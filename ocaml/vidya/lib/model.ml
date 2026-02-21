(* model.ml — Model definition, hyperparameters, RoPE, initialization
   ===================================================================

   Weight-tied, residual-scaled GPT-2 style transformer.
   256-dim, 8 heads, 12 layers, 256-token context, ~9.6M params.

   Weight tying: lm_head shares wte's matrix. The output projection
   logit for token t = dot(hidden, embedding[t]). Saves vocab_size ×
   n_embd params and creates tight embedding ↔ prediction feedback.

   Residual scaling: attn_wo and mlp_fc2 initialized with
   std = 0.08 / sqrt(2 * n_layer). Each layer adds two residual
   connections; scaling keeps variance bounded at ~1x through the network. *)

let n_layer = 12
let n_embd = 256
let block_size = 256
let n_head = 8
let head_dim = n_embd / n_head   (* = 32 *)
let half_dim = head_dim / 2      (* = 16 *)

(* RoPE (Rotary Position Embeddings) frequency tables.
   Precomputed for all positions and frequency indices. *)
let rope_freqs = Array.init half_dim (fun i ->
  1.0 /. (10000.0 ** (float_of_int (2 * i) /. float_of_int head_dim)))

let rope_cos = Array.init block_size (fun pos ->
  Array.init half_dim (fun i ->
    cos (float_of_int pos *. rope_freqs.(i))))

let rope_sin = Array.init block_size (fun pos ->
  Array.init half_dim (fun i ->
    sin (float_of_int pos *. rope_freqs.(i))))

type layer_weights = {
  attn_wq : Tensor.value; attn_wk : Tensor.value;
  attn_wv : Tensor.value; attn_wo : Tensor.value;
  mlp_fc1 : Tensor.value; mlp_fc2 : Tensor.value;
  ln1 : Tensor.value;  (* pre-attention RMSNorm scale *)
  ln2 : Tensor.value;  (* pre-MLP RMSNorm scale *)
}

type kv_cache = {
  k_cache : float array array;
  v_cache : float array array;
  k_nodes : Tensor.value array;
  v_nodes : Tensor.value array;
  mutable len : int;
}

let make_kv_cache () = {
  k_cache = Array.init n_head (fun _ -> Array.create_float (block_size * head_dim));
  v_cache = Array.init n_head (fun _ -> Array.create_float (block_size * head_dim));
  k_nodes = Array.make block_size (Tensor.dummy_node ());
  v_nodes = Array.make block_size (Tensor.dummy_node ());
  len = 0;
}

type t = {
  wte : Tensor.value;
  lm_head : Tensor.value;
  layers : layer_weights array;
  embed_norm : Tensor.value;  (* post-embed RMSNorm scale *)
  final_norm : Tensor.value;  (* pre-lm_head RMSNorm scale *)
}

let init_matrix ?(std = 0.08) nout nin =
  let data = Array.init (nout * nin) (fun _ -> Utils.random_gauss ~std ()) in
  Tensor.make_param [|nout; nin|] data

let init_ones dim =
  Tensor.make_param [|dim|] (Array.make dim 1.0)

(* init: Create a fresh model with random weights.
   Weight tying: lm_head = wte (same tensor, shared gradients).
   Residual scaling: attn_wo, mlp_fc2 init with reduced std. *)
let init vocab_size =
  let wte = init_matrix vocab_size n_embd in
  let lm_head = wte in
  let embed_norm = init_ones n_embd in
  let final_norm = init_ones n_embd in
  let residual_std = 0.08 /. sqrt (float_of_int (2 * n_layer)) in
  let make_layer () =
    let attn_wq = init_matrix n_embd n_embd in
    let attn_wk = init_matrix n_embd n_embd in
    let attn_wv = init_matrix n_embd n_embd in
    let attn_wo = init_matrix ~std:residual_std n_embd n_embd in
    let mlp_fc1 = init_matrix (4 * n_embd) n_embd in
    let mlp_fc2 = init_matrix ~std:residual_std n_embd (4 * n_embd) in
    let ln1 = init_ones n_embd in
    let ln2 = init_ones n_embd in
    { attn_wq; attn_wk; attn_wv; attn_wo; mlp_fc1; mlp_fc2; ln1; ln2 }
  in
  let layers = Array.init n_layer (fun _ -> make_layer ()) in
  { wte; lm_head; layers; embed_norm; final_norm }

(* collect_params: Gather all unique parameter tensors in a fixed order.
   Only includes wte once (lm_head is weight-tied to wte).
   Order: wte, then per-layer [wq, wk, wv, wo, fc1, fc2]. *)
let collect_params model =
  let layer_params l =
    [l.attn_wq; l.attn_wk; l.attn_wv; l.attn_wo; l.mlp_fc1; l.mlp_fc2;
     l.ln1; l.ln2] in
  [model.wte; model.embed_norm]
  @ (model.layers |> Array.to_list |> List.map layer_params |> List.flatten)
  @ [model.final_norm]
  |> Array.of_list

(* ── Head extraction / insertion helpers ──────────────────────────── *)

(* Our tensors are [S, n_embd] with heads interleaved in the inner dim:
     row i, head h = data[i * n_embd + h * head_dim .. + head_dim - 1]

   BLAS needs contiguous [S, head_dim] arrays per head. These helpers
   extract/insert the head slices. Cost: ~1KB memcpy per call at
   S=64, d=16 — negligible vs the BLAS compute. *)

(* Extract head h from [S, n_embd] → contiguous [S, head_dim] *)
let extract_head src dst h s =
  let h_off = h * head_dim in
  for i = 0 to s - 1 do
    Array.blit src (i * n_embd + h_off) dst (i * head_dim) head_dim
  done

(* Insert [S, head_dim] into head h of [S, n_embd] — overwrite *)
let insert_head src dst h s =
  let h_off = h * head_dim in
  for i = 0 to s - 1 do
    Array.blit src (i * head_dim) dst (i * n_embd + h_off) head_dim
  done

(* Insert [S, head_dim] into head h of [S, n_embd] — accumulate *)
let insert_head_accum src dst h s =
  let h_off = h * head_dim in
  for i = 0 to s - 1 do
    let d_off = i * n_embd + h_off in
    let s_off = i * head_dim in
    for j = 0 to head_dim - 1 do
      dst.(d_off + j) <- dst.(d_off + j) +. src.(s_off + j)
    done
  done
