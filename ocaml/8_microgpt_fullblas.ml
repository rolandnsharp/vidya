(* 8_microgpt_fullblas.ml — Stage 8: BLAS Everywhere
   ================================================================

   Based on Stage 7 (batched training, 27s).

   WHAT CHANGES
   ============
   Stage 7 used BLAS for the linear projections (Wq, Wk, Wv, Wo, fc1, fc2,
   lm_head) but kept scalar loops for the attention inner operations (score
   computation and value weighting). Stage 8 BLAS-ifies those too:

   Attention forward per head:
     Scores[S,S] = Q_h[S,d] @ K_h[S,d]^T    — was a triple scalar loop
     Attn_h[S,d] = Softmax[S,S] @ V_h[S,d]  — was a triple scalar loop

   Attention backward per head:
     dWeights[S,S] = dAttn_h[S,d] @ V_h[S,d]^T
     dV_h[S,d]     = Weights^T[S,S] @ dAttn_h[S,d]
     dQ_rot_h[S,d] = dScores[S,S] @ K_rot_h[S,d]
     dK_rot_h[S,d] = dScores^T[S,S] @ Q_rot_h[S,d]

   To use BLAS, we extract per-head data from the interleaved [S, n_embd]
   layout into contiguous [S, head_dim] arrays. This is a small memcpy
   (~1KB per extraction at S=64, d=16) that's negligible vs the BLAS wins.

   The causal mask is applied after the BLAS score computation: we compute
   the full [S,S] score matrix then set the upper triangle to -inf before
   softmax. After softmax, the upper triangle is 0, so all subsequent
   operations are mathematically correct.

   Everything else identical to Stage 7.

   Compile:
     ocamlopt -O2 -o microgpt_fullblas \
       blas_stubs.c 8_microgpt_fullblas.ml \
       -ccopt "-I/usr/include/x86_64-linux-gnu" -cclib -lopenblas

   Run:
     ./microgpt_fullblas *)

let () = Random.init 42

(* ══════════════════════════════════════════════════════════════════════
   BLAS FFI — 6 dgemm variants via 3-bit op flag
   ══════════════════════════════════════════════════════════════════════ *)
external blas_dgemm
  : int -> int -> int -> int
    -> float array -> float array -> float array
    -> unit
  = "caml_dgemm_byte" "caml_dgemm"

(* ══════════════════════════════════════════════════════════════════════
   UTILITIES (unchanged)
   ══════════════════════════════════════════════════════════════════════ *)

let random_gauss ?(mean = 0.0) ?(std = 1.0) () =
  let rec sample () =
    let u = Random.float 2.0 -. 1.0 in
    let v = Random.float 2.0 -. 1.0 in
    let s = u *. u +. v *. v in
    if s >= 1.0 || s = 0.0 then sample ()
    else mean +. std *. u *. sqrt (-2.0 *. log s /. s)
  in
  sample ()

let shuffle arr =
  for i = Array.length arr - 1 downto 1 do
    let j = Random.int (i + 1) in
    let tmp = arr.(i) in
    arr.(i) <- arr.(j);
    arr.(j) <- tmp
  done

let weighted_choice weights =
  let total = Array.fold_left (+.) 0.0 weights in
  let r = Random.float total in
  Array.fold_left (fun (chosen, remaining) w ->
    if remaining <= 0.0 then (chosen, remaining)
    else if remaining -. w <= 0.0 then (chosen, remaining -. w)
    else (chosen + 1, remaining -. w)
  ) (0, r) weights
  |> fst

let load_docs filename =
  if not (Sys.file_exists filename) then begin
    Printf.eprintf "Error: %s not found.\n" filename;
    exit 1
  end;
  let ic = open_in filename in
  let docs = ref [] in
  (try while true do
    let line = input_line ic |> String.trim in
    if String.length line > 0 then docs := line :: !docs
  done with End_of_file -> ());
  close_in ic;
  let arr = !docs |> List.rev |> Array.of_list in
  shuffle arr;
  arr

let build_vocab docs =
  let char_set = Hashtbl.create 128 in
  docs |> Array.iter (fun doc ->
    doc |> String.iter (fun ch -> Hashtbl.replace char_set ch ())
  );
  let uchars =
    Hashtbl.fold (fun ch () acc -> ch :: acc) char_set []
    |> List.sort Char.compare
    |> Array.of_list
  in
  let bos_id = Array.length uchars in
  (uchars, bos_id, bos_id + 1)

let build_char_to_id uchars =
  let tbl = Hashtbl.create (Array.length uchars) in
  Array.iteri (fun i ch -> Hashtbl.replace tbl ch i) uchars;
  tbl

let tokenize char_to_id bos_id doc =
  let n = String.length doc in
  let result = Array.make (n + 2) bos_id in
  for i = 0 to n - 1 do
    result.(i + 1) <- Hashtbl.find char_to_id doc.[i]
  done;
  result

(* ══════════════════════════════════════════════════════════════════════
   TENSOR AUTOGRAD ENGINE (unchanged)
   ══════════════════════════════════════════════════════════════════════ *)

let next_id = ref 0
let fresh_id () = let id = !next_id in incr next_id; id

type value = {
  id : int;
  data : float array;
  grad : float array;
  shape : int array;
  children : value array;
  backward_fn : unit -> unit;
}

let make_param shape data =
  { id = fresh_id (); data; grad = Array.make (Array.length data) 0.0;
    shape; children = [||]; backward_fn = (fun () -> ()) }

let dummy_node = { id = -1; data = [||]; grad = [||]; shape = [||];
                   children = [||]; backward_fn = (fun () -> ()) }

(* ══════════════════════════════════════════════════════════════════════
   SHARED OPS — element-wise, work for any shape
   ══════════════════════════════════════════════════════════════════════ *)

let tensor_add a b =
  let n = Array.length a.data in
  let y_data = Array.create_float n in
  for i = 0 to n - 1 do y_data.(i) <- a.data.(i) +. b.data.(i) done;
  let y_grad = Array.make n 0.0 in
  let backward () =
    for i = 0 to n - 1 do
      a.grad.(i) <- a.grad.(i) +. y_grad.(i);
      b.grad.(i) <- b.grad.(i) +. y_grad.(i)
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|a; b|]; backward_fn = backward }

let tensor_scale x s =
  let n = Array.length x.data in
  let y_data = Array.create_float n in
  for i = 0 to n - 1 do y_data.(i) <- x.data.(i) *. s done;
  let y_grad = Array.make n 0.0 in
  let backward () =
    for i = 0 to n - 1 do
      x.grad.(i) <- x.grad.(i) +. y_grad.(i) *. s
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

let tensor_gelu x =
  let n = Array.length x.data in
  let a = 0.7978845608028654 in
  let b = 0.044715 in
  let y_data = Array.create_float n in
  for i = 0 to n - 1 do
    let xi = x.data.(i) in
    let inner = a *. (xi +. b *. xi *. xi *. xi) in
    let t = tanh inner in
    y_data.(i) <- 0.5 *. xi *. (1.0 +. t)
  done;
  let y_grad = Array.make n 0.0 in
  let backward () =
    for i = 0 to n - 1 do
      let xi = x.data.(i) in
      let inner = a *. (xi +. b *. xi *. xi *. xi) in
      let t = tanh inner in
      let dtanh = 1.0 -. t *. t in
      let dgelu = 0.5 *. (1.0 +. t)
        +. 0.5 *. xi *. dtanh *. a *. (1.0 +. 3.0 *. b *. xi *. xi) in
      x.grad.(i) <- x.grad.(i) +. y_grad.(i) *. dgelu
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

let tensor_softmax x =
  let n = Array.length x.data in
  let max_val = ref neg_infinity in
  for i = 0 to n - 1 do
    if x.data.(i) > !max_val then max_val := x.data.(i)
  done;
  let y_data = Array.create_float n in
  let sum_exp = ref 0.0 in
  for i = 0 to n - 1 do
    let e = exp (x.data.(i) -. !max_val) in
    y_data.(i) <- e;
    sum_exp := !sum_exp +. e
  done;
  let inv_sum = 1.0 /. !sum_exp in
  for i = 0 to n - 1 do y_data.(i) <- y_data.(i) *. inv_sum done;
  let y_grad = Array.make n 0.0 in
  let backward () =
    let dot = ref 0.0 in
    for j = 0 to n - 1 do dot := !dot +. y_grad.(j) *. y_data.(j) done;
    for i = 0 to n - 1 do
      x.grad.(i) <- x.grad.(i) +. y_data.(i) *. (y_grad.(i) -. !dot)
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

let tensor_nll probs target =
  let y_data = [| -. log probs.data.(target) |] in
  let y_grad = [| 0.0 |] in
  let backward () =
    probs.grad.(target) <- probs.grad.(target)
      -. y_grad.(0) /. probs.data.(target) in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|1|]; children = [|probs|]; backward_fn = backward }

let tensor_mean losses =
  let n = Array.length losses in
  let nf = float_of_int n in
  let total = Array.fold_left (fun acc l -> acc +. l.data.(0)) 0.0 losses in
  let y_data = [| total /. nf |] in
  let y_grad = [| 0.0 |] in
  let backward () =
    Array.iter (fun l ->
      l.grad.(0) <- l.grad.(0) +. y_grad.(0) /. nf
    ) losses in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|1|]; children = Array.copy losses; backward_fn = backward }

let tensor_row mat row_idx =
  let cols = mat.shape.(1) in
  let off = row_idx * cols in
  let y_data = Array.create_float cols in
  for j = 0 to cols - 1 do y_data.(j) <- mat.data.(off + j) done;
  let y_grad = Array.make cols 0.0 in
  let backward () =
    for j = 0 to cols - 1 do
      mat.grad.(off + j) <- mat.grad.(off + j) +. y_grad.(j)
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|cols|]; children = [|mat|]; backward_fn = backward }

(* ══════════════════════════════════════════════════════════════════════
   SINGLE-VECTOR OPS — inference only (one token at a time with KV cache)
   ══════════════════════════════════════════════════════════════════════ *)

let matmul w x =
  let m = w.shape.(0) and n = w.shape.(1) in
  let y_data = Array.create_float m in
  blas_dgemm 0 m 1 n w.data x.data y_data;
  let y_grad = Array.make m 0.0 in
  let backward () =
    blas_dgemm 1 m n 1 y_grad x.data w.grad;
    blas_dgemm 5 n 1 m w.data y_grad x.grad
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|m|]; children = [|w; x|]; backward_fn = backward }

let tensor_rmsnorm x =
  let n = Array.length x.data in
  let nf = float_of_int n in
  let ms = ref 0.0 in
  for i = 0 to n - 1 do ms := !ms +. x.data.(i) *. x.data.(i) done;
  let rms = sqrt (!ms /. nf +. 1e-5) in
  let y_data = Array.create_float n in
  for i = 0 to n - 1 do y_data.(i) <- x.data.(i) /. rms done;
  let y_grad = Array.make n 0.0 in
  let backward () =
    let dot_gy = ref 0.0 in
    for i = 0 to n - 1 do dot_gy := !dot_gy +. y_grad.(i) *. y_data.(i) done;
    let mean_gy = !dot_gy /. nf in
    for i = 0 to n - 1 do
      x.grad.(i) <- x.grad.(i) +. (y_grad.(i) -. y_data.(i) *. mean_gy) /. rms
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

(* ══════════════════════════════════════════════════════════════════════
   BATCHED OPS — training (all positions at once)
   ══════════════════════════════════════════════════════════════════════ *)

let batch_matmul w x =
  let m = w.shape.(0) and n = w.shape.(1) in
  let s = Array.length x.data / n in
  let y_data = Array.create_float (s * m) in
  blas_dgemm 2 s m n x.data w.data y_data;
  let y_grad = Array.make (s * m) 0.0 in
  let backward () =
    blas_dgemm 5 m n s y_grad x.data w.grad;
    blas_dgemm 1 s n m y_grad w.data x.grad
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|s; m|]; children = [|w; x|]; backward_fn = backward }

let batch_rmsnorm ~dim x =
  let s = Array.length x.data / dim in
  let dimf = float_of_int dim in
  let y_data = Array.create_float (s * dim) in
  let rms_vals = Array.create_float s in
  for i = 0 to s - 1 do
    let off = i * dim in
    let ms = ref 0.0 in
    for j = 0 to dim - 1 do
      let v = x.data.(off + j) in
      ms := !ms +. v *. v
    done;
    let rms = sqrt (!ms /. dimf +. 1e-5) in
    rms_vals.(i) <- rms;
    for j = 0 to dim - 1 do
      y_data.(off + j) <- x.data.(off + j) /. rms
    done
  done;
  let y_grad = Array.make (s * dim) 0.0 in
  let backward () =
    for i = 0 to s - 1 do
      let off = i * dim in
      let rms = rms_vals.(i) in
      let dot_gy = ref 0.0 in
      for j = 0 to dim - 1 do
        dot_gy := !dot_gy +. y_grad.(off + j) *. y_data.(off + j)
      done;
      let mean_gy = !dot_gy /. dimf in
      for j = 0 to dim - 1 do
        x.grad.(off + j) <- x.grad.(off + j)
          +. (y_grad.(off + j) -. y_data.(off + j) *. mean_gy) /. rms
      done
    done
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|s; dim|]; children = [|x|]; backward_fn = backward }

let batch_embed wte tokens n_embd =
  let s = Array.length tokens in
  let data = Array.create_float (s * n_embd) in
  for i = 0 to s - 1 do
    Array.blit wte.data (tokens.(i) * n_embd) data (i * n_embd) n_embd
  done;
  let grad = Array.make (s * n_embd) 0.0 in
  let backward () =
    for i = 0 to s - 1 do
      let src = tokens.(i) * n_embd in
      let dst = i * n_embd in
      for j = 0 to n_embd - 1 do
        wte.grad.(src + j) <- wte.grad.(src + j) +. grad.(dst + j)
      done
    done
  in
  { id = fresh_id (); data; grad;
    shape = [|s; n_embd|]; children = [|wte|]; backward_fn = backward }

(* ══════════════════════════════════════════════════════════════════════
   BACKWARD PASS (unchanged)
   ══════════════════════════════════════════════════════════════════════ *)

let topological_sort root =
  let visited = Hashtbl.create 1024 in
  let topo = ref [] in
  let rec build_topo v =
    if not (Hashtbl.mem visited v.id) then begin
      Hashtbl.add visited v.id ();
      Array.iter build_topo v.children;
      topo := v :: !topo
    end
  in
  build_topo root;
  !topo

let backward loss =
  let topo = topological_sort loss in
  loss.grad.(0) <- 1.0;
  List.iter (fun v -> v.backward_fn ()) topo

(* ══════════════════════════════════════════════════════════════════════
   MODEL DEFINITION — identical to Stage 5/6/7
   ══════════════════════════════════════════════════════════════════════ *)

let n_layer = 4
let n_embd = 64
let block_size = 64
let n_head = 4
let head_dim = n_embd / n_head   (* = 16 *)
let half_dim = head_dim / 2      (* = 8 *)

let rope_freqs = Array.init half_dim (fun i ->
  1.0 /. (10000.0 ** (float_of_int (2 * i) /. float_of_int head_dim)))

let rope_cos = Array.init block_size (fun pos ->
  Array.init half_dim (fun i ->
    cos (float_of_int pos *. rope_freqs.(i))))

let rope_sin = Array.init block_size (fun pos ->
  Array.init half_dim (fun i ->
    sin (float_of_int pos *. rope_freqs.(i))))

type layer_weights = {
  attn_wq : value; attn_wk : value;
  attn_wv : value; attn_wo : value;
  mlp_fc1 : value; mlp_fc2 : value;
}

type kv_cache = {
  k_cache : float array array;
  v_cache : float array array;
  k_nodes : value array;
  v_nodes : value array;
  mutable len : int;
}

let make_kv_cache () = {
  k_cache = Array.init n_head (fun _ -> Array.create_float (block_size * head_dim));
  v_cache = Array.init n_head (fun _ -> Array.create_float (block_size * head_dim));
  k_nodes = Array.make block_size dummy_node;
  v_nodes = Array.make block_size dummy_node;
  len = 0;
}

type model = {
  wte : value;
  lm_head : value;
  layers : layer_weights array;
}

let init_matrix ?(std = 0.08) nout nin =
  let data = Array.init (nout * nin) (fun _ -> random_gauss ~std ()) in
  make_param [|nout; nin|] data

let init_model vocab_size =
  let wte = init_matrix vocab_size n_embd in
  let lm_head = init_matrix vocab_size n_embd in
  let make_layer () =
    let attn_wq = init_matrix n_embd n_embd in
    let attn_wk = init_matrix n_embd n_embd in
    let attn_wv = init_matrix n_embd n_embd in
    let attn_wo = init_matrix n_embd n_embd in
    let mlp_fc1 = init_matrix (4 * n_embd) n_embd in
    let mlp_fc2 = init_matrix n_embd (4 * n_embd) in
    { attn_wq; attn_wk; attn_wv; attn_wo; mlp_fc1; mlp_fc2 }
  in
  let layers = Array.init n_layer (fun _ -> make_layer ()) in
  { wte; lm_head; layers }

let collect_params model =
  let layer_params l =
    [l.attn_wq; l.attn_wk; l.attn_wv; l.attn_wo; l.mlp_fc1; l.mlp_fc2] in
  [model.wte; model.lm_head]
  @ (model.layers |> Array.to_list |> List.map layer_params |> List.flatten)
  |> Array.of_list

(* ══════════════════════════════════════════════════════════════════════
   HEAD EXTRACTION / INSERTION HELPERS

   Our tensors are [S, n_embd] with heads interleaved in the inner dim:
     row i, head h = data[i * n_embd + h * head_dim .. + head_dim - 1]

   BLAS needs contiguous [S, head_dim] arrays per head. These helpers
   extract/insert the head slices. Cost: ~1KB of memcpy per call at
   S=64, d=16 — negligible vs the BLAS compute.
   ══════════════════════════════════════════════════════════════════════ *)

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

(* ══════════════════════════════════════════════════════════════════════
   BATCHED TRAINING FORWARD
   ══════════════════════════════════════════════════════════════════════ *)

(* Batched multi-head attention with RoPE, causal mask, and BLAS.

   BLAS operations per head (forward):
     Scores[S,S] = Q_h[S,d] @ K_h[S,d]^T     op=2  (NT overwrite)
     Attn_h[S,d] = Weights[S,S] @ V_h[S,d]    op=0  (NN overwrite)

   BLAS operations per head (backward):
     dWeights[S,S] = dAttn_h[S,d] @ V_h[S,d]^T    op=2
     dV_h[S,d]     = Weights^T[S,S] @ dAttn_h[S,d] op=4  (TN overwrite)
     dQ_rot_h[S,d] = dScores[S,S] @ K_rot_h[S,d]   op=0
     dK_rot_h[S,d] = dScores^T[S,S] @ Q_rot_h[S,d] op=4

   Scalar loops handle: causal mask, softmax forward/backward, RoPE rotation.

   At S=64, d=16: each BLAS call processes 64×64×16 = 65,536 multiply-adds.
   With 4 heads × 6 BLAS calls × 2 (fwd+bwd) = 48 BLAS calls per layer,
   4 layers = 192 additional BLAS calls per step. *)
let batch_fused_attention q k v seq_len =
  (* Apply RoPE rotation to all positions of Q and K *)
  let q_rot = Array.create_float (seq_len * n_embd) in
  let k_rot = Array.create_float (seq_len * n_embd) in
  for pos = 0 to seq_len - 1 do
    let cos_p = rope_cos.(pos) and sin_p = rope_sin.(pos) in
    let row = pos * n_embd in
    for h = 0 to n_head - 1 do
      let base = row + h * head_dim in
      for i = 0 to half_dim - 1 do
        let j0 = base + 2 * i and j1 = base + 2 * i + 1 in
        let c = cos_p.(i) and s = sin_p.(i) in
        q_rot.(j0) <- q.data.(j0) *. c -. q.data.(j1) *. s;
        q_rot.(j1) <- q.data.(j0) *. s +. q.data.(j1) *. c;
        k_rot.(j0) <- k.data.(j0) *. c -. k.data.(j1) *. s;
        k_rot.(j1) <- k.data.(j0) *. s +. k.data.(j1) *. c
      done
    done
  done;

  let scale = 1.0 /. sqrt (float_of_int head_dim) in
  let out_data = Array.create_float (seq_len * n_embd) in
  let all_weights = Array.init n_head (fun _ ->
    Array.create_float (seq_len * seq_len)) in

  (* Temporary buffers for head extraction — reused across heads *)
  let q_h = Array.create_float (seq_len * head_dim) in
  let k_h = Array.create_float (seq_len * head_dim) in
  let v_h = Array.create_float (seq_len * head_dim) in
  let scores = Array.create_float (seq_len * seq_len) in
  let attn_h = Array.create_float (seq_len * head_dim) in

  for h = 0 to n_head - 1 do
    let wt = all_weights.(h) in

    (* Extract head data into contiguous arrays *)
    extract_head q_rot q_h h seq_len;
    extract_head k_rot k_h h seq_len;

    (* Scores[S,S] = Q_h[S,d] @ K_h[S,d]^T — BLAS! *)
    blas_dgemm 2 seq_len seq_len head_dim q_h k_h scores;

    (* Scale + causal mask + row-wise softmax *)
    for i = 0 to seq_len - 1 do
      let max_s = ref neg_infinity in
      for j = 0 to i do
        let s = scores.(i * seq_len + j) *. scale in
        wt.(i * seq_len + j) <- s;
        if s > !max_s then max_s := s
      done;
      let sum_exp = ref 0.0 in
      for j = 0 to i do
        let e = exp (wt.(i * seq_len + j) -. !max_s) in
        wt.(i * seq_len + j) <- e;
        sum_exp := !sum_exp +. e
      done;
      let inv = 1.0 /. !sum_exp in
      for j = 0 to i do
        wt.(i * seq_len + j) <- wt.(i * seq_len + j) *. inv
      done;
      (* Zero upper triangle — needed for correct BLAS in value weighting *)
      for j = i + 1 to seq_len - 1 do wt.(i * seq_len + j) <- 0.0 done
    done;

    (* Attn_h[S,d] = Weights[S,S] @ V_h[S,d] — BLAS! *)
    extract_head v.data v_h h seq_len;
    blas_dgemm 0 seq_len head_dim seq_len wt v_h attn_h;

    (* Write back to interleaved output *)
    insert_head attn_h out_data h seq_len
  done;

  let out_grad = Array.make (seq_len * n_embd) 0.0 in
  let backward () =
    let dq_rot = Array.make (seq_len * n_embd) 0.0 in
    let dk_rot = Array.make (seq_len * n_embd) 0.0 in

    (* Temporary buffers — reused across heads *)
    let dout_h = Array.create_float (seq_len * head_dim) in
    let v_h = Array.create_float (seq_len * head_dim) in
    let dw = Array.create_float (seq_len * seq_len) in
    let dv_h = Array.create_float (seq_len * head_dim) in
    let k_h = Array.create_float (seq_len * head_dim) in
    let q_h = Array.create_float (seq_len * head_dim) in
    let dq_h = Array.create_float (seq_len * head_dim) in
    let dk_h = Array.create_float (seq_len * head_dim) in

    for h = 0 to n_head - 1 do
      let wt = all_weights.(h) in

      (* Extract dAttn_h and V_h *)
      extract_head out_grad dout_h h seq_len;
      extract_head v.data v_h h seq_len;

      (* dWeights[S,S] = dAttn_h[S,d] @ V_h[S,d]^T — BLAS *)
      blas_dgemm 2 seq_len seq_len head_dim dout_h v_h dw;

      (* dV_h[S,d] = Weights^T[S,S] @ dAttn_h[S,d] — BLAS *)
      blas_dgemm 4 seq_len head_dim seq_len wt dout_h dv_h;
      insert_head_accum dv_h v.grad h seq_len;

      (* Softmax backward: dscores[i,j] = wt[i,j] * (dw[i,j] - dot_i) * scale
         Reuse dw array for dscores in-place. *)
      for i = 0 to seq_len - 1 do
        let dot = ref 0.0 in
        for j = 0 to i do
          dot := !dot +. dw.(i * seq_len + j) *. wt.(i * seq_len + j)
        done;
        for j = 0 to i do
          dw.(i * seq_len + j) <-
            wt.(i * seq_len + j) *. (dw.(i * seq_len + j) -. !dot) *. scale
        done;
        (* Zero upper triangle — dscores[i,j]=0 for j>i since wt[i,j]=0 *)
        for j = i + 1 to seq_len - 1 do dw.(i * seq_len + j) <- 0.0 done
      done;
      (* dw is now dScores *)

      (* dQ_rot_h[S,d] = dScores[S,S] @ K_rot_h[S,d] — BLAS *)
      extract_head k_rot k_h h seq_len;
      blas_dgemm 0 seq_len head_dim seq_len dw k_h dq_h;
      insert_head_accum dq_h dq_rot h seq_len;

      (* dK_rot_h[S,d] = dScores^T[S,S] @ Q_rot_h[S,d] — BLAS *)
      extract_head q_rot q_h h seq_len;
      blas_dgemm 4 seq_len head_dim seq_len dw q_h dk_h;
      insert_head_accum dk_h dk_rot h seq_len
    done;

    (* Inverse RoPE: rotate gradients back to pre-rotation space *)
    for pos = 0 to seq_len - 1 do
      let cos_p = rope_cos.(pos) and sin_p = rope_sin.(pos) in
      let row = pos * n_embd in
      for h = 0 to n_head - 1 do
        let base = row + h * head_dim in
        for i = 0 to half_dim - 1 do
          let j0 = base + 2 * i and j1 = base + 2 * i + 1 in
          let c = cos_p.(i) and s = sin_p.(i) in
          q.grad.(j0) <- q.grad.(j0) +. dq_rot.(j0) *. c +. dq_rot.(j1) *. s;
          q.grad.(j1) <- q.grad.(j1) -. dq_rot.(j0) *. s +. dq_rot.(j1) *. c;
          k.grad.(j0) <- k.grad.(j0) +. dk_rot.(j0) *. c +. dk_rot.(j1) *. s;
          k.grad.(j1) <- k.grad.(j1) -. dk_rot.(j0) *. s +. dk_rot.(j1) *. c
        done
      done
    done
  in
  { id = fresh_id (); data = out_data; grad = out_grad;
    shape = [|seq_len; n_embd|]; children = [|q; k; v|];
    backward_fn = backward }

(* Batched transformer block *)
let batch_transformer_block x layer seq_len =
  let xn = batch_rmsnorm ~dim:n_embd x in
  let q = batch_matmul layer.attn_wq xn in
  let k = batch_matmul layer.attn_wk xn in
  let v = batch_matmul layer.attn_wv xn in
  let attn_out = batch_fused_attention q k v seq_len in
  let attn_proj = batch_matmul layer.attn_wo attn_out in
  let x = tensor_add x attn_proj in
  let xn = batch_rmsnorm ~dim:n_embd x in
  let mlp_h = batch_matmul layer.mlp_fc1 xn |> tensor_gelu in
  let mlp_out = batch_matmul layer.mlp_fc2 mlp_h in
  tensor_add x mlp_out

let gpt_forward_batch model tokens seq_len =
  let x = batch_embed model.wte tokens n_embd in
  let x = batch_rmsnorm ~dim:n_embd x in
  let x = Array.fold_left (fun x layer ->
    batch_transformer_block x layer seq_len
  ) x model.layers in
  batch_matmul model.lm_head x

(* ══════════════════════════════════════════════════════════════════════
   INFERENCE — single-token with KV cache (unchanged from Stage 6/7)
   ══════════════════════════════════════════════════════════════════════ *)

let embed_token model token_id =
  tensor_row model.wte token_id |> tensor_rmsnorm

let fused_multi_head_attention x layer kv =
  let q_raw = matmul layer.attn_wq x in
  let k_raw = matmul layer.attn_wk x in
  let v = matmul layer.attn_wv x in
  let pos = kv.len in
  kv.k_nodes.(pos) <- k_raw;
  kv.v_nodes.(pos) <- v;
  let q_rot = Array.create_float n_embd in
  let cos_p = rope_cos.(pos) and sin_p = rope_sin.(pos) in
  for h = 0 to n_head - 1 do
    let h_off = h * head_dim in
    for i = 0 to half_dim - 1 do
      let j0 = h_off + 2 * i and j1 = h_off + 2 * i + 1 in
      let c = cos_p.(i) and s = sin_p.(i) in
      q_rot.(j0) <- q_raw.data.(j0) *. c -. q_raw.data.(j1) *. s;
      q_rot.(j1) <- q_raw.data.(j0) *. s +. q_raw.data.(j1) *. c;
      kv.k_cache.(h).(pos * head_dim + 2 * i) <-
        k_raw.data.(j0) *. c -. k_raw.data.(j1) *. s;
      kv.k_cache.(h).(pos * head_dim + 2 * i + 1) <-
        k_raw.data.(j0) *. s +. k_raw.data.(j1) *. c
    done;
    for j = 0 to head_dim - 1 do
      kv.v_cache.(h).(pos * head_dim + j) <- v.data.(h_off + j)
    done
  done;
  kv.len <- pos + 1;
  let t_len = pos + 1 in
  let scale = 1.0 /. sqrt (float_of_int head_dim) in
  let out_data = Array.create_float n_embd in
  let all_weights = Array.init n_head (fun _ -> Array.create_float t_len) in
  for h = 0 to n_head - 1 do
    let h_off = h * head_dim in
    let weights = all_weights.(h) in
    let max_score = ref neg_infinity in
    for t = 0 to t_len - 1 do
      let s = ref 0.0 in
      for j = 0 to head_dim - 1 do
        s := !s +. kv.k_cache.(h).(t * head_dim + j) *. q_rot.(h_off + j)
      done;
      let score = !s *. scale in
      weights.(t) <- score;
      if score > !max_score then max_score := score
    done;
    let sum_exp = ref 0.0 in
    for t = 0 to t_len - 1 do
      let e = exp (weights.(t) -. !max_score) in
      weights.(t) <- e;
      sum_exp := !sum_exp +. e
    done;
    let inv_sum = 1.0 /. !sum_exp in
    for t = 0 to t_len - 1 do
      weights.(t) <- weights.(t) *. inv_sum
    done;
    for j = 0 to head_dim - 1 do
      let s = ref 0.0 in
      for t = 0 to t_len - 1 do
        s := !s +. weights.(t) *. kv.v_cache.(h).(t * head_dim + j)
      done;
      out_data.(h_off + j) <- !s
    done
  done;
  let out_grad = Array.make n_embd 0.0 in
  let backward () = () in
  let attn_out = { id = fresh_id (); data = out_data; grad = out_grad;
    shape = [|n_embd|]; children = [||]; backward_fn = backward } in
  matmul layer.attn_wo attn_out

let mlp_block x layer =
  matmul layer.mlp_fc1 x |> tensor_gelu |> fun h -> matmul layer.mlp_fc2 h

let transformer_block x layer kv =
  let x =
    tensor_rmsnorm x |> fun xn ->
    fused_multi_head_attention xn layer kv |> tensor_add x in
  tensor_rmsnorm x |> fun xn -> mlp_block xn layer |> tensor_add x

let gpt_forward model token_id kv_caches =
  let x = embed_token model token_id in
  let x = Array.to_list model.layers
    |> List.mapi (fun li layer -> (li, layer))
    |> List.fold_left (fun x (li, layer) ->
      transformer_block x layer kv_caches.(li)) x in
  matmul model.lm_head x

(* ══════════════════════════════════════════════════════════════════════
   TRAINING (unchanged)
   ══════════════════════════════════════════════════════════════════════ *)

type adam_state = { m : float array; v : float array }
let learning_rate = 0.01
let beta1 = 0.85
let beta2 = 0.99
let eps_adam = 1e-8
let max_grad_norm = 1.0
let warmup_steps = 100

let init_adam params =
  let total = Array.fold_left (fun acc p -> acc + Array.length p.data) 0 params in
  { m = Array.make total 0.0; v = Array.make total 0.0 }

let get_lr step num_steps =
  if step < warmup_steps then
    learning_rate *. float_of_int step /. float_of_int warmup_steps
  else
    let progress =
      float_of_int (step - warmup_steps)
      /. float_of_int (num_steps - warmup_steps) in
    learning_rate *. 0.5 *. (1.0 +. cos (Float.pi *. progress))

let clip_grad_norm params =
  let norm_sq = ref 0.0 in
  Array.iter (fun p ->
    for i = 0 to Array.length p.grad - 1 do
      norm_sq := !norm_sq +. p.grad.(i) *. p.grad.(i)
    done
  ) params;
  let norm = sqrt !norm_sq in
  if norm > max_grad_norm then begin
    let scale = max_grad_norm /. norm in
    Array.iter (fun p ->
      for i = 0 to Array.length p.grad - 1 do
        p.grad.(i) <- p.grad.(i) *. scale
      done
    ) params
  end

let adam_step params adam step num_steps =
  let lr_t = get_lr step num_steps in
  let bc1 = 1.0 /. (1.0 -. beta1 ** float_of_int (step + 1)) in
  let bc2 = 1.0 /. (1.0 -. beta2 ** float_of_int (step + 1)) in
  let offset = ref 0 in
  params |> Array.iter (fun p ->
    let n = Array.length p.data in
    for i = 0 to n - 1 do
      let fi = !offset + i in
      adam.m.(fi) <- beta1 *. adam.m.(fi) +. (1.0 -. beta1) *. p.grad.(i);
      adam.v.(fi) <- beta2 *. adam.v.(fi) +. (1.0 -. beta2) *. p.grad.(i) *. p.grad.(i);
      let m_hat = adam.m.(fi) *. bc1 in
      let v_hat = adam.v.(fi) *. bc2 in
      p.data.(i) <- p.data.(i) -. lr_t *. m_hat /. (sqrt v_hat +. eps_adam);
      p.grad.(i) <- 0.0
    done;
    offset := !offset + n)

let compute_loss model tokens =
  let seq_len = min block_size (Array.length tokens - 1) in
  let input_tokens = Array.sub tokens 0 seq_len in
  let logits = gpt_forward_batch model input_tokens seq_len in
  let losses = Array.init seq_len (fun i ->
    let logits_i = tensor_row logits i in
    let probs_i = tensor_softmax logits_i in
    tensor_nll probs_i tokens.(i + 1)
  ) in
  (tensor_mean losses, seq_len)

let train model params docs char_to_id bos_id num_steps =
  let adam = init_adam params in
  for step = 0 to num_steps - 1 do
    let doc = docs.(step mod Array.length docs) in
    let tokens = tokenize char_to_id bos_id doc in
    let (loss, _) = compute_loss model tokens in
    backward loss;
    clip_grad_norm params;
    adam_step params adam step num_steps;
    if (step + 1) mod 500 = 0 then
      Printf.printf "step %4d / %4d | loss %.4f\n%!"
        (step + 1) num_steps loss.data.(0)
  done

(* ══════════════════════════════════════════════════════════════════════
   INFERENCE + MAIN (unchanged)
   ══════════════════════════════════════════════════════════════════════ *)

let generate_sample model uchars bos_id _vocab_size temperature =
  let kv_caches = Array.init n_layer (fun _ -> make_kv_cache ()) in
  let token_id = ref bos_id in
  let buf = Buffer.create 64 in
  let pos = ref 0 in
  while !pos < block_size do
    let logits = gpt_forward model !token_id kv_caches in
    let probs = tensor_scale logits (1.0 /. temperature) |> tensor_softmax in
    token_id := weighted_choice probs.data;
    if !token_id = bos_id then pos := block_size
    else begin Buffer.add_char buf uchars.(!token_id); incr pos end
  done;
  Buffer.contents buf

let () =
  let t_start = Sys.time () in
  let docs = load_docs "input.txt" in
  Printf.printf "num docs: %d\n" (Array.length docs);
  let (uchars, bos_id, vocab_size) = build_vocab docs in
  Printf.printf "vocab size: %d\n" vocab_size;
  let char_to_id = build_char_to_id uchars in
  let model = init_model vocab_size in
  let params = collect_params model in
  let total_params =
    Array.fold_left (fun acc p -> acc + Array.length p.data) 0 params in
  Printf.printf "num params: %d\n" total_params;
  train model params docs char_to_id bos_id 2000;
  Printf.printf "--- inference (new, hallucinated text) ---\n";
  for i = 1 to 20 do
    generate_sample model uchars bos_id vocab_size 0.5
    |> Printf.printf "sample %2d: %s\n" i
  done;
  Printf.printf "total time: %.2fs\n" (Sys.time () -. t_start)
