(* 4b_microgpt_rope.ml — Stage 4b: Rotary Position Embeddings (RoPE)
   ================================================================

   Based on Stage 4a (GeLU, cosine LR, gradient clipping).
   One structural change: replace learned position embeddings (wpe) with
   Rotary Position Embeddings (RoPE).

   WHAT ROPE IS
   ============
   Instead of adding a learned position vector to each token embedding,
   RoPE applies a position-dependent ROTATION to the query and key
   vectors in the attention mechanism. The rotation angle depends on
   both the position in the sequence and the dimension index.

   For each pair of dimensions (2i, 2i+1) at sequence position p:

     q_rot[2i]   = q[2i] * cos(p*θ_i) - q[2i+1] * sin(p*θ_i)
     q_rot[2i+1] = q[2i] * sin(p*θ_i) + q[2i+1] * cos(p*θ_i)

   where θ_i = 1/10000^(2i/d) and d is the head dimension.

   The same rotation is applied to keys. Values are NOT rotated.

   WHY ROPE IS BETTER THAN LEARNED POSITIONS
   ==========================================
   1. RELATIVE POSITION: The dot product q_m · k_n only depends on the
      relative distance (m-n), not the absolute positions. This is because
      rotating q by angle α and k by angle β gives a dot product that
      depends only on (α-β). The model naturally learns relative attention.

   2. NO LEARNED PARAMETERS: RoPE is purely mathematical — no wpe matrix.
      Saves parameters (256 in our case) and removes a potential overfitting
      source.

   3. LENGTH EXTRAPOLATION: Since RoPE uses continuous rotation angles
      (not a lookup table), the model can potentially handle sequences
      longer than those seen during training. A wpe matrix of size
      [block_size, n_embd] hard-limits context to block_size.

   4. MULTI-SCALE ENCODING: The frequency spectrum (θ_0=1.0, θ_1=0.01
      for our head_dim=4) gives each dimension pair a different "wavelength"
      for encoding position. Low-frequency pairs capture coarse position,
      high-frequency pairs capture fine position.

   RoPE was introduced by Su et al. (2021) and is used by LLaMA, Mistral,
   GPT-NeoX, and essentially all modern open-source LLMs.

   CHANGES FROM STAGE 4a
   =====================
   - Removed: wpe parameter matrix, position embedding in embed_token
   - Added: rope_freqs, rope_cos, rope_sin pre-computed tables
   - Modified: fused_multi_head_attention applies RoPE rotation to q and k
     in forward, and inverse rotation to gradients in backward
   - The KV cache now stores ROTATED key data (not raw projections)
   - Parameter count: 5632 (was 5888 — saved 256 from removing wpe)

   Compile: ocamlopt -O2 -o microgpt_rope 4b_microgpt_rope.ml
   Run:     ./microgpt_rope *)

let () = Random.init 42

(* ══════════════════════════════════════════════════════════════════════
   UTILITIES
   ══════════════════════════════════════════════════════════════════════ *)

(* Box-Muller transform for normally distributed random numbers. *)
let random_gauss ?(mean = 0.0) ?(std = 1.0) () =
  let rec sample () =
    let u = Random.float 2.0 -. 1.0 in
    let v = Random.float 2.0 -. 1.0 in
    let s = u *. u +. v *. v in
    if s >= 1.0 || s = 0.0 then sample ()
    else mean +. std *. u *. sqrt (-2.0 *. log s /. s)
  in
  sample ()

(* Fisher-Yates shuffle — O(n), in-place, uniform distribution. *)
let shuffle arr =
  for i = Array.length arr - 1 downto 1 do
    let j = Random.int (i + 1) in
    let tmp = arr.(i) in
    arr.(i) <- arr.(j);
    arr.(j) <- tmp
  done

(* Multinomial sampling from an unnormalized weight array. *)
let weighted_choice weights =
  let total = Array.fold_left (+.) 0.0 weights in
  let r = Random.float total in
  Array.fold_left (fun (chosen, remaining) w ->
    if remaining <= 0.0 then (chosen, remaining)
    else if remaining -. w <= 0.0 then (chosen, remaining -. w)
    else (chosen + 1, remaining -. w)
  ) (0, r) weights
  |> fst

(* Load training data: one document (sentence) per line. *)
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

(* Build vocabulary: sorted unique chars + BOS/EOS token. *)
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

(* O(1) char->int hashtable for tokenization. *)
let build_char_to_id uchars =
  let tbl = Hashtbl.create (Array.length uchars) in
  Array.iteri (fun i ch -> Hashtbl.replace tbl ch i) uchars;
  tbl

(* Tokenize: [BOS; char_ids...; BOS] *)
let tokenize char_to_id bos_id doc =
  let n = String.length doc in
  let result = Array.make (n + 2) bos_id in
  for i = 0 to n - 1 do
    result.(i + 1) <- Hashtbl.find char_to_id doc.[i]
  done;
  result

(* ══════════════════════════════════════════════════════════════════════
   TENSOR AUTOGRAD ENGINE

   Each computation graph node holds data (forward output), grad
   (accumulated gradient), shape, children (inputs), and a backward_fn
   closure. The graph is built during forward and traversed in reverse
   topological order during backward.
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
   TENSOR OPERATIONS
   ══════════════════════════════════════════════════════════════════════ *)

(* Matrix-vector multiply: [m,n] @ [n] -> [m]
   Backward: dw_{i,j} += dy_i * x_j,  dx_j += w_{i,j} * dy_i *)
let matmul w x =
  let m = w.shape.(0) and n = w.shape.(1) in
  let y_data = Array.create_float m in
  for i = 0 to m - 1 do
    let s = ref 0.0 in
    for j = 0 to n - 1 do s := !s +. w.data.(i * n + j) *. x.data.(j) done;
    y_data.(i) <- !s
  done;
  let y_grad = Array.make m 0.0 in
  let backward () =
    for i = 0 to m - 1 do
      for j = 0 to n - 1 do
        w.grad.(i * n + j) <- w.grad.(i * n + j) +. y_grad.(i) *. x.data.(j);
        x.grad.(j) <- x.grad.(j) +. w.data.(i * n + j) *. y_grad.(i)
      done
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|m|]; children = [|w; x|]; backward_fn = backward }

(* Element-wise add: [n] + [n] -> [n] *)
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

(* Multiply by scalar constant: [n] * s -> [n] *)
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

(* GeLU activation: x * Phi(x) via tanh approximation.
   Backward: dgelu = 0.5*(1+t) + 0.5*x*(1-t²)*a*(1+3*b*x²) *)
let tensor_gelu x =
  let n = Array.length x.data in
  let a = 0.7978845608028654 in  (* sqrt(2/pi) *)
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

(* Fused softmax: find max, exp+sum, normalize (3 passes).
   Backward: dx_i += y_i * (dy_i - sum_j(dy_j * y_j)) *)
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

(* RMS normalization: y_i = x_i / sqrt(mean(x²) + eps)
   Backward: dx_j = (dy_j - y_j * mean(dy * y)) / rms *)
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

(* Extract row from 2D matrix: [m,n] -> [n].  Used for embedding lookup. *)
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

(* Negative log-likelihood: -log(probs[target]) -> scalar *)
let tensor_nll probs target =
  let y_data = [| -. log probs.data.(target) |] in
  let y_grad = [| 0.0 |] in
  let backward () =
    probs.grad.(target) <- probs.grad.(target)
      -. y_grad.(0) /. probs.data.(target) in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|1|]; children = [|probs|]; backward_fn = backward }

(* Mean of scalar losses *)
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

(* ══════════════════════════════════════════════════════════════════════
   BACKWARD PASS
   ══════════════════════════════════════════════════════════════════════ *)

let topological_sort root =
  let visited = Hashtbl.create 256 in
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
   MODEL DEFINITION
   ══════════════════════════════════════════════════════════════════════ *)

let n_layer = 1
let n_embd = 16
let block_size = 16
let n_head = 4
let head_dim = n_embd / n_head   (* = 4 *)
let half_dim = head_dim / 2      (* = 2 — number of rotation pairs per head *)

(* ── RoPE Tables ──────────────────────────────────────────────────────

   Pre-computed rotation frequencies and cos/sin tables.

   The frequency for dimension pair i is:
     θ_i = 1 / 10000^(2i/d)

   For our head_dim=4 (half_dim=2):
     θ_0 = 1/10000^(0/4) = 1.0      — fast rotation, ~2.5 full turns over 16 positions
     θ_1 = 1/10000^(2/4) = 0.01     — slow rotation, ~8.6° over 16 positions

   This gives each head two frequency bands for encoding position:
   a high-frequency pair that distinguishes nearby positions and a
   low-frequency pair that distinguishes distant positions. Larger
   head dimensions would have more frequency bands for finer encoding.

   cos/sin tables are indexed as rope_cos.(pos).(freq_idx). *)

let rope_freqs = Array.init half_dim (fun i ->
  1.0 /. (10000.0 ** (float_of_int (2 * i) /. float_of_int head_dim)))

let rope_cos = Array.init block_size (fun pos ->
  Array.init half_dim (fun i ->
    cos (float_of_int pos *. rope_freqs.(i))))

let rope_sin = Array.init block_size (fun pos ->
  Array.init half_dim (fun i ->
    sin (float_of_int pos *. rope_freqs.(i))))

(* ── Layer and Model Types ────────────────────────────────────────── *)

type layer_weights = {
  attn_wq : value; attn_wk : value;
  attn_wv : value; attn_wo : value;
  mlp_fc1 : value; mlp_fc2 : value;
}

(* KV cache stores ROTATED key data (after RoPE) and unrotated value data.
   k_nodes stores the UNROTATED key autograd nodes — backward inverse-rotates
   gradients before writing into k_nodes[t].grad, so the matmul backward
   for Wk receives the correct (unrotated) gradient. *)
type kv_cache = {
  k_cache : float array array;   (* rotated key data: [n_head][block_size * head_dim] *)
  v_cache : float array array;   (* unrotated value data *)
  k_nodes : value array;         (* unrotated key autograd nodes *)
  v_nodes : value array;         (* value autograd nodes *)
  mutable len : int;
}

let make_kv_cache () = {
  k_cache = Array.init n_head (fun _ -> Array.create_float (block_size * head_dim));
  v_cache = Array.init n_head (fun _ -> Array.create_float (block_size * head_dim));
  k_nodes = Array.make block_size dummy_node;
  v_nodes = Array.make block_size dummy_node;
  len = 0;
}

(* [STAGE 4b CHANGE] Model no longer has wpe (learned position embeddings).
   Position information is encoded entirely through RoPE rotations in
   the attention mechanism. This saves block_size * n_embd = 256 parameters. *)
type model = {
  wte : value;               (* token embeddings: [vocab_size, n_embd] *)
  lm_head : value;           (* output projection: [vocab_size, n_embd] *)
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
   FORWARD PASS
   ══════════════════════════════════════════════════════════════════════ *)

(* [STAGE 4b CHANGE] Embed a token: just token embedding + RMSNorm.
   No position embedding — position is encoded by RoPE in attention. *)
let embed_token model token_id =
  tensor_row model.wte token_id |> tensor_rmsnorm

(* ── Fused Multi-Head Attention with RoPE ─────────────────────────────

   This is the core function that combines RoPE with the fused KV cache
   attention from Stage 3. The key changes vs Stage 4a:

   FORWARD:
   1. Project x -> q_raw, k_raw, v  (standard matmul, autograd nodes)
   2. Apply RoPE rotation to q_raw and k_raw at current position:
      For each head h, for each dimension pair (2i, 2i+1):
        q_rot[2i]   = q_raw[2i] * cos(pos*θ_i) - q_raw[2i+1] * sin(pos*θ_i)
        q_rot[2i+1] = q_raw[2i] * sin(pos*θ_i) + q_raw[2i+1] * cos(pos*θ_i)
      Same for k_rot.
   3. Write ROTATED k into cache (so cached keys are position-aware)
   4. Compute attention scores using rotated q and cached rotated k
   5. Softmax + weighted sum of cached values (v is NOT rotated)
   6. Project through Wo

   BACKWARD:
   The gradients flow back through the attention mechanism as before,
   but at the boundary between rotated and unrotated space, we apply
   the INVERSE rotation (transpose of the rotation matrix):

     dq_raw[2i]   =  dq_rot[2i] * cos(pos*θ) + dq_rot[2i+1] * sin(pos*θ)
     dq_raw[2i+1] = -dq_rot[2i] * sin(pos*θ) + dq_rot[2i+1] * cos(pos*θ)

   This works because rotation matrices are orthogonal: R^T = R^{-1}.

   For cached keys at position t, the inverse rotation uses position t's
   angle (not the current position), because each key was rotated at its
   own position when it was cached.

   The q_rot array (plain float, not autograd) is captured by the backward
   closure because it's needed to compute dk gradients. *)
let fused_multi_head_attention x layer kv =
  (* Step 1: project input to q, k, v (autograd nodes) *)
  let q_raw = matmul layer.attn_wq x in
  let k_raw = matmul layer.attn_wk x in
  let v = matmul layer.attn_wv x in

  (* Current position = number of entries already in cache *)
  let pos = kv.len in

  (* Store UNROTATED autograd nodes for gradient routing in backward *)
  kv.k_nodes.(pos) <- k_raw;
  kv.v_nodes.(pos) <- v;

  (* Step 2: apply RoPE rotation to q and k *)
  let q_rot = Array.create_float n_embd in
  let cos_p = rope_cos.(pos) and sin_p = rope_sin.(pos) in
  for h = 0 to n_head - 1 do
    let h_off = h * head_dim in
    for i = 0 to half_dim - 1 do
      let j0 = h_off + 2 * i and j1 = h_off + 2 * i + 1 in
      let c = cos_p.(i) and s = sin_p.(i) in
      (* Rotate q: [cos -sin; sin cos] @ [q0; q1] *)
      q_rot.(j0) <- q_raw.data.(j0) *. c -. q_raw.data.(j1) *. s;
      q_rot.(j1) <- q_raw.data.(j0) *. s +. q_raw.data.(j1) *. c;
      (* Rotate k and write directly into cache *)
      kv.k_cache.(h).(pos * head_dim + 2 * i) <-
        k_raw.data.(j0) *. c -. k_raw.data.(j1) *. s;
      kv.k_cache.(h).(pos * head_dim + 2 * i + 1) <-
        k_raw.data.(j0) *. s +. k_raw.data.(j1) *. c
    done;
    (* Write v to cache unchanged (RoPE only applies to q and k) *)
    for j = 0 to head_dim - 1 do
      kv.v_cache.(h).(pos * head_dim + j) <- v.data.(h_off + j)
    done
  done;
  kv.len <- pos + 1;
  let t_len = pos + 1 in
  let scale = 1.0 /. sqrt (float_of_int head_dim) in

  (* Steps 3-5: attention using rotated q and cached rotated k *)
  let out_data = Array.create_float n_embd in
  let all_weights = Array.init n_head (fun _ -> Array.create_float t_len) in

  for h = 0 to n_head - 1 do
    let h_off = h * head_dim in
    let weights = all_weights.(h) in
    (* Attention scores: q_rot · k_rot_cached / sqrt(d) *)
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
    (* Softmax *)
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
    (* Weighted sum of cached values *)
    for j = 0 to head_dim - 1 do
      let s = ref 0.0 in
      for t = 0 to t_len - 1 do
        s := !s +. weights.(t) *. kv.v_cache.(h).(t * head_dim + j)
      done;
      out_data.(h_off + j) <- !s
    done
  done;

  (* Backward with RoPE inverse rotation *)
  let out_grad = Array.make n_embd 0.0 in
  let backward () =
    let dweight = Array.create_float t_len in
    (* Scratch buffer for accumulated dq in rotated space, per head *)
    let dq_rot_buf = Array.create_float head_dim in

    for h = 0 to n_head - 1 do
      let h_off = h * head_dim in
      let weights = all_weights.(h) in

      (* 1. Gradient through weighted sum → dweight_t, dv
         (values are not rotated, so this is unchanged from Stage 3/4a) *)
      for t = 0 to t_len - 1 do
        let s = ref 0.0 in
        for j = 0 to head_dim - 1 do
          let dy_j = out_grad.(h_off + j) in
          s := !s +. dy_j *. kv.v_cache.(h).(t * head_dim + j);
          kv.v_nodes.(t).grad.(h_off + j) <-
            kv.v_nodes.(t).grad.(h_off + j) +. weights.(t) *. dy_j
        done;
        dweight.(t) <- !s
      done;

      (* 2. Softmax backward → dscore (unchanged) *)
      let dot = ref 0.0 in
      for t = 0 to t_len - 1 do
        dot := !dot +. dweight.(t) *. weights.(t)
      done;

      (* 3. Score backward with RoPE inverse rotation.

         In rotated space:
           dq_rot[j] = sum_t(dscore_t * k_rot_cached[t][j])
           dk_rot_t[j] = dscore_t * q_rot[j]

         Then inverse-rotate to get gradients for the unrotated autograd nodes:
           dq_raw = R_pos^T * dq_rot     (inverse rotation at current position)
           dk_raw_t = R_t^T * dk_rot_t   (inverse rotation at position t) *)

      (* Zero the dq_rot scratch buffer for this head *)
      for j = 0 to head_dim - 1 do dq_rot_buf.(j) <- 0.0 done;

      for t = 0 to t_len - 1 do
        let dscore = weights.(t) *. (dweight.(t) -. !dot) *. scale in

        (* Accumulate dq in rotated space *)
        for j = 0 to head_dim - 1 do
          dq_rot_buf.(j) <- dq_rot_buf.(j)
            +. dscore *. kv.k_cache.(h).(t * head_dim + j)
        done;

        (* dk at position t: compute in rotated space, inverse-rotate immediately.
           dk_rot = dscore * q_rot, then multiply by R_t^T *)
        let cos_t = rope_cos.(t) and sin_t = rope_sin.(t) in
        for i = 0 to half_dim - 1 do
          let j0 = h_off + 2 * i and j1 = h_off + 2 * i + 1 in
          let dk_rot_0 = dscore *. q_rot.(h_off + 2 * i) in
          let dk_rot_1 = dscore *. q_rot.(h_off + 2 * i + 1) in
          (* R^T * [dk0; dk1] = [dk0*cos + dk1*sin; -dk0*sin + dk1*cos] *)
          kv.k_nodes.(t).grad.(j0) <- kv.k_nodes.(t).grad.(j0)
            +. dk_rot_0 *. cos_t.(i) +. dk_rot_1 *. sin_t.(i);
          kv.k_nodes.(t).grad.(j1) <- kv.k_nodes.(t).grad.(j1)
            -. dk_rot_0 *. sin_t.(i) +. dk_rot_1 *. cos_t.(i)
        done
      done;

      (* Inverse-rotate accumulated dq_rot into q_raw.grad at current position *)
      for i = 0 to half_dim - 1 do
        let j0 = h_off + 2 * i and j1 = h_off + 2 * i + 1 in
        q_raw.grad.(j0) <- q_raw.grad.(j0)
          +. dq_rot_buf.(2 * i) *. cos_p.(i)
          +. dq_rot_buf.(2 * i + 1) *. sin_p.(i);
        q_raw.grad.(j1) <- q_raw.grad.(j1)
          -. dq_rot_buf.(2 * i) *. sin_p.(i)
          +. dq_rot_buf.(2 * i + 1) *. cos_p.(i)
      done
    done
  in

  (* Children: q_raw + all cached k/v nodes for topo sort ordering *)
  let children = Array.init (1 + 2 * t_len) (fun i ->
    if i = 0 then q_raw
    else let idx = i - 1 in
      if idx < t_len then kv.k_nodes.(idx)
      else kv.v_nodes.(idx - t_len)
  ) in

  (* Step 6: project through Wo *)
  let attn_out = { id = fresh_id (); data = out_data; grad = out_grad;
    shape = [|n_embd|]; children; backward_fn = backward } in
  matmul layer.attn_wo attn_out

(* MLP block: FC1 (expand 4x) -> GeLU -> FC2 (contract) *)
let mlp_block x layer =
  matmul layer.mlp_fc1 x |> tensor_gelu |> fun h -> matmul layer.mlp_fc2 h

(* Transformer block: pre-norm with residual connections.
   x -> RMSNorm -> Attention(+RoPE) -> +x -> RMSNorm -> MLP(GeLU) -> +x *)
let transformer_block x layer kv =
  let x =
    tensor_rmsnorm x |> fun xn ->
    fused_multi_head_attention xn layer kv |> tensor_add x in
  tensor_rmsnorm x |> fun xn -> mlp_block xn layer |> tensor_add x

(* Full forward: embed token (no position), transformer layers, project to logits.
   Position is encoded implicitly via RoPE inside the attention mechanism. *)
let gpt_forward model token_id kv_caches =
  let x = embed_token model token_id in
  let x = Array.to_list model.layers
    |> List.mapi (fun li layer -> (li, layer))
    |> List.fold_left (fun x (li, layer) ->
      transformer_block x layer kv_caches.(li)) x in
  matmul model.lm_head x

(* ══════════════════════════════════════════════════════════════════════
   TRAINING
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

(* Cosine LR with linear warmup. *)
let get_lr step num_steps =
  if step < warmup_steps then
    learning_rate *. float_of_int step /. float_of_int warmup_steps
  else
    let progress =
      float_of_int (step - warmup_steps)
      /. float_of_int (num_steps - warmup_steps) in
    learning_rate *. 0.5 *. (1.0 +. cos (Float.pi *. progress))

(* Gradient clipping by global L2 norm. *)
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

(* Adam step with hoisted bias correction. *)
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

(* Compute mean NLL loss over all positions in a document. *)
let compute_loss model tokens kv_caches =
  let n = min block_size (Array.length tokens - 1) in
  let losses = Array.init n (fun _pos_id ->
    let logits = gpt_forward model tokens.(_pos_id) kv_caches in
    let probs = tensor_softmax logits in
    tensor_nll probs tokens.(_pos_id + 1)) in
  (tensor_mean losses, n)

(* Training loop: forward, backward, clip, Adam update. *)
let train model params docs char_to_id bos_id num_steps =
  let adam = init_adam params in
  for step = 0 to num_steps - 1 do
    let doc = docs.(step mod Array.length docs) in
    let tokens = tokenize char_to_id bos_id doc in
    let kv_caches = Array.init n_layer (fun _ -> make_kv_cache ()) in
    let (loss, _) = compute_loss model tokens kv_caches in
    backward loss;
    clip_grad_norm params;
    adam_step params adam step num_steps;
    if step = num_steps - 1 then
      Printf.printf "step %4d / %4d | loss %.4f\n" (step + 1) num_steps loss.data.(0)
  done

(* ══════════════════════════════════════════════════════════════════════
   INFERENCE
   ══════════════════════════════════════════════════════════════════════ *)

let generate_sample model uchars bos_id _vocab_size temperature =
  let kv_caches = Array.init n_layer (fun _ -> make_kv_cache ()) in
  let token_id = ref bos_id in
  let buf = Buffer.create 32 in
  let pos = ref 0 in
  while !pos < block_size do
    let logits = gpt_forward model !token_id kv_caches in
    let probs = tensor_scale logits (1.0 /. temperature) |> tensor_softmax in
    token_id := weighted_choice probs.data;
    if !token_id = bos_id then pos := block_size
    else begin Buffer.add_char buf uchars.(!token_id); incr pos end
  done;
  Buffer.contents buf

(* ══════════════════════════════════════════════════════════════════════
   MAIN
   ══════════════════════════════════════════════════════════════════════ *)

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
  train model params docs char_to_id bos_id 1000;
  Printf.printf "--- inference (new, hallucinated text) ---\n";
  for i = 1 to 20 do
    generate_sample model uchars bos_id vocab_size 0.5
    |> Printf.printf "sample %2d: %s\n" i
  done;
  Printf.printf "total time: %.2fs\n" (Sys.time () -. t_start)
