(* 5_microgpt_scaled.ml — Stage 5: Scaled-Up Model
   ================================================================

   Based on Stage 4b (RoPE, GeLU, cosine LR, gradient clipping).
   This stage scales the model from toy size to something that should
   produce recognizably English output.

   SCALE CHANGES
   =============
   Before (4b)          After (5)            Ratio
   ─────────────        ─────────────        ─────
   1 layer              4 layers             4x depth
   16 embedding dim     64 embedding dim     4x width
   4 heads, d=4         4 heads, d=16        4x head dim
   16 context           64 context           4x context
   ~5,600 params        ~207,000 params      37x parameters
   1000 steps           2000 steps           2x training

   WHY THESE NUMBERS
   =================
   - 4 layers: enough depth for multi-level abstraction (character →
     character patterns → word-level features → phrase patterns)
   - 64 embedding dim: 16x capacity per position vs Stage 4b. Each
     position can represent much richer information.
   - 4 heads with head_dim=16: each head has 8 RoPE rotation pairs
     (vs 2 before), giving much finer position encoding. 16-dim
     attention subspace is enough to learn meaningful query-key patterns.
   - 64 context: sees up to 64 characters of history. Most English
     words are 4-8 characters, so this is ~8-16 words of context.
     Enough for local coherence within a sentence.
   - 2000 steps: with 37x more parameters, the model needs more
     training. 2000 steps sees 2000 documents from the 16K available.

   EXPECTED RESULTS
   ================
   - Loss should converge lower than Stage 4b (~1.2-1.5 range)
   - Samples should contain more recognizable English words
   - Some word-level patterns (articles, prepositions, common words)
   - Training time: ~2-5 minutes on a modern CPU (vs 0.6s for Stage 4b)

   Compile: ocamlopt -O2 -o microgpt_scaled 5_microgpt_scaled.ml
   Run:     ./microgpt_scaled *)

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

(* Fisher-Yates shuffle. *)
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
   This is now the performance-critical operation. At 64-dim embeddings,
   each matmul does 64*64 = 4096 multiply-adds (forward), and another
   4096 each for dw and dx (backward). With 6 matmuls per layer,
   4 layers, and 64 positions, that's ~3M multiply-adds per step just
   for the weight matmuls. Still fast enough for pure OCaml at this scale;
   BLAS would help at 128+ dimensions.
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
  let visited = Hashtbl.create 1024 in  (* larger table for bigger graph *)
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

   [STAGE 5 CHANGE] Scaled-up architecture:
   - 4 transformer layers (was 1) — enough depth for hierarchical features
   - 64-dim embeddings (was 16) — 16x more capacity per position
   - 4 heads with head_dim=16 (was 4) — richer attention patterns
   - 64-token context (was 16) — sees ~8-16 words of history
   - ~207K total parameters (was ~5.6K)
   ══════════════════════════════════════════════════════════════════════ *)

let n_layer = 4              (* 4 transformer layers *)
let n_embd = 64              (* 64-dim embeddings *)
let block_size = 64          (* 64-token context window *)
let n_head = 4               (* 4 attention heads *)
let head_dim = n_embd / n_head   (* = 16 dimensions per head *)
let half_dim = head_dim / 2      (* = 8 rotation pairs per head for RoPE *)

(* ── RoPE Tables ──────────────────────────────────────────────────────

   With head_dim=16 (half_dim=8), each head now has 8 rotation pairs
   spanning a wide frequency range:

     θ_0 = 1.0          — completes ~10 full rotations over 64 positions
     θ_1 ≈ 0.178        — ~1.8 rotations
     θ_2 ≈ 0.0316       — ~0.3 rotations (~115°)
     θ_3 ≈ 0.00562      — ~20° total rotation
     θ_4 ≈ 0.001        — ~3.7°
     θ_5 ≈ 0.000178     — barely rotates
     θ_6 ≈ 0.0000316    — barely rotates
     θ_7 ≈ 0.00000562   — essentially stationary

   This gives the model a rich multi-scale encoding: high-frequency pairs
   distinguish adjacent positions, low-frequency pairs encode coarse
   position within the context. Much better resolution than Stage 4b's
   2 rotation pairs. *)

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

(* KV cache: rotated k data, unrotated v data, unrotated autograd nodes.
   With 4 layers × 4 heads × 64 positions × 16 head_dim, the total
   cache size is 4 × 4 × 64 × 16 = 16,384 floats (~128 KB). *)
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
  wte : value;               (* token embeddings: [vocab_size, n_embd] *)
  lm_head : value;           (* output projection: [vocab_size, n_embd] *)
  layers : layer_weights array;
}

(* Initialize weights with std=0.08. For a 64-dim model, Xavier init
   would be 1/sqrt(64) ≈ 0.125, so 0.08 is slightly conservative —
   a reasonable starting point. *)
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
    (* MLP with 4x expansion: 64 -> 256 -> 64 *)
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

(* Embed a token: just token embedding + RMSNorm (no position embedding). *)
let embed_token model token_id =
  tensor_row model.wte token_id |> tensor_rmsnorm

(* Fused multi-head attention with RoPE and pre-allocated KV cache.
   See Stage 4b for full RoPE explanation. At this scale (4 heads,
   head_dim=16, up to 64 positions), the attention computation involves
   up to 4 × 16 × 64 = 4096 multiply-adds per position for scores,
   plus the same for the value weighted sum. *)
let fused_multi_head_attention x layer kv =
  let q_raw = matmul layer.attn_wq x in
  let k_raw = matmul layer.attn_wk x in
  let v = matmul layer.attn_wv x in
  let pos = kv.len in
  kv.k_nodes.(pos) <- k_raw;
  kv.v_nodes.(pos) <- v;
  (* Apply RoPE rotation to q and k *)
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
  (* Forward: attention using rotated q and cached rotated k *)
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
  (* Backward with RoPE inverse rotation *)
  let out_grad = Array.make n_embd 0.0 in
  let backward () =
    let dweight = Array.create_float t_len in
    let dq_rot_buf = Array.create_float head_dim in
    for h = 0 to n_head - 1 do
      let h_off = h * head_dim in
      let weights = all_weights.(h) in
      (* dweight and dv *)
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
      (* softmax backward *)
      let dot = ref 0.0 in
      for t = 0 to t_len - 1 do
        dot := !dot +. dweight.(t) *. weights.(t)
      done;
      (* Score backward with RoPE inverse rotation *)
      for j = 0 to head_dim - 1 do dq_rot_buf.(j) <- 0.0 done;
      for t = 0 to t_len - 1 do
        let dscore = weights.(t) *. (dweight.(t) -. !dot) *. scale in
        for j = 0 to head_dim - 1 do
          dq_rot_buf.(j) <- dq_rot_buf.(j)
            +. dscore *. kv.k_cache.(h).(t * head_dim + j)
        done;
        (* dk: inverse-rotate at position t *)
        let cos_t = rope_cos.(t) and sin_t = rope_sin.(t) in
        for i = 0 to half_dim - 1 do
          let j0 = h_off + 2 * i and j1 = h_off + 2 * i + 1 in
          let dk0 = dscore *. q_rot.(h_off + 2 * i) in
          let dk1 = dscore *. q_rot.(h_off + 2 * i + 1) in
          kv.k_nodes.(t).grad.(j0) <- kv.k_nodes.(t).grad.(j0)
            +. dk0 *. cos_t.(i) +. dk1 *. sin_t.(i);
          kv.k_nodes.(t).grad.(j1) <- kv.k_nodes.(t).grad.(j1)
            -. dk0 *. sin_t.(i) +. dk1 *. cos_t.(i)
        done
      done;
      (* dq: inverse-rotate at current position *)
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
  let children = Array.init (1 + 2 * t_len) (fun i ->
    if i = 0 then q_raw
    else let idx = i - 1 in
      if idx < t_len then kv.k_nodes.(idx)
      else kv.v_nodes.(idx - t_len)
  ) in
  let attn_out = { id = fresh_id (); data = out_data; grad = out_grad;
    shape = [|n_embd|]; children; backward_fn = backward } in
  matmul layer.attn_wo attn_out

(* MLP block: FC1 (expand 4x: 64 -> 256) -> GeLU -> FC2 (contract: 256 -> 64) *)
let mlp_block x layer =
  matmul layer.mlp_fc1 x |> tensor_gelu |> fun h -> matmul layer.mlp_fc2 h

(* Transformer block: pre-norm with residual connections.
   The residual connections are essential at 4 layers — without them,
   gradients would vanish through the depth. With residuals, the gradient
   has a direct path back through each block via the skip connection. *)
let transformer_block x layer kv =
  let x =
    tensor_rmsnorm x |> fun xn ->
    fused_multi_head_attention xn layer kv |> tensor_add x in
  tensor_rmsnorm x |> fun xn -> mlp_block xn layer |> tensor_add x

(* Full forward: embed token, 4 transformer layers, project to logits. *)
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

(* Compute mean NLL loss over all positions in a document.
   With block_size=64, processes up to 64 next-token predictions
   per document (fewer for shorter documents). *)
let compute_loss model tokens kv_caches =
  let n = min block_size (Array.length tokens - 1) in
  let losses = Array.init n (fun _pos_id ->
    let logits = gpt_forward model tokens.(_pos_id) kv_caches in
    let probs = tensor_softmax logits in
    tensor_nll probs tokens.(_pos_id + 1)) in
  (tensor_mean losses, n)

(* [STAGE 5 CHANGE] Training loop prints progress every 500 steps
   since training takes longer with the scaled model. *)
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
    if (step + 1) mod 500 = 0 then
      Printf.printf "step %4d / %4d | loss %.4f\n%!" (step + 1) num_steps loss.data.(0)
  done

(* ══════════════════════════════════════════════════════════════════════
   INFERENCE
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
  train model params docs char_to_id bos_id 2000;
  Printf.printf "--- inference (new, hallucinated text) ---\n";
  for i = 1 to 20 do
    generate_sample model uchars bos_id vocab_size 0.5
    |> Printf.printf "sample %2d: %s\n" i
  done;
  Printf.printf "total time: %.2fs\n" (Sys.time () -. t_start)
