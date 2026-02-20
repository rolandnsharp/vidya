(* 6_microgpt_blas.ml — Stage 6: BLAS-Accelerated Matrix Multiply
   ================================================================

   Based on Stage 5 (scaled model: 4 layers, 64-dim, 207K params).
   The ONLY change is replacing our hand-rolled matmul loops with
   calls to OpenBLAS's cblas_dgemm via OCaml's C FFI.

   WHY BLAS
   ========
   Stage 5 took 143s. The dominant cost is matrix multiplication:
   each training step does ~25 matmuls (forward) + ~50 BLAS calls
   (backward: each matmul has da and db gradients) = ~75 BLAS calls
   per step × 2000 steps = 150,000 matmul operations total.

   Our pure OCaml matmul is a scalar triple loop — it processes one
   multiply-add per clock cycle. OpenBLAS uses:
   - AVX2/AVX-512: processes 4-8 doubles per instruction
   - Cache blocking: keeps working data in L1/L2 cache
   - Register tiling: maximizes FPU utilization
   - CPU-specific micro-kernels: hand-tuned assembly per architecture

   At 64×64, this gives ~5-10x speedup on the matmul alone.

   WHAT CHANGED
   ============
   1. Added: external blas_dgemm declaration (1 line)
   2. Changed: matmul forward — single blas_dgemm call replaces triple loop
   3. Changed: matmul backward — two blas_dgemm calls replace two triple loops
   4. Changed: matmul now works on BATCHED positions (matrix × matrix)
      instead of matrix × vector, so we can feed all positions at once

   Wait — actually we keep the matrix × vector interface for now since
   our autograd processes one position at a time through the KV cache.
   The BLAS speedup comes from the inner matmul, not from batching.

   Everything else is identical to Stage 5.

   COMPILE
   =======
   ocamlopt -O2 -o microgpt_blas \
     blas_stubs.c 6_microgpt_blas.ml \
     -ccopt "-I/usr/include/x86_64-linux-gnu" \
     -cclib -lopenblas

   RUN
   ===
   ./microgpt_blas *)

let () = Random.init 42

(* ══════════════════════════════════════════════════════════════════════
   BLAS FFI
   ══════════════════════════════════════════════════════════════════════

   This single external declaration is the bridge to OpenBLAS.
   The C stub (blas_stubs.c) wraps cblas_dgemm with three modes:

     op=0: C[m,n]  = A[m,k] @ B[k,n]      (forward, overwrites C)
     op=1: C[m,k] += A[m,n] @ B[k,n]^T    (backward da, accumulates)
     op=2: C[k,n] += A[m,k]^T @ B[m,n]    (backward db, accumulates)

   The "bytecode" string is for ocamlc; the "native" string for ocamlopt.
   Both point to C functions in blas_stubs.c. *)
external blas_dgemm
  : int -> int -> int -> int
    -> float array -> float array -> float array
    -> unit
  = "caml_dgemm_byte" "caml_dgemm"

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

(* ── Matrix-vector multiply: [m,n] @ [n] -> [m] ──────────────────────

   THIS IS THE KEY CHANGE FROM STAGE 5.

   Old (Stage 5): hand-rolled scalar triple loop
     for i = 0 to m-1 do
       for j = 0 to n-1 do
         y.(i) += w.(i,j) * x.(j)    ← one multiply-add per iteration
       done
     done

   New (Stage 6): OpenBLAS dgemm
     Treats the vector x[n] as a matrix x[n,1], so the matmul becomes
     [m,n] @ [n,1] -> [m,1]. BLAS processes 4-8 multiply-adds per SIMD
     instruction and uses cache-optimal access patterns.

   The backward pass similarly uses BLAS:
     dw[m,n] += dy[m,1] @ x[1,n]    (outer product → op=2, A^T @ B)
     dx[n]   += w[m,n]^T @ dy[m,1]  (matvec      → op=1, A @ B^T...
                                       but we need w^T @ dy, which is
                                       op=2 with swapped roles)

   Actually, for the backward:
     dw_{i,j} += dy_i * x_j  →  dw[m,n] += dy[m] ⊗ x[n] (outer product)
       This is: dw[m,n] += dy[m,1] @ x[1,n]
       Using op=0 with m=m, n=n, k=1: C[m,n] = A[m,1] @ B[1,n]
       But op=0 overwrites! We need accumulate (beta=1).
       Use op=2: C[k,n] += A[m,k]^T @ B[m,n]
       With m_blas=m, k_blas=1, n_blas=n: C[1,n] += A[m,1]^T @ B[m,n]
       Hmm, that gives [1,n] not [m,n].

   Let me think about this differently. The BLAS ops are:
     op=0: C[m,n]  = A[m,k] @ B[k,n]
     op=1: C[m,k] += A[m,n] @ B[k,n]^T
     op=2: C[k,n] += A[m,k]^T @ B[m,n]

   For dw[m,n] += dy[m] ⊗ x[n]:
     This is an outer product. In BLAS terms, treat dy as [m,1] and x as [1,n]:
     C[m,n] += A[m,1] @ B[1,n]
     Use a direct dgemm(N, N, m, n, 1, 1.0, dy, 1, x, n, 1.0, dw, n)
     None of our three ops do beta=1 with NoTrans,NoTrans...
     BUT op=2 with different dims: C[k,n] += A[m,k]^T @ B[m,n]
     Set m_blas=1, k_blas=m, n_blas=n:
       C[m,n] += A[1,m]^T @ B[1,n]  — nope, that's C[m,n] += [m,1] @ [1,n]
       Wait: A is [m_blas, k_blas] = [1, m], A^T is [m, 1].
       B is [m_blas, n_blas] = [1, n].
       C is [k_blas, n_blas] = [m, n].
       So: C[m,n] += A[1,m]^T @ B[1,n] = [m,1] @ [1,n] ✓
     Yes! op=2 with (m=1, k=m_orig, n=n_orig) gives us the outer product.

   For dx[n] += w[m,n]^T @ dy[m]:
     dx[n,1] += w[m,n]^T @ dy[m,1]  →  [n,1] += [n,m] @ [m,1]
     Use op=2: C[k,n] += A[m,k]^T @ B[m,n]
     Set m_blas=m, k_blas=n, n_blas=1:
       C[n,1] += A[m,n]^T @ B[m,1] = w^T[n,m] @ dy[m,1] ✓
     Yes! op=2 with (m=m_orig, k=n_orig, n=1).

   So both backward ops use op=2 with different dimension mappings.
   This is clean — the BLAS call handles transposition internally via
   its CblasTrans flag, avoiding any explicit transpose allocation. *)
let matmul w x =
  let m = w.shape.(0) and n = w.shape.(1) in

  (* Forward: y[m] = W[m,n] @ x[n]
     Treat as: C[m,1] = A[m,n] @ B[n,1]
     blas_dgemm op=0 m=m n=1 k=n *)
  let y_data = Array.create_float m in
  blas_dgemm 0 m 1 n w.data x.data y_data;

  let y_grad = Array.make m 0.0 in
  let backward () =
    (* dw[m,n] += dy[m] ⊗ x[n]  (outer product)
       Treat as: C[m,n] += A[1,m]^T @ B[1,n]
       blas_dgemm op=2 m=1 k=m n=n
       A = dy[m] viewed as [1,m], B = x[n] viewed as [1,n], C = dw[m,n] *)
    blas_dgemm 2 1 n m y_grad x.data w.grad;

    (* dx[n] += W[m,n]^T @ dy[m]
       Treat as: C[n,1] += A[m,n]^T @ B[m,1]
       blas_dgemm op=2 m=m k=n n=1
       A = W[m,n], B = dy[m] viewed as [m,1], C = dx[n] viewed as [n,1] *)
    blas_dgemm 2 m 1 n w.data y_grad x.grad
  in
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
   gelu(x) = 0.5 * x * (1 + tanh(a * (x + b*x³)))
   where a = sqrt(2/π) ≈ 0.7979, b = 0.044715.
   Backward: dgelu/dx = 0.5*(1+t) + 0.5*x*(1-t²)*a*(1+3*b*x²) *)
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
   MODEL DEFINITION

   Identical to Stage 5: 4 layers, 64-dim, 4 heads, 64 context, ~207K params.
   ══════════════════════════════════════════════════════════════════════ *)

let n_layer = 4
let n_embd = 64
let block_size = 64
let n_head = 4
let head_dim = n_embd / n_head   (* = 16 *)
let half_dim = head_dim / 2      (* = 8 rotation pairs for RoPE *)

(* ── RoPE Tables ──────────────────────────────────────────────────────
   Pre-computed rotation tables. θ_i = 1/10000^(2i/head_dim).
   With head_dim=16: 8 frequency bands from θ_0=1.0 (fast rotation)
   down to θ_7≈0.0000056 (essentially stationary). *)

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

(* KV cache: stores rotated k and raw v per head per position.
   k_nodes/v_nodes hold the autograd graph nodes for backward. *)
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
   FORWARD PASS
   ══════════════════════════════════════════════════════════════════════ *)

(* Embed a token: token embedding + RMSNorm (no position embedding;
   position info comes from RoPE in the attention layer). *)
let embed_token model token_id =
  tensor_row model.wte token_id |> tensor_rmsnorm

(* Fused multi-head attention with RoPE and pre-allocated KV cache.

   NOTE: The attention inner loops (score computation, softmax, weighted
   sum over values) are NOT replaced with BLAS. These operate per-head
   with head_dim=16 and sequence length up to 64 — too small for BLAS
   overhead to pay off. The win comes from the weight matmuls (Wq, Wk,
   Wv, Wo) which go through our BLAS-accelerated `matmul` function. *)
let fused_multi_head_attention x layer kv =
  (* These three matmuls are now BLAS-accelerated: [64,64] @ [64] *)
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
  (* Attention scores, softmax, weighted value sum — kept as scalar loops
     because head_dim=16 is too small for BLAS to beat loop overhead. *)
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
  (* Backward: RoPE inverse rotation for dk, dq gradients *)
  let out_grad = Array.make n_embd 0.0 in
  let backward () =
    let dweight = Array.create_float t_len in
    let dq_rot_buf = Array.create_float head_dim in
    for h = 0 to n_head - 1 do
      let h_off = h * head_dim in
      let weights = all_weights.(h) in
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
      let dot = ref 0.0 in
      for t = 0 to t_len - 1 do
        dot := !dot +. dweight.(t) *. weights.(t)
      done;
      for j = 0 to head_dim - 1 do dq_rot_buf.(j) <- 0.0 done;
      for t = 0 to t_len - 1 do
        let dscore = weights.(t) *. (dweight.(t) -. !dot) *. scale in
        for j = 0 to head_dim - 1 do
          dq_rot_buf.(j) <- dq_rot_buf.(j)
            +. dscore *. kv.k_cache.(h).(t * head_dim + j)
        done;
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
  (* This matmul is also BLAS-accelerated *)
  matmul layer.attn_wo attn_out

(* MLP block: FC1 (expand 4x: 64 -> 256) -> GeLU -> FC2 (contract: 256 -> 64)
   Both matmuls here are BLAS-accelerated. The fc1 matmul is [256,64] @ [64],
   which is 16,384 multiply-adds — the largest single matmul in the model. *)
let mlp_block x layer =
  matmul layer.mlp_fc1 x |> tensor_gelu |> fun h -> matmul layer.mlp_fc2 h

(* Transformer block: pre-norm attention + pre-norm MLP, both with residuals. *)
let transformer_block x layer kv =
  let x =
    tensor_rmsnorm x |> fun xn ->
    fused_multi_head_attention xn layer kv |> tensor_add x in
  tensor_rmsnorm x |> fun xn -> mlp_block xn layer |> tensor_add x

(* Full forward: embed → 4 transformer layers → project to vocab logits. *)
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

(* Adam with hoisted bias correction (bc1, bc2 computed once per step). *)
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

(* Training loop: same as Stage 5, progress every 500 steps. *)
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
