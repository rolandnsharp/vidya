(* 3_microgpt_optimized.ml — Stage 3: Optimized Tensor Autograd
   ================================================================

   A character-level GPT trained from scratch with a custom autograd engine.
   Based on Stage 2 (tensor autograd), with targeted allocation optimizations
   that bring runtime from ~1.5s down to ~0.5s.

   Architecture: 1-layer transformer, 16-dim embeddings, 4 attention heads,
   16-token context window. ~5,888 parameters. Trains on input.txt (one
   sentence per line) for 1000 steps, then generates 20 samples.

   Optimizations applied (vs Stage 2):
   1. PRE-ALLOCATED KV CACHE — the big one. Replaces per-position list
      accumulation with fixed float arrays. Eliminates ~1300 small array
      allocations per training step (tensor_slice, tensor_stack, tensor_concat).
   2. HOISTED ADAM BIAS CORRECTION — beta1^step and beta2^step computed once
      per step instead of 11,776 times (once per parameter).
   3. HASHTABLE TOKENIZE — O(1) char->int lookup instead of O(n) linear scan.
   4. TOPO SORT FIX — removed unnecessary double List.rev.
   5. FUSED SOFTMAX — exp+sum in one pass (3 passes instead of 4).
   6. Array.create_float — avoids closure overhead in Array.init.

   Compile: ocamlopt -O2 -o microgpt_optimized 3_microgpt_optimized.ml
   Run:     ./microgpt_optimized *)

let () = Random.init 42

(* ══════════════════════════════════════════════════════════════════════
   UTILITIES
   ══════════════════════════════════════════════════════════════════════ *)

(* Box-Muller transform: generates normally distributed random numbers
   from uniform samples. Rejection-samples until (u,v) lands inside the
   unit circle, then applies the transform. *)
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

(* Multinomial sampling from an unnormalized weight array.
   Draws a uniform random number in [0, total_weight) and walks the
   array until the cumulative weight exceeds it. Returns the index. *)
let weighted_choice weights =
  let total = Array.fold_left (+.) 0.0 weights in
  let r = Random.float total in
  Array.fold_left (fun (chosen, remaining) w ->
    if remaining <= 0.0 then (chosen, remaining)
    else if remaining -. w <= 0.0 then (chosen, remaining -. w)
    else (chosen + 1, remaining -. w)
  ) (0, r) weights
  |> fst

(* Load training data: one document (sentence) per line. Blank lines
   are skipped. The array is shuffled so training sees documents in
   random order. *)
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

(* Build vocabulary from all unique characters across all documents.
   Returns: (uchars array for decoding, bos_id, vocab_size).
   bos_id serves as both BOS and EOS token — it's the first ID past
   the character range. vocab_size = num_chars + 1. *)
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

(* [OPT #3] Build a char->int hashtable for O(1) tokenization.
   Stage 2 used Array.to_list + List.find per character — O(vocab_size)
   per char, allocating ~160 objects per character tokenized. *)
let build_char_to_id uchars =
  let tbl = Hashtbl.create (Array.length uchars) in
  Array.iteri (fun i ch -> Hashtbl.replace tbl ch i) uchars;
  tbl

(* Tokenize a document string into an int array: [BOS; char_ids...; BOS].
   Array is pre-allocated at the right size and filled with bos_id, so
   positions 0 and n+1 are automatically correct. *)
let tokenize char_to_id bos_id doc =
  let n = String.length doc in
  let result = Array.make (n + 2) bos_id in
  for i = 0 to n - 1 do
    result.(i + 1) <- Hashtbl.find char_to_id doc.[i]
  done;
  result

(* ══════════════════════════════════════════════════════════════════════
   TENSOR AUTOGRAD ENGINE

   Each node in the computation graph holds:
   - data:        forward pass output (flat float array)
   - grad:        accumulated gradient (same size as data, zeroed initially)
   - shape:       logical dimensions (e.g. [|16|] for a vector, [|80;16|] for a matrix)
   - children:    input nodes (for topological sort ordering)
   - backward_fn: closure that computes d(loss)/d(inputs) from d(loss)/d(output)

   The graph is built implicitly during the forward pass. Each operation
   creates a new node whose backward_fn captures references to its inputs.
   Calling `backward loss` topologically sorts the graph and calls each
   node's backward_fn in reverse dependency order (root first, leaves last).
   ══════════════════════════════════════════════════════════════════════ *)

let next_id = ref 0
let fresh_id () = let id = !next_id in incr next_id; id

type value = {
  id : int;                    (* unique node ID for topo sort visited set *)
  data : float array;          (* forward output *)
  grad : float array;          (* accumulated gradient, same length as data *)
  shape : int array;           (* logical shape, e.g. [|rows; cols|] *)
  children : value array;      (* input nodes — defines graph edges *)
  backward_fn : unit -> unit;  (* gradient computation closure *)
}

(* Create a leaf node (trainable parameter). No children, no backward. *)
let make_param shape data =
  { id = fresh_id (); data; grad = Array.make (Array.length data) 0.0;
    shape; children = [||]; backward_fn = (fun () -> ()) }

(* Placeholder node for uninitialized KV cache slots. Never actually
   used in computation — slots are overwritten before they're read. *)
let dummy_node = { id = -1; data = [||]; grad = [||]; shape = [||];
                   children = [||]; backward_fn = (fun () -> ()) }

(* ══════════════════════════════════════════════════════════════════════
   TENSOR OPERATIONS

   Each function below implements both forward and backward for one op.
   The pattern is always:
   1. Compute output data (forward)
   2. Allocate grad array (zeros)
   3. Define backward closure (captures inputs by reference)
   4. Return new value node with children set to inputs

   [OPT #6] All use Array.create_float instead of Array.init to avoid
   per-element closure dispatch overhead.

   Gradient formulas are documented inline. All gradients accumulate (+=)
   because a node may be used multiple times in the graph.
   ══════════════════════════════════════════════════════════════════════ *)

(* Matrix-vector multiply: [m,n] @ [n] -> [m]
   y_i = sum_j w_{i,j} * x_j
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

(* Element-wise add: [n] + [n] -> [n]
   Backward: da_i += dy_i,  db_i += dy_i  (gradient passes through unchanged) *)
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

(* Multiply by scalar constant: [n] * s -> [n]
   Backward: dx_i += dy_i * s *)
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

(* Element-wise ReLU: max(0, x_i) -> [n]
   Backward: dx_i += dy_i if x_i > 0, else 0  (subgradient at 0 is 0) *)
let tensor_relu x =
  let n = Array.length x.data in
  let y_data = Array.create_float n in
  for i = 0 to n - 1 do y_data.(i) <- Float.max 0.0 x.data.(i) done;
  let y_grad = Array.make n 0.0 in
  let backward () =
    for i = 0 to n - 1 do
      if x.data.(i) > 0.0 then
        x.grad.(i) <- x.grad.(i) +. y_grad.(i)
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

(* [OPT #5] Softmax with fused exp+sum pass (3 passes instead of 4).
   Pass 1: find max (for numerical stability)
   Pass 2: compute exp(x_i - max) AND accumulate sum in one loop
   Pass 3: normalize by 1/sum
   Backward: dx_i += y_i * (dy_i - sum_j(dy_j * y_j))
   This is the Jacobian-vector product for softmax. *)
let tensor_softmax x =
  let n = Array.length x.data in
  (* Pass 1: find max for numerical stability *)
  let max_val = ref neg_infinity in
  for i = 0 to n - 1 do
    if x.data.(i) > !max_val then max_val := x.data.(i)
  done;
  (* Pass 2: exp and sum in one loop *)
  let y_data = Array.create_float n in
  let sum_exp = ref 0.0 in
  for i = 0 to n - 1 do
    let e = exp (x.data.(i) -. !max_val) in
    y_data.(i) <- e;
    sum_exp := !sum_exp +. e
  done;
  (* Pass 3: normalize *)
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

(* RMS normalization: y_i = x_i / rms,  where rms = sqrt(mean(x^2) + eps)
   Simpler than LayerNorm (no learned scale/bias, no mean subtraction).
   Backward: dx_j = (dy_j - y_j * mean(dy * y)) / rms
   Derivation: chain rule through x/rms, with rms depending on all x_i. *)
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

(* Extract one row from a 2D parameter matrix: [m,n] -> [n]
   Used for embedding lookup (wte, wpe). The row is copied out.
   Backward: only the extracted row's gradient gets updated. *)
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

(* Negative log-likelihood loss for a single prediction.
   loss = -log(probs[target])
   Backward: d(probs[target]) += -1/probs[target] * dy
   Only the target index gets gradient — all others are zero. *)
let tensor_nll probs target =
  let y_data = [| -. log probs.data.(target) |] in
  let y_grad = [| 0.0 |] in
  let backward () =
    probs.grad.(target) <- probs.grad.(target)
      -. y_grad.(0) /. probs.data.(target) in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|1|]; children = [|probs|]; backward_fn = backward }

(* Mean of scalar [1] losses — used to average NLL across positions.
   Backward: each loss gets dy / n *)
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

   Topological sort via DFS, then iterate in root-to-leaf order.
   Each node's backward_fn reads its own grad and writes to its
   children's grads (accumulating via +=).

   [OPT #4] Stage 2 did: topological_sort returns List.rev !topo,
   then backward does topo |> List.rev |> List.iter — two reversals
   that cancel out. Now we return !topo directly (root-first order)
   and iterate without reversing.
   ══════════════════════════════════════════════════════════════════════ *)

(* Build topological ordering via DFS. Children are visited before the
   current node is prepended, so !topo naturally has root first and
   leaves last — exactly the order needed for backward. *)
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

(* Backpropagation: set root gradient to 1.0, then propagate gradients
   from root toward leaves. Each backward_fn reads its own node's grad
   and accumulates into its children's grads. *)
let backward loss =
  let topo = topological_sort loss in
  loss.grad.(0) <- 1.0;
  List.iter (fun v -> v.backward_fn ()) topo

(* ══════════════════════════════════════════════════════════════════════
   MODEL DEFINITION

   Tiny GPT architecture matching Karpathy's microgpt:
   - 1 transformer layer
   - 16-dimensional embeddings
   - 4 attention heads (head_dim = 4)
   - 16-token context window (block_size)
   - Character-level vocabulary (~80 chars + BOS token)
   - ~5,888 total parameters
   ══════════════════════════════════════════════════════════════════════ *)

let n_layer = 1
let n_embd = 16
let block_size = 16
let n_head = 4
let head_dim = n_embd / n_head   (* = 4 *)

(* Each transformer layer has 6 weight matrices:
   - attn_wq/wk/wv: [n_embd, n_embd] — project input to queries/keys/values
   - attn_wo: [n_embd, n_embd] — project concatenated head outputs back
   - mlp_fc1: [4*n_embd, n_embd] — MLP expand (16 -> 64)
   - mlp_fc2: [n_embd, 4*n_embd] — MLP contract (64 -> 16) *)
type layer_weights = {
  attn_wq : value; attn_wk : value;
  attn_wv : value; attn_wo : value;
  mlp_fc1 : value; mlp_fc2 : value;
}

(* [OPT #1] Pre-allocated KV cache.

   In Stage 2, each position's k and v were accumulated in a `value list ref`
   per layer. At each new position, every head sliced every cached k/v
   (tensor_slice), stacked them into matrices (tensor_stack), and concatenated
   head outputs (tensor_concat). This created ~1300 small array allocations
   per training step.

   The optimized cache stores k/v data directly in pre-allocated float arrays
   indexed by [head][position * head_dim + dim]. The autograd nodes for each
   position's k and v are stored separately in k_nodes/v_nodes so backward
   can route gradients back through the projection matrices.

   k_cache.(h).(t * head_dim + j) = key data for head h, position t, dim j
   v_cache.(h).(t * head_dim + j) = value data for head h, position t, dim j
   k_nodes.(t) = the autograd node for position t's full key vector [n_embd]
   v_nodes.(t) = the autograd node for position t's full value vector [n_embd] *)
type kv_cache = {
  k_cache : float array array;   (* [n_head][block_size * head_dim] *)
  v_cache : float array array;
  k_nodes : value array;         (* [block_size] — autograd nodes for backprop *)
  v_nodes : value array;
  mutable len : int;             (* number of positions filled so far *)
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
  wpe : value;               (* position embeddings: [block_size, n_embd] *)
  lm_head : value;           (* output projection: [vocab_size, n_embd] *)
  layers : layer_weights array;
}

(* Initialize a weight matrix with small random values (std=0.08).
   Stored as a flat 1D array in row-major order. *)
let init_matrix ?(std = 0.08) nout nin =
  let data = Array.init (nout * nin) (fun _ -> random_gauss ~std ()) in
  make_param [|nout; nin|] data

let init_model vocab_size =
  let wte = init_matrix vocab_size n_embd in
  let wpe = init_matrix block_size n_embd in
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
  { wte; wpe; lm_head; layers }

(* Collect all trainable parameters into a flat array for the optimizer. *)
let collect_params model =
  let layer_params l =
    [l.attn_wq; l.attn_wk; l.attn_wv; l.attn_wo; l.mlp_fc1; l.mlp_fc2] in
  [model.wte; model.wpe; model.lm_head]
  @ (model.layers |> Array.to_list |> List.map layer_params |> List.flatten)
  |> Array.of_list

(* ══════════════════════════════════════════════════════════════════════
   FORWARD PASS
   ══════════════════════════════════════════════════════════════════════ *)

(* Embed a single token: look up token embedding + position embedding,
   then RMS-normalize. Returns a [n_embd] vector. *)
let embed_token model token_id pos_id =
  let tok = tensor_row model.wte token_id in
  let pos = tensor_row model.wpe pos_id in
  tensor_add tok pos |> tensor_rmsnorm

(* [OPT #1] Fused multi-head attention with pre-allocated KV cache.

   This replaces three separate functions from Stage 2:
   - multi_head_attention (which called tensor_slice per head per cached k/v)
   - attention_head (which called tensor_stack and matmul_tv)
   - tensor_concat (which concatenated head outputs)

   FORWARD:
   1. Project input x through Wq, Wk, Wv to get q, k, v vectors [n_embd]
   2. Write k and v data into the pre-allocated cache at the current position
   3. For each attention head h:
      a. Compute attention scores: score_t = (q_h . k_t_h) / sqrt(d)
         where q_h and k_t_h are the head-h slices of q and cached k
      b. Apply softmax over all positions 0..t_len-1 (causal: only past + current)
      c. Compute weighted sum of cached values: out_h = sum_t(weight_t * v_t_h)
   4. Write all head outputs directly into one [n_embd] array (no concat needed)
   5. Project through Wo

   BACKWARD:
   The backward closure knows:
   - The attention weights per head (stored in all_weights)
   - The cached k/v float data (in kv.k_cache, kv.v_cache)
   - References to the q, k, v autograd nodes (for gradient routing)

   It computes gradients in reverse through:
   1. Weighted sum → dweight (how much each position's weight should change)
                   → dv (gradient for each cached value)
   2. Softmax → dscore (Jacobian-vector product)
   3. Score computation → dq, dk (gradient for query and each cached key)

   CHILDREN:
   The children array includes q plus ALL k_nodes and v_nodes for positions
   0..t_len-1. This ensures topological sort processes this node's backward
   BEFORE the k/v nodes' backward functions, which propagate gradients to
   the Wk/Wv weight matrices. Without this, gradients would be lost. *)
let fused_multi_head_attention x layer kv =
  (* Step 1: compute q, k, v projections (these are autograd nodes) *)
  let q = matmul layer.attn_wq x in
  let k = matmul layer.attn_wk x in
  let v = matmul layer.attn_wv x in

  (* Step 2: write k, v float data into cache at current position.
     Also store the autograd node references for backward gradient routing. *)
  let pos = kv.len in
  kv.k_nodes.(pos) <- k;
  kv.v_nodes.(pos) <- v;
  for h = 0 to n_head - 1 do
    for j = 0 to head_dim - 1 do
      kv.k_cache.(h).(pos * head_dim + j) <- k.data.(h * head_dim + j);
      kv.v_cache.(h).(pos * head_dim + j) <- v.data.(h * head_dim + j)
    done
  done;
  kv.len <- pos + 1;
  let t_len = pos + 1 in  (* number of positions to attend over *)
  let scale = 1.0 /. sqrt (float_of_int head_dim) in

  (* Step 3-4: compute attention output for all heads into one array *)
  let out_data = Array.create_float n_embd in
  let all_weights = Array.init n_head (fun _ -> Array.create_float t_len) in

  for h = 0 to n_head - 1 do
    let h_off = h * head_dim in    (* offset into the n_embd-sized vectors *)
    let weights = all_weights.(h) in

    (* 3a. Compute attention scores: q_h . k_t_h for each cached position t *)
    let max_score = ref neg_infinity in
    for t = 0 to t_len - 1 do
      let s = ref 0.0 in
      for j = 0 to head_dim - 1 do
        s := !s +. kv.k_cache.(h).(t * head_dim + j) *. q.data.(h_off + j)
      done;
      let score = !s *. scale in
      weights.(t) <- score;
      if score > !max_score then max_score := score
    done;

    (* 3b. Softmax: fused exp+sum then normalize *)
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

    (* 3c. Weighted sum of cached values for this head *)
    for j = 0 to head_dim - 1 do
      let s = ref 0.0 in
      for t = 0 to t_len - 1 do
        s := !s +. weights.(t) *. kv.v_cache.(h).(t * head_dim + j)
      done;
      (* 4. Write directly to the correct offset — no concat needed *)
      out_data.(h_off + j) <- !s
    done
  done;

  (* Backward: compute gradients for q, and for each cached k/v *)
  let out_grad = Array.make n_embd 0.0 in
  let backward () =
    (* Scratch array for per-position weight gradients, reused across heads *)
    let dweight = Array.create_float t_len in
    for h = 0 to n_head - 1 do
      let h_off = h * head_dim in
      let weights = all_weights.(h) in

      (* 1. Gradient through weighted sum: out_h_j = sum_t(w_t * v_t_h_j)
         dweight_t = sum_j(dy_h_j * v_cache[h][t*d+j])
         dv[t][h*d+j] += w_t * dy_h_j *)
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

      (* 2. Gradient through softmax: dscore_t = w_t * (dweight_t - dot)
         where dot = sum_t(dweight_t * w_t) *)
      let dot = ref 0.0 in
      for t = 0 to t_len - 1 do
        dot := !dot +. dweight.(t) *. weights.(t)
      done;

      (* 3. Gradient through score = (q_h . k_t_h) * scale
         dq[h*d+j] += dscore_t * k_cache[h][t*d+j]
         dk[t][h*d+j] += dscore_t * q[h*d+j] *)
      for t = 0 to t_len - 1 do
        let dscore = weights.(t) *. (dweight.(t) -. !dot) *. scale in
        for j = 0 to head_dim - 1 do
          q.grad.(h_off + j) <- q.grad.(h_off + j)
            +. dscore *. kv.k_cache.(h).(t * head_dim + j);
          kv.k_nodes.(t).grad.(h_off + j) <-
            kv.k_nodes.(t).grad.(h_off + j) +. dscore *. q.data.(h_off + j)
        done
      done
    done
  in

  (* Children array: [q, k_0, k_1, ..., k_{t-1}, v_0, v_1, ..., v_{t-1}]
     This ensures the topo sort places this node before all k/v nodes in
     backward order, so gradients are accumulated before k/v backward runs. *)
  let children = Array.init (1 + 2 * t_len) (fun i ->
    if i = 0 then q
    else let idx = i - 1 in
      if idx < t_len then kv.k_nodes.(idx)
      else kv.v_nodes.(idx - t_len)
  ) in

  (* Step 5: project through Wo *)
  let attn_out = { id = fresh_id (); data = out_data; grad = out_grad;
    shape = [|n_embd|]; children; backward_fn = backward } in
  matmul layer.attn_wo attn_out

(* MLP block: x -> FC1 -> ReLU -> FC2
   FC1 expands from n_embd to 4*n_embd, FC2 contracts back. *)
let mlp_block x layer =
  matmul layer.mlp_fc1 x |> tensor_relu |> fun h -> matmul layer.mlp_fc2 h

(* One transformer block: pre-norm architecture with residual connections.
   x -> RMSNorm -> MultiHeadAttn -> +x -> RMSNorm -> MLP -> +x *)
let transformer_block x layer kv =
  let x =
    tensor_rmsnorm x |> fun xn ->
    fused_multi_head_attention xn layer kv |> tensor_add x in
  tensor_rmsnorm x |> fun xn -> mlp_block xn layer |> tensor_add x

(* Full forward pass: embed token, run through all transformer layers,
   project to vocabulary logits via lm_head. *)
let gpt_forward model token_id pos_id kv_caches =
  let x = embed_token model token_id pos_id in
  let x = Array.to_list model.layers
    |> List.mapi (fun li layer -> (li, layer))
    |> List.fold_left (fun x (li, layer) ->
      transformer_block x layer kv_caches.(li)) x in
  matmul model.lm_head x

(* ══════════════════════════════════════════════════════════════════════
   TRAINING

   Adam optimizer with linear learning rate warmdown.
   Trains on one document per step (no batching).
   ══════════════════════════════════════════════════════════════════════ *)

type adam_state = {
  m : float array;   (* first moment (momentum) — one entry per parameter *)
  v : float array;   (* second moment (RMSprop) — one entry per parameter *)
}
let learning_rate = 0.01
let beta1 = 0.85     (* momentum decay — lower than standard 0.9 for small model *)
let beta2 = 0.99     (* variance decay — lower than standard 0.999 for small model *)
let eps_adam = 1e-8

let init_adam params =
  let total = Array.fold_left (fun acc p -> acc + Array.length p.data) 0 params in
  { m = Array.make total 0.0; v = Array.make total 0.0 }

(* [OPT #2] Adam step with hoisted bias correction.
   Standard Adam computes m_hat = m/(1-beta1^t) and v_hat = v/(1-beta2^t).
   Stage 2 computed beta1^t and beta2^t inside the inner loop — 11,776
   redundant exponentiation calls per step. Now computed once per step. *)
let adam_step params adam step num_steps =
  (* Linear warmdown: lr decreases from learning_rate to 0 over training *)
  let lr_t = learning_rate *. (1.0 -. float_of_int step /. float_of_int num_steps) in
  (* Bias correction factors — computed ONCE, used for all parameters *)
  let bc1 = 1.0 /. (1.0 -. beta1 ** float_of_int (step + 1)) in
  let bc2 = 1.0 /. (1.0 -. beta2 ** float_of_int (step + 1)) in
  let offset = ref 0 in
  params |> Array.iter (fun p ->
    let n = Array.length p.data in
    for i = 0 to n - 1 do
      let fi = !offset + i in
      (* Update biased first moment estimate *)
      adam.m.(fi) <- beta1 *. adam.m.(fi) +. (1.0 -. beta1) *. p.grad.(i);
      (* Update biased second moment estimate *)
      adam.v.(fi) <- beta2 *. adam.v.(fi) +. (1.0 -. beta2) *. p.grad.(i) *. p.grad.(i);
      (* Bias-corrected estimates *)
      let m_hat = adam.m.(fi) *. bc1 in
      let v_hat = adam.v.(fi) *. bc2 in
      (* Parameter update *)
      p.data.(i) <- p.data.(i) -. lr_t *. m_hat /. (sqrt v_hat +. eps_adam);
      (* Zero gradient for next step *)
      p.grad.(i) <- 0.0
    done;
    offset := !offset + n)

(* Compute mean cross-entropy loss over all positions in a document.
   For each position, feeds the token at that position, predicts the next
   token, and computes NLL loss. All positions share the same KV cache
   (autoregressive: position t sees positions 0..t). *)
let compute_loss model tokens kv_caches =
  let n = min block_size (Array.length tokens - 1) in
  let losses = Array.init n (fun pos_id ->
    let logits = gpt_forward model tokens.(pos_id) pos_id kv_caches in
    let probs = tensor_softmax logits in
    tensor_nll probs tokens.(pos_id + 1)) in
  (tensor_mean losses, n)

(* Main training loop: for each step, pick a document, tokenize it,
   build the computation graph (forward), backpropagate, and update
   parameters with Adam. *)
let train model params docs char_to_id bos_id num_steps =
  let adam = init_adam params in
  for step = 0 to num_steps - 1 do
    let doc = docs.(step mod Array.length docs) in
    let tokens = tokenize char_to_id bos_id doc in
    (* Fresh KV cache per training step — each document is independent *)
    let kv_caches = Array.init n_layer (fun _ -> make_kv_cache ()) in
    let (loss, _) = compute_loss model tokens kv_caches in
    backward loss;
    adam_step params adam step num_steps;
    if step = num_steps - 1 then
      Printf.printf "step %4d / %4d | loss %.4f\n" (step + 1) num_steps loss.data.(0)
  done

(* ══════════════════════════════════════════════════════════════════════
   INFERENCE

   Autoregressive text generation: start with BOS token, feed each
   predicted token back as input. The KV cache persists across positions
   within a single sample (no recomputation). Temperature controls
   randomness: lower = more deterministic, higher = more creative.
   ══════════════════════════════════════════════════════════════════════ *)

let generate_sample model uchars bos_id _vocab_size temperature =
  let kv_caches = Array.init n_layer (fun _ -> make_kv_cache ()) in
  let token_id = ref bos_id in
  let buf = Buffer.create 32 in
  let pos = ref 0 in
  while !pos < block_size do
    let logits = gpt_forward model !token_id !pos kv_caches in
    (* Scale logits by 1/temperature before softmax for sampling *)
    let probs = tensor_scale logits (1.0 /. temperature) |> tensor_softmax in
    token_id := weighted_choice probs.data;
    (* Stop if BOS/EOS token is generated *)
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
