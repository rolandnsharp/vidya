(* ================================================================
   MicroGPT in OCaml — Mutable Record Version
   ================================================================

   A faithful translation of Karpathy's microgpt.py into OCaml,
   using mutable record fields for the autograd Value type.

   APPROACH: Mutable Records
   -------------------------
   This version uses OCaml's `mutable` keyword on record fields.
   Each Value node has `mutable data` and `mutable grad` that are
   modified in-place during forward and backward passes.

   If you know TypeScript, this is equivalent to:

     interface Value {
       data: number;    // mutated in optimizer step
       grad: number;    // mutated in backward pass
       readonly children: Value[];
       readonly localGrads: number[];
     }

   In OCaml, `mutable` on a record field means "this field can be
   reassigned with `<-`". Non-mutable fields are frozen at creation.

   WHY TWO VERSIONS?
   -----------------
   We implement microgpt twice to compare OCaml styles:

   1. THIS FILE (mutable records) — Imperative, direct translation.
      Uses `v.data <- 1.0` syntax. Closest to the JS/Python original.
      Uses explicit for-loops and sequential mutation.

   2. microgpt_ref.ml (ref cells + functional style) — Uses
      `float ref` fields and idiomatic OCaml: pipe operators,
      Array.map, Array.fold_left, etc. Same algorithm, different
      flavor.

   HOW TO COMPILE & RUN:
     ocamlfind ocamlopt microgpt_mutable.ml -o microgpt_mutable
     ./microgpt_mutable

   Requires input.txt in the current directory (the names dataset
   or any text corpus, one document per line).

   ================================================================ *)


(* ================================================================
   OCAML SURVIVAL GUIDE FOR TYPESCRIPT DEVS
   ================================================================

   Key syntax differences you'll see throughout this file:

   1. FLOAT vs INT operators:
      OCaml has NO implicit type coercion. Separate operators:
        Int:   + - * / mod          (like TS for integers)
        Float: +. -. *. /. **       (+. is float-add, etc.)
      In TS: 1 + 2.0 === 3.0       (auto-coerced)
      In OCaml: 1 + 2.0 is a TYPE ERROR.
      Convert with: float_of_int, int_of_float

   2. LET bindings:
      `let x = 5` is like `const x = 5` in TS.
      `let x = ref 5` is like `let x = { value: 5 }` — a mutable cell.
      `!x` dereferences a ref (reads the value).
      `x := 10` assigns to a ref.

   3. ARRAYS vs LISTS:
      `[| 1; 2; 3 |]` is an Array — mutable, fixed-size, O(1) access.
      `[1; 2; 3]` is a List — immutable, linked, O(n) access.
      Array is like TS arrays. List is like a linked list.

   4. SEMICOLONS:
      `;` is a SEQUENCE operator, not a statement terminator.
      `a; b` means "evaluate a (for side effects), then evaluate b".

   5. PATTERN MATCHING:
      `match x with | A -> ... | B -> ...` is like TS switch
      but exhaustive — the compiler warns if you miss a case.

   6. FUNCTION APPLICATION:
      `f x y` means `f(x, y)`. No parentheses needed.
      `f (g x)` means `f(g(x))`. Parentheses only for grouping.

   7. TYPE ANNOTATIONS:
      Usually optional — OCaml infers types. When present:
      `let f (x : int) : float = ...` like TS `function f(x: number): number`

   8. PIPE OPERATOR:
      `x |> f |> g` means `g(f(x))`. Like proposed TS pipeline.

   ================================================================ *)


(* ================================================================
   Section 1: Seeded PRNG
   ================================================================
   OCaml's Random module works fine, but we seed it for
   reproducible results. We also need a Gaussian (normal)
   distribution, which OCaml doesn't provide built-in.

   We use the Box-Muller transform — same math as the JS version.
   ================================================================ *)

(* Initialize the global PRNG with a fixed seed.
   Like: random.seed(42) in Python, or srand(42) in C. *)
let () = Random.init 42

(* Box-Muller transform: generate Gaussian-distributed random floats.
   Given uniform random numbers u,v in (-1,1), if u^2 + v^2 < 1,
   then u * sqrt(-2*ln(s)/s) is normally distributed.

   TypeScript equivalent:
     function randGauss(mean: number, std: number): number { ... }
*)
let random_gauss ?(mean = 0.0) ?(std = 1.0) () =
  (* `?mean` and `?std` are OPTIONAL ARGUMENTS with defaults.
     Called like: random_gauss ()           — uses defaults
                  random_gauss ~std:0.08 () — override std
     The final () is needed because OCaml functions must take
     at least one non-optional argument. *)
  let rec sample () =
    let u = Random.float 2.0 -. 1.0 in
    let v = Random.float 2.0 -. 1.0 in
    let s = u *. u +. v *. v in
    if s >= 1.0 || s = 0.0 then sample ()  (* reject and retry *)
    else mean +. std *. u *. sqrt (-2.0 *. log s /. s)
  in
  sample ()

(* Fisher-Yates shuffle — mutates array in place.
   TypeScript equivalent:
     function shuffle<T>(arr: T[]): void { ... }
*)
let shuffle arr =
  let n = Array.length arr in
  for i = n - 1 downto 1 do
    let j = Random.int (i + 1) in
    (* OCaml swap: no destructuring assignment, so use a temp *)
    let tmp = arr.(i) in
    arr.(i) <- arr.(j);
    arr.(j) <- tmp
  done
  (* `done` ends a for-loop. No return value — this is `unit`. *)

(* Weighted random choice — pick an index with probability
   proportional to its weight. Like random.choices() in Python.

   TypeScript equivalent:
     function weightedChoice(weights: number[]): number { ... }
*)
let weighted_choice weights =
  let total = Array.fold_left (+.) 0.0 weights in
  (* `Array.fold_left (+.) 0.0 weights` is like:
     `weights.reduce((acc, w) => acc + w, 0.0)` in TS *)
  let r = ref (Random.float total) in
  let result = ref (Array.length weights - 1) in
  let found = ref false in
  Array.iteri (fun i w ->
    if not !found then begin
      r := !r -. w;
      if !r <= 0.0 then begin
        result := i;
        found := true
      end
    end
  ) weights;
  !result


(* ================================================================
   Section 2: Data Loading & Tokenization
   ================================================================
   Load a text file where each line is a "document" (e.g. a name).
   Build a character-level tokenizer: each unique character gets
   an integer ID, plus a special BOS (Beginning of Sequence) token.
   ================================================================ *)

(* Read all non-empty lines from a file.
   TypeScript equivalent:
     fs.readFileSync('input.txt', 'utf-8').split('\n').filter(Boolean)
*)
let load_docs filename =
  if not (Sys.file_exists filename) then begin
    Printf.eprintf "Error: %s not found.\n" filename;
    Printf.eprintf "Download it or copy input.txt from the project root.\n";
    exit 1
  end;
  let ic = open_in filename in
  (* `open_in` opens a file for reading — like fs.openSync *)
  let docs = ref [] in
  begin try
    while true do
      let line = input_line ic |> String.trim in
      if String.length line > 0 then
        docs := line :: !docs
        (* `::` is list prepend — O(1), builds list in reverse *)
    done
  with End_of_file -> ()
  end;
  close_in ic;
  (* Convert to array and shuffle *)
  let arr = Array.of_list (List.rev !docs) in
  shuffle arr;
  arr

(* Build vocabulary: sorted unique characters from all documents.
   Returns (uchars, bos_id, vocab_size) where:
     - uchars.(i) = the character for token i
     - bos_id = the special Beginning-of-Sequence token ID
     - vocab_size = total number of tokens (chars + BOS)

   TypeScript equivalent:
     const uchars = [...new Set(docs.join(''))].sort()
*)
let build_vocab docs =
  (* Collect all unique characters using a Hashtbl as a set *)
  let char_set = Hashtbl.create 128 in
  Array.iter (fun doc ->
    String.iter (fun ch ->
      Hashtbl.replace char_set ch ()
      (* `Hashtbl.replace` with unit value = "add to set" *)
    ) doc
  ) docs;
  (* Extract keys, sort them *)
  let chars = Hashtbl.fold (fun ch () acc -> ch :: acc) char_set [] in
  let sorted = List.sort Char.compare chars in
  let uchars = Array.of_list sorted in
  let bos_id = Array.length uchars in
  let vocab_size = bos_id + 1 in
  (uchars, bos_id, vocab_size)

(* Tokenize a string: convert each character to its index in uchars.
   Like: doc.split('').map(ch => uchars.indexOf(ch))

   Returns an int array wrapped with BOS on both sides:
     [BOS, char_ids..., BOS]
*)
let tokenize uchars bos_id doc =
  let find_char ch =
    (* Linear search — fine for vocab_size < 100 *)
    let result = ref (-1) in
    Array.iteri (fun i c -> if c = ch then result := i) uchars;
    !result
  in
  let n = String.length doc in
  let tokens = Array.make (n + 2) bos_id in
  (* tokens.[0] = BOS (already set by Array.make) *)
  for i = 0 to n - 1 do
    tokens.(i + 1) <- find_char doc.[i]
    (* `doc.[i]` is OCaml's string character access — like doc[i] in TS *)
  done;
  (* tokens.[n+1] = BOS (already set by Array.make) *)
  tokens


(* ================================================================
   Section 3: Autograd — The Value Type
   ================================================================
   This is the heart of the system. A `value` is a node in a
   computation graph. It holds:
     - data: the scalar float computed in the forward pass
     - grad: the derivative of the loss w.r.t. this node (backward)
     - children: the input nodes to the operation that created this
     - local_grads: the local derivatives (chain rule multipliers)

   TypeScript equivalent:
     class Value {
       id: number;
       data: number;
       grad: number;
       children: Value[];
       localGrads: number[];
     }

   The `mutable` keyword means these fields can be reassigned:
     v.data <- 1.0    (* like v.data = 1.0 in TS *)
   Non-mutable fields are frozen at creation — like readonly in TS.
   ================================================================ *)

(* Auto-incrementing ID counter for unique node identification.
   Used in the backward pass to track which nodes we've visited. *)
let next_id = ref 0

let fresh_id () =
  let id = !next_id in
  (* `incr` increments a ref in place: incr x is x := !x + 1 *)
  incr next_id;
  id

(* The Value record type.
   `mutable` fields can be changed after creation.
   Non-mutable fields are fixed forever. *)
type value = {
  id : int;                    (* unique identifier for graph traversal *)
  mutable data : float;        (* forward pass: the computed value *)
  mutable grad : float;        (* backward pass: accumulated gradient *)
  children : value array;      (* inputs to the operation that created this *)
  local_grads : float array;   (* d(this)/d(child_i) for each child *)
}

(* Constructor: create a leaf Value (no children, no gradients).
   Used for model parameters and constants.

   TypeScript equivalent:
     new Value(data)   or   { data, grad: 0, children: [], localGrads: [] }
*)
let make_value data =
  { id = fresh_id (); data; grad = 0.0;
    children = [||]; local_grads = [||] }
  (* [||] is an empty array — like [] in TS *)

(* Constructor: create a Value that is the result of an operation.
   `children` are the input Values, `local_grads` are the local
   derivatives used in the chain rule during backpropagation. *)
let make_op data children local_grads =
  { id = fresh_id (); data; grad = 0.0; children; local_grads }


(* ================================================================
   Section 4: Autograd — Operations on Values
   ================================================================
   Each operation creates a new Value node that records:
     1. The computed result (forward pass)
     2. The children (inputs) and local gradients (for backward)

   The local gradient is: d(output)/d(input) evaluated at the
   current values. The chain rule will later multiply this by
   the upstream gradient to get the full derivative.

   TypeScript equivalent pattern:
     function vadd(a: Value, b: Value): Value {
       return new Value(
         a.data + b.data,
         [a, b],
         [1.0, 1.0]  // d(a+b)/da = 1, d(a+b)/db = 1
       );
     }
   ================================================================ *)

(* Addition: d(a+b)/da = 1, d(a+b)/db = 1 *)
let vadd a b =
  make_op (a.data +. b.data) [|a; b|] [|1.0; 1.0|]

(* Multiplication: d(a*b)/da = b, d(a*b)/db = a *)
let vmul a b =
  make_op (a.data *. b.data) [|a; b|] [|b.data; a.data|]

(* Power (float exponent): d(a^n)/da = n * a^(n-1) *)
let vpow a n =
  make_op (a.data ** n) [|a|] [|n *. a.data ** (n -. 1.0)|]

(* Natural log: d(ln(a))/da = 1/a *)
let vlog a =
  make_op (log a.data) [|a|] [|1.0 /. a.data|]

(* Exponential: d(exp(a))/da = exp(a) *)
let vexp a =
  let e = exp a.data in
  make_op e [|a|] [|e|]

(* ReLU: d(relu(a))/da = 1 if a > 0, else 0 *)
let vrelu a =
  make_op
    (Float.max 0.0 a.data)
    [|a|]
    [|if a.data > 0.0 then 1.0 else 0.0|]

(* Negation: -a = a * (-1) *)
let vneg a = vmul a (make_value (-1.0))

(* Subtraction: a - b = a + (-b) *)
let vsub a b = vadd a (vneg b)

(* Division: a / b = a * b^(-1) *)
let vdiv a b = vmul a (vpow b (-1.0))

(* Multiply a Value by a plain float (convenience).
   Like: value * 0.5 — wraps the float in a Value first. *)
let vscale a s = vmul a (make_value s)

(* Sum an array of Values. Reduces left-to-right with vadd.
   TypeScript equivalent:
     function vsum(arr: Value[]): Value {
       return arr.reduce((acc, v) => vadd(acc, v));
     }
*)
let vsum arr =
  (* `Array.fold_left f init arr` is `arr.reduce(f, init)` in TS *)
  let n = Array.length arr in
  if n = 0 then make_value 0.0
  else Array.fold_left vadd arr.(0) (Array.sub arr 1 (n - 1))
  (* `Array.sub arr 1 (n-1)` is like `arr.slice(1)` in TS *)


(* ================================================================
   Section 5: Autograd — Backward Pass (Backpropagation)
   ================================================================
   The backward pass computes gradients via the chain rule.
   Starting from the loss (grad = 1.0), it walks the computation
   graph in reverse topological order, accumulating gradients.

   Algorithm:
     1. Build topological ordering of all nodes (DFS post-order)
     2. Set loss.grad = 1.0
     3. For each node in reverse order:
        For each child:
          child.grad += local_grad * node.grad

   This is the SAME algorithm in all ML frameworks (PyTorch,
   TensorFlow, JAX) — just operating on scalars instead of tensors.
   ================================================================ *)

(* Topological sort via depth-first search.
   Returns nodes in dependency order (leaves first, loss last).

   Uses a Hashtbl as a "visited set" — keyed by the Value's unique ID.

   TypeScript equivalent:
     function topoSort(root: Value): Value[] {
       const visited = new Set<number>();
       const topo: Value[] = [];
       function dfs(v: Value) {
         if (!visited.has(v.id)) {
           visited.add(v.id);
           v.children.forEach(dfs);
           topo.push(v);
         }
       }
       dfs(root);
       return topo;
     }
*)
let topological_sort root =
  let visited = Hashtbl.create 256 in
  let topo = ref [] in
  let rec build_topo v =
    if not (Hashtbl.mem visited v.id) then begin
      Hashtbl.add visited v.id ();
      Array.iter build_topo v.children;
      topo := v :: !topo
      (* Prepend to list — we'll reverse at the end *)
    end
  in
  build_topo root;
  (* `List.rev` because we prepended — now it's leaves-first *)
  List.rev !topo

(* The backward pass: compute all gradients via reverse-mode autodiff.
   Mutates every Value's `grad` field in the computation graph.

   This is the function that makes training possible. Without it,
   we'd have no way to know which direction to adjust the weights.
*)
let backward loss =
  let topo = topological_sort loss in
  (* The loss's gradient w.r.t. itself is 1.0 by definition *)
  loss.grad <- 1.0;
  (* Walk in REVERSE topological order (loss first, leaves last).
     For each node, propagate its gradient to its children
     using the chain rule: child.grad += local_grad * parent.grad *)
  List.iter (fun v ->
    (* `Array.iteri` is like `arr.forEach((child, i) => ...)` in TS *)
    Array.iteri (fun i child ->
      child.grad <- child.grad +. v.local_grads.(i) *. v.grad
    ) v.children
  ) (List.rev topo)
  (* We reverse again because we want loss-first order *)


(* ================================================================
   Section 6: Model Architecture — Hyperparameters & Initialization
   ================================================================
   Define the transformer's shape and create all weight matrices.

   We use OCaml record types for the model weights instead of a
   string-keyed dictionary (like Python's state_dict). This gives
   us compile-time checking — a typo in a field name is a compiler
   error, not a silent runtime bug.

   TypeScript equivalent:
     interface LayerWeights {
       attn_wq: Value[][];
       attn_wk: Value[][];
       ...
     }
     interface Model {
       wte: Value[][];
       wpe: Value[][];
       lm_head: Value[][];
       layers: LayerWeights[];
     }
   ================================================================ *)

(* Hyperparameters — same as the original microgpt *)
let n_layer = 1        (* depth: number of transformer layers *)
let n_embd = 16        (* width: embedding dimension *)
let block_size = 16    (* maximum context length *)
let n_head = 4         (* number of attention heads *)
let head_dim = n_embd / n_head  (* dimension per head: 16/4 = 4 *)

(* Per-layer weight matrices *)
type layer_weights = {
  attn_wq : value array array;  (* query projection *)
  attn_wk : value array array;  (* key projection *)
  attn_wv : value array array;  (* value projection *)
  attn_wo : value array array;  (* output projection *)
  mlp_fc1 : value array array;  (* MLP first layer *)
  mlp_fc2 : value array array;  (* MLP second layer *)
}

(* Full model weights *)
type model = {
  wte : value array array;       (* token embedding table *)
  wpe : value array array;       (* position embedding table *)
  lm_head : value array array;   (* final output projection *)
  layers : layer_weights array;  (* per-layer weights *)
}

(* Initialize a random weight matrix of shape (nout, nin).
   Each entry is a Value initialized from a Gaussian distribution.

   TypeScript equivalent:
     function matrix(nout: number, nin: number, std = 0.08): Value[][] {
       return Array.from({length: nout}, () =>
         Array.from({length: nin}, () => new Value(randGauss(0, std)))
       );
     }
*)
let init_matrix ?(std = 0.08) nout nin =
  Array.init nout (fun _ ->
    Array.init nin (fun _ ->
      make_value (random_gauss ~std ())
    )
  )

(* Initialize the full model with all weight matrices. *)
let init_model vocab_size =
  {
    wte = init_matrix vocab_size n_embd;
    wpe = init_matrix block_size n_embd;
    lm_head = init_matrix vocab_size n_embd;
    layers = Array.init n_layer (fun _ -> {
      attn_wq = init_matrix n_embd n_embd;
      attn_wk = init_matrix n_embd n_embd;
      attn_wv = init_matrix n_embd n_embd;
      attn_wo = init_matrix n_embd n_embd;
      mlp_fc1 = init_matrix (4 * n_embd) n_embd;
      mlp_fc2 = init_matrix n_embd (4 * n_embd);
    });
  }

(* Collect ALL model parameters into a flat array.
   Needed for the optimizer, which iterates over all params.

   TypeScript equivalent:
     const params = Object.values(stateDict).flatMap(m => m.flat())
*)
let collect_params model =
  let matrix_to_list m =
    (* Flatten a 2D array into a 1D list *)
    Array.to_list m
    |> List.map Array.to_list
    |> List.flatten
    (* `|>` is the pipe operator: x |> f means f(x)
       Like the proposed TS pipeline: x |> f |> g *)
  in
  let layer_params l =
    List.concat [
      matrix_to_list l.attn_wq;
      matrix_to_list l.attn_wk;
      matrix_to_list l.attn_wv;
      matrix_to_list l.attn_wo;
      matrix_to_list l.mlp_fc1;
      matrix_to_list l.mlp_fc2;
    ]
  in
  List.concat [
    matrix_to_list model.wte;
    matrix_to_list model.wpe;
    matrix_to_list model.lm_head;
    Array.to_list model.layers
    |> List.map layer_params
    |> List.flatten;
  ]
  |> Array.of_list
  (* Convert to array for O(1) indexed access in the optimizer *)


(* ================================================================
   Section 7: Model Architecture — Layer Functions
   ================================================================
   Each function below implements one building block of the
   transformer. They're intentionally small and named so they
   can be individually tested, replaced, or modified in future
   stages.

   The architecture follows GPT-2 with minor simplifications:
     - RMSNorm instead of LayerNorm (no bias, no mean subtraction)
     - ReLU instead of GeLU (simpler, same idea)
     - No bias terms anywhere
   ================================================================ *)

(* Linear layer: matrix-vector multiply.
   Takes input vector x and weight matrix w.
   Returns w @ x (each row of w dot-producted with x).

   This is the fundamental building block — the "neuron".
   Every layer in the transformer is built from linear transforms.

   TypeScript equivalent:
     function linear(x: Value[], w: Value[][]): Value[] {
       return w.map(row => row.reduce((sum, wi, i) => vadd(sum, vmul(wi, x[i])), ZERO));
     }
*)
let linear x w =
  Array.map (fun w_row ->
    (* Dot product: sum of element-wise products *)
    let products = Array.mapi (fun i wi -> vmul wi x.(i)) w_row in
    vsum products
  ) w

(* Softmax: convert logits to probabilities.
   Subtracts the max for numerical stability (prevents exp overflow).

   Math: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))

   TypeScript equivalent:
     function softmax(logits: Value[]): Value[] {
       const maxVal = Math.max(...logits.map(v => v.data));
       const exps = logits.map(v => vexp(vsub(v, maxVal)));
       const total = vsum(exps);
       return exps.map(e => vdiv(e, total));
     }
*)
let softmax logits =
  (* Find the maximum value (as a plain float, for stability) *)
  let max_val = Array.fold_left
    (fun acc v -> Float.max acc v.data) neg_infinity logits
  in
  (* Subtract max and exponentiate *)
  let exps = Array.map (fun v ->
    vexp (vsub v (make_value max_val))
  ) logits in
  (* Normalize by the sum *)
  let total = vsum exps in
  Array.map (fun e -> vdiv e total) exps

(* RMS Normalization: normalize a vector by its root-mean-square.
   Simpler than LayerNorm (no learned bias/scale, no mean subtraction).

   Math: rmsnorm(x) = x / sqrt(mean(x^2) + epsilon)

   The epsilon (1e-5) prevents division by zero.
*)
let rmsnorm x =
  let n = Array.length x in
  (* Mean of squared elements *)
  let mean_sq =
    vsum (Array.map (fun xi -> vmul xi xi) x)
    |> fun ms -> vdiv ms (make_value (float_of_int n))
  in
  (* Scale factor: 1 / sqrt(mean_sq + eps) *)
  let scale = vpow (vadd mean_sq (make_value 1e-5)) (-0.5) in
  (* Multiply each element by the scale *)
  Array.map (fun xi -> vmul xi scale) x

(* Add two vectors element-wise.
   Used for residual connections: x_new = layer(x) + x_original *)
let vector_add a b =
  Array.init (Array.length a) (fun i -> vadd a.(i) b.(i))

(* Single attention head.
   Takes the query vector for this head, plus all cached keys and
   values for this head, and returns the attention-weighted output.

   This is the core of the transformer: "which past tokens should
   I pay attention to, and what should I extract from them?"

   Args:
     q_h:  query vector for this head, shape [head_dim]
     k_h:  list of key vectors (one per past token), each [head_dim]
     v_h:  list of value vectors (one per past token), each [head_dim]

   Returns: weighted combination of values, shape [head_dim]
*)
let attention_head q_h k_h v_h =
  let n_cached = List.length k_h in
  (* Compute attention scores: dot(query, each key) / sqrt(head_dim) *)
  let scale = sqrt (float_of_int head_dim) in
  let attn_logits = List.map (fun kt ->
    let dot = Array.init head_dim (fun j ->
      vmul q_h.(j) kt.(j)
    ) |> vsum in
    vdiv dot (make_value scale)
  ) k_h |> Array.of_list in
  (* Convert scores to probabilities *)
  let attn_weights = softmax attn_logits in
  (* Weighted sum of value vectors *)
  let v_arr = Array.of_list v_h in
  Array.init head_dim (fun j ->
    Array.init n_cached (fun t ->
      vmul attn_weights.(t) v_arr.(t).(j)
    ) |> vsum
  )

(* Multi-head attention block.
   Splits Q, K, V into heads, runs attention_head on each,
   concatenates the results, and projects back to n_embd.

   Args:
     x:       input vector [n_embd]
     layer:   this layer's weight matrices
     keys:    KV cache for keys — list of [n_embd] vectors, one per past token
     values:  KV cache for values — same shape

   Mutates keys and values (appends new entries for this token).
   Returns: output vector [n_embd]
*)
let multi_head_attention x layer keys values =
  (* Project input to queries, keys, values *)
  let q = linear x layer.attn_wq in
  let k = linear x layer.attn_wk in
  let v = linear x layer.attn_wv in
  (* Append this token's K and V to the cache *)
  keys := k :: !keys;
  values := v :: !values;
  (* Split into heads, compute attention, concatenate *)
  let x_attn = Array.make n_embd (make_value 0.0) in
  for h = 0 to n_head - 1 do
    let hs = h * head_dim in
    (* Extract this head's slice of Q *)
    let q_h = Array.sub q hs head_dim in
    (* Extract this head's slice from all cached K and V *)
    let k_h = List.rev !keys |> List.map (fun ki -> Array.sub ki hs head_dim) in
    let v_h = List.rev !values |> List.map (fun vi -> Array.sub vi hs head_dim) in
    (* Run attention for this head *)
    let head_out = attention_head q_h k_h v_h in
    (* Copy head output into the concatenated result *)
    Array.blit head_out 0 x_attn hs head_dim
    (* `Array.blit src srcpos dst dstpos len` copies elements.
       Like: dst.splice(dstpos, len, ...src.slice(srcpos, srcpos+len)) *)
  done;
  (* Project concatenated heads back to n_embd *)
  linear x_attn layer.attn_wo

(* MLP (feed-forward) block.
   Two linear layers with ReLU activation in between.
   The inner dimension is 4x the embedding dimension.

   Math: mlp(x) = W2 @ relu(W1 @ x)
*)
let mlp_block x layer =
  let hidden = linear x layer.mlp_fc1 in
  let activated = Array.map vrelu hidden in
  linear activated layer.mlp_fc2

(* Single transformer block: attention + MLP, both with
   residual connections and RMS normalization.

   This is the repeating unit of the transformer.
   Stack N of these and you get a deeper model.
*)
let transformer_block x layer keys values =
  (* --- Attention sub-block --- *)
  let x_residual = x in
  let x_norm = rmsnorm x in
  let x_attn = multi_head_attention x_norm layer keys values in
  let x = vector_add x_attn x_residual in
  (* --- MLP sub-block --- *)
  let x_residual = x in
  let x_norm = rmsnorm x in
  let x_mlp = mlp_block x_norm layer in
  vector_add x_mlp x_residual


(* ================================================================
   Section 8: Model Architecture — GPT Forward Pass
   ================================================================
   The full forward pass: embedding -> normalize -> N transformer
   blocks -> output logits.

   Takes a single token ID and position, returns logits over the
   full vocabulary (scores for what the next token should be).
   ================================================================ *)

(* Embed a token: look up its token embedding and position embedding,
   add them together, and normalize. *)
let embed_token model token_id pos_id =
  let tok_emb = model.wte.(token_id) in
  let pos_emb = model.wpe.(pos_id) in
  let x = vector_add tok_emb pos_emb in
  rmsnorm x

(* Full GPT forward pass for a single token.
   Returns logits: an array of vocab_size Values, one score per
   possible next token.

   Args:
     model:       all weight matrices
     token_id:    the input token (integer)
     pos_id:      position in the sequence (integer)
     layer_keys:  array of KV caches, one per layer (mutated)
     layer_values: same for values
*)
let gpt_forward model token_id pos_id layer_keys layer_values =
  (* Start with token + position embedding *)
  let x = embed_token model token_id pos_id in
  (* Pass through each transformer layer *)
  let x = ref x in
  for li = 0 to n_layer - 1 do
    x := transformer_block !x model.layers.(li)
           layer_keys.(li) layer_values.(li)
  done;
  (* Final projection to vocabulary logits *)
  linear !x model.lm_head


(* ================================================================
   Section 9: Training — Adam Optimizer & Training Loop
   ================================================================
   Adam is the standard optimizer for neural networks.
   It maintains running averages of the gradient (momentum)
   and the squared gradient (adaptive learning rate).

   The training loop:
     1. Pick a document from the dataset
     2. Tokenize it
     3. For each position, predict the next token
     4. Compute loss (how wrong we were)
     5. Backpropagate (compute gradients)
     6. Update weights with Adam
   ================================================================ *)

(* Adam optimizer state: first and second moment buffers *)
type adam_state = {
  m : float array;   (* first moment: running average of gradients *)
  v : float array;   (* second moment: running average of gradient squares *)
}

let init_adam n_params = {
  m = Array.make n_params 0.0;
  v = Array.make n_params 0.0;
}

(* Adam optimizer hyperparameters *)
let learning_rate = 0.01
let beta1 = 0.85     (* momentum decay *)
let beta2 = 0.99     (* squared gradient decay *)
let eps_adam = 1e-8   (* prevents division by zero *)

(* Single Adam update step.
   Updates all parameter values based on their accumulated gradients.
   Then resets all gradients to 0 for the next step.

   The learning rate decays linearly from `learning_rate` to 0
   over the course of training.
*)
let adam_step params adam step num_steps =
  let lr_t = learning_rate *. (1.0 -. float_of_int step /. float_of_int num_steps) in
  Array.iteri (fun i p ->
    (* Update moment estimates *)
    adam.m.(i) <- beta1 *. adam.m.(i) +. (1.0 -. beta1) *. p.grad;
    adam.v.(i) <- beta2 *. adam.v.(i) +. (1.0 -. beta2) *. p.grad *. p.grad;
    (* Bias correction (accounts for zero-initialization of moments) *)
    let m_hat = adam.m.(i) /. (1.0 -. beta1 ** float_of_int (step + 1)) in
    let v_hat = adam.v.(i) /. (1.0 -. beta2 ** float_of_int (step + 1)) in
    (* Update the parameter *)
    p.data <- p.data -. lr_t *. m_hat /. (sqrt v_hat +. eps_adam);
    (* Reset gradient for next iteration *)
    p.grad <- 0.0
  ) params

(* Compute the cross-entropy loss for one document.
   For each position, we predict the next token and measure
   how wrong we were using -log(probability_of_correct_token).

   Returns (loss_value, n_positions) *)
let compute_loss model tokens layer_keys layer_values =
  let n = min block_size (Array.length tokens - 1) in
  let losses = Array.init n (fun pos_id ->
    let token_id = tokens.(pos_id) in
    let target_id = tokens.(pos_id + 1) in
    let logits = gpt_forward model token_id pos_id layer_keys layer_values in
    let probs = softmax logits in
    (* Cross-entropy loss: -log(prob of correct answer) *)
    vneg (vlog probs.(target_id))
  ) in
  (* Average loss over all positions *)
  let loss = vdiv (vsum losses) (make_value (float_of_int n)) in
  (loss, n)

(* Training loop: train for `num_steps` steps on the dataset. *)
let train model params docs uchars bos_id num_steps =
  let adam = init_adam (Array.length params) in
  for step = 0 to num_steps - 1 do
    (* Pick a document (cycling through the dataset) *)
    let doc = docs.(step mod Array.length docs) in
    let tokens = tokenize uchars bos_id doc in
    (* Fresh KV caches for this document *)
    let layer_keys = Array.init n_layer (fun _ -> ref []) in
    let layer_values = Array.init n_layer (fun _ -> ref []) in
    (* Forward pass: compute loss *)
    let (loss, _n) = compute_loss model tokens layer_keys layer_values in
    (* Backward pass: compute gradients *)
    backward loss;
    (* Optimizer step: update weights *)
    adam_step params adam step num_steps;
    (* Print progress (overwrite same line with \r) *)
    Printf.printf "\rstep %4d / %4d | loss %.4f%!" (step + 1) num_steps loss.data
    (* `%!` flushes stdout — needed for \r to work *)
  done;
  Printf.printf "\n"


(* ================================================================
   Section 10: Inference — Text Generation
   ================================================================
   After training, we use the model to generate new text.
   Starting from the BOS token, we repeatedly:
     1. Run the forward pass to get logits
     2. Convert to probabilities with temperature scaling
     3. Sample a token from the distribution
     4. Stop if we hit BOS (end of sequence)
   ================================================================ *)

let generate_sample model uchars bos_id vocab_size temperature =
  let layer_keys = Array.init n_layer (fun _ -> ref []) in
  let layer_values = Array.init n_layer (fun _ -> ref []) in
  let token_id = ref bos_id in
  let sample = Buffer.create 32 in
  (* `Buffer` is OCaml's string builder — like TS's string concatenation
     but more efficient for building strings character by character. *)
  let pos_id = ref 0 in
  let continue = ref true in
  while !continue && !pos_id < block_size do
    let logits = gpt_forward model !token_id !pos_id layer_keys layer_values in
    (* Temperature scaling: divide logits by temperature before softmax.
       Low temperature = more confident/repetitive.
       High temperature = more random/creative. *)
    let scaled = Array.map (fun l -> vdiv l (make_value temperature)) logits in
    let probs = softmax scaled in
    (* Sample from the probability distribution *)
    let weights = Array.map (fun p -> p.data) probs in
    token_id := weighted_choice weights;
    if !token_id = bos_id then
      continue := false  (* Hit end-of-sequence *)
    else begin
      Buffer.add_char sample uchars.(!token_id);
      incr pos_id
    end
  done;
  Buffer.contents sample

(* Generate multiple samples and print them. *)
let inference model uchars bos_id vocab_size n_samples temperature =
  Printf.printf "--- inference (new, hallucinated text) ---\n";
  for i = 1 to n_samples do
    let text = generate_sample model uchars bos_id vocab_size temperature in
    Printf.printf "sample %2d: %s\n" i text
  done


(* ================================================================
   Section 11: Main — Entry Point
   ================================================================
   Wire everything together: load data, build model, train, infer.

   In OCaml, there's no `main()` function. Top-level code runs
   when the program starts. We use `let () = ...` as convention
   for "this is the entry point."
   ================================================================ *)

let () =
  (* Load dataset *)
  let docs = load_docs "input.txt" in
  Printf.printf "num docs: %d\n" (Array.length docs);

  (* Build tokenizer *)
  let (uchars, bos_id, vocab_size) = build_vocab docs in
  Printf.printf "vocab size: %d\n" vocab_size;

  (* Initialize model *)
  let model = init_model vocab_size in
  let params = collect_params model in
  Printf.printf "num params: %d\n" (Array.length params);

  (* Train *)
  let num_steps = 1000 in
  train model params docs uchars bos_id num_steps;

  (* Generate samples *)
  let temperature = 0.5 in
  inference model uchars bos_id vocab_size 20 temperature
