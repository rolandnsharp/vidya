(* tensor.ml — GPU-backed autograd engine
   ========================================

   Reverse-mode automatic differentiation over GPU float32 tensors.
   Each value node holds device pointers (Gpu.gpu_buf) for data and
   gradients, plus shape metadata and a backward closure on the CPU.

   The OCaml side builds the computation graph and orchestrates
   backward passes. All numerical work dispatches to CUDA kernels
   via the Gpu module.

   This is the heart of Vidya's 103M parameter transformer.
   Blog context: we're building a new kind of LLM with memory —
   one that can retrain its frontal layers online after each
   interaction. Schuurmans showed LLMs are universal computers
   bottlenecked by constant compute per token. Our architecture
   gives variable compute (via dictionary lookup) and persistent
   memory (via selective weight updates). This GPU engine makes
   it trainable at scale on a single RTX 3060. *)

let next_id = ref 0
let fresh_id () = let id = !next_id in incr next_id; id

(* Training mode flag — dropout is active only when true. *)
let training = ref true
let dropout_rate = 0.1

type value = {
  id : int;
  data : Gpu.gpu_buf;
  grad : Gpu.gpu_buf;
  shape : int array;
  children : value array;
  backward_fn : unit -> unit;
}

(* Create a parameter tensor from a CPU float64 array.
   Uploads to GPU as float32 and allocates a zeroed gradient buffer. *)
let make_param shape data =
  let n = Array.length data in
  let d = Gpu.create n in
  Gpu.upload data d;
  let g = Gpu.create n in
  { id = fresh_id (); data = d; grad = g;
    shape; children = [||]; backward_fn = (fun () -> ()) }

(* Placeholder node for uninitialised slots (e.g. KV cache). *)
let dummy_node () =
  let d = Gpu.create 0 in
  { id = -1; data = d; grad = d; shape = [||];
    children = [||]; backward_fn = (fun () -> ()) }

(* ── Element-wise ops ─────────────────────────────────────────────── *)

(* add: y = a + b. Backward: da += dy, db += dy. *)
let add a b =
  let n = Gpu.numel a.data in
  let y_data = Gpu.create n in
  Gpu.add_forward a.data b.data y_data n;
  let y_grad = Gpu.create n in
  let backward () =
    (* da += dy — use cuBLAS saxpy via a simple add kernel *)
    Gpu.add_forward a.grad y_grad a.grad n;
    Gpu.add_forward b.grad y_grad b.grad n
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|a; b|]; backward_fn = backward }

(* scale: y = x * s. Backward: dx += dy * s. *)
let scale x s =
  let n = Gpu.numel x.data in
  let y_data = Gpu.create n in
  Gpu.scale_forward x.data s y_data n;
  let y_grad = Gpu.create n in
  let backward () =
    (* dx += dy * s — scale dy and accumulate into dx *)
    let tmp = Gpu.create n in
    Gpu.scale_forward y_grad s tmp n;
    Gpu.add_forward x.grad tmp x.grad n
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

(* gelu: y = gelu(x). Backward via gpu kernel. *)
let gelu x =
  let n = Gpu.numel x.data in
  let y_data = Gpu.create n in
  Gpu.gelu_forward x.data y_data n;
  let y_grad = Gpu.create n in
  let backward () =
    Gpu.gelu_backward x.data y_grad x.grad n
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

(* softmax: y = softmax(x). Single-row softmax of length n. *)
let softmax x =
  let n = Gpu.numel x.data in
  let y_data = Gpu.create n in
  Gpu.softmax_forward x.data y_data n;
  let y_grad = Gpu.create n in
  let backward () =
    Gpu.softmax_backward y_data y_grad x.grad n
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

(* nll: y = -log(probs[target]). Returns a scalar (1-element GPU buf).
   The loss value is fetched to CPU for the backward seed. *)
let nll probs target =
  let loss_val = Gpu.nll_forward probs.data target in
  let y_data = Gpu.create 1 in
  Gpu.upload [| loss_val |] y_data;
  let y_grad = Gpu.create 1 in
  let backward () =
    (* Download upstream gradient from GPU to seed NLL backward. *)
    let dy = Gpu.download y_grad in
    Gpu.nll_backward probs.data probs.grad target dy.(0)
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|1|]; children = [|probs|]; backward_fn = backward }

(* mean: y = mean(losses). losses is an array of scalar values. *)
let mean losses =
  let n = Array.length losses in
  let nf = float_of_int n in
  (* Sum all scalar loss values on CPU — tiny array. *)
  let total = Array.fold_left (fun acc l ->
    let v = Gpu.download l.data in
    acc +. v.(0)
  ) 0.0 losses in
  let y_data = Gpu.create 1 in
  Gpu.upload [| total /. nf |] y_data;
  let y_grad = Gpu.create 1 in
  let backward () =
    let dy = Gpu.download y_grad in
    let upstream = dy.(0) /. nf in
    Array.iter (fun l ->
      let cur = Gpu.download l.grad in
      Gpu.upload [| cur.(0) +. upstream |] l.grad
    ) losses
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|1|]; children = Array.copy losses; backward_fn = backward }

(* row: extract row i from a [rows, cols] matrix. *)
let row mat row_idx =
  let cols = mat.shape.(1) in
  let y_data = Gpu.create cols in
  Gpu.row_forward mat.data y_data row_idx cols;
  let y_grad = Gpu.create cols in
  let backward () =
    Gpu.row_backward mat.grad y_grad row_idx cols
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|cols|]; children = [|mat|]; backward_fn = backward }

(* ── Single-vector ops (inference) ────────────────────────────────── *)

(* matmul: y[m] = w[m,n] @ x[n]. Uses cuBLAS sgemm. *)
let matmul w x m n =
  let y_data = Gpu.create m in
  (* C[m,1] = A[m,n] @ B[n,1] — op=0: NN overwrite *)
  Gpu.sgemm 0 m 1 n w.data x.data y_data;
  let y_grad = Gpu.create m in
  let backward () =
    (* dW += dy @ x^T — op=1: NN accumulate, treating as outer product *)
    Gpu.sgemm 1 m n 1 y_grad x.data w.grad;
    (* dx += W^T @ dy — op=5: TN accumulate *)
    Gpu.sgemm 5 n 1 m w.data y_grad x.grad
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|m|]; children = [|w; x|]; backward_fn = backward }

(* rmsnorm: normalise a vector by its root-mean-square.
   Stores rms value on GPU for backward. *)
let rmsnorm x dim =
  let y_data = Gpu.create dim in
  let rms_buf = Gpu.create 1 in  (* single rms value *)
  Gpu.rmsnorm_forward x.data y_data rms_buf 1 dim;
  let y_grad = Gpu.create dim in
  let backward () =
    Gpu.rmsnorm_backward x.data y_grad rms_buf x.grad 1 dim
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|dim|]; children = [|x|]; backward_fn = backward }

(* rmsnorm_affine: normalise then scale by learnable gamma. *)
let rmsnorm_affine x gamma dim =
  let y_data = Gpu.create dim in
  let rms_buf = Gpu.create 1 in
  let norm_data = Gpu.create dim in  (* normalised values for backward *)
  Gpu.rmsnorm_affine_forward x.data gamma.data y_data rms_buf 1 dim;
  let y_grad = Gpu.create dim in
  let backward () =
    Gpu.rmsnorm_affine_backward x.data gamma.data y_grad
      norm_data rms_buf x.grad gamma.grad 1 dim
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|dim|]; children = [|x; gamma|]; backward_fn = backward }

(* ── Batched ops (training) ───────────────────────────────────────── *)

(* batch_matmul: y[s,m] = x[s,n] @ w[m,n]^T.
   cuBLAS: y[s,m] = x[s,n] @ w^T[n,m] — op=2 (NT overwrite). *)
let batch_matmul w x s m n =
  let y_data = Gpu.create (s * m) in
  Gpu.sgemm 2 s m n x.data w.data y_data;
  let y_grad = Gpu.create (s * m) in
  let backward () =
    (* dW += dy^T @ x — op=5: TN accumulate *)
    Gpu.sgemm 5 m n s y_grad x.data w.grad;
    (* dx += dy @ W — op=1: NN accumulate *)
    Gpu.sgemm 1 s n m y_grad w.data x.grad
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|s; m|]; children = [|w; x|]; backward_fn = backward }

(* batch_rmsnorm: normalise each row of [s, dim] independently. *)
let batch_rmsnorm x s dim =
  let y_data = Gpu.create (s * dim) in
  let rms_buf = Gpu.create s in  (* per-row rms values *)
  Gpu.rmsnorm_forward x.data y_data rms_buf s dim;
  let y_grad = Gpu.create (s * dim) in
  let backward () =
    Gpu.rmsnorm_backward x.data y_grad rms_buf x.grad s dim
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|s; dim|]; children = [|x|]; backward_fn = backward }

(* batch_rmsnorm_affine: per-row normalise then scale by shared gamma. *)
let batch_rmsnorm_affine x gamma s dim =
  let y_data = Gpu.create (s * dim) in
  let rms_buf = Gpu.create s in
  let norm_data = Gpu.create (s * dim) in
  Gpu.rmsnorm_affine_forward x.data gamma.data y_data rms_buf s dim;
  let y_grad = Gpu.create (s * dim) in
  let backward () =
    Gpu.rmsnorm_affine_backward x.data gamma.data y_grad
      norm_data rms_buf x.grad gamma.grad s dim
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|s; dim|]; children = [|x; gamma|]; backward_fn = backward }

(* batch_dropout: randomly zero elements during training. *)
let batch_dropout x s dim =
  if not !training || dropout_rate = 0.0 then x
  else begin
    let n = s * dim in
    let y_data = Gpu.create n in
    let mask = Gpu.create n in
    Gpu.dropout_forward x.data y_data mask dropout_rate n;
    let y_grad = Gpu.create n in
    let backward () =
      Gpu.dropout_backward y_grad mask x.grad n
    in
    { id = fresh_id (); data = y_data; grad = y_grad;
      shape = [|s; dim|]; children = [|x|]; backward_fn = backward }
  end

(* batch_embed: look up token embeddings from the embedding table.
   wte is [vocab, n_embd], tokens is a CPU int array. *)
let batch_embed wte tokens n_embd =
  let s = Array.length tokens in
  let y_data = Gpu.create (s * n_embd) in
  Gpu.embed_forward wte.data tokens y_data s n_embd;
  let y_grad = Gpu.create (s * n_embd) in
  let backward () =
    Gpu.embed_backward wte.grad tokens y_grad s n_embd
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|s; n_embd|]; children = [|wte|]; backward_fn = backward }

(* ── Backward pass ────────────────────────────────────────────────── *)

(* Topological sort via DFS, then reverse-mode gradient propagation.
   The graph structure lives on the CPU; backward_fn closures dispatch
   to GPU kernels. This is the beauty of the design: OCaml orchestrates,
   CUDA computes. *)
let backward loss =
  let visited = Hashtbl.create 1024 in
  let topo = ref [] in
  let rec build_topo v =
    if not (Hashtbl.mem visited v.id) then begin
      Hashtbl.add visited v.id ();
      Array.iter build_topo v.children;
      topo := v :: !topo
    end
  in
  build_topo loss;
  (* Seed the loss gradient with 1.0. *)
  Gpu.upload [| 1.0 |] loss.grad;
  List.iter (fun v -> v.backward_fn ()) !topo

(* ── CPU-side utilities ───────────────────────────────────────────── *)

(* Download a gpu_buf to a CPU float array. Used for logging,
   sampling, and checkpoint I/O. *)
let to_cpu buf = Gpu.download buf

(* Softmax on CPU for tiny arrays (sampling logits ~2K elements).
   Avoids GPU round-trip for generation. *)
let softmax_cpu arr =
  let n = Array.length arr in
  let max_val = ref neg_infinity in
  for i = 0 to n - 1 do
    if arr.(i) > !max_val then max_val := arr.(i)
  done;
  let result = Array.create_float n in
  let sum_exp = ref 0.0 in
  for i = 0 to n - 1 do
    let e = exp (arr.(i) -. !max_val) in
    result.(i) <- e;
    sum_exp := !sum_exp +. e
  done;
  let inv = 1.0 /. !sum_exp in
  for i = 0 to n - 1 do result.(i) <- result.(i) *. inv done;
  result
