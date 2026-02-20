(* tensor.ml — Autograd engine
   ============================

   A reverse-mode automatic differentiation engine over flat float arrays.
   Each value node stores data, gradients, shape, children, and a backward
   function that propagates gradients to its children.

   Three categories of ops:
   1. Element-wise: add, scale, gelu, softmax, nll, mean, row
   2. Single-vector: matmul, rmsnorm (inference — one token at a time)
   3. Batched: batch_matmul, batch_rmsnorm, batch_embed (training — all positions) *)

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

let dummy_node () =
  { id = -1; data = [||]; grad = [||]; shape = [||];
    children = [||]; backward_fn = (fun () -> ()) }

(* ── Element-wise ops ─────────────────────────────────────────────── *)

let add a b =
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

let scale x s =
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

let gelu x =
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

let softmax x =
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

let nll probs target =
  let y_data = [| -. log probs.data.(target) |] in
  let y_grad = [| 0.0 |] in
  let backward () =
    probs.grad.(target) <- probs.grad.(target)
      -. y_grad.(0) /. probs.data.(target) in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|1|]; children = [|probs|]; backward_fn = backward }

let mean losses =
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

let row mat row_idx =
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

(* ── Single-vector ops (inference) ────────────────────────────────── *)

(* matmul: w[m,n] @ x[n] → y[m].  Uses BLAS for the heavy lifting. *)
let matmul w x m n =
  let y_data = Array.create_float m in
  Blas.dgemm 0 m 1 n w.data x.data y_data;
  let y_grad = Array.make m 0.0 in
  let backward () =
    Blas.dgemm 1 m n 1 y_grad x.data w.grad;
    Blas.dgemm 5 n 1 m w.data y_grad x.grad
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|m|]; children = [|w; x|]; backward_fn = backward }

(* rmsnorm: normalize a vector by its root-mean-square. *)
let rmsnorm x dim =
  let nf = float_of_int dim in
  let ms = ref 0.0 in
  for i = 0 to dim - 1 do ms := !ms +. x.data.(i) *. x.data.(i) done;
  let rms = sqrt (!ms /. nf +. 1e-5) in
  let y_data = Array.create_float dim in
  for i = 0 to dim - 1 do y_data.(i) <- x.data.(i) /. rms done;
  let y_grad = Array.make dim 0.0 in
  let backward () =
    let dot_gy = ref 0.0 in
    for i = 0 to dim - 1 do dot_gy := !dot_gy +. y_grad.(i) *. y_data.(i) done;
    let mean_gy = !dot_gy /. nf in
    for i = 0 to dim - 1 do
      x.grad.(i) <- x.grad.(i) +. (y_grad.(i) -. y_data.(i) *. mean_gy) /. rms
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|dim|]; children = [|x|]; backward_fn = backward }

(* ── Batched ops (training) ───────────────────────────────────────── *)

(* batch_matmul: w[m,n] @ x[s,n]^T → y[s,m].
   BLAS: y[s,m] = x[s,n] @ w[m,n]^T — op=2 (NT overwrite). *)
let batch_matmul w x s m n =
  let y_data = Array.create_float (s * m) in
  Blas.dgemm 2 s m n x.data w.data y_data;
  let y_grad = Array.make (s * m) 0.0 in
  let backward () =
    Blas.dgemm 5 m n s y_grad x.data w.grad;
    Blas.dgemm 1 s n m y_grad w.data x.grad
  in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|s; m|]; children = [|w; x|]; backward_fn = backward }

(* batch_rmsnorm: normalize each row of [s, dim] independently. *)
let batch_rmsnorm x s dim =
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

(* batch_embed: look up token embeddings for a sequence of token IDs. *)
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

(* ── Backward pass ────────────────────────────────────────────────── *)

(* Topological sort via DFS, then reverse-mode gradient propagation. *)
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
  loss.grad.(0) <- 1.0;
  List.iter (fun v -> v.backward_fn ()) !topo
