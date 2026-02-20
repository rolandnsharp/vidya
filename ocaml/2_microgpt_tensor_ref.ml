(* 2_microgpt_tensor_ref.ml — Stage 2: Tensor Autograd
   Same GPT architecture as Stage 1, but each computation graph node
   holds a tensor (float array + shape) instead of a single scalar.
   This collapses the graph from ~20,000 nodes to ~200 nodes per step. *)

let () = Random.init 42

(* ── Utilities ─────────────────────────────────────────────────────── *)

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

let tokenize uchars bos_id doc =
  let find_char ch =
    Array.to_list (Array.mapi (fun i c -> (i, c)) uchars)
    |> List.find (fun (_, c) -> c = ch)
    |> fst
  in
  let inner = String.to_seq doc |> Seq.map find_char |> Array.of_seq in
  Array.concat [ [| bos_id |]; inner; [| bos_id |] ]

(* ── Tensor Autograd Engine ────────────────────────────────────────── *)

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

(* ── Tensor Operations ─────────────────────────────────────────────── *)

(* Matrix-vector multiply: [m,n] @ [n] -> [m] *)
let matmul w x =
  let m = w.shape.(0) and n = w.shape.(1) in
  let y_data = Array.init m (fun i ->
    let s = ref 0.0 in
    for j = 0 to n - 1 do s := !s +. w.data.(i * n + j) *. x.data.(j) done;
    !s) in
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

(* Transposed matrix-vector: [m,n]^T @ [m] -> [n] *)
let matmul_tv mat v =
  let rows = mat.shape.(0) and cols = mat.shape.(1) in
  let y_data = Array.init cols (fun j ->
    let s = ref 0.0 in
    for i = 0 to rows - 1 do s := !s +. mat.data.(i * cols + j) *. v.data.(i) done;
    !s) in
  let y_grad = Array.make cols 0.0 in
  let backward () =
    for i = 0 to rows - 1 do
      for j = 0 to cols - 1 do
        mat.grad.(i * cols + j) <- mat.grad.(i * cols + j) +. y_grad.(j) *. v.data.(i);
        v.grad.(i) <- v.grad.(i) +. mat.data.(i * cols + j) *. y_grad.(j)
      done
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|cols|]; children = [|mat; v|]; backward_fn = backward }

(* Element-wise add: [n] + [n] -> [n] *)
let tensor_add a b =
  let n = Array.length a.data in
  let y_data = Array.init n (fun i -> a.data.(i) +. b.data.(i)) in
  let y_grad = Array.make n 0.0 in
  let backward () =
    for i = 0 to n - 1 do
      a.grad.(i) <- a.grad.(i) +. y_grad.(i);
      b.grad.(i) <- b.grad.(i) +. y_grad.(i)
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|a; b|]; backward_fn = backward }

(* Scale by constant: [n] * float -> [n] *)
let tensor_scale x s =
  let n = Array.length x.data in
  let y_data = Array.init n (fun i -> x.data.(i) *. s) in
  let y_grad = Array.make n 0.0 in
  let backward () =
    for i = 0 to n - 1 do
      x.grad.(i) <- x.grad.(i) +. y_grad.(i) *. s
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

(* Element-wise ReLU: [n] -> [n] *)
let tensor_relu x =
  let n = Array.length x.data in
  let y_data = Array.init n (fun i -> Float.max 0.0 x.data.(i)) in
  let y_grad = Array.make n 0.0 in
  let backward () =
    for i = 0 to n - 1 do
      if x.data.(i) > 0.0 then
        x.grad.(i) <- x.grad.(i) +. y_grad.(i)
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

(* Softmax: [n] -> [n]
   Backward: dx_i = y_i * (dy_i - sum_j(dy_j * y_j)) *)
let tensor_softmax x =
  let n = Array.length x.data in
  let max_val = Array.fold_left Float.max neg_infinity x.data in
  let y_data = Array.init n (fun i -> exp (x.data.(i) -. max_val)) in
  let sum_exp = Array.fold_left (+.) 0.0 y_data in
  Array.iteri (fun i _ -> y_data.(i) <- y_data.(i) /. sum_exp) y_data;
  let y_grad = Array.make n 0.0 in
  let backward () =
    let dot = ref 0.0 in
    for j = 0 to n - 1 do dot := !dot +. y_grad.(j) *. y_data.(j) done;
    for i = 0 to n - 1 do
      x.grad.(i) <- x.grad.(i) +. y_data.(i) *. (y_grad.(i) -. !dot)
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|n|]; children = [|x|]; backward_fn = backward }

(* RMS normalization: [n] -> [n]
   Backward: dx_j = (dy_j - y_j * mean(dy * y)) / rms *)
let tensor_rmsnorm x =
  let n = Array.length x.data in
  let nf = float_of_int n in
  let ms = ref 0.0 in
  for i = 0 to n - 1 do ms := !ms +. x.data.(i) *. x.data.(i) done;
  let rms = sqrt (!ms /. nf +. 1e-5) in
  let y_data = Array.init n (fun i -> x.data.(i) /. rms) in
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

(* Extract row from 2D tensor: [m,n], int -> [n] *)
let tensor_row mat row_idx =
  let cols = mat.shape.(1) in
  let off = row_idx * cols in
  let y_data = Array.init cols (fun j -> mat.data.(off + j)) in
  let y_grad = Array.make cols 0.0 in
  let backward () =
    for j = 0 to cols - 1 do
      mat.grad.(off + j) <- mat.grad.(off + j) +. y_grad.(j)
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|cols|]; children = [|mat|]; backward_fn = backward }

(* Slice 1D tensor: [n], offset, len -> [len] *)
let tensor_slice x off len =
  let y_data = Array.init len (fun i -> x.data.(off + i)) in
  let y_grad = Array.make len 0.0 in
  let backward () =
    for i = 0 to len - 1 do
      x.grad.(off + i) <- x.grad.(off + i) +. y_grad.(i)
    done in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|len|]; children = [|x|]; backward_fn = backward }

(* Stack list of [d] tensors into [T, d] matrix *)
let tensor_stack tensors =
  let t_count = List.length tensors in
  let d = (List.hd tensors).shape.(0) in
  let y_data = Array.make (t_count * d) 0.0 in
  List.iteri (fun t v -> Array.blit v.data 0 y_data (t * d) d) tensors;
  let y_grad = Array.make (t_count * d) 0.0 in
  let backward () =
    List.iteri (fun t v ->
      for j = 0 to d - 1 do
        v.grad.(j) <- v.grad.(j) +. y_grad.(t * d + j)
      done
    ) tensors in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|t_count; d|]; children = Array.of_list tensors;
    backward_fn = backward }

(* Concatenate 1D tensors: [k_1] @ [k_2] @ ... -> [sum k_i] *)
let tensor_concat tensors =
  let total = List.fold_left (fun acc v -> acc + Array.length v.data) 0 tensors in
  let y_data = Array.make total 0.0 in
  let offsets = Array.make (List.length tensors) 0 in
  let pos = ref 0 in
  List.iteri (fun idx v ->
    offsets.(idx) <- !pos;
    Array.blit v.data 0 y_data !pos (Array.length v.data);
    pos := !pos + Array.length v.data
  ) tensors;
  let y_grad = Array.make total 0.0 in
  let backward () =
    List.iteri (fun idx v ->
      let off = offsets.(idx) in
      let n = Array.length v.data in
      for i = 0 to n - 1 do
        v.grad.(i) <- v.grad.(i) +. y_grad.(off + i)
      done
    ) tensors in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|total|]; children = Array.of_list tensors;
    backward_fn = backward }

(* Negative log-likelihood: -log(probs[target]) -> [1] *)
let tensor_nll probs target =
  let y_data = [| -. log probs.data.(target) |] in
  let y_grad = [| 0.0 |] in
  let backward () =
    probs.grad.(target) <- probs.grad.(target)
      -. y_grad.(0) /. probs.data.(target) in
  { id = fresh_id (); data = y_data; grad = y_grad;
    shape = [|1|]; children = [|probs|]; backward_fn = backward }

(* Mean of scalar [1] losses: mean -> [1] *)
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

(* ── Backward Pass ─────────────────────────────────────────────────── *)

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
  List.rev !topo

let backward loss =
  let topo = topological_sort loss in
  loss.grad.(0) <- 1.0;
  topo |> List.rev |> List.iter (fun v -> v.backward_fn ())

(* ── Model ─────────────────────────────────────────────────────────── *)

let n_layer = 1
let n_embd = 16
let block_size = 16
let n_head = 4
let head_dim = n_embd / n_head

type layer_weights = {
  attn_wq : value; attn_wk : value;
  attn_wv : value; attn_wo : value;
  mlp_fc1 : value; mlp_fc2 : value;
}

type model = {
  wte : value; wpe : value;
  lm_head : value; layers : layer_weights array;
}

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

let collect_params model =
  let layer_params l =
    [l.attn_wq; l.attn_wk; l.attn_wv; l.attn_wo; l.mlp_fc1; l.mlp_fc2] in
  [model.wte; model.wpe; model.lm_head]
  @ (model.layers |> Array.to_list |> List.map layer_params |> List.flatten)
  |> Array.of_list

(* ── Forward Pass ──────────────────────────────────────────────────── *)

let embed_token model token_id pos_id =
  let tok = tensor_row model.wte token_id in
  let pos = tensor_row model.wpe pos_id in
  tensor_add tok pos |> tensor_rmsnorm

let attention_head q_h keys_h values_h =
  let scale = 1.0 /. sqrt (float_of_int head_dim) in
  let k_matrix = tensor_stack keys_h in
  let scores = matmul k_matrix q_h |> fun s -> tensor_scale s scale in
  let weights = tensor_softmax scores in
  let v_matrix = tensor_stack values_h in
  matmul_tv v_matrix weights

let multi_head_attention x layer keys values =
  let q = matmul layer.attn_wq x in
  let k = matmul layer.attn_wk x in
  let v = matmul layer.attn_wv x in
  keys := k :: !keys;
  values := v :: !values;
  let head_outputs = Array.init n_head (fun h ->
    let off = h * head_dim in
    let q_h = tensor_slice q off head_dim in
    let keys_h = !keys |> List.rev
      |> List.map (fun ki -> tensor_slice ki off head_dim) in
    let values_h = !values |> List.rev
      |> List.map (fun vi -> tensor_slice vi off head_dim) in
    attention_head q_h keys_h values_h
  ) in
  let x_attn = tensor_concat (Array.to_list head_outputs) in
  matmul layer.attn_wo x_attn

let mlp_block x layer =
  matmul layer.mlp_fc1 x |> tensor_relu |> fun h -> matmul layer.mlp_fc2 h

let transformer_block x layer keys values =
  let x =
    tensor_rmsnorm x |> fun xn ->
    multi_head_attention xn layer keys values |> tensor_add x in
  tensor_rmsnorm x |> fun xn -> mlp_block xn layer |> tensor_add x

let gpt_forward model token_id pos_id layer_keys layer_values =
  let x = embed_token model token_id pos_id in
  let x = Array.to_list model.layers
    |> List.mapi (fun li layer -> (li, layer))
    |> List.fold_left (fun x (li, layer) ->
      transformer_block x layer layer_keys.(li) layer_values.(li)) x in
  matmul model.lm_head x

(* ── Training ──────────────────────────────────────────────────────── *)

type adam_state = { m : float array; v : float array }
let learning_rate = 0.01
let beta1 = 0.85
let beta2 = 0.99
let eps_adam = 1e-8

let init_adam params =
  let total = Array.fold_left (fun acc p -> acc + Array.length p.data) 0 params in
  { m = Array.make total 0.0; v = Array.make total 0.0 }

let adam_step params adam step num_steps =
  let lr_t = learning_rate *. (1.0 -. float_of_int step /. float_of_int num_steps) in
  let offset = ref 0 in
  params |> Array.iter (fun p ->
    let n = Array.length p.data in
    for i = 0 to n - 1 do
      let fi = !offset + i in
      adam.m.(fi) <- beta1 *. adam.m.(fi) +. (1.0 -. beta1) *. p.grad.(i);
      adam.v.(fi) <- beta2 *. adam.v.(fi) +. (1.0 -. beta2) *. p.grad.(i) *. p.grad.(i);
      let m_hat = adam.m.(fi) /. (1.0 -. beta1 ** float_of_int (step + 1)) in
      let v_hat = adam.v.(fi) /. (1.0 -. beta2 ** float_of_int (step + 1)) in
      p.data.(i) <- p.data.(i) -. lr_t *. m_hat /. (sqrt v_hat +. eps_adam);
      p.grad.(i) <- 0.0
    done;
    offset := !offset + n)

let compute_loss model tokens layer_keys layer_values =
  let n = min block_size (Array.length tokens - 1) in
  let losses = Array.init n (fun pos_id ->
    let logits = gpt_forward model tokens.(pos_id) pos_id layer_keys layer_values in
    let probs = tensor_softmax logits in
    tensor_nll probs tokens.(pos_id + 1)) in
  (tensor_mean losses, n)

let train model params docs uchars bos_id num_steps =
  let adam = init_adam params in
  for step = 0 to num_steps - 1 do
    let doc = docs.(step mod Array.length docs) in
    let tokens = tokenize uchars bos_id doc in
    let layer_keys = Array.init n_layer (fun _ -> ref []) in
    let layer_values = Array.init n_layer (fun _ -> ref []) in
    let (loss, _) = compute_loss model tokens layer_keys layer_values in
    backward loss;
    adam_step params adam step num_steps;
    Printf.printf "\rstep %4d / %4d | loss %.4f%!" (step + 1) num_steps loss.data.(0)
  done;
  Printf.printf "\n"

(* ── Inference ─────────────────────────────────────────────────────── *)

let generate_sample model uchars bos_id _vocab_size temperature =
  let layer_keys = Array.init n_layer (fun _ -> ref []) in
  let layer_values = Array.init n_layer (fun _ -> ref []) in
  let token_id = ref bos_id in
  let buf = Buffer.create 32 in
  let pos = ref 0 in
  while !pos < block_size do
    let logits = gpt_forward model !token_id !pos layer_keys layer_values in
    let probs = tensor_scale logits (1.0 /. temperature) |> tensor_softmax in
    token_id := weighted_choice probs.data;
    if !token_id = bos_id then pos := block_size
    else begin Buffer.add_char buf uchars.(!token_id); incr pos end
  done;
  Buffer.contents buf

(* ── Main ──────────────────────────────────────────────────────────── *)

let () =
  let t_start = Sys.time () in
  let docs = load_docs "input.txt" in
  Printf.printf "num docs: %d\n" (Array.length docs);
  let (uchars, bos_id, vocab_size) = build_vocab docs in
  Printf.printf "vocab size: %d\n" vocab_size;
  let model = init_model vocab_size in
  let params = collect_params model in
  let total_params =
    Array.fold_left (fun acc p -> acc + Array.length p.data) 0 params in
  Printf.printf "num params: %d\n" total_params;
  train model params docs uchars bos_id 1000;
  Printf.printf "--- inference (new, hallucinated text) ---\n";
  for i = 1 to 20 do
    generate_sample model uchars bos_id vocab_size 0.5
    |> Printf.printf "sample %2d: %s\n" i
  done;
  Printf.printf "total time: %.2fs\n" (Sys.time () -. t_start)
