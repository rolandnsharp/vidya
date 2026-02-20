let () = Random.init 42

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

let next_id = ref 0
let fresh_id () = let id = !next_id in incr next_id; id

type value = {
  id : int;
  data : float ref;
  grad : float ref;
  children : value array;
  local_grads : float array;
}

let make_value data =
  { id = fresh_id (); data = ref data; grad = ref 0.0;
    children = [||]; local_grads = [||] }

let make_op data children local_grads =
  { id = fresh_id (); data = ref data; grad = ref 0.0;
    children; local_grads }

let vadd a b = make_op (!(a.data) +. !(b.data)) [|a; b|] [|1.0; 1.0|]
let vmul a b = make_op (!(a.data) *. !(b.data)) [|a; b|] [|!(b.data); !(a.data)|]
let vpow a n = make_op (!(a.data) ** n) [|a|] [|n *. !(a.data) ** (n -. 1.0)|]
let vlog a = make_op (log !(a.data)) [|a|] [|1.0 /. !(a.data)|]
let vexp a = let e = exp !(a.data) in make_op e [|a|] [|e|]
let vrelu a = make_op (Float.max 0.0 !(a.data)) [|a|] [|if !(a.data) > 0.0 then 1.0 else 0.0|]
let vneg a = vmul a (make_value (-1.0))
let vsub a b = vadd a (vneg b)
let vdiv a b = vmul a (vpow b (-1.0))

let vsum arr =
  if Array.length arr = 0 then make_value 0.0
  else Array.sub arr 1 (Array.length arr - 1) |> Array.fold_left vadd arr.(0)

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
  loss.grad := 1.0;
  topo |> List.rev |> List.iter (fun v ->
    v.children |> Array.iteri (fun i child ->
      child.grad := !(child.grad) +. v.local_grads.(i) *. !(v.grad)
    )
  )

let n_layer = 1
let n_embd = 16
let block_size = 16
let n_head = 4
let head_dim = n_embd / n_head

type layer_weights = {
  attn_wq : value array array; attn_wk : value array array;
  attn_wv : value array array; attn_wo : value array array;
  mlp_fc1 : value array array; mlp_fc2 : value array array;
}

type model = {
  wte : value array array; wpe : value array array;
  lm_head : value array array; layers : layer_weights array;
}

let init_matrix ?(std = 0.08) nout nin =
  Array.init nout (fun _ ->
    Array.init nin (fun _ -> random_gauss ~std () |> make_value))

let init_model vocab_size =
  let make_layer () = {
    attn_wq = init_matrix n_embd n_embd; attn_wk = init_matrix n_embd n_embd;
    attn_wv = init_matrix n_embd n_embd; attn_wo = init_matrix n_embd n_embd;
    mlp_fc1 = init_matrix (4 * n_embd) n_embd; mlp_fc2 = init_matrix n_embd (4 * n_embd);
  } in
  { wte = init_matrix vocab_size n_embd; wpe = init_matrix block_size n_embd;
    lm_head = init_matrix vocab_size n_embd;
    layers = Array.init n_layer (fun _ -> make_layer ()) }

let collect_params model =
  let mat m = m |> Array.to_list |> List.map Array.to_list |> List.flatten in
  let layer l = [l.attn_wq; l.attn_wk; l.attn_wv; l.attn_wo; l.mlp_fc1; l.mlp_fc2]
    |> List.map mat |> List.flatten in
  [ mat model.wte; mat model.wpe; mat model.lm_head;
    model.layers |> Array.to_list |> List.map layer |> List.flatten ]
  |> List.flatten |> Array.of_list

let linear x w =
  w |> Array.map (fun w_row -> w_row |> Array.mapi (fun i wi -> vmul wi x.(i)) |> vsum)

let softmax logits =
  let max_val = logits |> Array.map (fun v -> !(v.data)) |> Array.fold_left Float.max neg_infinity in
  let exps = logits |> Array.map (fun v -> vsub v (make_value max_val) |> vexp) in
  let total = vsum exps in
  exps |> Array.map (fun e -> vdiv e total)

let rmsnorm x =
  let n = float_of_int (Array.length x) in
  let mean_sq = x |> Array.map (fun xi -> vmul xi xi) |> vsum |> fun ms -> vdiv ms (make_value n) in
  let scale = vadd mean_sq (make_value 1e-5) |> fun s -> vpow s (-0.5) in
  x |> Array.map (fun xi -> vmul xi scale)

let vector_add a b = Array.init (Array.length a) (fun i -> vadd a.(i) b.(i))

let attention_head q_h k_h v_h =
  let scale = float_of_int head_dim |> sqrt |> make_value in
  let attn_logits =
    k_h |> List.map (fun kt ->
      Array.init head_dim (fun j -> vmul q_h.(j) kt.(j)) |> vsum |> fun dot -> vdiv dot scale
    ) |> Array.of_list in
  let attn_weights = softmax attn_logits in
  let v_arr = Array.of_list v_h in
  Array.init head_dim (fun j ->
    Array.init (Array.length v_arr) (fun t -> vmul attn_weights.(t) v_arr.(t).(j)) |> vsum)

let multi_head_attention x layer keys values =
  let q = linear x layer.attn_wq in
  let k = linear x layer.attn_wk in
  let v = linear x layer.attn_wv in
  keys := k :: !keys;
  values := v :: !values;
  let x_attn = Array.init n_embd (fun idx ->
    let h = idx / head_dim in
    let hs = h * head_dim in
    let q_h = Array.sub q hs head_dim in
    let k_h = !keys |> List.rev |> List.map (fun ki -> Array.sub ki hs head_dim) in
    let v_h = !values |> List.rev |> List.map (fun vi -> Array.sub vi hs head_dim) in
    (attention_head q_h k_h v_h).(idx mod head_dim)
  ) in
  linear x_attn layer.attn_wo

let mlp_block x layer =
  x |> fun x -> linear x layer.mlp_fc1 |> Array.map vrelu |> fun h -> linear h layer.mlp_fc2

let transformer_block x layer keys values =
  let x =
    rmsnorm x |> fun xn -> multi_head_attention xn layer keys values |> vector_add x in
  rmsnorm x |> fun xn -> mlp_block xn layer |> vector_add x

let embed_token model token_id pos_id =
  vector_add model.wte.(token_id) model.wpe.(pos_id) |> rmsnorm

let gpt_forward model token_id pos_id layer_keys layer_values =
  let x = embed_token model token_id pos_id in
  let x = Array.to_list model.layers
    |> List.mapi (fun li layer -> (li, layer))
    |> List.fold_left (fun x (li, layer) ->
      transformer_block x layer layer_keys.(li) layer_values.(li)) x in
  linear x model.lm_head

type adam_state = { m : float array; v : float array }
let init_adam n = { m = Array.make n 0.0; v = Array.make n 0.0 }
let learning_rate = 0.01
let beta1 = 0.85
let beta2 = 0.99
let eps_adam = 1e-8

let adam_step params adam step num_steps =
  let lr_t = learning_rate *. (1.0 -. float_of_int step /. float_of_int num_steps) in
  params |> Array.iteri (fun i p ->
    adam.m.(i) <- beta1 *. adam.m.(i) +. (1.0 -. beta1) *. !(p.grad);
    adam.v.(i) <- beta2 *. adam.v.(i) +. (1.0 -. beta2) *. !(p.grad) *. !(p.grad);
    let m_hat = adam.m.(i) /. (1.0 -. beta1 ** float_of_int (step + 1)) in
    let v_hat = adam.v.(i) /. (1.0 -. beta2 ** float_of_int (step + 1)) in
    p.data := !(p.data) -. lr_t *. m_hat /. (sqrt v_hat +. eps_adam);
    p.grad := 0.0)

let compute_loss model tokens layer_keys layer_values =
  let n = min block_size (Array.length tokens - 1) in
  let losses = Array.init n (fun pos_id ->
    let logits = gpt_forward model tokens.(pos_id) pos_id layer_keys layer_values in
    let probs = softmax logits in
    probs.(tokens.(pos_id + 1)) |> vlog |> vneg) in
  (vdiv (vsum losses) (make_value (float_of_int n)), n)

let train model params docs uchars bos_id num_steps =
  let adam = init_adam (Array.length params) in
  for step = 0 to num_steps - 1 do
    let doc = docs.(step mod Array.length docs) in
    let tokens = tokenize uchars bos_id doc in
    let layer_keys = Array.init n_layer (fun _ -> ref []) in
    let layer_values = Array.init n_layer (fun _ -> ref []) in
    let (loss, _) = compute_loss model tokens layer_keys layer_values in
    backward loss;
    adam_step params adam step num_steps;
    Printf.printf "\rstep %4d / %4d | loss %.4f%!" (step + 1) num_steps !(loss.data)
  done;
  Printf.printf "\n"

let generate_sample model uchars bos_id vocab_size temperature =
  let layer_keys = Array.init n_layer (fun _ -> ref []) in
  let layer_values = Array.init n_layer (fun _ -> ref []) in
  let token_id = ref bos_id in
  let buf = Buffer.create 32 in
  let pos = ref 0 in
  while !pos < block_size do
    let logits = gpt_forward model !token_id !pos layer_keys layer_values in
    let probs = logits |> Array.map (fun l -> vdiv l (make_value temperature)) |> softmax in
    let weights = probs |> Array.map (fun p -> !(p.data)) in
    token_id := weighted_choice weights;
    if !token_id = bos_id then pos := block_size
    else begin Buffer.add_char buf uchars.(!token_id); incr pos end
  done;
  Buffer.contents buf

let () =
  let docs = load_docs "input.txt" in
  Printf.printf "num docs: %d\n" (Array.length docs);
  let (uchars, bos_id, vocab_size) = build_vocab docs in
  Printf.printf "vocab size: %d\n" vocab_size;
  let model = init_model vocab_size in
  let params = collect_params model in
  Printf.printf "num params: %d\n" (Array.length params);
  train model params docs uchars bos_id 1000;
  Printf.printf "--- inference (new, hallucinated text) ---\n";
  for i = 1 to 20 do
    generate_sample model uchars bos_id vocab_size 0.5
    |> Printf.printf "sample %2d: %s\n" i
  done
