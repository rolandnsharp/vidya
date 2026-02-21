(* generate.ml — Text generation with symbolic constraints
   =========================================================

   Three generation modes:
   1. sample: unprompted generation from BOS token
   2. prompted: encode a prompt, prefill KV cache, then sample
   3. chat: format user input with turn markers, generate response

   All take a Symbolic.knowledge (built once, shared across runs)
   and an optional Knowledge.t (concept knowledge).
   A fresh Symbolic.context is created per generation call.
   Constraints are applied to logits before sampling. *)

(* sample: Generate text from scratch (starting from BOS).
   Feeds BOS, then autoregressively samples until BOS is produced
   again or block_size is reached. *)
let sample model know ?concept_know bos_id temperature =
  Tensor.training := false;
  let sym_ctx = Symbolic.create ?concept_know know in
  let kv_caches = Array.init Model.n_layer (fun _ -> Model.make_kv_cache ()) in
  let token_id = ref bos_id in
  let buf = Buffer.create 256 in
  let pos = ref 0 in
  while !pos < Model.block_size do
    let logits = Forward.gpt_forward model !token_id kv_caches in
    let constrained = Symbolic.apply sym_ctx logits.Tensor.data in
    let scaled = Array.map (fun x -> x /. temperature) constrained in
    let probs = Tensor.softmax (Tensor.make_param [|Array.length scaled|] scaled) in
    token_id := Utils.weighted_choice probs.Tensor.data;
    if !token_id = bos_id then pos := Model.block_size
    else begin
      Symbolic.td_update sym_ctx logits.Tensor.data !token_id;
      Symbolic.record_token sym_ctx !token_id;
      Buffer.add_string buf know.Symbolic.vocab.(!token_id);
      incr pos
    end
  done;
  Buffer.contents buf

(* prompted: Encode a prompt via BPE, prefill the KV cache with
   prompt tokens (no sampling during prefill), then sample
   continuation tokens autoregressively with symbolic constraints. *)
let prompted model know ?concept_know tok bos_id prompt temperature =
  Tensor.training := false;
  let sym_ctx = Symbolic.create ?concept_know know in
  let kv_caches = Array.init Model.n_layer (fun _ -> Model.make_kv_cache ()) in
  let prompt_ids = Bpe.encode tok prompt in
  let vocab = know.Symbolic.vocab in
  let buf = Buffer.create 256 in
  let n_prompt = Array.length prompt_ids in
  for i = 0 to n_prompt - 2 do
    ignore (Forward.gpt_forward model prompt_ids.(i) kv_caches);
    if i > 0 then begin
      Symbolic.record_token sym_ctx prompt_ids.(i);
      Buffer.add_string buf vocab.(prompt_ids.(i))
    end
  done;
  let token_id = ref prompt_ids.(n_prompt - 1) in
  if !token_id <> bos_id then begin
    Symbolic.record_token sym_ctx !token_id;
    Buffer.add_string buf vocab.(!token_id)
  end;
  let pos = ref n_prompt in
  while !pos < Model.block_size do
    let logits = Forward.gpt_forward model !token_id kv_caches in
    let constrained = Symbolic.apply sym_ctx logits.Tensor.data in
    let scaled = Array.map (fun x -> x /. temperature) constrained in
    let probs = Tensor.softmax (Tensor.make_param [|Array.length scaled|] scaled) in
    token_id := Utils.weighted_choice probs.Tensor.data;
    if !token_id = bos_id then pos := Model.block_size
    else begin
      Symbolic.td_update sym_ctx logits.Tensor.data !token_id;
      Symbolic.record_token sym_ctx !token_id;
      Buffer.add_string buf vocab.(!token_id);
      incr pos
    end
  done;
  Buffer.contents buf

(* chat: Format user input as a chat turn, generate assistant response.
   Encodes "<|user|> {input} <|assistant|>" as the prompt, prefills
   the KV cache, then samples until <|user|> or BOS is produced. *)
let chat model know ?concept_know tok user_input temperature =
  Tensor.training := false;
  let sym_ctx = Symbolic.create ?concept_know know in
  let kv_caches = Array.init Model.n_layer (fun _ -> Model.make_kv_cache ()) in
  let prompt = Printf.sprintf "<|user|> %s <|assistant|>" user_input in
  let prompt_ids = Bpe.encode tok prompt in
  let buf = Buffer.create 256 in
  let n_prompt = Array.length prompt_ids in
  (* Prefill: feed prompt tokens without sampling *)
  for i = 0 to n_prompt - 2 do
    ignore (Forward.gpt_forward model prompt_ids.(i) kv_caches);
    if i > 0 then
      Symbolic.record_token sym_ctx prompt_ids.(i)
  done;
  let token_id = ref prompt_ids.(n_prompt - 1) in
  Symbolic.record_token sym_ctx !token_id;
  (* Generate: sample until <|user|> or BOS or max length *)
  let pos = ref n_prompt in
  while !pos < Model.block_size do
    let logits = Forward.gpt_forward model !token_id kv_caches in
    let constrained = Symbolic.apply sym_ctx logits.Tensor.data in
    let scaled = Array.map (fun x -> x /. temperature) constrained in
    let probs = Tensor.softmax (Tensor.make_param [|Array.length scaled|] scaled) in
    token_id := Utils.weighted_choice probs.Tensor.data;
    if !token_id = tok.Bpe.bos_id || !token_id = tok.Bpe.user_id then
      pos := Model.block_size
    else begin
      Symbolic.td_update sym_ctx logits.Tensor.data !token_id;
      Symbolic.record_token sym_ctx !token_id;
      Buffer.add_string buf know.Symbolic.vocab.(!token_id);
      incr pos
    end
  done;
  String.trim (Buffer.contents buf)
