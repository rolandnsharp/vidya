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

(* chat: Generate assistant response given full conversation history.
   history is everything up to and including the latest "<|assistant|>".
   Prefills KV cache with the full history, then samples. *)
let chat model know ?concept_know tok history temperature =
  Tensor.training := false;
  let sym_ctx = Symbolic.create ?concept_know know in
  let kv_caches = Array.init Model.n_layer (fun _ -> Model.make_kv_cache ()) in
  (* Truncate history to fit in context window, keeping most recent turns *)
  let all_ids = Bpe.encode tok history in
  let max_prompt = Model.block_size - 50 in
  let prompt_ids =
    if Array.length all_ids <= max_prompt then all_ids
    else Array.sub all_ids (Array.length all_ids - max_prompt) max_prompt
  in
  let buf = Buffer.create 256 in
  let n_prompt = Array.length prompt_ids in
  let special_ids = [tok.Bpe.bos_id; tok.Bpe.user_id; tok.Bpe.assistant_id] in
  (* Prefill: feed prompt tokens without sampling *)
  for i = 0 to n_prompt - 2 do
    ignore (Forward.gpt_forward model prompt_ids.(i) kv_caches);
    if i > 0 then
      Symbolic.record_token sym_ctx prompt_ids.(i)
  done;
  let token_id = ref prompt_ids.(n_prompt - 1) in
  Symbolic.record_token sym_ctx !token_id;
  (* Generate: sample up to 200 content tokens.
     Special tokens are suppressed from logits to force content generation.
     Stop early if the model's top unsuppressed pick is a special token. *)
  let max_gen = 200 in
  let gen = ref 0 in
  let stop = ref false in
  let recent_tokens = Hashtbl.create 64 in
  let rep_penalty = 1.3 in
  while !gen < max_gen && not !stop do
    let logits = Forward.gpt_forward model !token_id kv_caches in
    let logit_data = logits.Tensor.data in
    (* Check if model naturally wants to stop (top token is special) *)
    let top_id = ref 0 in
    let top_val = ref neg_infinity in
    Array.iteri (fun i v -> if v > !top_val then (top_id := i; top_val := v))
      logit_data;
    if !gen > 5 && List.mem !top_id special_ids then
      stop := true
    else begin
      (* Suppress special tokens *)
      let suppressed = Array.copy logit_data in
      List.iter (fun id -> suppressed.(id) <- -1e9) special_ids;
      (* Repetition penalty: divide positive logits, multiply negative ones *)
      Hashtbl.iter (fun tok_id _ ->
        if suppressed.(tok_id) > 0.0 then
          suppressed.(tok_id) <- suppressed.(tok_id) /. rep_penalty
        else
          suppressed.(tok_id) <- suppressed.(tok_id) *. rep_penalty
      ) recent_tokens;
      let filtered = Utils.top_k 40 suppressed in
      let scaled = Array.map (fun x -> x /. temperature) filtered in
      let probs = Tensor.softmax (Tensor.make_param [|Array.length scaled|] scaled) in
      token_id := Utils.weighted_choice probs.Tensor.data;
      Hashtbl.replace recent_tokens !token_id true;
      Symbolic.td_update sym_ctx logits.Tensor.data !token_id;
      Symbolic.record_token sym_ctx !token_id;
      Buffer.add_string buf know.Symbolic.vocab.(!token_id);
      incr gen
    end
  done;
  String.trim (Buffer.contents buf)

(* chat_rollout: Like chat but returns generated token IDs as an array.
   Used by RL training to build training sequences and compute rewards. *)
let chat_rollout model tok history temperature =
  Tensor.training := false;
  let kv_caches = Array.init Model.n_layer (fun _ -> Model.make_kv_cache ()) in
  let all_ids = Bpe.encode tok history in
  let max_prompt = Model.block_size - 50 in
  let prompt_ids =
    if Array.length all_ids <= max_prompt then all_ids
    else Array.sub all_ids (Array.length all_ids - max_prompt) max_prompt
  in
  let n_prompt = Array.length prompt_ids in
  let special_ids = [tok.Bpe.bos_id; tok.Bpe.user_id; tok.Bpe.assistant_id] in
  for i = 0 to n_prompt - 2 do
    ignore (Forward.gpt_forward model prompt_ids.(i) kv_caches)
  done;
  let token_id = ref prompt_ids.(n_prompt - 1) in
  let max_gen = 200 in
  let gen_tokens = Array.make max_gen 0 in
  let gen = ref 0 in
  let stop = ref false in
  while !gen < max_gen && not !stop do
    let logits = Forward.gpt_forward model !token_id kv_caches in
    let logit_data = logits.Tensor.data in
    let top_id = ref 0 in
    let top_val = ref neg_infinity in
    Array.iteri (fun i v -> if v > !top_val then (top_id := i; top_val := v))
      logit_data;
    if !gen > 5 && List.mem !top_id special_ids then
      stop := true
    else begin
      let suppressed = Array.copy logit_data in
      List.iter (fun id -> suppressed.(id) <- -1e9) special_ids;
      let filtered = Utils.top_k 40 suppressed in
      let scaled = Array.map (fun x -> x /. temperature) filtered in
      let probs = Tensor.softmax (Tensor.make_param [|Array.length scaled|] scaled) in
      token_id := Utils.weighted_choice probs.Tensor.data;
      gen_tokens.(!gen) <- !token_id;
      incr gen
    end
  done;
  (prompt_ids, Array.sub gen_tokens 0 !gen)
