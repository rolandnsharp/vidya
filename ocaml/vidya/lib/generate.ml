(* generate.ml — Text generation (no symbolic constraints)
   ========================================================

   Four generation modes:
   1. sample: unprompted generation from BOS token
   2. prompted: encode a prompt, prefill KV cache, then sample
   3. chat: format user input with turn markers, generate response
   4. chat_rollout: like chat but returns token IDs (for RL training)

   All use raw logits with top-k sampling and special token
   suppression. No symbolic constraint layer. *)

(* sample: Generate text from scratch (starting from BOS).
   Feeds BOS, then autoregressively samples until BOS is produced
   again or block_size is reached. *)
let sample model tok bos_id temperature =
  Tensor.training := false;
  let kv_caches = Array.init Model.n_layer (fun _ -> Model.make_kv_cache ()) in
  let vocab = tok.Bpe.vocab in
  let token_id = ref bos_id in
  let buf = Buffer.create 256 in
  let pos = ref 0 in
  let recent_tokens = Hashtbl.create 64 in
  let rep_penalty = 1.2 in
  while !pos < Model.block_size do
    let logits = Forward.gpt_forward model !token_id kv_caches in
    let l = Array.copy logits.Tensor.data in
    Hashtbl.iter (fun tid _ ->
      if l.(tid) > 0.0 then l.(tid) <- l.(tid) /. rep_penalty
      else l.(tid) <- l.(tid) *. rep_penalty
    ) recent_tokens;
    let scaled = Array.map (fun x -> x /. temperature) l in
    let filtered = Utils.top_k 40 scaled in
    let probs = Tensor.softmax (Tensor.make_param [|Array.length filtered|] filtered) in
    token_id := Utils.weighted_choice probs.Tensor.data;
    if !token_id = bos_id then pos := Model.block_size
    else begin
      Hashtbl.replace recent_tokens !token_id true;
      Buffer.add_string buf vocab.(!token_id);
      incr pos
    end
  done;
  Buffer.contents buf

(* prompted: Encode a prompt via BPE, prefill the KV cache with
   prompt tokens (no sampling during prefill), then sample
   continuation tokens autoregressively. *)
let prompted model tok bos_id prompt temperature =
  Tensor.training := false;
  let kv_caches = Array.init Model.n_layer (fun _ -> Model.make_kv_cache ()) in
  let prompt_ids = Bpe.encode tok prompt in
  let vocab = tok.Bpe.vocab in
  let buf = Buffer.create 256 in
  let n_prompt = Array.length prompt_ids in
  for i = 0 to n_prompt - 2 do
    ignore (Forward.gpt_forward model prompt_ids.(i) kv_caches);
    if i > 0 then
      Buffer.add_string buf vocab.(prompt_ids.(i))
  done;
  let token_id = ref prompt_ids.(n_prompt - 1) in
  if !token_id <> bos_id then
    Buffer.add_string buf vocab.(!token_id);
  let pos = ref n_prompt in
  let recent_tokens = Hashtbl.create 64 in
  let rep_penalty = 1.2 in
  while !pos < Model.block_size do
    let logits = Forward.gpt_forward model !token_id kv_caches in
    let l = Array.copy logits.Tensor.data in
    Hashtbl.iter (fun tid _ ->
      if l.(tid) > 0.0 then l.(tid) <- l.(tid) /. rep_penalty
      else l.(tid) <- l.(tid) *. rep_penalty
    ) recent_tokens;
    let scaled = Array.map (fun x -> x /. temperature) l in
    let filtered = Utils.top_k 40 scaled in
    let probs = Tensor.softmax (Tensor.make_param [|Array.length filtered|] filtered) in
    token_id := Utils.weighted_choice probs.Tensor.data;
    if !token_id = bos_id then pos := Model.block_size
    else begin
      Hashtbl.replace recent_tokens !token_id true;
      Buffer.add_string buf vocab.(!token_id);
      incr pos
    end
  done;
  Buffer.contents buf

(* chat: Generate assistant response given full conversation history.
   Prefills KV cache with the full history, then samples with
   special token suppression and top-k filtering. *)
let chat model tok history temperature =
  Tensor.training := false;
  let kv_caches = Array.init Model.n_layer (fun _ -> Model.make_kv_cache ()) in
  let all_ids = Bpe.encode tok history in
  let max_prompt = Model.block_size - 50 in
  let prompt_ids =
    if Array.length all_ids <= max_prompt then all_ids
    else Array.sub all_ids (Array.length all_ids - max_prompt) max_prompt
  in
  let vocab = tok.Bpe.vocab in
  let buf = Buffer.create 256 in
  let n_prompt = Array.length prompt_ids in
  let special_ids = [tok.Bpe.bos_id; tok.Bpe.user_id; tok.Bpe.assistant_id] in
  (* Prefill: feed prompt tokens without sampling *)
  for i = 0 to n_prompt - 2 do
    ignore (Forward.gpt_forward model prompt_ids.(i) kv_caches)
  done;
  let token_id = ref prompt_ids.(n_prompt - 1) in
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
      (* Suppress special tokens, apply repetition penalty, then top-k *)
      let suppressed = Array.copy logit_data in
      List.iter (fun id -> suppressed.(id) <- -1e9) special_ids;
      Hashtbl.iter (fun tid _ ->
        if suppressed.(tid) > 0.0 then
          suppressed.(tid) <- suppressed.(tid) /. rep_penalty
        else
          suppressed.(tid) <- suppressed.(tid) *. rep_penalty
      ) recent_tokens;
      let filtered = Utils.top_k 40 suppressed in
      let scaled = Array.map (fun x -> x /. temperature) filtered in
      let probs = Tensor.softmax (Tensor.make_param [|Array.length scaled|] scaled) in
      token_id := Utils.weighted_choice probs.Tensor.data;
      Hashtbl.replace recent_tokens !token_id true;
      Buffer.add_string buf vocab.(!token_id);
      incr gen
    end
  done;
  String.trim (Buffer.contents buf)

(* chat_rollout: Like chat but returns generated token IDs as an array.
   Used by interactive RL training to build training sequences. *)
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
  let recent_tokens = Hashtbl.create 64 in
  let rep_penalty = 1.2 in
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
      Hashtbl.iter (fun tid _ ->
        if suppressed.(tid) > 0.0 then
          suppressed.(tid) <- suppressed.(tid) /. rep_penalty
        else
          suppressed.(tid) <- suppressed.(tid) *. rep_penalty
      ) recent_tokens;
      let filtered = Utils.top_k 40 suppressed in
      let scaled = Array.map (fun x -> x /. temperature) filtered in
      let probs = Tensor.softmax (Tensor.make_param [|Array.length scaled|] scaled) in
      token_id := Utils.weighted_choice probs.Tensor.data;
      Hashtbl.replace recent_tokens !token_id true;
      gen_tokens.(!gen) <- !token_id;
      incr gen
    end
  done;
  (prompt_ids, Array.sub gen_tokens 0 !gen)
