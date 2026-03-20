(* generate.ml — Text generation with GPU-backed inference
   ========================================================

   Four generation modes:
   1. sample: unprompted generation from BOS token
   2. prompted: encode a prompt, prefill KV cache, then sample
   3. chat: format user input with turn markers, generate response
   4. chat_rollout: like chat but returns token IDs (for RL training)

   Logits are computed on GPU, then downloaded to CPU for sampling.
   Sampling is tiny (~2K vocab) so there's no benefit to keeping
   it on GPU. The GPU does the heavy lifting (matmuls, attention);
   the CPU handles the cheap part (top-k, softmax, random choice). *)

(* ── Shared sampling logic ───────────────────────────────────────── *)

(* Download logits from GPU, apply repetition penalty, temperature
   scale, top-k filter, softmax, and sample one token. *)
let sample_next_token logits recent_tokens rep_penalty temperature special_suppress =
  let logit_cpu = Tensor.to_cpu logits.Tensor.data in
  (* Suppress special tokens if requested *)
  List.iter (fun id -> logit_cpu.(id) <- -1e9) special_suppress;
  (* Repetition penalty *)
  Hashtbl.iter (fun tid _ ->
    if logit_cpu.(tid) > 0.0 then
      logit_cpu.(tid) <- logit_cpu.(tid) /. rep_penalty
    else
      logit_cpu.(tid) <- logit_cpu.(tid) *. rep_penalty
  ) recent_tokens;
  (* Temperature + top-k + softmax on CPU *)
  let scaled = Array.map (fun x -> x /. temperature) logit_cpu in
  let filtered = Utils.top_k 40 scaled in
  let probs = Tensor.softmax_cpu filtered in
  Utils.weighted_choice probs

(* Check if the model's top logit is a special token (natural stop). *)
let wants_to_stop logits special_ids =
  let logit_cpu = Tensor.to_cpu logits.Tensor.data in
  let top_id = ref 0 in
  let top_val = ref neg_infinity in
  Array.iteri (fun i v ->
    if v > !top_val then (top_id := i; top_val := v)
  ) logit_cpu;
  List.mem !top_id special_ids

(* ── Generation modes ────────────────────────────────────────────── *)

(* sample: Generate from BOS token until BOS appears again. *)
let sample model tok bos_id temperature =
  Tensor.training := false;
  let kv_caches = Array.init Model.n_layer (fun _ -> Model.make_kv_cache ()) in
  let vocab = tok.Bpe.vocab in
  let buf = Buffer.create 256 in
  let token_id = ref bos_id in
  let pos = ref 0 in
  let recent = Hashtbl.create 64 in
  while !pos < Model.block_size do
    let logits = Forward.gpt_forward model !token_id kv_caches in
    token_id := sample_next_token logits recent 1.2 temperature [];
    if !token_id = bos_id then pos := Model.block_size
    else begin
      Hashtbl.replace recent !token_id true;
      Buffer.add_string buf vocab.(!token_id);
      incr pos
    end
  done;
  Buffer.contents buf

(* prompted: Prefill KV cache with prompt, then sample continuation. *)
let prompted model tok bos_id prompt temperature =
  Tensor.training := false;
  let kv_caches = Array.init Model.n_layer (fun _ -> Model.make_kv_cache ()) in
  let prompt_ids = Bpe.encode tok prompt in
  let vocab = tok.Bpe.vocab in
  let buf = Buffer.create 256 in
  let n_prompt = Array.length prompt_ids in
  (* Prefill: process prompt tokens without sampling *)
  for i = 0 to n_prompt - 2 do
    ignore (Forward.gpt_forward model prompt_ids.(i) kv_caches);
    if i > 0 then Buffer.add_string buf vocab.(prompt_ids.(i))
  done;
  let token_id = ref prompt_ids.(n_prompt - 1) in
  if !token_id <> bos_id then Buffer.add_string buf vocab.(!token_id);
  let pos = ref n_prompt in
  let recent = Hashtbl.create 64 in
  while !pos < Model.block_size do
    let logits = Forward.gpt_forward model !token_id kv_caches in
    token_id := sample_next_token logits recent 1.2 temperature [];
    if !token_id = bos_id then pos := Model.block_size
    else begin
      Hashtbl.replace recent !token_id true;
      Buffer.add_string buf vocab.(!token_id);
      incr pos
    end
  done;
  Buffer.contents buf

(* chat: Generate assistant response given conversation history.
   Suppresses special tokens during generation to force content. *)
let chat model tok history temperature =
  Tensor.training := false;
  let kv_caches = Array.init Model.n_layer (fun _ -> Model.make_kv_cache ()) in
  let all_ids = Bpe.encode tok history in
  let max_prompt = Model.block_size - 50 in
  let prompt_ids =
    if Array.length all_ids <= max_prompt then all_ids
    else Array.sub all_ids (Array.length all_ids - max_prompt) max_prompt in
  let vocab = tok.Bpe.vocab in
  let buf = Buffer.create 256 in
  let n_prompt = Array.length prompt_ids in
  let special_ids = [tok.Bpe.bos_id; tok.Bpe.user_id; tok.Bpe.assistant_id] in
  (* Prefill *)
  for i = 0 to n_prompt - 2 do
    ignore (Forward.gpt_forward model prompt_ids.(i) kv_caches)
  done;
  let token_id = ref prompt_ids.(n_prompt - 1) in
  let max_gen = 200 in
  let gen = ref 0 in
  let stop = ref false in
  let recent = Hashtbl.create 64 in
  while !gen < max_gen && not !stop do
    let logits = Forward.gpt_forward model !token_id kv_caches in
    if !gen > 5 && wants_to_stop logits special_ids then
      stop := true
    else begin
      token_id := sample_next_token logits recent 1.3 temperature special_ids;
      Hashtbl.replace recent !token_id true;
      Buffer.add_string buf vocab.(!token_id);
      incr gen
    end
  done;
  String.trim (Buffer.contents buf)

(* chat_rollout: Like chat but returns token IDs for RL training. *)
let chat_rollout model tok history temperature =
  Tensor.training := false;
  let kv_caches = Array.init Model.n_layer (fun _ -> Model.make_kv_cache ()) in
  let all_ids = Bpe.encode tok history in
  let max_prompt = Model.block_size - 50 in
  let prompt_ids =
    if Array.length all_ids <= max_prompt then all_ids
    else Array.sub all_ids (Array.length all_ids - max_prompt) max_prompt in
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
  let recent = Hashtbl.create 64 in
  while !gen < max_gen && not !stop do
    let logits = Forward.gpt_forward model !token_id kv_caches in
    if !gen > 5 && wants_to_stop logits special_ids then
      stop := true
    else begin
      token_id := sample_next_token logits recent 1.2 temperature special_ids;
      Hashtbl.replace recent !token_id true;
      gen_tokens.(!gen) <- !token_id;
      incr gen
    end
  done;
  (prompt_ids, Array.sub gen_tokens 0 !gen)
