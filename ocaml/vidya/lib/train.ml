(* train.ml — Adam optimizer, LR schedule, training loop, checkpoints
   ====================================================================

   Adam with cosine LR schedule, warmup, and gradient clipping.
   Lower peak LR (0.001) stretched over 200K steps for the 10M
   param model. Longer warmup (2000 steps) for stability.
   Gradient clipping at norm 1.0 prevents explosions.

   Running average loss over 2500-step windows shows the real trend
   (per-doc loss bounces 2.4-4.6 depending on doc difficulty). *)

type adam_state = { m : float array; v : float array }

let learning_rate = 0.001
let beta1 = 0.9
let beta2 = 0.999
let eps_adam = 1e-8
let max_grad_norm = 1.0
let warmup_steps = 2000

let init_adam params =
  let total = Array.fold_left (fun acc p -> acc + Array.length p.Tensor.data) 0 params in
  { m = Array.make total 0.0; v = Array.make total 0.0 }

(* Cosine LR schedule with linear warmup. *)
let get_lr step num_steps =
  if step < warmup_steps then
    learning_rate *. float_of_int step /. float_of_int warmup_steps
  else
    let progress =
      float_of_int (step - warmup_steps)
      /. float_of_int (num_steps - warmup_steps) in
    learning_rate *. 0.5 *. (1.0 +. cos (Float.pi *. progress))

(* Clip gradient norm to max_grad_norm. *)
let clip_grad_norm params =
  let norm_sq = ref 0.0 in
  Array.iter (fun p ->
    for i = 0 to Array.length p.Tensor.grad - 1 do
      norm_sq := !norm_sq +. p.Tensor.grad.(i) *. p.Tensor.grad.(i)
    done
  ) params;
  let norm = sqrt !norm_sq in
  if norm > max_grad_norm then begin
    let scale = max_grad_norm /. norm in
    Array.iter (fun p ->
      for i = 0 to Array.length p.Tensor.grad - 1 do
        p.Tensor.grad.(i) <- p.Tensor.grad.(i) *. scale
      done
    ) params
  end

(* One Adam step: update parameters, zero gradients. *)
let adam_step params adam step num_steps =
  let lr_t = get_lr step num_steps in
  let bc1 = 1.0 /. (1.0 -. beta1 ** float_of_int (step + 1)) in
  let bc2 = 1.0 /. (1.0 -. beta2 ** float_of_int (step + 1)) in
  let offset = ref 0 in
  params |> Array.iter (fun p ->
    let n = Array.length p.Tensor.data in
    for i = 0 to n - 1 do
      let fi = !offset + i in
      adam.m.(fi) <- beta1 *. adam.m.(fi) +. (1.0 -. beta1) *. p.Tensor.grad.(i);
      adam.v.(fi) <- beta2 *. adam.v.(fi) +. (1.0 -. beta2) *. p.Tensor.grad.(i) *. p.Tensor.grad.(i);
      let m_hat = adam.m.(fi) *. bc1 in
      let v_hat = adam.v.(fi) *. bc2 in
      p.Tensor.data.(i) <- p.Tensor.data.(i) -. lr_t *. m_hat /. (sqrt v_hat +. eps_adam);
      p.Tensor.grad.(i) <- 0.0
    done;
    offset := !offset + n)

(* Compute NLL loss for one document. *)
let compute_loss model tokens =
  Tensor.training := true;
  let seq_len = min Model.block_size (Array.length tokens - 1) in
  let input_tokens = Array.sub tokens 0 seq_len in
  let logits = Forward.gpt_forward_batch model input_tokens seq_len in
  let losses = Array.init seq_len (fun i ->
    let logits_i = Tensor.row logits i in
    let probs_i = Tensor.softmax logits_i in
    Tensor.nll probs_i tokens.(i + 1)
  ) in
  (Tensor.mean losses, seq_len)

(* Pre-tokenize all docs at startup. Saves ~500 hash lookups × doc_len
   per training step. *)
let pre_tokenize docs tok =
  let t0 = Sys.time () in
  let tokenized = Array.map (fun doc ->
    Bpe.encode tok doc
  ) docs in
  Printf.printf "pre-tokenized %d docs in %.2fs\n%!"
    (Array.length docs) (Sys.time () -. t0);
  tokenized

(* ── Checkpoint save / load ───────────────────────────────────────── *)

(* Save all parameter data arrays to a binary file using Marshal.
   Format: Marshal'd float array array (one entry per param tensor). *)
let save_checkpoint filename params =
  let oc = open_out_bin filename in
  let data_arrays = Array.map (fun (p : Tensor.value) -> p.data) params in
  Marshal.to_channel oc data_arrays [Marshal.No_sharing];
  close_out oc;
  let total = Array.fold_left (fun acc (p : Tensor.value) -> acc + Array.length p.data) 0 params in
  Printf.printf "saved checkpoint to %s (%d params)\n%!" filename total

(* Load saved weights into live parameter tensors. Validates counts
   and sizes match before blitting. *)
let load_checkpoint filename params =
  let ic = open_in_bin filename in
  let data_arrays : float array array = Marshal.from_channel ic in
  close_in ic;
  if Array.length data_arrays <> Array.length params then
    failwith (Printf.sprintf "checkpoint has %d param tensors, model has %d"
      (Array.length data_arrays) (Array.length params));
  Array.iteri (fun i saved ->
    if Array.length saved <> Array.length params.(i).Tensor.data then
      failwith (Printf.sprintf "param %d: checkpoint has %d elements, model has %d"
        i (Array.length saved) (Array.length params.(i).Tensor.data));
    Array.blit saved 0 params.(i).Tensor.data 0 (Array.length saved)
  ) data_arrays;
  let total = Array.fold_left (fun acc a -> acc + Array.length a) 0 data_arrays in
  Printf.printf "loaded checkpoint from %s (%d params)\n%!" filename total

(* Main training loop. *)
let format_duration secs =
  let h = int_of_float secs / 3600 in
  let m = (int_of_float secs mod 3600) / 60 in
  let s = int_of_float secs mod 60 in
  if h > 0 then Printf.sprintf "%dh%02dm%02ds" h m s
  else if m > 0 then Printf.sprintf "%dm%02ds" m s
  else Printf.sprintf "%ds" s

let train model params tokenized_docs num_steps ?(checkpoint_base="") () =
  let adam = init_adam params in
  let loss_sum = ref 0.0 in
  let t_start = Sys.time () in
  for step = 0 to num_steps - 1 do
    let tokens = tokenized_docs.(step mod Array.length tokenized_docs) in
    let (loss, _) = compute_loss model tokens in
    loss_sum := !loss_sum +. loss.Tensor.data.(0);
    Tensor.backward loss;
    clip_grad_norm params;
    adam_step params adam step num_steps;
    if (step + 1) mod 2500 = 0 then begin
      let elapsed = Sys.time () -. t_start in
      let steps_done = float_of_int (step + 1) in
      let steps_left = float_of_int (num_steps - step - 1) in
      let eta = elapsed *. steps_left /. steps_done in
      Printf.printf "step %5d / %d | loss %.4f | %s elapsed | %s remaining\n%!"
        (step + 1) num_steps (!loss_sum /. 2500.0)
        (format_duration elapsed) (format_duration eta);
      loss_sum := 0.0
    end;
    (* Intermediate checkpoints every 50K steps *)
    if checkpoint_base <> "" && (step + 1) mod 50000 = 0 && (step + 1) < num_steps then begin
      let filename = Printf.sprintf "%s_%dk.bin" checkpoint_base ((step + 1) / 1000) in
      save_checkpoint filename params
    end
  done

(* ── RL Training ─────────────────────────────────────────────────── *)

let rl_learning_rate = 1e-5
let rl_baseline_alpha = 0.05

(* Find first occurrence of needle in haystack starting from start.
   Returns index or -1 if not found. *)
let find_substring haystack needle start =
  let h_len = String.length haystack in
  let n_len = String.length needle in
  let result = ref (-1) in
  let i = ref start in
  while !result = -1 && !i <= h_len - n_len do
    if String.sub haystack !i n_len = needle then
      result := !i
    else
      incr i
  done;
  !result

(* Extract first-turn prompts from raw training docs.
   Each prompt is everything up to and including the first <|assistant|> marker.
   Returns shuffled array of prompt strings ready for chat_rollout. *)
let extract_prompts docs =
  let marker = "<|assistant|>" in
  let marker_len = String.length marker in
  let prompts = ref [] in
  Array.iter (fun doc ->
    let pos = find_substring doc marker 0 in
    if pos >= 0 then
      prompts := String.sub doc 0 (pos + marker_len) :: !prompts
  ) docs;
  let arr = Array.of_list !prompts in
  Utils.shuffle arr;
  Printf.printf "extracted %d prompts for RL training\n%!" (Array.length arr);
  arr

(* Hand-crafted reward function for a generated response.
   Scores length, diversity, repetition, ending quality, and relevance.
   Returns a float in roughly [-1, +1]. *)
let compute_rl_reward response_tokens prompt_tokens vocab =
  let n = Array.length response_tokens in
  if n = 0 then -1.0
  else
    (* Length: prefer 15-80 tokens *)
    let length_r =
      if n < 5 then -1.0
      else if n < 15 then float_of_int (n - 5) /. 10.0 *. 0.5
      else if n <= 80 then 1.0
      else if n <= 120 then 1.0 -. float_of_int (n - 80) /. 40.0
      else -0.5
    in
    (* Diversity: unique tokens / total tokens *)
    let seen = Hashtbl.create 64 in
    Array.iter (fun t -> Hashtbl.replace seen t true) response_tokens;
    let diversity_r =
      float_of_int (Hashtbl.length seen) /. float_of_int n in
    (* No repeat: 1 - repeated_bigrams / total_bigrams *)
    let no_repeat_r =
      if n < 2 then 1.0
      else begin
        let bigrams = Hashtbl.create 64 in
        let repeated = ref 0 in
        for i = 0 to n - 2 do
          let key = response_tokens.(i) * 65536 + response_tokens.(i + 1) in
          if Hashtbl.mem bigrams key then incr repeated
          else Hashtbl.replace bigrams key true
        done;
        1.0 -. float_of_int !repeated /. float_of_int (n - 1)
      end
    in
    (* Ending: reward natural sentence endings *)
    let last_str = vocab.(response_tokens.(n - 1)) in
    let last_ch =
      if String.length last_str > 0 then
        last_str.[String.length last_str - 1]
      else ' '
    in
    let ending_r =
      if last_ch = '.' || last_ch = '?' || last_ch = '!' then 0.5
      else -0.5
    in
    (* Relevance: prompt token overlap *)
    let prompt_set = Hashtbl.create 64 in
    Array.iter (fun t -> Hashtbl.replace prompt_set t true) prompt_tokens;
    let overlap = ref 0 in
    Array.iter (fun t ->
      if Hashtbl.mem prompt_set t then incr overlap
    ) response_tokens;
    let relevance_r = min 1.0 (float_of_int !overlap /. 3.0) in
    0.3 *. length_r +. 0.3 *. diversity_r +. 0.2 *. no_repeat_r
    +. 0.1 *. ending_r +. 0.1 *. relevance_r

(* Compute NLL loss only on response token positions.
   Position i predicts tokens.(i+1). Response tokens start at
   index response_start in the tokens array. Only positions where
   the target is a response token contribute to the loss. *)
let compute_loss_response model tokens response_start =
  Tensor.training := true;
  let seq_len = min Model.block_size (Array.length tokens - 1) in
  let input_tokens = Array.sub tokens 0 seq_len in
  let logits = Forward.gpt_forward_batch model input_tokens seq_len in
  let first_pos = max 0 (response_start - 1) in
  let last_pos = seq_len - 1 in
  if first_pos > last_pos then None
  else begin
    let n_loss = last_pos - first_pos + 1 in
    let losses = Array.init n_loss (fun j ->
      let i = first_pos + j in
      let logits_i = Tensor.row logits i in
      let probs_i = Tensor.softmax logits_i in
      Tensor.nll probs_i tokens.(i + 1)
    ) in
    Some (Tensor.mean losses)
  end

(* Adam step with fixed learning rate (no cosine schedule). *)
let adam_step_fixed params adam step lr =
  let bc1 = 1.0 /. (1.0 -. beta1 ** float_of_int (step + 1)) in
  let bc2 = 1.0 /. (1.0 -. beta2 ** float_of_int (step + 1)) in
  let offset = ref 0 in
  params |> Array.iter (fun p ->
    let n = Array.length p.Tensor.data in
    for i = 0 to n - 1 do
      let fi = !offset + i in
      adam.m.(fi) <- beta1 *. adam.m.(fi) +. (1.0 -. beta1) *. p.Tensor.grad.(i);
      adam.v.(fi) <- beta2 *. adam.v.(fi) +. (1.0 -. beta2) *. p.Tensor.grad.(i) *. p.Tensor.grad.(i);
      let m_hat = adam.m.(fi) *. bc1 in
      let v_hat = adam.v.(fi) *. bc2 in
      p.Tensor.data.(i) <- p.Tensor.data.(i) -. lr *. m_hat /. (sqrt v_hat +. eps_adam);
      p.Tensor.grad.(i) <- 0.0
    done;
    offset := !offset + n)

(* RL training loop.
   ExIt: generate N=4 responses per prompt, keep best, SFT on it.
   REINFORCE: generate → score → policy gradient update on response tokens.

   Usage:
     rl_train model params tok docs ~num_steps:1000 ~mode:`ExIt ()
     rl_train model params tok docs ~num_steps:1000 ~mode:`Reinforce () *)
let rl_train model params tok docs
    ?(num_steps = 1000) ?(mode = `ExIt) () =
  let prompts = extract_prompts docs in
  if Array.length prompts = 0 then
    Printf.printf "no prompts found, skipping RL\n%!"
  else begin
    let adam = init_adam params in
    let baseline = ref 0.0 in
    let reward_sum = ref 0.0 in
    let loss_sum = ref 0.0 in
    let steps_counted = ref 0 in
    let t_start = Sys.time () in
    let mode_name = match mode with
      | `ExIt -> "ExIt" | `Reinforce -> "REINFORCE" in
    Printf.printf "RL training: %s, %d steps, lr=%.1e\n%!"
      mode_name num_steps rl_learning_rate;

    for step = 0 to num_steps - 1 do
      let prompt = prompts.(Random.int (Array.length prompts)) in

      (match mode with
      | `ExIt ->
        (* Generate 4 responses, keep best, supervised train on it *)
        let best_reward = ref neg_infinity in
        let best_prompt_ids = ref [||] in
        let best_gen_tokens = ref [||] in
        for _ = 0 to 3 do
          let (prompt_ids, gen_tokens) =
            Generate.chat_rollout model tok prompt 0.7 in
          let r = compute_rl_reward gen_tokens prompt_ids tok.Bpe.vocab in
          if r > !best_reward then begin
            best_reward := r;
            best_prompt_ids := prompt_ids;
            best_gen_tokens := gen_tokens
          end
        done;
        reward_sum := !reward_sum +. !best_reward;
        if Array.length !best_gen_tokens > 0 then begin
          let seq = Array.append !best_prompt_ids !best_gen_tokens in
          let (loss, _) = compute_loss model seq in
          loss_sum := !loss_sum +. loss.Tensor.data.(0);
          incr steps_counted;
          Tensor.backward loss;
          clip_grad_norm params;
          adam_step_fixed params adam step rl_learning_rate
        end

      | `Reinforce ->
        let (prompt_ids, gen_tokens) =
          Generate.chat_rollout model tok prompt 0.7 in
        let r = compute_rl_reward gen_tokens prompt_ids tok.Bpe.vocab in
        let advantage = r -. !baseline in
        baseline := !baseline +. rl_baseline_alpha *. (r -. !baseline);
        reward_sum := !reward_sum +. r;
        if Array.length gen_tokens > 0
           && Float.abs advantage > 1e-6 then begin
          let seq = Array.append prompt_ids gen_tokens in
          let response_start = Array.length prompt_ids in
          match compute_loss_response model seq response_start with
          | None -> ()
          | Some nll_loss ->
            (* loss = advantage * NLL
               positive advantage → minimize NLL → reinforce
               negative advantage → maximize NLL → suppress *)
            let loss = Tensor.scale nll_loss advantage in
            loss_sum := !loss_sum +. loss.Tensor.data.(0);
            incr steps_counted;
            Tensor.backward loss;
            clip_grad_norm params;
            adam_step_fixed params adam step rl_learning_rate
        end);

      if (step + 1) mod 50 = 0 then begin
        let elapsed = Sys.time () -. t_start in
        let n = float_of_int (max 1 !steps_counted) in
        Printf.printf
          "rl %4d / %d | reward %.3f | loss %.4f | baseline %.3f | %s\n%!"
          (step + 1) num_steps
          (!reward_sum /. 50.0) (!loss_sum /. n)
          !baseline (format_duration elapsed);
        reward_sum := 0.0;
        loss_sum := 0.0;
        steps_counted := 0
      end
    done
  end
