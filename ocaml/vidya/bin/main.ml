(* main.ml — CLI entry point for vidya
   =====================================

   Usage:
     dune exec bin/main.exe              → train from scratch + save + generate
     dune exec bin/main.exe -- --resume  → resume training from latest checkpoint
     dune exec bin/main.exe -- --load    → load checkpoint + generate
     dune exec bin/main.exe -- --load --prompt "The Soul is"
                                         → load + prompted generation
     dune exec bin/main.exe -- --load --chat
                                         → load + interactive chat mode
     dune exec bin/main.exe -- --load --train
                                         → load + interactive RL training
     dune exec bin/main.exe -- --load --book file.txt
                                         → load + train on a text file
     dune exec bin/main.exe -- --load --book file.txt --epochs 50
                                         → same but repeat file 50 times
     dune exec bin/main.exe -- --load --book file.txt --start-step 15000
                                         → skip ahead (first run only, ignored if checkpoint exists)

   BPE tokenizer is cached to disk after first run for instant startup.
   No symbolic constraints — raw logits with top-k sampling only. *)

let () = Random.init 42
let () = Vidya.Gpu.init ()

(* ── CLI helpers ─────────────────────────────────────────────────── *)

(* Check if a flag appears anywhere in argv *)
let has_flag flag =
  let found = ref false in
  for i = 1 to Array.length Sys.argv - 1 do
    if Sys.argv.(i) = flag then found := true
  done;
  !found

(* Get the value after a flag in argv, or "" if not found *)
let get_flag_value flag =
  let value = ref "" in
  for i = 1 to Array.length Sys.argv - 2 do
    if Sys.argv.(i) = flag then value := Sys.argv.(i + 1)
  done;
  !value

(* Read a file into a string, or "" if it doesn't exist *)
let read_file filename =
  if Sys.file_exists filename then begin
    let ic = open_in filename in
    let n = in_channel_length ic in
    let s = really_input_string ic n in
    close_in ic;
    s
  end else ""

(* ── Training helpers ────────────────────────────────────────────── *)

(* Single training step: forward pass → backprop → gradient clip → Adam.
   Truncates tokens to fit within block_size if needed.
   Returns the scalar loss value. *)
let _train_step ?(lr=1e-5) model params adam step tokens =
  let n = Array.length tokens in
  let tokens =
    if n <= Vidya.Model.block_size + 1 then tokens
    else Array.sub tokens (n - Vidya.Model.block_size - 1)
           (Vidya.Model.block_size + 1) in
  let (loss, _) = Vidya.Train.compute_loss model tokens in
  Vidya.Tensor.backward loss;
  Vidya.Train.clip_grad_norm params;
  Vidya.Train.adam_step_fixed params adam step lr;
  (Vidya.Gpu.download loss.Vidya.Tensor.data).(0)

(* Decode token IDs back to a string using the BPE vocab table.
   v3 BPE doesn't have a decode function, so we do it manually
   by concatenating the vocab entry for each token ID. *)
let decode_tokens tok gen =
  let buf = Buffer.create 256 in
  Array.iter (fun id ->
    Buffer.add_string buf tok.Vidya.Bpe.vocab.(id)) gen;
  String.trim (Buffer.contents buf)

(* Generate n candidate responses for a given conversation history.
   Uses chat_rollout (no symbolic constraints, top-k 40) at the
   given temperature. Returns a string array of decoded responses. *)
let generate_candidates model tok history n temperature =
  Array.init n (fun _ ->
    let (_, gen) =
      Vidya.Generate.chat_rollout model tok history temperature in
    decode_tokens tok gen)

(* Parse user selection: "1"-"5" returns the corresponding response
   from the array, anything else is treated as a typed-in response. *)
let parse_selection sel responses =
  if String.length sel = 1
     && sel.[0] >= '1' && sel.[0] <= '5' then
    responses.(Char.code sel.[0] - Char.code '1')
  else
    sel

(* Accumulate gradients for one Q+A pair WITHOUT updating weights.
   Call this for each turn in a conversation, then call
   flush_sparse_update at the end to apply one batched step. *)
let accumulate_grad model tok prompt response =
  let train_text =
    Printf.sprintf "<|user|> %s <|assistant|> %s" prompt response in
  let tokens = Vidya.Bpe.encode tok train_text in
  let n = Array.length tokens in
  let tokens =
    if n <= Vidya.Model.block_size + 1 then tokens
    else Array.sub tokens (n - Vidya.Model.block_size - 1)
           (Vidya.Model.block_size + 1) in
  let (loss, _) = Vidya.Train.compute_loss model tokens in
  Vidya.Tensor.backward loss;
  (Vidya.Gpu.download loss.Vidya.Tensor.data).(0)

(* Apply one sparse Adam step from accumulated gradients.
   Only the top 1% of gradients by magnitude get applied.
   Call after accumulating gradients from all turns in a conversation. *)
let flush_sparse_update params adam step ?(top_frac=0.01) ?(lr=1e-4) () =
  let kept = Vidya.Train.sparse_grad_mask params top_frac in
  Vidya.Train.clip_grad_norm params;
  Vidya.Train.adam_step_fixed params adam !step lr;
  incr step;
  kept

(* Contrastive gradient accumulation: compute forward+backward on chosen
   and rejected responses WITHOUT updating weights. Gradients accumulate
   in the parameter tensors for a later batched Adam step.
   Returns (loss, margin) for display. *)
let _contrastive_accumulate model tok prompt chosen rejected beta =
  let chosen_text =
    Printf.sprintf "<|user|> %s <|assistant|> %s" prompt chosen in
  let rejected_text =
    Printf.sprintf "<|user|> %s <|assistant|> %s" prompt rejected in
  let chosen_tokens = Vidya.Bpe.encode tok chosen_text in
  let rejected_tokens = Vidya.Bpe.encode tok rejected_text in
  Vidya.Train.contrastive_loss model chosen_tokens rejected_tokens beta

(* ── Progress bar ────────────────────────────────────────────────── *)

(* Render a progress bar to stdout using \r to overwrite in place.
   Example: [################..............] 512/1600 | loss 3.214 | 42s left
   bar_width controls the number of characters in the bar itself. *)
let print_progress ~step ~n_steps ~avg_loss ~t0 =
  let pct = float_of_int step /. float_of_int n_steps in
  let bar_width = 30 in
  let filled = int_of_float (pct *. float_of_int bar_width) in
  let bar = String.make filled '#' ^ String.make (bar_width - filled) '.' in
  let elapsed = Sys.time () -. t0 in
  (* ETA: extrapolate remaining time from elapsed time and progress.
     Guard against division by zero on the first step. *)
  let eta = if step > 1 then elapsed *. (1.0 -. pct) /. pct else 0.0 in
  Printf.printf "\r[%s] %d/%d | loss %.3f | %.0fs left%!"
    bar step n_steps avg_loss eta

(* ── Mode: Interactive RL training ───────────────────────────────── *)

(* Interactive RL with sparse gradient masking.

   The model generates 5 candidate responses for each prompt.
   The user picks the best one (or types a better response).
   Selected Q+A pairs are trained with sparse gradients (top 1%)
   and auto-saved to vidya_personality.txt.

   Sparse masking: only the ~490K weights (out of 49M) that got
   the strongest gradient signal from this specific example get
   updated. The other 99% don't move. Like biological learning —
   only the synapses that fired hardest actually change. *)
let run_interactive_training model params tok train_checkpoint
    _personality_docs _corpus_docs =
  let adam = Vidya.Train.init_adam params in
  (* Try loading Adam state from personality checkpoint.
     Old format: flat (float array * float array). New format: per-param arrays.
     Skip loading if dimensions don't match (architecture change). *)
  let adam_file = "../../checkpoints/49m_v1_vidya_personality/adam.bin" in
  if Sys.file_exists adam_file then begin
    Printf.printf "skipping old-format adam state (architecture changed)\n%!"
  end;
  let step = ref 0 in
  Printf.printf "\n";
  Printf.printf "  ╔══════════════════════════════════════════╗\n";
  Printf.printf "  ║     interactive RL (sparse gradients)    ║\n";
  Printf.printf "  ║  top 1%% of gradients updated per step   ║\n";
  Printf.printf "  ║  responses auto-saved to personality     ║\n";
  Printf.printf "  ║  type 'quit' or ctrl-c to exit           ║\n";
  Printf.printf "  ╚══════════════════════════════════════════╝\n\n%!";
  let personality_file = "../../vidya_personality.txt" in
  let history = Buffer.create 4096 in
  let turn = ref 0 in
  let turns_per_convo = 5 in
  let convo_count = ref 0 in
  let saved_count = ref 0 in
  let turn_losses = Array.make turns_per_convo 0.0 in
  let turns_accumulated = ref 0 in
  let running = ref true in
  let total_params =
    Array.fold_left (fun acc p ->
      acc + Vidya.Gpu.numel p.Vidya.Tensor.data) 0 params in
  Sys.set_signal Sys.sigint (Sys.Signal_handle (fun _ ->
    Printf.printf "\n  saving checkpoint...\n%!";
    Vidya.Train.save_checkpoint train_checkpoint params;
    Printf.printf "  %d responses saved to %s\n%!" !saved_count personality_file;
    exit 0
  ));
  while !running do
    Printf.printf "  [%d/%d] you > %!" (!turn + 1) turns_per_convo;
    match input_line stdin with
    | exception End_of_file -> running := false
    | input ->
      let trimmed = String.trim input in
      if trimmed = "quit" || trimmed = "exit" then
        running := false
      else begin
        Buffer.add_string history
          (Printf.sprintf "<|user|> %s <|assistant|>" trimmed);
        let hist_str = Buffer.contents history in
        let responses = generate_candidates model tok hist_str 5 0.7 in
        Printf.printf "\n";
        Array.iteri (fun i text ->
          Printf.printf "    %d. %s\n\n" (i + 1) text
        ) responses;
        Printf.printf "  [1-5, type, or 0 to skip] > %!";
        match input_line stdin with
        | exception End_of_file -> running := false
        | sel_input ->
          let sel = String.trim sel_input in
          if sel = "0" then
            Printf.printf "  ── skipped ──\n\n%!"
          else begin
            let response_text = parse_selection sel responses in
            Buffer.add_string history
              (Printf.sprintf " %s " response_text);
            (* Accumulate gradients — no weight update yet *)
            let loss = accumulate_grad model tok trimmed response_text in
            turn_losses.(!turn) <- loss;
            incr turns_accumulated;
            Printf.printf "  ── loss %.3f (accumulated %d/5) ──\n%!"
              loss !turns_accumulated;
            (* Auto-save Q+A to personality file *)
            let oc = open_out_gen [Open_append; Open_creat] 0o644
              personality_file in
            Printf.fprintf oc "<|user|> %s <|assistant|> %s\n"
              trimmed response_text;
            close_out oc;
            incr saved_count;
          end;
          incr turn;
          if !turn >= turns_per_convo then begin
            (* Flush: one sparse Adam step from all 5 turns *)
            if !turns_accumulated > 0 then begin
              let kept = flush_sparse_update params adam step () in
              let avg_loss = Array.fold_left (+.) 0.0
                (Array.sub turn_losses 0 !turns_accumulated)
                /. float_of_int !turns_accumulated in
              Printf.printf
                "  ── updated: avg loss %.3f | %d/%d weights | conversation %d ──\n\n%!"
                avg_loss kept total_params (!convo_count + 1)
            end;
            turn := 0;
            turns_accumulated := 0;
            Buffer.clear history;
            incr convo_count
          end
      end
  done;
  if !saved_count > 0 then begin
    Printf.printf "  saving checkpoint...\n%!";
    Vidya.Train.save_checkpoint train_checkpoint params
  end;
  Printf.printf "  %d responses saved to %s\n%!" !saved_count personality_file

(* ── Mode: Book training ─────────────────────────────────────────── *)

(* Continued pre-training on a text file (--book flag).

   Each line is treated as an independent document — tokenized
   separately so the model doesn't learn cross-document patterns.
   This is critical for Q+A training files where each line is a
   separate conversation.

   Two key lessons from our first attempt at large-scale training
   (2.4M conversations, loss stuck flat at 2.8 for 16 hours):

   1. COSINE LR SCHEDULE, NOT FIXED LR
      The original code used a fixed learning rate of 1e-4 for all
      steps. This is fine for small RL fine-tuning (few hundred
      steps), but terrible for large-scale continued pre-training.
      The model needs a warmup phase while Adam builds its momentum
      estimates, then a high peak LR to learn aggressively, then a
      slow decay to settle into fine details.

      We use peak 3e-4 — lower than from-scratch training (1e-3)
      because we're building on an existing foundation, not starting
      from random weights. The cosine curve looks like:

        LR
        3e-4 |     ╭──────╮
             |    ╱        ╲
             |   ╱          ╲
             |  ╱            ╲
        0    | ╱              ╲___
             +─────────────────────
               warmup    decay

   2. PER-EPOCH SHUFFLING
      The original code cycled through documents in the same order
      every epoch (step mod n_docs). This means the model always
      sees doc 0 first, then doc 1, then doc 2... and the same
      sequence repeats in epoch 2. The model can learn spurious
      patterns from this ordering (e.g., "conversation about weather
      always follows conversation about food").

      We now use a shuffle index array that gets re-randomized at
      each epoch boundary via Fisher-Yates shuffle. Each epoch sees
      every document exactly once, but in a different random order.

   --epochs N repeats the file N times (default 1). For small
   files you need many epochs to get enough gradient steps.

   Progress bar shows completion, loss, current LR, and ETA. *)
let run_book_training model params tok train_checkpoint book_checkpoint_dir =
  let book_file = get_flag_value "--book" in
  if not (Sys.file_exists book_file) then begin
    Printf.printf "error: %s not found\n%!" book_file;
    exit 1
  end;
  let epochs =
    let e = get_flag_value "--epochs" in
    if e <> "" then int_of_string e else 1 in
  (* Read file and split into non-empty lines, each is a document *)
  let text = read_file book_file in
  let n_chars = String.length text in
  let all_lines = String.split_on_char '\n' text in
  let lines = List.filter (fun s -> String.trim s <> "") all_lines
    |> Array.of_list in
  let n_docs = Array.length lines in
  let tokenized = Array.map (fun line -> Vidya.Bpe.encode tok line) lines in
  let n_tokens = Array.fold_left (fun acc t -> acc + Array.length t) 0 tokenized in
  let n_steps = n_docs * epochs in

  (* Cosine LR schedule with linear warmup.
     Peak 3e-4 (0.3x of from-scratch 1e-3) for continued pre-training.
     Works correctly with resume — step is absolute position. *)
  let peak_lr = 3e-4 in
  let warmup_steps = 2000 in
  let get_lr step =
    if step < warmup_steps then
      peak_lr *. float_of_int step /. float_of_int warmup_steps
    else
      let progress =
        float_of_int (step - warmup_steps)
        /. float_of_int (n_steps - warmup_steps) in
      peak_lr *. 0.5 *. (1.0 +. cos (Float.pi *. progress))
  in

  Printf.printf "--- book training: %s (%d chars, %d tokens, %d docs x %d epochs = %d steps) ---\n%!"
    book_file n_chars n_tokens n_docs epochs n_steps;
  Printf.printf "    cosine LR: peak %.1e, warmup %d steps, ctrl+c safe\n%!" peak_lr warmup_steps;

  (* ── Resume or fresh start ──
     Check the checkpoint directory for saved progress. If found and
     the book file matches, restore weights + Adam state + shuffle order
     and resume from the saved step. Otherwise start fresh. *)
  let bhash = Vidya.Train.book_hash book_file n_docs in
  let (adam, start_step, order) =
    match Vidya.Train.load_book_checkpoint book_checkpoint_dir params with
    | Some (adam, progress) ->
      if progress.bp_book_hash <> bhash then begin
        Printf.printf "WARNING: book file changed since last checkpoint!\n%!";
        Printf.printf "  delete %s/ to start fresh\n%!" book_checkpoint_dir;
        let adam = Vidya.Train.init_adam params in
        let order = Array.init n_docs (fun i -> i) in
        Vidya.Utils.shuffle order;
        (adam, 0, order)
      end else begin
        Printf.printf "    resuming from step %d / %d | lr %.2e | epoch %d\n%!"
          progress.bp_step progress.bp_n_steps
          (get_lr progress.bp_step) progress.bp_epoch;
        (adam, progress.bp_step, progress.bp_order)
      end
    | None ->
      let adam = Vidya.Train.init_adam params in
      let order = Array.init n_docs (fun i -> i) in
      Vidya.Utils.shuffle order;
      (* --start-step N: manually set starting step for first run
         (e.g. to skip data already trained on from previous restarts) *)
      let manual_start =
        let s = get_flag_value "--start-step" in
        if s <> "" then int_of_string s else 0 in
      if manual_start > 0 then
        Printf.printf "    starting from step %d (manual) | lr %.2e\n%!"
          manual_start (get_lr manual_start);
      (adam, manual_start, order)
  in

  let make_progress step = Vidya.Train.{
    bp_step = step;
    bp_epoch = step / n_docs;
    bp_n_docs = n_docs;
    bp_n_steps = n_steps;
    bp_book_hash = bhash;
    bp_order = order;
  } in

  let current_step = ref start_step in
  let loss_sum = ref 0.0 in
  let t0 = Sys.time () in

  (* Ctrl+C handler: save full state (weights + Adam + progress).
     Adam is only saved here (not on periodic saves) to reduce SSD writes. *)
  Sys.set_signal Sys.sigint (Sys.Signal_handle (fun _ ->
    Printf.printf "\n  saving checkpoint (with adam state)...\n%!";
    Vidya.Train.save_book_checkpoint book_checkpoint_dir params adam
      (make_progress !current_step) ~save_adam:true;
    Vidya.Train.save_checkpoint train_checkpoint params;
    exit 0
  ));

  for step = start_step to n_steps - 1 do
    current_step := step;
    (* Reshuffle at epoch boundaries *)
    if step > 0 && step mod n_docs = 0 then begin
      Vidya.Utils.shuffle order;
      Printf.printf "--- epoch %d ---\n%!" (step / n_docs + 1)
    end;
    let doc_idx = order.(step mod n_docs) in
    let tokens = tokenized.(doc_idx) in

    (* Single training step: truncate → forward → loss → backward → clip → Adam *)
    let n = Array.length tokens in
    let tokens =
      if n <= Vidya.Model.block_size + 1 then tokens
      else Array.sub tokens (n - Vidya.Model.block_size - 1)
             (Vidya.Model.block_size + 1) in
    let (loss, _) = Vidya.Train.compute_loss model tokens in
    Vidya.Tensor.backward loss;
    Vidya.Train.clip_grad_norm params;
    let lr = get_lr step in
    Vidya.Train.adam_step_fixed params adam step lr;
    loss_sum := !loss_sum +. (Vidya.Gpu.download loss.Vidya.Tensor.data).(0);

    if (step + 1) mod 100 = 0 then begin
      let avg = !loss_sum /. 100.0 in
      let elapsed = Sys.time () -. t0 in
      Printf.printf "\rstep %d / %d | loss %.4f | lr %.2e | %.0fs\n%!"
        (step + 1) n_steps avg lr elapsed;
      loss_sum := 0.0
    end else begin
      let window = (step mod 100) + 1 in
      let avg = !loss_sum /. float_of_int window in
      print_progress ~step:(step + 1) ~n_steps ~avg_loss:avg ~t0
    end;

    (* Periodic saves: weights + progress only (no Adam, SSD-friendly).
       Every 10K steps (~5.5h). Ctrl+C saves immediately with full state. *)
    if (step + 1) mod 10000 = 0 && (step + 1) < n_steps then begin
      Vidya.Train.save_book_checkpoint book_checkpoint_dir params adam
        (make_progress (step + 1)) ~save_adam:false;
      Vidya.Train.save_checkpoint train_checkpoint params
    end
  done;
  Printf.printf "\n%!";
  (* Final save: full state including Adam *)
  Vidya.Train.save_book_checkpoint book_checkpoint_dir params adam
    (make_progress n_steps) ~save_adam:true;
  Vidya.Train.save_checkpoint train_checkpoint params

(* ── Mode: Interactive chat ──────────────────────────────────────── *)

(* Chat mode: user types, model responds. No training.
   Conversation history accumulates so the model sees prior
   turns when generating (up to the context window limit). *)
let run_chat model tok =
  Printf.printf "--- chat mode (type 'quit' to exit) ---\n%!";
  let history = Buffer.create 1024 in
  let running = ref true in
  while !running do
    Printf.printf "> %!";
    match input_line stdin with
    | exception End_of_file -> running := false
    | input ->
      let trimmed = String.trim input in
      if trimmed = "quit" || trimmed = "exit" then
        running := false
      else begin
        Buffer.add_string history
          (Printf.sprintf "<|user|> %s <|assistant|>" trimmed);
        let response = Vidya.Generate.chat model tok
          (Buffer.contents history) 0.5 in
        Printf.printf "%s\n%!" response;
        Buffer.add_string history (Printf.sprintf " %s " response)
      end
  done

(* ── Mode: Prompted generation ───────────────────────────────────── *)

(* Generate 10 completions for a given prompt at temperature 0.5. *)
let run_prompted model tok =
  let prompt_text = get_flag_value "--prompt" in
  Printf.printf "--- prompted generation ---\n";
  Printf.printf "prompt: %s\n" prompt_text;
  for i = 1 to 10 do
    Vidya.Generate.prompted model tok tok.bos_id prompt_text 0.5
    |> Printf.printf "  %2d: %s\n" i
  done

(* ── Mode: Default inference test ────────────────────────────────── *)

(* Run a batch of test prompts at several temperatures to evaluate
   the model's chat quality. No training, no interaction. *)
let run_inference_test model tok =
  Printf.printf "--- inference (chat test) ---\n";
  let test_prompts = [
    "Hello";
    "How are you?";
    "What is the meaning of life?";
    "Tell me a story";
    "What do you think about music?";
  ] in
  List.iter (fun temp ->
    Printf.printf "--- temperature %.1f ---\n" temp;
    List.iter (fun p ->
      Printf.printf "user: %s\n" p;
      let history = Printf.sprintf "<|user|> %s <|assistant|>" p in
      let response = Vidya.Generate.chat model tok history temp in
      Printf.printf "  → %s\n\n" response
    ) test_prompts
  ) [0.3; 0.5; 0.8]

(* ── Main ────────────────────────────────────────────────────────── *)

let () =
  (* Safety: require --load for modes that use the model.
     Without this, forgetting --load trains on random weights
     and could overwrite the checkpoint with garbage. *)
  if not (has_flag "--load") && not (has_flag "--resume")
     && not (get_flag_value "--expand-from" <> "") then begin
    if has_flag "--train" || get_flag_value "--book" <> ""
       || has_flag "--chat" || get_flag_value "--prompt" <> "" then begin
      Printf.printf "error: --train, --book, --chat, --prompt require --load\n%!";
      Printf.printf "       (this prevents accidentally wiping Mr . Classic's brain)\n%!";
      exit 1
    end
  end;
  let t_start = Sys.time () in
  let checkpoint_file = "../../vidya_103m.bin" in
  let train_checkpoint = "../../vidya_103m_train.bin" in
  let book_checkpoint_dir =
    let book = get_flag_value "--book" in
    if book <> "" then
      let base = Filename.remove_extension (Filename.basename book) in
      "../../checkpoints/103m_" ^ base
    else "../../checkpoints/103m_book" in
  let input_file = "../../chat_input.txt" in

  let tokenizer_file = "../../tokenizer_v3.bin" in

  (* BPE tokenizer: load from cache if available, otherwise train
     from corpus and save. Training is deterministic so the cached
     version is always identical — this just skips the ~3s startup. *)
  let tok =
    if Sys.file_exists tokenizer_file then
      Vidya.Bpe.load_tokenizer tokenizer_file
    else begin
      let docs = Vidya.Utils.load_docs input_file in
      Printf.printf "num docs: %d\n" (Array.length docs);
      let tok = Vidya.Bpe.train docs Vidya.Bpe.n_merges in
      Vidya.Bpe.save_tokenizer tokenizer_file tok;
      tok
    end
  in

  (* Init model with random weights, then either load a checkpoint
     or train from scratch. Prefers _train.bin if it exists (so
     interactive training resumes from where you left off). *)
  let model = Vidya.Model.init tok.vocab_size in
  let params = Vidya.Model.collect_params model in
  let total_params =
    Array.fold_left (fun acc p -> acc + Vidya.Gpu.numel p.Vidya.Tensor.data) 0 params in
  Printf.printf "num params: %d\n" total_params;

  let num_steps = 200000 in
  let checkpoint_base = "../../vidya_103m" in

  (* ── Expand: widen a 10M checkpoint to 15.5M ──────────────────── *)
  if get_flag_value "--expand-from" <> "" then begin
    let old_file = get_flag_value "--expand-from" in
    Printf.printf "expanding %s to 49M format...\n%!" old_file;
    let ic = open_in_bin old_file in
    let old_arrays : float array array = Marshal.from_channel ic in
    close_in ic;
    Printf.printf "loaded %d tensors from old checkpoint\n%!"
      (Array.length old_arrays);
    (* Old model dimensions — detect from wte size using known vocab *)
    let vocab = tok.vocab_size in
    let old_embd = Array.length old_arrays.(0) / vocab in
    Printf.printf "detected old embedding dim: %d (vocab %d)\n%!" old_embd vocab;
    let old_ffn = 4 * old_embd in
    (* New model dimensions from current Model constants *)
    let new_embd = Vidya.Model.n_embd in  (* 320 *)
    let new_ffn = 4 * new_embd in  (* 1280 *)
    (* Expand a row-major matrix [old_rows, old_cols] → [new_rows, new_cols]
       Copies old data into top-left block, zeros elsewhere *)
    let expand_matrix old_data old_rows old_cols new_rows new_cols =
      let arr = Array.make (new_rows * new_cols) 0.0 in
      for r = 0 to old_rows - 1 do
        Array.blit old_data (r * old_cols) arr (r * new_cols) old_cols
      done;
      arr
    in
    (* Expand a norm vector [old_dim] → [new_dim], pad with 1.0 *)
    let expand_norm old_data old_dim new_dim =
      let arr = Array.make new_dim 1.0 in
      Array.blit old_data 0 arr 0 old_dim;
      arr
    in
    (* Expand embedding [vocab, old_embd] → [vocab, new_embd]
       Pad new cols with small noise to avoid RMSNorm issues *)
    let expand_embedding old_data vocab old_dim new_dim =
      let arr = Array.init (vocab * new_dim) (fun i ->
        let col = i mod new_dim in
        if col < old_dim then old_data.(i / new_dim * old_dim + col)
        else Vidya.Utils.random_gauss ~std:0.01 ()
      ) in
      arr
    in
    Printf.printf "vocab size: %d\n%!" vocab;
    (* Expand each tensor according to its position in collect_params:
       [0] wte, [1] embed_norm,
       then per layer: [wq, wk, wv, wo, fc1, fc2, ln1, ln2],
       final: [final_norm] *)
    let n_tensors = Array.length old_arrays in
    let new_arrays = Array.init n_tensors (fun i ->
      let old = old_arrays.(i) in
      let old_len = Array.length old in
      if i = 0 then
        (* wte: [vocab, old_embd] → [vocab, new_embd] *)
        expand_embedding old vocab old_embd new_embd
      else if i = 1 || i = n_tensors - 1 then
        (* embed_norm or final_norm *)
        expand_norm old old_embd new_embd
      else begin
        (* Layer tensor: figure out which one by position within layer *)
        let layer_pos = (i - 2) mod 8 in
        match layer_pos with
        | 0 | 1 | 2 | 3 ->
          (* wq, wk, wv, wo: [old_embd, old_embd] → [new_embd, new_embd] *)
          expand_matrix old old_embd old_embd new_embd new_embd
        | 4 ->
          (* fc1: [old_ffn, old_embd] → [new_ffn, new_embd] *)
          expand_matrix old old_ffn old_embd new_ffn new_embd
        | 5 ->
          (* fc2: [old_embd, old_ffn] → [new_embd, new_ffn] *)
          expand_matrix old old_embd old_ffn new_embd new_ffn
        | 6 | 7 ->
          (* ln1, ln2: [old_embd] → [new_embd] *)
          expand_norm old old_embd new_embd
        | _ ->
          Printf.printf "warning: unknown tensor %d (len %d), copying as-is\n%!" i old_len;
          old
      end
    ) in
    (* Save expanded checkpoint — blit into the model's param arrays *)
    Array.iteri (fun i new_data ->
      let p = params.(i) in
      let n_model = Vidya.Gpu.numel p.Vidya.Tensor.data in
      if Array.length new_data <> n_model then
        Printf.printf "ERROR: tensor %d size mismatch: expanded=%d model=%d\n%!"
          i (Array.length new_data) n_model
      else
        Vidya.Gpu.upload new_data p.Vidya.Tensor.data
    ) new_arrays;
    Vidya.Train.save_checkpoint checkpoint_file params;
    Printf.printf "saved expanded checkpoint to %s\n%!" checkpoint_file;
    Printf.printf "total time: %.2fs\n" (Sys.time () -. t_start);
    exit 0
  end;

  if has_flag "--resume" then begin
    (* Resume training from the latest intermediate checkpoint.
       Scans for files like microgpt_chat_10m_v3_10k.bin, _20k.bin, etc.
       and picks the highest step number found. *)
    let step_increment = 2500 in
    let latest_step = ref 0 in
    let latest_file = ref "" in
    let s = ref step_increment in
    while !s < num_steps do
      let f = Printf.sprintf "%s_%dk.bin" checkpoint_base (!s / 1000) in
      if Sys.file_exists f then begin
        latest_step := !s;
        latest_file := f
      end;
      s := !s + step_increment
    done;
    (* Fall back to the final checkpoint if no intermediates exist *)
    if !latest_file = "" then begin
      if Sys.file_exists checkpoint_file then begin
        latest_file := checkpoint_file;
        latest_step := 0
      end else begin
        Printf.printf "no checkpoint found to resume from\n%!";
        exit 1
      end
    end;
    Printf.printf "resuming from step %d\n%!" !latest_step;
    Vidya.Train.load_checkpoint !latest_file params;
    let docs = Vidya.Utils.load_docs input_file in
    Printf.printf "num docs: %d\n" (Array.length docs);
    let tokenized_docs = Vidya.Train.pre_tokenize docs tok in
    Vidya.Train.train model params tokenized_docs num_steps
      ~start_step:!latest_step ~checkpoint_base ();
    Vidya.Train.save_checkpoint checkpoint_file params

  end else if has_flag "--load" then begin
    Random.self_init ();
    let file =
      if Sys.file_exists train_checkpoint then train_checkpoint
      else checkpoint_file in
    Vidya.Train.load_checkpoint file params

  end else begin
    (* Train from scratch with intermediate checkpoints every 10K steps *)
    let docs = Vidya.Utils.load_docs input_file in
    Printf.printf "num docs: %d\n" (Array.length docs);
    let tokenized_docs = Vidya.Train.pre_tokenize docs tok in
    Vidya.Train.train model params tokenized_docs num_steps
      ~checkpoint_base ();
    Vidya.Train.save_checkpoint checkpoint_file params
  end;

  (* Dispatch to the appropriate mode *)
  if has_flag "--train" then begin
    (* Load and pre-tokenize both replay sources for experience replay:
       1. Personality docs (339) — reinforces Vidya's identity
       2. General corpus (37K) — preserves broad conversational ability
       Without the general corpus, RL quickly causes catastrophic forgetting. *)
    let p_docs = Vidya.Utils.load_docs "../../vidya_personality.txt" in
    let personality_docs = Vidya.Train.pre_tokenize p_docs tok in
    let c_docs = Vidya.Utils.load_docs input_file in
    let corpus_docs = Vidya.Train.pre_tokenize c_docs tok in
    run_interactive_training model params tok train_checkpoint
      personality_docs corpus_docs
  end
  else if get_flag_value "--book" <> "" then
    run_book_training model params tok train_checkpoint book_checkpoint_dir
  else if has_flag "--chat" then
    run_chat model tok
  else if get_flag_value "--prompt" <> "" then
    run_prompted model tok
  else
    run_inference_test model tok;

  Printf.printf "total time: %.2fs\n" (Sys.time () -. t_start)
