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

   BPE tokenizer is cached to disk after first run for instant startup.
   No symbolic constraints — raw logits with top-k sampling only. *)

let () = Random.init 42

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
let train_step model params adam step tokens =
  let n = Array.length tokens in
  let tokens =
    if n <= Vidya.Model.block_size + 1 then tokens
    else Array.sub tokens (n - Vidya.Model.block_size - 1)
           (Vidya.Model.block_size + 1) in
  let (loss, _) = Vidya.Train.compute_loss model tokens in
  Vidya.Tensor.backward loss;
  Vidya.Train.clip_grad_norm params;
  Vidya.Train.adam_step_fixed params adam step 1e-4;
  loss.Vidya.Tensor.data.(0)

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

(* Train on a single Q+A pair by repeating the gradient step
   n_repeats times. Returns the final loss value and updates
   the step counter. This hammers the pair into the model —
   more repeats = stronger learning but more forgetting risk. *)
let train_on_pair model params adam step tok prompt response n_repeats =
  let train_text =
    Printf.sprintf "<|user|> %s <|assistant|> %s" prompt response in
  let tokens = Vidya.Bpe.encode tok train_text in
  let last_loss = ref 0.0 in
  for _ = 1 to n_repeats do
    last_loss := train_step model params adam !step tokens;
    incr step
  done;
  !last_loss

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

(* Human-in-the-loop reinforcement learning loop.
   1. User types a prompt (5 turns per conversation)
   2. Model generates 5 candidate responses
   3. User picks the best (1-5) or types a better one
   4. Model trains on the chosen Q+A pair (3 gradient steps)
   5. Experience replay: one step on a random base corpus doc
   6. After 5 turns, context resets (new conversation)

   Context resets every 5 turns to keep generation quality high
   (256-token window fills up fast). Adam optimizer state persists
   across resets so momentum builds smoothly over the whole session.

   Experience replay prevents catastrophic forgetting: after each
   RL step, we train on a random doc from the original corpus to
   remind the model of its base knowledge. *)
let run_interactive_training model params tok train_checkpoint replay_docs =
  Printf.printf "\n";
  Printf.printf "  ╔══════════════════════════════════════════╗\n";
  Printf.printf "  ║       interactive training mode          ║\n";
  Printf.printf "  ║  type 'quit' or ctrl-c to save & exit   ║\n";
  Printf.printf "  ╚══════════════════════════════════════════╝\n\n%!";
  let history = Buffer.create 4096 in
  let adam = Vidya.Train.init_adam params in
  let step = ref 0 in
  let turn = ref 0 in
  let turns_per_convo = 5 in
  let n_replay = Array.length replay_docs in
  let running = ref true in
  (* Install Ctrl+C handler so we save the checkpoint on interrupt.
     Without this, Ctrl+C kills the process and all training is lost. *)
  Sys.set_signal Sys.sigint (Sys.Signal_handle (fun _ ->
    Printf.printf "\n  saving checkpoint...\n%!";
    Vidya.Train.save_checkpoint train_checkpoint params;
    exit 0
  ));
  while !running do
    (* Show turn counter so the user knows when context resets *)
    Printf.printf "  [%d/%d] you > %!" (!turn + 1) turns_per_convo;
    match input_line stdin with
    | exception End_of_file -> running := false
    | input ->
      let trimmed = String.trim input in
      if trimmed = "quit" || trimmed = "exit" then
        running := false
      else begin
        (* Append user turn to history for generation context *)
        Buffer.add_string history
          (Printf.sprintf "<|user|> %s <|assistant|>" trimmed);
        let hist_str = Buffer.contents history in
        (* Generate 5 candidate responses at temperature 0.7 *)
        let responses = generate_candidates model tok hist_str 5 0.7 in
        Printf.printf "\n";
        Array.iteri (fun i text ->
          Printf.printf "    %d. %s\n\n" (i + 1) text
        ) responses;
        (* User picks a response or types their own *)
        Printf.printf "  [1-5 or type] > %!";
        match input_line stdin with
        | exception End_of_file -> running := false
        | sel_input ->
          let response_text = parse_selection (String.trim sel_input) responses in
          (* Append chosen response to history for future context *)
          Buffer.add_string history
            (Printf.sprintf " %s " response_text);
          (* Train on just this turn, 3 repeats — enough to nudge
             without catastrophic forgetting *)
          let loss = train_on_pair model params adam step tok
            trimmed response_text 3 in
          (* Experience replay: train on a random doc from the base
             corpus to prevent forgetting. One step, same Adam state. *)
          let replay_idx = Random.int n_replay in
          let replay_loss = train_step model params adam !step
            replay_docs.(replay_idx) in
          incr step;
          Printf.printf "  ── trained (loss %.2f, replay %.2f, step %d) ──\n\n%!"
            loss replay_loss !step;
          (* Auto-reset context every 5 turns *)
          incr turn;
          if !turn >= turns_per_convo then begin
            turn := 0;
            Buffer.clear history;
            Printf.printf "  ── new conversation ──\n\n%!"
          end
      end
  done;
  Vidya.Train.save_checkpoint train_checkpoint params

(* ── Mode: Book training ─────────────────────────────────────────── *)

(* Continued pre-training on a text file.
   Each line is treated as an independent document — tokenized
   separately so the model doesn't learn cross-document patterns.
   This is critical for Q+A training files where each line is a
   separate conversation.

   Each epoch shuffles the lines and trains on every one.
   --epochs N repeats the file N times (default 1). For small
   files you need many epochs to get enough gradient steps.

   Progress bar shows completion, loss, and ETA. *)
let run_book_training model params tok train_checkpoint =
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
  (* Pre-tokenize each line independently *)
  let tokenized = Array.map (fun line -> Vidya.Bpe.encode tok line) lines in
  let n_tokens = Array.fold_left (fun acc t -> acc + Array.length t) 0 tokenized in
  let n_steps = n_docs * epochs in
  Printf.printf "--- book training: %s (%d chars, %d tokens, %d docs x %d epochs = %d steps) ---\n%!"
    book_file n_chars n_tokens n_docs epochs n_steps;
  let adam = Vidya.Train.init_adam params in
  let loss_sum = ref 0.0 in
  let t0 = Sys.time () in
  for step = 0 to n_steps - 1 do
    (* Pick the next doc, cycling through each epoch *)
    let doc_idx = step mod n_docs in
    let tokens = tokenized.(doc_idx) in
    let loss = train_step model params adam step tokens in
    loss_sum := !loss_sum +. loss;
    (* Print a permanent log line every 100 steps so you can see
       the loss curve, then show a progress bar between prints *)
    if (step + 1) mod 100 = 0 then begin
      let avg = !loss_sum /. 100.0 in
      let elapsed = Sys.time () -. t0 in
      Printf.printf "\rstep %d / %d | loss %.4f | %.0fs\n%!"
        (step + 1) n_steps avg elapsed;
      loss_sum := 0.0
    end else begin
      let window = (step mod 100) + 1 in
      let avg = !loss_sum /. float_of_int window in
      print_progress ~step:(step + 1) ~n_steps ~avg_loss:avg ~t0
    end
  done;
  Printf.printf "\n%!";
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
  let t_start = Sys.time () in
  let checkpoint_file = "../../microgpt_chat_10m_v3.bin" in
  let train_checkpoint = "../../microgpt_chat_10m_v3_train.bin" in
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
    Array.fold_left (fun acc p -> acc + Array.length p.Vidya.Tensor.data) 0 params in
  Printf.printf "num params: %d\n" total_params;

  let num_steps = 200000 in
  let checkpoint_base = "../../microgpt_chat_10m_v3" in

  if has_flag "--resume" then begin
    (* Resume training from the latest intermediate checkpoint.
       Scans for files like microgpt_chat_10m_v3_10k.bin, _20k.bin, etc.
       and picks the highest step number found. *)
    let step_increment = 10000 in
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
    (* Load and pre-tokenize the base corpus for experience replay.
       This adds ~3s startup but prevents catastrophic forgetting
       during interactive training. *)
    let docs = Vidya.Utils.load_docs input_file in
    let replay_docs = Vidya.Train.pre_tokenize docs tok in
    run_interactive_training model params tok train_checkpoint replay_docs
  end
  else if get_flag_value "--book" <> "" then
    run_book_training model params tok train_checkpoint
  else if has_flag "--chat" then
    run_chat model tok
  else if get_flag_value "--prompt" <> "" then
    run_prompted model tok
  else
    run_inference_test model tok;

  Printf.printf "total time: %.2fs\n" (Sys.time () -. t_start)
