(* main.ml — CLI entry point for vidya
   =====================================

   Usage:
     dune exec bin/main.exe              → train from scratch + save + generate
     dune exec bin/main.exe -- --load    → load checkpoint + generate
     dune exec bin/main.exe -- --load --chat
                                         → load + interactive chat mode
     dune exec bin/main.exe -- --load --train
                                         → load + interactive training mode
     dune exec bin/main.exe -- --load --prompt "hello"
                                         → generate 5 responses (CLI mode)
     dune exec bin/main.exe -- --load --teach 3
                                         → train on response #3 (CLI mode)
     dune exec bin/main.exe -- --load --teach "response text"
                                         → train on typed response (CLI mode)
     dune exec bin/main.exe -- --load --rl
                                         → load + ExIt RL fine-tuning
     dune exec bin/main.exe -- --load --rl --reinforce
                                         → load + REINFORCE RL fine-tuning
     dune exec bin/main.exe -- --save-bpe
                                         → save BPE tokenizer and exit

   BPE tokenizer is saved to disk after first training. Subsequent calls
   load it directly (<1s startup). Symbolic + concept knowledge only built
   for modes that need them (chat, inference, rl).

   Interactive training modes (--train, --prompt/--teach) accumulate
   conversation history in context.md and save trained weights to a
   separate checkpoint file. *)

let () = Random.init 42

(* Check if a flag appears anywhere in argv *)
let has_flag flag =
  let found = ref false in
  for i = 1 to Array.length Sys.argv - 1 do
    if Sys.argv.(i) = flag then found := true
  done;
  !found

(* Get the value after a flag in argv *)
let get_flag_value flag =
  let value = ref "" in
  for i = 1 to Array.length Sys.argv - 2 do
    if Sys.argv.(i) = flag then value := Sys.argv.(i + 1)
  done;
  !value

(* Read a file, return contents or empty string if missing *)
let read_file filename =
  if Sys.file_exists filename then begin
    let ic = open_in filename in
    let n = in_channel_length ic in
    let s = really_input_string ic n in
    close_in ic;
    s
  end else ""

(* Append text to a file *)
let append_file filename text =
  let oc = open_out_gen [Open_append; Open_creat] 0o644 filename in
  output_string oc text;
  close_out oc

(* Count <|user|> markers in a string *)
let count_turns s =
  let n = ref 0 in
  let marker = "<|user|>" in
  let mlen = String.length marker in
  let slen = String.length s in
  let i = ref 0 in
  while !i <= slen - mlen do
    if String.sub s !i mlen = marker then begin incr n; i := !i + mlen end
    else incr i
  done;
  !n

(* State file for CLI --prompt / --teach pair *)
type cli_state = {
  prompt_text : string;
  history : string;
  responses : string array;
}

let save_state filename (state : cli_state) =
  let oc = open_out_bin filename in
  Marshal.to_channel oc state [];
  close_out oc

let load_state filename : cli_state =
  let ic = open_in_bin filename in
  let state : cli_state = Marshal.from_channel ic in
  close_in ic;
  state

(* Train on a token sequence: loss → backward → clip → adam step *)
let train_step model params adam step tokens =
  let n = Array.length tokens in
  let tokens =
    if n <= Vidya.Model.block_size + 1 then tokens
    else Array.sub tokens (n - Vidya.Model.block_size - 1)
           (Vidya.Model.block_size + 1) in
  let (loss, _) = Vidya.Train.compute_loss model tokens in
  Vidya.Tensor.backward loss;
  Vidya.Train.clip_grad_norm params;
  Vidya.Train.adam_step_fixed params adam step 1e-5;
  loss.Vidya.Tensor.data.(0)

let () =
  let t_start = Sys.time () in
  let checkpoint_file = "../../microgpt_chat_10m_v4.bin" in
  let train_checkpoint = "../../microgpt_chat_10m_v4_train.bin" in
  let input_file = "../../chat_input_v4.txt" in
  let tokenizer_file = "../../tokenizer.bin" in
  let context_file = "context.md" in
  let state_file = "train_state.bin" in

  (* Load BPE tokenizer (fast) or train from corpus and save *)
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

  if has_flag "--save-bpe" then
    exit 0;

  (* Init model *)
  let model = Vidya.Model.init tok.vocab_size in
  let params = Vidya.Model.collect_params model in
  let total_params =
    Array.fold_left (fun acc p ->
      acc + Array.length p.Vidya.Tensor.data) 0 params in
  Printf.printf "num params: %d\n" total_params;

  (* Load checkpoint or train from scratch *)
  if has_flag "--load" then begin
    Random.self_init ();
    let file =
      if Sys.file_exists train_checkpoint then train_checkpoint
      else checkpoint_file in
    Vidya.Train.load_checkpoint file params
  end else begin
    let docs = Vidya.Utils.load_docs input_file in
    Printf.printf "num docs: %d\n" (Array.length docs);
    let tokenized_docs = Vidya.Train.pre_tokenize docs tok in
    Vidya.Train.train model params tokenized_docs 300000
      ~checkpoint_base:"../../microgpt_chat_10m_v4" ();
    Vidya.Train.save_checkpoint checkpoint_file params
  end;

  (* === Interactive training mode === *)
  if has_flag "--train" then begin
    let context = read_file context_file in
    let history = Buffer.create 4096 in
    Buffer.add_string history context;
    Printf.printf "--- interactive training mode (type 'quit' to exit) ---\n%!";
    Printf.printf "(loaded context: %d turns)\n%!" (count_turns context);
    let adam = Vidya.Train.init_adam params in
    let step = ref 0 in
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
          let hist_str = Buffer.contents history in
          (* Generate 5 responses *)
          let responses = Array.init 5 (fun _ ->
            let (_, gen) =
              Vidya.Generate.chat_rollout model tok hist_str 0.7 in
            Vidya.Bpe.decode tok gen
          ) in
          Array.iteri (fun i text ->
            Printf.printf "%d: %s\n" (i + 1) text
          ) responses;
          Printf.printf "[1-5 or type response] > %!";
          match input_line stdin with
          | exception End_of_file -> running := false
          | sel_input ->
            let sel = String.trim sel_input in
            let response_text =
              if String.length sel = 1
                 && sel.[0] >= '1' && sel.[0] <= '5' then
                responses.(Char.code sel.[0] - Char.code '1')
              else sel
            in
            Buffer.add_string history
              (Printf.sprintf " %s " response_text);
            let tokens =
              Vidya.Bpe.encode tok (Buffer.contents history) in
            let loss = train_step model params adam !step tokens in
            incr step;
            Printf.printf "trained. (loss %.2f, step %d)\n%!" loss !step;
            append_file context_file
              (Printf.sprintf "<|user|> %s <|assistant|> %s\n"
                trimmed response_text)
        end
    done;
    Vidya.Train.save_checkpoint train_checkpoint params

  (* === CLI teach mode === *)
  end else if has_flag "--teach" then begin
    let teach_arg = get_flag_value "--teach" in
    if teach_arg = "" then begin
      Printf.printf "error: --teach requires a value (1-5 or response text)\n%!";
      exit 1
    end;
    if not (Sys.file_exists state_file) then begin
      Printf.printf "error: no state file. run --prompt first.\n%!";
      exit 1
    end;
    let state = load_state state_file in
    let response_text =
      if String.length teach_arg = 1
         && teach_arg.[0] >= '1' && teach_arg.[0] <= '5' then
        state.responses.(Char.code teach_arg.[0] - Char.code '1')
      else teach_arg
    in
    let train_text = state.history ^ " " ^ response_text ^ " " in
    let tokens = Vidya.Bpe.encode tok train_text in
    let adam = Vidya.Train.init_adam params in
    let loss = train_step model params adam 0 tokens in
    Printf.printf "trained. (loss %.2f)\n%!" loss;
    Vidya.Train.save_checkpoint train_checkpoint params;
    append_file context_file
      (Printf.sprintf "<|user|> %s <|assistant|> %s\n"
        state.prompt_text response_text)

  (* === CLI prompt mode === *)
  end else if get_flag_value "--prompt" <> "" then begin
    let prompt_text = get_flag_value "--prompt" in
    let context = read_file context_file in
    let history = context ^
      Printf.sprintf "<|user|> %s <|assistant|>" prompt_text in
    let responses = Array.init 5 (fun _ ->
      let (_, gen) =
        Vidya.Generate.chat_rollout model tok history 0.7 in
      Vidya.Bpe.decode tok gen
    ) in
    Array.iteri (fun i text ->
      Printf.printf "%d: %s\n" (i + 1) text
    ) responses;
    save_state state_file { prompt_text; history; responses }

  (* === Modes that need corpus + symbolic + concept knowledge === *)
  end else begin
    let docs = Vidya.Utils.load_docs input_file in
    let special_ids = [tok.bos_id; tok.user_id; tok.assistant_id] in
    let know = Vidya.Symbolic.build tok.vocab docs tok.bos_id
      ~special_ids () in
    let concept_know = Vidya.Knowledge.build tok.vocab docs tok.bos_id in
    Vidya.Knowledge.load_weights "td_weights.bin" concept_know;

    if has_flag "--rl" then begin
      let mode =
        if has_flag "--reinforce" then `Reinforce else `ExIt in
      let num_steps =
        let s = get_flag_value "--rl-steps" in
        if s <> "" then int_of_string s else 1000 in
      Vidya.Train.rl_train model params tok docs ~num_steps ~mode ();
      let rl_file =
        (Filename.chop_extension checkpoint_file) ^ "_rl.bin" in
      Vidya.Train.save_checkpoint rl_file params
    end else if has_flag "--chat" then begin
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
            let response = Vidya.Generate.chat model know ~concept_know
              tok (Buffer.contents history) 0.5 in
            Printf.printf "%s\n%!" response;
            Buffer.add_string history (Printf.sprintf " %s " response)
          end
      done
    end else begin
      Printf.printf "--- inference (chat test) ---\n";
      let test_prompts = [
        "Hello"; "How are you?"; "What is the meaning of life?";
        "Tell me a story"; "What do you think about music?";
      ] in
      List.iter (fun temp ->
        Printf.printf "--- temperature %.1f ---\n" temp;
        List.iter (fun p ->
          Printf.printf "user: %s\n" p;
          let history = Printf.sprintf "<|user|> %s <|assistant|>" p in
          let response = Vidya.Generate.chat model know ~concept_know
            tok history temp in
          Printf.printf "  → %s\n\n" response
        ) test_prompts
      ) [0.3; 0.5; 0.8]
    end;
    Vidya.Knowledge.save_weights "td_weights.bin" concept_know
  end;
  Printf.printf "total time: %.2fs\n" (Sys.time () -. t_start)
