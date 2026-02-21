(* main.ml — CLI entry point for vidya
   =====================================

   Usage:
     dune exec bin/main.exe              → train from scratch + save + generate
     dune exec bin/main.exe -- --load    → load checkpoint + generate
     dune exec bin/main.exe -- --load --prompt "The Soul is"
                                         → load + prompted generation
     dune exec bin/main.exe -- --load --chat
                                         → load + interactive chat mode

   BPE training always runs first (fast, ~3s, deterministic from corpus).
   Symbolic + concept knowledge built once, shared across all generations.
   The checkpoint format is unchanged from the monolithic stage files. *)

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

let () =
  let t_start = Sys.time () in
  let checkpoint_file = "../../microgpt_chat_10m_v3.bin" in
  let input_file = "../../chat_input.txt" in

  (* BPE is always needed (fast, ~3s) — deterministic from corpus *)
  let docs = Vidya.Utils.load_docs input_file in
  Printf.printf "num docs: %d\n" (Array.length docs);
  let tok = Vidya.Bpe.train docs Vidya.Bpe.n_merges in
  let special_ids = [tok.bos_id; tok.user_id; tok.assistant_id] in

  (* Init model, then either load checkpoint or train from scratch *)
  let model = Vidya.Model.init tok.vocab_size in
  let params = Vidya.Model.collect_params model in
  let total_params =
    Array.fold_left (fun acc p -> acc + Array.length p.Vidya.Tensor.data) 0 params in
  Printf.printf "num params: %d\n" total_params;

  if has_flag "--load" then begin
    Random.self_init ();
    Vidya.Train.load_checkpoint checkpoint_file params;
    Printf.printf "skipping training (loaded from %s)\n%!" checkpoint_file
  end else begin
    let tokenized_docs = Vidya.Train.pre_tokenize docs tok in
    Vidya.Train.train model params tokenized_docs 200000;
    Vidya.Train.save_checkpoint checkpoint_file params
  end;

  (* Build symbolic knowledge once — shared across all generations *)
  let know = Vidya.Symbolic.build tok.vocab docs tok.bos_id
    ~special_ids () in

  (* Build concept knowledge — co-occurrence associations *)
  let concept_know = Vidya.Knowledge.build tok.vocab docs tok.bos_id in
  Vidya.Knowledge.load_weights "td_weights.bin" concept_know;

  let prompt_text = get_flag_value "--prompt" in

  if has_flag "--chat" then begin
    (* Interactive chat mode *)
    Printf.printf "--- chat mode (type 'quit' to exit) ---\n%!";
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
          let response = Vidya.Generate.chat model know ~concept_know
            tok trimmed 0.5 in
          Printf.printf "%s\n%!" response
        end
    done
  end else if prompt_text <> "" then begin
    Printf.printf "--- prompted generation ---\n";
    Printf.printf "prompt: %s\n" prompt_text;
    for i = 1 to 10 do
      Vidya.Generate.prompted model know ~concept_know
        tok tok.bos_id prompt_text 0.5
      |> Printf.printf "  %2d: %s\n" i
    done
  end else begin
    Printf.printf "--- inference (new, hallucinated text) ---\n";
    for i = 1 to 20 do
      Vidya.Generate.sample model know ~concept_know
        tok.bos_id 0.5
      |> Printf.printf "sample %2d: %s\n" i
    done
  end;
  Vidya.Knowledge.save_weights "td_weights.bin" concept_know;
  Printf.printf "total time: %.2fs\n" (Sys.time () -. t_start)
