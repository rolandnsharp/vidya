(* main.ml — CLI entry point for vidya
   =====================================

   Usage:
     dune exec bin/main.exe              → train from scratch + save + generate
     dune exec bin/main.exe -- --load    → load checkpoint + generate
     dune exec bin/main.exe -- --load --prompt "The Soul is"
                                         → load + prompted generation

   BPE training always runs first (fast, ~3s, deterministic from corpus).
   The checkpoint format is unchanged from the monolithic stage files. *)

let () = Random.init 42

let () =
  let t_start = Sys.time () in
  let checkpoint_file = "../microgpt_tuned.bin" in
  let input_file = "../input.txt" in

  (* BPE is always needed (fast, ~3s) — deterministic from corpus *)
  let docs = Vidya.Utils.load_docs input_file in
  Printf.printf "num docs: %d\n" (Array.length docs);
  let tok = Vidya.Bpe.train docs Vidya.Bpe.n_merges in

  (* Init model, then either load checkpoint or train from scratch *)
  let model = Vidya.Model.init tok.vocab_size in
  let params = Vidya.Model.collect_params model in
  let total_params =
    Array.fold_left (fun acc p -> acc + Array.length p.Vidya.Tensor.data) 0 params in
  Printf.printf "num params: %d\n" total_params;

  if Array.length Sys.argv > 1 && Sys.argv.(1) = "--load" then begin
    (* Load saved weights — skip training entirely.
       Re-seed RNG from clock so each run gives different samples
       (training mode uses seed 42 for reproducibility). *)
    Random.self_init ();
    Vidya.Train.load_checkpoint checkpoint_file params;
    Printf.printf "skipping training (loaded from %s)\n%!" checkpoint_file
  end else begin
    (* Train from scratch and save *)
    let tokenized_docs = Vidya.Train.pre_tokenize docs tok in
    Vidya.Train.train model params tokenized_docs 100000;
    Vidya.Train.save_checkpoint checkpoint_file params
  end;

  (* Build symbolic knowledge once — shared across all generations *)
  let know = Vidya.Symbolic.build tok.vocab docs tok.bos_id in

  (* Check for --prompt "text" in argv *)
  let prompt_text = ref "" in
  for i = 1 to Array.length Sys.argv - 2 do
    if Sys.argv.(i) = "--prompt" then prompt_text := Sys.argv.(i + 1)
  done;

  if !prompt_text <> "" then begin
    Printf.printf "--- prompted generation ---\n";
    Printf.printf "prompt: %s\n" !prompt_text;
    for i = 1 to 10 do
      Vidya.Generate.prompted model know tok tok.bos_id !prompt_text 0.5
      |> Printf.printf "  %2d: %s\n" i
    done
  end else begin
    Printf.printf "--- inference (new, hallucinated text) ---\n";
    for i = 1 to 20 do
      Vidya.Generate.sample model know tok.bos_id 0.5
      |> Printf.printf "sample %2d: %s\n" i
    done
  end;
  Printf.printf "total time: %.2fs\n" (Sys.time () -. t_start)
