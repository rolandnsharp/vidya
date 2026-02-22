(* main.ml — CLI entry point for symbolic-eval
   =============================================

   Usage:
     dune exec bin/main.exe -- --model model.gguf --corpus corpus.txt
     dune exec bin/main.exe -- --model model.gguf --corpus corpus.txt --baseline

   Loads a pre-trained model via llama.cpp, builds concept knowledge from
   the corpus, and generates responses with/without symbolic constraints. *)

let () =
  Printf.printf "symbolic-eval: not yet implemented\n";
  Printf.printf "next step: build llama.cpp and write FFI bindings\n"
