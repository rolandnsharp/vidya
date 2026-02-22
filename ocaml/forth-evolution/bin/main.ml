(* main.ml — CLI entry point for forth-evolution
   ===============================================

   Usage:
     dune exec bin/main.exe -- --model path/to/model.gguf

   Runs the evolution loop: model proposes Forth definitions,
   validator accepts or rejects, dictionary grows. *)

let () =
  Printf.printf "forth-evolution: not yet implemented\n";
  Printf.printf "next step: build llama.cpp and write FFI bindings\n";
  (* Quick test: create a dictionary and validate a definition *)
  let dict = Forth_evolution.Forth.create () in
  let body = ["DUP"; "*"] in
  match Forth_evolution.Forth.validate dict body with
  | Ok effect ->
    Printf.printf "SQUARE (DUP *) → valid: %d in, %d out\n"
      effect.consumed effect.produced
  | Error msg ->
    Printf.printf "SQUARE (DUP *) → invalid: %s\n" msg
