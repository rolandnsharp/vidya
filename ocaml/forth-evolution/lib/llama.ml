(* llama.ml — OCaml interface to llama.cpp
   ========================================

   TODO: implement once llama.cpp FFI stubs are written.

   For the evolution loop we primarily need text generation
   (prompt in, text out). Raw logit access is optional —
   the model proposes definitions as text, Forth validates them. *)

type model = unit
type context = unit

let load_model _path = failwith "TODO: implement llama.cpp bindings"
let create_context _model = failwith "TODO: implement llama.cpp bindings"
let free_context _ctx = ()
let free_model _model = ()

let generate _ctx _prompt _temperature =
  failwith "TODO: implement llama.cpp bindings"
