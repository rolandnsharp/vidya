(* llama.ml — OCaml interface to llama.cpp
   ========================================

   Wraps the C stubs into a clean OCaml API.
   TODO: implement once llama.cpp FFI stubs are written. *)

(* Placeholder types *)
type model = unit
type context = unit

let load_model _path = failwith "TODO: implement llama.cpp bindings"
let create_context _model = failwith "TODO: implement llama.cpp bindings"
let free_context _ctx = ()
let free_model _model = ()

let tokenize _model _text = failwith "TODO: implement llama.cpp bindings"
let token_to_string _model _token_id = failwith "TODO: implement llama.cpp bindings"
let n_vocab _model = failwith "TODO: implement llama.cpp bindings"

let eval _ctx _tokens = failwith "TODO: implement llama.cpp bindings"
let get_logits _ctx = failwith "TODO: implement llama.cpp bindings"
