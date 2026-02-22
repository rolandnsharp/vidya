/* llama_stubs.c — OCaml FFI bindings for llama.cpp
   =================================================

   Minimal bindings: load model, generate text, get logits.
   TODO: implement after llama.cpp is built. */

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/fail.h>

CAMLprim value caml_llama_stub_placeholder(value unit) {
  CAMLparam1(unit);
  CAMLreturn(Val_unit);
}
