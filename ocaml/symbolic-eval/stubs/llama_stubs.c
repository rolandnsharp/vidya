/* llama_stubs.c — OCaml FFI bindings for llama.cpp
   =================================================

   Minimal bindings: load model, feed tokens, get logits.
   Same pattern as vidya's BLAS stubs. */

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/fail.h>

/* TODO: #include "llama.h" once llama.cpp is built */

/* Placeholder — implement after llama.cpp is installed */
CAMLprim value caml_llama_stub_placeholder(value unit) {
  CAMLparam1(unit);
  CAMLreturn(Val_unit);
}
