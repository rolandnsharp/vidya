(* blas.ml — BLAS FFI declaration
   ================================

   Single external binding to the C stub in vidya_stubs.
   The stub wraps cblas_dgemm with a 3-bit op flag:

     bit 2 (4): transpose A
     bit 1 (2): transpose B
     bit 0 (1): accumulate (beta=1) vs overwrite (beta=0)

   op=0: C = A @ B         NN overwrite
   op=1: C += A @ B        NN accumulate
   op=2: C = A @ B^T       NT overwrite
   op=3: C += A @ B^T      NT accumulate
   op=4: C = A^T @ B       TN overwrite
   op=5: C += A^T @ B      TN accumulate

   m,n are result dims; k is the contracted dimension. *)

external dgemm : int -> int -> int -> int
  -> float array -> float array -> float array -> unit
  = "caml_dgemm_byte" "caml_dgemm"
