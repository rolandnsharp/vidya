/* blas_stubs.c — Minimal C glue between OCaml and OpenBLAS.
 *
 * This file provides exactly ONE function: a wrapper around cblas_dgemm
 * (double-precision general matrix multiply). Everything else in the
 * project stays pure OCaml.
 *
 * WHY WE NEED THIS
 * =================
 * OCaml's ocamlopt compiler generates good scalar code but cannot emit
 * SIMD instructions (AVX2/AVX-512). For matrix multiplication — the
 * dominant cost in transformer training — OpenBLAS uses hand-tuned
 * assembly kernels that exploit SIMD, cache blocking, and micro-
 * architecture-specific optimizations. At 64x64 matrices this gives
 * ~5-10x speedup; at 256+ dimensions, ~20-50x.
 *
 * HOW OCAML C FFI WORKS
 * =====================
 * OCaml float arrays are stored as flat, contiguous, unboxed doubles
 * in the heap. The value pointer points directly to the first double
 * (after the block header). So (double *)Op_val(v) gives us a raw
 * double* that BLAS can use directly — no copying needed.
 *
 * The GC won't move these arrays during our cblas_dgemm call because
 * the GC only runs when OCaml code allocates, and cblas_dgemm doesn't
 * allocate on the OCaml heap.
 *
 * OPERATIONS
 * ==========
 * We encode three matrix multiply variants in a single function via
 * an 'op' flag, keeping the OCaml external interface simple:
 *
 *   op=0 (forward):     C[m,n]  = A[m,k] @ B[k,n]     beta=0 (overwrite)
 *   op=1 (backward da): C[m,k] += A[m,n] @ B[k,n]^T   beta=1 (accumulate)
 *   op=2 (backward db): C[k,n] += A[m,k]^T @ B[m,n]   beta=1 (accumulate)
 *
 * The m,n,k dimensions always match the original forward pass, so the
 * caller doesn't need to think about transposition.
 *
 * Compile: ocamlopt -O2 -o microgpt_blas blas_stubs.c 6_microgpt_blas.ml \
 *          -ccopt "-I/usr/include/x86_64-linux-gnu" -cclib -lopenblas
 */

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <cblas.h>

CAMLprim value caml_dgemm(value vop, value vm, value vn, value vk,
                          value va, value vb, value vc) {
    int op = Int_val(vop);
    int m  = Int_val(vm);
    int n  = Int_val(vn);
    int k  = Int_val(vk);

    /* OCaml float array -> raw double pointer.
     * Op_val(v) returns (value*)(v), which for float arrays points
     * to the first double in the contiguous unboxed storage. */
    double *a = (double *)Op_val(va);
    double *b = (double *)Op_val(vb);
    double *c = (double *)Op_val(vc);

    switch (op) {
    case 0:
        /* Forward: C[m,n] = A[m,k] @ B[k,n]
         * Row-major: lda=k, ldb=n, ldc=n */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0, a, k, b, n, 0.0, c, n);
        break;

    case 1:
        /* Backward da: C[m,k] += A[m,n] @ B[k,n]^T
         * A=dC[m,n], B=original_B[k,n], C=da[m,k]
         * BLAS: dgemm(N, T, m, k, n, 1.0, dC, n, B, n, 1.0, da, k) */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m, k, n, 1.0, a, n, b, n, 1.0, c, k);
        break;

    case 2:
        /* Backward db: C[k,n] += A[m,k]^T @ B[m,n]
         * A=original_A[m,k], B=dC[m,n], C=db[k,n]
         * BLAS: dgemm(T, N, k, n, m, 1.0, A, k, dC, n, 1.0, db, n) */
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    k, n, m, 1.0, a, k, b, n, 1.0, c, n);
        break;
    }

    return Val_unit;
}

/* Bytecode wrapper: OCaml bytecode can only pass ≤5 args directly to C.
 * For 6+ args, it passes an argv array instead. We only compile with
 * ocamlopt (native), but include this for completeness / portability. */
CAMLprim value caml_dgemm_byte(value *argv, int argn) {
    (void)argn;
    return caml_dgemm(argv[0], argv[1], argv[2], argv[3],
                      argv[4], argv[5], argv[6]);
}
