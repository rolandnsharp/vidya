/* blas_stubs.c — OCaml ↔ OpenBLAS bridge (Stage 7+)
 *
 * Single C function wrapping cblas_dgemm with 6 operation modes,
 * encoded as a 3-bit flag:
 *
 *   bit 2 (4): transpose A
 *   bit 1 (2): transpose B
 *   bit 0 (1): accumulate (beta=1) vs overwrite (beta=0)
 *
 *   op=0: C[m,n]  = A[m,k]  @ B[k,n]      NN overwrite
 *   op=1: C[m,n] += A[m,k]  @ B[k,n]      NN accumulate
 *   op=2: C[m,n]  = A[m,k]  @ B^T[k,n]    NT overwrite  (B stored [n,k])
 *   op=3: C[m,n] += A[m,k]  @ B^T[k,n]    NT accumulate
 *   op=4: C[m,n]  = A^T[m,k] @ B[k,n]     TN overwrite  (A stored [k,m])
 *   op=5: C[m,n] += A^T[m,k] @ B[k,n]     TN accumulate
 *
 * m,n are always the result dimensions. k is the contracted dimension.
 *
 * Compile with: -ccopt "-I/usr/include/x86_64-linux-gnu" -cclib -lopenblas
 */

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <cblas.h>

/* Limit OpenBLAS to 1 thread — at 64×64 matrices, thread overhead
 * exceeds the benefit of parallelism. Single-threaded is faster. */
extern void openblas_set_num_threads(int);
__attribute__((constructor))
static void init_blas(void) { openblas_set_num_threads(1); }

CAMLprim value caml_dgemm(value vop, value vm, value vn, value vk,
                          value va, value vb, value vc) {
    int op = Int_val(vop);
    int m  = Int_val(vm);
    int n  = Int_val(vn);
    int k  = Int_val(vk);
    double *a = (double *)Op_val(va);
    double *b = (double *)Op_val(vb);
    double *c = (double *)Op_val(vc);

    CBLAS_TRANSPOSE ta = (op & 4) ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tb = (op & 2) ? CblasTrans : CblasNoTrans;
    double beta        = (op & 1) ? 1.0 : 0.0;

    /* Row-major leading dimensions:
     *   NoTrans: matrix stored [rows, cols], ld = cols
     *   Trans:   matrix stored [cols, rows], ld = rows
     * A logical [m,k]: stored [m,k] if N (ld=k), [k,m] if T (ld=m)
     * B logical [k,n]: stored [k,n] if N (ld=n), [n,k] if T (ld=k) */
    int lda = (op & 4) ? m : k;
    int ldb = (op & 2) ? k : n;

    cblas_dgemm(CblasRowMajor, ta, tb, m, n, k,
                1.0, a, lda, b, ldb, beta, c, n);
    return Val_unit;
}

CAMLprim value caml_dgemm_byte(value *argv, int argn) {
    (void)argn;
    return caml_dgemm(argv[0], argv[1], argv[2], argv[3],
                      argv[4], argv[5], argv[6]);
}
