/* gpu_bridge.c — OCaml FFI bridge to CUDA kernels
 * ================================================
 *
 * This file is compiled as plain C (not C++) because OCaml's FFI
 * requires CAMLprim functions with C linkage. It wraps the CUDA
 * functions declared in gpu_stubs.h, marshalling between OCaml
 * values and C types.
 *
 * GPU tensors are represented as OCaml custom blocks containing
 * a gpu_tensor struct (device pointer + element count). The GC
 * finalizer calls cudaFree automatically. */

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <string.h>
#include <stdlib.h>

#include "gpu_stubs.h"

/* Upload int array to device and return device pointer.
 * Caller must call gpu_free_int_buf to free. */
void gpu_upload_int_buf(const int *host, int **d_out, int n);
void gpu_free_int_buf(int *d_ptr);

/* ── Custom block for gpu_tensor ─────────────────────────────────── */

#define Gpu_tensor_val(v) ((gpu_tensor *)Data_custom_val(v))

static void gpu_tensor_finalize(value v) {
    gpu_tensor *t = Gpu_tensor_val(v);
    gpu_free(t);
}

static struct custom_operations gpu_tensor_ops = {
    .identifier  = "vidya.gpu_tensor",
    .finalize    = gpu_tensor_finalize,
    .compare     = custom_compare_default,
    .hash        = custom_hash_default,
    .serialize   = custom_serialize_default,
    .deserialize = custom_deserialize_default,
    .compare_ext = custom_compare_ext_default,
    .fixed_length = NULL,
};

/* Helper: allocate an OCaml custom block holding a gpu_tensor. */
static value alloc_gpu_tensor(gpu_tensor t) {
    value v = caml_alloc_custom(&gpu_tensor_ops, sizeof(gpu_tensor), 0, 1);
    *Gpu_tensor_val(v) = t;
    return v;
}

/* ── Init ────────────────────────────────────────────────────────── */

CAMLprim value caml_gpu_init(value unit) {
    CAMLparam1(unit);
    gpu_init();
    CAMLreturn(Val_unit);
}

/* ── Memory management ───────────────────────────────────────────── */

CAMLprim value caml_gpu_create(value vn) {
    CAMLparam1(vn);
    int n = Int_val(vn);
    gpu_tensor t = gpu_create(n);
    CAMLreturn(alloc_gpu_tensor(t));
}

CAMLprim value caml_gpu_upload(value varr, value vgpu) {
    CAMLparam2(varr, vgpu);
    gpu_tensor *t = Gpu_tensor_val(vgpu);
    /* OCaml float array: flat array of boxed doubles. */
    double *host = (double *)Op_val(varr);
    gpu_upload(host, t);
    CAMLreturn(Val_unit);
}

CAMLprim value caml_gpu_download(value vgpu) {
    CAMLparam1(vgpu);
    gpu_tensor *t = Gpu_tensor_val(vgpu);
    int n = t->numel;
    /* Allocate a fresh OCaml float array. */
    value arr = caml_alloc_float_array(n);
    double *host = (double *)Op_val(arr);
    gpu_download(t, host);
    CAMLreturn(arr);
}

CAMLprim value caml_gpu_numel(value vgpu) {
    CAMLparam1(vgpu);
    gpu_tensor *t = Gpu_tensor_val(vgpu);
    CAMLreturn(Val_int(t->numel));
}

CAMLprim value caml_gpu_zero(value vgpu) {
    CAMLparam1(vgpu);
    gpu_tensor *t = Gpu_tensor_val(vgpu);
    gpu_zero(t);
    CAMLreturn(Val_unit);
}

CAMLprim value caml_gpu_copy(value vsrc, value vdst, value vn) {
    CAMLparam3(vsrc, vdst, vn);
    gpu_tensor *src = Gpu_tensor_val(vsrc);
    gpu_tensor *dst = Gpu_tensor_val(vdst);
    gpu_copy(src, dst, Int_val(vn));
    CAMLreturn(Val_unit);
}

/* ── GEMM ────────────────────────────────────────────────────────── */

CAMLprim value caml_gpu_sgemm(value vop, value vm, value vn, value vk,
                               value va, value vb, value vc) {
    int op = Int_val(vop);
    int m  = Int_val(vm);
    int n  = Int_val(vn);
    int k  = Int_val(vk);
    gpu_tensor *a = Gpu_tensor_val(va);
    gpu_tensor *b = Gpu_tensor_val(vb);
    gpu_tensor *c = Gpu_tensor_val(vc);
    gpu_sgemm(op, m, n, k, a->d_data, b->d_data, c->d_data);
    return Val_unit;
}

CAMLprim value caml_gpu_sgemm_byte(value *argv, int argn) {
    (void)argn;
    return caml_gpu_sgemm(argv[0], argv[1], argv[2], argv[3],
                           argv[4], argv[5], argv[6]);
}

/* ── Element-wise ops ────────────────────────────────────────────── */

CAMLprim value caml_gpu_gelu_forward(value vx, value vy, value vn) {
    CAMLparam3(vx, vy, vn);
    gpu_gelu_forward(Gpu_tensor_val(vx)->d_data,
                     Gpu_tensor_val(vy)->d_data, Int_val(vn));
    CAMLreturn(Val_unit);
}

CAMLprim value caml_gpu_gelu_backward(value vx, value vdy,
                                       value vdx, value vn) {
    CAMLparam4(vx, vdy, vdx, vn);
    gpu_gelu_backward(Gpu_tensor_val(vx)->d_data,
                      Gpu_tensor_val(vdy)->d_data,
                      Gpu_tensor_val(vdx)->d_data, Int_val(vn));
    CAMLreturn(Val_unit);
}

CAMLprim value caml_gpu_add_forward(value va, value vb,
                                     value vy, value vn) {
    CAMLparam4(va, vb, vy, vn);
    gpu_add_forward(Gpu_tensor_val(va)->d_data,
                    Gpu_tensor_val(vb)->d_data,
                    Gpu_tensor_val(vy)->d_data, Int_val(vn));
    CAMLreturn(Val_unit);
}

CAMLprim value caml_gpu_scale_forward(value vx, value vs,
                                       value vy, value vn) {
    CAMLparam4(vx, vs, vy, vn);
    gpu_scale_forward(Gpu_tensor_val(vx)->d_data, Double_val(vs),
                      Gpu_tensor_val(vy)->d_data, Int_val(vn));
    CAMLreturn(Val_unit);
}

CAMLprim value caml_gpu_dropout_forward(value vx, value vy, value vmask,
                                         value vrate, value vn) {
    CAMLparam5(vx, vy, vmask, vrate, vn);
    gpu_dropout_forward(Gpu_tensor_val(vx)->d_data,
                        Gpu_tensor_val(vy)->d_data,
                        Gpu_tensor_val(vmask)->d_data,
                        Double_val(vrate), Int_val(vn));
    CAMLreturn(Val_unit);
}

CAMLprim value caml_gpu_dropout_backward(value vdy, value vmask,
                                          value vdx, value vn) {
    CAMLparam4(vdy, vmask, vdx, vn);
    gpu_dropout_backward(Gpu_tensor_val(vdy)->d_data,
                         Gpu_tensor_val(vmask)->d_data,
                         Gpu_tensor_val(vdx)->d_data, Int_val(vn));
    CAMLreturn(Val_unit);
}

/* ── Softmax ─────────────────────────────────────────────────────── */

CAMLprim value caml_gpu_softmax_forward(value vx, value vy, value vn) {
    CAMLparam3(vx, vy, vn);
    gpu_softmax_forward(Gpu_tensor_val(vx)->d_data,
                        Gpu_tensor_val(vy)->d_data, Int_val(vn));
    CAMLreturn(Val_unit);
}

CAMLprim value caml_gpu_softmax_backward(value vy, value vdy,
                                          value vdx, value vn) {
    CAMLparam4(vy, vdy, vdx, vn);
    gpu_softmax_backward(Gpu_tensor_val(vy)->d_data,
                         Gpu_tensor_val(vdy)->d_data,
                         Gpu_tensor_val(vdx)->d_data, Int_val(vn));
    CAMLreturn(Val_unit);
}

/* ── RMSNorm ─────────────────────────────────────────────────────── */

CAMLprim value caml_gpu_rmsnorm_forward(value vx, value vy, value vrms,
                                         value vrows, value vdim) {
    CAMLparam5(vx, vy, vrms, vrows, vdim);
    gpu_rmsnorm_forward(Gpu_tensor_val(vx)->d_data,
                        Gpu_tensor_val(vy)->d_data,
                        Gpu_tensor_val(vrms)->d_data,
                        Int_val(vrows), Int_val(vdim));
    CAMLreturn(Val_unit);
}

/* Bytecode wrappers for functions with >5 args. */

CAMLprim value caml_gpu_rmsnorm_affine_forward(value vx, value vgamma,
                                                value vy, value vrms,
                                                value vrows, value vdim) {
    gpu_rmsnorm_affine_forward(Gpu_tensor_val(vx)->d_data,
                               Gpu_tensor_val(vgamma)->d_data,
                               Gpu_tensor_val(vy)->d_data,
                               Gpu_tensor_val(vrms)->d_data,
                               Int_val(vrows), Int_val(vdim));
    return Val_unit;
}

CAMLprim value caml_gpu_rmsnorm_affine_forward_byte(value *argv, int argn) {
    (void)argn;
    return caml_gpu_rmsnorm_affine_forward(
        argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

CAMLprim value caml_gpu_rmsnorm_backward(value vx, value vdy, value vrms,
                                          value vdx, value vrows,
                                          value vdim) {
    gpu_rmsnorm_backward(Gpu_tensor_val(vx)->d_data,
                         Gpu_tensor_val(vdy)->d_data,
                         Gpu_tensor_val(vrms)->d_data,
                         Gpu_tensor_val(vdx)->d_data,
                         Int_val(vrows), Int_val(vdim));
    return Val_unit;
}

CAMLprim value caml_gpu_rmsnorm_backward_byte(value *argv, int argn) {
    (void)argn;
    return caml_gpu_rmsnorm_backward(
        argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

CAMLprim value caml_gpu_rmsnorm_affine_backward(value vx, value vgamma,
    value vdy, value vnorm, value vrms, value vdx, value vdgamma,
    value vrows, value vdim) {
    /* Stub — TODO Step 4 */
    (void)vx; (void)vgamma; (void)vdy; (void)vnorm; (void)vrms;
    (void)vdx; (void)vdgamma; (void)vrows; (void)vdim;
    return Val_unit;
}

CAMLprim value caml_gpu_rmsnorm_affine_backward_byte(value *argv, int argn) {
    (void)argn;
    return caml_gpu_rmsnorm_affine_backward(
        argv[0], argv[1], argv[2], argv[3], argv[4],
        argv[5], argv[6], argv[7], argv[8]);
}

/* ── Embedding ───────────────────────────────────────────────────── */

CAMLprim value caml_gpu_embed_forward(value vwte, value vtokens,
                                       value vout, value vseq, value vdim) {
    CAMLparam5(vwte, vtokens, vout, vseq, vdim);
    int seq_len = Int_val(vseq);
    int dim = Int_val(vdim);

    /* Copy token IDs from OCaml int array to host buffer. */
    int *h_tokens = (int *)malloc(seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++)
        h_tokens[i] = Int_val(Field(vtokens, i));

    /* Upload to device via CUDA helper (in gpu_stubs.cu). */
    int *d_tokens;
    gpu_upload_int_buf(h_tokens, &d_tokens, seq_len);

    gpu_embed_forward(Gpu_tensor_val(vwte)->d_data, d_tokens,
                      Gpu_tensor_val(vout)->d_data, seq_len, dim);

    gpu_free_int_buf(d_tokens);
    free(h_tokens);
    CAMLreturn(Val_unit);
}

CAMLprim value caml_gpu_embed_backward(value vwte_grad, value vtokens,
                                        value vdout, value vseq, value vdim) {
    CAMLparam5(vwte_grad, vtokens, vdout, vseq, vdim);
    int seq_len = Int_val(vseq);
    int dim = Int_val(vdim);

    int *h_tokens = (int *)malloc(seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++)
        h_tokens[i] = Int_val(Field(vtokens, i));

    int *d_tokens;
    gpu_upload_int_buf(h_tokens, &d_tokens, seq_len);

    gpu_embed_backward(Gpu_tensor_val(vwte_grad)->d_data, d_tokens,
                       Gpu_tensor_val(vdout)->d_data, seq_len, dim);

    gpu_free_int_buf(d_tokens);
    free(h_tokens);
    CAMLreturn(Val_unit);
}

/* ── Loss ────────────────────────────────────────────────────────── */

CAMLprim value caml_gpu_nll_forward(value vprobs, value vtarget) {
    CAMLparam2(vprobs, vtarget);
    float loss = gpu_nll_forward(Gpu_tensor_val(vprobs)->d_data,
                                 Int_val(vtarget));
    CAMLreturn(caml_copy_double((double)loss));
}

CAMLprim value caml_gpu_nll_backward(value vprobs, value vgrad,
                                      value vtarget, value vupstream) {
    CAMLparam4(vprobs, vgrad, vtarget, vupstream);
    gpu_nll_backward(Gpu_tensor_val(vprobs)->d_data,
                     Gpu_tensor_val(vgrad)->d_data,
                     Int_val(vtarget), (float)Double_val(vupstream));
    CAMLreturn(Val_unit);
}

/* ── Row ops ─────────────────────────────────────────────────────── */

CAMLprim value caml_gpu_row_forward(value vmat, value vrow,
                                     value vidx, value vcols) {
    CAMLparam4(vmat, vrow, vidx, vcols);
    gpu_row_forward(Gpu_tensor_val(vmat)->d_data,
                    Gpu_tensor_val(vrow)->d_data,
                    Int_val(vidx), Int_val(vcols));
    CAMLreturn(Val_unit);
}

CAMLprim value caml_gpu_row_backward(value vmat_grad, value vrow_grad,
                                      value vidx, value vcols) {
    CAMLparam4(vmat_grad, vrow_grad, vidx, vcols);
    gpu_row_backward(Gpu_tensor_val(vmat_grad)->d_data,
                     Gpu_tensor_val(vrow_grad)->d_data,
                     Int_val(vidx), Int_val(vcols));
    CAMLreturn(Val_unit);
}

/* ── Head extraction/insertion ────────────────────────────────────── */

void gpu_extract_head(const float *src, float *dst, int h,
                      int seq_len, int n_embd, int head_dim);
void gpu_insert_head(const float *src, float *dst, int h,
                     int seq_len, int n_embd, int head_dim);
void gpu_insert_head_accum(const float *src, float *dst, int h,
                           int seq_len, int n_embd, int head_dim);

CAMLprim value caml_gpu_extract_head(value vsrc, value vdst, value vh,
                                      value vseq, value vembd, value vhdim) {
    gpu_extract_head(Gpu_tensor_val(vsrc)->d_data,
                     Gpu_tensor_val(vdst)->d_data,
                     Int_val(vh), Int_val(vseq),
                     Int_val(vembd), Int_val(vhdim));
    return Val_unit;
}
CAMLprim value caml_gpu_extract_head_byte(value *argv, int argn) {
    (void)argn;
    return caml_gpu_extract_head(argv[0], argv[1], argv[2],
                                 argv[3], argv[4], argv[5]);
}

CAMLprim value caml_gpu_insert_head(value vsrc, value vdst, value vh,
                                     value vseq, value vembd, value vhdim) {
    gpu_insert_head(Gpu_tensor_val(vsrc)->d_data,
                    Gpu_tensor_val(vdst)->d_data,
                    Int_val(vh), Int_val(vseq),
                    Int_val(vembd), Int_val(vhdim));
    return Val_unit;
}
CAMLprim value caml_gpu_insert_head_byte(value *argv, int argn) {
    (void)argn;
    return caml_gpu_insert_head(argv[0], argv[1], argv[2],
                                argv[3], argv[4], argv[5]);
}

CAMLprim value caml_gpu_insert_head_accum(value vsrc, value vdst, value vh,
                                           value vseq, value vembd, value vhdim) {
    gpu_insert_head_accum(Gpu_tensor_val(vsrc)->d_data,
                          Gpu_tensor_val(vdst)->d_data,
                          Int_val(vh), Int_val(vseq),
                          Int_val(vembd), Int_val(vhdim));
    return Val_unit;
}
CAMLprim value caml_gpu_insert_head_accum_byte(value *argv, int argn) {
    (void)argn;
    return caml_gpu_insert_head_accum(argv[0], argv[1], argv[2],
                                      argv[3], argv[4], argv[5]);
}

/* ── RoPE ────────────────────────────────────────────────────────── */

CAMLprim value caml_gpu_rope_forward(value vdata, value vcos, value vsin,
                                      value vseq, value vembd,
                                      value vhead, value vhdim) {
    gpu_rope_forward(Gpu_tensor_val(vdata)->d_data,
                     Gpu_tensor_val(vcos)->d_data,
                     Gpu_tensor_val(vsin)->d_data,
                     Int_val(vseq), Int_val(vembd),
                     Int_val(vhead), Int_val(vhdim));
    return Val_unit;
}

CAMLprim value caml_gpu_rope_forward_byte(value *argv, int argn) {
    (void)argn;
    return caml_gpu_rope_forward(
        argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
}

CAMLprim value caml_gpu_rope_backward(value vgrad, value vcos, value vsin,
                                       value vseq, value vembd,
                                       value vhead, value vhdim) {
    gpu_rope_backward(Gpu_tensor_val(vgrad)->d_data,
                      Gpu_tensor_val(vcos)->d_data,
                      Gpu_tensor_val(vsin)->d_data,
                      Int_val(vseq), Int_val(vembd),
                      Int_val(vhead), Int_val(vhdim));
    return Val_unit;
}

CAMLprim value caml_gpu_rope_backward_byte(value *argv, int argn) {
    (void)argn;
    return caml_gpu_rope_backward(
        argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
}

/* ── Causal mask ─────────────────────────────────────────────────── */

CAMLprim value caml_gpu_causal_mask_scale(value vscores, value vscale,
                                           value vseq) {
    CAMLparam3(vscores, vscale, vseq);
    gpu_causal_mask_scale(Gpu_tensor_val(vscores)->d_data,
                          Double_val(vscale), Int_val(vseq));
    CAMLreturn(Val_unit);
}

/* ── Optimiser ───────────────────────────────────────────────────── */

CAMLprim value caml_gpu_adam_step(value vparam, value vgrad,
                                   value vm, value vv,
                                   value vlr, value vb1, value vb2,
                                   value vbc1, value vbc2, value vn) {
    gpu_adam_step(Gpu_tensor_val(vparam)->d_data,
                  Gpu_tensor_val(vgrad)->d_data,
                  Gpu_tensor_val(vm)->d_data,
                  Gpu_tensor_val(vv)->d_data,
                  Double_val(vlr), Double_val(vb1), Double_val(vb2),
                  Double_val(vbc1), Double_val(vbc2), Int_val(vn));
    return Val_unit;
}

CAMLprim value caml_gpu_adam_step_byte(value *argv, int argn) {
    (void)argn;
    return caml_gpu_adam_step(
        argv[0], argv[1], argv[2], argv[3], argv[4],
        argv[5], argv[6], argv[7], argv[8], argv[9]);
}

CAMLprim value caml_gpu_grad_clip_norm(value vgrads, value vmax_norm) {
    /* TODO: Step 7 */
    (void)vgrads; (void)vmax_norm;
    return Val_unit;
}

CAMLprim value caml_gpu_sparse_grad_mask(value vgrads, value vfrac) {
    /* TODO: Step 7 */
    (void)vgrads; (void)vfrac;
    return Val_int(0);
}

CAMLprim value caml_gpu_elastic_pull(value vparam, value vanchor,
                                      value valpha, value vn) {
    CAMLparam4(vparam, vanchor, valpha, vn);
    gpu_elastic_pull(Gpu_tensor_val(vparam)->d_data,
                     Gpu_tensor_val(vanchor)->d_data,
                     Double_val(valpha), Int_val(vn));
    CAMLreturn(Val_unit);
}
