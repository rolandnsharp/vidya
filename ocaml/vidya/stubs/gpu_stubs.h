/* gpu_stubs.h — Shared declarations between CUDA kernels and OCaml bridge
 *
 * The gpu_tensor struct lives inside an OCaml custom block.
 * OCaml's GC owns the block; the finalizer calls cudaFree.
 * All kernel functions take raw float* device pointers. */

#ifndef GPU_STUBS_H
#define GPU_STUBS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a float32 array on the GPU.
 * Embedded in an OCaml custom block via caml_alloc_custom. */
typedef struct {
    float *d_data;  /* cudaMalloc'd device pointer (NULL after free) */
    int    numel;   /* number of float32 elements */
} gpu_tensor;

/* ── Lifecycle ───────────────────────────────────────────────────── */

/* Initialise cuBLAS handle + cuRAND generator. Call once. */
void gpu_init(void);

/* Allocate n float32 elements on device, zero-initialised. */
gpu_tensor gpu_create(int n);

/* Free device memory. Safe to call on already-freed tensors. */
void gpu_free(gpu_tensor *t);

/* ── Transfers (host ↔ device) ───────────────────────────────────── */

/* Upload: convert float64 host array → float32 device buffer.
 * host_f64 has t->numel elements. */
void gpu_upload(const double *host_f64, gpu_tensor *t);

/* Download: convert float32 device buffer → float64 host array.
 * host_f64 must have room for t->numel elements. */
void gpu_download(const gpu_tensor *t, double *host_f64);

/* Zero all elements on device. */
void gpu_zero(gpu_tensor *t);

/* Copy n floats from src to dst (device → device). */
void gpu_copy(const gpu_tensor *src, gpu_tensor *dst, int n);

/* ── cuBLAS GEMM ─────────────────────────────────────────────────── */

/* sgemm with 3-bit op encoding (same convention as blas_stubs.c):
 *   bit 2 (4): transpose A
 *   bit 1 (2): transpose B
 *   bit 0 (1): accumulate (beta=1) vs overwrite (beta=0)
 * m,n = result dims, k = contracted dim.
 * All matrices are logically row-major. */
void gpu_sgemm(int op, int m, int n, int k,
               const float *a, const float *b, float *c);

/* ── Element-wise kernels ────────────────────────────────────────── */

void gpu_gelu_forward(const float *x, float *y, int n);
void gpu_gelu_backward(const float *x, const float *dy, float *dx, int n);

void gpu_add_forward(const float *a, const float *b, float *y, int n);
void gpu_scale_forward(const float *x, float s, float *y, int n);

/* Dropout: mask is a float buffer (0.0 or scale). seed for cuRAND. */
void gpu_dropout_forward(const float *x, float *y, float *mask,
                         float rate, int n);
void gpu_dropout_backward(const float *dy, const float *mask,
                          float *dx, int n);

/* ── Reduction kernels ───────────────────────────────────────────── */

/* Softmax over a vector of length n (single row). */
void gpu_softmax_forward(const float *x, float *y, int n);
void gpu_softmax_backward(const float *y, const float *dy,
                          float *dx, int n);

/* RMSNorm: rows × dim matrix, normalise each row.
 * rms_out stores per-row RMS values for backward. */
void gpu_rmsnorm_forward(const float *x, float *y, float *rms_out,
                         int rows, int dim);
void gpu_rmsnorm_affine_forward(const float *x, const float *gamma,
                                float *y, float *rms_out,
                                int rows, int dim);
void gpu_rmsnorm_backward(const float *x, const float *dy,
                          const float *rms_out, float *dx,
                          int rows, int dim);
void gpu_rmsnorm_affine_backward(const float *x, const float *gamma,
                                 const float *dy, const float *norm_data,
                                 const float *rms_out,
                                 float *dx, float *dgamma,
                                 int rows, int dim);

/* ── Embedding ───────────────────────────────────────────────────── */

/* Gather rows from wte[vocab, dim] by token IDs → out[seq, dim]. */
void gpu_embed_forward(const float *wte, const int *tokens,
                       float *out, int seq_len, int dim);

/* Scatter-add gradients back into wte_grad. Uses atomicAdd. */
void gpu_embed_backward(float *wte_grad, const int *tokens,
                        const float *dout, int seq_len, int dim);

/* ── Loss ────────────────────────────────────────────────────────── */

/* NLL: returns -log(probs[target]) as a host float. */
float gpu_nll_forward(const float *probs, int target);

/* NLL backward: probs_grad[target] -= upstream / probs[target]. */
void gpu_nll_backward(const float *probs, float *probs_grad,
                      int target, float upstream);

/* ── Row extraction ──────────────────────────────────────────────── */

void gpu_row_forward(const float *mat, float *row, int row_idx, int cols);
void gpu_row_backward(float *mat_grad, const float *row_grad,
                      int row_idx, int cols);

/* ── Head extraction/insertion ────────────────────────────────────── */

/* Extract head h from [seq, n_embd] → [seq, head_dim].
 * Copies strided data into a contiguous buffer on device. */
void gpu_extract_head(const float *src, float *dst, int h,
                      int seq_len, int n_embd, int head_dim);

/* Insert [seq, head_dim] into head h of [seq, n_embd] — overwrite. */
void gpu_insert_head(const float *src, float *dst, int h,
                     int seq_len, int n_embd, int head_dim);

/* Insert [seq, head_dim] into head h of [seq, n_embd] — accumulate. */
void gpu_insert_head_accum(const float *src, float *dst, int h,
                           int seq_len, int n_embd, int head_dim);

/* ── RoPE ────────────────────────────────────────────────────────── */

/* Apply rotary embeddings to Q or K tensor [seq, n_embd].
 * cos_table, sin_table: [block_size, half_dim] on device.
 * Modifies data in-place. */
void gpu_rope_forward(float *data, const float *cos_tab,
                      const float *sin_tab,
                      int seq_len, int n_embd, int n_head, int head_dim);
void gpu_rope_backward(float *grad, const float *cos_tab,
                       const float *sin_tab,
                       int seq_len, int n_embd, int n_head, int head_dim);

/* ── Causal mask ─────────────────────────────────────────────────── */

/* Scale attention scores and set upper triangle to -inf.
 * scores is [seq, seq]. */
void gpu_causal_mask_scale(float *scores, float scale, int seq_len);

/* ── Optimiser ───────────────────────────────────────────────────── */

/* Adam update on a single parameter tensor. */
void gpu_adam_step(float *param, float *grad, float *m, float *v,
                   float lr, float beta1, float beta2,
                   float bc1, float bc2, int n);

/* Compute global gradient norm across multiple tensors, clip if needed.
 * Returns the computed norm. */
float gpu_grad_clip_norm(float **grads, const int *sizes,
                         int n_tensors, float max_norm);

/* Keep top frac of gradients by magnitude, zero the rest.
 * Operates across multiple tensors. Returns count kept. */
int gpu_sparse_grad_mask(float **grads, const int *sizes,
                         int n_tensors, float top_frac);

/* Elastic pull: param = (1-alpha)*param + alpha*anchor. */
void gpu_elastic_pull(float *param, const float *anchor,
                      float alpha, int n);

#ifdef __cplusplus
}
#endif

#endif /* GPU_STUBS_H */
