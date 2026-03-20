/* gpu_stubs.cu — CUDA kernels and cuBLAS wrappers for Vidya
 * =========================================================
 *
 * All tensor computation lives here. The OCaml side builds the
 * autograd graph and dispatches operations through gpu_bridge.c
 * into these functions.
 *
 * Conventions:
 *   - All device pointers are float32.
 *   - Block size is 256 threads unless noted otherwise.
 *   - Reductions use shared memory within a block.
 *   - Error checking via GPU_CHECK macro (debug builds). */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

#include "gpu_stubs.h"

/* ── Error checking ──────────────────────────────────────────────── */

#define GPU_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t stat = (call); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                __FILE__, __LINE__, (int)stat); \
        exit(1); \
    } \
} while(0)

/* ── Global handles ──────────────────────────────────────────────── */

static cublasHandle_t g_cublas = nullptr;
static curandGenerator_t g_curand = nullptr;

extern "C" void gpu_init(void) {
    if (!g_cublas) {
        CUBLAS_CHECK(cublasCreate(&g_cublas));
        /* Use TF32 on Ampere+ for faster matmul with minimal precision loss. */
        cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    if (!g_curand) {
        curandCreateGenerator(&g_curand, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(g_curand, 42ULL);
    }
}

/* ── Lifecycle ───────────────────────────────────────────────────── */

extern "C" gpu_tensor gpu_create(int n) {
    gpu_tensor t;
    t.numel = n;
    GPU_CHECK(cudaMalloc(&t.d_data, (size_t)n * sizeof(float)));
    GPU_CHECK(cudaMemset(t.d_data, 0, (size_t)n * sizeof(float)));
    return t;
}

extern "C" void gpu_free(gpu_tensor *t) {
    if (t->d_data) {
        cudaFree(t->d_data);
        t->d_data = nullptr;
    }
    t->numel = 0;
}

/* ── Transfers ───────────────────────────────────────────────────── */

/* Convert float64 → float32 and upload to device. */
extern "C" void gpu_upload(const double *host_f64, gpu_tensor *t) {
    int n = t->numel;
    float *tmp = (float *)malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) tmp[i] = (float)host_f64[i];
    GPU_CHECK(cudaMemcpy(t->d_data, tmp, (size_t)n * sizeof(float),
                         cudaMemcpyHostToDevice));
    free(tmp);
}

/* Download from device and convert float32 → float64. */
extern "C" void gpu_download(const gpu_tensor *t, double *host_f64) {
    int n = t->numel;
    float *tmp = (float *)malloc((size_t)n * sizeof(float));
    GPU_CHECK(cudaMemcpy(tmp, t->d_data, (size_t)n * sizeof(float),
                         cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; i++) host_f64[i] = (double)tmp[i];
    free(tmp);
}

extern "C" void gpu_zero(gpu_tensor *t) {
    GPU_CHECK(cudaMemset(t->d_data, 0, (size_t)t->numel * sizeof(float)));
}

/* ── Int buffer helpers (for token IDs) ───────────────────────────── */

extern "C" void gpu_upload_int_buf(const int *host, int **d_out, int n) {
    GPU_CHECK(cudaMalloc(d_out, (size_t)n * sizeof(int)));
    GPU_CHECK(cudaMemcpy(*d_out, host, (size_t)n * sizeof(int),
                         cudaMemcpyHostToDevice));
}

extern "C" void gpu_free_int_buf(int *d_ptr) {
    if (d_ptr) cudaFree(d_ptr);
}

extern "C" void gpu_copy(const gpu_tensor *src, gpu_tensor *dst, int n) {
    GPU_CHECK(cudaMemcpy(dst->d_data, src->d_data, (size_t)n * sizeof(float),
                         cudaMemcpyDeviceToDevice));
}

/* ── cuBLAS GEMM ───────────────────────────────────────────────────
 *
 * cuBLAS is column-major. Our data is row-major. The standard trick:
 *   row-major A[m,k] @ B[k,n] = C[m,n]
 * is equivalent to:
 *   col-major B^T[n,k] @ A^T[k,m] = C^T[n,m]
 * So we swap A↔B and flip transpose flags, then call cuBLAS
 * with (n, m, k) instead of (m, n, k). */

extern "C" void gpu_sgemm(int op, int m, int n, int k,
                           const float *a, const float *b, float *c) {
    float alpha = 1.0f;
    float beta  = (op & 1) ? 1.0f : 0.0f;

    /* Flip transposes for the row→col major trick.
     * Original row-major: ta applied to A, tb applied to B.
     * In cuBLAS col-major with swapped args: B gets ta, A gets tb. */
    int ta = (op & 4) ? 1 : 0;  /* original: transpose A */
    int tb = (op & 2) ? 1 : 0;  /* original: transpose B */

    /* cuBLAS sees B as the "first" matrix, A as the "second".
     * The transpose flags get applied to the swapped matrix. */
    cublasOperation_t opB = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opA = ta ? CUBLAS_OP_T : CUBLAS_OP_N;

    /* Leading dimensions in row-major storage:
     *   A[m,k] stored row-major: lda = k (or m if transposed)
     *   B[k,n] stored row-major: ldb = n (or k if transposed)
     *   C[m,n] stored row-major: ldc = n
     * cuBLAS with swapped args sees these as column-major leading dims. */
    int lda = ta ? m : k;  /* A's row-major ld */
    int ldb = tb ? k : n;  /* B's row-major ld */
    int ldc = n;

    CUBLAS_CHECK(cublasSgemm(g_cublas,
        opB, opA,         /* swapped transpose ops */
        n, m, k,          /* swapped m,n */
        &alpha,
        b, ldb,           /* B is "first" in cuBLAS */
        a, lda,           /* A is "second" in cuBLAS */
        &beta,
        c, ldc));
}

/* ── Element-wise kernels ────────────────────────────────────────── */

static const int BLOCK = 256;

__global__ void k_gelu_forward(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    float inner = 0.7978845608f * (xi + 0.044715f * xi * xi * xi);
    float t = tanhf(inner);
    y[i] = 0.5f * xi * (1.0f + t);
}

__global__ void k_gelu_backward(const float *x, const float *dy,
                                float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    float inner = 0.7978845608f * (xi + 0.044715f * xi * xi * xi);
    float t = tanhf(inner);
    float dtanh = 1.0f - t * t;
    float dgelu = 0.5f * (1.0f + t)
        + 0.5f * xi * dtanh * 0.7978845608f * (1.0f + 3.0f * 0.044715f * xi * xi);
    dx[i] += dy[i] * dgelu;
}

extern "C" void gpu_gelu_forward(const float *x, float *y, int n) {
    k_gelu_forward<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, y, n);
}

extern "C" void gpu_gelu_backward(const float *x, const float *dy,
                                  float *dx, int n) {
    k_gelu_backward<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, dy, dx, n);
}

__global__ void k_add(const float *a, const float *b, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] + b[i];
}

extern "C" void gpu_add_forward(const float *a, const float *b,
                                float *y, int n) {
    k_add<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(a, b, y, n);
}

__global__ void k_scale(const float *x, float s, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] * s;
}

extern "C" void gpu_scale_forward(const float *x, float s, float *y, int n) {
    k_scale<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, s, y, n);
}

/* ── Dropout ──────────────────────────────────────────────────────
 *
 * Generate uniform randoms via cuRAND, threshold to create a binary
 * mask, scale surviving activations by 1/(1-rate). The mask stores
 * the scale value (or 0.0) for use in backward. */

__global__ void k_dropout_forward(const float *x, float *y,
                                  float *mask, float rate, float scale,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    /* mask contains uniform [0,1) from cuRAND. */
    if (mask[i] >= rate) {
        y[i] = x[i] * scale;
        mask[i] = scale;
    } else {
        y[i] = 0.0f;
        mask[i] = 0.0f;
    }
}

__global__ void k_dropout_backward(const float *dy, const float *mask,
                                   float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dx[i] += dy[i] * mask[i];
}

extern "C" void gpu_dropout_forward(const float *x, float *y, float *mask,
                                    float rate, int n) {
    float scale = 1.0f / (1.0f - rate);
    curandGenerateUniform(g_curand, mask, n);
    int blocks = (n + BLOCK - 1) / BLOCK;
    k_dropout_forward<<<blocks, BLOCK>>>(x, y, mask, rate, scale, n);
}

extern "C" void gpu_dropout_backward(const float *dy, const float *mask,
                                     float *dx, int n) {
    k_dropout_backward<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(dy, mask, dx, n);
}

/* ── Softmax ───────────────────────────────────────────────────────
 *
 * Row-wise softmax. Each block handles one row.
 * Uses shared memory for max-find and sum-exp reductions.
 * For vocab-sized rows (~2K), one block of 256 threads suffices. */

__global__ void k_softmax_forward(const float *x, float *y,
                                  int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * cols;
    float *yi = y + row * cols;

    extern __shared__ float sdata[];

    /* Pass 1: find max. */
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        local_max = fmaxf(local_max, xi[j]);
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x],
                                        sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    /* Pass 2: exp and sum. */
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float e = expf(xi[j] - row_max);
        yi[j] = e;
        local_sum += e;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float sum = sdata[0];

    /* Pass 3: normalise. */
    float inv_sum = 1.0f / sum;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        yi[j] *= inv_sum;
}

__global__ void k_softmax_backward(const float *y, const float *dy,
                                   float *dx, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *yi = y + row * cols;
    const float *dyi = dy + row * cols;
    float *dxi = dx + row * cols;

    extern __shared__ float sdata[];

    /* dot(dy, y) */
    float local_dot = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        local_dot += dyi[j] * yi[j];
    sdata[threadIdx.x] = local_dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float dot = sdata[0];

    /* dx += y * (dy - dot) */
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        dxi[j] += yi[j] * (dyi[j] - dot);
}

extern "C" void gpu_softmax_forward(const float *x, float *y, int n) {
    /* Single-row softmax. One block, n elements. */
    int threads = (n < 256) ? n : 256;
    k_softmax_forward<<<1, threads, threads * sizeof(float)>>>(x, y, 1, n);
}

extern "C" void gpu_softmax_backward(const float *y, const float *dy,
                                     float *dx, int n) {
    int threads = (n < 256) ? n : 256;
    k_softmax_backward<<<1, threads, threads * sizeof(float)>>>(y, dy, dx, 1, n);
}

/* ── RMSNorm ───────────────────────────────────────────────────────
 *
 * Each block handles one row. Shared memory for sum-of-squares. */

__global__ void k_rmsnorm_forward(const float *x, float *y, float *rms_out,
                                  int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * dim;
    float *yi = y + row * dim;

    extern __shared__ float sdata[];

    float local_ss = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        local_ss += xi[j] * xi[j];
    sdata[threadIdx.x] = local_ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = sqrtf(sdata[0] / (float)dim + 1e-5f);
    if (threadIdx.x == 0) rms_out[row] = rms;

    float inv_rms = 1.0f / rms;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        yi[j] = xi[j] * inv_rms;
}

__global__ void k_rmsnorm_affine_forward(const float *x, const float *gamma,
                                         float *y, float *norm_data,
                                         float *rms_out, int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * dim;
    float *yi = y + row * dim;
    float *ni = norm_data + row * dim;

    extern __shared__ float sdata[];

    float local_ss = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        local_ss += xi[j] * xi[j];
    sdata[threadIdx.x] = local_ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = sqrtf(sdata[0] / (float)dim + 1e-5f);
    if (threadIdx.x == 0) rms_out[row] = rms;

    float inv_rms = 1.0f / rms;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float n = xi[j] * inv_rms;
        ni[j] = n;
        yi[j] = n * gamma[j];
    }
}

extern "C" void gpu_rmsnorm_forward(const float *x, float *y,
                                    float *rms_out, int rows, int dim) {
    int threads = (dim < 256) ? dim : 256;
    k_rmsnorm_forward<<<rows, threads, threads * sizeof(float)>>>(
        x, y, rms_out, rows, dim);
}

/* Simpler affine forward: compute rms, normalise, scale by gamma.
 * We store rms per row for backward. norm_data is not stored —
 * the backward recomputes it from x and rms. */
__global__ void k_rmsnorm_affine_fwd_simple(const float *x, const float *gamma,
                                             float *y, float *rms_out,
                                             int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * dim;
    float *yi = y + row * dim;

    extern __shared__ float sdata[];

    float local_ss = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        local_ss += xi[j] * xi[j];
    sdata[threadIdx.x] = local_ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = sqrtf(sdata[0] / (float)dim + 1e-5f);
    if (threadIdx.x == 0) rms_out[row] = rms;

    float inv_rms = 1.0f / rms;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        yi[j] = xi[j] * inv_rms * gamma[j];
}

extern "C" void gpu_rmsnorm_affine_forward(const float *x, const float *gamma,
                                           float *y, float *rms_out,
                                           int rows, int dim) {
    int threads = (dim < 256) ? dim : 256;
    k_rmsnorm_affine_fwd_simple<<<rows, threads, threads * sizeof(float)>>>(
        x, gamma, y, rms_out, rows, dim);
}

/* RMSNorm backward:
 *   dx_i += (dy_i - norm_i * mean(dy · norm)) / rms
 * where norm_i = x_i / rms. Each block handles one row. */
__global__ void k_rmsnorm_backward(const float *x, const float *dy,
                                   const float *rms_out, float *dx,
                                   int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * dim;
    const float *dyi = dy + row * dim;
    float *dxi = dx + row * dim;
    float rms = rms_out[row];
    float inv_rms = 1.0f / rms;
    float dimf = (float)dim;

    extern __shared__ float sdata[];

    /* Compute normalized values and dot(dy, norm) */
    float local_dot = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float ni = xi[j] * inv_rms;
        local_dot += dyi[j] * ni;
    }
    sdata[threadIdx.x] = local_dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean_dot = sdata[0] / dimf;

    /* dx += (dy - norm * mean_dot) / rms */
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float ni = xi[j] * inv_rms;
        dxi[j] += (dyi[j] - ni * mean_dot) * inv_rms;
    }
}

extern "C" void gpu_rmsnorm_backward(const float *x, const float *dy,
                                     const float *rms_out, float *dx,
                                     int rows, int dim) {
    int threads = (dim < 256) ? dim : 256;
    k_rmsnorm_backward<<<rows, threads, threads * sizeof(float)>>>(
        x, dy, rms_out, dx, rows, dim);
}

/* RMSNorm affine backward:
 *   Forward was: norm_i = x_i / rms, y_i = norm_i * gamma_i
 *   Backward:
 *     dgamma_j += sum_over_rows(dy_ij * norm_ij)
 *     dn_i = dy_i * gamma_i
 *     dx_i += (dn_i - norm_i * mean(dn · norm)) / rms */
__global__ void k_rmsnorm_affine_backward(
    const float *x, const float *gamma, const float *dy,
    const float *rms_out, float *dx, float *dgamma,
    int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * dim;
    const float *dyi = dy + row * dim;
    float *dxi = dx + row * dim;
    float rms = rms_out[row];
    float inv_rms = 1.0f / rms;
    float dimf = (float)dim;

    extern __shared__ float sdata[];

    /* Compute dot(dn, norm) where dn_j = dy_j * gamma_j */
    float local_dot = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float ni = xi[j] * inv_rms;
        float dn_j = dyi[j] * gamma[j];
        local_dot += dn_j * ni;
    }
    sdata[threadIdx.x] = local_dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean_dot = sdata[0] / dimf;

    /* Accumulate dgamma and dx */
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float ni = xi[j] * inv_rms;
        /* dgamma accumulates across rows — use atomicAdd */
        atomicAdd(&dgamma[j], dyi[j] * ni);
        /* dx */
        float dn_j = dyi[j] * gamma[j];
        dxi[j] += (dn_j - ni * mean_dot) * inv_rms;
    }
}

extern "C" void gpu_rmsnorm_affine_backward(const float *x, const float *gamma,
                                            const float *dy, const float *norm_data,
                                            const float *rms_out,
                                            float *dx, float *dgamma,
                                            int rows, int dim) {
    (void)norm_data; /* We recompute norm from x and rms instead */
    int threads = (dim < 256) ? dim : 256;
    k_rmsnorm_affine_backward<<<rows, threads, threads * sizeof(float)>>>(
        x, gamma, dy, rms_out, dx, dgamma, rows, dim);
}

/* ── Head extraction/insertion ────────────────────────────────────
 *
 * Tensors are [seq, n_embd] with heads interleaved:
 *   row i, head h = data[i * n_embd + h * head_dim .. + head_dim - 1]
 * These kernels copy between the interleaved layout and contiguous
 * per-head [seq, head_dim] buffers entirely on GPU. */

__global__ void k_extract_head(const float *src, float *dst, int h,
                               int seq_len, int n_embd, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * head_dim;
    if (idx >= total) return;
    int pos = idx / head_dim;
    int j = idx % head_dim;
    dst[idx] = src[pos * n_embd + h * head_dim + j];
}

__global__ void k_insert_head(const float *src, float *dst, int h,
                              int seq_len, int n_embd, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * head_dim;
    if (idx >= total) return;
    int pos = idx / head_dim;
    int j = idx % head_dim;
    dst[pos * n_embd + h * head_dim + j] = src[idx];
}

__global__ void k_insert_head_accum(const float *src, float *dst, int h,
                                    int seq_len, int n_embd, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * head_dim;
    if (idx >= total) return;
    int pos = idx / head_dim;
    int j = idx % head_dim;
    dst[pos * n_embd + h * head_dim + j] += src[idx];
}

extern "C" void gpu_extract_head(const float *src, float *dst, int h,
                                 int seq_len, int n_embd, int head_dim) {
    int n = seq_len * head_dim;
    k_extract_head<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        src, dst, h, seq_len, n_embd, head_dim);
}

extern "C" void gpu_insert_head(const float *src, float *dst, int h,
                                int seq_len, int n_embd, int head_dim) {
    int n = seq_len * head_dim;
    k_insert_head<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        src, dst, h, seq_len, n_embd, head_dim);
}

extern "C" void gpu_insert_head_accum(const float *src, float *dst, int h,
                                      int seq_len, int n_embd, int head_dim) {
    int n = seq_len * head_dim;
    k_insert_head_accum<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        src, dst, h, seq_len, n_embd, head_dim);
}

/* ── Embedding ───────────────────────────────────────────────────── */

__global__ void k_embed_forward(const float *wte, const int *tokens,
                                float *out, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * dim;
    if (idx >= total) return;
    int pos = idx / dim;
    int j = idx % dim;
    out[idx] = wte[tokens[pos] * dim + j];
}

__global__ void k_embed_backward(float *wte_grad, const int *tokens,
                                 const float *dout, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * dim;
    if (idx >= total) return;
    int pos = idx / dim;
    int j = idx % dim;
    atomicAdd(&wte_grad[tokens[pos] * dim + j], dout[idx]);
}

extern "C" void gpu_embed_forward(const float *wte, const int *tokens,
                                  float *out, int seq_len, int dim) {
    int n = seq_len * dim;
    k_embed_forward<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        wte, tokens, out, seq_len, dim);
}

extern "C" void gpu_embed_backward(float *wte_grad, const int *tokens,
                                   const float *dout, int seq_len, int dim) {
    int n = seq_len * dim;
    k_embed_backward<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        wte_grad, tokens, dout, seq_len, dim);
}

/* ── Loss ────────────────────────────────────────────────────────── */

extern "C" float gpu_nll_forward(const float *probs, int target) {
    /* Single float transfer — negligible cost. */
    float p;
    GPU_CHECK(cudaMemcpy(&p, probs + target, sizeof(float),
                         cudaMemcpyDeviceToHost));
    return -logf(p);
}

extern "C" void gpu_nll_backward(const float *probs, float *probs_grad,
                                 int target, float upstream) {
    /* Read probs[target], compute gradient, write back. */
    float p;
    GPU_CHECK(cudaMemcpy(&p, probs + target, sizeof(float),
                         cudaMemcpyDeviceToHost));
    float grad = -upstream / p;
    /* Read-modify-write: add to existing gradient. */
    float existing;
    GPU_CHECK(cudaMemcpy(&existing, probs_grad + target, sizeof(float),
                         cudaMemcpyDeviceToHost));
    existing += grad;
    GPU_CHECK(cudaMemcpy(probs_grad + target, &existing, sizeof(float),
                         cudaMemcpyHostToDevice));
}

/* ── Row extraction ──────────────────────────────────────────────── */

extern "C" void gpu_row_forward(const float *mat, float *row,
                                int row_idx, int cols) {
    GPU_CHECK(cudaMemcpy(row, mat + row_idx * cols,
                         (size_t)cols * sizeof(float),
                         cudaMemcpyDeviceToDevice));
}

extern "C" void gpu_row_backward(float *mat_grad, const float *row_grad,
                                 int row_idx, int cols) {
    /* Accumulate: mat_grad[row] += row_grad. */
    float alpha = 1.0f;
    CUBLAS_CHECK(cublasSaxpy(g_cublas, cols, &alpha, row_grad, 1,
                             mat_grad + row_idx * cols, 1));
}

/* ── RoPE ────────────────────────────────────────────────────────── */

__global__ void k_rope(float *data, const float *cos_tab,
                       const float *sin_tab,
                       int seq_len, int n_embd, int n_head,
                       int head_dim, int half_dim, int sign) {
    /* sign=+1 for forward, -1 for backward (inverse rotation). */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_head * half_dim;
    if (idx >= total) return;

    int pos = idx / (n_head * half_dim);
    int rem = idx % (n_head * half_dim);
    int h = rem / half_dim;
    int f = rem % half_dim;

    int base = pos * n_embd + h * head_dim;
    float c = cos_tab[pos * half_dim + f];
    float s = sin_tab[pos * half_dim + f];

    float x0 = data[base + f];
    float x1 = data[base + f + half_dim];
    data[base + f]            = x0 * c - (float)sign * x1 * s;
    data[base + f + half_dim] = (float)sign * x0 * s + x1 * c;
}

extern "C" void gpu_rope_forward(float *data, const float *cos_tab,
                                 const float *sin_tab,
                                 int seq_len, int n_embd,
                                 int n_head, int head_dim) {
    int half_dim = head_dim / 2;
    int n = seq_len * n_head * half_dim;
    k_rope<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        data, cos_tab, sin_tab, seq_len, n_embd, n_head,
        head_dim, half_dim, 1);
}

extern "C" void gpu_rope_backward(float *grad, const float *cos_tab,
                                  const float *sin_tab,
                                  int seq_len, int n_embd,
                                  int n_head, int head_dim) {
    int half_dim = head_dim / 2;
    int n = seq_len * n_head * half_dim;
    k_rope<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        grad, cos_tab, sin_tab, seq_len, n_embd, n_head,
        head_dim, half_dim, -1);
}

/* ── Causal mask ─────────────────────────────────────────────────── */

__global__ void k_causal_mask_scale(float *scores, float scale,
                                    int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * seq_len;
    if (idx >= total) return;
    int row = idx / seq_len;
    int col = idx % seq_len;
    if (col > row)
        scores[idx] = -1e9f;  /* upper triangle → -inf */
    else
        scores[idx] *= scale;
}

extern "C" void gpu_causal_mask_scale(float *scores, float scale,
                                      int seq_len) {
    int n = seq_len * seq_len;
    k_causal_mask_scale<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        scores, scale, seq_len);
}

/* ── Optimiser kernels ───────────────────────────────────────────── */

__global__ void k_adam_step(float *param, float *grad, float *m, float *v,
                            float lr, float beta1, float beta2,
                            float bc1, float bc2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
    float m_hat = m[i] * bc1;
    float v_hat = v[i] * bc2;
    param[i] -= lr * m_hat / (sqrtf(v_hat) + 1e-8f);
    grad[i] = 0.0f;  /* zero grad after step */
}

extern "C" void gpu_adam_step(float *param, float *grad, float *m, float *v,
                              float lr, float beta1, float beta2,
                              float bc1, float bc2, int n) {
    k_adam_step<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        param, grad, m, v, lr, beta1, beta2, bc1, bc2, n);
}

__global__ void k_elastic_pull(float *param, const float *anchor,
                               float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    param[i] = (1.0f - alpha) * param[i] + alpha * anchor[i];
}

extern "C" void gpu_elastic_pull(float *param, const float *anchor,
                                 float alpha, int n) {
    k_elastic_pull<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        param, anchor, alpha, n);
}

/* grad_clip_norm and sparse_grad_mask are more complex (multi-tensor
 * reductions). Stubs for now — will be implemented in Step 7. */
extern "C" float gpu_grad_clip_norm(float **grads, const int *sizes,
                                    int n_tensors, float max_norm) {
    /* TODO: Step 7 — reduction across all grad tensors. */
    (void)grads; (void)sizes; (void)n_tensors; (void)max_norm;
    return 0.0f;
}

extern "C" int gpu_sparse_grad_mask(float **grads, const int *sizes,
                                    int n_tensors, float top_frac) {
    /* TODO: Step 7 — sort + threshold across all grad tensors. */
    (void)grads; (void)sizes; (void)n_tensors; (void)top_frac;
    return 0;
}
