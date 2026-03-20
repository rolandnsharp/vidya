/* kernels.cu — CUDA kernels for Vidya (Nim version)
 *
 * Clean reimplementation. No OCaml bridge overhead.
 * Called directly from Nim via {.importc.} pragmas. */

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define BLOCK 256

extern "C" {

/* ── GELU ────────────────────────────────────────────────────────── */

__global__ void k_gelu_fwd(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    float t = tanhf(0.7978845608f * (xi + 0.044715f * xi * xi * xi));
    y[i] = 0.5f * xi * (1.0f + t);
}

__global__ void k_gelu_bwd(const float *x, const float *dy, float *dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    float inner = 0.7978845608f * (xi + 0.044715f * xi * xi * xi);
    float t = tanhf(inner);
    float dt = 1.0f - t * t;
    float dg = 0.5f * (1.0f + t)
        + 0.5f * xi * dt * 0.7978845608f * (1.0f + 3.0f * 0.044715f * xi * xi);
    dx[i] += dy[i] * dg;
}

void gpu_gelu_fwd(const float *x, float *y, int n) {
    k_gelu_fwd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, y, n);
}

void gpu_gelu_bwd(const float *x, const float *dy, float *dx, int n) {
    k_gelu_bwd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, dy, dx, n);
}

/* ── Element-wise ────────────────────────────────────────────────── */

__global__ void k_add(const float *a, const float *b, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] + b[i];
}

__global__ void k_add_inplace(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

__global__ void k_scale(const float *x, float s, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] * s;
}

void gpu_add(const float *a, const float *b, float *y, int n) {
    k_add<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(a, b, y, n);
}

void gpu_add_inplace(float *a, const float *b, int n) {
    k_add_inplace<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(a, b, n);
}

void gpu_scale(const float *x, float s, float *y, int n) {
    k_scale<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, s, y, n);
}

/* ── RMSNorm ─────────────────────────────────────────────────────
 *
 * Each block handles one row. Shared memory for sum-of-squares. */

__global__ void k_rmsnorm_affine(const float *x, const float *gamma,
                                 float *y, float *rms_out,
                                 int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * dim;
    float *yi = y + row * dim;

    extern __shared__ float sdata[];

    float ss = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        ss += xi[j] * xi[j];
    sdata[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = sqrtf(sdata[0] / (float)dim + 1e-5f);
    if (threadIdx.x == 0) rms_out[row] = rms;

    float inv = 1.0f / rms;
    for (int j = threadIdx.x; j < dim; j += blockDim.x)
        yi[j] = xi[j] * inv * gamma[j];
}

__global__ void k_rmsnorm_affine_bwd(const float *x, const float *gamma,
                                     const float *dy, const float *rms_out,
                                     float *dx, float *dgamma,
                                     int rows, int dim) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * dim;
    const float *dyi = dy + row * dim;
    float *dxi = dx + row * dim;
    float inv = 1.0f / rms_out[row];
    float dimf = (float)dim;

    extern __shared__ float sdata[];

    float dot = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float dn = dyi[j] * gamma[j];
        dot += dn * xi[j] * inv;
    }
    sdata[threadIdx.x] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float mean_dot = sdata[0] / dimf;

    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        float ni = xi[j] * inv;
        atomicAdd(&dgamma[j], dyi[j] * ni);
        float dn = dyi[j] * gamma[j];
        dxi[j] += (dn - ni * mean_dot) * inv;
    }
}

void gpu_rmsnorm_affine_fwd(const float *x, const float *gamma,
                            float *y, float *rms_out, int rows, int dim) {
    int threads = dim < 256 ? dim : 256;
    k_rmsnorm_affine<<<rows, threads, threads * sizeof(float)>>>(
        x, gamma, y, rms_out, rows, dim);
}

void gpu_rmsnorm_affine_bwd(const float *x, const float *gamma,
                            const float *dy, const float *rms_out,
                            float *dx, float *dgamma, int rows, int dim) {
    int threads = dim < 256 ? dim : 256;
    k_rmsnorm_affine_bwd<<<rows, threads, threads * sizeof(float)>>>(
        x, gamma, dy, rms_out, dx, dgamma, rows, dim);
}

/* ── Causal mask + scale ─────────────────────────────────────────── */

__global__ void k_causal_mask(float *scores, float scale, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * seq_len;
    if (idx >= total) return;
    int row = idx / seq_len;
    int col = idx % seq_len;
    if (col > row)
        scores[idx] = -1e9f;
    else
        scores[idx] *= scale;
}

void gpu_causal_mask(float *scores, float scale, int seq_len) {
    int n = seq_len * seq_len;
    k_causal_mask<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(scores, scale, seq_len);
}

/* ── Softmax (row-wise) ──────────────────────────────────────────── */

__global__ void k_softmax(const float *x, float *y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *xi = x + row * cols;
    float *yi = y + row * cols;

    extern __shared__ float sdata[];

    float mx = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        mx = fmaxf(mx, xi[j]);
    sdata[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    float sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float e = expf(xi[j] - row_max);
        yi[j] = e;
        sum += e;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv = 1.0f / sdata[0];
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        yi[j] *= inv;
}

__global__ void k_softmax_bwd(const float *y, const float *dy,
                               float *dx, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float *yi = y + row * cols;
    const float *dyi = dy + row * cols;
    float *dxi = dx + row * cols;

    extern __shared__ float sdata[];

    float dot = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        dot += dyi[j] * yi[j];
    sdata[threadIdx.x] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float d = sdata[0];
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        dxi[j] += yi[j] * (dyi[j] - d);
}

void gpu_softmax_fwd(const float *x, float *y, int rows, int cols) {
    int threads = cols < 256 ? cols : 256;
    k_softmax<<<rows, threads, threads * sizeof(float)>>>(x, y, rows, cols);
}

void gpu_softmax_bwd(const float *y, const float *dy, float *dx,
                     int rows, int cols) {
    int threads = cols < 256 ? cols : 256;
    k_softmax_bwd<<<rows, threads, threads * sizeof(float)>>>(y, dy, dx, rows, cols);
}

/* ── Head extraction/insertion ───────────────────────────────────── */

__global__ void k_extract_head(const float *src, float *dst, int h,
                               int seq_len, int n_embd, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * head_dim) return;
    int pos = idx / head_dim;
    int j = idx % head_dim;
    dst[idx] = src[pos * n_embd + h * head_dim + j];
}

__global__ void k_insert_head(const float *src, float *dst, int h,
                              int seq_len, int n_embd, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * head_dim) return;
    int pos = idx / head_dim;
    int j = idx % head_dim;
    dst[pos * n_embd + h * head_dim + j] = src[idx];
}

__global__ void k_insert_head_acc(const float *src, float *dst, int h,
                                  int seq_len, int n_embd, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * head_dim) return;
    int pos = idx / head_dim;
    int j = idx % head_dim;
    dst[pos * n_embd + h * head_dim + j] += src[idx];
}

void gpu_extract_head(const float *src, float *dst, int h,
                      int seq_len, int n_embd, int head_dim) {
    int n = seq_len * head_dim;
    k_extract_head<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        src, dst, h, seq_len, n_embd, head_dim);
}

void gpu_insert_head(const float *src, float *dst, int h,
                     int seq_len, int n_embd, int head_dim) {
    int n = seq_len * head_dim;
    k_insert_head<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        src, dst, h, seq_len, n_embd, head_dim);
}

void gpu_insert_head_acc(const float *src, float *dst, int h,
                         int seq_len, int n_embd, int head_dim) {
    int n = seq_len * head_dim;
    k_insert_head_acc<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        src, dst, h, seq_len, n_embd, head_dim);
}

/* ── RoPE ────────────────────────────────────────────────────────── */

__global__ void k_rope(float *data, const float *cos_tab,
                       const float *sin_tab, int seq_len, int n_embd,
                       int n_head, int head_dim, int half_dim, int sign) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * n_head * half_dim) return;
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

void gpu_rope_fwd(float *data, const float *cos_tab, const float *sin_tab,
                  int seq_len, int n_embd, int n_head, int head_dim) {
    int hd = head_dim / 2;
    int n = seq_len * n_head * hd;
    k_rope<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        data, cos_tab, sin_tab, seq_len, n_embd, n_head, head_dim, hd, 1);
}

void gpu_rope_bwd(float *grad, const float *cos_tab, const float *sin_tab,
                  int seq_len, int n_embd, int n_head, int head_dim) {
    int hd = head_dim / 2;
    int n = seq_len * n_head * hd;
    k_rope<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        grad, cos_tab, sin_tab, seq_len, n_embd, n_head, head_dim, hd, -1);
}

/* ── Embedding ───────────────────────────────────────────────────── */

__global__ void k_embed_fwd(const float *wte, const int *tokens,
                            float *out, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * dim) return;
    int pos = idx / dim;
    int j = idx % dim;
    out[idx] = wte[tokens[pos] * dim + j];
}

__global__ void k_embed_bwd(float *wte_grad, const int *tokens,
                            const float *dout, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * dim) return;
    int pos = idx / dim;
    int j = idx % dim;
    atomicAdd(&wte_grad[tokens[pos] * dim + j], dout[idx]);
}

void gpu_embed_fwd(const float *wte, const int *tokens, float *out,
                   int seq_len, int dim) {
    int n = seq_len * dim;
    k_embed_fwd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(wte, tokens, out, seq_len, dim);
}

void gpu_embed_bwd(float *wte_grad, const int *tokens, const float *dout,
                   int seq_len, int dim) {
    int n = seq_len * dim;
    k_embed_bwd<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(wte_grad, tokens, dout, seq_len, dim);
}

/* ── Adam ────────────────────────────────────────────────────────── */

__global__ void k_adam(float *param, float *grad, float *m, float *v,
                       float lr, float b1, float b2,
                       float bc1, float bc2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    m[i] = b1 * m[i] + (1.0f - b1) * g;
    v[i] = b2 * v[i] + (1.0f - b2) * g * g;
    param[i] -= lr * (m[i] * bc1) / (sqrtf(v[i] * bc2) + 1e-8f);
    grad[i] = 0.0f;
}

void gpu_adam(float *param, float *grad, float *m, float *v,
             float lr, float b1, float b2, float bc1, float bc2, int n) {
    k_adam<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
        param, grad, m, v, lr, b1, b2, bc1, bc2, n);
}

/* ── Elastic pull ────────────────────────────────────────────────── */

__global__ void k_elastic(float *param, const float *anchor, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    param[i] = (1.0f - alpha) * param[i] + alpha * anchor[i];
}

void gpu_elastic(float *param, const float *anchor, float alpha, int n) {
    k_elastic<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(param, anchor, alpha, n);
}

} /* extern "C" */
