# OCaml Vidya — Reference for Nim Port

This documents the current state of the OCaml implementation so the Nim
port can replicate the architecture without reading 2000 lines of OCaml + CUDA.

## Architecture (103M params)

```
n_layer   = 8
n_embd    = 1024
block_size = 512
n_head    = 16
head_dim  = 64
half_dim  = 32
ffn_dim   = 4096
vocab     = 2188 (BPE with 2000 merges + 185 chars + 3 special tokens)
```

## Model Structure

Weight-tied GPT-2 style transformer. `lm_head` IS `wte` — same tensor.

**Per layer (8 total):**
- `attn_wq` [1024, 1024] — query projection
- `attn_wk` [1024, 1024] — key projection
- `attn_wv` [1024, 1024] — value projection
- `attn_wo` [1024, 1024] — output projection (residual-scaled init)
- `mlp_fc1` [4096, 1024] — FFN up-projection
- `mlp_fc2` [1024, 4096] — FFN down-projection (residual-scaled init)
- `ln1` [1024] — pre-attention RMSNorm scale
- `ln2` [1024] — pre-MLP RMSNorm scale

**Global:**
- `wte` [2188, 1024] — token embeddings (weight-tied as lm_head)
- `embed_norm` [1024] — post-embedding RMSNorm scale
- `final_norm` [1024] — pre-output RMSNorm scale

**Init:** Gaussian(0, 0.08) for most weights. attn_wo and mlp_fc2 use
std = 0.08 / sqrt(2 * n_layer) for residual scaling. Norm scales init to 1.0.

## Forward Pass (Training)

```
tokens [S]
  → batch_embed: lookup wte rows → [S, 1024]
  → batch_rmsnorm_affine(embed_norm) → [S, 1024]
  → batch_dropout(0.1) → [S, 1024]
  → 8x transformer_block → [S, 1024]
  → batch_rmsnorm_affine(final_norm) → [S, 1024]
  → batch_matmul(lm_head=wte) → [S, 2188] (logits)
```

**Transformer block:**
```
x → rmsnorm_affine(ln1) → Q,K,V matmuls → fused_attention → Wo matmul → dropout → + x (residual)
  → rmsnorm_affine(ln2) → fc1 matmul → GELU → fc2 matmul → dropout → + x (residual)
```

## Attention Detail

**RoPE (Rotary Position Embeddings):**
- Precomputed cos/sin tables [block_size, half_dim]
- Applied to Q and K before attention scores
- Rotation: for each head, each pair (j, j+half_dim):
  ```
  q_rot[j]          = q[j] * cos - q[j+half_dim] * sin
  q_rot[j+half_dim] = q[j] * sin + q[j+half_dim] * cos
  ```

**Multi-head attention (per head):**
1. Extract head h from [S, n_embd] → [S, head_dim] (strided copy)
2. scores = Q_h @ K_h^T → [S, S]
3. Scale by 1/sqrt(head_dim)
4. Causal mask: upper triangle → -inf
5. Row-wise softmax
6. output = softmax(scores) @ V_h → [S, head_dim]
7. Insert head back into [S, n_embd]

**Heads are interleaved** in the embedding dimension:
row i, head h = data[i * n_embd + h * head_dim .. + head_dim - 1]

## RMSNorm

```
rms = sqrt(mean(x^2) + 1e-5)
norm = x / rms
output = norm * gamma   (affine variant)
```

## GELU Activation

```
inner = 0.7978845608 * (x + 0.044715 * x^3)
gelu(x) = 0.5 * x * (1 + tanh(inner))
```

## Loss

Standard cross-entropy via NLL:
- For each position i: softmax(logits[i]) → probs, then -log(probs[target])
- Average over all positions

## Training

**Optimizer:** Adam
- lr = 0.0003 (cosine annealing with 5000 warmup steps)
- beta1 = 0.9, beta2 = 0.999, eps = 1e-8
- Gradient clipping: max norm 1.0
- num_steps = 200000

**Checkpoints:** Every 2500 steps. Filename pattern: vidya_103m_2k.bin, _5k.bin, etc.

## Memory Mechanism (RL Fine-tuning)

**Sparse gradient mask:** After backprop, zero out all but top 1% of
gradients by magnitude. Only ~1M weights (out of 103M) update per
interaction. This is the "frontal cortex" — selective retraining.

**Elastic weight consolidation:** After each RL step, pull all weights
back toward the base model: `w = (1-alpha)*w + alpha*anchor`.
Alpha = 0.01 (gentle). Weights that fire hard repeatedly resist the pull.

**Contrastive loss (DPO):** Push probability toward chosen response,
away from rejected. Loss = -log(sigmoid(beta * (log_pi_chosen - log_pi_rejected))).

## BPE Tokenizer

- 2000 merge rounds on the conversation corpus
- Special tokens: `<|bos|>` (id 0), `<|user|>`, `<|assistant|>`
- Vocab size: ~2188 (185 byte-level chars + 2000 merges + 3 special)
- Saved as binary: tokenizer_v3.bin (OCaml Marshal format)
- The Nim version should reimplement BPE or use a portable format

## Data

- `chat_input.txt`: 37,492 conversations, 25MB
- Format: `<|user|> text <|assistant|> text <|user|> text ...`
- One conversation per line
- Source: HuggingFace datasets (soda, dailydialog, etc.)

## Key Lessons from OCaml Port

1. **GC + GPU finalizers:** OCaml GC calling cudaFree on autograd graph
   buffers caused use-after-free. Fixed with Gc.full_major() between
   training steps. Nim's deterministic destructors should handle this
   better — Arraymancer tensors free on scope exit.

2. **CPU softmax bottleneck:** Causal softmax is still done on CPU with
   download/upload per head. 32 round trips per step. A GPU kernel for
   batched causal softmax would double training speed.

3. **Head extraction:** Initially done via CPU download/extract/upload —
   extremely slow. Fixed with GPU scatter/gather kernels. Arraymancer
   should handle this natively with slicing.

4. **Float64 → Float32:** OCaml uses float64 natively. The GPU port
   required converting everything. Nim + Arraymancer uses float32
   natively on GPU — no conversion needed.

5. **Build system pain:** OCaml dune + nvcc + cuBLAS linking was fragile.
   Nim compiles to C and links CUDA libraries directly. Much simpler.
