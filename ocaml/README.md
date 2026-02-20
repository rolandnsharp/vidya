# MicroGPT in OCaml — Vidya

A from-scratch GPT training framework in OCaml, evolving from 5K params to 1.25M params across 11 stages. Trains on Plotinus's Enneads.

This is the [Vidya](../PLAN.md) project — started as a faithful translation of [Karpathy's microgpt.py](../microgpt.py), now a BPE-tokenized, BLAS-accelerated, weight-tied transformer.

## Quick Start — Stage 11 (latest)

Requires OCaml and OpenBLAS (`sudo apt install libopenblas-dev`).

**Compile:**
```bash
ocamlopt -O2 -o microgpt_tuned blas_stubs.c eleven_microgpt_tuned.ml \
  -ccopt "-I/usr/include/x86_64-linux-gnu" -cclib -lopenblas
```

**Train** (100K steps, ~48 min, saves checkpoint):
```bash
./microgpt_tuned
```

**Load saved weights + generate** (instant, ~3s):
```bash
./microgpt_tuned --load
```

**Prompted generation:**
```bash
./microgpt_tuned --load --prompt "what is the Absolute?"
```

## All Stages

Each stage is a self-contained `.ml` file. Stages 1-5 are pure OCaml, stages 6+ use OpenBLAS via `blas_stubs.c`.

| Stage | File | Params | What changed |
|-------|------|--------|-------------|
| 1 | `1_microgpt_mutable.ml` | 5,888 | Scalar autograd, char tokenizer |
| 1 | `1_microgpt_ref.ml` | 5,888 | Same, functional style |
| 5 | `5_microgpt_scaled.ml` | ~50K | Tensor autograd, scaled up |
| 6 | `6_microgpt_blas.ml` | ~50K | BLAS matmul via FFI |
| 7 | `7_microgpt_batched.ml` | ~50K | Batched training (all positions at once) |
| 8 | `8_microgpt_fullblas.ml` | ~50K | Full BLAS attention |
| 9 | `9_microgpt_bpe.ml` | ~270K | BPE tokenizer (200 merges) |
| 10 | `ten_microgpt_scaled.ml` | 1.3M | 128-dim, 8 heads, 6 layers, 128 context |
| 11 | `eleven_microgpt_tuned.ml` | 1.25M | Weight tying, residual scaling, checkpoints, prompted generation |

**Compile stages 1-5** (pure OCaml, no deps):
```bash
ocamlopt -o microgpt_mutable 1_microgpt_mutable.ml
```

**Compile stages 6-11** (need OpenBLAS):
```bash
ocamlopt -O2 -o <binary> blas_stubs.c <stage>.ml \
  -ccopt "-I/usr/include/x86_64-linux-gnu" -cclib -lopenblas
```

## Architecture (Stage 11)

```
input.txt (Plotinus Enneads, 16,478 docs)
    │
    ▼
BPE Tokenizer (500 merges, 580 vocab, 2.7 chars/token)
    │
    ▼
Transformer (128-dim, 8 heads, 6 layers, 128-token context):
    Token Embedding (weight-tied with output) ──► RoPE
        │
        ▼
    Transformer Block (×6):
        ├── RMSNorm → Multi-Head Attention (RoPE, KV cache)
        ├── Residual connection (scaled init)
        ├── RMSNorm → MLP (linear → ReLU → linear)
        └── Residual connection (scaled init)
        │
        ▼
    Output = input @ Embedding^T (weight tying)
        │
        ▼
    Softmax → Cross-Entropy Loss
    │
    ▼
Backward Pass (tensor autograd, BLAS-accelerated)
    │
    ▼
Adam Optimizer (cosine LR, warmup, grad clipping)
```

## Training Data

`input.txt` — Plotinus's Enneads (MacKenna translation), one paragraph per line.

## Next Steps

See [PLAN.md](../PLAN.md) for the full roadmap. Next: GPU acceleration.
