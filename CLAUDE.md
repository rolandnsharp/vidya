# Vidya — From-Scratch Language Model Framework in OCaml

## What This Is

A complete language model framework built from first principles in OCaml.
Autograd engine, BLAS-accelerated tensor ops, BPE tokenizer, pre-norm
transformer with RoPE, training loop with Adam optimizer, and symbolic
constrained decoding. No PyTorch, no Python dependencies.

## Project Structure

```
ocaml/vidya/          Main framework
  lib/
    tensor.ml         Autograd engine — all ops, forward+backward
    model.ml          Model definition, hyperparams, RoPE, KV cache
    forward.ml        Training (batched) and inference (KV cache) forward passes
    train.ml          Adam optimizer, cosine LR, gradient clipping, checkpoints
    generate.ml       Text generation (sample, prompted, chat)
    bpe.ml            BPE tokenizer (train + encode/decode)
    symbolic.ml       Constrained decoding (5 constraints + TD learning)
    knowledge.ml      Concept extraction from corpus (co-occurrence, associations)
    utils.ml          Shared helpers (word boundaries, RNG, weighted choice)
    blas.ml           FFI declaration for BLAS dgemm
  bin/
    main.ml           CLI entry point
  stubs/
    blas_stubs.c      C bridge to OpenBLAS cblas_dgemm
```

## Build and Run

```bash
cd ocaml/vidya
dune build                              # compile
dune exec bin/main.exe                  # train from scratch (200K steps)
dune exec bin/main.exe -- --load        # load checkpoint, generate samples
dune exec bin/main.exe -- --load --chat # interactive chat mode
dune exec bin/main.exe -- --load --prompt "Hello"  # prompted generation
```

## Current Model

- 10M params, 12 layers, 256 dim, 8 heads, 256-token context
- BPE: 2000 merges, ~2188 vocab, 4.2 chars/token
- Training data: 37K conversations (DailyDialog + SODA), 24MB
- Checkpoint: `microgpt_chat_10m_v3.bin`
- Architecture: pre-norm transformer, learnable RMSNorm, RoPE, dropout p=0.1
- Weight-tied embeddings (lm_head = wte)

## Key Patterns

### Tensor autograd
All ops in tensor.ml follow: compute forward → allocate grad → define backward
closure → return node. Gradients always accumulate (+=, never overwrite).

### Adding a new op
1. Add to tensor.ml (single-token for inference, batch for training)
2. Wire into forward.ml in both paths
3. If it has learnable params: add to model.ml types + init + collect_params
4. collect_params ordering = checkpoint format — changing it breaks old checkpoints

### Training vs inference
- Training: `gpt_forward_batch` — full sequence, builds autograd graph
- Inference: `gpt_forward` — single token + KV cache, no autograd
- Both paths must stay architecturally in sync
- Dropout only in training path (Tensor.training ref)

### BLAS integration
Single dgemm wrapper with 3-bit op flag (transpose A, transpose B, accumulate).
OpenBLAS pinned to 1 thread — at 256×256 matrices, threading overhead > benefit.

## Checkpoint Compatibility

Binary checkpoints are flat arrays matching collect_params order. Adding or
reordering params requires a new checkpoint file. Bump the checkpoint name
(e.g. _v3 → _v4) when collect_params changes.

## What NOT to Do

- Don't retrain while a training run is active (dune build will interfere)
- Don't change collect_params ordering without bumping the checkpoint name
- Don't add symbolic constraints to the inference path in forward.ml — they
  go in generate.ml after the forward pass returns logits
- Don't use Array.make for grad arrays that should be zeros — use Array.make
  (which does init to 0.0) not Array.create_float (which is uninitialized)

## Training Data

- `chat_input.txt` — 37K conversations, one per line, shuffled
- Format: multi-turn dialogues with `<|user|>` and `<|assistant|>` markers
- Sources: DailyDialog, SODA (Allen AI)
- Backup: `chat_input_backup.txt`

## Next Steps (from ROADMAP.md)

1. Evaluate v3 training loss curve (running now)
2. Get RTX 3090 for GPU training
3. CUDA backend (replace BLAS calls with GPU kernels)
4. Scale to 100M params (32 layers, 512 dim, 16 heads)
5. Tune symbolic layer at scale
