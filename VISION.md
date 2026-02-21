# Vidya: A From-Scratch Language Model Framework

Vidya is a language model framework built entirely in OCaml. No PyTorch, no
Python, no dependencies you don't understand. Autograd engine, tensor ops,
BPE tokenizer, transformer architecture, training loop, inference, and
symbolic constrained decoding — all from first principles.

The goal: train the best conversational model we can run locally, with a
framework we fully control and fully understand.

## What We're Building

A complete ML framework in OCaml that trains transformer language models:

- **Autograd engine** — forward/backward with gradient accumulation
- **BLAS-accelerated tensor ops** — matmul via C FFI, everything else in OCaml
- **BPE tokenizer** — trained from corpus, configurable merge count
- **Pre-norm transformer** — learnable RMSNorm, RoPE, KV cache, dropout
- **Symbolic constrained decoding** — logit biasing at inference time using
  corpus-derived concept knowledge, repetition penalties, word validation
- **TD learning** — concept associations strengthen/weaken based on model
  confidence during generation

## Why OCaml

- Fast enough for real training (BLAS for the hot path, OCaml for everything else)
- Type system catches bugs that Python silently propagates
- No dependency hell — the framework is self-contained
- Every line is ours to read, debug, and modify

## Current State

10M params, 12 layers, 256 dim, 37K conversations. Training on CPU.
Architecture: learnable RMSNorm, dropout, 2000-merge BPE vocab (~2100 tokens).

## The Road

1. **Now:** 10M param model training on CPU. Evaluating loss curves.
2. **Next:** GPU (RTX 3090) to iterate at 100M param scale — 32 layers,
   512 dim, 16 heads. This is where coherent conversation should emerge.
3. **Then:** Constrained decoding becomes genuinely useful once the base
   model is fluent. The symbolic layer steers; the neural model generates.

## The Symbolic Layer

Not a bolted-on rule engine. The symbolic system operates at the logit level:

- **Word validation** — prevent the model from generating non-words
- **Concept coherence** — boost tokens related to recently activated topics
- **Topic depth control** — prevent repetitive loops on the same concepts
- **TD learning** — associations that predict high-confidence tokens strengthen

The neural model decides what to say. The symbolic system decides what's valid.
This split becomes powerful once the base model is fluent enough to benefit
from steering rather than fighting it.

## What Makes This Different

- **From scratch** — not a wrapper around PyTorch. Every op has a hand-written
  backward pass. Every architectural choice is deliberate.
- **Single language** — OCaml end to end. No Python glue, no YAML configs,
  no notebook workflows.
- **Symbolic integration** — constrained decoding using corpus-derived knowledge,
  not just pure neural sampling.
- **Transparent** — small enough codebase to fit in your head. ~2000 lines of
  OCaml for the entire framework.
