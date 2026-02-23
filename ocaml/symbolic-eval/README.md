# Symbolic-Eval: Constrained Decoding on Pre-Trained LLMs

Does symbolic constrained decoding improve pre-trained language model output
without retraining?

This experiment loads a pre-trained model (Phi-3 mini, Llama 3.2 3B, etc.)
via llama.cpp, applies the symbolic layer from Vidya at the logit level, and
measures whether output quality improves.

## The Claim

Lightweight symbolic post-processing — repetition penalty, word validation,
concept coherence, topic depth control, and TD-learned association weights —
can measurably improve a pre-trained LLM's output without touching its weights.

## How It Works

1. llama.cpp loads a GGUF model and runs the forward pass
2. OCaml receives the raw logit array via FFI
3. The symbolic layer biases logits before sampling:
   - Repetition penalty (penalize recent tokens)
   - Word validation (mask invalid word continuations)
   - Concept coherence (boost tokens related to active topics)
   - Topic depth (prevent repetitive topic loops)
   - TD learning (reinforce associations that predict confident tokens)
4. OCaml samples from the biased distribution
5. The chosen token is fed back to llama.cpp for the next step

## Setup

```bash
# Install llama.cpp (build from source for the shared library)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build

# Download a model
# (use huggingface-cli or manual download of a GGUF file)

# Build this project
cd ocaml/symbolic-eval
dune build

# Run
dune exec bin/main.exe -- --model path/to/model.gguf --corpus path/to/corpus.txt
```

## Evaluation

Compare generations with and without the symbolic layer on the same prompts:
- Repetition rate (n-gram overlap within a response)
- Topical coherence (do responses stay on topic?)
- Word validity (are all generated words real words?)
- Human preference (blind A/B comparison)
- TD learning curve (do associations improve over time?)

## Relationship to Vidya

This is a side project to the main Vidya work. Vidya trains a 10M parameter
model from scratch in OCaml — full control over weights, gradients, and
training loops. The interactive training mode (`--train`, `--prompt`/`--teach`)
does real gradient steps on Mr. Classic's weights.

Symbolic-eval asks a different question: can the symbolic layer we built for
Vidya improve a model we didn't train? If yes, it validates the symbolic
approach independently of our specific model. If the same constraints help
both a 10M model we trained and a 3B model someone else trained, the technique
generalises.

The practical angle: a Llama 3.2 3B running on CPU with Vidya's symbolic
layer and TD-learned weights could be a capable interactive system. The model
provides generation quality. The symbolic layer provides adaptive refinement
from human feedback. The context file provides multi-turn memory. No GPU
needed for any of it.

## Hardware Requirements

For CPU-only inference with llama.cpp (16GB RAM system):
- Llama 3.2 1B Q4: ~0.8GB, ~20-30 tok/s
- Llama 3.2 3B Q4: ~2GB, ~8-12 tok/s (recommended sweet spot)
- Phi-3.5 mini 3.8B Q4: ~2.5GB, ~6-10 tok/s
- 7-8B Q4: ~5GB, ~2-4 tok/s (slow for interactive use)

## Structure

```
symbolic-eval/
  lib/          OCaml symbolic layer (from vidya) + llama.cpp wrapper
  bin/          CLI entry point
  stubs/        C stubs for llama.cpp FFI
  PLAN.md       Implementation plan
  README.md     This file
```
