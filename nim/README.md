# NimLLM

A grockable language model. No black boxes except the matrix multiplier.

One binary that grows with you. Feed it books. Talk to it. It learns from
your answers. Feed it your Claude Code logs. It learns how you think.

```
nimllm train data.txt          # train on text
nimllm chat                    # talk to it — it retrains on every exchange
nimllm read book.txt           # absorb a book into the weights
nimllm code                    # agent mode — uses your shell tools
```

## Why

Every other LLM is a black box. Billions of parameters trained on data
you never saw, running on code you can't read, behind an API you don't
control.

NimLLM is ~600 lines of Nim + ~300 lines of CUDA. You can read every
line. The only thing you can't fully see into is `cublasSgemm` — the
matrix multiplier. Everything else — attention, activation functions,
normalization, the optimizer, the backward pass — is code you wrote
and can change.

The model starts small. You train it on what matters to you. It
remembers your conversations. It grows as you feed it more.

## What It Does

**Trains from scratch.** No downloading someone else's weights. You
choose the data. You own the model.

**Remembers.** Every conversation retrains the model. Important things
stick through selective weight updates. Small talk fades. You don't
manage memory — the model figures out what matters.

**Reads.** `nimllm read` absorbs a document into the weights. Not RAG,
not context stuffing — actual learning. The knowledge is in the model,
not a search index.

**Grows with you.** Start at 4M parameters on a laptop GPU. Train it.
Talk to it. When it plateaus, scale up — widen the layers, add depth.
The model grows incrementally with your hardware and your data. 4M → 28M
→ 100M → 1B. Same code, same binary, just bigger numbers in the config.
You never need a datacenter because you never train from scratch at
full size — you grow into it.

## The Stack

```
Your text
  ↓
BPE tokenizer (trained on your corpus)
  ↓
Nim (model, forward, backward, optimizer)
  ↓
Flash attention kernel (numerically stable, fused)
  ↓
cuBLAS sgemm (matrix multiply — the one black box)
  ↓
Your GPU
```

No PyTorch. No Python. No frameworks. Nim compiles to C. C calls CUDA.
The binary is 4MB.

## Quick Start

```bash
# Install Nim
curl https://nim-lang.org/choosenim/init.sh -sSf | sh
export PATH=$HOME/.nimble/bin:$PATH

# Build
git clone https://github.com/rolandnsharp/vidya.git
cd vidya/nim/src
nvcc -c -O2 -Xcompiler -fPIC -o kernels.o kernels.cu
cd ..
nim c -d:release src/microgpt.nim

# Train
./src/microgpt
```

## Architecture

GPT-2 style transformer with flash attention.

| | Default | Scale up |
|---|---|---|
| Parameters | 28M | 100M+ |
| Layers | 8 | 12+ |
| Dimension | 512 | 1024+ |
| Heads | 8 | 16+ |
| Context | 512 | 1024+ |
| Attention | Flash (fused, numerically stable) | Same |
| Activation | GELU | SwiGLU (kernel ready) |
| Norm | RMSNorm with learnable gamma | Same |
| Optimizer | AdamW | Same |
| Precision | float32 | Same |

Flash attention makes overflow impossible — the online softmax never
computes `exp(large_number)`. This is what fixed the float32 numerical
stability issue that took two days to debug.

## Feed It Your Claude Code Logs

Claude Code stores conversations as JSONL. NimLLM can train on them:

```bash
# Convert Claude logs to training format
cat ~/.claude/logs/*.jsonl | jq -r '.messages[] | "<|" + .role + "|> " + .content' > claude_convos.txt
nimllm train claude_convos.txt
```

The model learns your coding patterns, your architecture preferences,
your problem-solving approach. It becomes a reflection of how you work.

## Customise

The model constants are at the top of `microgpt.nim`:

```nim
const
  nLayer   = 8       # more layers = deeper reasoning
  nEmbd    = 512     # wider = more capacity per layer
  nHead    = 8       # more heads = more attention patterns
  headDim  = 64      # keep at 64
  blockSize = 512    # context window in tokens
  ffnMul   = 4       # FFN width multiplier
```

Change, recompile, retrain. Everything is explicit.

## Hardware Portability

Nim compiles to C (NVIDIA/CUDA) and C++ (Tenstorrent/TT-Metalium).
The model, tokenizer, and training loop are hardware-agnostic. Only
`gpu.nim` and `kernels.cu` touch the hardware.

## Files

```
src/
  microgpt.nim       # the whole thing: model, forward, backward, training
  gpu.nim            # CUDA bindings and GPU memory management
  kernels.cu         # CUDA kernels: flash attention, GELU, RMSNorm, Adam
  bpe.nim            # BPE tokenizer
  autograd.nim       # tracked allocation helpers (no autograd graph used)
```

~600 lines of Nim. ~300 lines of CUDA. That's the entire LLM.

## Status

Training stably at loss 5.0 on 37K conversations. 28M parameters.
Flash attention forward and backward. No NaN at any learning rate.

## License

MIT
