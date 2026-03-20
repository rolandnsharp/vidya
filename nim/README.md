# NimLLM

A local language model you own. Train from scratch on your data, chat with it,
it retrains on your conversations and remembers.

One Nim binary. No Python. No PyTorch. Runs on NVIDIA (CUDA) today,
[Tenstorrent Blackhole](https://tenstorrent.com) (TT-Metalium) tomorrow.
Nim compiles to C and C++ — swap one file to target different silicon.

## Quick Start

```bash
# Install Nim
curl https://nim-lang.org/choosenim/init.sh -sSf | sh
export PATH=$HOME/.nimble/bin:$PATH

# Clone and build
git clone https://github.com/rolandnsharp/vidya.git
cd vidya/nim

# Compile CUDA kernels
cd src && nvcc -c -O2 -Xcompiler -fPIC -o kernels.o kernels.cu && cd ..

# Build the trainer
nim c -d:release src/ag_train.nim

# Train on your data (one conversation per line)
./src/ag_train
```

## Training Data

Put your training text in `chat_input.txt` at the repo root. One conversation
per line. Use `<|user|>` and `<|assistant|>` markers:

```
<|user|> Hello, how are you? <|assistant|> I'm doing well, thanks for asking.
<|user|> Tell me about Nim. <|assistant|> Nim is a language that compiles to C.
```

The tokenizer trains automatically on your data the first run.

## What You Need

- An NVIDIA GPU with CUDA installed (any card with 2+ GB VRAM works)
- Nim 2.0+ (`choosenim` handles this)
- ~400 MB disk per checkpoint

## Customise for Your Machine

Ask Claude Code to optimise the model for your hardware:

> "I have an RTX 4090 with 24GB VRAM. Resize the model to use most of it."

The key parameters are in `src/gpu_model.nim`:

```nim
const
  nLayer   = 8      # more layers = deeper reasoning, slower training
  nEmbd    = 1024   # wider = more capacity per layer
  nHead    = 16     # more heads = more attention patterns
  headDim  = 64     # nEmbd / nHead — keep at 64 for best results
  blockSize = 512   # context window in tokens
```

**Scaling guide:**

| GPU VRAM | nEmbd | nLayer | Params | Training speed |
|----------|-------|--------|--------|----------------|
| 2 GB     | 512   | 6      | ~20M   | ~20 steps/s    |
| 4 GB     | 768   | 6      | ~50M   | ~15 steps/s    |
| 8 GB     | 768   | 8      | ~70M   | ~12 steps/s    |
| 12 GB    | 1024  | 8      | ~103M  | ~7 steps/s     |
| 24 GB    | 1536  | 12     | ~350M  | ~3 steps/s     |

Keep `headDim = 64` (change `nHead` to match `nEmbd / 64`).

Training hyperparameters are in `src/ag_train.nim`:

```nim
const
  learningRate = 0.0001    # lower for larger models
  warmupSteps = 2000       # longer for more data
  weightDecay = 0.1        # keep this, prevents explosion
```

## Checkpoints

The model saves every 2500 steps to `nimllm_2500.bin`, `nimllm_5000.bin`, etc.
Also saves `nimllm_latest.bin` for easy resume. Restart and it picks up where
it left off.

## Architecture

103M parameter GPT-2 style transformer. Wide and shallow for memory
experiments.

- 8 transformer layers with pre-norm (RMSNorm)
- Multi-head attention with rotary position embeddings (RoPE)
- GELU activation in feed-forward layers
- Weight-tied embedding/output projection
- BPE tokenizer trained on your corpus
- AdamW optimizer with cosine LR decay and gradient clipping

The entire stack: ~1,000 lines of Nim + ~300 lines of CUDA.

## Files

```
src/
  ag_train.nim      — training loop (forward + backward + Adam)
  ag_forward.nim    — autograd-aware forward pass
  autograd.nim      — reverse-mode automatic differentiation
  gpu_model.nim     — model definition, weights on GPU
  gpu.nim           — CUDA bindings (cuBLAS, memory, kernels)
  kernels.cu        — CUDA kernels (GELU, RMSNorm, softmax, RoPE, Adam)
  bpe.nim           — BPE tokenizer
  gpu_forward.nim   — forward-only pass (no autograd, 94 steps/s)
  grad_check.nim    — numerical gradient verification
```

## Status

Training stably at loss 4.7 (from 9.4 at start). Gradient norms healthy at
3-9. Not yet conversational — needs loss ~3.5 to form coherent sentences.

## Hardware Portability

Nim compiles to C (for CUDA/NVIDIA) and C++ (for TT-Metalium/Tenstorrent).
The model, autograd, tokenizer, and training loop are hardware-agnostic. Only
`gpu.nim` and `kernels.cu` touch the hardware. To port to Blackhole:

```
gpu.nim      → replace cuBLAS calls with TT-NN calls
kernels.cu   → rewrite as TT-Metalium kernels (or run on Tensix RISC-V cores)
```

Everything else stays the same. One model definition, any silicon.

## Why Nim

Nim compiles to C. CUDA interop is just C function calls — no FFI bridge, no
GC issues, no build system hacks. You get Python's readability with C's
performance. The GPU layer is 200 lines. The autograd is 200 lines. PyTorch
is 3 million lines.

## License

MIT
