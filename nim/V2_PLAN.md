# NimLLM V2 Plan

V1 is training. V2 is the production model.

## V1 → V2 Changes

| | V1 (current) | V2 (planned) |
|---|---|---|
| Params | 103M | 140M |
| Layers | 8 | 12 |
| Dim | 1024 | 1024 |
| Q heads | 16 | 16 |
| KV heads | 16 | 4 (GQA) |
| FFN | GELU, 4096 | SwiGLU, 2730 |
| Vocab | 2259 (2K merges) | 8000+ (8K merges) |
| Context | 512 | 1024 |
| Init std | 0.02 | 0.02 |
| Data | 37K convos (25MB) | 2.4M convos (2.5GB) |
| Dropout | none | 0.1 |

## What's Already Built

- [x] SwiGLU forward/backward CUDA kernel
- [x] GQA head extraction/insertion CUDA kernels (with atomicAdd)
- [x] `agSwiGLU` autograd op
- [x] `agGqaAttention` autograd op with full backward
- [x] `ag_forward_v2.nim` — transformer block compiles
- [x] `model_v2.nim` — type definitions
- [x] 2.4M conversations converted to `chat_input_all.txt`

## What's Left to Build

- [ ] `initGpuModelV2` — allocate and upload weights
- [ ] V2 training entry point (`ag_train_v2.nim`)
- [ ] BPE tokenizer with 8K merges on big dataset
- [ ] Dropout in forward pass (training only)
- [ ] Checkpoint save/load for V2 model shape
- [ ] Data loader that handles 2.5GB without loading all into RAM
- [ ] Gradient accumulation (batch 8+) — now that init std is fixed

## Architecture Decisions

### GQA (Grouped Query Attention)

16 Q heads, 4 KV heads. Every 4 Q heads share one K,V pair.
Same quality as full attention at 40% fewer attention parameters.
Saves VRAM during inference (smaller KV cache).

K projection: [nKvHead * headDim, nEmbd] = [256, 1024]
V projection: [nKvHead * headDim, nEmbd] = [256, 1024]

vs V1:
K projection: [nEmbd, nEmbd] = [1024, 1024]
V projection: [nEmbd, nEmbd] = [1024, 1024]

### SwiGLU

Replaces GELU in the FFN. Two projections (gate + up) instead of one:

```
V1 GELU FFN:          V2 SwiGLU FFN:
  h = gelu(x @ fc1)     gate = x @ Wgate
  out = h @ fc2          up = x @ Wup
                         h = swish(gate) * up
                         out = h @ Wdown
```

FFN dim is 8/3 * nEmbd ≈ 2730 (vs 4 * nEmbd = 4096 for GELU).
This keeps param count similar despite the extra projection.

### Larger Vocab

2259 tokens means common words take 3-5 tokens. "conversation" might
be 5 tokens. With 8K+ vocab, common words are single tokens. Training
sees more content per step. Context window goes further.

Need to retrain BPE on the 2.5GB dataset. Takes a few hours but only
once.

### Data

`chat_input_all.txt`: 2.4M conversations, 2.5GB. Sources:
- SODA (social dialogues)
- WildChat (real user conversations)
- Anthropic HH (helpful/harmless)
- OASST2 (assistant conversations)
- ShareGPT (GPT conversations)
- DailyDialog, Cornell Movie, Blended Skill Talk, etc.

V1 trains on `chat_input.txt` (37K, 25MB) — a tiny subset.

### Dropout

V1 has none. V2 adds 0.1 dropout after attention projection and
after MLP output. Only during training. Prevents overfitting,
especially important with 140M params on "only" 2.4M conversations.

## Training Plan

1. Train BPE tokenizer (8K merges on chat_input_all.txt)
2. Init V2 model (140M params)
3. Pre-train on chat_input_all.txt (2.4M convos)
   - LR: 0.0003 (can go higher with batch 8 + proper init)
   - Warmup: 2000 steps
   - Batch: 8 (gradient accumulation)
   - Weight decay: 0.1
   - Grad clip: 1.0
   - Target loss: ~3.0
4. Save base checkpoint
5. Implement `nimllm chat` interface
6. Implement memory mechanism (sparse grad mask + elastic pull)
7. Test memory: how many facts at 140M params?

## Gradient Spike Investigation

V1 hits gradient spikes (norm 50-200K) around step 4000-6000.
Root causes identified:
- Init std 0.08 was too large (fixed: 0.02)
- Softmax backward aliasing bug (fixed)
- Missing causal mask in backward (fixed)
- Non-ASCII data (9% of docs) produces rare tokens with high loss
- Batch size 1 amplifies outlier documents

V2 mitigations:
- Larger vocab handles unicode better
- Gradient accumulation (batch 8) smooths outliers
- Dropout prevents activation growth
- Shuffled data order prevents clustering of bad docs
- Skip steps with grad norm > 50

## Timeline

V2 can start training once V1 proves the memory experiments work.
If V1 at 103M shows persistent memory through selective retraining,
V2 is the production version with better architecture and more data.

If V1 memory fails, we reconsider the approach before scaling V2.
