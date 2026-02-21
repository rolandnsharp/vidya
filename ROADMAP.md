# Vidya Roadmap

## Current State

10M params, 12 layers, 256 dim, 37K conversations, checkpoint: `microgpt_chat_10m_v3.bin`

## Completed

### ~~1. Expand BPE vocab from 500 to 2000 merges~~ done
`n_merges = 2000` in `lib/bpe.ml`. Vocab goes from ~580 to ~2100 tokens.

### ~~2. Learnable RMSNorm + final norm~~ done
Added learnable gamma scale vectors to all norm positions. Added `final_norm`
before lm_head. +6,400 params, initialized to 1.0.

### ~~3. More and better training data~~ done
Grew from 16K to 37K conversations (24MB). Cleaned existing data, added 25K
from SODA (Allen AI) + 576 new DailyDialog lines.

### ~~4. Dropout~~ done
`batch_dropout` in `tensor.ml` (p=0.1, inverted scaling). Applied after
attention projection, after MLP output, and after embedding norm in training.

### ~~5. Remove Forth interpreter, keep symbolic ideas~~ done
Deleted `forth.ml`. Rewrote `knowledge.ml` to use plain OCaml hashtables
instead of a Forth dictionary. Symbolic constrained decoding preserved —
concept coherence, TD learning, word validation all work the same way,
just without the Forth runtime underneath.

## Up Next

### 6. Check loss curve, retrain v3 on CPU
When v1 finishes: evaluate loss curve, then retrain with v3 arch (new vocab +
learnable norms + 37K data + dropout). This is the last CPU-only run.

### 7. Get a GPU (RTX 3090)
Used RTX 3090 on eBay AU: ~AU $1,000-1,350 for 24GB VRAM. This is the gate
to everything that follows — 100M param training drops from 3+ days to hours.

### 8. CUDA backend
Port matmul and key ops to CUDA kernels. The tensor.ml autograd stays in OCaml,
just the inner BLAS calls get replaced with GPU equivalents.

### 9. Scale to 100M params
32 layers, 512 dim, 16 heads (~102M params). First model that should produce
consistently coherent sentences. Train on GPU — hours not days.

### 10. Tune symbolic layer at scale
Once the base model is coherent at 100M params, the constrained decoding
becomes genuinely useful. Tune the concept coherence, topic depth, and TD
learning parameters against the fluent base model. The symbolic layer steers
output that's already good, rather than fighting output that's uncertain.

## Notes
- Binary checkpoints are architecture-specific — dimension changes require retrain
- 10M v1 training is running now (200K steps, ~8hrs) — don't retrain until it finishes
- GPU is the critical path — everything after step 9 depends on it
- Forth is gone from the codebase; the symbolic ideas live on in pure OCaml
