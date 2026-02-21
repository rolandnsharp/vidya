# Vidya Roadmap

## Current State

10M params, 12 layers, 256 dim, 37K conversations, checkpoint: `microgpt_chat_10m_v3.bin`

## Completed

### ~~1. Expand BPE vocab from 500 to 2000 merges~~ ✓
Done. `n_merges = 2000` in `lib/bpe.ml`. Vocab goes from ~580 → ~2100 tokens.
New checkpoint file (`_v2`) since old one is incompatible. Expect ~4-5 chars/token
(was ~2.8) and slightly higher param count (~9.98M vs 9.59M).

### ~~2. Learnable RMSNorm + final norm~~ ✓
RMSNorm was already in use (pre-norm pattern). Added learnable γ scale vectors
to all norm positions (`rmsnorm_affine` / `batch_rmsnorm_affine` in tensor.ml).
Added `ln1`/`ln2` per layer, `embed_norm`, and `final_norm` before lm_head.
+6,400 params (25 × 256), initialized to 1.0. Checkpoint bumped to `_v3`.

## Up Next (while training runs)

### ~~3. More and better training data~~ ✓
Grew from 16K → 37K conversations (24MB, up from 7MB). Cleaned existing data
(removed 4,276 duplicates + 77 trivial lines), added 25K from SODA (Allen AI)
+ 576 new DailyDialog lines. Avg turn depth improved (more 3-5 turn convos),
avg line length up from 465 → 676 chars. Backup at `chat_input_backup.txt`.

### ~~4. Dropout~~ ✓
Added `batch_dropout` to `tensor.ml` (p=0.1, inverted scaling, autograd-aware).
Applied after attention projection, after MLP output, and after embedding norm
in the training forward pass. Training flag `Tensor.training` toggles on/off —
set true in `train.ml`, false in `generate.ml`. No new params, no checkpoint change.
KV cache already exists for inference — nothing to add there.

## After current training finishes

### 5. Check loss curve, decide next run
If loss is still dropping at 200K steps, consider extending to 500K-1M. But by
then RMSNorm will be ready, so it may be better to retrain with the new arch
instead of continuing the old one. Evaluate.

### 6. Scale to 50-100M params
512 dim, 16-24 layers. CPU-trainable but takes days. Where coherent sentences
emerge consistently.

### 7. Strengthen the symbolic layer
Expand Forth dictionary, add concept relationships, tune TD reward signal.
Only valuable once base model produces semi-coherent text.

## Notes
- Vocab + data quality are the biggest levers at this scale
- A 10M model with 2K vocab + 50K clean docs beats a 50M model with 580 vocab + 16K docs
- Binary checkpoints are architecture-specific — dimension changes require retrain
- 10M v1 training is running now (200K steps, ~8hrs) — don't retrain until it finishes
- When it does: evaluate loss curve, then retrain with new vocab + learnable norms + bigger dataset
