# Vidya Dev Log

## 2026-02-21 — Architecture overhaul (pre-retrain)

Four changes made while the 10M v1 model trains (200K steps, ~8hrs). All require
retraining from scratch, so we batched them together for the next run.

### 1. BPE vocab: 500 → 2000 merges

`lib/bpe.ml` — single constant change (`n_merges = 2000`).

Everything downstream is dynamic: vocab_size computed at runtime, wte sized by
`tok.vocab_size`, special tokens shift automatically. Pair counting array goes
from ~2.7MB to ~35MB (fine).

Expected: ~2100 vocab (was ~580), ~4-5 chars/token (was ~2.8), +390K wte params.

### 2. Learnable RMSNorm + final norm

RMSNorm was already implemented and used (pre-norm pattern). Two improvements:

- **Learnable γ scale vectors** — `rmsnorm_affine` / `batch_rmsnorm_affine` in
  `tensor.ml`. Forward: `y = (x / rms) * γ`. Backward propagates to both x and γ.
  Standard in LLaMA, GPT-NeoX, etc.
- **Final norm before lm_head** — was missing. Added `model.final_norm` applied
  after all layers, before the output projection. LLaMA-standard architecture.

New params in `model.ml`: `ln1`/`ln2` per layer (24 × 256), `embed_norm` (256),
`final_norm` (256) = 6,656 total. All initialized to 1.0 (identity at start).

`collect_params` updated to include norm weights for checkpointing and Adam.

### 3. Training data: 16K → 37K conversations

Cleaned existing data: removed 4,276 exact duplicates and 77 trivial lines
(under 60 chars). Discovered existing corpus was ~95% DailyDialog already.

Added ~25K conversations from SODA (Allen AI) — diverse social dialogues with
richer multi-turn structure. Avg line length up from 465 → 676 chars.

Final: 37,492 unique conversations, 24MB (was 7MB). Shuffled deterministically.
Original backed up at `chat_input_backup.txt`.

### 4. Dropout (p=0.1)

`batch_dropout` in `tensor.ml` — inverted scaling (survivors × 1/(1-p)), mask
stored in closure for backward pass. No-op when `Tensor.training = false`.

Applied at 3 points in the training forward pass (`forward.ml`):
1. After embedding norm
2. After attention output projection (before residual add)
3. After MLP output (before residual add)

Inference path untouched. `Tensor.training` flag set true in `train.ml`,
false in all `generate.ml` functions.

No learnable parameters, no checkpoint impact.

### Checkpoint

Bumped to `microgpt_chat_10m_v3.bin` (incompatible with v1/v2 due to new norm
params in `collect_params` ordering).

### Net effect on param count

| Component | v1 | v3 |
|-----------|----|----|
| wte (+ lm_head tied) | 580 × 256 = 148K | ~2100 × 256 = 538K |
| Layer weights (12 layers) | 12 × 6 × 256² = 4,718K | same |
| Norm γ vectors | 0 | 25 × 256 = 6.4K |
| **Total** | **~9.59M** | **~9.99M** |
