# Vidya Dev Log

## 2026-02-22 — Forth removed, symbolic layer reimplemented in pure OCaml

Deleted `forth.ml` entirely. The Forth interpreter was being used as nothing
more than a hashtable — the stack operations, arithmetic primitives, execution
engine, and validation logic were never called by the symbolic layer.

Rewrote `knowledge.ml` to store concepts in plain OCaml data structures:

```ocaml
type concept = {
  mutable associations : (string * float) list;
  strength : float;
  token_ids : int list;
}

type t = {
  concepts : (string, concept) Hashtbl.t;
  token_to_concepts : (int, string list) Hashtbl.t;
  concept_to_tokens : (string, int list) Hashtbl.t;
}
```

Updated `symbolic.ml` to reference `Knowledge.t` directly instead of going
through `Forth.dictionary`. All five constraints preserved:
1. Repetition penalty (unchanged)
2. Word boundary bias (unchanged)
3. Word validation (unchanged)
4. Concept coherence — now calls `Knowledge.associations` instead of `Forth.associations`
5. Topic depth penalty — uses `knowledge.concept_to_tokens` directly

TD learning preserved — `Knowledge.update_association_weight` replaces
`Forth.update_association_weight`. Same logic, simpler types.

Renamed `forth_know` to `concept_know` throughout `generate.ml` and `main.ml`.

Project goal refocused: best local model we can train, no microprocessor
deployment target. The framework is the product.

---

## 2026-02-22 — Rethinking Forth and the symbolic layer

### What we're changing

The Forth-based symbolic layer (TD learning, concept dictionaries, constrained
decoding) is being pulled out of the core training/inference path. At 10M params
the base model can barely form coherent sentences — the symbolic constraints are
fighting uncertainty rather than steering fluent output. It's pushing on a rope.

### What we learned

The Forth/symbolic work wasn't wasted. Key ideas to carry forward:

**Constrained decoding works in principle.** The `Symbolic.context` in
`generate.ml` that adjusts logits before sampling is a clean interface. When the
base model is coherent enough (100M+ params), this becomes genuinely useful for
steering output toward factual consistency, domain rules, and style constraints.

**The dictionary-as-knowledge pattern is sound.** Forth's flat dictionary model —
named entries, compositional definitions, instant lookup — is a good mental model
for how symbolic knowledge should integrate with a neural model. The specific
implementation (TD-weighted associations, Forth evaluation) was premature, but the
architecture pattern of "neural generates, symbolic constrains" is right.

**Image persistence is the killer feature for embedded.** From the Forth9 work:
a model that persists in the OS image and wakes up with the device is fundamentally
different from a model you launch as an application. This shapes how we think about
the Pico deployment — the chatbot is a resident, not a program.

### Forth as deployment target, not core architecture

The Forth inference engine stays as a deployment option for the PicoCalc. But the
core Vidya development happens in OCaml. The symbolic ideas get reimplemented
cleanly in OCaml when we're ready (post-100M params), without the Forth runtime
in the loop.

For the PicoCalc (Pico Plus 2W with 8MB PSRAM), the use case isn't a keyboard
autocomplete gadget — it's a **resident chatbot inside Forth9**:

- 4MB model budget → 2-4M params in INT8, or up to 8M in INT4
- 2-4 layers, 128 dim — small but real language model
- Trained on personal/domain corpus using the same OCaml framework
- Runs as a native Forth9 word, persists in the image
- Can interact with the Forth9 environment (sensors, mesh peers, files)
- Neural side handles language, Forth9 environment provides facts and actions
- The neurosymbolic split happens naturally: model generates, OS acts

### What to remove

- `lib/symbolic.ml` — TD learning, concept associations. Premature at 10M scale.
- `lib/knowledge.ml` — Forth dictionary wrapper. Same reasoning.
- `lib/forth.ml` — Forth evaluator in the training loop. Not needed.
- `forth/` directory — keep for reference but decouple from the build.
- TD weights file (`td_weights.bin`) — artifact of the current symbolic layer.
- References in `generate.ml` and `main.ml` to symbolic/knowledge/forth.

Keep the `Symbolic.context` interface pattern — we'll want something like it
later when the base model deserves steering.

### What stays

- `tensor.ml`, `model.ml`, `forward.ml`, `train.ml` — the core framework
- `bpe.ml` — tokenizer
- `generate.ml` — text generation (simplified without symbolic hooks)
- The architecture: pre-norm transformer, learnable RMSNorm, dropout, BLAS
- The training pipeline: OCaml train → quantize → export → deploy

---

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

### v1 training loss curve (baseline)

500 merges, 580 vocab, 16K docs, no learnable norms, no dropout.

```
step  2500  loss 4.7787
step  5000  loss 3.7486
step  7500  loss 3.0879
step 10000  loss 2.8329
step 12500  loss 2.6329
step 15000  loss 2.5097
step 17500  loss 2.4422
step 20000  loss 2.3721
step 22500  loss 2.3212
step 25000  loss 2.2789
step 27500  loss 2.2418
step 30000  loss 2.2101
```

Fast drop to ~2.5 in first 15K steps, then gradual descent. Starting to flatten
around 2.2 at 30K (15% through). Killed at step 40K (loss 2.1125) — curve had
told us what it was going to tell us. Plateau would have been ~1.9-2.0.

### v3 training loss curve (in progress)

2000 merges, 2188 vocab, 37K docs, learnable norms, dropout p=0.1.

```
step  2500  loss 5.1794
step  5000  loss 4.0907
step  7500  loss 3.8123
step 10000  loss 3.6526
step 12500  loss 3.5279
step 15000  loss 3.4533
step 17500  loss 3.3583
step 20000  loss 3.2915
step 22500  loss 3.2455
step 25000  loss 3.1927
step 27500  loss 3.1621
step 30000  loss 3.1022
step 32500  loss 3.0888
step 35000  loss 3.0437
step 37500  loss 3.0128
step 40000  loss 2.9762
step 42500  loss 2.9389
step 45000  loss 2.9071
```

Higher absolute loss than v1 at same steps — expected because vocab is 3.6x
larger (2188 vs 606). Relative to random baseline (ln(vocab)):

    v1: 2.11 / ln(606) = 33% of random
    v3: 2.91 / ln(2188) = 38% of random (at 45K, still descending)

Descent rate per 10K steps: -1.53, -0.36, -0.19, -0.13, -0.14. Not flattening
yet at 45K — v1 was already plateauing at 40K. Each token covers 4.2 chars
(vs 2.8), so the model is learning bigger semantic chunks.

## Scaling to 100M params — planning notes

### Target configs

| Config | Layers | Dim | Heads | Params |
|--------|--------|-----|-------|--------|
| A: Deep | 32 | 512 | 16 | ~102M |
| B: Wide | 20 | 640 | 20 | ~100M |

Config A (deep) is more standard and likely better for language at this scale.

### Code changes

Just 3 constants in `model.ml`:
```ocaml
let n_layer = 32    (* was 12 *)
let n_embd = 512    (* was 256 *)
let n_head = 16     (* was 8 *)
```

Everything else is parametric — tensor ops, forward pass, optimizer, BPE, generate
all adapt automatically. May need to lower LR from 0.001 to 0.0003-0.0005 for
stability at 32 layers.

### Resource estimates

- **Training time (CPU):** ~80hrs (3.3 days) for 200K steps, ~8 days for 500K
- **Memory:** ~3.2GB params+grads+Adam, ~5-8GB total with activations
- **GPU estimate:** single GPU would cut 3 days → 3-6 hours

### Bottleneck

Wall time. 3+ day CPU runs mean slow iteration. GPU (CUDA backend, stage 8 in
PLAN.md) is the real unlock for 100M+ scale.
