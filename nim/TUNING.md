# Tuning Notes

What we learned getting 103M params to train stably on a single RTX 3060.
Written during the session so nothing is lost.

## The Init Std Bug

The single biggest issue. Weight initialization standard deviation was 0.08
(carried over from the OCaml 10M model). At 103M params this produced
gradient norms of 195,000 — so extreme that gradient clipping at 1.0
effectively zeroed the learning signal.

**Fix:** Changed init std to 0.02 (matches nanoGPT). Gradient norms dropped
to 3-13. Loss immediately started dropping. This one change took us from
"explodes every 300 steps" to "trains stably for hours."

**Lesson:** Init std should scale with model size. 0.02 is standard for
100M+ models. The OCaml version worked at 0.08 because it was smaller (49M)
and used float64 which hides numerical issues.

## The Softmax Backward Aliasing Bug

The attention backward computed softmax gradients in-place — writing `dx`
into the same buffer as `dy` while the kernel was still reading `dy`. This
produced subtly wrong gradients that accumulated over layers and exploded
after a few hundred steps.

**Fix:** Separate input and output buffers for softmax backward. Added
`gpu_zero_upper` kernel to properly zero the causal mask in the backward
gradient matrix.

## The Fused Cross-Entropy

We originally computed softmax and NLL as separate autograd operations.
The NLL backward produces a gradient of `-1/p` where `p` is the predicted
probability. For an untrained model with vocab 2259, most probs are ~0.0004,
giving gradients of ~2500 per element. This made the gradient norm enormous
even when the actual learning signal was fine.

**Fix:** Fused softmax + NLL into a single `agCrossEntropy` operation where
the backward is simply `probs - one_hot`, bounded between -1 and 1. This is
what PyTorch's `CrossEntropyLoss` does internally.

## Learning Rate

Tested values:
- 0.001: explodes immediately (step ~100)
- 0.0006: explodes at step ~150
- 0.0003: explodes at step ~200
- 0.0001: stable with 0.02 init std, loss drops to 4.7+

nanoGPT uses 0.0006 but with batch size 480. We're at batch size 1. The
effective learning rate scales with batch size — our lr=0.0001 at batch 1
is equivalent to their lr=0.0006 at batch ~6.

Could go higher with gradient accumulation (batch size 8) but we saw
explosions even at lr=0.00015 with batch 8, likely because the softmax
aliasing bug was still present during those tests. Worth retesting with
all fixes in place.

## AdamW vs Adam

Plain Adam without weight decay caused weights to grow unbounded — the
model would train for 300-600 steps then suddenly explode. AdamW with
weight_decay=0.1 prevents this by pulling weights toward zero each step.
Keep weight decay at 0.1.

## Gradient Clipping

We clip the global gradient norm to 1.0. The kernel computes sum of squares
across all 103M parameters using atomicAdd in shared memory blocks, then
scales all gradients if the norm exceeds the threshold. Verified working
independently (clip_test.nim).

With proper init std (0.02), gradient norms are naturally 3-13, so clipping
is mild. Without it, rare large gradients could still cause issues.

## What We Haven't Tried Yet

**Gradient accumulation with all fixes.** The softmax bug and init std were
both broken during our batch-8 experiments. With both fixed, batch size 8
at lr=0.0003 might work and would converge faster.

**Higher learning rate.** Now that gradients are healthy (norm 3-13), we
might be able to push lr to 0.0003 or even 0.0006. The model plateaued
around loss 4.7 at lr=0.0001 — higher lr would push past this.

**Learning rate warmup is 2000 steps.** Could try shorter (500) to reach
peak lr faster, or longer (5000) for more stability.

**Dropout.** Not currently applied in the Nim version. Would help prevent
overfitting on the 37K conversation dataset, which is small for 103M params.

**Larger vocab.** Our BPE has 2259 tokens (2000 merges). nanoGPT uses 50K+.
Larger vocab means shorter sequences and more efficient training. Would need
to retrain the tokenizer.

**More data.** 37K conversations (25MB) is thin for 103M params. We have
additional HuggingFace datasets (soda, wildchat, sharegpt, anthropic_hh) in
the data/ directory ready to convert. More data → less overfitting → lower
final loss.

**Mixed precision.** Currently pure float32. Using float16 for forward and
float32 for accumulation would halve VRAM and roughly double throughput.

**Flash attention.** Our attention downloads scores to CPU for causal softmax
in the OCaml version. The Nim version does it on GPU but uses separate
per-head buffers. A fused flash attention kernel would be significantly faster.

## Hyperparameter Summary (Current Working Config)

```
init_std       = 0.02
learning_rate  = 0.0001
min_lr         = 0.00001
beta1          = 0.9
beta2          = 0.95
weight_decay   = 0.1
warmup_steps   = 2000
grad_clip_norm = 1.0
grad_accum     = 1 (batch size 1)
```

## Training Progress

```
step   100: loss 9.45  (random baseline ~7.7)
step   500: loss 6.89
step  1000: loss 5.73
step  1500: loss 5.23
step  2000: loss 5.30  (warmup complete, lr at peak)
step  2500: loss 5.07  (first checkpoint saved)
step  3000: loss 5.04
step  3500: loss 4.80
```

Loss needs to reach ~3.5 for coherent words, ~3.0 for basic conversation.
