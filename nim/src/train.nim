## train.nim — Training loop with Adam and memory mechanism
##
## The sparse gradient mask + elastic pull is our "frontal cortex"
## retraining. At 103M params, the top 1% is ~1M weights per
## interaction — enough room for persistent memories.

import arraymancer
import model, forward
import std/[math, times, strformat]

const
  learningRate = 0.0003f
  beta1 = 0.9f
  beta2 = 0.999f
  epsAdam = 1e-8f
  maxGradNorm = 1.0f
  warmupSteps = 5000

# ── Learning rate schedule ────────────────────────────────────────

proc getLr(step, numSteps: int): float32 =
  if step < warmupSteps:
    learningRate * float32(step) / float32(warmupSteps)
  else:
    let progress = float32(step - warmupSteps) / float32(numSteps - warmupSteps)
    learningRate * 0.5f * (1.0f + cos(PI.float32 * progress))

# ── Loss ──────────────────────────────────────────────────────────

proc softmaxRows(logits: Tensor[float32]): Tensor[float32] =
  ## Row-wise softmax for [S, vocab] tensor.
  let rows = logits.shape[0]
  let cols = logits.shape[1]
  result = zeros[float32](rows, cols)
  for i in 0 ..< rows:
    let row = logits[i, _].squeeze(0)
    let maxVal = row.max()
    let expRow = map_inline(row -. maxVal): exp(x)
    let s = expRow.sum()
    result[i, _] = (expRow / s).unsqueeze(0)

proc crossEntropyLoss*(logits: Tensor[float32], targets: seq[int]): float32 =
  ## Average NLL loss across all positions.
  let probs = softmaxRows(logits)
  let seqLen = targets.len
  var totalLoss = 0.0f
  for i in 0 ..< seqLen:
    let p = probs[i, targets[i]]
    totalLoss += -ln(max(p, 1e-10f))
  totalLoss / float32(seqLen)

# ── Checkpoint save/load ──────────────────────────────────────────

proc saveCheckpoint*(m: Model, filename: string) =
  ## Save model weights to binary file.
  echo &"saving checkpoint to {filename}..."
  # TODO: implement binary save with Arraymancer's write_npy or custom format
  echo "  (checkpoint save not yet implemented)"

proc loadCheckpoint*(m: var Model, filename: string) =
  ## Load model weights from binary file.
  echo &"loading checkpoint from {filename}..."
  # TODO: implement binary load
  echo "  (checkpoint load not yet implemented)"

# ── Training loop ─────────────────────────────────────────────────

proc formatDuration(secs: float): string =
  let h = int(secs) div 3600
  let m = (int(secs) mod 3600) div 60
  let s = int(secs) mod 60
  if h > 0: &"{h}h{m:02d}m{s:02d}s"
  elif m > 0: &"{m}m{s:02d}s"
  else: &"{s}s"

proc train*(m: var Model, tokenizedDocs: seq[seq[int]], numSteps: int) =
  ## Main training loop. No autograd — manual forward + numerical gradients
  ## for now. Will add Arraymancer autograd in next iteration.
  ##
  ## For now this is inference-only validation: compute forward pass and
  ## loss to verify the architecture works, then add backprop.
  echo &"training for {numSteps} steps on {tokenizedDocs.len} docs"

  let tStart = cpuTime()
  var lossSum = 0.0f
  let logInterval = 100

  for step in 0 ..< min(numSteps, tokenizedDocs.len):
    let tokens = tokenizedDocs[step mod tokenizedDocs.len]
    let seqLen = min(blockSize, tokens.len - 1)
    if seqLen < 2: continue

    # Forward pass
    let inputTokens = tokens[0 ..< seqLen]
    let targetTokens = tokens[1 .. seqLen]
    let logits = forward(m, inputTokens, seqLen)

    # Compute loss
    let loss = crossEntropyLoss(logits, targetTokens)
    lossSum += loss

    if (step + 1) mod logInterval == 0:
      let elapsed = cpuTime() - tStart
      let stepsDone = float(step + 1)
      let stepsLeft = float(numSteps - step - 1)
      let eta = elapsed * stepsLeft / stepsDone
      let avgLoss = lossSum / float32(logInterval)
      echo &"step {step + 1:>5} / {numSteps} | loss {avgLoss:.4f} | {formatDuration(elapsed)} elapsed | {formatDuration(eta)} remaining"
      lossSum = 0.0f
