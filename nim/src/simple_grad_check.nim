## simple_grad_check.nim — Minimal gradient check
##
## Build up complexity one op at a time:
## 1. Just embedding + matmul + cross-entropy
## 2. Add RMSNorm
## 3. Add attention
## 4. Add residual
## Each step: compare analytic vs numeric gradient.

import gpu, gpu_model, autograd
import std/[strformat, math]

const eps = 0.1f  # large eps needed for high-dimensional normalized layers

proc numericGrad(paramBuf: GpuBuf, idx: int,
                 forward: proc(): float32): float32 =
  let hw = gpuDownload(paramBuf)
  var wp = hw
  wp[idx] += eps
  gpuUpload(paramBuf, wp)
  let lp = forward()
  wp[idx] = hw[idx] - eps
  gpuUpload(paramBuf, wp)
  let lm = forward()
  gpuUpload(paramBuf, hw)
  (lp - lm) / (2.0f * eps)

when isMainModule:
  gpuInit()
  trackingEnabled = true

  let vocabSize = 20
  let seqLen = 3
  let dim = nEmbd  # 1024

  # Tiny random weight matrix [vocab, dim]
  var wHost = newSeq[float32](vocabSize * dim)
  for i in 0 ..< wHost.len: wHost[i] = 0.01f * float32(i mod 37 - 18)
  let wBuf = toGpu(wHost)
  let wGradBuf = gpuCreate(vocabSize * dim)

  var tokens = @[int32(1), int32(5), int32(3), int32(10)]  # 3 input + 1 target

  echo "=== Test 1: embedding + project + cross-entropy ==="

  # Forward: embed tokens, project to vocab, cross-entropy
  proc forwardLoss(): float32 =
    trackingEnabled = true
    let wNode = paramNode(wBuf, wGradBuf, vocabSize * dim)

    # Embed: just lookup rows from W
    let embedded = agEmbed(wNode, tokens[0 ..< seqLen], seqLen, dim)

    # Project back to vocab (weight-tied): logits = embedded @ W^T
    let logits = agMatmul(wNode, embedded, seqLen, vocabSize, dim)

    # Cross-entropy on last position only (simpler)
    let lastLogits = agRow(logits, seqLen - 1, vocabSize)
    let loss = agCrossEntropy(lastLogits, tokens[seqLen].int, vocabSize)

    let lossVal = gpuDownload(loss.data)[0]
    freeStepAllocations()
    lossVal

  # Analytic gradient
  gpuZero(wGradBuf)
  let wNode = paramNode(wBuf, wGradBuf, vocabSize * dim)
  let embedded = agEmbed(wNode, tokens[0 ..< seqLen], seqLen, dim)
  let logits = agMatmul(wNode, embedded, seqLen, vocabSize, dim)
  let lastLogits = agRow(logits, seqLen - 1, vocabSize)
  let loss = agCrossEntropy(lastLogits, tokens[seqLen].int, vocabSize)
  backward(loss)

  let analyticGrad = gpuDownload(wGradBuf)
  freeStepAllocations()

  echo "  checking 10 elements..."
  var maxRel = 0.0f
  for i in 0 ..< 10:
    let num = numericGrad(wBuf, i, forwardLoss)
    let ana = analyticGrad[i]
    let rel = abs(ana - num) / max(max(abs(ana), abs(num)), 1e-8f)
    if rel > maxRel: maxRel = rel
    echo &"    [{i}] analytic={ana:.8f} numeric={num:.8f} rel={rel:.6f}"

  let status = if maxRel < 0.01: "OK" elif maxRel < 0.1: "WARN" else: "FAIL"
  echo &"  {status}: max_rel_err = {maxRel:.6f}"
  echo ""

  echo "=== Test 2: embed + rmsnorm + project + cross-entropy ==="

  # Add a gamma parameter for RMSNorm
  var gHost = newSeq[float32](dim)
  for i in 0 ..< dim: gHost[i] = 1.0f
  let gBuf = toGpu(gHost)
  let gGradBuf = gpuCreate(dim)

  proc forwardLoss2(): float32 =
    trackingEnabled = true
    gpuZero(wGradBuf)
    gpuZero(gGradBuf)
    let wNode = paramNode(wBuf, wGradBuf, vocabSize * dim)
    let gNode = paramNode(gBuf, gGradBuf, dim)
    let embedded = agEmbed(wNode, tokens[0 ..< seqLen], seqLen, dim)
    let normed = agRmsNormAffine(embedded, gNode, seqLen, dim)
    let logits = agMatmul(wNode, normed, seqLen, vocabSize, dim)
    let lastLogits = agRow(logits, seqLen - 1, vocabSize)
    let loss = agCrossEntropy(lastLogits, tokens[seqLen].int, vocabSize)
    let v = gpuDownload(loss.data)[0]
    freeStepAllocations()
    v

  gpuZero(wGradBuf)
  gpuZero(gGradBuf)
  block:
    let wNode = paramNode(wBuf, wGradBuf, vocabSize * dim)
    let gNode = paramNode(gBuf, gGradBuf, dim)
    let embedded = agEmbed(wNode, tokens[0 ..< seqLen], seqLen, dim)
    let normed = agRmsNormAffine(embedded, gNode, seqLen, dim)
    let logits = agMatmul(wNode, normed, seqLen, vocabSize, dim)
    let lastLogits = agRow(logits, seqLen - 1, vocabSize)
    let loss = agCrossEntropy(lastLogits, tokens[seqLen].int, vocabSize)
    backward(loss)
    freeStepAllocations()

  let anaGradG = gpuDownload(gGradBuf)
  echo "  checking gamma gradient..."
  var maxRel2 = 0.0f
  proc forwardLoss2g(): float32 =
    trackingEnabled = true
    gpuZero(wGradBuf)
    gpuZero(gGradBuf)
    let wNode = paramNode(wBuf, wGradBuf, vocabSize * dim)
    let gNode = paramNode(gBuf, gGradBuf, dim)
    let embedded = agEmbed(wNode, tokens[0 ..< seqLen], seqLen, dim)
    let normed = agRmsNormAffine(embedded, gNode, seqLen, dim)
    let logits = agMatmul(wNode, normed, seqLen, vocabSize, dim)
    let lastLogits = agRow(logits, seqLen - 1, vocabSize)
    let loss = agCrossEntropy(lastLogits, tokens[seqLen].int, vocabSize)
    let v = gpuDownload(loss.data)[0]
    freeStepAllocations()
    v
  for i in 0 ..< 5:
    let num = numericGrad(gBuf, i, forwardLoss2g)
    let ana = anaGradG[i]
    let rel = abs(ana - num) / max(max(abs(ana), abs(num)), 1e-8f)
    if rel > maxRel2: maxRel2 = rel
    echo &"    [{i}] analytic={ana:.8f} numeric={num:.8f} rel={rel:.6f}"
  let status2 = if maxRel2 < 0.01: "OK" elif maxRel2 < 0.1: "WARN" else: "FAIL"
  echo &"  {status2}: max_rel_err = {maxRel2:.6f}"

  echo "=== Done ==="
