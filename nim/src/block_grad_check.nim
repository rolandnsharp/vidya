## block_grad_check.nim — Numerical gradient check on a full transformer block
##
## For each parameter in the block, perturb one weight by eps,
## compute the loss, perturb by -eps, compute the loss, compare
## (loss+ - loss-) / (2*eps) to the analytic gradient.
##
## This tests the ENTIRE composed backward — if any op has a bug
## in composition, it shows up here.

import gpu, gpu_model, autograd, ag_forward
import std/[strformat, math]

# Override to use 1 layer for testing
const testLayers = 1

const eps = 1e-3f

proc computeBlockLoss(m: var GpuModel, seqLen: int, tokens: seq[int32]): float32 =
  ## Forward one block + cross-entropy, return scalar loss.
  trackingEnabled = true
  let (lossNode, lossVal) = agComputeLoss(m, tokens)
  freeStepAllocations()
  lossVal

proc checkParam(m: var GpuModel, paramBuf: GpuBuf, paramName: string,
                seqLen: int, tokens: seq[int32], analyticGrad: seq[float32],
                numCheck: int = 10) =
  ## Check first numCheck elements of one parameter's gradient.
  let hostWeights = gpuDownload(paramBuf)
  var maxRel = 0.0f
  var maxIdx = 0

  for i in 0 ..< min(numCheck, hostWeights.len):
    # Perturb +eps
    var wp = hostWeights
    wp[i] += eps
    gpuUpload(paramBuf, wp)
    let lossPlus = computeBlockLoss(m, seqLen, tokens)

    # Perturb -eps
    wp[i] = hostWeights[i] - eps
    gpuUpload(paramBuf, wp)
    let lossMinus = computeBlockLoss(m, seqLen, tokens)

    # Restore
    gpuUpload(paramBuf, hostWeights)

    let numeric = (lossPlus - lossMinus) / (2.0f * eps)
    let analytic = analyticGrad[i]
    let diff = abs(analytic - numeric)
    let denom = max(max(abs(analytic), abs(numeric)), 1e-8f)
    let rel = diff / denom

    if rel > maxRel:
      maxRel = rel
      maxIdx = i

  let status = if maxRel < 0.01: "OK" elif maxRel < 0.1: "WARN" else: "FAIL"
  echo &"  {status:4} {paramName:20} max_rel_err={maxRel:.6f} at [{maxIdx}]"
  # Print first 3 elements for debugging
  let hw = gpuDownload(paramBuf)
  for i in 0 ..< min(3, hw.len):
    var wp = hw
    wp[i] += eps
    gpuUpload(paramBuf, wp)
    let lp = computeBlockLoss(m, seqLen, tokens)
    wp[i] = hw[i] - eps
    gpuUpload(paramBuf, wp)
    let lm = computeBlockLoss(m, seqLen, tokens)
    gpuUpload(paramBuf, hw)
    let num = (lp - lm) / (2.0f * eps)
    echo &"       [{i}] analytic={analyticGrad[i]:.8f} numeric={num:.8f} ratio={analyticGrad[i]/max(abs(num),1e-10f):.4f}"

when isMainModule:
  gpuInit()
  trackingEnabled = true
  echo "=== Full block gradient check ==="

  # Small model for speed
  var m = initGpuModel(100)  # tiny vocab

  # Fake tokens
  let seqLen = 4  # tiny sequence
  var tokens = newSeq[int32](seqLen + 1)
  for i in 0 ..< tokens.len: tokens[i] = int32(i mod 100)

  # Forward + backward to get analytic gradients
  echo "computing analytic gradients..."

  # Zero all grads
  var params = collectParams(m)
  for p in params: gpuZero(p[].grad)

  let (lossNode, lossVal) = agComputeLoss(m, tokens)
  echo &"  loss = {lossVal:.4f}"

  if lossNode != nil:
    backward(lossNode)

    # Check key parameters
    echo "checking gradients (10 elements each)..."

    let wteGrad = gpuDownload(m.wte.grad)
    checkParam(m, m.wte.data, "wte", seqLen, tokens, wteGrad)

    let wqGrad = gpuDownload(m.layers[0].attnWq.grad)
    checkParam(m, m.layers[0].attnWq.data, "layer0.attnWq", seqLen, tokens, wqGrad)

    let wkGrad = gpuDownload(m.layers[0].attnWk.grad)
    checkParam(m, m.layers[0].attnWk.data, "layer0.attnWk", seqLen, tokens, wkGrad)

    let wvGrad = gpuDownload(m.layers[0].attnWv.grad)
    checkParam(m, m.layers[0].attnWv.data, "layer0.attnWv", seqLen, tokens, wvGrad)

    let woGrad = gpuDownload(m.layers[0].attnWo.grad)
    checkParam(m, m.layers[0].attnWo.data, "layer0.attnWo", seqLen, tokens, woGrad)

    let fc1Grad = gpuDownload(m.layers[0].mlpFc1.grad)
    checkParam(m, m.layers[0].mlpFc1.data, "layer0.mlpFc1", seqLen, tokens, fc1Grad)

    let fc2Grad = gpuDownload(m.layers[0].mlpFc2.grad)
    checkParam(m, m.layers[0].mlpFc2.data, "layer0.mlpFc2", seqLen, tokens, fc2Grad)

    let ln1Grad = gpuDownload(m.layers[0].ln1.grad)
    checkParam(m, m.layers[0].ln1.data, "layer0.ln1", seqLen, tokens, ln1Grad)

    # Also check last layer to verify gradient flows through all 8 layers
    let wq7Grad = gpuDownload(m.layers[7].attnWq.grad)
    checkParam(m, m.layers[7].attnWq.data, "layer7.attnWq", seqLen, tokens, wq7Grad)

    let fc1_7Grad = gpuDownload(m.layers[7].mlpFc1.grad)
    checkParam(m, m.layers[7].mlpFc1.data, "layer7.mlpFc1", seqLen, tokens, fc1_7Grad)

    freeStepAllocations()

  echo "=== Done ==="
