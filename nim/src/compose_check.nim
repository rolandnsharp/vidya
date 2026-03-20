import gpu, autograd
import std/[strformat, math]

const eps = 0.001f

when isMainModule:
  gpuInit()
  trackingEnabled = true

  let dim = 8
  let vocab = 10
  let seqLen = 2

  var wHost = newSeq[float32](vocab * dim)
  for i in 0 ..< wHost.len: wHost[i] = 0.05f * float32((i * 7 + 3) mod 19 - 9)
  let wBuf = toGpu(wHost)
  let wGrad = gpuCreate(vocab * dim)

  var wqHost = newSeq[float32](dim * dim)
  for i in 0 ..< wqHost.len: wqHost[i] = 0.02f * float32((i * 3 + 1) mod 11 - 5)
  let wqBuf = toGpu(wqHost)
  let wqGrad = gpuCreate(dim * dim)

  var gHost = newSeq[float32](dim)
  for i in 0 ..< dim: gHost[i] = 1.0f
  let gBuf = toGpu(gHost)
  let gGrad = gpuCreate(dim)

  var tokens = @[int32(3), int32(7), int32(1)]

  proc fwd(): float32 =
    gpuZero(wGrad); gpuZero(gGrad); gpuZero(wqGrad)
    let wN = paramNode(wBuf, wGrad, vocab * dim)
    let gN = paramNode(gBuf, gGrad, dim)
    let wqN = paramNode(wqBuf, wqGrad, dim * dim)
    let emb = agEmbed(wN, tokens[0 ..< seqLen], seqLen, dim)
    let normed = agRmsNormAffine(emb, gN, seqLen, dim)
    let hidden = agMatmul(wqN, normed, seqLen, dim, dim)
    let logits = agMatmul(wN, hidden, seqLen, vocab, dim)
    let lastLogits = agRow(logits, seqLen - 1, vocab)
    let loss = agCrossEntropy(lastLogits, tokens[seqLen].int, vocab)
    let v = gpuDownload(loss.data)[0]
    freeStepAllocations()
    v

  gpuZero(wGrad); gpuZero(gGrad); gpuZero(wqGrad)
  block:
    let wN = paramNode(wBuf, wGrad, vocab * dim)
    let gN = paramNode(gBuf, gGrad, dim)
    let wqN = paramNode(wqBuf, wqGrad, dim * dim)
    let emb = agEmbed(wN, tokens[0 ..< seqLen], seqLen, dim)
    let normed = agRmsNormAffine(emb, gN, seqLen, dim)
    let hidden = agMatmul(wqN, normed, seqLen, dim, dim)
    let logits = agMatmul(wN, hidden, seqLen, vocab, dim)
    let lastLogits = agRow(logits, seqLen - 1, vocab)
    let loss = agCrossEntropy(lastLogits, tokens[seqLen].int, vocab)
    backward(loss)
    echo "  wqGrad right after backward: ", gpuDownload(wqGrad)[0..4]
    echo "  wGrad right after backward: ", gpuDownload(wGrad)[0..4]
    # DON'T free yet — need to read param grads first

  echo "  wqGrad OUTSIDE block: ", gpuDownload(wqGrad)[0..4]
  echo "  wGrad OUTSIDE block: ", gpuDownload(wGrad)[0..4]
  # Save analytic grads BEFORE fwd() zeros them
  let wqAna = gpuDownload(wqGrad)
  let wAna = gpuDownload(wGrad)
  echo "loss = ", fwd()

  echo "Wq gradient:"
  for i in 0 ..< 5:
    let hw = gpuDownload(wqBuf)
    var wp = hw; wp[i] += eps
    gpuUpload(wqBuf, wp)
    let lp = fwd()
    wp[i] = hw[i] - eps
    gpuUpload(wqBuf, wp)
    let lm = fwd()
    gpuUpload(wqBuf, hw)
    let num = (lp - lm) / (2.0f * eps)
    let rel = abs(wqAna[i] - num) / max(max(abs(wqAna[i]), abs(num)), 1e-8f)
    echo &"  [{i}] ana={wqAna[i]:.6f} num={num:.6f} rel={rel:.4f}"

  echo "W gradient (weight-tied):"
  for i in 0 ..< 5:
    let hw = gpuDownload(wBuf)
    var wp = hw; wp[i] += eps
    gpuUpload(wBuf, wp)
    let lp = fwd()
    wp[i] = hw[i] - eps
    gpuUpload(wBuf, wp)
    let lm = fwd()
    gpuUpload(wBuf, hw)
    let num = (lp - lm) / (2.0f * eps)
    let rel = abs(wAna[i] - num) / max(max(abs(wAna[i]), abs(num)), 1e-8f)
    echo &"  [{i}] ana={wAna[i]:.6f} num={num:.6f} rel={rel:.4f}"
