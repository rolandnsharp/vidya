## Test: two matmuls, NO weight tying. Separate W_embed and W_proj.
import gpu, autograd
import std/[strformat, math]

const eps = 0.0001f

when isMainModule:
  gpuInit()
  trackingEnabled = true

  let dim = 4
  let vocab = 10
  let seqLen = 2

  # Separate embed and projection weights
  var weHost = newSeq[float32](vocab * dim)
  for i in 0 ..< weHost.len: weHost[i] = 0.05f * float32((i * 7 + 3) mod 19 - 9)
  let weBuf = toGpu(weHost)
  let weGrad = gpuCreate(vocab * dim)

  var wpHost = newSeq[float32](vocab * dim)
  for i in 0 ..< wpHost.len: wpHost[i] = 0.03f * float32((i * 11 + 5) mod 23 - 11)
  let wpBuf = toGpu(wpHost)
  let wpGrad = gpuCreate(vocab * dim)

  var wqHost = newSeq[float32](dim * dim)
  for i in 0 ..< wqHost.len: wqHost[i] = 0.02f * float32((i * 3 + 1) mod 11 - 5)
  let wqBuf = toGpu(wqHost)
  let wqGrad = gpuCreate(dim * dim)

  var tokens = @[int32(3), int32(7), int32(1)]

  proc fwd(): float32 =
    gpuZero(weGrad); gpuZero(wpGrad); gpuZero(wqGrad)
    let weN = paramNode(weBuf, weGrad, vocab * dim)
    let wpN = paramNode(wpBuf, wpGrad, vocab * dim)
    let wqN = paramNode(wqBuf, wqGrad, dim * dim)
    let emb = agEmbed(weN, tokens[0 ..< seqLen], seqLen, dim)
    let hidden = agMatmul(wqN, emb, seqLen, dim, dim)
    let logits = agMatmul(wpN, hidden, seqLen, vocab, dim)
    let lastLogits = agRow(logits, seqLen - 1, vocab)
    let loss = agCrossEntropy(lastLogits, tokens[seqLen].int, vocab)
    let v = gpuDownload(loss.data)[0]
    freeStepAllocations()
    v

  # Analytic
  gpuZero(weGrad); gpuZero(wpGrad); gpuZero(wqGrad)
  block:
    let weN = paramNode(weBuf, weGrad, vocab * dim)
    let wpN = paramNode(wpBuf, wpGrad, vocab * dim)
    let wqN = paramNode(wqBuf, wqGrad, dim * dim)
    let emb = agEmbed(weN, tokens[0 ..< seqLen], seqLen, dim)
    let hidden = agMatmul(wqN, emb, seqLen, dim, dim)
    let logits = agMatmul(wpN, hidden, seqLen, vocab, dim)
    let lastLogits = agRow(logits, seqLen - 1, vocab)
    let loss = agCrossEntropy(lastLogits, tokens[seqLen].int, vocab)
    backward(loss)

  let wqAna = gpuDownload(wqGrad)
  echo "loss = ", fwd()

  echo "Wq gradient (no weight tying):"
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
    echo &"  [{i}] ana={wqAna[i]:.6f} num={num:.6f} ratio={wqAna[i]/max(abs(num),1e-10f):.4f}"
