import gpu
import std/[strformat, math]

when isMainModule:
  gpuInit()

  let dim = 4
  let vocab = 5
  var x = @[0.1f, 0.2f, 0.3f, 0.4f]  # one token embedding
  var gamma = @[1.0f, 1.0f, 1.0f, 1.0f]
  var w = newSeq[float32](vocab * dim)
  for i in 0 ..< w.len: w[i] = 0.1f * float32(i - 10)
  
  proc fwdLoss(xh, gh, wh: seq[float32], target: int): float32 =
    let xBuf = toGpu(xh)
    let gBuf = toGpu(gh)
    let wBuf = toGpu(wh)
    let yBuf = gpuCreate(dim)
    let rmsBuf = gpuCreate(1)
    rmsnormAffineFwd(xBuf, gBuf, yBuf, rmsBuf, 1, dim)
    # logits = y @ W^T
    let logits = gpuCreate(vocab)
    gpuSgemm(2, 1, vocab, dim, yBuf, wBuf, logits)
    # softmax + nll
    let probs = gpuCreate(vocab)
    softmaxFwd(logits, probs, 1, vocab)
    let p = gpuDownload(probs)
    -ln(max(p[target], 1e-10f))
  
  let target = 2
  let loss0 = fwdLoss(x, gamma, w, target)
  echo &"base loss: {loss0:.6f}"
  
  # Numeric gradient for gamma
  echo "gamma gradient:"
  let eps = 0.01f
  for i in 0 ..< dim:
    var gp = gamma; gp[i] += eps
    var gm = gamma; gm[i] -= eps
    let lp = fwdLoss(x, gp, w, target)
    let lm = fwdLoss(x, gm, w, target)
    let grad = (lp - lm) / (2.0f * eps)
    echo &"  gamma[{i}] numeric_grad = {grad:.6f}"
