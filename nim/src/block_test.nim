## block_test.nim — Test one transformer block forward+backward
##
## If this explodes, the bug is inside the block.
## If this is stable, the bug is in how blocks chain.

import gpu, gpu_model, autograd, ag_forward
import std/[strformat, math]

when isMainModule:
  gpuInit()
  trackingEnabled = true
  echo "testing single transformer block..."

  var m = initGpuModel(100)  # tiny vocab for testing

  # Create a fake input: [seqLen=8, nEmbd=1024]
  let seqLen = 8
  var xHost = newSeq[float32](seqLen * nEmbd)
  for i in 0 ..< xHost.len: xHost[i] = 0.01f * float32(i mod 100)
  let xBuf = toGpu(xHost)
  let xNode = newNode(xBuf, seqLen * nEmbd)

  # Run 100 forward+backward steps, check if gradients stay bounded
  for step in 0 ..< 100:
    # Zero xNode grad
    gpuZero(xNode.grad)

    let wq = paramNode(m.layers[0].attnWq)
    let wk = paramNode(m.layers[0].attnWk)
    let wv = paramNode(m.layers[0].attnWv)
    let wo = paramNode(m.layers[0].attnWo)
    let fc1 = paramNode(m.layers[0].mlpFc1)
    let fc2 = paramNode(m.layers[0].mlpFc2)
    let ln1 = paramNode(m.layers[0].ln1)
    let ln2 = paramNode(m.layers[0].ln2)

    # Attention sub-block
    let xn = agRmsNormAffine(xNode, ln1, seqLen, nEmbd)
    let q = agMatmul(wq, xn, seqLen, nEmbd, nEmbd)
    let k = agMatmul(wk, xn, seqLen, nEmbd, nEmbd)
    let v = agMatmul(wv, xn, seqLen, nEmbd, nEmbd)
    let attnOut = agAttention(q, k, v, m.ropeCos, m.ropeSin, seqLen)
    let projected = agMatmul(wo, attnOut, seqLen, nEmbd, nEmbd)
    let afterAttn = agAdd(xNode, projected)

    # MLP sub-block
    let xn2 = agRmsNormAffine(afterAttn, ln2, seqLen, nEmbd)
    let hidden = agGelu(agMatmul(fc1, xn2, seqLen, ffnDim, nEmbd))
    let mlpOut = agMatmul(fc2, hidden, seqLen, nEmbd, ffnDim)
    let output = agAdd(afterAttn, mlpOut)

    # Fake loss: sum of output
    let outCpu = gpuDownload(output.data)
    var lossVal = 0.0f
    for v in outCpu: lossVal += v
    lossVal /= float32(outCpu.len)

    # Seed gradient — use the grad buffer's actual size
    echo &"  output.numel={output.numel} grad.numel={output.grad.numel}"
    var seed = newSeq[float32](output.grad.numel)
    for i in 0 ..< seed.len: seed[i] = 1.0f / float32(seed.len)
    gpuUpload(output.grad, seed)

    backward(output)

    # Check gradient norms
    let xGrad = gpuDownload(xNode.grad)
    var xNorm = 0.0f
    for v in xGrad: xNorm += v * v
    xNorm = sqrt(xNorm)

    let wqGrad = gpuDownload(m.layers[0].attnWq.grad)
    var wqNorm = 0.0f
    for v in wqGrad: wqNorm += v * v
    wqNorm = sqrt(wqNorm)

    echo &"step {step:>3} | loss {lossVal:.4f} | x_grad_norm {xNorm:.4f} | wq_grad_norm {wqNorm:.6f}"

    if xNorm != xNorm or wqNorm != wqNorm:
      echo "NaN detected!"
      break
    if xNorm > 1e6 or wqNorm > 1e6:
      echo "Gradient explosion!"
      break

    freeStepAllocations()
    # Zero param grads for next step
    gpuZero(m.layers[0].attnWq.grad)
    gpuZero(m.layers[0].attnWk.grad)
    gpuZero(m.layers[0].attnWv.grad)
    gpuZero(m.layers[0].attnWo.grad)
    gpuZero(m.layers[0].mlpFc1.grad)
    gpuZero(m.layers[0].mlpFc2.grad)
    gpuZero(m.layers[0].ln1.grad)
    gpuZero(m.layers[0].ln2.grad)

  echo "done"
