## ag_forward.nim — Autograd-enabled forward pass
##
## Builds a computation graph during forward. backward() on the
## loss node propagates gradients through the entire graph.
## Then Adam updates the weights. The model learns.

import gpu, gpu_model, autograd
import std/math

proc agTransformerBlock(x: Node, layer: GpuLayer, ropeCos, ropeSin: GpuBuf,
                        seqLen: int): Node =
  ## One transformer block with autograd tracking.
  let n = nEmbd
  let wq = paramNode(layer.attnWq)
  let wk = paramNode(layer.attnWk)
  let wv = paramNode(layer.attnWv)
  let wo = paramNode(layer.attnWo)
  let fc1 = paramNode(layer.mlpFc1)
  let fc2 = paramNode(layer.mlpFc2)
  let ln1 = paramNode(layer.ln1)
  let ln2 = paramNode(layer.ln2)

  # Attention sub-block
  let xn = agRmsNormAffine(x, ln1, seqLen, n)
  let q = agMatmul(wq, xn, seqLen, n, n)
  let k = agMatmul(wk, xn, seqLen, n, n)
  let v = agMatmul(wv, xn, seqLen, n, n)
  let attnOut = agAttention(q, k, v, ropeCos, ropeSin, seqLen)
  let projected = agMatmul(wo, attnOut, seqLen, n, n)
  let afterAttn = agAdd(x, projected)

  # MLP sub-block
  let xn2 = agRmsNormAffine(afterAttn, ln2, seqLen, n)
  let hidden = agGelu(agMatmul(fc1, xn2, seqLen, ffnDim, n))
  let mlpOut = agMatmul(fc2, hidden, seqLen, n, ffnDim)
  agAdd(afterAttn, mlpOut)

proc agForward*(m: var GpuModel, tokenIds: seq[int32], seqLen: int): Node =
  ## Full forward pass with autograd. Returns logits node.
  let n = nEmbd
  let wte = paramNode(m.wte)
  let embedNorm = paramNode(m.embedNorm)
  let finalNorm = paramNode(m.finalNorm)

  # Embedding
  var x = agEmbed(wte, tokenIds, seqLen, n)

  # Post-embed norm
  x = agRmsNormAffine(x, embedNorm, seqLen, n)

  # Transformer layers
  for i in 0 ..< nLayer:
    x = agTransformerBlock(x, m.layers[i], m.ropeCos, m.ropeSin, seqLen)

  # Final norm + project to vocab (weight-tied)
  x = agRmsNormAffine(x, finalNorm, seqLen, n)
  agMatmul(wte, x, seqLen, m.vocabSize, n)

proc agComputeLoss*(m: var GpuModel, tokens: seq[int32]): (Node, float32) =
  ## Forward + cross-entropy loss with autograd graph.
  ## Returns (loss_node, loss_value).
  let seqLen = min(blockSize, tokens.len - 1)
  if seqLen < 2: return (nil, 0.0f)

  let logits = agForward(m, tokens[0 ..< seqLen], seqLen)

  # Per-position fused cross-entropy (softmax + NLL in one op).
  # Backward is (probs - one_hot), bounded [-1, 1] — no gradient explosion.
  var losses: seq[Node]
  for i in 0 ..< seqLen:
    let logitsI = agRow(logits, i, m.vocabSize)
    let ce = agCrossEntropy(logitsI, tokens[i + 1].int, m.vocabSize)
    losses.add(ce)

  # Mean loss — sum the scalar NLL values
  # Download each to compute mean, then create a node
  var totalLoss = 0.0f
  for l in losses:
    let v = gpuDownload(l.data)
    totalLoss += v[0]
  let meanLoss = totalLoss / float32(seqLen)

  # Create a mean node that distributes gradient equally
  let meanBuf = gpuCreate(1)
  gpuUpload(meanBuf, @[meanLoss])
  let meanNode = newNode(meanBuf, 1, losses)
  let mGrad = meanNode.grad
  meanNode.backwardFn = proc() =
    let dy = gpuDownload(mGrad)
    let scale = dy[0] / float32(seqLen)
    for l in losses:
      gpuUpload(l.grad, @[scale])
  (meanNode, meanLoss)
