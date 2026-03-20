## ag_forward_v2.nim — V2 forward pass with GQA + SwiGLU
##
## GQA: 16 Q heads share 4 KV heads. Each KV head serves 4 Q heads.
## SwiGLU: gate + up projections with swish gating instead of GELU.
## Everything else same as V1.

import gpu, model_v2, autograd
import std/math

var trainingMode* = true

# ── Dropout ───────────────────────────────────────────────────────

proc agDropout*(x: Node, rate: float32 = 0.1f): Node =
  ## Randomly zero elements during training. No-op during inference.
  if not trainingMode or rate == 0.0f:
    return x
  let n = x.numel
  let mask = trackedCreate(n)
  let yBuf = trackedCreate(n)
  gpu_dropout_fwd(x.data.data, yBuf.data, mask.data, rate, cint(n))
  result = newNode(yBuf, n, @[x])
  let yGrad = result.grad
  result.backwardFn = proc() =
    gpu_dropout_bwd(yGrad.data, mask.data, x.grad.data, cint(n))

# ── GQA Attention ─────────────────────────────────────────────────

proc agGqaAttention*(q, k, v: Node, ropeCos, ropeSin: GpuBuf,
                     seqLen: int): Node =
  ## Grouped Query Attention. q is [S, nEmbd], k/v are [S, nKvHead*headDim].
  ## Each Q head attends to its corresponding KV group.
  let n = nEmbdV2
  let hd = headDimV2
  let nKvDim = nKvHeadV2 * hd  # total KV dimension
  let scale = 1.0f / sqrt(float32(hd))

  # RoPE on Q (full nEmbd) and K (nKvDim)
  let qRot = trackedCreate(seqLen * n)
  gpuCopy(q.data, qRot, seqLen * n)
  let kRot = trackedCreate(seqLen * nKvDim)
  gpuCopy(k.data, kRot, seqLen * nKvDim)
  # RoPE on Q — standard, all heads
  ropeFwd(qRot, ropeCos, ropeSin, seqLen, n, nHeadV2, hd)
  # RoPE on K — fewer heads
  ropeFwd(kRot, ropeCos, ropeSin, seqLen, nKvDim, nKvHeadV2, hd)

  let outBuf = trackedCreate(seqLen * n)

  # Per Q-head temporaries
  let qH = trackedCreate(seqLen * hd)
  let kH = trackedCreate(seqLen * hd)
  let vH = trackedCreate(seqLen * hd)
  let scores = trackedCreate(seqLen * seqLen)
  let weights = trackedCreate(seqLen * seqLen)
  let attnH = trackedCreate(seqLen * hd)

  # Save weights per head for backward
  var allWeights = newSeq[GpuBuf](nHeadV2)

  for h in 0 ..< nHeadV2:
    # Extract Q head (from full nEmbd)
    extractHead(qRot, qH, h, seqLen, n, hd)
    # Extract KV head (from nKvDim, with repetition)
    gpu_extract_kv_head(kRot.data, kH.data,
                        cint(h), cint(kvRepeatV2),
                        cint(seqLen), cint(nKvDim), cint(hd))
    gpu_extract_kv_head(v.data.data, vH.data,
                        cint(h), cint(kvRepeatV2),
                        cint(seqLen), cint(nKvDim), cint(hd))

    # Scores = Q_h @ K_h^T
    gpuSgemm(2, seqLen, seqLen, hd, qH, kH, scores)
    causalMask(scores, scale, seqLen)
    softmaxFwd(scores, weights, seqLen, seqLen)

    allWeights[h] = trackedCreate(seqLen * seqLen)
    gpuCopy(weights, allWeights[h], seqLen * seqLen)

    # Output = weights @ V_h
    gpuSgemm(0, seqLen, hd, seqLen, weights, vH, attnH)
    # Insert into output at Q head position
    insertHead(attnH, outBuf, h, seqLen, n, hd)

  result = newNode(outBuf, seqLen * n, @[q, k, v])
  let outGrad = result.grad
  result.backwardFn = proc() =
    let dqRot = trackedCreate(seqLen * n)
    let dkRot = trackedCreate(seqLen * nKvDim)
    let doutH = trackedCreate(seqLen * hd)
    let dvH = trackedCreate(seqLen * hd)
    let dw = trackedCreate(seqLen * seqLen)
    let dScores = trackedCreate(seqLen * seqLen)
    let dqH = trackedCreate(seqLen * hd)
    let dkH = trackedCreate(seqLen * hd)

    for h in 0 ..< nHeadV2:
      # Extract Q head gradient
      extractHead(outGrad, doutH, h, seqLen, n, hd)
      # Extract V for this Q head (shared KV)
      gpu_extract_kv_head(v.data.data, vH.data,
                          cint(h), cint(kvRepeatV2),
                          cint(seqLen), cint(nKvDim), cint(hd))

      # dWeights = dAttn @ V^T
      gpuSgemm(2, seqLen, seqLen, hd, doutH, vH, dw)
      # dV += Weights^T @ dAttn (accumulate into shared KV head)
      gpuSgemm(4, seqLen, hd, seqLen, allWeights[h], doutH, dvH)
      gpu_insert_kv_head_acc(dvH.data, v.grad.data,
                             cint(h), cint(kvRepeatV2),
                             cint(seqLen), cint(nKvDim), cint(hd))

      # Causal softmax backward — CPU-side, matches OCaml exactly
      let dwCpu = gpuDownload(dw)
      let wtCpu = gpuDownload(allWeights[h])
      var dScoresCpu = newSeq[float32](seqLen * seqLen)
      for i in 0 ..< seqLen:
        var dot = 0.0f
        for j in 0 .. i:
          dot += dwCpu[i * seqLen + j] * wtCpu[i * seqLen + j]
        for j in 0 .. i:
          dScoresCpu[i * seqLen + j] =
            wtCpu[i * seqLen + j] * (dwCpu[i * seqLen + j] - dot) * scale
        for j in i + 1 ..< seqLen:
          dScoresCpu[i * seqLen + j] = 0.0f
      gpuUpload(dScores, dScoresCpu)

      # dQ_rot += dScores @ K_rot
      gpu_extract_kv_head(kRot.data, kH.data,
                          cint(h), cint(kvRepeatV2),
                          cint(seqLen), cint(nKvDim), cint(hd))
      gpuSgemm(0, seqLen, hd, seqLen, dScores, kH, dqH)
      insertHeadAcc(dqH, dqRot, h, seqLen, n, hd)

      # dK_rot += dScores^T @ Q_rot
      extractHead(qRot, qH, h, seqLen, n, hd)
      gpuSgemm(4, seqLen, hd, seqLen, dScores, qH, dkH)
      gpu_insert_kv_head_acc(dkH.data, dkRot.data,
                             cint(h), cint(kvRepeatV2),
                             cint(seqLen), cint(nKvDim), cint(hd))

    # Inverse RoPE
    ropeBwd(dqRot, ropeCos, ropeSin, seqLen, n, nHeadV2, hd)
    ropeBwd(dkRot, ropeCos, ropeSin, seqLen, nKvDim, nKvHeadV2, hd)
    gpu_add_inplace(q.grad.data, dqRot.data, cint(seqLen * n))
    gpu_add_inplace(k.grad.data, dkRot.data, cint(seqLen * nKvDim))

# ── V2 Transformer Block ─────────────────────────────────────────

proc agTransformerBlockV2*(x: Node, layer: GpuLayerV2,
                           ropeCos, ropeSin: GpuBuf,
                           seqLen: int): Node =
  let n = nEmbdV2
  let nKvDim = nKvHeadV2 * headDimV2
  proc pn(p: GpuParamV2): Node = paramNode(p.data, p.grad, p.numel)
  let wq = pn(layer.attnWq)
  let wk = pn(layer.attnWk)
  let wv = pn(layer.attnWv)
  let wo = pn(layer.attnWo)
  let wgate = pn(layer.mlpWgate)
  let wup = pn(layer.mlpWup)
  let wdown = pn(layer.mlpWdown)
  let ln1 = pn(layer.ln1)
  let ln2 = pn(layer.ln2)

  # Attention sub-block
  let xn = agRmsNormAffine(x, ln1, seqLen, n)
  let q = agMatmul(wq, xn, seqLen, n, n)
  let k = agMatmul(wk, xn, seqLen, nKvDim, n)     # fewer KV heads
  let v = agMatmul(wv, xn, seqLen, nKvDim, n)     # fewer KV heads
  let attnOut = agGqaAttention(q, k, v, ropeCos, ropeSin, seqLen)
  let projected = agDropout(agMatmul(wo, attnOut, seqLen, n, n))
  let afterAttn = agAdd(x, projected)

  # SwiGLU MLP sub-block
  let xn2 = agRmsNormAffine(afterAttn, ln2, seqLen, n)
  let gate = agMatmul(wgate, xn2, seqLen, ffnDimV2, n)
  let up = agMatmul(wup, xn2, seqLen, ffnDimV2, n)
  let hidden = agSwiGLU(gate, up)
  let mlpOut = agDropout(agMatmul(wdown, hidden, seqLen, n, ffnDimV2))
  agAdd(afterAttn, mlpOut)

# ── Full V2 forward pass ─────────────────────────────────────────

proc agForwardV2*(m: var GpuModelV2, tokenIds: seq[int32],
                  seqLen: int): Node =
  let n = nEmbdV2
  let wte = paramNode(m.wte.data, m.wte.grad, m.wte.numel)
  let embedNorm = paramNode(m.embedNorm.data, m.embedNorm.grad, m.embedNorm.numel)
  let finalNorm = paramNode(m.finalNorm.data, m.finalNorm.grad, m.finalNorm.numel)

  # Embedding
  var x = agEmbed(wte, tokenIds, seqLen, n)
  x = agRmsNormAffine(x, embedNorm, seqLen, n)
  x = agDropout(x)

  # Transformer layers
  for i in 0 ..< nLayerV2:
    x = agTransformerBlockV2(x, m.layers[i], m.ropeCos, m.ropeSin, seqLen)

  # Final norm + project to vocab (weight-tied)
  x = agRmsNormAffine(x, finalNorm, seqLen, n)
  agMatmul(wte, x, seqLen, m.vocabSize, n)

proc agComputeLossV2*(m: var GpuModelV2, tokens: seq[int32]): (Node, float32) =
  let seqLen = min(blockSizeV2, tokens.len - 1)
  if seqLen < 2: return (nil, 0.0f)

  let logits = agForwardV2(m, tokens[0 ..< seqLen], seqLen)

  var losses: seq[Node]
  for i in 0 ..< seqLen:
    let logitsI = agRow(logits, i, m.vocabSize)
    let ce = agCrossEntropy(logitsI, tokens[i + 1].int, m.vocabSize)
    losses.add(ce)

  var totalLoss = 0.0f
  for l in losses:
    let v = gpuDownload(l.data)
    totalLoss += v[0]
  let meanLoss = totalLoss / float32(seqLen)

  let meanBuf = trackedCreate(1)
  gpuUpload(meanBuf, @[meanLoss])
  let meanNode = newNode(meanBuf, 1, losses)
  let mGrad = meanNode.grad
  let sl = seqLen
  meanNode.backwardFn = proc() =
    let dy = gpuDownload(mGrad)
    let scale = dy[0] / float32(sl)
    for l in losses:
      gpuUpload(l.grad, @[scale])
  (meanNode, meanLoss)
