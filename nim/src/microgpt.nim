## microgpt.nim — Karpathy's microGPT ported to Nim with GPU
##
## Explicit forward + backward. No autograd graph. No closures.
## Each operation saves what backward needs. Backward runs in reverse.
## Same structure as microgpt.py, but with GPU tensors.
##
## Start small (verify correctness), then scale up.

import gpu, bpe, autograd
import std/[math, random, strformat, os, streams, times]

# ── Configuration ─────────────────────────────────────────────────

const
  nLayer   = 4
  nEmbd    = 256
  nHead    = 4
  headDim  = nEmbd div nHead  # 64
  blockSize = 256
  ffnMul   = 4

# ── Model ─────────────────────────────────────────────────────────

type
  Layer = object
    wq, wk, wv, wo: GpuBuf       # attention weights [nEmbd, nEmbd]
    fc1: GpuBuf                    # MLP up [ffnMul*nEmbd, nEmbd]
    fc2: GpuBuf                    # MLP down [nEmbd, ffnMul*nEmbd]
    ln1g, ln2g: GpuBuf            # learnable RMSNorm gamma [nEmbd]
    # Gradients
    dwq, dwk, dwv, dwo: GpuBuf
    dfc1, dfc2: GpuBuf
    dln1g, dln2g: GpuBuf

  Model = object
    wte: GpuBuf                    # token embeddings [vocab, nEmbd]
    wpe: GpuBuf                    # position embeddings [blockSize, nEmbd]
    lmHead: GpuBuf                 # output projection [vocab, nEmbd]
    lnFg: GpuBuf                   # final RMSNorm gamma [nEmbd]
    layers: seq[Layer]
    vocabSize: int
    # Gradients
    dwte, dwpe, dlmHead: GpuBuf
    dlnFg: GpuBuf

proc initModel(vocabSize: int): Model =
  randomize(42)
  let std = 0.02f

  proc randBuf(n: int, s: float32 = std): GpuBuf =
    var host = newSeq[float32](n)
    for i in 0 ..< n:
      let u1 = rand(1.0).float32
      let u2 = rand(1.0).float32
      host[i] = s * sqrt(-2f * ln(max(u1, 1e-10f))) * cos(2f * PI.float32 * u2)
    toGpu(host)

  result.vocabSize = vocabSize
  result.wte = randBuf(vocabSize * nEmbd)
  result.wpe = randBuf(blockSize * nEmbd)
  result.lmHead = randBuf(vocabSize * nEmbd)
  result.dwte = gpuCreate(vocabSize * nEmbd)
  result.dwpe = gpuCreate(blockSize * nEmbd)
  result.dlmHead = gpuCreate(vocabSize * nEmbd)

  proc onesBuf(n: int): GpuBuf =
    var h = newSeq[float32](n)
    for i in 0 ..< n: h[i] = 1.0f
    toGpu(h)

  result.lnFg = onesBuf(nEmbd)
  result.dlnFg = gpuCreate(nEmbd)

  for i in 0 ..< nLayer:
    let resStd = std / sqrt(float32(2 * nLayer))
    result.layers.add(Layer(
      wq: randBuf(nEmbd * nEmbd),
      wk: randBuf(nEmbd * nEmbd),
      wv: randBuf(nEmbd * nEmbd),
      wo: randBuf(nEmbd * nEmbd, resStd),
      fc1: randBuf(ffnMul * nEmbd * nEmbd),
      fc2: randBuf(nEmbd * ffnMul * nEmbd, resStd),
      ln1g: onesBuf(nEmbd),
      ln2g: onesBuf(nEmbd),
      dwq: gpuCreate(nEmbd * nEmbd),
      dwk: gpuCreate(nEmbd * nEmbd),
      dwv: gpuCreate(nEmbd * nEmbd),
      dwo: gpuCreate(nEmbd * nEmbd),
      dfc1: gpuCreate(ffnMul * nEmbd * nEmbd),
      dfc2: gpuCreate(nEmbd * ffnMul * nEmbd),
      dln1g: gpuCreate(nEmbd),
      dln2g: gpuCreate(nEmbd),
    ))

  let total = vocabSize * nEmbd * 2 + blockSize * nEmbd +
    nLayer * (4 * nEmbd * nEmbd + 2 * ffnMul * nEmbd * nEmbd)
  echo &"  params: {total}"

# ── Saved intermediates for backward ──────────────────────────────

type
  LayerCache = object
    xResidual1: GpuBuf   # input to attention block (for residual backward)
    xInput1: GpuBuf      # ORIGINAL input to rmsnorm1 (needed for backward!)
    xNorm1: GpuBuf       # after first rmsnorm
    rms1: GpuBuf         # rms value for rmsnorm backward
    q, k, v: GpuBuf      # after projection
    attnWeights: seq[GpuBuf]  # per-head softmax weights
    attnOut: GpuBuf       # concatenated attention output
    xResidual2: GpuBuf   # input to MLP block
    xInput2: GpuBuf      # ORIGINAL input to rmsnorm2 (needed for backward!)
    xNorm2: GpuBuf       # after second rmsnorm
    rms2: GpuBuf
    fc1Out: GpuBuf       # after fc1 (pre-relu)
    reluOut: GpuBuf      # after relu

  ForwardCache = object
    embedded: GpuBuf      # token + position embeddings
    layerCaches: seq[LayerCache]
    xPreFinalNorm: GpuBuf # ORIGINAL input to final rmsnorm
    finalNormed: GpuBuf
    rmsF: GpuBuf
    logits: GpuBuf
    logProbs: GpuBuf      # log-softmax output

# ── Forward pass ──────────────────────────────────────────────────

proc forward(m: Model, tokens: seq[int32], seqLen: int): (ForwardCache, float32) =
  var cache: ForwardCache
  let n = nEmbd
  let S = seqLen

  # Embedding: tok + pos
  let tokEmb = trackedCreate(S * n)
  var tokBuf: pointer
  discard cudaMalloc(addr tokBuf, csize_t(S * sizeof(int32)))
  discard cudaMemcpy(tokBuf, unsafeAddr tokens[0],
                     csize_t(S * sizeof(int32)), CudaMemcpyHostToDevice)
  gpu_embed_fwd(m.wte.data, tokBuf, tokEmb.data, cint(S), cint(n))

  let posEmb = trackedCreate(S * n)
  # Position IDs: 0, 1, 2, ..., S-1
  var posIds = newSeq[int32](S)
  for i in 0 ..< S: posIds[i] = int32(i)
  var posBuf: pointer
  discard cudaMalloc(addr posBuf, csize_t(S * sizeof(int32)))
  discard cudaMemcpy(posBuf, unsafeAddr posIds[0],
                     csize_t(S * sizeof(int32)), CudaMemcpyHostToDevice)
  gpu_embed_fwd(m.wpe.data, posBuf, posEmb.data, cint(S), cint(n))

  var x = trackedCreate(S * n)
  gpu_add(tokEmb.data, posEmb.data, x.data, cint(S * n))
  cache.embedded = x

  discard cudaFree(tokBuf)
  discard cudaFree(posBuf)

  # Transformer layers
  for li in 0 ..< nLayer:
    var lc: LayerCache
    let layer = m.layers[li]

    # Save residual
    lc.xResidual1 = trackedCreate(S * n)
    gpuCopy(x, lc.xResidual1, S * n)

    # RMSNorm 1 with learnable gamma — save original input for backward
    lc.xInput1 = trackedCreate(S * n)
    gpuCopy(x, lc.xInput1, S * n)
    lc.xNorm1 = trackedCreate(S * n)
    lc.rms1 = trackedCreate(S)
    gpu_rmsnorm_affine_fwd(x.data, layer.ln1g.data, lc.xNorm1.data, lc.rms1.data, cint(S), cint(n))

    # Q, K, V projections
    lc.q = trackedCreate(S * n)
    lc.k = trackedCreate(S * n)
    lc.v = trackedCreate(S * n)
    gpuSgemm(2, S, n, n, lc.xNorm1, layer.wq, lc.q)
    gpuSgemm(2, S, n, n, lc.xNorm1, layer.wk, lc.k)
    gpuSgemm(2, S, n, n, lc.xNorm1, layer.wv, lc.v)

    # Multi-head attention
    let scale = 1.0f / sqrt(float32(headDim))
    lc.attnOut = trackedCreate(S * n)
    lc.attnWeights = newSeq[GpuBuf](nHead)

    let qH = trackedCreate(S * headDim)
    let kH = trackedCreate(S * headDim)
    let vH = trackedCreate(S * headDim)
    let scores = trackedCreate(S * S)
    let weights = trackedCreate(S * S)
    let attnH = trackedCreate(S * headDim)

    for h in 0 ..< nHead:
      extractHead(lc.q, qH, h, S, n, headDim)
      extractHead(lc.k, kH, h, S, n, headDim)
      extractHead(lc.v, vH, h, S, n, headDim)

      gpuSgemm(2, S, S, headDim, qH, kH, scores)
      causalMask(scores, scale, S)
      # Clamp AFTER scaling — prevent any possibility of overflow
      gpu_clamp(scores.data, -10.0f, 10.0f, cint(S * S))
      softmaxFwd(scores, weights, S, S)

      lc.attnWeights[h] = trackedCreate(S * S)
      gpuCopy(weights, lc.attnWeights[h], S * S)

      gpuSgemm(0, S, headDim, S, weights, vH, attnH)
      insertHead(attnH, lc.attnOut, h, S, n, headDim)

    # Output projection
    var projected = trackedCreate(S * n)
    gpuSgemm(2, S, n, n, lc.attnOut, layer.wo, projected)

    # Residual 1
    lc.xResidual2 = trackedCreate(S * n)
    gpu_add(lc.xResidual1.data, projected.data, lc.xResidual2.data, cint(S * n))
    x = lc.xResidual2

    # RMSNorm 2 with learnable gamma — save original input for backward
    lc.xInput2 = trackedCreate(S * n)
    gpuCopy(x, lc.xInput2, S * n)
    lc.xNorm2 = trackedCreate(S * n)
    lc.rms2 = trackedCreate(S)
    gpu_rmsnorm_affine_fwd(x.data, layer.ln2g.data, lc.xNorm2.data, lc.rms2.data, cint(S), cint(n))

    # MLP: fc1 -> relu -> fc2
    lc.fc1Out = trackedCreate(S * ffnMul * n)
    gpuSgemm(2, S, ffnMul * n, n, lc.xNorm2, layer.fc1, lc.fc1Out)

    lc.reluOut = trackedCreate(S * ffnMul * n)
    gpu_gelu_fwd(lc.fc1Out.data, lc.reluOut.data, cint(S * ffnMul * n))
    # TODO: microGPT uses ReLU not GELU. For now GELU is fine.

    var mlpOut = trackedCreate(S * n)
    gpuSgemm(2, S, n, ffnMul * n, lc.reluOut, layer.fc2, mlpOut)

    # Residual 2
    let xNew = trackedCreate(S * n)
    gpu_add(lc.xResidual2.data, mlpOut.data, xNew.data, cint(S * n))
    x = xNew

    cache.layerCaches.add(lc)

  # Final norm with learnable gamma
  cache.xPreFinalNorm = trackedCreate(S * n)
  gpuCopy(x, cache.xPreFinalNorm, S * n)
  cache.finalNormed = trackedCreate(S * n)
  cache.rmsF = trackedCreate(S)
  gpu_rmsnorm_affine_fwd(x.data, m.lnFg.data, cache.finalNormed.data, cache.rmsF.data, cint(S), cint(n))

  # Logits
  cache.logits = trackedCreate(S * m.vocabSize)
  gpuSgemm(2, S, m.vocabSize, n, cache.finalNormed, m.lmHead, cache.logits)

  # Log-softmax + cross-entropy loss
  cache.logProbs = trackedCreate(S * m.vocabSize)
  gpu_log_softmax(cache.logits.data, cache.logProbs.data, cint(S), cint(m.vocabSize))

  # Compute mean loss on CPU
  let logProbsCpu = gpuDownload(cache.logProbs)
  var totalLoss = 0.0f
  for i in 0 ..< S:
    let target = tokens[i + 1]
    totalLoss += -logProbsCpu[i * m.vocabSize + target]
  let loss = totalLoss / float32(S)

  (cache, loss)

# ── Backward pass ─────────────────────────────────────────────────

proc backward(m: var Model, tokens: seq[int32], seqLen: int,
              cache: ForwardCache) =
  let S = seqLen
  let n = nEmbd
  let V = m.vocabSize

  # dLogits = softmax(logits) - one_hot(target), scaled by 1/S
  var dLogitsCpu = newSeq[float32](S * V)
  let logProbsCpu = gpuDownload(cache.logProbs)
  for i in 0 ..< S:
    for j in 0 ..< V:
      dLogitsCpu[i * V + j] = exp(logProbsCpu[i * V + j]) / float32(S)
    dLogitsCpu[i * V + tokens[i + 1]] -= 1.0f / float32(S)
  let dLogits = trackedCreate(S * V)
  gpuUpload(dLogits, dLogitsCpu)

  # dLmHead += dLogits^T @ finalNormed
  gpuSgemm(5, V, n, S, dLogits, cache.finalNormed, m.dlmHead)
  # dFinalNormed = dLogits @ lmHead
  var dx = trackedCreate(S * n)
  gpuSgemm(1, S, n, V, dLogits, m.lmHead, dx)

  # Final RMSNorm backward
  let dxPreNorm = trackedCreate(S * n)
  gpu_rmsnorm_affine_bwd(cache.xPreFinalNorm.data, m.lnFg.data, dx.data,
                         cache.rmsF.data, dxPreNorm.data, m.dlnFg.data,
                         cint(S), cint(n))
  dx = dxPreNorm

  # Backward through layers in reverse
  for li in countdown(nLayer - 1, 0):
    let lc = cache.layerCaches[li]
    let layer = m.layers[li]

    # Residual 2 backward: dx flows to both MLP path and skip
    let dMlpOut = trackedCreate(S * n)
    gpuCopy(dx, dMlpOut, S * n)
    # dx also flows through residual (already in dx)

    # MLP backward: fc2
    let dReluOut = trackedCreate(S * ffnMul * n)
    gpuSgemm(1, S, ffnMul * n, n, dMlpOut, layer.fc2, dReluOut)
    gpuSgemm(5, n, ffnMul * n, S, dMlpOut, lc.reluOut, m.layers[li].dfc2)

    # GELU backward
    let dFc1Out = trackedCreate(S * ffnMul * n)
    gpu_gelu_bwd(lc.fc1Out.data, dReluOut.data, dFc1Out.data, cint(S * ffnMul * n))

    # fc1 backward
    let dNorm2 = trackedCreate(S * n)
    gpuSgemm(1, S, n, ffnMul * n, dFc1Out, layer.fc1, dNorm2)
    gpuSgemm(5, ffnMul * n, n, S, dFc1Out, lc.xNorm2, m.layers[li].dfc1)

    # RMSNorm 2 backward
    let dResid2 = trackedCreate(S * n)
    gpu_rmsnorm_affine_bwd(lc.xInput2.data, layer.ln2g.data, dNorm2.data,
                           lc.rms2.data, dResid2.data, m.layers[li].dln2g.data,
                           cint(S), cint(n))
    # Add residual gradient
    gpu_add_inplace(dx.data, dResid2.data, cint(S * n))

    # Attention output projection backward
    let dAttnOut = trackedCreate(S * n)
    # dProjected = dx (from residual 1 backward)
    gpuSgemm(1, S, n, n, dx, layer.wo, dAttnOut)
    gpuSgemm(5, n, n, S, dx, lc.attnOut, m.layers[li].dwo)

    # Multi-head attention backward
    let dq = trackedCreate(S * n)
    let dk = trackedCreate(S * n)
    let dv = trackedCreate(S * n)
    let scale = 1.0f / sqrt(float32(headDim))

    let doutH = trackedCreate(S * headDim)
    let dvH = trackedCreate(S * headDim)
    let dw = trackedCreate(S * S)
    let dScores = trackedCreate(S * S)
    let dqH = trackedCreate(S * headDim)
    let dkH = trackedCreate(S * headDim)
    let qH = trackedCreate(S * headDim)
    let kH = trackedCreate(S * headDim)
    let vH = trackedCreate(S * headDim)

    for h in 0 ..< nHead:
      extractHead(dAttnOut, doutH, h, S, n, headDim)
      extractHead(lc.v, vH, h, S, n, headDim)

      # dWeights = dAttnOut_h @ V_h^T
      gpuSgemm(2, S, S, headDim, doutH, vH, dw)
      # dV_h = Weights^T @ dAttnOut_h
      gpuSgemm(4, S, headDim, S, lc.attnWeights[h], doutH, dvH)
      insertHeadAcc(dvH, dv, h, S, n, headDim)

      # Causal softmax backward (CPU, matches OCaml exactly)
      let dwCpu = gpuDownload(dw)
      let wtCpu = gpuDownload(lc.attnWeights[h])
      var dScoresCpu = newSeq[float32](S * S)
      for i in 0 ..< S:
        var dot = 0.0f
        for j in 0 .. i:
          dot += dwCpu[i * S + j] * wtCpu[i * S + j]
        for j in 0 .. i:
          dScoresCpu[i * S + j] =
            wtCpu[i * S + j] * (dwCpu[i * S + j] - dot) * scale
      gpuUpload(dScores, dScoresCpu)

      # dQ_h = dScores @ K_h
      extractHead(lc.k, kH, h, S, n, headDim)
      gpuSgemm(0, S, headDim, S, dScores, kH, dqH)
      insertHeadAcc(dqH, dq, h, S, n, headDim)

      # dK_h = dScores^T @ Q_h
      extractHead(lc.q, qH, h, S, n, headDim)
      gpuSgemm(4, S, headDim, S, dScores, qH, dkH)
      insertHeadAcc(dkH, dk, h, S, n, headDim)

    # QKV projection backward
    let dNorm1 = trackedCreate(S * n)
    gpuSgemm(1, S, n, n, dq, layer.wq, dNorm1)
    gpuSgemm(5, n, n, S, dq, lc.xNorm1, m.layers[li].dwq)

    let dNorm1k = trackedCreate(S * n)
    gpuSgemm(1, S, n, n, dk, layer.wk, dNorm1k)
    gpuSgemm(5, n, n, S, dk, lc.xNorm1, m.layers[li].dwk)
    gpu_add_inplace(dNorm1.data, dNorm1k.data, cint(S * n))

    let dNorm1v = trackedCreate(S * n)
    gpuSgemm(1, S, n, n, dv, layer.wv, dNorm1v)
    gpuSgemm(5, n, n, S, dv, lc.xNorm1, m.layers[li].dwv)
    gpu_add_inplace(dNorm1.data, dNorm1v.data, cint(S * n))

    # RMSNorm 1 backward
    let dResid1 = trackedCreate(S * n)
    gpu_rmsnorm_affine_bwd(lc.xInput1.data, layer.ln1g.data, dNorm1.data,
                           lc.rms1.data, dResid1.data, m.layers[li].dln1g.data,
                           cint(S), cint(n))

    # Residual 1: dx for next layer = dResid1 + dResid2_skip
    gpu_add(dResid1.data, dx.data, dx.data, cint(S * n))

  # Embedding backward
  var tokBuf: pointer
  discard cudaMalloc(addr tokBuf, csize_t(S * sizeof(int32)))
  discard cudaMemcpy(tokBuf, unsafeAddr tokens[0],
                     csize_t(S * sizeof(int32)), CudaMemcpyHostToDevice)
  gpu_embed_bwd(m.dwte.data, tokBuf, dx.data, cint(S), cint(n))
  discard cudaFree(tokBuf)

  # Position embedding backward
  var posBuf: pointer
  var posIds = newSeq[int32](S)
  for i in 0 ..< S: posIds[i] = int32(i)
  discard cudaMalloc(addr posBuf, csize_t(S * sizeof(int32)))
  discard cudaMemcpy(posBuf, unsafeAddr posIds[0],
                     csize_t(S * sizeof(int32)), CudaMemcpyHostToDevice)
  gpu_embed_bwd(m.dwpe.data, posBuf, dx.data, cint(S), cint(n))
  discard cudaFree(posBuf)

# ── Zero gradients ────────────────────────────────────────────────

proc zeroGrads(m: var Model) =
  gpuZero(m.dwte)
  gpuZero(m.dwpe)
  gpuZero(m.dlmHead)
  gpuZero(m.dlnFg)
  for i in 0 ..< m.layers.len:
    gpuZero(m.layers[i].dwq)
    gpuZero(m.layers[i].dwk)
    gpuZero(m.layers[i].dwv)
    gpuZero(m.layers[i].dwo)
    gpuZero(m.layers[i].dfc1)
    gpuZero(m.layers[i].dfc2)
    gpuZero(m.layers[i].dln1g)
    gpuZero(m.layers[i].dln2g)

# ── Adam optimizer ────────────────────────────────────────────────

type AdamState = object
  m, v: seq[GpuBuf]

proc initAdam(m: Model): AdamState =
  proc addPair(s: var AdamState, buf: GpuBuf) =
    s.m.add(gpuCreate(buf.numel))
    s.v.add(gpuCreate(buf.numel))
  addPair(result, m.wte)
  addPair(result, m.wpe)
  addPair(result, m.lmHead)
  addPair(result, m.lnFg)
  for layer in m.layers:
    addPair(result, layer.wq)
    addPair(result, layer.wk)
    addPair(result, layer.wv)
    addPair(result, layer.wo)
    addPair(result, layer.ln1g)
    addPair(result, layer.ln2g)
    addPair(result, layer.fc1)
    addPair(result, layer.fc2)

proc adamUpdate(m: var Model, adam: var AdamState, lr: float32,
                step: int, beta1 = 0.9f, beta2 = 0.999f, wd = 0.1f) =
  let bc1 = 1.0f / (1.0f - pow(beta1, float32(step + 1)))
  let bc2 = 1.0f / (1.0f - pow(beta2, float32(step + 1)))

  var pairs: seq[(GpuBuf, GpuBuf)] # (param, grad)
  pairs.add((m.wte, m.dwte))
  pairs.add((m.wpe, m.dwpe))
  pairs.add((m.lmHead, m.dlmHead))
  pairs.add((m.lnFg, m.dlnFg))
  for i in 0 ..< m.layers.len:
    pairs.add((m.layers[i].wq, m.layers[i].dwq))
    pairs.add((m.layers[i].wk, m.layers[i].dwk))
    pairs.add((m.layers[i].wv, m.layers[i].dwv))
    pairs.add((m.layers[i].wo, m.layers[i].dwo))
    pairs.add((m.layers[i].ln1g, m.layers[i].dln1g))
    pairs.add((m.layers[i].ln2g, m.layers[i].dln2g))
    pairs.add((m.layers[i].fc1, m.layers[i].dfc1))
    pairs.add((m.layers[i].fc2, m.layers[i].dfc2))

  for i in 0 ..< pairs.len:
    gpu_adamw(pairs[i][0].data, pairs[i][1].data,
              adam.m[i].data, adam.v[i].data,
              lr, beta1, beta2, bc1, bc2, wd,
              cint(pairs[i][0].numel))

# ── Training ──────────────────────────────────────────────────────

proc formatDuration(secs: float): string =
  let h = int(secs) div 3600
  let m = (int(secs) mod 3600) div 60
  let s = int(secs) mod 60
  if h > 0: &"{h}h{m:02d}m{s:02d}s"
  elif m > 0: &"{m}m{s:02d}s"
  else: &"{s}s"

when isMainModule:
  let baseDir = getAppDir().parentDir()
  let vidyaRoot = baseDir.parentDir()
  let dataFile = vidyaRoot / "chat_input.txt"  # small dataset for now
  let tokenizerFile = vidyaRoot / "tokenizer_nim.bin"

  gpuInit()
  echo "microgpt (nim) starting..."

  # Tokenizer
  var tok: Tokenizer
  if fileExists(tokenizerFile):
    tok = loadTokenizer(tokenizerFile)
  else:
    let docs = loadDocs(dataFile)
    tok = trainBpe(docs)
    saveTokenizer(tok, tokenizerFile)
  echo &"vocab: {tok.vocab.len}"

  # Model
  echo "init model..."
  var m = initModel(tok.vocab.len)

  # Data
  echo "loading data..."
  let docs = loadDocs(dataFile)
  let t0 = cpuTime()
  var tokenizedDocs: seq[seq[int32]]
  for doc in docs:
    let ids = tok.encode(doc)
    var ids32 = newSeq[int32](ids.len)
    for i in 0 ..< ids.len: ids32[i] = int32(ids[i])
    if ids32.len >= 3:
      tokenizedDocs.add(ids32)
  echo &"  {tokenizedDocs.len} docs in {cpuTime() - t0:.1f}s"

  # Shuffle
  randomize()
  var order = newSeq[int](tokenizedDocs.len)
  for i in 0 ..< order.len: order[i] = i
  shuffle(order)

  # Adam
  var adam = initAdam(m)

  # Use tracked allocations for automatic cleanup
  echo "  using tracked allocations"

  # Train
  let numSteps = 200000
  let warmupSteps = 2000
  let peakLr = 0.0003f
  let minLr = 0.00003f
  let gradAccum = 1      # batch 1 — research says it works with right settings
  trackingEnabled = true
  echo &"training {numSteps} steps (grad_accum={gradAccum})..."

  let tStart = cpuTime()
  var lossSum = 0.0f
  var microStep = 0
  var optStep = 0
  let logInterval = 50  # in optimizer steps

  for step in 0 ..< numSteps:
    let tokens = tokenizedDocs[order[step mod order.len]]
    let seqLen = min(blockSize, tokens.len - 1)
    if seqLen < 2: continue

    var (cache, loss) = forward(m, tokens[0 ..< seqLen + 1], seqLen)
    if loss != loss:  # NaN
      echo "NaN at step ", optStep
      proc chk(buf: GpuBuf, name: string) =
        let d = gpuDownload(buf)
        for i in 0 ..< d.len:
          if d[i] != d[i]: echo "  NaN: ", name, "[", i, "]"; return
          if d[i] > 1e30: echo "  INF: ", name, "[", i, "]=", d[i]; return
      chk(cache.embedded, "embedded")
      for li in 0 ..< nLayer:
        chk(cache.layerCaches[li].xInput1, "L" & $li & ".xInput1")
        chk(cache.layerCaches[li].xNorm1, "L" & $li & ".xNorm1")
        chk(cache.layerCaches[li].q, "L" & $li & ".q")
        chk(cache.layerCaches[li].k, "L" & $li & ".k")
        chk(cache.layerCaches[li].v, "L" & $li & ".v")
        chk(cache.layerCaches[li].attnOut, "L" & $li & ".attnOut")
        chk(cache.layerCaches[li].xInput2, "L" & $li & ".xInput2")
        chk(cache.layerCaches[li].xNorm2, "L" & $li & ".xNorm2")
        chk(cache.layerCaches[li].fc1Out, "L" & $li & ".fc1Out")
        chk(cache.layerCaches[li].reluOut, "L" & $li & ".reluOut")
      chk(cache.xPreFinalNorm, "xPreFinalNorm")
      chk(cache.finalNormed, "finalNormed")
      chk(cache.logits, "logits")
      chk(cache.logProbs, "logProbs")
      freeStepAllocations()
      continue
    backward(m, tokens[0 ..< seqLen + 1], seqLen, cache)
    freeStepAllocations()

    lossSum += loss
    microStep += 1

    if microStep < gradAccum:
      continue

    # Scale gradients by 1/gradAccum
    proc scaleAllGrads(m: var Model, s: float32) =
      gpu_scale(m.dwte.data, s, m.dwte.data, cint(m.dwte.numel))
      gpu_scale(m.dwpe.data, s, m.dwpe.data, cint(m.dwpe.numel))
      gpu_scale(m.dlmHead.data, s, m.dlmHead.data, cint(m.dlmHead.numel))
      for i in 0 ..< m.layers.len:
        gpu_scale(m.layers[i].dwq.data, s, m.layers[i].dwq.data, cint(m.layers[i].dwq.numel))
        gpu_scale(m.layers[i].dwk.data, s, m.layers[i].dwk.data, cint(m.layers[i].dwk.numel))
        gpu_scale(m.layers[i].dwv.data, s, m.layers[i].dwv.data, cint(m.layers[i].dwv.numel))
        gpu_scale(m.layers[i].dwo.data, s, m.layers[i].dwo.data, cint(m.layers[i].dwo.numel))
        gpu_scale(m.layers[i].dfc1.data, s, m.layers[i].dfc1.data, cint(m.layers[i].dfc1.numel))
        gpu_scale(m.layers[i].dfc2.data, s, m.layers[i].dfc2.data, cint(m.layers[i].dfc2.numel))
    scaleAllGrads(m, 1.0f / float32(gradAccum))

    # LR schedule
    let totalOptSteps = numSteps div gradAccum
    let lr = if optStep < warmupSteps:
        peakLr * float32(optStep) / float32(warmupSteps)
      else:
        let progress = float32(optStep - warmupSteps) / float32(totalOptSteps - warmupSteps)
        minLr + (peakLr - minLr) * 0.5f * (1f + cos(PI.float32 * progress))

    # Gradient clipping — clip global norm to 1.0
    var gradPtrs: seq[pointer]
    var gradSizes: seq[cint]
    gradPtrs.add(m.dwte.data); gradSizes.add(cint(m.dwte.numel))
    gradPtrs.add(m.dwpe.data); gradSizes.add(cint(m.dwpe.numel))
    gradPtrs.add(m.dlmHead.data); gradSizes.add(cint(m.dlmHead.numel))
    gradPtrs.add(m.dlnFg.data); gradSizes.add(cint(m.dlnFg.numel))
    for i in 0 ..< m.layers.len:
      gradPtrs.add(m.layers[i].dwq.data); gradSizes.add(cint(m.layers[i].dwq.numel))
      gradPtrs.add(m.layers[i].dwk.data); gradSizes.add(cint(m.layers[i].dwk.numel))
      gradPtrs.add(m.layers[i].dwv.data); gradSizes.add(cint(m.layers[i].dwv.numel))
      gradPtrs.add(m.layers[i].dwo.data); gradSizes.add(cint(m.layers[i].dwo.numel))
      gradPtrs.add(m.layers[i].dfc1.data); gradSizes.add(cint(m.layers[i].dfc1.numel))
      gradPtrs.add(m.layers[i].dfc2.data); gradSizes.add(cint(m.layers[i].dfc2.numel))
      gradPtrs.add(m.layers[i].dln1g.data); gradSizes.add(cint(m.layers[i].dln1g.numel))
      gradPtrs.add(m.layers[i].dln2g.data); gradSizes.add(cint(m.layers[i].dln2g.numel))
    clipGradNorm(gradPtrs, gradSizes, 1.0f)

    adamUpdate(m, adam, lr, optStep, beta2 = 0.9999f, wd = 0.0f)
    zeroGrads(m)
    microStep = 0
    optStep += 1

    if optStep mod logInterval == 0:
      let elapsed = cpuTime() - tStart
      let opsps = float(optStep) / elapsed
      let totalOpt = numSteps div gradAccum
      let eta = float(totalOpt - optStep) / max(opsps, 0.01)
      echo &"opt {optStep:>5} / {totalOpt} | loss {lossSum / float32(logInterval * gradAccum):.4f} | lr {lr:.6f} | {opsps:.1f} opt/s | {formatDuration(elapsed)} elapsed | {formatDuration(eta)} remaining"
      lossSum = 0.0f
