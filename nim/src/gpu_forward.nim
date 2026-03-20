## gpu_forward.nim — GPU forward pass with pre-allocated scratch buffers
##
## All temporaries are allocated once and reused across steps.
## No per-step allocation — the only cudaMalloc is at init.

import gpu, gpu_model
import std/math

type
  ForwardScratch* = object
    ## Pre-allocated GPU buffers for the forward pass.
    ## Allocated once, reused every step. Sized for max seq_len.
    xn, q, k, v: GpuBuf              # [S, nEmbd]
    qRot, kRot: GpuBuf               # [S, nEmbd]
    projected, afterAttn: GpuBuf      # [S, nEmbd]
    xn2, mlpOut: GpuBuf              # [S, nEmbd]
    hidden, activated: GpuBuf         # [S, ffnDim]
    attnOut: GpuBuf                   # [S, nEmbd]
    qH, kH, vH, attnH: GpuBuf       # [S, headDim] — per-head
    scores, weights: GpuBuf           # [S, S] — per-head
    rms1, rms2, rmsE, rmsF: GpuBuf   # [S] — rmsnorm intermediates
    xFinal: GpuBuf                    # [S, nEmbd]
    logits: GpuBuf                    # [S, vocab]
    probs: GpuBuf                     # [S, vocab]

proc initScratch*(maxSeq, vocabSize: int): ForwardScratch =
  ## Allocate all scratch buffers for max sequence length.
  let sn = maxSeq * nEmbd
  let sf = maxSeq * ffnDim
  let ss = maxSeq * maxSeq
  let sh = maxSeq * headDim
  let sv = maxSeq * vocabSize

  result.xn = gpuCreate(sn)
  result.q = gpuCreate(sn)
  result.k = gpuCreate(sn)
  result.v = gpuCreate(sn)
  result.qRot = gpuCreate(sn)
  result.kRot = gpuCreate(sn)
  result.projected = gpuCreate(sn)
  result.afterAttn = gpuCreate(sn)
  result.xn2 = gpuCreate(sn)
  result.mlpOut = gpuCreate(sn)
  result.hidden = gpuCreate(sf)
  result.activated = gpuCreate(sf)
  result.attnOut = gpuCreate(sn)
  result.qH = gpuCreate(sh)
  result.kH = gpuCreate(sh)
  result.vH = gpuCreate(sh)
  result.attnH = gpuCreate(sh)
  result.scores = gpuCreate(ss)
  result.weights = gpuCreate(ss)
  result.rms1 = gpuCreate(maxSeq)
  result.rms2 = gpuCreate(maxSeq)
  result.rmsE = gpuCreate(maxSeq)
  result.rmsF = gpuCreate(maxSeq)
  result.xFinal = gpuCreate(sn)
  result.logits = gpuCreate(sv)
  result.probs = gpuCreate(sv)

# ── Attention ─────────────────────────────────────────────────────

proc gpuAttention(x: GpuBuf, q, k, v: GpuBuf, m: GpuModel,
                  scratch: var ForwardScratch, seqLen: int): GpuBuf =
  ## Multi-head attention — all on GPU, reusing scratch buffers.
  let n = nEmbd
  let hd = headDim
  let scale = 1.0f / sqrt(float32(hd))

  # Copy for RoPE
  gpuCopy(q, scratch.qRot, seqLen * n)
  gpuCopy(k, scratch.kRot, seqLen * n)

  # Apply RoPE
  ropeFwd(scratch.qRot, m.ropeCos, m.ropeSin, seqLen, n, nHead, hd)
  ropeFwd(scratch.kRot, m.ropeCos, m.ropeSin, seqLen, n, nHead, hd)

  # Zero the output
  gpuZero(scratch.attnOut)

  for h in 0 ..< nHead:
    extractHead(scratch.qRot, scratch.qH, h, seqLen, n, hd)
    extractHead(scratch.kRot, scratch.kH, h, seqLen, n, hd)
    extractHead(v, scratch.vH, h, seqLen, n, hd)

    # Scores = Q @ K^T
    gpuSgemm(2, seqLen, seqLen, hd, scratch.qH, scratch.kH, scratch.scores)

    # Causal mask + scale
    causalMask(scratch.scores, scale, seqLen)

    # Softmax on GPU
    softmaxFwd(scratch.scores, scratch.weights, seqLen, seqLen)

    # Output = weights @ V
    gpuSgemm(0, seqLen, hd, seqLen, scratch.weights, scratch.vH, scratch.attnH)

    # Insert head
    insertHead(scratch.attnH, scratch.attnOut, h, seqLen, n, hd)

  scratch.attnOut

# ── Full forward pass ─────────────────────────────────────────────

proc gpuForward*(m: GpuModel, tokenIds: seq[int32], seqLen: int,
                 scratch: var ForwardScratch, x: var GpuBuf) =
  ## Full forward pass. Writes logits into scratch.logits.
  ## x is a pre-allocated [maxSeq, nEmbd] buffer for the hidden state.
  let n = nEmbd

  # Upload token IDs
  var tokBuf: pointer
  discard cudaMalloc(addr tokBuf, csize_t(seqLen * sizeof(int32)))
  discard cudaMemcpy(tokBuf, unsafeAddr tokenIds[0],
                     csize_t(seqLen * sizeof(int32)),
                     CudaMemcpyHostToDevice)

  # Embedding lookup
  gpu_embed_fwd(m.wte.data.data, tokBuf, x.data, cint(seqLen), cint(n))
  discard cudaFree(tokBuf)

  # Post-embed norm
  rmsnormAffineFwd(x, m.embedNorm.data, scratch.xn, scratch.rmsE, seqLen, n)
  gpuCopy(scratch.xn, x, seqLen * n)

  # Transformer layers
  for i in 0 ..< nLayer:
    let layer = m.layers[i]

    # Attention sub-block
    rmsnormAffineFwd(x, layer.ln1.data, scratch.xn, scratch.rms1, seqLen, n)
    gpuSgemm(2, seqLen, n, n, scratch.xn, layer.attnWq.data, scratch.q)
    gpuSgemm(2, seqLen, n, n, scratch.xn, layer.attnWk.data, scratch.k)
    gpuSgemm(2, seqLen, n, n, scratch.xn, layer.attnWv.data, scratch.v)

    let attnResult = gpuAttention(x, scratch.q, scratch.k, scratch.v,
                                   m, scratch, seqLen)
    gpuSgemm(2, seqLen, n, n, attnResult, layer.attnWo.data, scratch.projected)
    gpu_add(x.data, scratch.projected.data, scratch.afterAttn.data, cint(seqLen * n))

    # MLP sub-block
    rmsnormAffineFwd(scratch.afterAttn, layer.ln2.data, scratch.xn2,
                     scratch.rms2, seqLen, n)
    gpuSgemm(2, seqLen, ffnDim, n, scratch.xn2, layer.mlpFc1.data, scratch.hidden)
    geluFwd(scratch.hidden, scratch.activated)
    gpuSgemm(2, seqLen, n, ffnDim, scratch.activated, layer.mlpFc2.data, scratch.mlpOut)
    gpu_add(scratch.afterAttn.data, scratch.mlpOut.data, x.data, cint(seqLen * n))

  # Final norm + project to vocab
  rmsnormAffineFwd(x, m.finalNorm.data, scratch.xFinal, scratch.rmsF, seqLen, n)
  gpuSgemm(2, seqLen, m.vocabSize, n, scratch.xFinal, m.wte.data, scratch.logits)

# ── Loss ──────────────────────────────────────────────────────────

proc gpuComputeLoss*(m: GpuModel, tokens: seq[int32],
                     scratch: var ForwardScratch, x: var GpuBuf): float32 =
  ## Forward + cross-entropy loss. Returns scalar on CPU.
  let seqLen = min(blockSize, tokens.len - 1)
  if seqLen < 2: return 0.0f

  gpuForward(m, tokens[0 ..< seqLen], seqLen, scratch, x)

  # Softmax
  softmaxFwd(scratch.logits, scratch.probs, seqLen, m.vocabSize)

  # Download probs for NLL
  let probsCpu = gpuDownload(scratch.probs)
  var totalLoss = 0.0f
  for i in 0 ..< seqLen:
    let target = tokens[i + 1]
    let p = probsCpu[i * m.vocabSize + target]
    totalLoss += -ln(max(p, 1e-10f))

  totalLoss / float32(seqLen)
