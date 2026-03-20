## forward.nim — Transformer forward pass
##
## Arraymancer handles GPU dispatch, memory, and BLAS automatically.
## No manual cudaMalloc, no FFI bridge, no head extraction kernels.
## Just tensor operations that read like the math.
##
## Compare this to the 300-line OCaml forward.ml + 500-line gpu_stubs.cu.

import arraymancer
import model
import std/math

# ── Activations ───────────────────────────────────────────────────

proc gelu*(x: Tensor[float32]): Tensor[float32] =
  ## GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  let a = 0.7978845608f
  let b = 0.044715f
  result = x.map_inline:
    let inner = a * (x + b * x * x * x)
    0.5f * x * (1.0f + tanh(inner))

proc rmsNorm*(x: Tensor[float32], gamma: Tensor[float32]): Tensor[float32] =
  ## RMSNorm: normalize by root-mean-square, then scale by gamma.
  ## x is [S, dim] or [dim]. gamma is [dim].
  let dims = x.shape.len
  if dims == 1:
    let dim = x.shape[0]
    let ms = x.dot(x) / float32(dim)
    let rms = sqrt(ms + 1e-5f)
    result = (x / rms) *. gamma
  else:
    # Batched: normalize each row independently
    let seqLen = x.shape[0]
    let dim = x.shape[1]
    result = newTensor[float32](seqLen, dim)
    for i in 0 ..< seqLen:
      let row = x[i, _].squeeze(0)
      let ms = row.dot(row) / float32(dim)
      let rms = sqrt(ms + 1e-5f)
      result[i, _] = ((row / rms) *. gamma).unsqueeze(0)

# ── Attention ─────────────────────────────────────────────────────

proc applyRope*(x: Tensor[float32], seqLen: int): Tensor[float32] =
  ## Apply rotary position embeddings to [S, nEmbd] tensor.
  ## Rotates pairs within each head's dimensions.
  result = x.clone()
  for pos in 0 ..< seqLen:
    for h in 0 ..< nHead:
      let base = h * headDim
      for i in 0 ..< halfDim:
        let j0 = base + i
        let j1 = base + i + halfDim
        let c = ropeCos[pos, i]
        let s = ropeSin[pos, i]
        let x0 = x[pos, j0]
        let x1 = x[pos, j1]
        result[pos, j0] = x0 * c - x1 * s
        result[pos, j1] = x0 * s + x1 * c

proc extractHead*(x: Tensor[float32], h: int, seqLen: int): Tensor[float32] =
  ## Extract head h from [S, nEmbd] → [S, headDim].
  let offset = h * headDim
  x[_, offset ..< offset + headDim]

proc fusedAttention*(q, k, v: Tensor[float32], seqLen: int): Tensor[float32] =
  ## Multi-head attention with RoPE and causal masking.
  let qRot = applyRope(q, seqLen)
  let kRot = applyRope(k, seqLen)
  let scale = 1.0f / sqrt(float32(headDim))

  result = zeros[float32](seqLen, nEmbd)

  for h in 0 ..< nHead:
    let qH = extractHead(qRot, h, seqLen)  # [S, headDim]
    let kH = extractHead(kRot, h, seqLen)  # [S, headDim]
    let vH = extractHead(v, h, seqLen)     # [S, headDim]

    # Scores = Q @ K^T → [S, S]
    var scores = qH * kH.transpose() * scale

    # Causal mask: set upper triangle to -inf
    for i in 0 ..< seqLen:
      for j in i + 1 ..< seqLen:
        scores[i, j] = -1e9f

    # Row-wise softmax
    for i in 0 ..< seqLen:
      let row = scores[i, _].squeeze(0)
      let maxVal = row.max()
      let expRow = map_inline(row -. maxVal):
        exp(x)
      let sumExp = expRow.sum()
      scores[i, _] = (expRow / sumExp).unsqueeze(0)

    # Output = softmax(scores) @ V → [S, headDim]
    let attnOut = scores * vH

    # Insert head back
    let offset = h * headDim
    result[_, offset ..< offset + headDim] = attnOut

# ── Transformer block ─────────────────────────────────────────────

proc transformerBlock*(x: Tensor[float32], layer: LayerWeights,
                       seqLen: int): Tensor[float32] =
  ## RMSNorm → Attention → Residual → RMSNorm → MLP → Residual
  # Attention sub-block
  let xn = rmsNorm(x, layer.ln1)
  let q = xn * layer.attnWq.transpose()   # [S, nEmbd]
  let k = xn * layer.attnWk.transpose()
  let v = xn * layer.attnWv.transpose()
  let attnOut = fusedAttention(q, k, v, seqLen)
  let projected = attnOut * layer.attnWo.transpose()
  let afterAttn = x + projected

  # MLP sub-block
  let xn2 = rmsNorm(afterAttn, layer.ln2)
  let hidden = gelu(xn2 * layer.mlpFc1.transpose())
  let mlpOut = hidden * layer.mlpFc2.transpose()
  afterAttn + mlpOut

# ── Full forward pass ─────────────────────────────────────────────

proc forward*(m: Model, tokens: seq[int], seqLen: int): Tensor[float32] =
  ## Full forward pass. Returns logits [S, vocabSize].

  # Embedding lookup
  var x = zeros[float32](seqLen, nEmbd)
  for i in 0 ..< seqLen:
    x[i, _] = m.wte[tokens[i], _]

  # Post-embed norm
  x = rmsNorm(x, m.embedNorm)

  # Transformer layers
  for layer in m.layers:
    x = transformerBlock(x, layer, seqLen)

  # Final norm + project to vocab (weight-tied: lm_head = wte)
  x = rmsNorm(x, m.finalNorm)
  result = x * m.wte.transpose()  # [S, vocabSize]
