## model.nim — 103M parameter transformer definition
##
## Wide-and-shallow architecture: 8 layers, 1024 dim, 16 heads.
## Designed for memory experiments via selective weight retraining.
##
## Arraymancer tensors live on GPU natively — no FFI bridge needed.
## This is the clean rewrite of the OCaml version.

import arraymancer
import std/math

const
  nLayer*   = 8
  nEmbd*    = 1024
  blockSize* = 512
  nHead*    = 16
  headDim*  = nEmbd div nHead   # 64
  halfDim*  = headDim div 2     # 32
  ffnDim*   = 4 * nEmbd         # 4096

type
  LayerWeights* = object
    attnWq*, attnWk*, attnWv*, attnWo*: Tensor[float32]
    mlpFc1*, mlpFc2*: Tensor[float32]
    ln1*, ln2*: Tensor[float32]  # RMSNorm scale parameters

  Model* = object
    wte*: Tensor[float32]        # token embeddings [vocab, nEmbd]
    layers*: seq[LayerWeights]
    embedNorm*: Tensor[float32]  # post-embed RMSNorm scale
    finalNorm*: Tensor[float32]  # pre-lm_head RMSNorm scale
    vocabSize*: int

# ── RoPE frequency tables ─────────────────────────────────────────

let ropeFreqs* = block:
  var f = newTensor[float32](halfDim)
  for i in 0 ..< halfDim:
    f[i] = 1.0f / pow(10000.0f, float32(2 * i) / float32(headDim))
  f

let ropeCos* = block:
  var t = newTensor[float32](blockSize, halfDim)
  for pos in 0 ..< blockSize:
    for i in 0 ..< halfDim:
      t[pos, i] = cos(float32(pos) * ropeFreqs[i])
  t

let ropeSin* = block:
  var t = newTensor[float32](blockSize, halfDim)
  for pos in 0 ..< blockSize:
    for i in 0 ..< halfDim:
      t[pos, i] = sin(float32(pos) * ropeFreqs[i])
  t

# ── Initialization ────────────────────────────────────────────────

proc initMatrix(nout, nin: int, std: float32 = 0.08f): Tensor[float32] =
  ## Random normal initialization for weight matrices.
  result = randomNormalTensor[float32](nout, nin, 0.0f, std)

proc initOnes(dim: int): Tensor[float32] =
  ## All-ones initialization for RMSNorm scale parameters.
  result = ones[float32](dim)

proc initModel*(vocabSize: int): Model =
  ## Create a fresh 103M parameter model with random weights.
  ## Weight tying: lm_head shares wte (output projection = embedding).
  let residualStd = 0.08f / sqrt(float32(2 * nLayer))

  result.vocabSize = vocabSize
  result.wte = initMatrix(vocabSize, nEmbd)
  result.embedNorm = initOnes(nEmbd)
  result.finalNorm = initOnes(nEmbd)

  result.layers = newSeq[LayerWeights](nLayer)
  for i in 0 ..< nLayer:
    result.layers[i] = LayerWeights(
      attnWq: initMatrix(nEmbd, nEmbd),
      attnWk: initMatrix(nEmbd, nEmbd),
      attnWv: initMatrix(nEmbd, nEmbd),
      attnWo: initMatrix(nEmbd, nEmbd, residualStd),
      mlpFc1: initMatrix(ffnDim, nEmbd),
      mlpFc2: initMatrix(nEmbd, ffnDim, residualStd),
      ln1: initOnes(nEmbd),
      ln2: initOnes(nEmbd),
    )

proc paramCount*(m: Model): int =
  ## Count total trainable parameters.
  result = m.wte.size + m.embedNorm.size + m.finalNorm.size
  for layer in m.layers:
    result += layer.attnWq.size + layer.attnWk.size
    result += layer.attnWv.size + layer.attnWo.size
    result += layer.mlpFc1.size + layer.mlpFc2.size
    result += layer.ln1.size + layer.ln2.size
