## model_v2.nim — V2 architecture targeting SmolLM2-135M parity
##
## Changes from V1:
##   - SwiGLU activation (replaces GELU) — needs gate + up projections
##   - GQA: 12 Q heads, 4 KV heads (saves VRAM, same quality)
##   - 768 dim, 12 layers (deeper than V1's 8 layers)
##   - 1024 context window
##   - Designed for 8K+ vocab trained on 2.4M conversations
##
## Note: SwiGLU changes the FFN structure. Instead of:
##   hidden = gelu(x @ fc1)     — one up-projection
##   out = hidden @ fc2          — one down-projection
## We have:
##   gate = x @ Wgate            — gate projection
##   up = x @ Wup               — up projection
##   hidden = swiglu(gate, up)   — gated activation
##   out = hidden @ Wdown        — down projection
##
## The FFN hidden dim is typically 8/3 * nEmbd (rounded) for SwiGLU
## to match parameter count of 4 * nEmbd GELU FFN with two projections.

import gpu
import std/[math, random, streams, strformat]

const
  nLayerV2*   = 12
  nEmbdV2*    = 768
  blockSizeV2* = 1024
  nHeadV2*    = 12           # Q heads
  nKvHeadV2*  = 4            # KV heads (GQA: 3 Q heads per KV head)
  headDimV2*  = nEmbdV2 div nHeadV2   # 64
  halfDimV2*  = headDimV2 div 2        # 32
  ffnDimV2*   = (8 * nEmbdV2) div 3   # 2048 — SwiGLU-adjusted
  kvRepeatV2* = nHeadV2 div nKvHeadV2  # 3 — each KV head serves 3 Q heads

type
  GpuParamV2* = object
    data*: GpuBuf
    grad*: GpuBuf
    numel*: int

  GpuLayerV2* = object
    attnWq*: GpuParamV2          # [nEmbd, nEmbd]
    attnWk*: GpuParamV2          # [nKvHead * headDim, nEmbd] — fewer KV heads
    attnWv*: GpuParamV2          # [nKvHead * headDim, nEmbd]
    attnWo*: GpuParamV2          # [nEmbd, nEmbd]
    qNorm*, kNorm*: GpuParamV2   # QK-Norm: bounds Q·K^T, prevents overflow
    mlpWgate*: GpuParamV2        # [ffnDim, nEmbd] — SwiGLU gate
    mlpWup*: GpuParamV2          # [ffnDim, nEmbd] — SwiGLU up
    mlpWdown*: GpuParamV2        # [nEmbd, ffnDim] — down projection
    ln1*, ln2*: GpuParamV2       # RMSNorm scales

  GpuModelV2* = object
    wte*: GpuParamV2
    layers*: seq[GpuLayerV2]
    embedNorm*: GpuParamV2
    finalNorm*: GpuParamV2
    vocabSize*: int
    ropeCos*: GpuBuf
    ropeSin*: GpuBuf

proc paramCountV2*(m: GpuModelV2): int =
  result = m.wte.numel + m.embedNorm.numel + m.finalNorm.numel
  for layer in m.layers:
    result += layer.attnWq.numel + layer.attnWk.numel
    result += layer.attnWv.numel + layer.attnWo.numel
    result += layer.mlpWgate.numel + layer.mlpWup.numel + layer.mlpWdown.numel
    result += layer.ln1.numel + layer.ln2.numel

proc randomNormalV2(n: int, std: float32): seq[float32] =
  result = newSeq[float32](n)
  var i = 0
  while i < n - 1:
    let u1 = rand(1.0).float32
    let u2 = rand(1.0).float32
    let mag = std * sqrt(-2.0f * ln(max(u1, 1e-10f)))
    result[i]     = mag * cos(2.0f * PI.float32 * u2)
    result[i + 1] = mag * sin(2.0f * PI.float32 * u2)
    i += 2
  if i < n:
    let u1 = rand(1.0).float32
    let u2 = rand(1.0).float32
    result[i] = std * sqrt(-2.0f * ln(max(u1, 1e-10f))) * cos(2.0f * PI.float32 * u2)

proc makeParamV2(n: int, std: float32 = 0.02f): GpuParamV2 =
  let hostData = randomNormalV2(n, std)
  result.numel = n
  result.data = toGpu(hostData)
  result.grad = gpuCreate(n)

proc makeOnesParamV2(n: int): GpuParamV2 =
  var hostData = newSeq[float32](n)
  for i in 0 ..< n: hostData[i] = 1.0f
  result.numel = n
  result.data = toGpu(hostData)
  result.grad = gpuCreate(n)

proc initGpuModelV2*(vocabSize: int): GpuModelV2 =
  randomize(42)
  let residualStd = 0.02f / sqrt(float32(2 * nLayerV2))
  let nKvDim = nKvHeadV2 * headDimV2

  echo "  uploading wte..."
  result.vocabSize = vocabSize
  result.wte = makeParamV2(vocabSize * nEmbdV2)
  result.embedNorm = makeOnesParamV2(nEmbdV2)
  result.finalNorm = makeOnesParamV2(nEmbdV2)

  result.layers = newSeq[GpuLayerV2](nLayerV2)
  for i in 0 ..< nLayerV2:
    echo "  uploading layer ", i, "..."
    result.layers[i] = GpuLayerV2(
      attnWq: makeParamV2(nEmbdV2 * nEmbdV2),
      attnWk: makeParamV2(nKvDim * nEmbdV2),
      attnWv: makeParamV2(nKvDim * nEmbdV2),
      attnWo: makeParamV2(nEmbdV2 * nEmbdV2, residualStd),
      qNorm: makeOnesParamV2(nEmbdV2),
      kNorm: makeOnesParamV2(nKvDim),
      mlpWgate: makeParamV2(ffnDimV2 * nEmbdV2),
      mlpWup: makeParamV2(ffnDimV2 * nEmbdV2),
      mlpWdown: makeParamV2(nEmbdV2 * ffnDimV2, residualStd),
      ln1: makeOnesParamV2(nEmbdV2),
      ln2: makeOnesParamV2(nEmbdV2),
    )

  echo "  uploading RoPE tables..."
  var cosData = newSeq[float32](blockSizeV2 * halfDimV2)
  var sinData = newSeq[float32](blockSizeV2 * halfDimV2)
  for pos in 0 ..< blockSizeV2:
    for i in 0 ..< halfDimV2:
      let freq = 1.0f / pow(10000.0f, float32(2 * i) / float32(headDimV2))
      cosData[pos * halfDimV2 + i] = cos(float32(pos) * freq)
      sinData[pos * halfDimV2 + i] = sin(float32(pos) * freq)
  result.ropeCos = toGpu(cosData)
  result.ropeSin = toGpu(sinData)

proc collectParamsV2*(m: var GpuModelV2): seq[ptr GpuParamV2] =
  result.add(addr m.wte)
  result.add(addr m.embedNorm)
  for i in 0 ..< m.layers.len:
    result.add(addr m.layers[i].attnWq)
    result.add(addr m.layers[i].attnWk)
    result.add(addr m.layers[i].attnWv)
    result.add(addr m.layers[i].attnWo)
    result.add(addr m.layers[i].qNorm)
    result.add(addr m.layers[i].kNorm)
    result.add(addr m.layers[i].mlpWgate)
    result.add(addr m.layers[i].mlpWup)
    result.add(addr m.layers[i].mlpWdown)
    result.add(addr m.layers[i].ln1)
    result.add(addr m.layers[i].ln2)
  result.add(addr m.finalNorm)

proc writeParamV2(s: FileStream, p: GpuParamV2) =
  let data = gpuDownload(p.data)
  s.write(int32(data.len))
  for v in data: s.write(v)

proc readParamV2(s: FileStream, p: var GpuParamV2) =
  let n = s.readInt32().int
  var data = newSeq[float32](n)
  for i in 0 ..< n: data[i] = s.readFloat32()
  gpuUpload(p.data, data)

proc saveCheckpointV2*(m: GpuModelV2, filename: string) =
  echo &"saving checkpoint to {filename}..."
  let s = newFileStream(filename, fmWrite)
  defer: s.close()
  s.write(int32(m.vocabSize))
  s.write(int32(paramCountV2(m)))
  s.writeParamV2(m.wte)
  s.writeParamV2(m.embedNorm)
  for i in 0 ..< m.layers.len:
    s.writeParamV2(m.layers[i].attnWq)
    s.writeParamV2(m.layers[i].attnWk)
    s.writeParamV2(m.layers[i].attnWv)
    s.writeParamV2(m.layers[i].attnWo)
    s.writeParamV2(m.layers[i].qNorm)
    s.writeParamV2(m.layers[i].kNorm)
    s.writeParamV2(m.layers[i].mlpWgate)
    s.writeParamV2(m.layers[i].mlpWup)
    s.writeParamV2(m.layers[i].mlpWdown)
    s.writeParamV2(m.layers[i].ln1)
    s.writeParamV2(m.layers[i].ln2)
  s.writeParamV2(m.finalNorm)
  echo "  done"

proc loadCheckpointV2*(m: var GpuModelV2, filename: string) =
  echo &"loading checkpoint from {filename}..."
  let s = newFileStream(filename, fmRead)
  defer: s.close()
  let vocabSize = s.readInt32().int
  let nParams = s.readInt32().int
  echo &"  vocab={vocabSize} params={nParams}"
  s.readParamV2(m.wte)
  s.readParamV2(m.embedNorm)
  for i in 0 ..< m.layers.len:
    s.readParamV2(m.layers[i].attnWq)
    s.readParamV2(m.layers[i].attnWk)
    s.readParamV2(m.layers[i].attnWv)
    s.readParamV2(m.layers[i].attnWo)
    s.readParamV2(m.layers[i].qNorm)
    s.readParamV2(m.layers[i].kNorm)
    s.readParamV2(m.layers[i].mlpWgate)
    s.readParamV2(m.layers[i].mlpWup)
    s.readParamV2(m.layers[i].mlpWdown)
    s.readParamV2(m.layers[i].ln1)
    s.readParamV2(m.layers[i].ln2)
  s.readParamV2(m.finalNorm)
  echo "  done"
