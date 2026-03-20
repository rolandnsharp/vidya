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
import std/[math, random, streams, os, strformat]

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

## Parameter count estimate:
##   wte: vocab * 768
##   Per layer:
##     Q: 768*768 = 589K
##     K: 256*768 = 196K (4 KV heads * 64)
##     V: 256*768 = 196K
##     O: 768*768 = 589K
##     gate: 2048*768 = 1.57M
##     up: 2048*768 = 1.57M
##     down: 768*2048 = 1.57M
##     norms: 2*768 = 1.5K
##     total per layer: ~6.3M
##   12 layers: ~75M
##   embeddings (8K vocab): ~6M
##   Total: ~81M (need to bump ffnDim to hit 135M)
