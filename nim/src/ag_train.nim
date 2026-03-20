## ag_train.nim — Training with autograd (weights actually update)
##
## This is the real thing. Forward builds graph, backward propagates
## gradients, Adam updates weights. Loss should decrease.

import gpu, gpu_model, autograd, ag_forward, bpe
import std/[math, times, strformat, os]

const
  learningRate = 0.00002f  # very conservative — batch size 1 is noisy
  beta1 = 0.9f
  beta2 = 0.999f
  warmupSteps = 200
  logInterval = 50
  checkpointInterval = 2500
  maxGradNorm = 1.0f

proc formatDuration(secs: float): string =
  let h = int(secs) div 3600
  let m = (int(secs) mod 3600) div 60
  let s = int(secs) mod 60
  if h > 0: &"{h}h{m:02d}m{s:02d}s"
  elif m > 0: &"{m}m{s:02d}s"
  else: &"{s}s"

proc getLr(step, numSteps: int): float32 =
  if step < warmupSteps:
    learningRate * float32(step) / float32(warmupSteps)
  else:
    let progress = float32(step - warmupSteps) / float32(numSteps - warmupSteps)
    learningRate * 0.5f * (1.0f + cos(PI.float32 * progress))

when isMainModule:
  let baseDir = getAppDir().parentDir()
  let vidyaRoot = baseDir.parentDir()
  let dataFile = vidyaRoot / "chat_input.txt"
  let tokenizerFile = vidyaRoot / "tokenizer_nim.bin"

  gpuInit()
  echo "vidya (nim/autograd) starting..."

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
  echo "init model on GPU..."
  var m = initGpuModel(tok.vocab.len)
  echo &"params: {paramCount(m)}"

  # Collect param pointers for optimizer
  var params = collectParams(m)
  echo &"  {params.len} parameter tensors"

  # Adam state: per-param m and v buffers
  var adamM = newSeq[GpuBuf](params.len)
  var adamV = newSeq[GpuBuf](params.len)
  for i in 0 ..< params.len:
    adamM[i] = gpuCreate(params[i][].numel)
    adamV[i] = gpuCreate(params[i][].numel)

  # Data
  echo "loading data..."
  let docs = loadDocs(dataFile)
  let t0 = cpuTime()
  var tokenizedDocs: seq[seq[int32]]
  for doc in docs:
    let ids = tok.encode(doc)
    var ids32 = newSeq[int32](ids.len)
    for i in 0 ..< ids.len: ids32[i] = int32(ids[i])
    tokenizedDocs.add(ids32)
  echo &"  tokenized {tokenizedDocs.len} docs in {cpuTime() - t0:.2f}s"

  # Training
  let numSteps = 200000
  echo &"training {numSteps} steps..."
  trackingEnabled = true
  let tStart = cpuTime()
  var lossSum = 0.0f
  var stepCount = 0

  for step in 0 ..< numSteps:
    let tokens = tokenizedDocs[step mod tokenizedDocs.len]
    if tokens.len < 3: continue

    # Zero gradients
    zeroGrad(params)

    # Forward + loss (builds autograd graph, all allocs tracked)
    let (lossNode, lossVal) = agComputeLoss(m, tokens)
    if lossNode == nil:
      freeStepAllocations()
      continue
    if lossVal != lossVal or lossVal > 100:
      freeStepAllocations()
      continue

    # Backward (propagates gradients, may allocate more tracked buffers)
    backward(lossNode)

    # Gradient clipping — proper L2 norm on GPU, clip to maxGradNorm
    var gradPtrs = newSeq[pointer](params.len)
    var gradSizes = newSeq[cint](params.len)
    for i in 0 ..< params.len:
      gradPtrs[i] = params[i][].grad.data
      gradSizes[i] = cint(params[i][].numel)
    clipGradNorm(gradPtrs, gradSizes, maxGradNorm)

    # Adam update
    let lr = getLr(step, numSteps)
    let bc1 = 1.0f / (1.0f - pow(beta1, float32(step + 1)))
    let bc2 = 1.0f / (1.0f - pow(beta2, float32(step + 1)))
    for i in 0 ..< params.len:
      adamStep(params[i][].data, params[i][].grad, adamM[i], adamV[i],
               lr, beta1, beta2, bc1, bc2)

    # Free ALL intermediate GPU buffers from this step
    freeStepAllocations()

    lossSum += lossVal
    stepCount += 1

    if stepCount mod logInterval == 0:
      let elapsed = cpuTime() - tStart
      let stepsPerSec = float(stepCount) / elapsed
      let stepsLeft = float(numSteps - step - 1)
      let eta = stepsLeft / max(stepsPerSec, 0.01)
      echo &"step {step + 1:>6} / {numSteps} | loss {lossSum / float32(logInterval):.4f} | lr {lr:.6f} | {stepsPerSec:.1f} steps/s | {formatDuration(elapsed)} elapsed | {formatDuration(eta)} remaining"
      lossSum = 0.0f
