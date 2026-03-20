## ag_train.nim — Training with autograd (weights actually update)
##
## This is the real thing. Forward builds graph, backward propagates
## gradients, Adam updates weights. Loss should decrease.

import gpu, gpu_model, autograd, ag_forward, bpe
import std/[math, times, strformat, os, random]

const
  learningRate = 0.00005f    # half the previous — stable longer
  minLr = 0.000005f
  beta1 = 0.9f
  beta2 = 0.95f
  weightDecay = 0.1f
  warmupSteps = 2000
  gradAccumSteps = 1       # batch size 1 — matches OCaml recipe
  logInterval = 100
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
  ## Cosine decay with warmup and minimum LR floor.
  if step < warmupSteps:
    learningRate * float32(step) / float32(warmupSteps)
  else:
    let progress = float32(step - warmupSteps) / float32(numSteps - warmupSteps)
    let cosDecay = 0.5f * (1.0f + cos(PI.float32 * progress))
    minLr + (learningRate - minLr) * cosDecay

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

  # Load checkpoint if one exists
  let latestCp = vidyaRoot / "nimllm_latest.bin"
  if fileExists(latestCp):
    loadCheckpoint(m, latestCp)

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
  # Shuffle training data so each run sees different order
  randomize()
  var shuffled = newSeq[int](tokenizedDocs.len)
  for i in 0 ..< shuffled.len: shuffled[i] = i
  shuffle(shuffled)
  echo "  shuffled document order"

  let numSteps = 200000
  echo &"training {numSteps} steps (grad_accum={gradAccumSteps}, effective_batch={gradAccumSteps})..."
  trackingEnabled = true
  let tStart = cpuTime()
  var lossSum = 0.0f
  var stepCount = 0
  var microStep = 0  # counts within gradient accumulation

  # Pre-build grad pointer arrays for clipping
  var gradPtrs = newSeq[pointer](params.len)
  var gradSizes = newSeq[cint](params.len)
  for i in 0 ..< params.len:
    gradPtrs[i] = params[i][].grad.data
    gradSizes[i] = cint(params[i][].numel)

  for step in 0 ..< numSteps:
    let docIdx = shuffled[step mod shuffled.len]
    let tokens = tokenizedDocs[docIdx]
    if tokens.len < 3: continue

    # Forward + loss (builds autograd graph)
    let (lossNode, lossVal) = agComputeLoss(m, tokens)
    if lossNode == nil:
      freeStepAllocations()
      continue
    if lossVal != lossVal or lossVal > 100:
      freeStepAllocations()
      continue

    # Backward — gradients accumulate across micro-steps
    backward(lossNode)
    freeStepAllocations()

    lossSum += lossVal
    microStep += 1

    # Only update weights every gradAccumSteps micro-steps
    if microStep >= gradAccumSteps:
      # Scale gradients by 1/gradAccumSteps to average
      for i in 0 ..< params.len:
        gpu_scale(params[i][].grad.data, 1.0f / float32(gradAccumSteps),
                  params[i][].grad.data, cint(params[i][].numel))

      # Gradient clipping — skip if NaN
      let preNorm = gpu_grad_norm(addr gradPtrs[0], addr gradSizes[0], cint(params.len))
      if stepCount mod 50 == 0:
        echo &"  grad_norm: {preNorm:.2f} → clipped to {maxGradNorm}"
      if preNorm != preNorm or preNorm > 50.0:
        # NaN or extreme gradient — zero grads and skip this step
        echo &"  WARNING: bad gradient norm {preNorm:.2f} at step {stepCount}, skipping"
        zeroGrad(params)
        microStep = 0
        stepCount += 1
        continue
      clipGradNorm(gradPtrs, gradSizes, maxGradNorm)

      # AdamW update with weight decay
      let lr = getLr(stepCount, numSteps div gradAccumSteps)
      let bc1 = 1.0f / (1.0f - pow(beta1, float32(stepCount + 1)))
      let bc2 = 1.0f / (1.0f - pow(beta2, float32(stepCount + 1)))
      for i in 0 ..< params.len:
        adamwStep(params[i][].data, params[i][].grad, adamM[i], adamV[i],
                  lr, beta1, beta2, bc1, bc2, weightDecay)

      # Zero gradients for next accumulation cycle
      zeroGrad(params)
      microStep = 0
      stepCount += 1

    # Save checkpoint every checkpointInterval optimizer steps
    if stepCount > 0 and stepCount mod checkpointInterval == 0 and microStep == 0:
      let cpFile = vidyaRoot / &"nimllm_{stepCount}.bin"
      saveCheckpoint(m, cpFile)
      # Also save as latest for easy resume
      saveCheckpoint(m, vidyaRoot / "nimllm_latest.bin")

    if stepCount > 0 and stepCount mod logInterval == 0 and microStep == 0:
      let elapsed = cpuTime() - tStart
      let stepsPerSec = float(stepCount) / elapsed
      let totalOptSteps = numSteps div gradAccumSteps
      let stepsLeft = float(totalOptSteps - stepCount)
      let eta = stepsLeft / max(stepsPerSec, 0.01)
      let lr = getLr(stepCount, totalOptSteps)
      echo &"step {stepCount:>6} / {totalOptSteps} | loss {lossSum / float32(logInterval * gradAccumSteps):.4f} | lr {lr:.6f} | {stepsPerSec:.1f} opt/s | {formatDuration(elapsed)} elapsed | {formatDuration(eta)} remaining"
      lossSum = 0.0f
