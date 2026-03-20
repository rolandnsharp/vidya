## ag_train_v2.nim — V2 training: GQA + SwiGLU + dropout + big data

import gpu, model_v2, autograd, ag_forward_v2, bpe
import std/[math, times, strformat, os, random, streams]

const
  learningRate = 0.0001f   # conservative — proven stable in V1 up to step 3800
  minLr = 0.00003f
  beta1 = 0.9f
  beta2 = 0.95f
  weightDecay = 0.1f
  warmupSteps = 2000
  gradAccumSteps = 8
  logInterval = 50
  checkpointInterval = 2500
  maxGradNorm = 1.0f
  bpeMerges = 4000         # good enough — retrain on books later

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
    let cosDecay = 0.5f * (1.0f + cos(PI.float32 * progress))
    minLr + (learningRate - minLr) * cosDecay

when isMainModule:
  let baseDir = getAppDir().parentDir()
  let vidyaRoot = baseDir.parentDir()
  let dataFile = vidyaRoot / "chat_input_all.txt"  # 2.4M conversations
  let tokenizerFile = vidyaRoot / "tokenizer_v2.bin"
  let checkpointBase = vidyaRoot / "nimllm_v2"

  gpuInit()
  echo "nimllm v2 starting..."

  # Tokenizer — 8K merges on big dataset
  var tok: Tokenizer
  if fileExists(tokenizerFile):
    echo &"loading tokenizer from {tokenizerFile}"
    tok = loadTokenizer(tokenizerFile)
  else:
    echo &"training tokenizer ({bpeMerges} merges) on {dataFile}..."
    echo "  (this takes a while on 2.5GB — go get coffee)"
    let docs = loadDocs(dataFile)
    tok = trainBpe(docs, bpeMerges)
    saveTokenizer(tok, tokenizerFile)
  echo &"vocab: {tok.vocab.len}"

  # Model
  echo "init V2 model on GPU..."
  var m = initGpuModelV2(tok.vocab.len)
  echo &"params: {paramCountV2(m)}"

  # Load checkpoint if exists
  let latestCp = checkpointBase & "_latest.bin"
  if fileExists(latestCp):
    loadCheckpointV2(m, latestCp)

  var params = collectParamsV2(m)
  echo &"  {params.len} parameter tensors"

  # Adam state
  var adamM = newSeq[GpuBuf](params.len)
  var adamV = newSeq[GpuBuf](params.len)
  for i in 0 ..< params.len:
    adamM[i] = gpuCreate(params[i][].numel)
    adamV[i] = gpuCreate(params[i][].numel)

  # Tokenize data — use first 50K docs for speed.
  # Full 2.4M would take hours to encode with 8K merges.
  const maxTrainDocs = 50000

  let tokenCacheFile = vidyaRoot / "tokens_v2.bin"
  var tokenizedDocs: seq[seq[int32]]

  if fileExists(tokenCacheFile):
    echo "loading cached tokens from ", tokenCacheFile, "..."
    let s = newFileStream(tokenCacheFile, fmRead)
    let nDocs = s.readInt32().int
    for i in 0 ..< nDocs:
      let n = s.readInt32().int
      var ids = newSeq[int32](n)
      for j in 0 ..< n: ids[j] = s.readInt32()
      tokenizedDocs.add(ids)
    s.close()
    echo &"  loaded {tokenizedDocs.len} docs from cache"
  else:
    echo "tokenizing data (first run, will cache)..."
    let allDocs = loadDocs(dataFile)
    let docs = if allDocs.len > maxTrainDocs: allDocs[0 ..< maxTrainDocs]
               else: allDocs
    echo &"  {docs.len} docs (of {allDocs.len} total)"
    let t0 = cpuTime()
    for i, doc in docs:
      let ids = tok.encode(doc)
      var ids32 = newSeq[int32](ids.len)
      for j in 0 ..< ids.len: ids32[j] = int32(ids[j])
      if ids32.len >= 3:
        tokenizedDocs.add(ids32)
      if (i + 1) mod 10000 == 0:
        echo &"    {i + 1}/{docs.len} tokenized"
    echo &"  tokenized {tokenizedDocs.len} docs in {cpuTime() - t0:.2f}s"

    echo "  saving token cache..."
    let s = newFileStream(tokenCacheFile, fmWrite)
    s.write(int32(tokenizedDocs.len))
    for doc in tokenizedDocs:
      s.write(int32(doc.len))
      for id in doc: s.write(id)
    s.close()
    echo "  cached to ", tokenCacheFile

  # Shuffle
  randomize()
  var shuffled = newSeq[int](tokenizedDocs.len)
  for i in 0 ..< shuffled.len: shuffled[i] = i
  shuffle(shuffled)
  echo "  shuffled"

  # Grad pointer arrays
  var gradPtrs = newSeq[pointer](params.len)
  var gradSizes = newSeq[cint](params.len)
  for i in 0 ..< params.len:
    gradPtrs[i] = params[i][].grad.data
    gradSizes[i] = cint(params[i][].numel)

  # Zero grads
  proc zeroGradV2() =
    for p in params: gpuZero(p[].grad)

  # Training
  let numSteps = tokenizedDocs.len * 3  # 3 epochs
  let numOptSteps = numSteps div gradAccumSteps
  echo &"training {numSteps} steps ({numSteps div tokenizedDocs.len} epochs, {numOptSteps} optimizer steps)..."
  trainingMode = true
  trackingEnabled = true
  let tStart = cpuTime()
  var lossSum = 0.0f
  var stepCount = 0
  var microStep = 0

  for step in 0 ..< numSteps:
    let docIdx = shuffled[step mod shuffled.len]
    let tokens = tokenizedDocs[docIdx]

    let (lossNode, lossVal) = agComputeLossV2(m, tokens)
    if lossNode == nil:
      freeStepAllocations()
      continue
    if lossVal != lossVal or lossVal > 100:
      freeStepAllocations()
      continue

    backward(lossNode)
    freeStepAllocations()

    lossSum += lossVal
    microStep += 1

    if microStep >= gradAccumSteps:
      for i in 0 ..< params.len:
        gpu_scale(params[i][].grad.data, 1.0f / float32(gradAccumSteps),
                  params[i][].grad.data, cint(params[i][].numel))

      let preNorm = gpu_grad_norm(addr gradPtrs[0], addr gradSizes[0], cint(params.len))
      if stepCount mod 50 == 0:
        echo &"  grad_norm: {preNorm:.2f}"
      if preNorm != preNorm or preNorm > 1000.0:
        echo &"  WARNING: bad gradient norm {preNorm:.2f} at step {stepCount}, skipping"
        zeroGradV2()
        microStep = 0
        stepCount += 1
        continue
      clipGradNorm(gradPtrs, gradSizes, maxGradNorm)

      let lr = getLr(stepCount, numOptSteps)
      let bc1 = 1.0f / (1.0f - pow(beta1, float32(stepCount + 1)))
      let bc2 = 1.0f / (1.0f - pow(beta2, float32(stepCount + 1)))
      for i in 0 ..< params.len:
        gpu_adamw(params[i][].data.data, params[i][].grad.data,
                  adamM[i].data, adamV[i].data,
                  lr, beta1, beta2, bc1, bc2, weightDecay,
                  cint(params[i][].numel))

      zeroGradV2()
      microStep = 0
      stepCount += 1

    if stepCount > 0 and stepCount mod checkpointInterval == 0 and microStep == 0:
      saveCheckpointV2(m, &"{checkpointBase}_{stepCount}.bin")
      saveCheckpointV2(m, &"{checkpointBase}_latest.bin")

    if stepCount > 0 and stepCount mod logInterval == 0 and microStep == 0:
      let elapsed = cpuTime() - tStart
      let stepsPerSec = float(stepCount) / elapsed
      let stepsLeft = float(numOptSteps - stepCount)
      let eta = stepsLeft / max(stepsPerSec, 0.01)
      let lr = getLr(stepCount, numOptSteps)
      echo &"step {stepCount:>6} / {numOptSteps} | loss {lossSum / float32(logInterval * gradAccumSteps):.4f} | lr {lr:.6f} | {stepsPerSec:.1f} opt/s | {formatDuration(elapsed)} elapsed | {formatDuration(eta)} remaining"
      lossSum = 0.0f

  saveCheckpointV2(m, &"{checkpointBase}_final.bin")
  echo "training complete"
