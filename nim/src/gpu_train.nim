## gpu_train.nim — GPU training loop for Vidya (Nim)
##
## Forward-only training (no autograd yet — loss computation validates
## the architecture). Autograd comes next — for now we verify the
## forward pass produces decreasing loss with manual gradient updates.
##
## TODO: Add backward pass with autograd. For now this is forward-only
## loss monitoring to validate the pipeline end-to-end.

import gpu, gpu_model, gpu_forward, bpe
import std/[math, times, strformat, os, random]

const
  learningRate = 0.0003f
  beta1 = 0.9f
  beta2 = 0.999f
  maxGradNorm = 1.0f
  warmupSteps = 5000
  logInterval = 100
  checkpointInterval = 2500

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

# ── Training ──────────────────────────────────────────────────────

proc train*(m: var GpuModel, tokenizedDocs: seq[seq[int32]], numSteps: int) =
  ## Training loop. Forward pass computes loss on GPU.
  ## Loss should decrease from ~7.7 (random) toward ~3-4 (learned).
  echo &"training {numSteps} steps on {tokenizedDocs.len} docs"
  echo &"  lr={learningRate}, warmup={warmupSteps}"
  echo &"  log every {logInterval}, checkpoint every {checkpointInterval}"

  # Pre-allocate all scratch buffers — no per-step allocation
  echo "  allocating scratch buffers..."
  var scratch = initScratch(blockSize, m.vocabSize)
  var x = gpuCreate(blockSize * nEmbd)
  echo "  scratch ready"

  let tStart = cpuTime()
  var lossSum = 0.0f
  var stepCount = 0

  for step in 0 ..< numSteps:
    let tokens = tokenizedDocs[step mod tokenizedDocs.len]
    if tokens.len < 3: continue

    let loss = gpuComputeLoss(m, tokens, scratch, x)

    if loss != loss or loss > 1e6:
      echo &"WARNING: loss={loss} at step {step}, skipping"
      continue

    lossSum += loss
    stepCount += 1

    if stepCount mod logInterval == 0:
      let elapsed = cpuTime() - tStart
      let stepsPerSec = float(stepCount) / elapsed
      let stepsLeft = float(numSteps - step - 1)
      let eta = stepsLeft / stepsPerSec
      let avgLoss = lossSum / float32(logInterval)
      echo &"step {step + 1:>6} / {numSteps} | loss {avgLoss:.4f} | {stepsPerSec:.1f} steps/s | {formatDuration(elapsed)} elapsed | {formatDuration(eta)} remaining"
      lossSum = 0.0f

# ── Entry point ───────────────────────────────────────────────────

when isMainModule:
  let baseDir = getAppDir().parentDir()
  let vidyaRoot = baseDir.parentDir()
  let dataFile = vidyaRoot / "chat_input.txt"
  let tokenizerFile = vidyaRoot / "tokenizer_nim.bin"

  gpuInit()
  echo "vidya (nim/gpu) starting..."

  # Tokenizer
  var tok: Tokenizer
  if fileExists(tokenizerFile):
    echo &"loading tokenizer from {tokenizerFile}"
    tok = loadTokenizer(tokenizerFile)
  else:
    echo "training tokenizer..."
    let docs = loadDocs(dataFile)
    tok = trainBpe(docs)
    saveTokenizer(tok, tokenizerFile)
  echo &"vocab: {tok.vocab.len}"

  # Model
  echo "initialising model on GPU..."
  var m = initGpuModel(tok.vocab.len)
  echo &"params: {paramCount(m)}"

  # Data
  echo "loading and tokenizing data..."
  let docs = loadDocs(dataFile)
  echo &"  {docs.len} docs"
  let t0 = cpuTime()
  var tokenizedDocs: seq[seq[int32]]
  for doc in docs:
    let ids = tok.encode(doc)
    var ids32 = newSeq[int32](ids.len)
    for i in 0 ..< ids.len: ids32[i] = int32(ids[i])
    tokenizedDocs.add(ids32)
  echo &"  tokenized in {cpuTime() - t0:.2f}s"

  # Train
  train(m, tokenizedDocs, numSteps = 200000)
