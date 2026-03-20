## Quick test: init GPU model, run one forward pass, compute loss.
import gpu, gpu_model, gpu_forward
import std/[strformat, times, math]

when isMainModule:
  gpuInit()
  echo "initialising 103M model on GPU..."
  let t0 = cpuTime()
  var m = initGpuModel(2188)
  echo &"  {paramCount(m)} params in {cpuTime() - t0:.2f}s"

  # Fake tokens for testing
  var tokens = newSeq[int32](100)
  for i in 0 ..< 100: tokens[i] = int32(i mod 2188)

  echo "running forward pass..."
  let t1 = cpuTime()
  let loss = gpuComputeLoss(m, tokens)
  echo &"  loss = {loss:.4f} in {cpuTime() - t1:.2f}s"
  echo &"  expected ~{ln(2188.0):.4f} (random weights)"
  echo "done!"
