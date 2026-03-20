## Quick test: allocate GPU memory, upload, matmul, download.
import gpu
import std/strformat

when isMainModule:
  echo "initialising GPU..."
  gpuInit()

  echo "allocating..."
  var hostA = @[1f, 0, 0, 0, 1, 0, 0, 0, 1]  # identity
  var hostB = @[1f, 2, 3, 4, 5, 6, 7, 8, 9]

  let a = toGpu(hostA)
  let b = toGpu(hostB)
  let c = gpuCreate(9)

  echo "matmul (identity @ B should equal B)..."
  gpuSgemm(0, 3, 3, 3, a, b, c)

  let result = gpuDownload(c)
  echo &"result: {result}"
  echo &"expected: {hostB}"

  if result == hostB:
    echo "GPU MATMUL WORKS!"
  else:
    echo "MISMATCH — debug needed"
