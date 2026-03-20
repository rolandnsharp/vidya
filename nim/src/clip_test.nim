import gpu
import std/[strformat, math]

when isMainModule:
  gpuInit()
  
  # Create a buffer with large values
  var data = @[100.0f, 200.0f, 300.0f, 400.0f]
  let buf = toGpu(data)
  
  # Expected norm: sqrt(100^2 + 200^2 + 300^2 + 400^2) = sqrt(300000) ≈ 547.7
  echo "before clip: ", gpuDownload(buf)
  
  var ptrs = @[buf.data]
  var sizes = @[cint(4)]
  
  # Get norm
  let norm = gpu_grad_norm(addr ptrs[0], addr sizes[0], 1)
  echo &"norm = {norm:.4f} (expected {sqrt(300000.0):.4f})"
  
  # Clip to 1.0
  clipGradNorm(ptrs, sizes, 1.0f)
  
  let after = gpuDownload(buf)
  echo "after clip: ", after
  
  var normAfter = 0.0f
  for v in after: normAfter += v * v
  echo &"norm after = {sqrt(normAfter):.4f} (should be ~1.0)"
