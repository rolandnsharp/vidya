import gpu, autograd
import std/strformat

when isMainModule:
  gpuInit()
  trackingEnabled = true

  let n = 4
  var aHost = @[1.0f, 2.0f, 3.0f, 4.0f]
  let aBuf = toGpu(aHost)
  let aGrad = gpuCreate(n)

  # Simple: a -> matmul(identity) -> sum
  var wHost = newSeq[float32](n * n)
  for i in 0 ..< n: wHost[i * n + i] = 1.0f  # identity
  let wBuf = toGpu(wHost)
  let wGrad = gpuCreate(n * n)

  let aNode = paramNode(aBuf, aGrad, n)
  let wNode = paramNode(wBuf, wGrad, n * n)
  let y = agMatmul(wNode, aNode, 1, n, n)

  # Check y value
  echo "y = ", gpuDownload(y.data)

  # Seed gradient
  gpuUpload(y.grad, @[1.0f, 1.0f, 1.0f, 1.0f])
  
  # Manual backward of just this one node
  echo "calling y.backwardFn..."
  y.backwardFn()
  echo "aGrad after manual backward: ", gpuDownload(aGrad)
  echo "wGrad after manual backward: ", gpuDownload(wGrad)[0..3]

  # Now try full backward
  gpuZero(aGrad)
  gpuZero(wGrad)
  echo ""
  echo "calling backward(y)..."
  backward(y)
  echo "aGrad after backward(): ", gpuDownload(aGrad)
  echo "wGrad after backward(): ", gpuDownload(wGrad)[0..3]
