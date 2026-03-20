import gpu
import std/strformat

when isMainModule:
  gpuInit()
  
  # Test: does rmsnormAffineFwd actually use gamma?
  var x = @[1.0f, 2.0f, 3.0f, 4.0f]
  var g1 = @[1.0f, 1.0f, 1.0f, 1.0f]
  var g2 = @[2.0f, 2.0f, 2.0f, 2.0f]  # double gamma
  
  let xBuf = toGpu(x)
  let g1Buf = toGpu(g1)
  let g2Buf = toGpu(g2)
  let y1 = gpuCreate(4)
  let y2 = gpuCreate(4)
  let rms1 = gpuCreate(1)
  let rms2 = gpuCreate(1)
  
  rmsnormAffineFwd(xBuf, g1Buf, y1, rms1, 1, 4)
  rmsnormAffineFwd(xBuf, g2Buf, y2, rms2, 1, 4)
  
  echo "gamma=1: ", gpuDownload(y1)
  echo "gamma=2: ", gpuDownload(y2)
  echo "should be 2x"
