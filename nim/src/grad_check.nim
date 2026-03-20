## grad_check.nim — Find the broken backward pass
##
## For each autograd operation, compute gradients two ways:
## 1. Our backward function (analytic)
## 2. Finite differences (numeric): df/dx ≈ (f(x+eps) - f(x-eps)) / (2*eps)
##
## If they don't match, that op has a bug.

import gpu, gpu_model, autograd
import std/[strformat, math]

const eps = 1e-3f  # finite difference step (float32 needs larger eps)

proc maxAbsDiff(a, b: seq[float32]): (float32, int) =
  ## Return max |a[i] - b[i]| and which index.
  var maxD = 0.0f
  var maxI = 0
  for i in 0 ..< min(a.len, b.len):
    let d = abs(a[i] - b[i])
    if d > maxD: (maxD, maxI) = (d, i)
  (maxD, maxI)

proc relError(analytic, numeric: seq[float32]): float32 =
  ## Relative error: max|a-n| / max(max|a|, max|n|, 1e-8)
  var maxA = 0.0f
  var maxN = 0.0f
  var maxDiff = 0.0f
  for i in 0 ..< min(analytic.len, numeric.len):
    maxA = max(maxA, abs(analytic[i]))
    maxN = max(maxN, abs(numeric[i]))
    maxDiff = max(maxDiff, abs(analytic[i] - numeric[i]))
  maxDiff / max(max(maxA, maxN), 1e-8f)

proc checkOp(name: string, analytic, numeric: seq[float32]) =
  let rel = relError(analytic, numeric)
  let (maxD, maxI) = maxAbsDiff(analytic, numeric)
  let status = if rel < 0.01: "OK" elif rel < 0.1: "WARN" else: "FAIL"
  echo &"  {status:4} {name:25} rel_err={rel:.6f} max_diff={maxD:.6f} at [{maxI}]"
  if rel > 0.01:
    echo &"       analytic[{maxI}]={analytic[maxI]:.6f} numeric[{maxI}]={numeric[maxI]:.6f}"

# ── Test individual operations ────────────────────────────────────

proc testMatmul() =
  echo "testing agMatmul..."
  let s = 4
  let m = 8
  let n = 6
  # Random inputs
  var wHost = newSeq[float32](m * n)
  var xHost = newSeq[float32](s * n)
  for i in 0 ..< wHost.len: wHost[i] = float32(i mod 7) * 0.1f - 0.3f
  for i in 0 ..< xHost.len: xHost[i] = float32(i mod 5) * 0.1f - 0.2f

  # Analytic gradient
  let wBuf = toGpu(wHost)
  let xBuf = toGpu(xHost)
  let wNode = newNode(wBuf, m * n)
  let xNode = newNode(xBuf, s * n)
  let yNode = agMatmul(wNode, xNode, s, m, n)

  # Seed with ones
  var seed = newSeq[float32](s * m)
  for i in 0 ..< seed.len: seed[i] = 1.0f
  gpuUpload(yNode.grad, seed)
  yNode.backwardFn()

  let dwAnalytic = gpuDownload(wNode.grad)
  let dxAnalytic = gpuDownload(xNode.grad)

  # Numeric gradient for W
  var dwNumeric = newSeq[float32](m * n)
  for i in 0 ..< m * n:
    var wp = wHost
    var wm = wHost
    wp[i] += eps
    wm[i] -= eps
    let wpBuf = toGpu(wp)
    let wmBuf = toGpu(wm)
    let yp = gpuCreate(s * m)
    let ym = gpuCreate(s * m)
    gpuSgemm(2, s, m, n, xBuf, wpBuf, yp)
    gpuSgemm(2, s, m, n, xBuf, wmBuf, ym)
    let ypH = gpuDownload(yp)
    let ymH = gpuDownload(ym)
    var dw = 0.0f
    for j in 0 ..< s * m: dw += (ypH[j] - ymH[j]) * seed[j]
    dwNumeric[i] = dw / (2.0f * eps)

  checkOp("dW (matmul)", dwAnalytic, dwNumeric)

  # Numeric gradient for X
  var dxNumeric = newSeq[float32](s * n)
  for i in 0 ..< s * n:
    var xp = xHost
    var xm = xHost
    xp[i] += eps
    xm[i] -= eps
    let xpBuf = toGpu(xp)
    let xmBuf = toGpu(xm)
    let yp = gpuCreate(s * m)
    let ym = gpuCreate(s * m)
    gpuSgemm(2, s, m, n, xpBuf, wBuf, yp)
    gpuSgemm(2, s, m, n, xmBuf, wBuf, ym)
    let ypH = gpuDownload(yp)
    let ymH = gpuDownload(ym)
    var dx = 0.0f
    for j in 0 ..< s * m: dx += (ypH[j] - ymH[j]) * seed[j]
    dxNumeric[i] = dx / (2.0f * eps)

  checkOp("dX (matmul)", dxAnalytic, dxNumeric)

proc testAdd() =
  echo "testing agAdd..."
  let n = 16
  var aHost = newSeq[float32](n)
  var bHost = newSeq[float32](n)
  for i in 0 ..< n:
    aHost[i] = float32(i) * 0.1f
    bHost[i] = float32(n - i) * 0.1f

  let aNode = newNode(toGpu(aHost), n)
  let bNode = newNode(toGpu(bHost), n)
  let yNode = agAdd(aNode, bNode)

  var seed = newSeq[float32](n)
  for i in 0 ..< n: seed[i] = 1.0f
  gpuUpload(yNode.grad, seed)
  yNode.backwardFn()

  let daAnalytic = gpuDownload(aNode.grad)
  let dbAnalytic = gpuDownload(bNode.grad)

  # For add, numeric gradient is trivially 1.0 for both
  var expected = newSeq[float32](n)
  for i in 0 ..< n: expected[i] = 1.0f

  checkOp("dA (add)", daAnalytic, expected)
  checkOp("dB (add)", dbAnalytic, expected)

proc testGelu() =
  echo "testing agGelu..."
  let n = 16
  var xHost = newSeq[float32](n)
  for i in 0 ..< n: xHost[i] = float32(i - 8) * 0.3f

  let xNode = newNode(toGpu(xHost), n)
  let yNode = agGelu(xNode)

  var seed = newSeq[float32](n)
  for i in 0 ..< n: seed[i] = 1.0f
  gpuUpload(yNode.grad, seed)
  yNode.backwardFn()
  let dxAnalytic = gpuDownload(xNode.grad)

  var dxNumeric = newSeq[float32](n)
  for i in 0 ..< n:
    var xp = xHost; xp[i] += eps
    var xm = xHost; xm[i] -= eps
    let ypBuf = gpuCreate(n); geluFwd(toGpu(xp), ypBuf)
    let ymBuf = gpuCreate(n); geluFwd(toGpu(xm), ymBuf)
    let yp = gpuDownload(ypBuf)
    let ym = gpuDownload(ymBuf)
    dxNumeric[i] = (yp[i] - ym[i]) / (2.0f * eps)

  checkOp("dX (gelu)", dxAnalytic, dxNumeric)

proc testSoftmax() =
  echo "testing agSoftmax..."
  let n = 8
  var xHost = newSeq[float32](n)
  for i in 0 ..< n: xHost[i] = float32(i) * 0.5f - 2.0f

  let xNode = newNode(toGpu(xHost), n)
  let yNode = agSoftmax(xNode, 1, n)

  var seed = newSeq[float32](n)
  for i in 0 ..< n: seed[i] = if i == 3: 1.0f else: 0.0f
  gpuUpload(yNode.grad, seed)
  yNode.backwardFn()
  let dxAnalytic = gpuDownload(xNode.grad)

  var dxNumeric = newSeq[float32](n)
  for i in 0 ..< n:
    var xp = xHost; xp[i] += eps
    var xm = xHost; xm[i] -= eps
    let ypBuf = gpuCreate(n); softmaxFwd(toGpu(xp), ypBuf, 1, n)
    let ymBuf = gpuCreate(n); softmaxFwd(toGpu(xm), ymBuf, 1, n)
    let yp = gpuDownload(ypBuf)
    let ym = gpuDownload(ymBuf)
    dxNumeric[i] = (yp[3] - ym[3]) / (2.0f * eps)

  checkOp("dX (softmax)", dxAnalytic, dxNumeric)

proc testRmsNorm() =
  echo "testing agRmsNormAffine..."
  let dim = 8
  var xHost = newSeq[float32](dim)
  var gHost = newSeq[float32](dim)
  for i in 0 ..< dim:
    xHost[i] = float32(i) * 0.3f - 1.0f
    gHost[i] = 1.0f + float32(i) * 0.1f

  let xNode = newNode(toGpu(xHost), dim)
  let gNode = newNode(toGpu(gHost), dim)
  let yNode = agRmsNormAffine(xNode, gNode, 1, dim)

  var seed = newSeq[float32](dim)
  for i in 0 ..< dim: seed[i] = 1.0f
  gpuUpload(yNode.grad, seed)
  yNode.backwardFn()
  let dxAnalytic = gpuDownload(xNode.grad)

  var dxNumeric = newSeq[float32](dim)
  for i in 0 ..< dim:
    var xp = xHost; xp[i] += eps
    var xm = xHost; xm[i] -= eps
    let ypBuf = gpuCreate(dim); let rp = gpuCreate(1)
    let ymBuf = gpuCreate(dim); let rm = gpuCreate(1)
    rmsnormAffineFwd(toGpu(xp), toGpu(gHost), ypBuf, rp, 1, dim)
    rmsnormAffineFwd(toGpu(xm), toGpu(gHost), ymBuf, rm, 1, dim)
    let yp = gpuDownload(ypBuf)
    let ym = gpuDownload(ymBuf)
    var dx = 0.0f
    for j in 0 ..< dim: dx += (yp[j] - ym[j]) * seed[j]
    dxNumeric[i] = dx / (2.0f * eps)

  checkOp("dX (rmsnorm)", dxAnalytic, dxNumeric)

when isMainModule:
  gpuInit()
  trackingEnabled = true
  echo "=== Gradient checks ==="
  testAdd()
  freeStepAllocations()
  testGelu()
  freeStepAllocations()
  testSoftmax()
  freeStepAllocations()
  testRmsNorm()
  freeStepAllocations()
  testMatmul()
  freeStepAllocations()
  echo "=== Done ==="
