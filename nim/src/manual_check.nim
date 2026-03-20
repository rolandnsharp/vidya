## Pure CPU matmul backward check — no CUDA, no cuBLAS.
## If this also shows 11% error, the bug is in the math, not the GPU.
## If this is correct, the bug is in the GPU matmul wrapper.

import std/[strformat, math]

const eps = 0.0001f

proc matmul(a: seq[float32], b: seq[float32], m, n, k: int): seq[float32] =
  ## C[m,n] = A[m,k] @ B[k,n]
  result = newSeq[float32](m * n)
  for i in 0 ..< m:
    for j in 0 ..< n:
      var s = 0.0f
      for p in 0 ..< k:
        s += a[i * k + p] * b[p * n + j]
      result[i * n + j] = s

proc matmulNT(a: seq[float32], b: seq[float32], m, n, k: int): seq[float32] =
  ## C[m,n] = A[m,k] @ B[n,k]^T  (B stored as [n,k])
  result = newSeq[float32](m * n)
  for i in 0 ..< m:
    for j in 0 ..< n:
      var s = 0.0f
      for p in 0 ..< k:
        s += a[i * k + p] * b[j * k + p]
      result[i * n + j] = s

proc softmax(x: seq[float32]): seq[float32] =
  let n = x.len
  var mx = -1e30f
  for v in x: mx = max(mx, v)
  result = newSeq[float32](n)
  var s = 0.0f
  for i in 0 ..< n:
    result[i] = exp(x[i] - mx)
    s += result[i]
  for i in 0 ..< n: result[i] /= s

when isMainModule:
  let dim = 4
  let vocab = 10
  let seqLen = 2

  var we = newSeq[float32](vocab * dim)
  for i in 0 ..< we.len: we[i] = 0.05f * float32((i * 7 + 3) mod 19 - 9)
  var wp = newSeq[float32](vocab * dim)
  for i in 0 ..< wp.len: wp[i] = 0.03f * float32((i * 11 + 5) mod 23 - 11)
  var wq = newSeq[float32](dim * dim)
  for i in 0 ..< wq.len: wq[i] = 0.02f * float32((i * 3 + 1) mod 11 - 5)

  var tokens = @[3, 7, 1]

  proc computeLoss(we, wp, wq: seq[float32]): float32 =
    # Embed
    var emb = newSeq[float32](seqLen * dim)
    for i in 0 ..< seqLen:
      for j in 0 ..< dim:
        emb[i * dim + j] = we[tokens[i] * dim + j]
    # hidden = emb @ Wq^T
    let hidden = matmulNT(emb, wq, seqLen, dim, dim)
    # logits = hidden @ Wp^T
    let logits = matmulNT(hidden, wp, seqLen, vocab, dim)
    # CE on last position
    let lastLogits = logits[((seqLen-1) * vocab) ..< (seqLen * vocab)]
    let probs = softmax(lastLogits)
    -ln(max(probs[tokens[seqLen]], 1e-10f))

  let loss0 = computeLoss(we, wp, wq)
  echo &"loss = {loss0:.6f}"

  echo "Wq gradient (pure CPU, no CUDA):"
  for i in 0 ..< 5:
    var wqp = wq; wqp[i] += eps
    var wqm = wq; wqm[i] -= eps
    let lp = computeLoss(we, wp, wqp)
    let lm = computeLoss(we, wp, wqm)
    let num = (lp - lm) / (2.0f * eps)
    echo &"  [{i}] numeric={num:.6f}"

  echo ""
  echo "Now compare with GPU numeric from notied_check."
