## Verify each GEMM op code produces correct results.
import gpu
import std/strformat

when isMainModule:
  gpuInit()

  # A[2,3] = [[1,2,3],[4,5,6]]
  # B[3,2] = [[1,2],[3,4],[5,6]]
  # A @ B = [[22,28],[49,64]]      (2x2)
  # A^T @ ... needs A stored as [3,2] (transposed storage)
  # B^T @ ... needs B stored as [2,3]

  let a = toGpu(@[1f,2,3, 4,5,6])       # [2,3]
  let b = toGpu(@[1f,2, 3,4, 5,6])      # [3,2]
  let bT = toGpu(@[1f,3,5, 2,4,6])      # [2,3] — B transposed storage

  # op=0: C = A @ B (NN overwrite)
  let c0 = gpuCreate(4)
  gpuSgemm(0, 2, 2, 3, a, b, c0)
  echo "op=0 A@B: ", gpuDownload(c0), " expect [22,28,49,64]"

  # op=2: C = A @ B^T (NT overwrite) — B stored as [2,3]
  let c2 = gpuCreate(4)
  gpuSgemm(2, 2, 2, 3, a, bT, c2)
  echo "op=2 A@B^T: ", gpuDownload(c2), " expect [22,28,49,64]"

  # op=4: C = A^T @ B (TN overwrite) — A stored as [3,2]
  # A^T[3,2] = [[1,4],[2,5],[3,6]] stored as [2,3]: [1,4,2,5,3,6]
  let aT = toGpu(@[1f,4, 2,5, 3,6])    # [3,2] — A^T stored row-major
  let c4 = gpuCreate(6)  # result [3,2]
  gpuSgemm(4, 3, 2, 2, aT, b, c4)
  # A^T[3,2] @ B[2,2]... wait, dims don't match
  # Let me redo: A^T is [3,2], we need A^T @ something[2,k]
  # A^T[3,2] @ B_small[2,2] → [3,2]
  let bSmall = toGpu(@[1f,2, 3,4])  # [2,2]
  gpuSgemm(4, 3, 2, 2, a, bSmall, c4)
  # op=4: C[3,2] = A^T[3,2] @ B[2,2] where A stored as [2,3]
  # A^T = [[1,4],[2,5],[3,6]], B = [[1,2],[3,4]]
  # result = [[13,18],[17,24],[21,30]]
  echo "op=4 A^T@B: ", gpuDownload(c4)[0..5], " expect [13,18,17,24,21,30]"

  # op=1: C += A @ B (NN accumulate)
  let c1 = gpuCreate(4)
  gpuUpload(c1, @[10f, 10, 10, 10])
  gpuSgemm(1, 2, 2, 3, a, b, c1)
  echo "op=1 10+A@B: ", gpuDownload(c1), " expect [32,38,59,74]"

  # op=5: C += A^T @ B (TN accumulate)
  let c5 = gpuCreate(6)
  gpuUpload(c5, @[100f, 100, 100, 100, 100, 100])
  gpuSgemm(5, 3, 2, 2, a, bSmall, c5)
  echo "op=5 100+A^T@B: ", gpuDownload(c5)[0..5], " expect [113,118,117,124,121,130]"
