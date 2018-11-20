#!/usr/bin/env python3
"""
Based on: https://cnugteren.github.io/tutorial/pages/page3.html
Important note: by default numpy array is row-major (not column-major)
"""
import numpy as np
import pyopencl as cl

a_np = np.random.rand(3,2).astype(np.float32)
b_np = np.random.rand(2,1).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

prg = cl.Program(ctx, """
__kernel void matmul(const int M, const int N, const int K,
    __global const float* A, __global const float* B, __global float* C)
{
  const int row = get_global_id(0); // 0..M-1
  const int col = get_global_id(1); // 0..N-1
  float acc = 0;

  for (int k = 0; k < K; ++k) {
    acc += A[row*K + k]*B[k*N + col];
  }

  C[row*N + col] = acc;
}
""").build()

m, k = a_np.shape
k, n = b_np.shape
res_np = np.empty((m,n)).astype(np.float32)

mf = cl.mem_flags
A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
C = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)

prg.matmul(queue, (m,n), None,
  np.int32(m), np.int32(n), np.int32(k), A, B, C)
cl.enqueue_copy(queue, res_np, C)

# Check on CPU with Numpy:
res_np_compare = np.matmul(a_np, b_np)

print(res_np - res_np_compare)
print(np.linalg.norm(res_np - res_np_compare))
