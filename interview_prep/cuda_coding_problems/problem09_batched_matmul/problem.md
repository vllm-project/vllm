# Problem 9: Batched Matrix Multiplication

**Difficulty:** Medium
**Estimated Time:** 40-50 minutes
**Tags:** Batching, 3D Tensors, GEMM

## Problem Statement

Implement batched matrix multiplication in CUDA. Given batches of matrices A (batch_size × M × K) and B (batch_size × K × N), compute C = A @ B for each batch independently.

## Requirements

- Process multiple matrix multiplications in parallel
- Use shared memory tiling
- Handle arbitrary batch sizes and dimensions
- Optimize for GPU occupancy

## Function Signature

```cuda
__global__ void batchedMatMulKernel(float* A, float* B, float* C,
                                    int batch_size, int M, int K, int N);
```

## Input/Output

**Input:**
- A: (batch_size × M × K)
- B: (batch_size × K × N)

**Output:**
- C: (batch_size × M × N)

## Example

```
batch_size=2, M=2, K=2, N=2

A = [[[1,2], [3,4]], [[5,6], [7,8]]]
B = [[[1,0], [0,1]], [[1,1], [1,1]]]
C = [[[1,2], [3,4]], [[11,11], [15,15]]]
```

## Success Criteria

- ✅ Correct batched multiplication
- ✅ Tiled implementation
- ✅ Handles multiple batches efficiently
- ✅ Good GPU utilization
