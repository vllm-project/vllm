# Problem 3: Element-wise Operations & Kernel Fusion

**Difficulty:** Easy
**Estimated Time:** 25-35 minutes
**Tags:** Kernel Fusion, Memory Bandwidth, Vector Operations

## Problem Statement

Implement fused element-wise operations in CUDA. Given three arrays A, B, C, compute:
`D[i] = alpha * A[i] + beta * B[i] * C[i] + gamma` for all i.

Optimize for memory bandwidth by fusing operations into a single kernel.

## Function Signature

```cuda
__global__ void fusedElementWise(float* A, float* B, float* C, float* D,
                                 float alpha, float beta, float gamma, int n);
```

## Input/Output

**Input:**
- Arrays A, B, C of length n
- Scalars alpha, beta, gamma
- n: array length (1 ≤ n ≤ 10^8)

**Output:**
- Array D: D[i] = alpha * A[i] + beta * B[i] * C[i] + gamma

## Example

```
Input:
A = [1, 2, 3]
B = [2, 3, 4]
C = [1, 1, 1]
alpha = 2.0, beta = 3.0, gamma = 1.0

Output:
D = [2*1 + 3*2*1 + 1, 2*2 + 3*3*1 + 1, 2*3 + 3*4*1 + 1]
  = [9, 14, 19]
```

## Requirements

1. Fuse all operations in single kernel
2. Achieve >80% memory bandwidth utilization
3. Handle arrays not divisible by block size
4. Use grid-stride loop for large arrays
5. Vectorized loads (float4) for bonus

## Success Criteria

- ✅ Correct computation for all test cases
- ✅ Single kernel (no separate kernels per operation)
- ✅ Coalesced memory access
- ✅ Handles arbitrary array sizes
- ✅ Near-peak bandwidth performance
