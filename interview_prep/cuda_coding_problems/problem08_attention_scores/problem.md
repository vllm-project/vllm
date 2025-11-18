# Problem 8: Attention Score Computation (Q*K^T)

**Difficulty:** Medium
**Estimated Time:** 45-50 minutes
**Tags:** Transformer, Matrix Multiply, Tiling

## Problem Statement

Implement an optimized kernel for computing attention scores: `scores = Q @ K^T / sqrt(d_k)`. This is a key operation in transformer models.

## Requirements

- Compute matrix multiplication Q * K^T where Q is (seq_len_q × d_k) and K is (seq_len_k × d_k)
- Scale by 1/sqrt(d_k) for attention stability
- Use shared memory tiling for optimization
- Handle arbitrary dimensions
- Output: (seq_len_q × seq_len_k) matrix

## Function Signature

```cuda
__global__ void attentionScoresKernel(float* Q, float* K, float* scores,
                                      int seq_len_q, int seq_len_k, int d_k);
```

## Example

```
Q = [[1, 0],   # 3x2
     [0, 1],
     [1, 1]]

K = [[1, 0],   # 2x2
     [0, 1]]

d_k = 2
scale = 1/sqrt(2) ≈ 0.707

scores = Q @ K^T / sqrt(d_k)
       = [[0.707, 0],
          [0, 0.707],
          [0.707, 0.707]]
```

## Success Criteria

- ✅ Correct Q*K^T computation
- ✅ Proper scaling by sqrt(d_k)
- ✅ Tiled implementation using shared memory
- ✅ Handles arbitrary dimensions
- ✅ Coalesced memory access
