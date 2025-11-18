# Problem 11: Warp-Level Reduction Primitives

**Difficulty:** Medium
**Estimated Time:** 35-45 minutes
**Tags:** Warp Primitives, Shuffle Operations, Lock-Free

## Problem Statement

Implement array reduction using warp-level primitives (`__shfl_down_sync`, `__shfl_xor_sync`). Demonstrate understanding of warp-synchronous programming and avoid unnecessary `__syncthreads()`.

## Requirements

- Use warp shuffle instructions for intra-warp reduction
- No shared memory within warps (use shuffles instead)
- Support sum, max, min operations
- Handle arrays not divisible by warp size
- Combine warp results efficiently

## Function Signature

```cuda
__device__ float warpReduceSum(float val);
__device__ float warpReduceMax(float val);
__global__ void warpReduceKernel(float* input, float* output, int n, int op);
```

## Key Concepts

**Warp Shuffle Down:**
```cuda
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
// Thread 0 now has sum of all 32 threads
```

## Success Criteria

- ✅ Correct reduction using warp primitives
- ✅ No shared memory for intra-warp reduction
- ✅ Proper synchronization masks
- ✅ Handles multiple operations (sum, max, min)
- ✅ Efficient for small arrays
