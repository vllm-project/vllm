# Problem 12: Memory Coalescing Optimization

**Difficulty:** Medium
**Estimated Time:** 40-50 minutes
**Tags:** Memory Access Patterns, Performance Optimization, Profiling

## Problem Statement

Given an uncoalesced memory access pattern, identify the issue and rewrite the kernel to achieve coalesced access. Demonstrate understanding of memory coalescing and its impact on bandwidth.

## Task

Fix the following uncoalesced kernel:

```cuda
// BAD: Strided access pattern
__global__ void unstridedCopy(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx * stride];  // Uncoalesced!
    }
}
```

## Requirements

- Identify coalescing issues
- Rewrite for coalesced access
- Handle arbitrary strides
- Use shared memory if needed
- Demonstrate performance improvement

## Success Criteria

- ✅ Identifies coalescing problem
- ✅ Coalesced solution implemented
- ✅ Handles multiple stride patterns
- ✅ Measurable performance improvement
- ✅ Explains trade-offs
