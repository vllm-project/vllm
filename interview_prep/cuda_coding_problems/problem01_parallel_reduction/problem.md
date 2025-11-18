# Problem 1: Parallel Reduction (Sum Array)

**Difficulty:** Easy
**Estimated Time:** 30-45 minutes
**Tags:** CUDA Basics, Parallel Reduction, Shared Memory

## Problem Statement

Implement an efficient parallel reduction algorithm in CUDA to compute the sum of all elements in an array. Your solution should utilize shared memory and minimize warp divergence.

## Requirements

Write a CUDA kernel `parallelSum` that:
1. Takes an input array of floats
2. Returns the sum of all elements
3. Uses shared memory for efficient reduction
4. Handles arrays of arbitrary size (not necessarily power of 2)
5. Minimizes bank conflicts and warp divergence

## Function Signature

```cuda
__global__ void reductionKernel(float* input, float* output, int n);
float parallelSum(float* h_input, int n);
```

## Input/Output Specifications

**Input:**
- `h_input`: Host array of floats (length n)
- `n`: Number of elements (1 ≤ n ≤ 10^8)

**Output:**
- Sum of all elements as a single float

## Constraints

- Time Complexity: O(log n) per element
- Space Complexity: O(n) for input, O(n/block_size) for intermediate results
- Must use shared memory
- Block size should be configurable (typically 256 or 512)

## Examples

### Example 1
```
Input: [1.0, 2.0, 3.0, 4.0, 5.0]
Output: 15.0
```

### Example 2
```
Input: [1.5, -2.3, 4.7, 0.0, -1.5, 3.6]
Output: 6.0
```

### Example 3 (Edge Case)
```
Input: [42.0]
Output: 42.0
```

### Example 4 (Large Array)
```
Input: Array of 1,000,000 ones
Output: 1000000.0
```

## Key Considerations

1. **Shared Memory:** Use shared memory to reduce global memory accesses
2. **Warp Divergence:** Minimize divergent branches in reduction loop
3. **Bank Conflicts:** Avoid bank conflicts in shared memory access
4. **Grid-Stride:** Handle arrays larger than grid size
5. **Numerical Stability:** Consider floating-point precision

## Common Pitfalls

- Forgetting to synchronize threads after shared memory writes
- Not handling non-power-of-2 array sizes
- Bank conflicts in shared memory access patterns
- Race conditions in reduction tree
- Not handling the final reduction of partial sums

## Follow-Up Questions

1. How would you extend this to compute other reductions (min, max, product)?
2. What's the impact of different block sizes on performance?
3. How would you handle different data types (double, int)?
4. Can you use warp-level primitives to optimize further?
5. How does this compare to CUB library's DeviceReduce?

## Optimization Opportunities

- Use `__shfl_down_sync()` for warp-level reduction
- Sequential addressing to avoid bank conflicts
- Multiple elements per thread for better memory coalescing
- Template kernels for different data types

## Success Criteria

- ✅ Produces correct sum for all test cases
- ✅ Uses shared memory effectively
- ✅ Handles arbitrary array sizes
- ✅ No race conditions or synchronization bugs
- ✅ Reasonable performance (within 2x of cuBLAS)
