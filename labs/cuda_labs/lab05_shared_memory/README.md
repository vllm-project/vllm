# Lab 05: Advanced Shared Memory - Bank Conflicts and Optimization

## Problem Statement

Master shared memory usage patterns, identify and resolve bank conflicts, and implement efficient stencil operations. Shared memory is crucial for high-performance LLM kernels.

**Relevance to LLM Inference:**
- Multi-head attention tile caching
- Local reduction operations in LayerNorm/Softmax
- Convolution-based models (though less common)
- Tiling strategies in general GEMM

## Learning Objectives

1. Understand shared memory bank organization
2. Detect and resolve bank conflicts
3. Implement padding strategies
4. Optimize stencil computations
5. Use shared memory for inter-thread communication

## Prerequisites

- Completed Labs 01-04
- Understanding of memory banking
- Knowledge of synchronization primitives

## Estimated Time

2-3 hours

## Key Concepts

**Bank Conflicts**: Multiple threads in a warp accessing same bank (different addresses)
- 32 banks on modern GPUs
- Sequential 4-byte words map to sequential banks
- Broadcast (all threads read same address): No conflict
- Multicast (all threads read same word): No conflict
- Conflict: Different addresses in same bank → serialized

**Resolution Strategies**:
1. Padding (+1 on fastest dimension)
2. Access pattern restructuring
3. Using registers for temporary storage

## Expected Performance

For 1D stencil with radius=3 on 16M elements:
- Naive (global memory): ~8.0 ms
- Shared memory with conflicts: ~3.0 ms
- Shared memory optimized: ~1.2 ms

## Key Takeaways

- Bank conflicts can reduce performance by up to 32×
- Padding is simple and effective for most patterns
- Profile with Nsight Compute to detect conflicts
- Shared memory is 100× faster than global when used correctly

## Next Steps

Proceed to Lab 06: Atomic Operations
