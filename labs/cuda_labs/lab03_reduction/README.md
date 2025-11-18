# Lab 03: Parallel Reduction - Warp Primitives

## Problem Statement

Implement efficient parallel reduction algorithms using various CUDA optimization techniques. Reduction operations (sum, max, min) are critical for LLM inference in operations like computing attention scores, normalizing activations, and calculating statistics.

**Relevance to LLM Inference:**
- Softmax computation requires sum/max reductions
- LayerNorm requires mean and variance calculation
- Beam search uses max reductions
- Loss computation aggregates values across tokens

## Learning Objectives

1. Understand reduction pattern and work-efficient algorithms
2. Master warp-level primitives (`__shfl_down_sync`, `__reduce_add_sync`)
3. Avoid thread divergence and synchronization issues
4. Implement sequential addressing to avoid bank conflicts
5. Measure and optimize occupancy

## Prerequisites

- Completed Labs 01 and 02
- Understanding of parallel reduction concept
- Knowledge of shared memory and synchronization

## Estimated Time

2-3 hours

## Instructions

Implement three reduction kernels:
1. **Naive**: Divergent threads, bank conflicts
2. **Optimized**: Sequential addressing, no conflicts
3. **Warp**: Using warp-level primitives

## Expected Performance

For 16M elements (RTX 3080):
- Naive: ~2.0 ms
- Optimized: ~0.4 ms (5× speedup)
- Warp: ~0.2 ms (10× speedup)

## Profiling

```bash
nsys profile ./test
ncu --metrics warp_execution_efficiency,shared_efficiency ./solution
```

## Common Mistakes

1. **Divergent threads**: Avoid half-warp branches
2. **Bank conflicts**: Use sequential addressing
3. **Missing sync**: Must sync after shared memory writes
4. **Warp size assumption**: Always use `warpSize` constant

## Optimization Challenges

1. Implement min/max reductions
2. Multi-pass reduction for very large arrays
3. CUB library integration for comparison
4. Strided reduction for attention softmax

## Key Takeaways

- Warp primitives eliminate shared memory and synchronization
- Sequential addressing eliminates bank conflicts
- Work-efficient algorithms minimize total operations
- Reduction is often memory-bound despite computation

## Next Steps

Proceed to Lab 04: Memory Coalescing
