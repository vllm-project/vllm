# Lab 04: Memory Coalescing - Optimizing Access Patterns

## Problem Statement

Learn how memory access patterns affect performance by implementing kernels with different access patterns. Understanding coalescing is critical for achieving high bandwidth utilization in LLM operations.

**Relevance to LLM Inference:**
- Transpose operations in attention mechanisms
- Embedding lookups
- Token gathering and scattering
- Weight matrix layout optimization

## Learning Objectives

1. Understand coalesced vs non-coalesced memory access
2. Implement efficient matrix transpose
3. Use shared memory to enable coalescing
4. Measure memory bandwidth utilization
5. Identify and fix strided access patterns

## Prerequisites

- Completed Labs 01-03
- Understanding of memory hierarchies
- Knowledge of shared memory

## Estimated Time

2-3 hours

## Key Concepts

**Coalesced Access**: Consecutive threads access consecutive memory locations
- Achieved: Threads 0-31 access addresses 0-31 (single transaction)
- Not achieved: Threads 0-31 access stride-N pattern (N transactions)

**Matrix Transpose Challenge**:
- Row-wise read + column-wise write = non-coalesced
- Solution: Use shared memory as staging area

## Expected Performance

For 4096×4096 matrix transpose (RTX 3080):
- Naive (non-coalesced): ~5.0 ms
- Coalesced (shared mem): ~1.2 ms (4× speedup)
- Optimized (padded): ~0.9 ms (5.5× speedup)

## Profiling

```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
             dram__bytes_read.sum,\
             dram__throughput.avg.pct_of_peak_sustained_elapsed \
     ./solution
```

## Key Takeaways

- Non-coalesced accesses can reduce effective bandwidth by 10-32×
- Shared memory enables transforming access patterns
- Bank conflict padding (+1) often necessary
- Real LLM kernels require careful layout planning

## Next Steps

Proceed to Lab 05: Advanced Shared Memory Techniques
