# Lab 08: Occupancy Optimization - Maximizing GPU Utilization

## Problem Statement

Learn to analyze and optimize kernel occupancy to maximize GPU utilization. Understanding occupancy is crucial for achieving peak performance in compute-bound kernels common in LLM inference.

**Relevance to LLM Inference:**
- Attention kernel optimization
- Large GEMM operations in transformers
- Custom activation functions
- Fused kernels requiring careful resource management

## Learning Objectives

1. Understand occupancy and its impact on performance
2. Use CUDA Occupancy Calculator and APIs
3. Optimize register usage and shared memory allocation
4. Balance resource usage vs parallelism
5. Measure achieved occupancy with profiling tools

## Prerequisites

- Completed Labs 01-07
- Understanding of SM architecture
- Knowledge of resource limitations

## Estimated Time

2-3 hours

## Key Concepts

**Occupancy**: Ratio of active warps to maximum possible warps per SM

**Factors Limiting Occupancy**:
1. **Registers per thread**: More registers → fewer threads per SM
2. **Shared memory per block**: Larger allocations → fewer blocks per SM
3. **Block size**: Too small → underutilization, too large → resource exhaustion
4. **Architectural limits**: Max threads/blocks per SM

**Optimization Strategies**:
- Launch bounds: `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)`
- Register pressure reduction: loop unrolling, spilling control
- Shared memory tuning: dynamic allocation, padding strategies
- Block size experimentation: find sweet spot for your kernel

## Expected Results

For compute-bound kernel (complex math operations):
- Low occupancy (25%): 100 ms
- Medium occupancy (50%): 60 ms
- High occupancy (75%+): 45 ms

**Note**: High occupancy doesn't always mean better performance!
- Memory-bound kernels: Occupancy less critical
- Compute-bound kernels: Occupancy very important

## Profiling Occupancy

### Using Nsight Compute

```bash
# Check theoretical and achieved occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
           sm__maximum_warps_per_active_cycle_pct \
    ./solution

# Full occupancy analysis
ncu --set full --section SpeedOfLight --section Occupancy ./solution

# Launch statistics
ncu --metrics launch__* ./solution
```

### Using CUDA Occupancy API

```cuda
int maxActiveBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
    myKernel, blockSize, dynamicSMemSize);

// Calculate occupancy
float occupancy = (maxActiveBlocks * blockSize) /
                  (float)deviceProp.maxThreadsPerMultiprocessor;
```

## Common Patterns

### Pattern 1: Register-Limited Kernel
```cuda
// Too many registers → low occupancy
__global__ void heavyRegisterUsage() {
    float temp[64];  // Lots of arrays → many registers
    // Solution: Reduce array sizes, use shared memory
}
```

### Pattern 2: Shared Memory-Limited
```cuda
// Large shared memory → few blocks per SM
__global__ void largeSharedMem() {
    __shared__ float data[8192];  // 32KB
    // Solution: Reduce size, use dynamic allocation wisely
}
```

### Pattern 3: Block Size Tuning
```cuda
// Experiment with different block sizes
// Not always bigger is better!
// 128, 256, 512 often optimal
```

## Optimization Challenges

1. **Auto-tuning**: Write script to find optimal block size automatically
2. **Multi-kernel**: Balance occupancy across multiple kernels in pipeline
3. **Mixed precision**: Optimize FP16/FP32 mixed kernels
4. **Dynamic shared memory**: Optimize allocation based on problem size

## Key Takeaways

- Occupancy is means to an end, not goal itself
- High occupancy helps hide latency in compute-bound kernels
- Balance occupancy with ILP and memory access patterns
- Use profiling tools to measure achieved vs theoretical occupancy
- `__launch_bounds__` helps compiler optimize for target occupancy
- Block size significantly impacts occupancy

## Trade-offs

**High Occupancy**:
- ✓ Better latency hiding
- ✓ More parallel work
- ✗ Less resources per thread
- ✗ Potentially more register spilling

**Low Occupancy**:
- ✓ More resources per thread
- ✓ Less contention
- ✗ Poor latency hiding
- ✗ Underutilized hardware

## Real-World Application

In LLM inference:
- **Small batch**: May benefit from lower occupancy (more resources per thread)
- **Large batch**: Need high occupancy to process many sequences
- **Attention kernels**: Often compute-bound, need high occupancy
- **Memory ops**: Occupancy less critical, focus on bandwidth

## Next Steps

Congratulations on completing the CUDA fundamentals labs!

**Next:**
1. Apply these concepts to real LLM kernels
2. Study advanced topics: Tensor Cores, Multi-GPU, Streams
3. Explore optimization of FlashAttention, PagedAttention
4. Dive into vLLM codebase and contribute!

## References

- [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/)
- [Nsight Compute Occupancy Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#occupancy)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
