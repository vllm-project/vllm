# Tutorial 01: CUDA Memory Hierarchy

## Learning Objectives

After completing this tutorial, you will be able to:

1. Understand the CUDA memory hierarchy and access latency characteristics
2. Identify and optimize global memory access patterns for coalescing
3. Leverage shared memory to reduce global memory bandwidth requirements
4. Recognize and eliminate shared memory bank conflicts
5. Apply memory optimization techniques to vLLM kernels

## Prerequisites

- Basic CUDA programming knowledge (kernel launches, thread indexing)
- Understanding of C/C++ memory concepts (pointers, arrays)
- Familiarity with GPU architecture basics
- vLLM development environment setup

## Table of Contents

1. [CUDA Memory Types](#cuda-memory-types)
2. [Memory Hierarchy and Performance](#memory-hierarchy-and-performance)
3. [Global Memory Access Patterns](#global-memory-access-patterns)
4. [Shared Memory Fundamentals](#shared-memory-fundamentals)
5. [Bank Conflicts](#bank-conflicts)
6. [vLLM Examples](#vllm-examples)
7. [Hands-on Exercises](#hands-on-exercises)
8. [Best Practices](#best-practices)
9. [References](#references)

## CUDA Memory Types

CUDA provides several memory spaces with different characteristics:

```
┌─────────────────────────────────────────────────────────────┐
│                        GPU Memory Hierarchy                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │  Registers   │     │    Shared    │     │  Constant   │ │
│  │  (per thread)│     │    Memory    │     │   Memory    │ │
│  │              │     │  (per block) │     │  (cached)   │ │
│  │  ~1 cycle    │     │  ~5-20 cycles│     │  ~5 cycles  │ │
│  │  32-64 KB    │     │  0-164 KB    │     │  64 KB      │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│         │                     │                     │        │
│         └─────────────────────┴─────────────────────┘        │
│                               │                              │
│                    ┌──────────▼──────────┐                   │
│                    │    L1/L2 Cache      │                   │
│                    │   (128 KB - 40 MB)  │                   │
│                    │    ~100 cycles      │                   │
│                    └──────────┬──────────┘                   │
│                               │                              │
│                    ┌──────────▼──────────┐                   │
│                    │   Global Memory     │                   │
│                    │  (GB - hundreds)    │                   │
│                    │  300-600 cycles     │                   │
│                    └─────────────────────┘                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1. Register Memory

- **Fastest** memory available (~1 cycle latency)
- Private to each thread
- Limited: 32-64 KB per SM, typically 255 registers per thread
- Automatically managed by compiler
- Register spilling to local memory (global) when exceeded

### 2. Shared Memory

- **Fast** on-chip memory (~5-20 cycles)
- Shared among threads in a block
- Explicitly managed by programmer
- Size: 0-164 KB per block (architecture dependent)
- Perfect for data reuse within a block

### 3. Constant Memory

- Read-only memory cached on-chip
- Fast when all threads read the same address
- 64 KB total size
- Good for uniform data access patterns

### 4. Global Memory

- **Slowest** but largest (GBs)
- Accessible by all threads
- High latency (300-600 cycles)
- High bandwidth when accessed correctly
- Requires coalescing for optimal performance

### 5. Local Memory

- Actually resides in global memory (slow)
- Used for register spills and large arrays
- Private to each thread
- Avoid when possible

## Memory Hierarchy and Performance

### Latency vs Bandwidth Trade-off

| Memory Type | Latency | Bandwidth | Size | Scope |
|------------|---------|-----------|------|-------|
| Registers | 1 cycle | ~20 TB/s | 32-64 KB/SM | Thread |
| Shared | 5-20 cycles | ~15 TB/s | 0-164 KB/block | Block |
| L1 Cache | ~30 cycles | ~10 TB/s | 128 KB/SM | SM |
| L2 Cache | ~100 cycles | ~5 TB/s | 40 MB | Device |
| Global | 300-600 cycles | 1-2 TB/s | GBs | Device |

### Key Insights

1. **Minimize global memory access**: Use shared memory for frequently accessed data
2. **Maximize memory bandwidth**: Ensure coalesced access patterns
3. **Optimize register usage**: Keep working set in registers when possible
4. **Hide latency**: Use enough threads to hide memory latency

## Global Memory Access Patterns

### Coalesced Memory Access

For optimal performance, adjacent threads should access adjacent memory locations:

```
Coalesced Access (Good):
Thread:  0    1    2    3    4    5    6    7
         │    │    │    │    │    │    │    │
Memory: [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  → Single transaction

Strided Access (Poor):
Thread:  0    1    2    3    4    5    6    7
         │    │    │    │    │    │    │    │
Memory: [0]  [2]  [4]  [6]  [8]  [10] [12] [14] → Multiple transactions

Random Access (Worst):
Thread:  0    1    2    3    4    5    6    7
         │    │    │    │    │    │    │    │
Memory: [17] [3]  [9]  [21] [5]  [13] [1]  [19] → Many transactions
```

### Example: Coalesced vs Strided

```cpp
// BAD: Strided access (stride = 2)
__global__ void strided_kernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float value = data[idx * 2];  // Non-coalesced!
    }
}

// GOOD: Coalesced access
__global__ void coalesced_kernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float value = data[idx];  // Coalesced!
    }
}
```

## Shared Memory Fundamentals

Shared memory enables cooperation between threads in a block:

```cpp
__global__ void shared_memory_example(float* input, float* output, int n) {
    // Allocate shared memory
    __shared__ float shared_data[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from global to shared (coalesced)
    if (gid < n) {
        shared_data[tid] = input[gid];
    }
    __syncthreads();  // Ensure all threads have loaded data

    // Process data in shared memory
    if (tid < blockDim.x - 1) {
        float result = shared_data[tid] + shared_data[tid + 1];
        output[gid] = result;
    }
    __syncthreads();  // Synchronize before next iteration
}
```

### Shared Memory Benefits

1. **Data reuse**: Load once from global, read many times
2. **Communication**: Threads in a block can share data
3. **Reduction operations**: Efficient parallel summation
4. **Tiling**: Break large problems into shared-memory-sized tiles

## Bank Conflicts

Shared memory is divided into banks (typically 32 banks on modern GPUs):

```
Bank Organization (32 banks, 4-byte words):

Address: 0    1    2    3    ... 31   32   33   34   ...
Bank:    0    1    2    3    ... 31   0    1    2    ...

┌────┬────┬────┬────┬─────┬────┬────┬────┬────┐
│ B0 │ B1 │ B2 │ B3 │ ... │B31 │ B0 │ B1 │ B2 │
└────┴────┴────┴────┴─────┴────┴────┴────┴────┘
```

### Conflict-Free Access

```cpp
// NO CONFLICT: Each thread accesses different bank
__shared__ float data[32];
int tid = threadIdx.x;
float value = data[tid];  // Threads 0-31 access banks 0-31

// 2-WAY CONFLICT: Two threads access same bank
float value = data[tid / 2];  // Half the threads conflict

// BROADCAST: All threads read same address (no conflict)
float value = data[0];  // Broadcast from bank 0
```

### Common Patterns

```cpp
// BAD: 2-way bank conflict (stride = 2)
__shared__ float shared[64];
int tid = threadIdx.x;
float value = shared[tid * 2];  // Half the threads conflict

// GOOD: Conflict-free access
float value = shared[tid];

// PADDING to avoid conflicts in 2D arrays
#define PADDING 1
__shared__ float tile[32][32 + PADDING];  // Avoid column access conflicts
```

## vLLM Examples

### Example 1: Cache Kernel - Block Copying

**File**: `/home/user/vllm-learn/csrc/cache_kernels.cu` (Lines 73-98)

```cpp
template <typename scalar_t>
__global__ void copy_blocks_kernel(int64_t* key_cache_ptrs,
                                   int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache =
      reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;

  // Coalesced memory access pattern
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}
```

**Memory Analysis**:
- **Coalesced access**: `i = threadIdx.x + k * blockDim.x` ensures adjacent threads access adjacent elements
- **Global memory**: Direct global-to-global copy (no shared memory needed for simple copy)
- **Striding**: Loop stride matches warp size for optimal memory transactions

### Example 2: Activation Kernel - Register Usage

**File**: `/home/user/vllm-learn/csrc/activation_kernels.cu` (Lines 22-32)

```cpp
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}
```

**Memory Analysis**:
- **Register optimization**: `x` and `y` loaded into registers via `VLLM_LDG` (read-only cache hint)
- **Coalesced reads**: Adjacent threads read adjacent memory locations
- **Minimal memory footprint**: No shared memory, only register usage
- **Read-only hint**: `__restrict__` and `VLLM_LDG` allow compiler optimizations

### Example 3: Paged Attention - Shared Memory Configuration

**File**: `/home/user/vllm-learn/csrc/attention/paged_attention_v2.cu` (Lines 81-91)

```cpp
const int NUM_WARPS = NUM_THREADS / WARP_SIZE;
int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
int logits_size = PARTITION_SIZE * sizeof(float);
int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

// For paged attention v2 kernel.
dim3 grid(num_heads, num_seqs, max_num_partitions);
int shared_mem_size = std::max(logits_size, outputs_size);
// For paged attention v2 reduce kernel.
dim3 reduce_grid(num_heads, num_seqs);
int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);
```

**Memory Analysis**:
- **Dynamic shared memory allocation**: Size calculated based on problem dimensions
- **Memory reuse**: Same shared memory used for different stages (logits vs outputs)
- **Partition strategy**: Reduces global memory bandwidth by computing in tiles
- **Warp-level organization**: Shared memory sized for efficient warp operations

## Performance Analysis

### Memory Bandwidth Utilization

To measure memory bandwidth efficiency:

```bash
# Using NVIDIA Nsight Compute
ncu --metrics dram_throughput,l1tex_throughput,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
    ./your_kernel

# Key metrics:
# - dram_throughput: Actual DRAM bandwidth used
# - Memory throughput %: Percentage of peak bandwidth
# - Coalescing efficiency: Bytes requested / Bytes transferred
```

### Example Output:
```
Memory Throughput:           850 GB/s (85% of peak)
L1 Cache Hit Rate:          65%
Global Load Efficiency:     82% (good coalescing)
Shared Memory Bank Conflicts: 0.5% (negligible)
```

## Hands-on Exercises

### Exercise 1: Fix Strided Access Pattern

Given this kernel with poor memory access:

```cpp
__global__ void transpose_naive(float* out, const float* in,
                                int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];  // Non-coalesced write!
    }
}
```

**Task**: Modify to use shared memory for coalesced writes.

**Hint**: Use a shared memory tile and transpose in shared memory.

### Exercise 2: Eliminate Bank Conflicts

Analyze and fix bank conflicts in this reduction kernel:

```cpp
__global__ void reduce_with_conflicts(float* g_data, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_data[i] : 0;
    __syncthreads();

    // Reduction with bank conflicts
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];  // Divergence and conflicts!
        }
        __syncthreads();
    }

    if (tid == 0) g_data[blockIdx.x] = sdata[0];
}
```

**Task**: Optimize for sequential addressing and no bank conflicts.

### Exercise 3: Optimize vLLM Cache Kernel

Study the `copy_blocks_kernel` from vLLM:

**Tasks**:
1. Calculate theoretical bandwidth for copying 1GB of FP16 data
2. Measure actual performance using CUDA events
3. Experiment with different block sizes (128, 256, 512 threads)
4. Compare with `cudaMemcpy` performance

## Best Practices

### Memory Access Optimization

1. **Always ensure coalesced access**
   - Adjacent threads access adjacent addresses
   - Align starting addresses to segment boundaries (32/64/128 bytes)
   - Use structure-of-arrays (SoA) instead of array-of-structures (AoS)

2. **Use shared memory for data reuse**
   - Load once, use multiple times
   - Stage data through shared memory for non-coalesced patterns
   - Size tiles based on shared memory limits (48-164 KB)

3. **Minimize bank conflicts**
   - Use padding for multi-dimensional shared arrays
   - Access in stride-1 patterns when possible
   - Sequential addressing in reductions

4. **Optimize register usage**
   - Keep frequently accessed data in registers
   - Monitor register spilling (use `--ptxas-options=-v`)
   - Balance register usage vs occupancy

5. **Use memory qualifiers**
   - `__restrict__`: Indicates no pointer aliasing
   - `const`: Read-only data (enables texture/L1 caching)
   - `__ldg()`: Force read-only cache load

### vLLM-Specific Patterns

1. **Paged memory layout**: Design kernels for block-based addressing
2. **Token-level parallelism**: One block per token in many kernels
3. **Head-level parallelism**: Separate computation for each attention head
4. **Fused operations**: Combine multiple operations to reduce memory traffic

## References

### Official Documentation

1. [CUDA C Programming Guide - Memory Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
2. [CUDA Best Practices Guide - Memory Optimization](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
3. [NVIDIA Nsight Compute - Memory Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

### Academic Papers

1. Hong, S., & Kim, H. (2009). "An analytical model for a GPU architecture with memory-level and thread-level parallelism awareness"
2. Volkov, V. (2010). "Better performance at lower occupancy"
3. Wong, H. et al. (2010). "Demystifying GPU microarchitecture through microbenchmarking"

### Books

1. "Programming Massively Parallel Processors" by Kirk & Hwu (Chapter 5: Memory Architecture)
2. "CUDA by Example" by Sanders & Kandrot (Chapter 6: Memory)
3. "Professional CUDA C Programming" by Cheng et al. (Chapter 4: Memory)

### vLLM Resources

1. [vLLM GitHub - CUDA Kernels](https://github.com/vllm-project/vllm/tree/main/csrc)
2. [vLLM Documentation - Performance Optimization](https://docs.vllm.ai/)
3. vLLM kernel implementations in `/csrc/` directory

## Summary

Understanding CUDA memory hierarchy is fundamental to writing efficient GPU kernels:

- **Registers** are fastest but limited - keep hot data here
- **Shared memory** enables cooperation and reduces global memory traffic
- **Global memory** is slow but large - optimize access patterns for coalescing
- **Bank conflicts** can severely degrade shared memory performance
- **vLLM kernels** demonstrate production-quality memory optimization techniques

Master these concepts before proceeding to advanced optimization techniques in subsequent tutorials.

---

**Next Tutorial**: [02_kernel_optimization_basics.md](02_kernel_optimization_basics.md) - Learn about occupancy, thread block configuration, and warp-level operations.
