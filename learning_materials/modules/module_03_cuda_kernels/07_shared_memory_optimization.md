# Tutorial 07: Shared Memory Optimization

## Learning Objectives

After completing this tutorial, you will be able to:

1. Understand shared memory organization and bank conflicts
2. Design access patterns to avoid bank conflicts
3. Use shared memory for efficient data reuse in attention kernels
4. Apply padding techniques to eliminate conflicts
5. Optimize shared memory usage in production LLM kernels

## Prerequisites

- Completion of Tutorials 01-06
- Understanding of shared memory hierarchy
- Knowledge of warp execution model
- Familiarity with attention mechanism implementation

## Table of Contents

1. [Shared Memory Architecture](#shared-memory-architecture)
2. [Bank Conflicts](#bank-conflicts)
3. [Access Pattern Analysis](#access-pattern-analysis)
4. [Conflict Avoidance Techniques](#conflict-avoidance-techniques)
5. [Shared Memory in Attention](#shared-memory-in-attention)
6. [Dynamic Shared Memory](#dynamic-shared-memory)
7. [vLLM Examples](#vllm-examples)
8. [Hands-on Exercises](#hands-on-exercises)
9. [Best Practices](#best-practices)
10. [References](#references)

## Shared Memory Architecture

### Memory Banks

Shared memory is divided into 32 banks (on modern GPUs):

```
Shared Memory Organization (32 banks):

Address (4-byte words):
0    1    2    3    ...  31   32   33   34   ...  63
┌────┬────┬────┬────┬───┬────┬────┬────┬────┬───┬────┐
│ B0 │ B1 │ B2 │ B3 │...│B31 │ B0 │ B1 │ B2 │...│B31 │
└────┴────┴────┴────┴───┴────┴────┴────┴────┴───┴────┘

Bank(address) = (address / 4) % 32

Example:
Word 0 → Bank 0
Word 1 → Bank 1
...
Word 31 → Bank 31
Word 32 → Bank 0  (wraps around)
Word 33 → Bank 1
```

### Bandwidth Characteristics

```
Access Scenarios:

1. Conflict-Free (BEST):
   - All 32 threads access different banks
   - Bandwidth: 32 elements per cycle
   - Latency: ~20-30 cycles

2. Broadcast (GOOD):
   - All threads read same address
   - Bandwidth: 1 read, broadcast to 32
   - Latency: ~20-30 cycles

3. 2-Way Conflict (WORSE):
   - 2 threads per bank
   - Bandwidth: 16 elements per cycle
   - Latency: ~40-60 cycles (serialized)

4. 32-Way Conflict (WORST):
   - All threads access same bank
   - Bandwidth: 1 element per cycle
   - Latency: ~640-960 cycles (fully serialized)
```

### Conflict-Free Access Pattern

```cpp
__shared__ float data[1024];

// Thread ID in warp
int lane = threadIdx.x % 32;

// CONFLICT-FREE: Each thread accesses different bank
float val1 = data[lane];       // Bank 0-31
float val2 = data[lane + 32];  // Bank 0-31 (different address)

// BROADCAST: All threads read same address (no conflict)
float shared_val = data[0];  // All threads read from bank 0

// 2-WAY CONFLICT: Two threads per bank
float val3 = data[lane * 2];  // Even banks used twice

// 32-WAY CONFLICT: All threads access same bank
float val4 = data[lane % 1];  // All access bank 0 (worst case!)
```

## Bank Conflicts

### Detecting Conflicts

Bank conflicts occur when multiple threads in a warp access different addresses in the same bank.

```
Example: Stride-2 Access

Threads:  0    1    2    3   ...  31
Addresses: 0    2    4    6   ...  62
Banks:    0    2    4    6   ...  30  (even banks only)

Result: 16 banks used, 2 threads per bank → 2-way conflict
```

### Mathematical Formula

```cpp
// For stride S and thread T:
address = T * S
bank = (address / 4) % 32

// Conflicts occur when:
// bank(T1 * S) == bank(T2 * S) AND T1 != T2

// Number of conflicts = ceil(32 / GCD(S, 32))
//
// S=1:  GCD=1  → No conflicts (best)
// S=2:  GCD=2  → 2-way conflicts
// S=4:  GCD=4  → 4-way conflicts
// S=32: GCD=32 → No conflicts (but poor utilization)
```

### Profiling Bank Conflicts

```bash
# Using NVIDIA Nsight Compute
ncu --metrics smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct \
    --kernel-name my_kernel \
    ./program

# Output:
# Shared Memory Bank Conflicts: 0.0% (ideal)
# Shared Memory Bank Conflicts: 50.0% (2-way conflict)
# Shared Memory Bank Conflicts: 95.0% (32-way conflict - fix immediately!)
```

## Access Pattern Analysis

### Pattern 1: Sequential Access (Conflict-Free)

```cpp
__shared__ float data[1024];
int tid = threadIdx.x;

// Perfect: stride = 1
float val = data[tid];  // No conflicts

/*
Thread 0 → Address 0 → Bank 0
Thread 1 → Address 1 → Bank 1
...
Thread 31 → Address 31 → Bank 31
All different banks!
*/
```

### Pattern 2: Strided Access (May Conflict)

```cpp
__shared__ float data[1024];
int tid = threadIdx.x;
int stride = 2;

// 2-way conflict
float val = data[tid * stride];

/*
Thread 0 → Address 0 → Bank 0  ┐
Thread 1 → Address 2 → Bank 2  │ 16 banks used
...                             │ 2 threads/bank
Thread 16 → Address 32 → Bank 0 ┘ → Conflict!
*/
```

### Pattern 3: Matrix Column Access (Conflicts)

```cpp
__shared__ float matrix[32][32];
int tid = threadIdx.x;
int col = 5;  // Fixed column

// 32-way conflict!
float val = matrix[tid][col];

/*
All threads access column 5:
Thread 0 → matrix[0][5] → Address 5 → Bank 5  ┐
Thread 1 → matrix[1][5] → Address 37 → Bank 5 │ All access
...                                            │ Bank 5!
Thread 31 → matrix[31][5] → Address 1029 → Bank 5 ┘
Fully serialized → 32× slower!
*/
```

### Pattern 4: Padded Matrix (Conflict-Free)

```cpp
// Add padding to shift addresses
__shared__ float matrix[32][33];  // +1 column padding
int tid = threadIdx.x;
int col = 5;

// No conflict!
float val = matrix[tid][col];

/*
With padding (stride = 33):
Thread 0 → matrix[0][5] → Address 5 → Bank 5
Thread 1 → matrix[1][5] → Address 38 → Bank 6  (shifted!)
Thread 2 → matrix[2][5] → Address 71 → Bank 7
...
All different banks due to padding!
*/
```

## Conflict Avoidance Techniques

### Technique 1: Padding

Add extra elements to break alignment:

```cpp
// Without padding: conflicts on column access
__shared__ float tile[32][32];

// With padding: conflict-free
__shared__ float tile[32][33];  // +1 column
__shared__ float tile[32][32+1];  // Explicit padding

// General formula for square tiles:
// Size = [TILE_DIM][TILE_DIM + PADDING]
// where PADDING breaks bank alignment
```

### Technique 2: Swizzling

Permute access pattern to distribute across banks:

```cpp
__shared__ float data[1024];
int tid = threadIdx.x;

// Instead of: data[tid * stride]
// Use XOR to swizzle:
int swizzled = (tid * stride) ^ (tid / 32);
float val = data[swizzled];

// XOR distributes accesses across banks more evenly
```

### Technique 3: Vectorization

Use vector types to reduce bank access frequency:

```cpp
__shared__ float4 data[256];  // Each element is 4 floats (16 bytes)
int tid = threadIdx.x;

// Each access loads 4 floats
float4 val = data[tid];

/*
Thread 0 → data[0] → Banks 0,1,2,3
Thread 1 → data[1] → Banks 4,5,6,7
...
Thread 7 → data[7] → Banks 28,29,30,31
Thread 8 → data[8] → Banks 0,1,2,3 (wraps)

Still no conflicts! Each thread uses 4 consecutive banks.
*/
```

### Technique 4: Reordering Computations

Change algorithm to access memory differently:

```cpp
// BEFORE: Column reduction (conflicts)
__shared__ float matrix[32][32];
for (int row = 0; row < 32; ++row) {
    sum += matrix[row][threadIdx.x];  // All threads same column
}

// AFTER: Transpose in shared memory first
__shared__ float matrix[32][33];     // Padded
__shared__ float transposed[32][33];

// Load and transpose (conflict-free both ways)
transposed[threadIdx.y][threadIdx.x] = matrix[threadIdx.x][threadIdx.y];
__syncthreads();

// Now row reduction (conflict-free)
for (int col = 0; col < 32; ++col) {
    sum += transposed[threadIdx.x][col];
}
```

## Shared Memory in Attention

### Use Case: Loading Query Vector

```cpp
// Attention kernel: Load Q to shared memory

template <int HEAD_SIZE, int BLOCK_SIZE>
__global__ void attention_kernel(/*...*/) {
    // Shared memory for query (reused for all K, V)
    __shared__ float Q_shared[HEAD_SIZE];

    // Collaborative load (conflict-free)
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
        Q_shared[i] = Q_global[head_idx * HEAD_SIZE + i];
    }
    __syncthreads();

    // Reuse Q_shared for computing all attention scores
    // Saves HEAD_SIZE reads from global memory per K/V pair
}
```

### Use Case: Attention Score Storage

```cpp
// Store attention scores in shared memory for softmax

template <int PARTITION_SIZE>
__global__ void attention_kernel(/*...*/) {
    extern __shared__ float logits[];  // Size: PARTITION_SIZE

    // Each thread computes scores for some tokens
    for (int token = warp_idx; token < partition_size; token += NUM_WARPS) {
        float score = compute_qk_score(Q, K[token]);

        // Write to shared memory (conflict-free if sequential)
        if (thread_group_offset == 0) {
            logits[token] = score;
        }
    }
    __syncthreads();

    // Compute softmax over shared logits
    // Reuse in computing attention output
}
```

### Use Case: Warp Reduction

```cpp
// Efficient warp-level reduction using shared memory

template <int NUM_WARPS>
__device__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[NUM_WARPS];

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    // Warp-level reduction (registers only)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // First thread in warp writes to shared memory
    // No bank conflict: different warps write to different locations
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warp sums
    if (warp_id == 0) {
        val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;

        // Final warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    return val;  // Thread 0 has final sum
}
```

## Dynamic Shared Memory

### Allocation

```cpp
// Kernel declaration with dynamic shared memory
__global__ void my_kernel(float* data, int size) {
    extern __shared__ float shared[];  // Size determined at launch

    // Use shared memory
    int tid = threadIdx.x;
    shared[tid] = data[tid];
    __syncthreads();
    // ...
}

// Launch with specified shared memory size
int shared_size = 1024 * sizeof(float);
my_kernel<<<blocks, threads, shared_size>>>(data, size);
```

### Multiple Arrays in Shared Memory

```cpp
__global__ void multi_array_kernel() {
    extern __shared__ char shared_memory[];

    // Partition shared memory into multiple arrays
    float* array1 = reinterpret_cast<float*>(shared_memory);
    float* array2 = array1 + SIZE1;
    int* array3 = reinterpret_cast<int*>(array2 + SIZE2);

    // Use arrays
    array1[threadIdx.x] = ...;
    array2[threadIdx.x] = ...;
    array3[threadIdx.x] = ...;
}

// Calculate total size
int shared_size = SIZE1 * sizeof(float) +
                  SIZE2 * sizeof(float) +
                  SIZE3 * sizeof(int);
multi_array_kernel<<<blocks, threads, shared_size>>>();
```

### vLLM Pattern: Reusing Shared Memory

```cpp
// vLLM attention kernel: Reuse shared memory for different phases

template <int NUM_THREADS, int PARTITION_SIZE, int HEAD_SIZE>
__global__ void paged_attention_kernel(/*...*/) {
    const int NUM_WARPS = NUM_THREADS / 32;

    extern __shared__ char shared_mem[];

    // Phase 1: Use as logits storage
    float* logits = reinterpret_cast<float*>(shared_mem);
    // Size: PARTITION_SIZE * sizeof(float)

    // Compute attention scores
    for (int token = ...) {
        logits[token] = compute_score(Q, K[token]);
    }
    __syncthreads();

    // Apply softmax to logits
    softmax_inplace(logits, partition_size);
    __syncthreads();

    // Phase 2: Reuse as output accumulator
    // (can reuse same memory after logits no longer needed)
    float* output_shared = reinterpret_cast<float*>(shared_mem);
    // Size: NUM_WARPS * HEAD_SIZE * sizeof(float)

    // Compute attention output
    for (int token = ...) {
        accumulate_output(output_shared, logits[token], V[token]);
    }

    // Total shared memory = max(logits_size, output_size)
}
```

## vLLM Examples

### Example 1: Attention Query Storage

**File**: `/home/user/vllm-learn/csrc/attention/attention_kernels.cuh` (Lines 175-182)

```cpp
// Load query to shared memory for reuse
__shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];

#pragma unroll
for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
     i += NUM_THREAD_GROUPS) {
  const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
  q_vecs[thread_group_offset][i] =
      *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
}
__syncthreads();
```

**Analysis**:
- Stores query vector in shared memory
- Accessed by all threads for each K computation
- Saves repeated global memory reads
- 2D array indexing: `[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD]`

### Example 2: Block Reduction

**File**: `/home/user/vllm-learn/csrc/attention/attention_kernels.cuh` (Lines 44-77)

```cpp
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Warp-level reduction
  #pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders write to shared memory
  if (lane == 0) {
    red_smem[warp] = sum;  // No bank conflict: different warps
  }
  __syncthreads();

  // Final reduction
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  #pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  return VLLM_SHFL_SYNC(sum, 0);  // Broadcast result
}
```

**Analysis**:
- Combines warp shuffle (no shared memory) with shared memory
- Shared memory only for inter-warp communication
- `red_smem[warp]`: Each warp writes to different location (no conflicts)
- Efficient two-level reduction

### Example 3: LayerNorm Reduction Storage

**File**: `/home/user/vllm-learn/csrc/layernorm_kernels.cu` (Lines 38-40)

```cpp
using BlockReduce = cub::BlockReduce<float, 1024>;
__shared__ typename BlockReduce::TempStorage reduceStore;
variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);
```

**Analysis**:
- Uses CUB library for efficient block reduction
- CUB handles bank conflict avoidance internally
- Shared memory size determined by CUB template
- Production-quality reduction with minimal code

## Hands-on Exercises

### Exercise 1: Detect and Fix Bank Conflicts

Profile and fix this kernel:

```cpp
__global__ void matrix_column_sum(float* out, const float* matrix,
                                  int rows, int cols) {
    __shared__ float tile[32][32];

    int row = threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    // Load tile (may have conflicts)
    if (col < cols && row < rows) {
        tile[row][col % 32] = matrix[row * cols + col];
    }
    __syncthreads();

    // Reduce (definitely has conflicts)
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int r = 0; r < 32; ++r) {
            sum += tile[r][threadIdx.y];  // All threads access column threadIdx.y
        }
        out[blockIdx.x * 32 + threadIdx.y] = sum;
    }
}

// TODO: Add padding and redesign access pattern
```

### Exercise 2: Optimize Transpose

Implement an optimized transpose using shared memory:

```cpp
template <int TILE_DIM>
__global__ void transpose_optimized(float* out, const float* in,
                                    int rows, int cols) {
    // TODO: Implement with proper padding
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 for padding

    // Load tile (coalesced)
    // Transpose in shared memory
    // Store tile (coalesced)
}

// Test with TILE_DIM = 16, 32, 64
// Measure bank conflicts and performance
```

### Exercise 3: Shared Memory Budget

Calculate shared memory usage for this configuration:

```cpp
constexpr int HEAD_SIZE = 128;
constexpr int BLOCK_SIZE = 16;
constexpr int PARTITION_SIZE = 512;
constexpr int NUM_THREADS = 128;
constexpr int NUM_WARPS = NUM_THREADS / 32;  // 4 warps

// Shared memory allocations:
__shared__ float q_vecs[HEAD_SIZE];           // Query vector
extern __shared__ float logits[];             // PARTITION_SIZE elements
__shared__ float red_smem[2 * NUM_WARPS];     // Reduction workspace

// Questions:
// 1. Total shared memory usage?
// 2. Does it fit in 48 KB limit?
// 3. How many blocks can run concurrently on an SM with 164 KB shared memory?
```

## Best Practices

### Designing for Conflict-Free Access

1. **Add padding to 2D arrays**:
   ```cpp
   // Square tiles: Add 1
   __shared__ float tile[32][33];

   // Rectangular: Add padding to break alignment
   __shared__ float tile[ROWS][COLS + PAD];
   ```

2. **Use sequential access when possible**:
   ```cpp
   // Good: Sequential
   data[threadIdx.x]

   // Bad: Strided
   data[threadIdx.x * stride]
   ```

3. **Prefer row-wise access in 2D arrays**:
   ```cpp
   // Good: Row-wise (stride = 1)
   tile[row][threadIdx.x]

   // Bad: Column-wise (stride = row_size)
   tile[threadIdx.x][col]
   ```

4. **Profile and verify**:
   ```bash
   ncu --metrics smsp__sass_average_data_bytes_per_wavefront_mem_shared \
       ./program
   ```

### Memory Budget Management

1. **Know your limits**:
   - 48 KB per block (Pascal, Turing)
   - 164 KB per block (Ampere, Hopper) with config
   - Check with `cudaDeviceGetAttribute()`

2. **Trade-off occupancy vs shared memory**:
   ```
   More shared memory per block
   → Fewer blocks per SM
   → Lower occupancy
   → May reduce performance if memory-bound
   ```

3. **Reuse shared memory across phases**:
   ```cpp
   // Phase 1
   __shared__ float temp1[SIZE1];
   // ... use temp1 ...
   __syncthreads();

   // Phase 2: Reuse same memory
   __shared__ float temp2[SIZE2];  // Must have SIZE2 ≤ SIZE1
   // ... use temp2 ...
   ```

### Testing for Correctness

```cpp
void verify_shared_memory_kernel() {
    // Test with different block sizes
    for (int block_size : {64, 128, 256, 512}) {
        // Run kernel
        my_kernel<<<grid, block_size, shared_size>>>(/*...*/);

        // Verify results identical
        assert(results_match(expected, actual));
    }

    // Test bank conflict metrics
    assert(bank_conflict_rate < 5.0);  // < 5% acceptable
}
```

## References

### Official Documentation

1. [CUDA C Programming Guide - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
2. [CUDA Best Practices - Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
3. [Nsight Compute - Shared Memory Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#shared-memory-tables)

### Papers and Articles

1. Volkov, V. (2010). "Better Performance at Lower Occupancy"
2. Harris, M. (2007). "Optimizing Parallel Reduction in CUDA"
3. Ruetsch, G., & Micikevicius, P. (2009). "Optimizing Matrix Transpose in CUDA"

### Tools

1. **NVIDIA Nsight Compute**: Bank conflict profiling
2. **CUDA Occupancy Calculator**: Shared memory vs occupancy
3. **Compute Sanitizer**: Memory access verification

### vLLM Code

1. Attention kernels: `/home/user/vllm-learn/csrc/attention/attention_kernels.cuh`
2. LayerNorm kernels: `/home/user/vllm-learn/csrc/layernorm_kernels.cu`
3. Reduction utilities: Various kernel files using block reductions

## Summary

Shared memory optimization is crucial for high-performance LLM kernels:

**Key Principles**:
1. **Avoid bank conflicts**: Add padding, use sequential access
2. **Maximize reuse**: Load once, use many times (e.g., query vector in attention)
3. **Manage budget**: Balance shared memory usage with occupancy
4. **Profile carefully**: Use tools to verify conflict-free access

**Common Patterns**:
- Query storage in attention: Loaded once, reused for all keys
- Logits storage: Computed and consumed within same kernel
- Block reductions: Two-level (warp + shared memory)
- Tiling: Load tiles to shared memory for reuse

**In vLLM**:
- Attention query vectors cached in shared memory
- Online softmax uses shared memory for intermediate results
- Reduction operations leverage CUB library
- Dynamic shared memory sizing based on problem dimensions

Proper shared memory optimization can provide 2-10× speedups for memory-reuse patterns common in transformers!

---

**Next Tutorial**: [08_tensor_cores_usage.md](08_tensor_cores_usage.md) - Leverage specialized hardware for matrix operations in LLM inference.
