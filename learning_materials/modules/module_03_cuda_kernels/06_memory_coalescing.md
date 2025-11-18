# Tutorial 06: Memory Coalescing

## Learning Objectives

After completing this tutorial, you will be able to:

1. Understand memory coalescing at the hardware level
2. Identify non-coalesced memory access patterns
3. Transform memory layouts for optimal coalescing
4. Analyze coalescing efficiency using profiling tools
5. Apply coalescing optimizations to LLM kernels

## Prerequisites

- Completion of Tutorial 01: CUDA Memory Hierarchy
- Understanding of GPU memory transactions
- Knowledge of warp-level execution model
- Familiarity with array layouts (row-major vs column-major)

## Table of Contents

1. [Memory Transaction Basics](#memory-transaction-basics)
2. [Coalescing Rules](#coalescing-rules)
3. [Common Access Patterns](#common-access-patterns)
4. [Layout Transformations](#layout-transformations)
5. [Profiling and Measurement](#profiling-and-measurement)
6. [vLLM Examples](#vllm-examples)
7. [Hands-on Exercises](#hands-on-exercises)
8. [Best Practices](#best-practices)
9. [References](#references)

## Memory Transaction Basics

### GPU Memory Hierarchy

```
Warp (32 threads) → Memory Request
         ↓
    L1 Cache (128 KB)
         ↓
    L2 Cache (40 MB)
         ↓
   Global Memory (HBM)
    (transaction size: 32/64/128 bytes)
```

### Memory Transaction Sizes

Modern GPUs use cache lines of specific sizes:

```
Memory Transaction Units:
┌─────────────────────────────────────────┐
│ 32 bytes  │ Minimum transaction size   │
│ 64 bytes  │ Common for L1 cache        │
│ 128 bytes │ L2 cache line              │
└─────────────────────────────────────────┘

Example: Warp reads 32 × 4-byte floats = 128 bytes
- Ideal: 1 transaction of 128 bytes
- Poor: 32 transactions of 4 bytes each
```

### Coalesced vs Non-Coalesced

```
Coalesced Access (GOOD):
Thread:   0    1    2    3  ... 31
Address: [0]  [4]  [8]  [12] ... [124]

Memory Layout (sequential):
┌────┬────┬────┬────┬─────┬────┐
│  0 │  4 │  8 │ 12 │ ... │124 │  ← 128 bytes
└────┴────┴────┴────┴─────┴────┘
         Single 128-byte transaction


Non-Coalesced Access (BAD):
Thread:   0    1    2    3  ... 31
Address: [0]  [128][256][384] ... [3968]

Memory Layout (strided):
┌────┐      ┌────┐      ┌────┐
│  0 │ ...  │128 │ ...  │256 │ ...
└────┘      └────┘      └────┘
  ↑           ↑            ↑
  Transaction 1  Transaction 2  Transaction 3 ...
        32 separate transactions!
```

### Performance Impact

```
Example Calculation:

Scenario: Warp reads 32 floats (128 bytes total)

Coalesced:
- Transactions: 1 × 128 bytes
- Memory bandwidth: 128 bytes / request
- Efficiency: 100%

Stride-2 (non-coalesced):
- Threads access: 0, 8, 16, 24, ..., 248
- Transactions: 2 × 128 bytes (two cache lines)
- Memory bandwidth: 256 bytes / request
- Efficiency: 50%

Random (worst case):
- Threads access random addresses
- Transactions: 32 × 128 bytes (worst case)
- Memory bandwidth: 4096 bytes / request
- Efficiency: 3.125%
```

## Coalescing Rules

### Rule 1: Sequential Access

**Best case**: Adjacent threads access adjacent addresses

```cpp
// PERFECT coalescing
__global__ void coalesced_read(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];  // Thread i accesses element i
    }
}

/*
Warp 0: Threads 0-31 access addresses 0-31
Address pattern: [0, 4, 8, 12, ..., 124] (consecutive)
Result: 1 memory transaction (128 bytes)
*/
```

### Rule 2: Alignment Matters

Memory transactions are aligned to cache line boundaries:

```cpp
// Aligned access (GOOD)
float* aligned_ptr = /* ... aligned to 128 bytes ... */;
idx = threadIdx.x;  // 0, 1, 2, ..., 31
val = aligned_ptr[idx];  // Addresses: 0, 4, 8, ..., 124
// Result: 1 transaction

// Misaligned access (WORSE)
float* misaligned_ptr = aligned_ptr + 1;  // +4 bytes offset
idx = threadIdx.x;
val = misaligned_ptr[idx];  // Addresses: 4, 8, 12, ..., 128
// Result: 2 transactions (crosses cache line boundary)
```

### Rule 3: Stride Affects Coalescing

```cpp
// Stride = 1 (PERFECT)
idx = threadIdx.x;
val = data[idx];  // 100% efficiency

// Stride = 2 (REDUCED)
idx = threadIdx.x * 2;
val = data[idx];  // 50% efficiency (2 transactions instead of 1)

// Stride = 32 (POOR)
idx = threadIdx.x * 32;
val = data[idx];  // ~3% efficiency (32 transactions)

// Stride = 64 or higher (WORST)
idx = threadIdx.x * 64;
val = data[idx];  // Each thread gets own cache line
```

### Rule 4: Permutations Can Coalesce

Order doesn't matter, only that addresses fall in same cache line:

```cpp
// Still coalesced (addresses within same 128-byte region)
__global__ void permuted_access(float* out, const float* in) {
    int idx = threadIdx.x;
    int permuted = (idx + 7) % 32;  // Shuffle order

    if (permuted < 32) {
        out[idx] = in[permuted];  // Still within [0..31]
    }
}
// Result: Still 1-2 transactions (addresses in same cache lines)
```

## Common Access Patterns

### Pattern 1: Row-Major Matrix Access

```cpp
// Matrix stored row-major: [row0...][row1...][row2...]

// GOOD: Row-wise access (coalesced)
__global__ void access_rows(float* A, int cols) {
    int row = blockIdx.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    float val = A[row * cols + col];  // Adjacent threads, adjacent cols
}

// BAD: Column-wise access (non-coalesced)
__global__ void access_cols(float* A, int rows, int cols) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = blockIdx.y;
    float val = A[row * cols + col];  // Adjacent threads, stride = cols
}

/*
For 1024×1024 matrix:
- Row access: Stride = 1 → Coalesced
- Col access: Stride = 1024 → 32 transactions per warp!
*/
```

### Pattern 2: Transpose

```cpp
// Naive transpose (non-coalesced writes)
__global__ void transpose_naive(float* out, const float* in,
                                int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Read coalesced, write non-coalesced!
        out[col * rows + row] = in[row * cols + col];
    }
}

// Optimized transpose (shared memory)
__global__ void transpose_shared(float* out, const float* in,
                                 int rows, int cols) {
    __shared__ float tile[32][33];  // +1 to avoid bank conflicts

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    // Coalesced read from input
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = in[row * cols + col];
    }
    __syncthreads();

    // Transpose in shared memory
    row = blockIdx.x * 32 + threadIdx.y;
    col = blockIdx.y * 32 + threadIdx.x;

    // Coalesced write to output
    if (row < cols && col < rows) {
        out[row * rows + col] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### Pattern 3: Gather/Scatter

```cpp
// Gather (non-coalesced reads)
__global__ void gather(float* out, const float* in, const int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[indices[idx]];  // Random access pattern
    }
}

// Scatter (non-coalesced writes)
__global__ void scatter(float* out, const float* in, const int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[indices[idx]] = in[idx];  // Random write pattern
    }
}

/*
Optimization: If indices have locality, can use shared memory to batch
*/
```

### Pattern 4: Structure of Arrays vs Array of Structures

```cpp
// Array of Structures (AoS) - BAD for coalescing
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};

__global__ void update_aos(Particle* particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Non-coalesced: Thread 0 at byte 0, Thread 1 at byte 24, etc.
        particles[idx].x += particles[idx].vx;
    }
}

// Structure of Arrays (SoA) - GOOD for coalescing
struct ParticlesSoA {
    float *x, *y, *z;
    float *vx, *vy, *vz;
};

__global__ void update_soa(ParticlesSoA particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Coalesced: Thread 0 at byte 0, Thread 1 at byte 4, etc.
        particles.x[idx] += particles.vx[idx];
    }
}
```

## Layout Transformations

### Transformation 1: Add Padding

Avoid bank conflicts and improve coalescing:

```cpp
// Original: May cause bank conflicts
__shared__ float tile[32][32];

// Padded: Avoids conflicts
__shared__ float tile[32][33];  // +1 column

/*
Accessing tile[i][j] where i varies:
- Without padding: stride = 32 elements → bank conflicts
- With padding: stride = 33 elements → no conflicts
*/
```

### Transformation 2: Tiling

Break large problems into tiles that fit in shared memory:

```cpp
template <int TILE_SIZE>
__global__ void tiled_matmul(float* C, const float* A, const float* B,
                              int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Process K dimension in tiles
    for (int t = 0; t < K; t += TILE_SIZE) {
        // Coalesced load into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();

        // Compute using shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

### Transformation 3: Vectorization

Load multiple elements per thread for better bandwidth:

```cpp
// Scalar loads (basic coalescing)
__global__ void copy_scalar(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];  // 4 bytes per thread
    }
}

// Vectorized loads (better bandwidth)
__global__ void copy_vectorized(float* out, const float* in, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 val = *reinterpret_cast<const float4*>(in + idx);
        *reinterpret_cast<float4*>(out + idx) = val;  // 16 bytes per thread
    }
}

/*
Benefits:
- Fewer memory instructions
- Better utilization of memory bandwidth
- Reduces warp scheduling overhead
*/
```

## Profiling and Measurement

### Using NVIDIA Nsight Compute

```bash
# Profile global load efficiency
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
    --kernel-name my_kernel \
    ./my_program

# Output interpretation:
# - 100%: Perfect coalescing
# - 50%: Half efficiency (e.g., stride-2 access)
# - <25%: Poor coalescing (investigate!)

# Detailed memory workload analysis
ncu --set full --section MemoryWorkloadAnalysis \
    --kernel-name my_kernel \
    ./my_program
```

### Metrics to Monitor

```
Key Metrics:

1. Global Load Efficiency
   = (Requested bytes) / (Loaded bytes) × 100%
   Target: >80%

2. Global Store Efficiency
   = (Requested bytes) / (Stored bytes) × 100%
   Target: >80%

3. L1 Hit Rate
   Higher is better for repeated accesses
   Target: >70% for reused data

4. Memory Throughput
   Percentage of peak bandwidth utilized
   Target: >70% for memory-bound kernels
```

### Example Analysis

```
Kernel: Strided Memory Access

Results:
┌──────────────────────────────────────────┐
│ Global Load Transactions: 1024           │
│ Global Load Throughput: 450 GB/s         │
│ Requested Global Load: 4 MB              │
│ Global Load Efficiency: 25.0%            │
└──────────────────────────────────────────┘

Analysis:
- Efficiency 25% → likely stride-4 access
- Loading 16 MB instead of needed 4 MB
- Wasting 12 MB of bandwidth
- Fix: Reorder memory layout or use shared memory
```

## vLLM Examples

### Example 1: Cache Kernel Coalesced Copy

**File**: `/home/user/vllm-learn/csrc/cache_kernels.cu` (Lines 89-93)

```cpp
// Coalesced memory access pattern
for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
  int64_t src_offset = src_block_offset + i;
  int64_t dst_offset = dst_block_offset + i;
  key_cache[dst_offset] = key_cache[src_offset];
}
```

**Analysis**:
- Thread i accesses element i (stride = 1)
- Adjacent threads access adjacent memory locations
- Perfect coalescing for both reads and writes
- Grid-stride loop maintains coalescing throughout

### Example 2: Activation Kernel Vectorization

**File**: `/home/user/vllm-learn/csrc/activation_kernels.cu` (Lines 27-30)

```cpp
for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
  const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
  const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);
  out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
}
```

**Analysis**:
- `VLLM_LDG`: Read-only cache hint for better coalescing
- Sequential access pattern (idx increments by 1)
- Memory layout supports coalescing for both x and y reads
- Output writes are also coalesced

### Example 3: LayerNorm Vectorized Access

**File**: `/home/user/vllm-learn/csrc/layernorm_kernels.cu` (Lines 48-61)

```cpp
scalar_t* out_row = out + blockIdx.x * hidden_size;
auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight);
auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(out_row);

for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
  vec_n_t<scalar_t, VEC_SIZE> dst;
  vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
  vec_n_t<scalar_t, VEC_SIZE> src2 = v_w[i];
  // ... computation ...
  v_out[i] = dst;
}
```

**Analysis**:
- Vectorized loads (VEC_SIZE elements per thread)
- Coalesced access: thread i accesses vector i
- For VEC_SIZE=4 with FP16: 8 bytes per load
- Better bandwidth utilization than scalar access

## Hands-on Exercises

### Exercise 1: Fix Non-Coalesced Access

Given this non-coalesced kernel:

```cpp
__global__ void column_sum(float* out, const float* matrix,
                           int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; ++row) {
            sum += matrix[row * cols + col];  // Stride = cols (BAD!)
        }
        out[col] = sum;
    }
}
```

**Task**: Rewrite using shared memory to achieve coalesced access.

**Hint**: Load tiles of the matrix with coalesced reads, then sum in shared memory.

### Exercise 2: Measure Coalescing Efficiency

Profile these two kernels and compare efficiency:

```cpp
// Version 1: Coalesced
__global__ void read_coalesced(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}

// Version 2: Strided
__global__ void read_strided(float* out, const float* in, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) out[idx] = in[idx];
}
```

**Tasks**:
1. Profile with ncu for strides = 1, 2, 4, 8, 16, 32
2. Plot efficiency vs stride
3. Explain the pattern you observe

### Exercise 3: Optimize Transpose

Implement and compare three transpose versions:

```cpp
// Version 1: Naive (provided above)
// Version 2: Shared memory without padding
// Version 3: Shared memory with padding

// Measure:
// - Global load efficiency
// - Global store efficiency
// - Shared memory bank conflicts
// - Total execution time
```

## Best Practices

### Design for Coalescing

1. **Start with access patterns**:
   ```cpp
   // Good mental model:
   // "Thread i should access element i"
   int idx = global_thread_id;
   data[idx] = ...;
   ```

2. **Use Structure of Arrays (SoA)**:
   ```cpp
   // Better than AoS for GPU
   struct DataSoA {
       float *positions_x, *positions_y, *positions_z;
       float *velocities_x, *velocities_y, *velocities_z;
   };
   ```

3. **Vectorize when possible**:
   ```cpp
   // Load 4 floats at once
   float4 data = *reinterpret_cast<const float4*>(ptr + idx * 4);
   ```

4. **Profile early and often**:
   - Don't assume coalescing is perfect
   - Use ncu to verify
   - Fix the biggest inefficiencies first

### Common Pitfalls

❌ **Avoid**:
```cpp
// Pitfall 1: Column-major access in row-major array
val = matrix[col * rows + row];  // Stride = rows

// Pitfall 2: Large strides
val = data[threadIdx.x * 1024];  // Each thread gets own cache line

// Pitfall 3: Misaligned accesses
float* misaligned = aligned_ptr + 1;
val = misaligned[idx];  // Crosses cache line boundaries

// Pitfall 4: Array of Structures
struct Data {int a; float b; double c;};
val = data_array[idx].b;  // Stride = sizeof(Data)
```

✅ **Do**:
```cpp
// Fix 1: Transpose or use shared memory
__shared__ float tile[32][32];
// Load row-wise, transpose in shared, write col-wise

// Fix 2: Reduce stride via shared memory or reordering
__shared__ float shared[1024];
shared[threadIdx.x] = data[threadIdx.x * 1024];
__syncthreads();
// Process from shared memory

// Fix 3: Ensure alignment
assert(((uintptr_t)ptr % 128) == 0);

// Fix 4: Use SoA
struct DataSoA {int *a; float *b; double *c;};
```

## References

### Official Documentation

1. [CUDA C Programming Guide - Device Memory Access](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
2. [CUDA Best Practices - Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
3. [Nsight Compute - Memory Workload Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-tables)

### Papers

1. Volkov, V., & Demmel, J. W. (2008). "Benchmarking GPUs to tune dense linear algebra"
2. Ruetsch, G., & Micikevicius, P. (2009). "Optimizing Matrix Transpose in CUDA"

### Tools

1. **NVIDIA Nsight Compute**: Memory profiling
2. **CUDA-MEMCHECK**: Detect misaligned accesses
3. **Compute Sanitizer**: Advanced memory checking

### vLLM Code

1. Cache kernels: `/home/user/vllm-learn/csrc/cache_kernels.cu`
2. Activation kernels: `/home/user/vllm-learn/csrc/activation_kernels.cu`
3. LayerNorm kernels: `/home/user/vllm-learn/csrc/layernorm_kernels.cu`

## Summary

Memory coalescing is critical for achieving peak GPU memory bandwidth:

**Key Principles**:
1. **Sequential access**: Adjacent threads → adjacent addresses
2. **Alignment**: Start at cache line boundaries
3. **Minimize stride**: Stride = 1 is ideal
4. **Vectorization**: Load multiple elements per thread

**Common Patterns**:
- Row-major matrix: Access rows, not columns
- Transpose: Use shared memory as staging area
- Gather/scatter: May require algorithmic changes
- SoA over AoS: Better for GPU memory access

**In vLLM**:
- Grid-stride loops maintain coalescing
- Vectorized loads (float2, float4, custom Vec types)
- Careful memory layout in paged cache
- Read-only cache hints (`__ldg`, `VLLM_LDG`)

Achieving 80%+ coalescing efficiency is essential for memory-bound LLM kernels!

---

**Next Tutorial**: [07_shared_memory_optimization.md](07_shared_memory_optimization.md) - Advanced shared memory techniques for LLM kernels.
