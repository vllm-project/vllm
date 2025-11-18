# Tutorial 02: Kernel Optimization Basics

## Learning Objectives

After completing this tutorial, you will be able to:

1. Understand GPU occupancy and its impact on performance
2. Configure thread blocks optimally for different workloads
3. Utilize warp-level primitives for efficient parallel operations
4. Apply `__launch_bounds__` to control register usage and occupancy
5. Analyze and optimize vLLM kernels for maximum throughput

## Prerequisites

- Completion of Tutorial 01: CUDA Memory Hierarchy
- Understanding of CUDA execution model (grids, blocks, threads, warps)
- Familiarity with GPU architecture (SMs, warps, schedulers)
- Basic profiling experience with NVIDIA Nsight Compute

## Table of Contents

1. [GPU Occupancy Fundamentals](#gpu-occupancy-fundamentals)
2. [Thread Block Configuration](#thread-block-configuration)
3. [Warp-Level Operations](#warp-level-operations)
4. [Launch Bounds Optimization](#launch-bounds-optimization)
5. [Register Pressure Management](#register-pressure-management)
6. [vLLM Kernel Examples](#vllm-kernel-examples)
7. [Hands-on Exercises](#hands-on-exercises)
8. [Best Practices](#best-practices)
9. [References](#references)

## GPU Occupancy Fundamentals

### What is Occupancy?

Occupancy is the ratio of active warps to maximum possible warps on a Streaming Multiprocessor (SM):

```
                    Active Warps per SM
Occupancy (%) = ─────────────────────────── × 100
                Maximum Warps per SM (64 for A100)
```

### Occupancy Visualization

```
┌──────────────────────────────────────────────────────────┐
│                   Streaming Multiprocessor (SM)           │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Maximum: 64 warps (2048 threads)                        │
│                                                           │
│  Low Occupancy (25%):        High Occupancy (75%):       │
│  ┌────┬────┬────┬────┐      ┌────┬────┬────┬────┐       │
│  │ W0 │ W1 │ W2 │ W3 │      │ W0 │ W1 │... │W47 │       │
│  └────┴────┴────┴────┘      └────┴────┴────┴────┘       │
│  ░░░░░░░░░░░░░░░░░░░        ░░░░░░░░░░░░                 │
│  Idle resources             Better latency hiding         │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Occupancy Limiters

Occupancy is limited by:

1. **Registers per thread** → Register file capacity
2. **Shared memory per block** → Shared memory capacity
3. **Threads per block** → Max threads per SM
4. **Blocks per SM** → Max blocks per SM

### Occupancy vs Performance

**IMPORTANT**: Higher occupancy ≠ Better performance

```
Performance

    │     ┌─────────────────  Plateau region
    │    ╱                    (sufficient to hide latency)
    │   ╱
    │  ╱
    │ ╱
    │╱________________________ Occupancy (%)
    0    25    50    75    100

Optimal occupancy depends on:
- Memory vs compute bound workload
- Memory access patterns
- Instruction latency
```

### Calculating Theoretical Occupancy

```cpp
// Example configuration
const int THREADS_PER_BLOCK = 256;
const int REGS_PER_THREAD = 32;
const int SHARED_MEM_PER_BLOCK = 16384;  // bytes

// For A100 GPU:
// - Max 64 warps per SM (2048 threads)
// - 65536 registers per SM
// - 164 KB shared memory per SM

// Register limitation:
int blocks_by_regs = 65536 / (THREADS_PER_BLOCK * REGS_PER_THREAD);
// blocks_by_regs = 65536 / (256 * 32) = 8 blocks

// Shared memory limitation:
int blocks_by_shmem = (164 * 1024) / SHARED_MEM_PER_BLOCK;
// blocks_by_shmem = 168960 / 16384 = 10 blocks

// Thread limitation:
int blocks_by_threads = 2048 / THREADS_PER_BLOCK;
// blocks_by_threads = 2048 / 256 = 8 blocks

// Actual blocks per SM = min(8, 10, 8) = 8 blocks
// Active warps = 8 * (256/32) = 64 warps
// Occupancy = 64/64 = 100%
```

## Thread Block Configuration

### Choosing Block Dimensions

Thread block size affects:
- Occupancy
- Shared memory usage
- Warp efficiency
- Synchronization overhead

### Block Size Guidelines

```
┌─────────────────┬──────────────┬─────────────────────────┐
│ Block Size      │ Warps        │ Use Case                │
├─────────────────┼──────────────┼─────────────────────────┤
│ 32-64 threads   │ 1-2 warps    │ Very low resource use   │
│                 │              │ High divergence         │
├─────────────────┼──────────────┼─────────────────────────┤
│ 128 threads     │ 4 warps      │ Balanced, good default  │
│                 │              │ Moderate resources      │
├─────────────────┼──────────────┼─────────────────────────┤
│ 256 threads     │ 8 warps      │ Common choice           │
│                 │              │ Good occupancy          │
├─────────────────┼──────────────┼─────────────────────────┤
│ 512-1024        │ 16-32 warps  │ Large shared memory     │
│ threads         │              │ Reduce sync overhead    │
└─────────────────┴──────────────┴─────────────────────────┘
```

### Multidimensional Blocks

```cpp
// 1D block - simple linear access
dim3 block(256);
dim3 grid((n + 255) / 256);

// 2D block - image/matrix operations
dim3 block(16, 16);  // 256 threads total
dim3 grid((width + 15) / 16, (height + 15) / 16);

// Thread indexing
int idx = threadIdx.x + blockIdx.x * blockDim.x;  // 1D
int row = blockIdx.y * blockDim.y + threadIdx.y;  // 2D
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### Grid Configuration

```cpp
// Maximize GPU utilization
int num_sms = 108;  // A100 has 108 SMs
int blocks_per_sm = 2;
int total_blocks = num_sms * blocks_per_sm;

// Or based on workload
int total_work = 1000000;
int work_per_block = 256;
int total_blocks = (total_work + work_per_block - 1) / work_per_block;
```

## Warp-Level Operations

A warp is a group of 32 threads that execute in lockstep (SIMT - Single Instruction, Multiple Threads).

### Warp Basics

```
Warp (32 threads):
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7│...              │31│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
 └─────────────────────────────────────────────┘
          Execute same instruction

Lane ID = threadIdx.x % 32
Warp ID = threadIdx.x / 32
```

### Warp-Level Primitives

#### 1. Warp Shuffle

Exchange data between threads in a warp without shared memory:

```cpp
__device__ int warpSum(int val) {
    // Butterfly reduction using warp shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Shuffle variants:
// __shfl_sync(mask, var, srcLane)      - Get from specific lane
// __shfl_up_sync(mask, var, delta)     - Get from lane (id - delta)
// __shfl_down_sync(mask, var, delta)   - Get from lane (id + delta)
// __shfl_xor_sync(mask, var, laneMask) - Get from (id XOR laneMask)
```

#### 2. Warp Vote

Collective decision-making across warp:

```cpp
__device__ bool warpHasWork(bool hasWork) {
    // Check if any thread in warp has work
    return __any_sync(0xffffffff, hasWork);
}

// Vote variants:
// __all_sync(mask, predicate)   - True if all threads true
// __any_sync(mask, predicate)   - True if any thread true
// __ballot_sync(mask, predicate) - Bitmask of results
```

#### 3. Warp Match

Find threads with matching values:

```cpp
__device__ void processMatchingGroups(int key) {
    unsigned mask = __match_any_sync(0xffffffff, key);
    // mask contains bits set for threads with same key value

    if (__popc(mask) > 1) {
        // Multiple threads have same key - can optimize
    }
}
```

### Warp Divergence

Avoid control flow divergence within a warp:

```cpp
// BAD: Warp divergence (50% efficiency)
if (threadIdx.x % 2 == 0) {
    computeA();  // Half warp executes
} else {
    computeB();  // Other half executes (serialized)
}

// GOOD: Warp-aligned branching (100% efficiency)
if (threadIdx.x / 32 == 0) {
    computeA();  // Entire warp 0 executes
} else {
    computeB();  // Entire warp 1 executes
}

// BETTER: Predication (no divergence)
float result = (threadIdx.x % 2 == 0) ? computeA() : computeB();
```

## Launch Bounds Optimization

`__launch_bounds__` helps the compiler optimize kernel configuration:

```cpp
__global__ void
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
my_kernel(/* params */) {
    // kernel code
}
```

### Parameters

1. **MAX_THREADS_PER_BLOCK**: Maximum threads per block for this kernel
2. **MIN_BLOCKS_PER_SM**: Minimum blocks per SM to achieve desired occupancy (optional)

### Benefits

1. **Register allocation**: Compiler limits registers to meet occupancy target
2. **Instruction scheduling**: Better scheduling for known block size
3. **Resource guarantee**: Ensures minimum occupancy

### Example

```cpp
// Target: 50% occupancy with 256 threads per block
// A100: 64 warps max → 32 warps needed for 50%
// 32 warps / (256/32 warps per block) = 4 blocks per SM

__global__ void
__launch_bounds__(256, 4)  // 256 threads, 4 blocks/SM minimum
optimized_kernel(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Compiler will limit registers to achieve 4 blocks/SM
}
```

## Register Pressure Management

### Monitoring Register Usage

```bash
# Compile with verbose PTXAS
nvcc -arch=sm_80 -Xptxas=-v kernel.cu

# Output shows:
# ptxas info : Used 32 registers, 16384 bytes smem,
#              328 bytes cmem[0]
```

### Register Spilling

When register usage exceeds limits, variables spill to local memory (slow!):

```
Register Usage vs Performance:

Throughput
    │
    │  ████████████╗
    │              ║ Registers OK
    │              ║
    │              ╚═════════════
    │                  ▼         ║ Register spilling
    │                            ║ (performance cliff)
    └──────────────────────────────── Registers/Thread
                  64              128
```

### Reducing Register Pressure

```cpp
// BEFORE: High register usage (many live variables)
__global__ void high_regs_kernel(float* out, const float* in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Many intermediate values → high register usage
    float a = in[idx];
    float b = in[idx + n];
    float c = in[idx + 2*n];
    float d = in[idx + 3*n];
    float e = computeE(a, b);
    float f = computeF(c, d);
    float g = computeG(e, f);
    out[idx] = g;
}

// AFTER: Reduced register usage (compute on-demand)
__global__ void low_regs_kernel(float* out, const float* in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Compute and consume immediately
    float result = computeG(
        computeE(in[idx], in[idx + n]),
        computeF(in[idx + 2*n], in[idx + 3*n])
    );
    out[idx] = result;
}
```

## vLLM Kernel Examples

### Example 1: LayerNorm with Occupancy Optimization

**File**: `/home/user/vllm-learn/csrc/layernorm_kernels.cu` (Lines 14-62)

```cpp
template <typename scalar_t, int VEC_SIZE>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,          // [..., hidden_size]
    const scalar_t* __restrict__ input,  // [..., hidden_size]
    const int64_t input_stride,
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  // Phase 1: Compute variance using vectorized loads
  const scalar_t* input_row = input + blockIdx.x * input_stride;
  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };

  // Phase 2: Warp-level reduction using CUB
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // Phase 3: Normalize using vectorized operations
  scalar_t* out_row = out + blockIdx.x * hidden_size;
  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight);
  auto* v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE>*>(out_row);

  for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
    vec_n_t<scalar_t, VEC_SIZE> dst;
    vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
    vec_n_t<scalar_t, VEC_SIZE> src2 = v_w[i];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(src1.val[j]);
      dst.val[j] = ((scalar_t)(x * s_variance)) * src2.val[j];
    }
    v_out[i] = dst;
  }
}
```

**Optimization Analysis**:

1. **Block size**: Uses 1024 threads (32 warps) for BlockReduce template
2. **Vectorization**: `VEC_SIZE` template parameter (2, 4, or 8) reduces memory transactions
3. **Shared memory**: Minimal usage (one float + CUB temp storage)
4. **Warp primitives**: CUB BlockReduce for efficient warp-level reduction
5. **Register usage**: Loop unrolling with `#pragma unroll` keeps data in registers

### Example 2: Launch Bounds in Attention Kernel

**File**: `/home/user/vllm-learn/csrc/rocm/attention.cu` (Line 330)

```cpp
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void
__launch_bounds__(NUM_THREADS, 5)  // 5 blocks minimum per SM
paged_attention_ll4mi_QKV_mfma16_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    const int num_seqs,
    const int num_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride) {
    // Complex attention computation
    // ...
}
```

**Optimization Analysis**:

1. **__launch_bounds__(NUM_THREADS, 5)**:
   - Guarantees 5 blocks per SM minimum
   - Forces compiler to limit register usage
   - Ensures predictable occupancy

2. **Template parameters**: Compile-time constants enable aggressive optimization
3. **Register allocation**: Compiler optimizes to keep 5 blocks resident
4. **Performance**: Trades max occupancy for optimized computation

### Example 3: Activation Kernel Thread Configuration

**File**: `/home/user/vllm-learn/csrc/activation_kernels.cu` (Lines 68-83)

```cpp
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, ACT_FIRST)                 \
  int d = input.size(-1) / 2;                                            \
  int64_t num_tokens = input.numel() / input.size(-1);                   \
  dim3 grid(num_tokens);                                                 \
  dim3 block(std::min(d, 1024));                                         \
  if (num_tokens == 0) {                                                 \
    return;                                                              \
  }                                                                      \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));      \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();          \
  VLLM_DISPATCH_FLOATING_TYPES(                                          \
      input.scalar_type(), "act_and_mul_kernel", [&] {                   \
        vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>, ACT_FIRST>  \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),       \
                                         input.data_ptr<scalar_t>(), d); \
      });
```

**Optimization Analysis**:

1. **Adaptive block size**: `std::min(d, 1024)` adjusts to problem size
2. **Token-level parallelism**: One block per token in grid
3. **No shared memory**: `0` bytes shared memory (all in registers)
4. **Warp efficiency**: Block size is multiple of 32 when d > 32

## Performance Analysis

### Using NVIDIA Nsight Compute

```bash
# Profile kernel occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    --metrics smsp__average_warps_issue_stalled_not_selected.pct \
    ./your_program

# Key metrics to watch:
# - Achieved Occupancy: Actual occupancy during execution
# - Warp Stall Reasons: Why warps are idle
# - Eligible Warps Per Scheduler: Warps ready to execute
```

### CUDA Occupancy Calculator

```bash
# Command-line tool
cuda-occupancy --threads 256 --registers 32 --shared 16384

# Or programmatic:
#include <cuda_occupancy.h>

int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                   my_kernel, 0, 0);
```

## Hands-on Exercises

### Exercise 1: Optimize Block Size

Given this simple vector addition kernel:

```cpp
__global__ void vector_add(float* c, const float* a, const float* b, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch with various block sizes
vector_add<<<(n+127)/128, 128>>>(c, a, b, n);  // 128 threads
vector_add<<<(n+255)/256, 256>>>(c, a, b, n);  // 256 threads
vector_add<<<(n+511)/512, 512>>>(c, a, b, n);  // 512 threads
```

**Tasks**:
1. Profile each configuration with Nsight Compute
2. Measure achieved occupancy and throughput
3. Determine optimal block size for your GPU
4. Explain why that size is optimal

### Exercise 2: Implement Warp Reduction

Implement an efficient warp-level sum using shuffle operations:

```cpp
__device__ float warpReduceSum(float val) {
    // TODO: Implement using __shfl_down_sync
    // Hint: Butterfly pattern with offsets 16, 8, 4, 2, 1
}

__global__ void block_sum(float* out, const float* in, int n) {
    __shared__ float warp_sums[32];  // Up to 32 warps per block

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val = (idx < n) ? in[idx] : 0.0f;

    // Sum within warp
    val = warpReduceSum(val);

    // Store warp results
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Final reduction (first warp only)
    if (warp_id == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? warp_sums[threadIdx.x] : 0.0f;
        val = warpReduceSum(val);
        if (threadIdx.x == 0) {
            out[blockIdx.x] = val;
        }
    }
}
```

### Exercise 3: Apply Launch Bounds

Take this kernel and optimize with launch bounds:

```cpp
// Current kernel (no optimization hints)
__global__ void matrix_multiply_tile(float* C, const float* A,
                                     const float* B, int N) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    float sum = 0.0f;
    for (int tile = 0; tile < N/32; ++tile) {
        As[threadIdx.y][threadIdx.x] = A[row * N + tile * 32 + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(tile * 32 + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < 32; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Tasks**:
1. Calculate register usage with `nvcc -Xptxas=-v`
2. Determine optimal occupancy target
3. Add `__launch_bounds__` to achieve target
4. Verify improved performance

## Best Practices

### Occupancy Optimization

1. **Start with standard sizes**: 128 or 256 threads per block
2. **Profile before optimizing**: Measure actual bottlenecks
3. **Don't over-optimize**: 50% occupancy often sufficient
4. **Consider workload type**:
   - Compute-bound: Lower occupancy OK
   - Memory-bound: Higher occupancy helps hide latency

### Thread Block Configuration

1. **Multiple of warp size**: Always use multiples of 32
2. **2D blocks for 2D data**: Matches memory access patterns
3. **Enough blocks**: At least 2-4× number of SMs for load balancing
4. **Test different sizes**: Optimal size varies by kernel

### Warp-Level Programming

1. **Prefer warp primitives**: Faster than shared memory for small reductions
2. **Avoid divergence**: Keep threads in a warp on same path
3. **Use full mask**: `0xffffffff` for all threads active
4. **Sync carefully**: Warp operations are implicit sync within warp

### Launch Bounds

1. **Use judiciously**: Only when you've identified occupancy issues
2. **Profile first**: Measure register usage and occupancy
3. **Balance resources**: Don't force unrealistic targets
4. **Document intent**: Comment why specific bounds chosen

### Register Management

1. **Monitor spilling**: Use `-Xptxas=-v` flag
2. **Limit live variables**: Compute and consume immediately
3. **Use const**: Enables register/constant memory optimization
4. **Unroll loops**: `#pragma unroll` for small, fixed-size loops

## References

### Official Documentation

1. [CUDA C Programming Guide - Occupancy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#occupancy)
2. [CUDA C Programming Guide - Warp Shuffle](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)
3. [CUDA Best Practices - Launch Bounds](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#launch-bounds)
4. [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

### Tools

1. [CUDA Occupancy Calculator](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)
2. [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
3. [Compiler Options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)

### Academic Papers

1. Volkov, V. (2010). "Better performance at lower occupancy"
2. Wong, H. et al. (2010). "Demystifying GPU microarchitecture through microbenchmarking"
3. Jia, Z. et al. (2019). "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking"

### vLLM Resources

1. vLLM Kernel Source: `/home/user/vllm-learn/csrc/`
2. [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
3. [vLLM Performance Guide](https://docs.vllm.ai/en/latest/performance/)

## Summary

Key takeaways from kernel optimization basics:

- **Occupancy** is important but not the only metric - profile to find actual bottlenecks
- **Thread blocks** should be multiples of 32, with 128-256 threads being common sweet spots
- **Warp primitives** enable efficient intra-warp communication without shared memory
- **Launch bounds** give compiler hints to optimize register allocation for target occupancy
- **Register pressure** can silently kill performance through spilling - monitor carefully

These fundamentals apply to all CUDA kernels, including the complex attention and fusion kernels in vLLM that we'll explore in upcoming tutorials.

---

**Next Tutorial**: [03_attention_kernel_walkthrough.md](03_attention_kernel_walkthrough.md) - Deep dive into attention mechanism implementation in CUDA.
