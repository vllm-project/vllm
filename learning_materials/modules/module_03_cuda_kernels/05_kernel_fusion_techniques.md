# Tutorial 05: Kernel Fusion Techniques

## Learning Objectives

After completing this tutorial, you will be able to:

1. Understand the benefits and trade-offs of kernel fusion
2. Identify fusion opportunities in LLM inference pipelines
3. Implement fused kernels for common operation patterns
4. Analyze performance improvements from fusion
5. Apply fusion techniques from vLLM's production kernels

## Prerequisites

- Completion of Tutorials 01-04
- Understanding of common neural network operations (LayerNorm, activations, etc.)
- Familiarity with kernel launch overhead and memory bandwidth
- Basic knowledge of compiler optimizations

## Table of Contents

1. [Why Kernel Fusion?](#why-kernel-fusion)
2. [Fusion Opportunities](#fusion-opportunities)
3. [Common Fusion Patterns](#common-fusion-patterns)
4. [vLLM Fusion Examples](#vllm-fusion-examples)
5. [Implementation Techniques](#implementation-techniques)
6. [Performance Analysis](#performance-analysis)
7. [Hands-on Exercises](#hands-on-exercises)
8. [Best Practices](#best-practices)
9. [References](#references)

## Why Kernel Fusion?

### The Problem: Kernel Launch Overhead

Each kernel launch incurs overhead:

```
Sequential Kernels (Unfused):

Kernel A          Kernel B          Kernel C
┌─────────┐      ┌─────────┐      ┌─────────┐
│Launch   │  →   │Launch   │  →   │Launch   │
│Execute  │      │Execute  │      │Execute  │
│Write    │      │Write    │      │Write    │
└─────────┘      └─────────┘      └─────────┘
    ↓                ↓                ↓
  Memory          Memory          Memory
  (HBM)           (HBM)           (HBM)

Overhead:
- 3× kernel launch latency (~5-20 μs each)
- 3× memory writes
- 2× intermediate memory reads
```

### The Solution: Kernel Fusion

```
Fused Kernel:

┌───────────────────────────────────┐
│ Single Launch                     │
│   - Execute A                     │
│   - Execute B (on A's output)     │
│   - Execute C (on B's output)     │
│   - Write final result only       │
└───────────────────────────────────┘
    ↓
  Memory (final result only)

Benefits:
- 1× kernel launch
- 1× memory write
- 0× intermediate reads (stay in registers)
```

### Performance Gains

**Sources of improvement**:

1. **Reduced kernel launch overhead**: ~10-60 μs saved
2. **Eliminated intermediate memory traffic**: Major savings for bandwidth-bound ops
3. **Increased data locality**: Data stays in registers/cache
4. **Better instruction-level parallelism**: Compiler can optimize across operations

**Example calculation**:
```
Unfused: LayerNorm → Activation → Add
- 3 kernel launches: 3 × 10 μs = 30 μs
- Memory: 3 writes + 2 reads = 5× (N × d × 4 bytes)
- For N=4096, d=4096: 5 × 64 MB = 320 MB
- Time at 1.5 TB/s: 213 μs
- Total: ~243 μs

Fused: LayerNorm+Activation+Add
- 1 kernel launch: 10 μs
- Memory: 1 write + 0 intermediate = 1× (N × d × 4 bytes)
- Time at 1.5 TB/s: 43 μs
- Total: ~53 μs

Speedup: 243 / 53 ≈ 4.6×
```

## Fusion Opportunities

### Common Patterns in LLM Inference

```
1. Normalization + Linear
   ┌──────────┐
   │LayerNorm │ → │Linear│
   └──────────┘
   Fusion: Normalize and apply linear in one kernel

2. Activation + Gating
   ┌────────────┐
   │ Activation │ → │Multiply│
   └────────────┘
   vLLM: SiLU_and_Mul kernel

3. QKV Projection + RoPE
   ┌──────┐   ┌──────┐   ┌──────┐
   │ Q    │ + │ RoPE │
   │ K    │   │      │
   │ V    │   └──────┘
   └──────┘
   vLLM: Fused QKNorm + RoPE

4. Residual + Normalization
   ┌─────┐
   │Add  │ → │LayerNorm│
   └─────┘
   vLLM: Fused Add + RMSNorm

5. Attention + Output Projection
   ┌──────────┐
   │Attention │ → │Linear│
   └──────────┘
```

### Fusion Decision Framework

**Should you fuse?**

✅ **Yes, fuse when**:
- Operations are elementwise or reduction-based
- Intermediate results fit in registers/shared memory
- Memory bandwidth is the bottleneck
- Operations applied to same tensor shape

❌ **No, keep separate when**:
- Operations have different parallelism patterns
- Intermediate results are reused elsewhere
- Operations have vastly different resource requirements
- Fusion increases register pressure excessively

## Common Fusion Patterns

### Pattern 1: Elementwise Fusion

Combine multiple elementwise operations:

```cpp
// BEFORE: Three separate kernels
__global__ void relu_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmaxf(0.0f, in[idx]);
}

__global__ void scale_kernel(float* out, const float* in, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * scale;
}

__global__ void add_bias_kernel(float* out, const float* in,
                                 const float* bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] + bias[idx];
}

// AFTER: Single fused kernel
__global__ void fused_relu_scale_bias_kernel(
    float* out, const float* in, const float* bias,
    float scale, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        val = fmaxf(0.0f, val);      // ReLU
        val = val * scale;            // Scale
        val = val + bias[idx];        // Add bias
        out[idx] = val;               // Write once
    }
}
```

**Key insight**: Data flows through registers, never written to memory

### Pattern 2: Reduction + Elementwise

Fuse reduction with subsequent elementwise operations:

```cpp
// LayerNorm = Reduce (mean, variance) + Elementwise (normalize)

template <typename scalar_t, int VEC_SIZE>
__global__ void fused_layernorm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    float epsilon, int hidden_size) {

    __shared__ float s_mean, s_var;
    float sum = 0.0f;
    float sq_sum = 0.0f;

    // Phase 1: Compute mean and variance (reduction)
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = static_cast<float>(input[blockIdx.x * hidden_size + i]);
        sum += val;
        sq_sum += val * val;
    }

    // Block-level reduction
    sum = blockReduceSum(sum);
    sq_sum = blockReduceSum(sq_sum);

    if (threadIdx.x == 0) {
        s_mean = sum / hidden_size;
        s_var = (sq_sum / hidden_size) - (s_mean * s_mean);
    }
    __syncthreads();

    // Phase 2: Normalize (elementwise, fused with phase 1)
    float inv_std = rsqrtf(s_var + epsilon);
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        int idx = blockIdx.x * hidden_size + i;
        float val = static_cast<float>(input[idx]);
        val = (val - s_mean) * inv_std;  // Normalize
        val = val * static_cast<float>(weight[i]) + static_cast<float>(bias[i]);
        out[idx] = static_cast<scalar_t>(val);
    }
}
```

### Pattern 3: Multi-Input Fusion

Fuse operations with multiple inputs:

```cpp
// Residual connection + LayerNorm

template <typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,      // Input + Output (in-place)
    scalar_t* __restrict__ residual,   // Residual connection
    const scalar_t* __restrict__ weight,
    float epsilon, int hidden_size) {

    __shared__ float s_variance;
    float variance = 0.0f;

    // Phase 1: Add residual and compute variance
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        int id = blockIdx.x * hidden_size + idx;

        // Fused add
        scalar_t temp = input[id] + residual[id];
        residual[id] = temp;  // Save for next layer

        // Accumulate for variance
        float x = static_cast<float>(temp);
        variance += x * x;
    }

    // Reduce variance
    variance = blockReduceSum(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();

    // Phase 2: Apply normalization
    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        int id = blockIdx.x * hidden_size + idx;
        float temp = static_cast<float>(residual[id]);
        temp *= s_variance;
        temp *= static_cast<float>(weight[idx]);
        input[id] = static_cast<scalar_t>(temp);
    }
}
```

## vLLM Fusion Examples

### Example 1: Activation and Gating Fusion

**File**: `/home/user/vllm-learn/csrc/activation_kernels.cu` (Lines 12-32)

```cpp
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y) {
  return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&),
          bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    // Load both values
    const scalar_t x = VLLM_LDG(&input[token_idx * 2 * d + idx]);
    const scalar_t y = VLLM_LDG(&input[token_idx * 2 * d + d + idx]);

    // Fused: activation + multiply
    out[token_idx * d + idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
  }
}
```

**Fusion benefit**:
- Combines SiLU/GELU activation with gating multiply
- Common in FFN layers (SwiGLU, GeGLU)
- Saves one kernel launch and one memory pass

### Example 2: Fused Add + RMSNorm

**File**: `/home/user/vllm-learn/csrc/layernorm_kernels.cu` (Lines 68-120)

```cpp
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,
    const int64_t input_stride,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float epsilon, const int num_tokens, const int hidden_size) {

  const int vec_hidden_size = hidden_size / width;
  __shared__ float s_variance;
  float variance = 0.0f;

  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  // Fused: add + variance computation
  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;

    _f16Vec<scalar_t, width> temp = input_v[strided_id];
    temp += residual_v[id];              // Add
    variance += temp.sum_squares();       // Accumulate variance
    residual_v[id] = temp;                // Store for next layer
  }

  // Reduce variance
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // Fused: normalize + scale
  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    int64_t strided_id = blockIdx.x * vec_input_stride + idx;

    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;                   // Normalize
    temp *= weight_v[idx];                // Scale by weight
    input_v[strided_id] = temp;
  }
}
```

**Fusion benefit**:
- Combines residual add with RMS normalization
- Common pattern: every transformer layer has residual + norm
- Vectorized operations for better memory bandwidth
- Saves ~40% memory traffic

### Example 3: Fused QK Norm + RoPE

**File**: `/home/user/vllm-learn/csrc/fused_qknorm_rope_kernel.cu` (Lines 88-100)

```cpp
// Perform per-head QK Norm and RoPE in a single kernel.
template <typename scalar_t_in, typename scalar_t_cache, int head_dim,
          bool interleave>
__global__ void fusedQKNormRopeKernel(
    scalar_t_in* __restrict__ qkv,
    const scalar_t_in* __restrict__ qk_weight,
    const scalar_t_cache* __restrict__ cos_sin_cache,
    const int* __restrict__ positions,
    const int rotary_dim, const int token_num,
    const int qkv_stride, const int head_num,
    const int kv_head_num, const float eps) {

  // This kernel fuses:
  // 1. Q and K head normalization (RMSNorm per head)
  // 2. Rotary position embedding (RoPE) application
  // 3. All in one pass over the data

  // Benefits:
  // - Single memory read of QKV
  // - Intermediate normalized values stay in registers
  // - RoPE applied immediately after normalization
  // - Significant reduction in memory bandwidth
}
```

**Fusion benefit**:
- Combines three operations: load QKV, normalize Q/K, apply RoPE
- Critical path in attention computation
- Adapted from TensorRT-LLM optimization
- Reduces memory bandwidth by ~3×

## Implementation Techniques

### Technique 1: Register Blocking

Keep intermediate values in registers:

```cpp
template <typename T>
__global__ void fused_kernel(T* out, const T* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Load once
        T val = in[idx];

        // Apply multiple operations (stays in register)
        val = operation1(val);
        val = operation2(val);
        val = operation3(val);

        // Write once
        out[idx] = val;
    }
}
```

### Technique 2: Shared Memory Staging

Use shared memory for operations requiring cross-thread communication:

```cpp
__global__ void fused_reduce_transform_kernel(float* out, const float* in,
                                               int n) {
    __shared__ float shared[256];
    __shared__ float reduced_val;

    // Phase 1: Load and reduce
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += in[blockIdx.x * n + i];
    }

    shared[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) sum += shared[i];
        reduced_val = sum / n;  // Mean
    }
    __syncthreads();

    // Phase 2: Transform using reduced value (fused!)
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = in[blockIdx.x * n + i];
        out[blockIdx.x * n + i] = (val - reduced_val) * reduced_val;
    }
}
```

### Technique 3: Template Metaprogramming

Compile-time fusion selection:

```cpp
template <bool FUSE_BIAS, bool FUSE_RELU>
__global__ void generic_kernel(float* out, const float* in,
                               const float* bias, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];

        if constexpr (FUSE_BIAS) {
            val += bias[idx];
        }

        if constexpr (FUSE_RELU) {
            val = fmaxf(0.0f, val);
        }

        out[idx] = val;
    }
}

// Usage:
// No overhead for disabled operations (compiled out)
generic_kernel<true, true><<<...>>>(...)  // Bias + ReLU
generic_kernel<true, false><<<...>>>(...) // Bias only
generic_kernel<false, true><<<...>>>(...)  // ReLU only
```

### Technique 4: Vectorization in Fused Kernels

```cpp
template <typename T, int VEC_SIZE>
__global__ void fused_vectorized_kernel(T* out, const T* in1,
                                        const T* in2, int n) {
    using VecT = typename Vec<T, VEC_SIZE>::Type;

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * VEC_SIZE;
    if (idx < n) {
        // Vectorized load
        VecT vec1 = *reinterpret_cast<const VecT*>(in1 + idx);
        VecT vec2 = *reinterpret_cast<const VecT*>(in2 + idx);

        // Elementwise operations on vectors
        VecT result;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            float val = vec1.data[i] + vec2.data[i];  // Add
            val = tanhf(val);                          // Activation
            result.data[i] = val * 0.5f;               // Scale
        }

        // Vectorized store
        *reinterpret_cast<VecT*>(out + idx) = result;
    }
}
```

## Performance Analysis

### Measuring Fusion Benefits

```bash
# Profile unfused version
ncu --metrics dram_throughput,sm__throughput.avg.pct_of_peak_sustained_active \
    --kernel-name "layernorm_kernel|add_kernel" \
    ./unfused_program

# Profile fused version
ncu --metrics dram_throughput,sm__throughput.avg.pct_of_peak_sustained_active \
    --kernel-name "fused_add_layernorm_kernel" \
    ./fused_program

# Compare:
# - Total kernel time
# - Memory throughput
# - Number of kernel launches
```

### Expected Improvements

| Fusion Pattern | Memory Reduction | Speedup (typical) |
|----------------|------------------|-------------------|
| 2 elementwise ops | 2× → 1× | 1.5-2× |
| 3 elementwise ops | 3× → 1× | 2-3× |
| Reduction + elementwise | 2× → 1× | 1.3-1.8× |
| Residual + norm | 3× → 1× | 1.8-2.5× |
| Activation + gate | 2× → 1× | 1.5-2× |

**Note**: Actual speedup depends on memory bandwidth utilization and kernel launch overhead.

### Profiling Example

```python
import torch
import time

def benchmark_fusion():
    n = 4096 * 4096
    x = torch.randn(n, device='cuda')
    y = torch.randn(n, device='cuda')

    # Warm-up
    for _ in range(10):
        _ = torch.relu(x + y)

    # Unfused: separate kernels
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        temp = x + y          # Kernel 1
        temp = torch.relu(temp)  # Kernel 2
        result = temp * 0.5   # Kernel 3
    torch.cuda.synchronize()
    unfused_time = (time.time() - start) / 100

    # Fused: custom kernel or JIT compilation
    # (implementation depends on framework)

    print(f"Unfused: {unfused_time*1000:.2f} ms")
    print(f"Speedup potential: ~2-3×")
```

## Hands-on Exercises

### Exercise 1: Implement Simple Fusion

Fuse these two kernels:

```cpp
__global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void relu_kernel(float* out, const float* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmaxf(0.0f, in[idx]);
}

// TODO: Implement fused version
__global__ void fused_add_relu_kernel(float* out, const float* a,
                                      const float* b, int n) {
    // Your code here
}
```

**Bonus**: Add vectorization for FP16 inputs.

### Exercise 2: Analyze Fusion Benefit

Calculate theoretical speedup for this fusion:

```
Unfused:
1. MatMul: (M×K) @ (K×N) = M×N
   Memory: Read 2 matrices, write 1 = (MK + KN + MN) × 4 bytes
2. Bias Add: M×N + N
   Memory: Read 2, write 1 = (MN + N + MN) × 4 bytes
3. GELU: M×N → M×N
   Memory: Read 1, write 1 = 2MN × 4 bytes

Fused MatMul+Bias+GELU:
Memory: Read 2 matrices + bias, write 1 = (MK + KN + N + MN) × 4 bytes

Calculate:
a) Total memory traffic for each version (M=4096, K=4096, N=4096)
b) Theoretical speedup from memory reduction
c) Expected wall-clock speedup (accounting for compute)
```

### Exercise 3: Implement Reduction + Elementwise Fusion

Create a fused kernel for mean-centering:

```cpp
// Fuse: compute mean + subtract mean from all elements

__global__ void fused_mean_center_kernel(float* data, int n) {
    __shared__ float mean;

    // TODO: Implement
    // 1. Compute mean (reduction)
    // 2. Subtract mean from each element (elementwise)
    // Both in single kernel
}
```

## Best Practices

### When to Fuse

1. **Profile first**: Measure if memory bandwidth is the bottleneck
2. **Consider register pressure**: Don't fuse if it causes spilling
3. **Check occupancy**: Fusion shouldn't reduce occupancy significantly
4. **Maintain readability**: Document what's fused and why

### Fusion Anti-Patterns

❌ **Don't fuse**:
```cpp
// Bad: Different parallelism patterns
// MatMul (2D) + ColumnSum (1D reduction)
// These need different thread organizations

// Bad: Intermediate result reused multiple times
x = normalize(input)
y = linear1(x)  // Uses x
z = linear2(x)  // Also uses x - can't fuse!

// Bad: Excessive register usage
// Fusing 10 operations → 50 registers → spilling
```

✅ **Do fuse**:
```cpp
// Good: Same shape, sequential operations
x = input + bias
x = activation(x)
output = x * scale

// Good: Reduction + broadcast
mean = compute_mean(input)
output = input - mean  // Fuse!

// Good: Multiple inputs, single output
output = alpha * input1 + beta * input2 + gamma * input3
```

### Testing Fused Kernels

```cpp
// Always verify correctness
void test_fusion() {
    // Generate test data
    std::vector<float> input = generate_random_data();

    // Run unfused version (reference)
    auto ref_output = run_unfused(input);

    // Run fused version
    auto fused_output = run_fused(input);

    // Compare (allow small numerical differences)
    for (size_t i = 0; i < ref_output.size(); ++i) {
        float diff = std::abs(ref_output[i] - fused_output[i]);
        assert(diff < 1e-5 && "Fusion changed results!");
    }
}
```

## References

### Papers

1. **Kernel Fusion**:
   - Chen, T. et al. (2018). "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning"
   - Jia, Z. et al. (2019). "Beyond Data and Model Parallelism for Deep Neural Networks"

2. **Operator Fusion in Practice**:
   - NVIDIA TensorRT Documentation
   - XLA (Accelerated Linear Algebra) from TensorFlow

### vLLM Code

1. **Activation kernels**: `/home/user/vllm-learn/csrc/activation_kernels.cu`
2. **LayerNorm kernels**: `/home/user/vllm-learn/csrc/layernorm_kernels.cu`
3. **Fused QKNorm+RoPE**: `/home/user/vllm-learn/csrc/fused_qknorm_rope_kernel.cu`

### Tools

1. **NVIDIA Nsight Compute**: Profile memory bandwidth and kernel launches
2. **PyTorch JIT**: Automatic fusion for simple patterns
3. **TensorRT**: Advanced fusion for inference

## Summary

Kernel fusion is a critical optimization for LLM inference:

**Key Principles**:
1. **Reduce memory traffic**: Main source of speedup
2. **Eliminate kernel launch overhead**: Secondary benefit
3. **Improve data locality**: Keep data in fast memory

**Common Patterns in vLLM**:
- Residual connections + normalization
- Activation functions + gating
- QKV projection + positional encoding
- Multiple elementwise operations

**Implementation Strategy**:
- Use registers for intermediate values
- Leverage shared memory for reductions
- Apply vectorization for memory efficiency
- Template metaprogramming for flexibility

Fusion is most effective for bandwidth-bound operations with sequential dependencies—exactly the pattern in transformer inference!

---

**Next Tutorial**: [06_memory_coalescing.md](06_memory_coalescing.md) - Deep dive into memory access patterns and coalescing optimization.
