# Tutorial 08: Tensor Cores Usage

## Learning Objectives

After completing this tutorial, you will be able to:

1. Understand tensor core architecture and capabilities
2. Use WMMA API for tensor core programming
3. Leverage CUTLASS library for optimized matrix operations
4. Integrate tensor cores in LLM inference kernels
5. Analyze performance characteristics of tensor core operations

## Prerequisites

- Completion of Tutorials 01-07
- Understanding of matrix multiplication algorithms
- Familiarity with C++ templates and modern C++
- Knowledge of quantization techniques (FP16, INT8, FP8)

## Table of Contents

1. [Tensor Core Architecture](#tensor-core-architecture)
2. [WMMA API Basics](#wmma-api-basics)
3. [Matrix Shapes and Data Types](#matrix-shapes-and-data-types)
4. [CUTLASS Library](#cutlass-library)
5. [Quantized Operations](#quantized-operations)
6. [vLLM Integration](#vllm-integration)
7. [Performance Analysis](#performance-analysis)
8. [Hands-on Exercises](#hands-on-exercises)
9. [Best Practices](#best-practices)
10. [References](#references)

## Tensor Core Architecture

### What Are Tensor Cores?

Tensor cores are specialized hardware units for matrix operations:

```
Traditional CUDA Core:
- Performs scalar operations (1×1 multiply-add per cycle)
- FP32/FP64 operations
- General purpose

Tensor Core:
- Performs matrix operations (16×16 or larger per cycle)
- Specialized for D = A × B + C (matrix multiply-accumulate)
- FP16, BF16, TF32, INT8, FP8, INT4 support
- 8-16× higher throughput for matrix operations
```

### Tensor Core Evolution

```
GPU Generation | Tensor Core Features
─────────────────────────────────────────────────────
Volta (V100)   │ FP16 input, FP32 accumulate
               │ 4×4×4 matrix ops
─────────────────────────────────────────────────────
Turing (T4)    │ + INT8, INT4, INT1
               │ Same as Volta for FP16
─────────────────────────────────────────────────────
Ampere (A100)  │ + BF16, TF32, FP64
               │ 8×8×4 and 16×8×16 ops
               │ Sparsity support (2:4)
─────────────────────────────────────────────────────
Hopper (H100)  │ + FP8 (E4M3, E5M2)
               │ Larger matrix dimensions
               │ FP8 Tensor Core ops
               │ DPX instructions
─────────────────────────────────────────────────────
```

### Performance Comparison

```
A100 GPU Peak Performance:

CUDA Cores (FP16):     ~78 TFLOPS
Tensor Cores (FP16):   312 TFLOPS   (4× improvement)
Tensor Cores (TF32):   156 TFLOPS
Tensor Cores (FP8):    624 TFLOPS   (8× improvement)

Key insight: Use tensor cores for matrix-heavy workloads!
```

## WMMA API Basics

### What is WMMA?

**WMMA** (Warp-level Matrix Multiply and Accumulate) provides a C++ API for tensor cores.

### Basic Structure

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Three steps to use WMMA:

// 1. Declare fragments (register-based matrices)
fragment<matrix_a, M, N, K, half, row_major> a_frag;
fragment<matrix_b, M, N, K, half, col_major> b_frag;
fragment<accumulator, M, N, K, float> c_frag;

// 2. Load from memory to fragments
load_matrix_sync(a_frag, a_ptr, lda);
load_matrix_sync(b_frag, b_ptr, ldb);
fill_fragment(c_frag, 0.0f);  // Initialize accumulator

// 3. Perform matrix multiply-accumulate
mma_sync(c_frag, a_frag, b_frag, c_frag);  // C = A × B + C

// 4. Store result back to memory
store_matrix_sync(c_ptr, c_frag, ldc, mem_row_major);
```

### Simple Matrix Multiplication Example

```cpp
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// Matrix multiplication using WMMA
// C[M×N] = A[M×K] × B[K×N]

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void wmma_matmul(
    half* __restrict__ C,
    const half* __restrict__ A,
    const half* __restrict__ B,
    int M, int N, int K) {

    // Warp and lane IDs
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   half> c_frag;

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Compute C = A × B in tiles
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        // Bounds check
        if (aRow < M && bCol < N) {
            // Load fragments from global memory
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Perform matrix multiply-accumulate
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store the result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N,
                                wmma::mem_row_major);
    }
}
```

## Matrix Shapes and Data Types

### Supported Configurations

```cpp
// Volta / Turing / Ampere (older arch)
// M=16, N=16, K=16 for FP16
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;

// Ampere (newer shapes)
// M=8, N=8, K=4 for FP64
wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> a_frag;

// Tensor Float 32 (TF32) on Ampere+
// Automatic conversion from FP32 → TF32 in tensor cores
wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
               wmma::row_major> a_frag;
```

### Data Type Mapping

| Input Type A | Input Type B | Accumulator Type | Supported Architectures |
|--------------|--------------|------------------|------------------------|
| half (FP16) | half (FP16) | float (FP32) | Volta, Turing, Ampere, Hopper |
| half (FP16) | half (FP16) | half (FP16) | Volta, Turing, Ampere, Hopper |
| bfloat16 | bfloat16 | float (FP32) | Ampere, Hopper |
| tf32 | tf32 | float (FP32) | Ampere, Hopper |
| int8 | int8 | int32 | Turing, Ampere, Hopper |
| fp8_e4m3 | fp8_e4m3 | float (FP32) | Hopper |
| fp8_e5m2 | fp8_e5m2 | float (FP32) | Hopper |

### Fragment Layouts

```cpp
// Row-major matrix A
wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;

// Column-major matrix B
wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;

// Accumulator (no layout - always in register)
wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

/*
Memory Layout:

Row-Major (A):          Column-Major (B):
[0 1 2 3]               [0 4 8  12]
[4 5 6 7]               [1 5 9  13]
[8 9 A B]               [2 6 10 14]
[C D E F]               [3 7 11 15]
*/
```

## CUTLASS Library

### What is CUTLASS?

**CUTLASS** (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's template library for high-performance GEMM and tensor operations.

### Benefits

1. **Performance**: Hand-optimized for each GPU architecture
2. **Flexibility**: Template-based, highly configurable
3. **Maintainability**: Production-quality code
4. **Quantization**: Built-in support for mixed-precision and quantized ops

### Basic CUTLASS GEMM

```cpp
#include "cutlass/gemm/device/gemm.h"

// Define the GEMM operation type
using Gemm = cutlass::gemm::device::Gemm<
    float,                           // ElementA
    cutlass::layout::RowMajor,       // LayoutA
    float,                           // ElementB
    cutlass::layout::ColumnMajor,    // LayoutB
    float,                           // ElementC
    cutlass::layout::RowMajor,       // LayoutC
    float,                           // ElementAccumulator
    cutlass::arch::OpClassTensorOp,  // Use Tensor Cores
    cutlass::arch::Sm80              // Target Ampere
>;

// Launch GEMM
Gemm gemm_op;
cutlass::Status status = gemm_op({
    {M, N, K},           // Problem size
    {A_ptr, lda},        // Matrix A
    {B_ptr, ldb},        // Matrix B
    {C_ptr, ldc},        // Matrix C
    {C_ptr, ldc},        // Matrix D (output)
    {alpha, beta}        // Scalars
});
```

### CUTLASS for FP8 (Hopper)

```cpp
#include "cutlass/gemm/device/gemm.h"

// FP8 GEMM on Hopper
using GemmFp8 = cutlass::gemm::device::Gemm<
    cutlass::float_e4m3_t,           // FP8 E4M3 for A
    cutlass::layout::RowMajor,
    cutlass::float_e4m3_t,           // FP8 E4M3 for B
    cutlass::layout::ColumnMajor,
    float,                           // FP32 output
    cutlass::layout::RowMajor,
    float,                           // FP32 accumulator
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90              // Hopper
>;

// Includes automatic scaling for FP8
```

## Quantized Operations

### Why Quantization?

```
Memory and Compute Savings:

FP32:  32 bits/element
FP16:  16 bits/element  → 2× memory reduction, 2× speedup
INT8:   8 bits/element  → 4× memory reduction, 4× speedup
FP8:    8 bits/element  → 4× memory reduction, 8× speedup (Hopper)
INT4:   4 bits/element  → 8× memory reduction
```

### INT8 Tensor Core Example

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void int8_gemm_kernel(
    int32_t* __restrict__ C,
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int M, int N, int K) {

    // INT8 fragments (16×16×16 on Turing/Ampere)
    fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, int8_t, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, int32_t> c_frag;

    // Initialize accumulator
    fill_fragment(c_frag, 0);

    // Load and compute
    // (similar to FP16 example, but with INT8/INT32 types)
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(a_frag, A + row * K + k, K);
        load_matrix_sync(b_frag, B + k * N + col, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    store_matrix_sync(C + row * N + col, c_frag, N, mem_row_major);
}
```

### FP8 Tensor Cores (Hopper H100)

```cpp
// FP8 E4M3 (4-bit exponent, 3-bit mantissa)
// Better for weights: [-448, 448]

// FP8 E5M2 (5-bit exponent, 2-bit mantissa)
// Better for activations: [-57344, 57344] but less precision

#include <cuda_fp8.h>

__global__ void fp8_gemm_kernel(
    float* __restrict__ C,
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    float scale_a, float scale_b,
    int M, int N, int K) {

    // Use CUTLASS or cuBLASLt for production
    // Manual WMMA for FP8 is complex due to scaling

    // Conceptually:
    // 1. Load FP8 values
    // 2. Implicit conversion to FP32 in tensor core
    // 3. Multiply-accumulate in FP32
    // 4. Apply scales
    // 5. Output in FP32 or FP8
}
```

## vLLM Integration

### CUTLASS in vLLM

vLLM uses CUTLASS extensively for quantized matrix operations:

**File**: `/home/user/vllm-learn/csrc/attention/mla/sm100_cutlass_mla_kernel.cu`

```cpp
// CUTLASS-based Multi-head Latent Attention kernel
// Optimized for SM100 (Hopper/Next-gen)

// Uses:
// - Tensor cores for attention QK^T matmul
// - Tensor cores for attention scores × V
// - FP8 quantization for memory efficiency
// - CUTLASS templates for flexibility
```

### Quantized GEMM

vLLM implements various quantized GEMM operations:

```
/csrc/quantization/w8a8/cutlass/
├── scaled_mm_entry.cu         # Entry point for W8A8 ops
├── scaled_mm_c2x.cu           # Ampere (compute capability 8.x)
├── scaled_mm_c3x_sm90.cu      # Hopper (compute capability 9.0)
├── scaled_mm_c3x_sm100.cu     # Next-gen architecture
└── moe/
    └── grouped_mm_c3x_sm90.cu # MoE-specific grouped GEMM
```

### W8A8 (8-bit Weight, 8-bit Activation)

```cpp
// Conceptual structure of vLLM's W8A8 GEMM

// Input:
// - A: Activations (INT8 or FP8)
// - B: Weights (INT8 or FP8)
// - scale_a: Per-tensor or per-token scale
// - scale_b: Per-tensor or per-channel scale

// Output:
// - C: FP16 or BF16 (dequantized)

// Kernel does:
// 1. Load quantized A, B
// 2. Tensor core matmul (INT8×INT8 → INT32)
// 3. Apply scales: C_fp = (A_int × B_int) × scale_a × scale_b
// 4. Optional: Add bias, apply activation
// 5. Store in FP16/BF16
```

### Example: Scaled Matrix Multiply

```cpp
// Simplified version of vLLM's scaled_mm

template <typename InputType, typename OutputType>
__global__ void scaled_mm_kernel(
    OutputType* __restrict__ C,
    const InputType* __restrict__ A,
    const InputType* __restrict__ B,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int M, int N, int K) {

    // Use CUTLASS GEMM with custom epilogue
    // Epilogue applies: output = (matmul_result) * scale_a * scale_b

    // Pseudo-code:
    // 1. Tensor core matmul: C_int = A_int × B_int
    // 2. Scale: C_float = C_int × scale_a × scale_b
    // 3. Convert to output type (FP16/BF16)
    // 4. Store
}
```

## Performance Analysis

### Tensor Core vs CUDA Core Performance

```
Matrix Multiplication: C[4096×4096] = A[4096×4096] × B[4096×4096]

CUDA Cores (FP32):
- FLOPs: 2 × 4096³ = 137 GFLOPs
- A100 peak: 19.5 TFLOPS
- Time: 137 / 19500 = 7.0 ms

Tensor Cores (FP16):
- FLOPs: 2 × 4096³ = 137 GFLOPs
- A100 peak: 312 TFLOPS
- Time: 137 / 312000 = 0.44 ms

Speedup: 7.0 / 0.44 = 16×
```

### Profiling with Nsight Compute

```bash
# Profile tensor core utilization
ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
    --kernel-name ".*gemm.*" \
    ./program

# Key metrics:
# - Tensor Active: Percentage of time tensor cores are active
# - SM Occupancy: Thread blocks per SM
# - Memory Throughput: Bandwidth utilization

# Good results:
# - Tensor Active > 80%
# - Memory Throughput > 70%
```

### Performance Tuning

```cpp
// Tile sizes matter for tensor cores

// Too small: Underutilize tensor cores
constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_K = 16;
// Result: Low tensor core utilization

// Optimal: Match tensor core dimensions, maximize reuse
constexpr int TILE_M = 128;  // Multiple of 16
constexpr int TILE_N = 128;  // Multiple of 16
constexpr int TILE_K = 32;   // Multiple of 16
// Result: High tensor core utilization, good data reuse
```

## Hands-on Exercises

### Exercise 1: Implement WMMA Matrix Multiply

Complete this WMMA-based matrix multiplication:

```cpp
#include <mma.h>
using namespace nvcuda;

template <int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void wmma_matmul_tiled(
    half* C, const half* A, const half* B,
    int M, int N, int K) {

    // TODO: Implement tiled matrix multiplication
    // 1. Calculate tile indices
    // 2. Declare fragments
    // 3. Load tiles in loop over K
    // 4. Accumulate using mma_sync
    // 5. Store result

    // Hints:
    // - Use shared memory for tiling
    // - Each warp processes one output tile
    // - Accumulate partial results
}

// Test with M=N=K=1024, compare with cuBLAS
```

### Exercise 2: Profile Tensor Core Utilization

Profile these configurations and analyze results:

```python
import torch

# Configuration 1: Small matrices (bad for tensor cores)
A = torch.randn(128, 128, dtype=torch.float16, device='cuda')
B = torch.randn(128, 128, dtype=torch.float16, device='cuda')
C = torch.matmul(A, B)

# Configuration 2: Large matrices (good for tensor cores)
A = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
B = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
C = torch.matmul(A, B)

# Configuration 3: Quantized (best throughput)
A = torch.randint(-128, 127, (4096, 4096), dtype=torch.int8, device='cuda')
B = torch.randint(-128, 127, (4096, 4096), dtype=torch.int8, device='cuda')
C = torch._int_mm(A, B)  # INT8 matmul

# Profile each with ncu and compare:
# - Tensor core utilization
# - Memory bandwidth
# - Achieved FLOPs
```

### Exercise 3: Implement INT8 Quantized GEMM

```cpp
// Implement INT8 quantized GEMM with scaling

__global__ void quantized_gemm(
    half* __restrict__ C,
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int M, int N, int K) {

    // TODO:
    // 1. Use WMMA for INT8×INT8→INT32
    // 2. Apply per-tensor or per-channel scaling
    // 3. Convert to FP16 output
    // 4. Store result

    // Bonus: Add per-token scaling support
}
```

## Best Practices

### When to Use Tensor Cores

✅ **Good candidates**:
- Matrix multiplications (GEMM): Linear layers, attention QK^T, PV
- Large matrices: M, N, K >= 128 (better utilization)
- Mixed precision: FP16/BF16 for speed, FP32 accumulator for accuracy
- Quantized inference: INT8, FP8 for maximum throughput

❌ **Poor candidates**:
- Small matrices: M, N, K < 64 (overhead dominates)
- Irregular shapes: Not multiples of 16
- Sparse matrices: Use sparsity-aware kernels instead
- Memory-bound ops: Tensor cores won't help

### Optimization Checklist

1. **Data types**:
   ```cpp
   // Use FP16 or BF16 for tensor core input
   half* A;
   half* B;
   float* C;  // FP32 accumulator for accuracy
   ```

2. **Alignment**:
   ```cpp
   // Ensure matrix dimensions are multiples of 16
   assert(M % 16 == 0 && N % 16 == 0 && K % 16 == 0);
   ```

3. **Tile sizes**:
   ```cpp
   // Use larger tiles for better utilization
   constexpr int TILE_M = 128;  // Not 16 or 32
   constexpr int TILE_N = 128;
   constexpr int TILE_K = 32;
   ```

4. **Launch configuration**:
   ```cpp
   // Enough blocks to saturate GPU
   int num_blocks = (M / TILE_M) * (N / TILE_N);
   int threads_per_block = 128;  // 4 warps
   ```

5. **Memory access**:
   ```cpp
   // Coalesced loads into shared memory
   // Use ldmatrix for optimal loading (on Ampere+)
   ```

### Common Pitfalls

❌ **Avoid**:
```cpp
// Pitfall 1: Non-aligned dimensions
// M=1023 → 1023 % 16 != 0 → Inefficient

// Pitfall 2: Wrong data type
float* A;  // Should be half* for tensor cores

// Pitfall 3: Small tiles
constexpr int TILE = 16;  // Too small, low utilization

// Pitfall 4: Not using accumulator properly
wmma::fragment<accumulator, 16, 16, 16, half> c_frag;
// Should use float for accumulator when possible
```

✅ **Do**:
```cpp
// Fix 1: Pad dimensions
int M_padded = ((M + 15) / 16) * 16;

// Fix 2: Convert to FP16
half* A = convert_to_fp16(A_fp32);

// Fix 3: Use larger tiles
constexpr int TILE = 128;

// Fix 4: Use FP32 accumulator
wmma::fragment<accumulator, 16, 16, 16, float> c_frag;
```

## References

### Official Documentation

1. [CUDA C Programming Guide - WMMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
2. [Using Tensor Cores in CUDA](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#tensor-cores)
3. [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)

### Papers

1. **Tensor Cores**:
   - Markidis, S. et al. (2018). "NVIDIA Tensor Core Programmability, Performance & Precision"
   - Jia, Z. et al. (2018). "Beyond Data and Model Parallelism for Deep Neural Networks"

2. **Quantization**:
   - Micikevicius, P. et al. (2022). "FP8 Formats for Deep Learning"
   - Dettmers, T. et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers"

### Tools

1. **CUTLASS**: NVIDIA's template library for GEMM
2. **cuBLASLt**: High-level GEMM library with tensor core support
3. **Nsight Compute**: Profiling tensor core utilization

### vLLM Code

1. CUTLASS kernels: `/home/user/vllm-learn/csrc/quantization/w8a8/cutlass/`
2. MLA kernel: `/home/user/vllm-learn/csrc/attention/mla/sm100_cutlass_mla_kernel.cu`
3. Quantization utils: `/home/user/vllm-learn/csrc/quantization/`

## Summary

Tensor cores are essential for high-performance LLM inference:

**Key Capabilities**:
1. **Massive parallelism**: 16×16 matrix ops per cycle
2. **Mixed precision**: FP16/BF16 input, FP32 accumulator
3. **Quantization**: INT8, FP8 for up to 8× speedup
4. **Specialized hardware**: Dedicated units, separate from CUDA cores

**Programming Models**:
- **WMMA API**: Low-level, explicit control
- **CUTLASS**: High-level, production-ready templates
- **cuBLAS**: Highest-level, easiest to use

**In vLLM**:
- Linear layer projections (Q, K, V, output)
- Quantized weight kernels (W8A8, W4A16)
- Attention mechanisms (QK^T, scores×V)
- MoE expert routing and computation

**Performance Impact**:
- 4-16× speedup for FP16 matrix operations
- Up to 8× for FP8 quantized operations
- Critical for achieving <10ms latency per token

Tensor cores transform LLM inference from impossible to practical at scale!

---

**Congratulations!** You've completed Module 3: CUDA Kernels & Optimization. You now have the knowledge to understand and optimize the core computational kernels in vLLM for maximum performance.

**Next Steps**: Apply these techniques to real vLLM kernels, profile your implementations, and contribute optimizations back to the community!
