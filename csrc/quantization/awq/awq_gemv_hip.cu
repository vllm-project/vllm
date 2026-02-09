// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// AWQ GEMV Kernel wrapper for ROCm/HIP
// The actual kernel is in benchmark_ddr/awq_gemv_kernel.hip

#ifdef USE_ROCM

  #include <torch/all.h>
  #include <ATen/hip/HIPContext.h>
  #include <c10/hip/HIPStream.h>
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>
  #include <algorithm>

  // IMPORTANT: Force assert to be active even with -DNDEBUG
  // The assert() call generates code that prevents the compiler from
  // over-optimizing and results in better register allocation and ~50% faster
  // code. See benchmark_ddr/vgpr_assert_minimal.hip for a minimal reproducer.
  #undef NDEBUG
  #include <cassert>

// ============================================================================
// FP16 bit-trick dequantization helpers
// Reinterpret a uint32 as a pair of fp16 values (__half2) via union cast.
// Used to construct half2 values by OR-ing 4-bit nibbles with the fp16
// encoding of 1024.0 (0x6400), avoiding expensive int-to-float conversions.
// ============================================================================
__forceinline__ __device__ __half2 uint32_as_half2(uint32_t val) {
  union {
    uint32_t u;
    __half2 h;
  } converter;
  converter.u = val;
  return converter.h;
}

// Two copies of fp16(1024.0) packed as half2: 0x6400_6400
// fp16 bit layout: [sign(1)][exponent(5)][mantissa(10)]
// 0x6400 = 0_11001_0000000000 = 2^(25-15) * 1.0 = 1024.0
// OR-ing a 4-bit value into the low nibble gives 1024 + value.
// OR-ing into bits [7:4] gives 1024 + value*16.
static constexpr uint32_t FP16_1024_PAIR = 0x64006400u;

// ============================================================================
// AWQ GEMV Kernel with Split-K parallelism
// Accumulates in fp32 for precision, supports SPLIT_K = 1, 2, 4, 8, 16
// ============================================================================
template <int OUTPUT_PER_THREAD, int SPLIT_K>
__global__ __launch_bounds__(SPLIT_K * 32) void awq_gemv_kernel_splitk(
    const __half* __restrict__ activation,  // [K]
    const uint32_t* __restrict__ qweight,   // [K, N/8]
    const __half* __restrict__ scales,      // [K/G, N]
    const uint32_t* __restrict__ qzeros,    // [K/G, N/8]
    __half* __restrict__ output,            // [N]
    size_t M, size_t K, size_t N, size_t G) {
  static_assert(OUTPUT_PER_THREAD == 8,
                "Split-K only supports OUTPUT_PER_THREAD=8");
  static_assert(SPLIT_K >= 1 && SPLIT_K <= 20,
                "SPLIT_K must be between 1 and 20");

  // Thread organization: SPLIT_K splits of 32 threads each
  // SPLIT_K=2: 64 threads/block, SPLIT_K=4: 128 threads/block, SPLIT_K=8: 256
  // threads/block
  constexpr int THREADS_PER_SPLIT = 32;
  constexpr int UINT32_PER_LOAD = OUTPUT_PER_THREAD / 8;  // 1
  constexpr int PIPELINE_DEPTH = 16;
  constexpr int ACC_HALF2_COUNT = OUTPUT_PER_THREAD / 2;  // 4

  // Runtime number of groups and iterations per group
  size_t TOTAL_GROUPS = K / G;
  size_t GROUPS_PER_SPLIT = TOTAL_GROUPS / SPLIT_K;
  size_t ITERS_PER_GROUP = G / PIPELINE_DEPTH;  // G=128: 8, G=64: 4, G=32: 2

  assert(TOTAL_GROUPS > 0 && "TOTAL_GROUPS must be positive");
  assert(TOTAL_GROUPS % SPLIT_K == 0 &&
         "TOTAL_GROUPS must be divisible by SPLIT_K");

  // Determine which split this thread belongs to
  int split_id = threadIdx.x / THREADS_PER_SPLIT;         // 0 or 1
  int thread_in_split = threadIdx.x % THREADS_PER_SPLIT;  // 0-31

  // Calculate column assignment (same for both splits in a pair)
  size_t tid = blockIdx.x * THREADS_PER_SPLIT + thread_in_split;
  size_t col_start = tid * OUTPUT_PER_THREAD;

  if (col_start >= N) return;

  // Shared memory for reduction (fp32 for precision)
  extern __shared__ float
      smem_f[];  // [SPLIT_K][THREADS_PER_SPLIT][OUTPUT_PER_THREAD]
  float* my_smem = &smem_f[split_id * THREADS_PER_SPLIT * OUTPUT_PER_THREAD +
                           thread_in_split * OUTPUT_PER_THREAD];

  typedef const uint32_t __attribute__((address_space(1))) * global_uint32_ptr;

  // Accumulators in fp32 for precision
  float acc[OUTPUT_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
    acc[i] = 0.0f;
  }

  // Starting group for this split
  size_t start_group = split_id * GROUPS_PER_SPLIT;
  size_t start_row = start_group * G;  // G elements per group

  // Pointer setup - offset to starting row
  size_t weight_row_stride = N / 8;  // in uint32_t elements
  const uint32_t* w_ptr =
      qweight + tid * UINT32_PER_LOAD + start_row * weight_row_stride;

  size_t zeros_row_stride = (N / 8) * sizeof(uint32_t);
  size_t zeros_ptr_val = reinterpret_cast<size_t>(qzeros) +
                         tid * UINT32_PER_LOAD * sizeof(uint32_t);

  size_t scales_row_stride = N * sizeof(__half);
  size_t scales_ptr_val =
      reinterpret_cast<size_t>(scales) + col_start * sizeof(__half);

  const __half* act_ptr = activation + start_row;

  // Simplified pointer arithmetic with memory clobber to limit cross-iteration
  // optimization The asm barrier prevents the compiler from over-optimizing
  // across loop iterations which would blow up register usage (205 VGPRs
  // without barrier, 140 with barrier) NOTE: We tried volatile, #pragma
  // nounroll, opaque casts, noinline functions,
  // __launch_bounds__(maxThreads, minBlocks), and partial unrolling (#pragma
  // unroll N) but only asm volatile("" ::: "memory") achieves the right balance
  // of preventing over-optimization while still allowing full loop unrolling
  // for ILP.
  #define W_PTR_ADD_ROW_SK()         \
    do {                             \
      w_ptr += weight_row_stride;    \
      asm volatile("" ::: "memory"); \
    } while (0)

  #define GET_W_PTR_SK() \
    reinterpret_cast<global_uint32_ptr>(reinterpret_cast<size_t>(w_ptr))

  // Pipeline registers
  uint32_t w[PIPELINE_DEPTH][UINT32_PER_LOAD];
  uint32_t packed_zeros[2][UINT32_PER_LOAD];
  __half2 zeros2[2][ACC_HALF2_COUNT];
  __half2 scales2[2][ACC_HALF2_COUNT];
  int curr_buf = 0;

  // Preloaded activations
  __half2 act2[PIPELINE_DEPTH / 2];

  #define LOAD_ACT2_SK(idx)                           \
    __halves2half2(act_ptr[act_row_base + (idx) * 2], \
                   act_ptr[act_row_base + (idx) * 2 + 1])

  #define PRELOAD_ACTIVATIONS_SK(base)                                 \
    do {                                                               \
      _Pragma("unroll") for (int i = 0; i < PIPELINE_DEPTH / 2; i++) { \
        act2[i] = LOAD_ACT2_SK(i);                                     \
      }                                                                \
    } while (0)

  #define LOAD_ZEROS_TO_BUF_SK(group_idx, buf_idx)                     \
    do {                                                               \
      global_uint32_ptr zp = reinterpret_cast<global_uint32_ptr>(      \
          zeros_ptr_val + (group_idx) * zeros_row_stride);             \
      _Pragma("unroll") for (int j = 0; j < UINT32_PER_LOAD; j++) {    \
        packed_zeros[buf_idx][j] = __builtin_nontemporal_load(zp + j); \
      }                                                                \
    } while (0)

  /* FP16 bit-trick zero extraction: OR nibble pairs with fp16(1024.0)    \
     to construct half2 values without int-to-float conversion.            \
     Low-nibble pairs get half2(1024+z0, 1024+z1).                         \
     High-nibble pairs get half2(1024+z2*16, 1024+z3*16).                  \
     The 1024 bias cancels when subtracting from weight half2. */
  #define EXTRACT_ZEROS_IN_BUF_SK(buf_idx)                          \
    do {                                                            \
      _Pragma("unroll") for (int j = 0; j < UINT32_PER_LOAD; j++) { \
        uint32_t za = packed_zeros[buf_idx][j];                     \
        zeros2[buf_idx][j * 4 + 0] =                                \
            uint32_as_half2((za & 0x000f000fu) | FP16_1024_PAIR);   \
        zeros2[buf_idx][j * 4 + 1] =                                \
            uint32_as_half2((za & 0x00f000f0u) | FP16_1024_PAIR);   \
        za >>= 8;                                                   \
        zeros2[buf_idx][j * 4 + 2] =                                \
            uint32_as_half2((za & 0x000f000fu) | FP16_1024_PAIR);   \
        zeros2[buf_idx][j * 4 + 3] =                                \
            uint32_as_half2((za & 0x00f000f0u) | FP16_1024_PAIR);   \
      }                                                             \
    } while (0)

  #define LOAD_SCALES_TO_BUF_SK(group_idx, buf_idx)                     \
    do {                                                                \
      const __half* sp = reinterpret_cast<const __half*>(               \
          scales_ptr_val + (group_idx) * scales_row_stride);            \
      _Pragma("unroll") for (int i = 0; i < ACC_HALF2_COUNT; i++) {     \
        scales2[buf_idx][i] = __halves2half2(sp[i * 2], sp[i * 2 + 1]); \
      }                                                                 \
    } while (0)

  /* FP16 bit-trick weight dequantization:                                   \
     Extract 4-bit nibble pairs as half2 by OR-ing with fp16(1024.0).        \
     Subtract zero half2 (same format) — the +1024 bias cancels exactly.     \
     Low-nibble pairs: diff = (w-z, w'-z') — exact result.                   \
     High-nibble pairs: diff = ((w-z)*16, (w'-z')*16) — divide by 16.       \
     Then multiply by scale and accumulate in fp32. */
  #define ACCUMULATE_SLOT_SK(slot)                                             \
    do {                                                                       \
      __half2 a2 = act2[(slot) / 2];                                           \
      __half act_val = ((slot) % 2 == 0) ? __low2half(a2) : __high2half(a2);   \
      float act_f = __half2float(act_val);                                     \
      _Pragma("unroll") for (int j = 0; j < UINT32_PER_LOAD; j++) {            \
        uint32_t qa = w[slot][j];                                              \
        /* Pair 0: low nibble → elements 0,1 */                                \
        {                                                                      \
          __half2 w_h2 = uint32_as_half2((qa & 0x000f000fu) | FP16_1024_PAIR); \
          __half2 diff = __hsub2(w_h2, zeros2[curr_buf][j * 4 + 0]);           \
          __half2 s2 = scales2[curr_buf][j * 4 + 0];                           \
          float d0 = __half2float(__low2half(diff));                           \
          float d1 = __half2float(__high2half(diff));                          \
          acc[(j * 4 + 0) * 2] += act_f * d0 * __half2float(__low2half(s2));   \
          acc[(j * 4 + 0) * 2 + 1] +=                                          \
              act_f * d1 * __half2float(__high2half(s2));                      \
        }                                                                      \
        /* Pair 1: high nibble → elements 2,3 (has ×16, corrected) */          \
        {                                                                      \
          __half2 w_h2 = uint32_as_half2((qa & 0x00f000f0u) | FP16_1024_PAIR); \
          __half2 diff = __hsub2(w_h2, zeros2[curr_buf][j * 4 + 1]);           \
          __half2 s2 = scales2[curr_buf][j * 4 + 1];                           \
          float d0 = __half2float(__low2half(diff)) * 0.0625f;                 \
          float d1 = __half2float(__high2half(diff)) * 0.0625f;                \
          acc[(j * 4 + 1) * 2] += act_f * d0 * __half2float(__low2half(s2));   \
          acc[(j * 4 + 1) * 2 + 1] +=                                          \
              act_f * d1 * __half2float(__high2half(s2));                      \
        }                                                                      \
        qa >>= 8;                                                              \
        /* Pair 2: low nibble → elements 4,5 */                                \
        {                                                                      \
          __half2 w_h2 = uint32_as_half2((qa & 0x000f000fu) | FP16_1024_PAIR); \
          __half2 diff = __hsub2(w_h2, zeros2[curr_buf][j * 4 + 2]);           \
          __half2 s2 = scales2[curr_buf][j * 4 + 2];                           \
          float d0 = __half2float(__low2half(diff));                           \
          float d1 = __half2float(__high2half(diff));                          \
          acc[(j * 4 + 2) * 2] += act_f * d0 * __half2float(__low2half(s2));   \
          acc[(j * 4 + 2) * 2 + 1] +=                                          \
              act_f * d1 * __half2float(__high2half(s2));                      \
        }                                                                      \
        /* Pair 3: high nibble → elements 6,7 (has ×16, corrected) */          \
        {                                                                      \
          __half2 w_h2 = uint32_as_half2((qa & 0x00f000f0u) | FP16_1024_PAIR); \
          __half2 diff = __hsub2(w_h2, zeros2[curr_buf][j * 4 + 3]);           \
          __half2 s2 = scales2[curr_buf][j * 4 + 3];                           \
          float d0 = __half2float(__low2half(diff)) * 0.0625f;                 \
          float d1 = __half2float(__high2half(diff)) * 0.0625f;                \
          acc[(j * 4 + 3) * 2] += act_f * d0 * __half2float(__low2half(s2));   \
          acc[(j * 4 + 3) * 2 + 1] +=                                          \
              act_f * d1 * __half2float(__high2half(s2));                      \
        }                                                                      \
      }                                                                        \
    } while (0)

  // Load zeros and scales for first group of this split
  LOAD_ZEROS_TO_BUF_SK(start_group, 0);
  EXTRACT_ZEROS_IN_BUF_SK(0);
  LOAD_SCALES_TO_BUF_SK(start_group, 0);

  // Load first 16 weight rows (partial unroll to reduce register pressure)
  #pragma unroll
  for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
    global_uint32_ptr p = GET_W_PTR_SK();
  #pragma unroll
    for (int j = 0; j < UINT32_PER_LOAD; j++) {
      w[slot][j] = __builtin_nontemporal_load(p + j);
    }
    W_PTR_ADD_ROW_SK();
  }

  // Prefetch zeros/scales for next group
  if (start_group + 1 < start_group + GROUPS_PER_SPLIT) {
    LOAD_ZEROS_TO_BUF_SK(start_group + 1, 1);
    LOAD_SCALES_TO_BUF_SK(start_group + 1, 1);
  }

  size_t act_row_base = 0;
  PRELOAD_ACTIVATIONS_SK(act_row_base);

  // Main loop: process GROUPS_PER_SPLIT groups
  for (size_t group = 0; group < GROUPS_PER_SPLIT - 1; group++) {
    for (size_t inner = 0; inner < ITERS_PER_GROUP; inner++) {
  #pragma unroll
      for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
        ACCUMULATE_SLOT_SK(slot);
      }
      act_row_base += PIPELINE_DEPTH;
      PRELOAD_ACTIVATIONS_SK(act_row_base);

  #pragma unroll
      for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
        global_uint32_ptr p = GET_W_PTR_SK();
  #pragma unroll
        for (int j = 0; j < UINT32_PER_LOAD; j++) {
          w[slot][j] = __builtin_nontemporal_load(p + j);
        }
        W_PTR_ADD_ROW_SK();
      }
    }

    int next_buf = 1 - curr_buf;
    EXTRACT_ZEROS_IN_BUF_SK(next_buf);
    curr_buf = next_buf;

    if (group + 2 < GROUPS_PER_SPLIT) {
      int prefetch_buf = 1 - curr_buf;
      LOAD_ZEROS_TO_BUF_SK(start_group + group + 2, prefetch_buf);
      LOAD_SCALES_TO_BUF_SK(start_group + group + 2, prefetch_buf);
    }
  }

  // Last group
  for (size_t inner = 0; inner < ITERS_PER_GROUP - 1; inner++) {
  #pragma unroll
    for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
      ACCUMULATE_SLOT_SK(slot);
    }
    act_row_base += PIPELINE_DEPTH;
    PRELOAD_ACTIVATIONS_SK(act_row_base);

  #pragma unroll
    for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
      global_uint32_ptr p = GET_W_PTR_SK();
  #pragma unroll
      for (int j = 0; j < UINT32_PER_LOAD; j++) {
        w[slot][j] = __builtin_nontemporal_load(p + j);
      }
      W_PTR_ADD_ROW_SK();
    }
  }

  // Epilogue
  #pragma unroll
  for (int slot = 0; slot < PIPELINE_DEPTH; slot++) {
    ACCUMULATE_SLOT_SK(slot);
  }

  #undef PRELOAD_ACTIVATIONS_SK
  #undef LOAD_ZEROS_TO_BUF_SK
  #undef EXTRACT_ZEROS_IN_BUF_SK
  #undef LOAD_SCALES_TO_BUF_SK
  #undef ACCUMULATE_SLOT_SK
  #undef LOAD_ACT2_SK
  #undef GET_W_PTR_SK
  #undef W_PTR_ADD_ROW_SK

  // For SPLIT_K=1, no reduction needed - write directly to output and return
  if constexpr (SPLIT_K == 1) {
  #pragma unroll
    for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
      size_t col = col_start + i;
      if (col < N) {
        output[col] = __float2half(acc[i]);
      }
    }
    return;
  }

  // Store partial results to shared memory (fp32)
  #pragma unroll
  for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
    my_smem[i] = acc[i];
  }

  __syncthreads();

  // Helper: check if SPLIT_K is a power of 2
  constexpr bool IS_POW2 = (SPLIT_K & (SPLIT_K - 1)) == 0;

  if constexpr (IS_POW2) {
    // Tree reduction across splits (in fp32)
    // For SPLIT_K=2: one reduction step
    // For SPLIT_K=4: two reduction steps
    // For SPLIT_K=8: three reduction steps
    // For SPLIT_K=16: four reduction steps
    if constexpr (SPLIT_K >= 16) {
      // Step 0: splits 0-7 add splits 8-15
      if (split_id < 8) {
        float* other_smem =
            &smem_f[(split_id + 8) * THREADS_PER_SPLIT * OUTPUT_PER_THREAD +
                    thread_in_split * OUTPUT_PER_THREAD];
  #pragma unroll
        for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
          acc[i] = my_smem[i] + other_smem[i];
          my_smem[i] = acc[i];
        }
      }
      __syncthreads();
    }

    if constexpr (SPLIT_K >= 8) {
      // Step 1: splits 0-3 add splits 4-7
      if (split_id < 4) {
        float* other_smem =
            &smem_f[(split_id + 4) * THREADS_PER_SPLIT * OUTPUT_PER_THREAD +
                    thread_in_split * OUTPUT_PER_THREAD];
  #pragma unroll
        for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
          acc[i] = my_smem[i] + other_smem[i];
          my_smem[i] = acc[i];
        }
      }
      __syncthreads();
    }

    if constexpr (SPLIT_K >= 4) {
      // Step 2: splits 0-1 add splits 2-3
      if (split_id < 2) {
        float* other_smem =
            &smem_f[(split_id + 2) * THREADS_PER_SPLIT * OUTPUT_PER_THREAD +
                    thread_in_split * OUTPUT_PER_THREAD];
  #pragma unroll
        for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
          acc[i] = my_smem[i] + other_smem[i];
          my_smem[i] = acc[i];
        }
      }
      __syncthreads();
    }

    // Final step: split 0 adds split 1
    if (split_id == 0) {
      float* other_smem = &smem_f[1 * THREADS_PER_SPLIT * OUTPUT_PER_THREAD +
                                  thread_in_split * OUTPUT_PER_THREAD];

  #pragma unroll
      for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
        acc[i] = my_smem[i] + other_smem[i];
      }

      // Write outputs (convert fp32 accumulators to fp16)
  #pragma unroll
      for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
        size_t col = col_start + i;
        if (col < N) {
          output[col] = __float2half(acc[i]);
        }
      }
    }
  } else {
    // Non-power-of-2 SPLIT_K: serial reduction by split 0
    // For SPLIT_K=5,10,20: split 0 sums all other splits' partial results.
    // The reduction cost is negligible (SPLIT_K-1 additions of 8 floats each)
    // compared to the main compute loop.
    if (split_id == 0) {
      for (int s = 1; s < SPLIT_K; s++) {
        float* other_smem = &smem_f[s * THREADS_PER_SPLIT * OUTPUT_PER_THREAD +
                                    thread_in_split * OUTPUT_PER_THREAD];
  #pragma unroll
        for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
          acc[i] += other_smem[i];
        }
      }

      // Write outputs (convert fp32 accumulators to fp16)
  #pragma unroll
      for (int i = 0; i < OUTPUT_PER_THREAD; i++) {
        size_t col = col_start + i;
        if (col < N) {
          output[col] = __float2half(acc[i]);
        }
      }
    }
  }
}

// PyTorch binding wrapper
torch::Tensor awq_gemv_hip(torch::Tensor activation,  // [M, K] or [K]
                           torch::Tensor qweight,     // [K, N/8]
                           torch::Tensor scales,      // [K/G, N]
                           torch::Tensor qzeros,      // [K/G, N/8]
                           int64_t split_k)           // 0=auto, 1/2/4/8/16
{
  // ========== Dimension checks ==========
  TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D, got ", qweight.dim(),
              "D");
  TORCH_CHECK(qzeros.dim() == 2, "qzeros must be 2D, got ", qzeros.dim(), "D");
  TORCH_CHECK(scales.dim() == 2, "scales must be 2D, got ", scales.dim(), "D");

  // Get dimensions from qweight (the authoritative source)
  int64_t K = qweight.size(0);
  int64_t N = qweight.size(1) * 8;  // Each uint32 packs 8 int4 values
  int64_t num_groups = qzeros.size(0);

  TORCH_CHECK(num_groups > 0, "num_groups must be positive, got ", num_groups);
  TORCH_CHECK(K % num_groups == 0, "K (", K,
              ") must be divisible by num_groups (", num_groups, ")");

  int64_t G = K / num_groups;  // Group size

  // ========== M=1 (GEMV) constraint ==========
  TORCH_CHECK(activation.dim() == 1 ||
                  (activation.dim() == 2 && activation.size(0) == 1),
              "awq_gemv_hip only supports M=1 (GEMV), got activation shape ",
              activation.sizes());

  // ========== Group size constraint ==========
  // Kernel uses PIPELINE_DEPTH=16, so G must be divisible by 16
  // Currently we only support G=128 for optimal performance
  TORCH_CHECK(G == 128, "awq_gemv_hip only supports group_size=128, got ", G);
  TORCH_CHECK(G % 16 == 0, "group_size (", G,
              ") must be divisible by PIPELINE_DEPTH (16)");

  // ========== N dimension constraints ==========
  // N must be divisible by 8 for weight packing (8 int4 per uint32)
  TORCH_CHECK(N % 8 == 0, "N (", N, ") must be divisible by 8");

  // ========== Activation size check ==========
  int64_t act_size =
      activation.dim() == 1 ? activation.size(0) : activation.size(1);
  TORCH_CHECK(act_size >= K, "activation size (", act_size, ") must be >= K (",
              K, ")");

  // ========== Scales shape validation ==========
  // scales must be [K/G, N] = [num_groups, N]
  TORCH_CHECK(scales.size(0) == num_groups, "scales.size(0) (", scales.size(0),
              ") must equal num_groups (", num_groups, ")");
  TORCH_CHECK(scales.size(1) == N, "scales.size(1) (", scales.size(1),
              ") must equal N (", N, ")");

  // ========== Qzeros shape validation ==========
  // qzeros must be [K/G, N/8] = [num_groups, N/8]
  TORCH_CHECK(qzeros.size(0) == num_groups, "qzeros.size(0) (", qzeros.size(0),
              ") must equal num_groups (", num_groups, ")");
  TORCH_CHECK(qzeros.size(1) == N / 8, "qzeros.size(1) (", qzeros.size(1),
              ") must equal N/8 (", N / 8, ")");

  // ========== Contiguity checks ==========
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(qweight.is_contiguous(), "qweight must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(qzeros.is_contiguous(), "qzeros must be contiguous");

  // ========== Dtype checks ==========
  TORCH_CHECK(activation.scalar_type() == at::ScalarType::Half,
              "activation must be float16, got ", activation.scalar_type());
  TORCH_CHECK(scales.scalar_type() == at::ScalarType::Half,
              "scales must be float16, got ", scales.scalar_type());
  TORCH_CHECK(qweight.scalar_type() == at::ScalarType::Int,
              "qweight must be int32, got ", qweight.scalar_type());
  TORCH_CHECK(qzeros.scalar_type() == at::ScalarType::Int,
              "qzeros must be int32, got ", qzeros.scalar_type());

  // ========== Device checks ==========
  TORCH_CHECK(activation.is_cuda(), "activation must be on CUDA device");
  TORCH_CHECK(qweight.is_cuda(), "qweight must be on CUDA device");
  TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA device");
  TORCH_CHECK(qzeros.is_cuda(), "qzeros must be on CUDA device");
  TORCH_CHECK(activation.device() == qweight.device(),
              "activation and qweight must be on the same device");
  TORCH_CHECK(scales.device() == qweight.device(),
              "scales and qweight must be on the same device");
  TORCH_CHECK(qzeros.device() == qweight.device(),
              "qzeros and qweight must be on the same device");

  // Create output tensor
  auto output = torch::empty({N}, scales.options());

  // Flatten activation if 2D
  auto act_flat = activation.dim() == 2 ? activation.squeeze(0) : activation;

  // Launch parameters
  constexpr int OUTPUT_PER_THREAD = 8;

  // Get current stream
  auto stream = at::hip::getCurrentHIPStream();

  // Choose split-k value
  int64_t total_outputs = (N + OUTPUT_PER_THREAD - 1) / OUTPUT_PER_THREAD;
  constexpr int THREADS_PER_SPLIT = 32;

  // Determine effective split-k: use passed value, or fall back to heuristic
  int64_t effective_splitk;
  if (split_k > 0) {
    // Explicit split-k from Python config
    effective_splitk = split_k;
  } else {
    // Fallback heuristic: find the largest divisor of num_groups (up to 20)
    // that provides good parallelism without excessive overhead.
    // Target: ~4-8 groups per split for good balance of parallelism vs.
    // reduction overhead, scaled by N (larger N needs less K-parallelism).
    int64_t target_groups_per_split;
    if (N <= 4096) {
      target_groups_per_split = 2;  // small N: maximize K-parallelism
    } else if (N <= 16384) {
      target_groups_per_split = 4;
    } else {
      target_groups_per_split = 5;  // large N: less K-parallelism needed
    }
    int64_t target_sk =
        std::max(int64_t(1), num_groups / target_groups_per_split);
    target_sk = std::min(target_sk, int64_t(20));

    // Find the largest divisor of num_groups that is <= target_sk
    effective_splitk = 1;
    for (int64_t sk = target_sk; sk >= 1; sk--) {
      if (num_groups % sk == 0) {
        effective_splitk = sk;
        break;
      }
    }
  }

  // Validate: effective_splitk must be in [1, 20] and divide num_groups.
  // If not, find the largest value <= effective_splitk that divides num_groups.
  effective_splitk = std::min(effective_splitk, int64_t(20));
  if (effective_splitk < 1 || num_groups % effective_splitk != 0) {
    int64_t orig = std::max(effective_splitk, int64_t(1));
    effective_splitk = 1;  // safe default
    for (int64_t sk = std::min(orig, int64_t(20)); sk >= 1; sk--) {
      if (num_groups % sk == 0) {
        effective_splitk = sk;
        break;
      }
    }
  }

  // Helper to launch the kernel with a specific SPLIT_K
  auto launch = [&]<int SPLIT_K>() {
    constexpr int THREADS_PER_BLOCK = THREADS_PER_SPLIT * SPLIT_K;
    int64_t num_blocks =
        (total_outputs + THREADS_PER_SPLIT - 1) / THREADS_PER_SPLIT;
    size_t smem_size = (SPLIT_K > 1) ? SPLIT_K * THREADS_PER_SPLIT *
                                           OUTPUT_PER_THREAD * sizeof(float)
                                     : 0;

    awq_gemv_kernel_splitk<8, SPLIT_K>
        <<<num_blocks, THREADS_PER_BLOCK, smem_size, stream>>>(
            reinterpret_cast<const __half*>(act_flat.data_ptr<at::Half>()),
            reinterpret_cast<const uint32_t*>(qweight.data_ptr<int32_t>()),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<const uint32_t*>(qzeros.data_ptr<int32_t>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()), 1,
            static_cast<size_t>(K), static_cast<size_t>(N),
            static_cast<size_t>(G));
  };

  // Dispatch to the appropriate template instantiation (1-20)
  switch (effective_splitk) {
    case 20:
      launch.template operator()<20>();
      break;
    case 19:
      launch.template operator()<19>();
      break;
    case 18:
      launch.template operator()<18>();
      break;
    case 17:
      launch.template operator()<17>();
      break;
    case 16:
      launch.template operator()<16>();
      break;
    case 15:
      launch.template operator()<15>();
      break;
    case 14:
      launch.template operator()<14>();
      break;
    case 13:
      launch.template operator()<13>();
      break;
    case 12:
      launch.template operator()<12>();
      break;
    case 11:
      launch.template operator()<11>();
      break;
    case 10:
      launch.template operator()<10>();
      break;
    case 9:
      launch.template operator()<9>();
      break;
    case 8:
      launch.template operator()<8>();
      break;
    case 7:
      launch.template operator()<7>();
      break;
    case 6:
      launch.template operator()<6>();
      break;
    case 5:
      launch.template operator()<5>();
      break;
    case 4:
      launch.template operator()<4>();
      break;
    case 3:
      launch.template operator()<3>();
      break;
    case 2:
      launch.template operator()<2>();
      break;
    default:
      launch.template operator()<1>();
      break;
  }

  return output;
}

#endif  // USE_ROCM
