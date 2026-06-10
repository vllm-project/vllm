// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused MoE W4A16 GPTQ kernel for RDNA3 (gfx1100).
//
// Combines expert routing (sorted_token_ids / expert_ids) with the RDNA3
// W4A16 dequant+dot from q_gemm_rdna3.cu into a single kernel launch.
// Each block processes BLOCK_SIZE_M tokens assigned to one expert, covering
// a tile of N output columns and K input positions.
//
// Weight format: same as the dense kernel — [E, K/8, N] uint32 shuffled,
// [E, groups, N] scales, [E, groups, N/8] packed zeros.
//
// Design: THREADS_X=256 (8 waves on wave32), BLOCK_KN_SIZE=256, each thread
// handles 4 N columns. Output via 64-bit packed CAS atomic-add directly to
// the pre-zeroed output tensor (no FP32 scratch buffer).

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "qdq_4_rdna3.cuh"

#if defined(__HIPCC__) && defined(__gfx1100__)
  #define __HIP__RDNA3__
#endif

namespace vllm {
namespace moe_gptq_rdna3 {

#define BLOCK_KN_SIZE 256
#define THREADS_X 256

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

using gptq_rdna3::bf162_t;
using gptq_rdna3::bf16_t;

// --- Helpers (same as q_gemm_rdna3.cu) ---

template <typename T>
__forceinline__ __device__ T tzero();

template <>
__forceinline__ __device__ half tzero<half>() {
  return __float2half_rn(0.0f);
}

template <>
__forceinline__ __device__ bf16_t tzero<bf16_t>() {
  return __float2bfloat16(0.0f);
}

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr) {
  float result = 0.0f;
  const half2* a2_ptr = (const half2*)a_ptr;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    result = __builtin_amdgcn_fdot2(dq[i], *a2_ptr++, result, /*clamp=*/false);
  }
  return result;
}

__forceinline__ __device__ float dot22_8_f(float (&dq)[8],
                                           const bf16_t* a_ptr) {
  float result = 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t aw;
    __builtin_memcpy(&aw, a_ptr + 2 * i, sizeof(uint32_t));
    float a_x = __uint_as_float((aw & 0xFFFFu) << 16);
    float a_y = __uint_as_float(aw & 0xFFFF0000u);
    result = __fmaf_rn(dq[2 * i + 0], a_x, result);
    result = __fmaf_rn(dq[2 * i + 1], a_y, result);
  }
  return result;
}

__forceinline__ __device__ void atomic_add_pk4_f16(half* addr, half2 v01,
                                                   half2 v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

__forceinline__ __device__ void atomic_add_pk4_bf16(bf16_t* addr, bf162_t v01,
                                                    bf162_t v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      bf162_t b2[2];
    } cur, sum;
    cur.u = old;
    sum.b2[0] = __hadd2(cur.b2[0], v01);
    sum.b2[1] = __hadd2(cur.b2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

__forceinline__ __device__ void load4_zeros(const uint32_t* qzeros_row, int n,
                                            int (&zeros)[4]) {
  int qcol = n / 8;
  int shift = (n & 0x07) * 4;
  uint32_t d = qzeros_row[qcol] >> shift;
  zeros[0] = (int)(d & 0xF);
  zeros[1] = (int)((d >> 4) & 0xF);
  zeros[2] = (int)((d >> 8) & 0xF);
  zeros[3] = (int)((d >> 12) & 0xF);
}

template <typename T>
__forceinline__ __device__ void load4_scales(const T* scales_row, int n,
                                             T (&scales)[4]) {
  scales[0] = scales_row[n + 0];
  scales[1] = scales_row[n + 1];
  scales[2] = scales_row[n + 2];
  scales[3] = scales_row[n + 3];
}

// ---------------------------------------------------------------------------
// Fused MoE kernel.
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_q4_kernel_rdna3(
    const T* __restrict__ a,                  // [size_m, size_k] or [M*topk, K]
    T* __restrict__ c,                        // [M*topk, size_n] pre-zeroed
    const uint32_t* __restrict__ b_q_weight,  // [E, K/8, N] packed
    const T* __restrict__ b_scales,           // [E, groups, N]
    const uint32_t* __restrict__ b_qzeros,    // [E, groups, N/8] packed
    const float* __restrict__ topk_weights,   // [M*topk] or nullptr
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    const int size_m,  // total tokens (original M, or M*topk for w2)
    const int size_n,  // output features per expert
    const int size_k,  // input features
    const int groups,  // K / group_size
    const int top_k,   // routing top-k (1 for w2 pass)
    // Per-expert strides (in elements, not bytes)
    const int expert_weight_stride,  // (K/8) * N
    const int expert_scales_stride,  // groups * N
    const int expert_zeros_stride,   // groups * (N/8)
    const bool mul_topk_weight,
    const int output_topk) {  // >0: reduce output by token_id/output_topk
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * BLOCK_KN_SIZE * 4;
  const int offset_k = blockIdx.z * BLOCK_KN_SIZE;
  const int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  const int n = offset_n + t * 4;

  // Early exit for padding blocks or invalid experts (expert_map = -1)
  if (token_block * BLOCK_SIZE_M >= num_tokens_post_padded[0]) return;

  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  // Expert-specific pointers
  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * expert_weight_stride;
  const T* expert_scales = b_scales + (int64_t)expert_id * expert_scales_stride;
  const uint32_t* expert_qzeros =
      b_qzeros + (int64_t)expert_id * expert_zeros_stride;

  // LDS for activations
  constexpr int LDS_PAD = 8;
  __shared__ T block_a[BLOCK_SIZE_M][BLOCK_KN_SIZE + LDS_PAD];

  static_assert(BLOCK_KN_SIZE == THREADS_X,
                "BLOCK_KN_SIZE must equal THREADS_X");

  // For bf16 M=1, we can skip LDS and read A from global (same as dense).
  // fp16 always needs LDS due to the dot22_8_f indexing pattern.
  constexpr bool USE_LDS_A = (BLOCK_SIZE_M > 1) || std::is_same<T, half>::value;

  const int offset_m_base = token_block * BLOCK_SIZE_M;

  if constexpr (USE_LDS_A) {
    if (offset_k + t < end_k) {
  #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        int32_t token_id = sorted_token_ids[offset_m_base + m];
        int token_row = token_id / top_k;
        T av;
        if (token_row < size_m) {
          av = a[(int64_t)token_row * size_k + offset_k + t];
        } else {
          av = tzero<T>();
        }
        block_a[m][t] = av;
      }
    }
    __syncthreads();
  }

  if (n >= size_n) return;

  // Group bookkeeping
  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  // Weight pointer for this expert
  int qk = offset_k / 8;
  const uint32_t* b_ptr = expert_weights + qk * size_n + n;

  // Per-column dequant constants (4 columns per thread)
  half2 z1z16_h[4][2], y1y16_h[4][2];
  float z_b_f[4], y_b_f[4];

  // GPTQv1: zero_offset = 1
  constexpr int zero_offset = 1;

  auto refresh_group = [&](int g) {
    const uint32_t* qz_row = expert_qzeros + g * (size_n / 8);
    const T* sc_row = expert_scales + g * size_n;
    int zeros[4];
    T scales[4];
    load4_zeros(qz_row, n, zeros);
    load4_scales<T>(sc_row, n, scales);
    if constexpr (std::is_same<T, half>::value) {
  #pragma unroll
      for (int i = 0; i < 4; ++i) {
        gptq_rdna3::prep_zero_scale_fp16((uint32_t)(zeros[i] + zero_offset),
                                         scales[i], z1z16_h[i], y1y16_h[i]);
      }
    } else {
  #pragma unroll
      for (int i = 0; i < 4; ++i) {
        gptq_rdna3::prep_zero_scale_bf16_f32((uint32_t)(zeros[i] + zero_offset),
                                             scales[i], z_b_f[i], y_b_f[i]);
      }
    }
  };

  refresh_group(group);

  float block_c[BLOCK_SIZE_M][4];
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
  #pragma unroll
    for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
  }

  // --- Main K-loop ---
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh_group(group);
    }

    // Prefetch 4 weight words (128 bytes)
    int4 b_w[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      b_w[j] = *(const int4*)(b_ptr + j * size_n);
    }
    b_ptr += 4 * size_n;

  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;

      if constexpr (std::is_same<T, half>::value) {
        // fp16 path: dequant via bit-trick, dot via v_dot2_f32_f16
        half2 dq[4][4];
        gptq_rdna3::dequant_4bit_8_fp16((uint32_t)b_w[j].x, dq[0], z1z16_h[0],
                                        y1y16_h[0]);
        gptq_rdna3::dequant_4bit_8_fp16((uint32_t)b_w[j].y, dq[1], z1z16_h[1],
                                        y1y16_h[1]);
        gptq_rdna3::dequant_4bit_8_fp16((uint32_t)b_w[j].z, dq[2], z1z16_h[2],
                                        y1y16_h[2]);
        gptq_rdna3::dequant_4bit_8_fp16((uint32_t)b_w[j].w, dq[3], z1z16_h[3],
                                        y1y16_h[3]);

  #pragma unroll
        for (int m = 0; m < BLOCK_SIZE_M; ++m) {
          const half* a_ptr = reinterpret_cast<const half*>(&block_a[m][a_off]);
          block_c[m][0] += dot22_8_f(dq[0], a_ptr);
          block_c[m][1] += dot22_8_f(dq[1], a_ptr);
          block_c[m][2] += dot22_8_f(dq[2], a_ptr);
          block_c[m][3] += dot22_8_f(dq[3], a_ptr);
        }
      } else if constexpr (BLOCK_SIZE_M == 1) {
        // bf16 M=1: v_dot2_f32_bf16 with InstCombine-defeating opacity
        typedef short __attribute__((ext_vector_type(2))) bf16x2_t;
        constexpr uint32_t BF16_MAGIC = 0x43004300u;
        constexpr uint32_t BF16_ONES = 0x3F803F80u;
        union pack4 {
          float f[4];
          uint32_t u[4];
        };

        uint32_t w[4];
        __builtin_memcpy(w, &b_w[j], sizeof(int4));

        // Load activations — read from global (no LDS for bf16 M=1)
        pack4 a_pack;
        {
          int32_t token_id = sorted_token_ids[offset_m_base];
          int token_row = token_id / top_k;
          if (token_row < size_m) {
            const uint32_t* a_words = reinterpret_cast<const uint32_t*>(
                a + (int64_t)token_row * size_k + offset_k + a_off);
            a_pack.u[0] = a_words[0];
            a_pack.u[1] = a_words[1];
            a_pack.u[2] = a_words[2];
            a_pack.u[3] = a_words[3];
          } else {
            a_pack.u[0] = 0;
            a_pack.u[1] = 0;
            a_pack.u[2] = 0;
            a_pack.u[3] = 0;
          }
        }

        // sum_a for bias correction
        float sum_a = 0.0f;
  #pragma unroll
        for (int b = 0; b < 4; ++b) {
          sum_a = __builtin_amdgcn_fdot2_f32_bf16(
              *((bf16x2_t*)(&a_pack.f[b])), *((const bf16x2_t*)&BF16_ONES),
              sum_a, /*clamp=*/false);
        }

  #pragma unroll 1
        for (int col = 0; col < 4; ++col) {
          pack4 q_pack;
          const uint32_t qa = w[col];
          q_pack.u[0] = ((qa >> 0) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[1] = ((qa >> 4) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[2] = ((qa >> 8) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[3] = ((qa >> 12) & 0x000F000Fu) | BF16_MAGIC;

          float partial = 0.0f;
  #pragma unroll
          for (int b = 0; b < 4; ++b) {
            partial = __builtin_amdgcn_fdot2_f32_bf16(
                *((bf16x2_t*)(&a_pack.f[b])), *((bf16x2_t*)(&q_pack.f[b])),
                partial, /*clamp=*/false);
          }

          block_c[0][col] =
              __fmaf_rn(y_b_f[col], partial,
                        __fmaf_rn(z_b_f[col], sum_a, block_c[0][col]));
        }
      } else {
        // bf16 M>1: v_dot2_f32_bf16 with LDS-staged activations
        typedef short __attribute__((ext_vector_type(2))) bf16x2_t;
        constexpr uint32_t BF16_MAGIC = 0x43004300u;
        constexpr uint32_t BF16_ONES = 0x3F803F80u;
        union pack4 {
          float f[4];
          uint32_t u[4];
        };

        uint32_t w[4];
        __builtin_memcpy(w, &b_w[j], sizeof(int4));

        pack4 a_pack[BLOCK_SIZE_M];
  #pragma unroll
        for (int m = 0; m < BLOCK_SIZE_M; ++m) {
          const uint32_t* a_words =
              reinterpret_cast<const uint32_t*>(&block_a[m][a_off]);
          a_pack[m].u[0] = a_words[0];
          a_pack[m].u[1] = a_words[1];
          a_pack[m].u[2] = a_words[2];
          a_pack[m].u[3] = a_words[3];
        }

        float sum_a[BLOCK_SIZE_M];
  #pragma unroll
        for (int m = 0; m < BLOCK_SIZE_M; ++m) {
          float s = 0.0f;
  #pragma unroll
          for (int b = 0; b < 4; ++b) {
            s = __builtin_amdgcn_fdot2_f32_bf16(*((bf16x2_t*)(&a_pack[m].f[b])),
                                                *((const bf16x2_t*)&BF16_ONES),
                                                s, /*clamp=*/false);
          }
          sum_a[m] = s;
        }

  #pragma unroll 1
        for (int col = 0; col < 4; ++col) {
          pack4 q_pack;
          const uint32_t qa = w[col];
          q_pack.u[0] = ((qa >> 0) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[1] = ((qa >> 4) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[2] = ((qa >> 8) & 0x000F000Fu) | BF16_MAGIC;
          q_pack.u[3] = ((qa >> 12) & 0x000F000Fu) | BF16_MAGIC;

  #pragma unroll
          for (int m = 0; m < BLOCK_SIZE_M; ++m) {
            float partial = 0.0f;
  #pragma unroll
            for (int b = 0; b < 4; ++b) {
              partial = __builtin_amdgcn_fdot2_f32_bf16(
                  *((bf16x2_t*)(&a_pack[m].f[b])), *((bf16x2_t*)(&q_pack.f[b])),
                  partial, /*clamp=*/false);
            }
            block_c[m][col] =
                __fmaf_rn(y_b_f[col], partial,
                          __fmaf_rn(z_b_f[col], sum_a[m], block_c[m][col]));
          }
        }
      }
    }
    k += 32;
  }

  // --- Epilogue: apply topk_weight and atomic-add to output ---
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    int32_t token_id = sorted_token_ids[offset_m_base + m];
    if (token_id / top_k >= size_m) continue;

    // Apply router weight
    if (mul_topk_weight && topk_weights != nullptr) {
      float tw = topk_weights[token_id];
  #pragma unroll
      for (int j = 0; j < 4; ++j) block_c[m][j] *= tw;
    }

    // output_topk > 0: reduce by mapping token_id back to original token
    // (multiple experts write to the same row via atomics)
    int64_t out_row = (output_topk > 0) ? (int64_t)(token_id / output_topk)
                                        : (int64_t)token_id;
    T* out = c + out_row * size_n + n;
    if constexpr (std::is_same<T, half>::value) {
      half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                 __float2half_rn(block_c[m][1]));
      half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                 __float2half_rn(block_c[m][3]));
      atomic_add_pk4_f16(out, r01, r23);
    } else {
      bf162_t r01;
      r01.x = __float2bfloat16(block_c[m][0]);
      r01.y = __float2bfloat16(block_c[m][1]);
      bf162_t r23;
      r23.x = __float2bfloat16(block_c[m][2]);
      r23.y = __float2bfloat16(block_c[m][3]);
      atomic_add_pk4_bf16(out, r01, r23);
    }
  }
}

#else  // non-RDNA3: empty stub for symbol parity

template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_q4_kernel_rdna3(
    const T*, T*, const uint32_t*, const T*, const uint32_t*, const float*,
    const int32_t*, const int32_t*, const int32_t*, const int, const int,
    const int, const int, const int, const int, const int, const int,
    const bool, const int) {}

#endif  // __HIP__RDNA3__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE_M>
void launch_moe_gemm_q4(
    const T* a, T* c, const uint32_t* b_q_weight, const T* b_scales,
    const uint32_t* b_qzeros, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int expert_weight_stride,
    int expert_scales_stride, int expert_zeros_stride, bool mul_topk_weight,
    int output_topk, cudaStream_t stream) {
  dim3 block(THREADS_X);
  dim3 grid(num_token_blocks,
            (size_n + BLOCK_KN_SIZE * 4 - 1) / (BLOCK_KN_SIZE * 4),
            (size_k + BLOCK_KN_SIZE - 1) / BLOCK_KN_SIZE);

  moe_gemm_q4_kernel_rdna3<T, BLOCK_SIZE_M><<<grid, block, 0, stream>>>(
      a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
      expert_ids, num_tokens_post_padded, size_m, size_n, size_k, groups, top_k,
      expert_weight_stride, expert_scales_stride, expert_zeros_stride,
      mul_topk_weight, output_topk);
}

template <typename T>
void dispatch_moe_gemm_q4(
    const T* a, T* c, const uint32_t* b_q_weight, const T* b_scales,
    const uint32_t* b_qzeros, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int block_size_m,
    int expert_weight_stride, int expert_scales_stride, int expert_zeros_stride,
    bool mul_topk_weight, int output_topk, cudaStream_t stream) {
  // Dispatch to template instantiation based on block_size_m
  switch (block_size_m) {
    case 1:
      launch_moe_gemm_q4<T, 1>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, groups, top_k, expert_weight_stride, expert_scales_stride,
          expert_zeros_stride, mul_topk_weight, output_topk, stream);
      break;
    case 2:
      launch_moe_gemm_q4<T, 2>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, groups, top_k, expert_weight_stride, expert_scales_stride,
          expert_zeros_stride, mul_topk_weight, output_topk, stream);
      break;
    case 4:
      launch_moe_gemm_q4<T, 4>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, groups, top_k, expert_weight_stride, expert_scales_stride,
          expert_zeros_stride, mul_topk_weight, output_topk, stream);
      break;
    case 8:
      launch_moe_gemm_q4<T, 8>(
          a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
          expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
          size_k, groups, top_k, expert_weight_stride, expert_scales_stride,
          expert_zeros_stride, mul_topk_weight, output_topk, stream);
      break;
    default:
      TORCH_CHECK(false,
                  "moe_gptq_gemm_rdna3: block_size_m must be 1, 2, 4, or 8, "
                  "got ",
                  block_size_m);
  }
}

}  // namespace moe_gptq_rdna3
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
//
// Inputs:
//   a                      [M, K] or [M*top_k, K]  half or bfloat16
//   c                      [M*top_k, N]             same dtype (pre-zeroed!)
//   b_q_weight             [E, K/8, N]              uint32 (shuffled)
//   b_scales               [E, groups, N]           same dtype as a
//   b_qzeros               [E, groups, N/8]         uint32 (packed 4-bit)
//   topk_weights           [M*top_k] or empty       float32
//   sorted_token_ids       [num_blocks * block_m]   int32
//   expert_ids             [num_blocks]              int32
//   num_tokens_post_padded [1]                       int32
//   top_k                  int
//   block_size_m           int (1, 2, 4, or 8)
//   mul_topk_weight        bool

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D");
  TORCH_CHECK(c.dim() == 2, "c must be 2D");
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be 3D [E, K/8, N]");
  TORCH_CHECK(b_scales.dim() == 3, "b_scales must be 3D [E, groups, N]");
  TORCH_CHECK(b_qzeros.dim() == 3, "b_qzeros must be 3D [E, groups, N/8]");
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf || a.scalar_type() == torch::kBFloat16,
      "a must be half or bfloat16");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(2);
  int groups = (int)b_scales.size(1);

  // Per-expert strides
  int expert_weight_stride = (int)(b_q_weight.size(1) * b_q_weight.size(2));
  int expert_scales_stride = (int)(b_scales.size(1) * b_scales.size(2));
  int expert_zeros_stride = (int)(b_qzeros.size(1) * b_qzeros.size(2));

  int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);

  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  // Manual dtype dispatch using HIP native types (c10::Half/BFloat16 don't
  // implicitly convert to half/__hip_bfloat16 in device code).
  using vllm::gptq_rdna3::bf16_t;

  auto dispatch = [&](auto* a_ptr, auto* c_ptr, const auto* s_ptr) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(a_ptr)>>;
    vllm::moe_gptq_rdna3::dispatch_moe_gemm_q4<T>(
        a_ptr, c_ptr, (const uint32_t*)b_q_weight.data_ptr<int32_t>(), s_ptr,
        (const uint32_t*)b_qzeros.data_ptr<int32_t>(), topk_w_ptr,
        sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks, size_m,
        size_n, size_k, groups, (int)top_k, (int)block_size_m,
        expert_weight_stride, expert_scales_stride, expert_zeros_stride,
        mul_topk_weight, (int)output_topk, stream);
  };

  if (a.scalar_type() == torch::kHalf) {
    dispatch((const half*)a.data_ptr(), (half*)c.data_ptr(),
             (const half*)b_scales.data_ptr());
  } else {
    dispatch((const bf16_t*)a.data_ptr(), (bf16_t*)c.data_ptr(),
             (const bf16_t*)b_scales.data_ptr());
  }
}
