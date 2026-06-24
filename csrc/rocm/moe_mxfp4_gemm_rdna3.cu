// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MXFP4 fused MoE GEMM for RDNA3 (gfx1100/1101/1102). Fork of
// moe_q_gemm_rdna3.cu: expert routing (sorted_token_ids / expert_ids) + the
// MXFP4 scalar dequant+dot, in one kernel. No zero-point; E8M0 block scale is
// [E, K/32, N] uint8. Per-expert weights are [E, K/8, N] uint32 (E2M1).

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "qdq_mxfp4_rdna3.cuh"

#if defined(__HIPCC__) && defined(__gfx1100__)
  #define __HIP__RDNA3__
#endif

#define MOE_MXFP4_BLOCK_KN 256
#define MOE_MXFP4_THREADS 256

namespace vllm {
namespace moe_mxfp4_rdna3 {

using bf16_t = __hip_bfloat16;
using bf162_t = __hip_bfloat162;
using mxfp4_rdna3::dequant_mxfp4_8_f32;
using mxfp4_rdna3::mxfp4_e8m0_bias;

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

template <typename T>
__device__ __forceinline__ T tzero();
template <>
__device__ __forceinline__ half tzero<half>() {
  return __float2half_rn(0.0f);
}
template <>
__device__ __forceinline__ bf16_t tzero<bf16_t>() {
  return __float2bfloat16(0.0f);
}

__device__ __forceinline__ float mxfp4_to_f(half v) { return __half2float(v); }
__device__ __forceinline__ float mxfp4_to_f(bf16_t v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ void atomic_add_pk4(half* addr, half2 v01,
                                               half2 v23) {
  unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *a;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(a, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}
__device__ __forceinline__ void atomic_add_pk4(bf16_t* addr, bf162_t v01,
                                               bf162_t v23) {
  unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *a;
  while (true) {
    union {
      unsigned long long u;
      bf162_t b2[2];
    } cur, sum;
    cur.u = old;
    sum.b2[0] = __hadd2(cur.b2[0], v01);
    sum.b2[1] = __hadd2(cur.b2[1], v23);
    unsigned long long prev = atomicCAS(a, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_mxfp4_kernel_rdna3(
    const T* __restrict__ a, T* __restrict__ c,
    const uint32_t* __restrict__ b_q_weight,
    const uint8_t* __restrict__ b_scales_e8m0,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded, const int size_m,
    const int size_n, const int size_k, const int groups, const int top_k,
    const int expert_weight_stride, const int expert_scales_stride,
    const bool mul_topk_weight, const int output_topk) {
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * MOE_MXFP4_BLOCK_KN * 4;
  const int offset_k = blockIdx.z * MOE_MXFP4_BLOCK_KN;
  const int end_k = min(offset_k + MOE_MXFP4_BLOCK_KN, size_k);
  const int n = offset_n + t * 4;

  if (token_block * BLOCK_SIZE_M >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  const uint32_t* expert_weights =
      b_q_weight + (int64_t)expert_id * expert_weight_stride;
  const uint8_t* expert_scales =
      b_scales_e8m0 + (int64_t)expert_id * expert_scales_stride;

  constexpr int LDS_PAD = 8;
  __shared__ T block_a[BLOCK_SIZE_M][MOE_MXFP4_BLOCK_KN + LDS_PAD];
  static_assert(MOE_MXFP4_BLOCK_KN == MOE_MXFP4_THREADS,
                "BLOCK_KN must equal THREADS");
  const int offset_m_base = token_block * BLOCK_SIZE_M;

  // Stage activations into LDS via the routing table.
  if (offset_k + t < end_k) {
  #pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
      int32_t token_id = sorted_token_ids[offset_m_base + m];
      int token_row = token_id / top_k;
      block_a[m][t] = (token_row < size_m)
                          ? a[(int64_t)token_row * size_k + offset_k + t]
                          : tzero<T>();
    }
  }
  __syncthreads();
  if (n >= size_n) return;

  const int groupsize = size_k / groups;  // == 32 for MXFP4
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;
  // dequant_mxfp4_8_f32 decodes via bf16 bits (mant 7) for both dtypes.
  constexpr uint32_t mant_bits = 7u;

  int qk = offset_k / 8;
  const uint32_t* b_ptr = expert_weights + qk * size_n + n;

  int32_t bias[4];
  auto refresh = [&](int g) {
    const uint8_t* sc = expert_scales + (int64_t)g * size_n;
  #pragma unroll
    for (int col = 0; col < 4; ++col)
      bias[col] = mxfp4_e8m0_bias(sc[n + col], mant_bits);
  };
  refresh(group);

  float block_c[BLOCK_SIZE_M][4];
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m)
  #pragma unroll
    for (int col = 0; col < 4; ++col) block_c[m][col] = 0.0f;

  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh(group);
    }
    int4 b_w[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) b_w[j] = *(const int4*)(b_ptr + j * size_n);
    b_ptr += 4 * size_n;

  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;
      uint32_t wcol[4];
      __builtin_memcpy(wcol, &b_w[j], sizeof(int4));
  #pragma unroll
      for (int col = 0; col < 4; ++col) {
        float dq[8];
        dequant_mxfp4_8_f32(wcol[col], bias[col], dq);
  #pragma unroll
        for (int m = 0; m < BLOCK_SIZE_M; ++m) {
          const T* ap = &block_a[m][a_off];
          float s = 0.0f;
  #pragma unroll
          for (int i = 0; i < 8; ++i) s += dq[i] * mxfp4_to_f(ap[i]);
          block_c[m][col] += s;
        }
      }
    }
    k += 32;
  }

  // Epilogue: topk weight + atomic-add (with optional output_topk reduction).
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    int32_t token_id = sorted_token_ids[offset_m_base + m];
    if (token_id / top_k >= size_m) continue;
    if (mul_topk_weight && topk_weights != nullptr) {
      float tw = topk_weights[token_id];
  #pragma unroll
      for (int j = 0; j < 4; ++j) block_c[m][j] *= tw;
    }
    int64_t out_row = (output_topk > 0) ? (int64_t)(token_id / output_topk)
                                        : (int64_t)token_id;
    T* out = c + out_row * size_n + n;
    if constexpr (std::is_same<T, half>::value) {
      half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                 __float2half_rn(block_c[m][1]));
      half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                 __float2half_rn(block_c[m][3]));
      atomic_add_pk4(out, r01, r23);
    } else {
      bf162_t r01, r23;
      r01.x = __float2bfloat16(block_c[m][0]);
      r01.y = __float2bfloat16(block_c[m][1]);
      r23.x = __float2bfloat16(block_c[m][2]);
      r23.y = __float2bfloat16(block_c[m][3]);
      atomic_add_pk4(out, r01, r23);
    }
  }
}

#else  // non-RDNA3: empty stub for symbol parity
template <typename T, int BLOCK_SIZE_M>
__global__ void moe_gemm_mxfp4_kernel_rdna3(
    const T*, T*, const uint32_t*, const uint8_t*, const float*,
    const int32_t*, const int32_t*, const int32_t*, const int, const int,
    const int, const int, const int, const int, const int, const bool,
    const int) {}
#endif

template <typename T, int BLOCK_SIZE_M>
void launch_moe_gemm_mxfp4(const T* a, T* c, const uint32_t* b_q_weight,
                           const uint8_t* b_scales_e8m0,
                           const float* topk_weights,
                           const int32_t* sorted_token_ids,
                           const int32_t* expert_ids,
                           const int32_t* num_tokens_post_padded,
                           int num_token_blocks, int size_m, int size_n,
                           int size_k, int groups, int top_k,
                           int expert_weight_stride, int expert_scales_stride,
                           bool mul_topk_weight, int output_topk,
                           cudaStream_t stream) {
  dim3 block(MOE_MXFP4_THREADS);
  dim3 grid(num_token_blocks,
            (size_n + MOE_MXFP4_BLOCK_KN * 4 - 1) / (MOE_MXFP4_BLOCK_KN * 4),
            (size_k + MOE_MXFP4_BLOCK_KN - 1) / MOE_MXFP4_BLOCK_KN);
  moe_gemm_mxfp4_kernel_rdna3<T, BLOCK_SIZE_M><<<grid, block, 0, stream>>>(
      a, c, b_q_weight, b_scales_e8m0, topk_weights, sorted_token_ids,
      expert_ids, num_tokens_post_padded, size_m, size_n, size_k, groups, top_k,
      expert_weight_stride, expert_scales_stride, mul_topk_weight, output_topk);
}

template <typename T>
void dispatch_moe_gemm_mxfp4(
    const T* a, T* c, const uint32_t* b_q_weight, const uint8_t* b_scales_e8m0,
    const float* topk_weights, const int32_t* sorted_token_ids,
    const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
    int num_token_blocks, int size_m, int size_n, int size_k, int groups,
    int top_k, int block_size_m, int expert_weight_stride,
    int expert_scales_stride, bool mul_topk_weight, int output_topk,
    cudaStream_t stream) {
  auto L = [&](auto bsm) {
    launch_moe_gemm_mxfp4<T, decltype(bsm)::value>(
        a, c, b_q_weight, b_scales_e8m0, topk_weights, sorted_token_ids,
        expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
        size_k, groups, top_k, expert_weight_stride, expert_scales_stride,
        mul_topk_weight, output_topk, stream);
  };
  switch (block_size_m) {
    case 1:
      L(std::integral_constant<int, 1>{});
      break;
    case 2:
      L(std::integral_constant<int, 2>{});
      break;
    case 4:
      L(std::integral_constant<int, 4>{});
      break;
    case 8:
      L(std::integral_constant<int, 8>{});
      break;
    default:
      TORCH_CHECK(false, "moe_mxfp4_gemm_rdna3: block_size_m must be 1/2/4/8");
  }
}

}  // namespace moe_mxfp4_rdna3
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
//   a                      [M, K] or [M*top_k, K]  half/bf16
//   c                      [M*top_k, N] or reduced  same dtype (pre-zeroed!)
//   b_q_weight             [E, K/8, N]              uint32 (E2M1)
//   b_scales_e8m0          [E, K/32, N]             uint8  (E8M0)
//   topk_weights           [M*top_k] or empty       float32
//   sorted_token_ids       [num_blocks * block_m]   int32
//   expert_ids             [num_blocks]             int32
//   num_tokens_post_padded [1]                      int32
// ---------------------------------------------------------------------------
void moe_mxfp4_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                          torch::Tensor b_q_weight, torch::Tensor b_scales_e8m0,
                          torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t top_k,
                          int64_t block_size_m, bool mul_topk_weight,
                          int64_t output_topk) {
  TORCH_CHECK(a.is_cuda() && c.is_cuda() && b_q_weight.is_cuda(),
              "tensors must be CUDA/HIP");
  TORCH_CHECK(a.dim() == 2 && c.dim() == 2, "a and c must be 2D");
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be [E, K/8, N]");
  TORCH_CHECK(b_scales_e8m0.dim() == 3, "b_scales_e8m0 must be [E, K/32, N]");
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf || a.scalar_type() == torch::kBFloat16,
      "a must be half or bfloat16");
  TORCH_CHECK(b_scales_e8m0.scalar_type() == torch::kUInt8,
              "b_scales_e8m0 must be uint8 (E8M0)");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(2);
  int groups = (int)b_scales_e8m0.size(1);  // K/32
  int expert_weight_stride = (int)(b_q_weight.size(1) * b_q_weight.size(2));
  int expert_scales_stride =
      (int)(b_scales_e8m0.size(1) * b_scales_e8m0.size(2));
  int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);

  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  using bf16_t = vllm::moe_mxfp4_rdna3::bf16_t;
  const uint32_t* bq = (const uint32_t*)b_q_weight.data_ptr<int32_t>();
  const uint8_t* bs = (const uint8_t*)b_scales_e8m0.data_ptr();
  const int32_t* sti = sorted_token_ids.data_ptr<int32_t>();
  const int32_t* eid = expert_ids.data_ptr<int32_t>();
  const int32_t* ntp = num_tokens_post_padded.data_ptr<int32_t>();

  if (a.scalar_type() == torch::kHalf) {
    vllm::moe_mxfp4_rdna3::dispatch_moe_gemm_mxfp4<half>(
        (const half*)a.data_ptr(), (half*)c.data_ptr(), bq, bs, topk_w_ptr, sti,
        eid, ntp, num_token_blocks, size_m, size_n, size_k, groups, (int)top_k,
        (int)block_size_m, expert_weight_stride, expert_scales_stride,
        mul_topk_weight, (int)output_topk, stream);
  } else {
    vllm::moe_mxfp4_rdna3::dispatch_moe_gemm_mxfp4<bf16_t>(
        (const bf16_t*)a.data_ptr(), (bf16_t*)c.data_ptr(), bq, bs, topk_w_ptr,
        sti, eid, ntp, num_token_blocks, size_m, size_n, size_k, groups,
        (int)top_k, (int)block_size_m, expert_weight_stride,
        expert_scales_stride, mul_topk_weight, (int)output_topk, stream);
  }
}
