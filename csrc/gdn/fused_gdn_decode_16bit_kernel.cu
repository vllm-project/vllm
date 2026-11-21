/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 *
 * Fused GDN MTP decode for BF16/FP16 recurrent state, used when V is not split
 * across CTAs (fused_gdn_decode_kernel.cu handles FP32 state and small
 * batches). Same semantics as that kernel, and bit-identical to its unsplit
 * (SplitV == 1) path, the one it replaces.
 *
 * The lane layout, 128-bit register-direct state access, padded q/k shared
 * layout and packed f32x2 math are adapted from FlashInfer's cake_gdn
 * BF16-state MTP decode kernel, cake_gdn_decode_pretranspose_mtp_t4_
 * bf16state_wide128 in csrc/gdn/cake/ (FlashInfer v0.7.3), added in
 * https://github.com/flashinfer-ai/flashinfer/pull/4581.
 * Copyright (c) 2026 by FlashInfer team, licensed under Apache-2.0.
 * The reduction is our own: it keeps the summation order of
 * fused_gdn_decode_kernel.cu, so results stay bit-identical to it.
 */

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "fused_gdn_decode.h"

namespace {

constexpr int kDim = 128;  // K == V
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kTokenPasses = (kMaxMtpTokens + kWarps - 1) / kWarps;
// q/k are stored with a 12-float stride per 8-value block, so the two 16-byte
// reads per lane don't bank-conflict. Within a block the order is
// {K0,K4,K1,K5}, {K2,K6,K3,K7}: each float4 then holds two ready f32x2 pairs.
constexpr int kQkWidth = kDim / 8 * 12;

__device__ __forceinline__ int qk_column(int dim) {
  return (dim >> 3) * 12 + ((dim & 3) << 1) + ((dim >> 2) & 1);
}

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_fast(float x) {
  return x * sigmoid_fast(x);
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head,
                                              int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
  }
  if (dt_bias_type == kDtBiasFloat16) {
    return __half2float(static_cast<const __half*>(dt_bias)[head]);
  }
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <typename StateT>
__device__ __forceinline__ void load_state8(const StateT* state,
                                            float (&v)[8]) {
  const uint4 packed = *reinterpret_cast<const uint4*>(state);
  const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if constexpr (std::is_same_v<StateT, __half>) {
      __half2 pair;
      memcpy(&pair, &words[i], 4);
      v[2 * i] = __half2float(pair.x);
      v[2 * i + 1] = __half2float(pair.y);
    } else {
      __nv_bfloat162 pair;
      memcpy(&pair, &words[i], 4);
      v[2 * i] = __bfloat162float(pair.x);
      v[2 * i + 1] = __bfloat162float(pair.y);
    }
  }
}

template <typename StateT>
__device__ __forceinline__ void store_state8(StateT* state,
                                             const float (&v)[8]) {
  uint32_t words[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if constexpr (std::is_same_v<StateT, __half>) {
      const __half2 pair = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
      memcpy(&words[i], &pair, 4);
    } else {
      const __nv_bfloat162 pair = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
      memcpy(&words[i], &pair, 4);
    }
  }
  *reinterpret_cast<uint4*>(state) =
      make_uint4(words[0], words[1], words[2], words[3]);
}

// Packed FP32 math (FFMA2/FMUL2) exists on SM10x only; SM11x/SM12x accept it
// but lower it to scalar ops. Each half is exactly the scalar op (including
// --use_fast_math's ftz, like the original kernel), so other targets run the
// same op per half.
__device__ __forceinline__ float2 fma2(float2 a, float2 b, float2 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
  return __ffma2_rn(a, b, c);
#else
  return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
#endif
}

__device__ __forceinline__ float2 mul2(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
  return __fmul2_rn(a, b);
#else
  return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
#endif
}

// Reduce 8 partials {row r: a, b} over the 16 lanes of a group. Levels follow
// the order of the 32-lane butterfly in fused_gdn_decode_kernel.cu (xor
// 8/4/2/1 here = xor 16/8/4/2 there) and the a/b halves (xor 1 there) are
// added last, so sums are bit-identical. xor 8 keeps rows {0,1} or {2,3}, xor 4
// one row, xor 2 the a or b half. Lane ends with the total of row
// (lane >> 2) & 3: 9 shuffles.
__device__ __forceinline__ float reduce_scatter(const float (&p)[8], int lane) {
  const bool u8 = lane & 8, u4 = lane & 4, u2 = lane & 2;
  float k4[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float keep = u8 ? p[4 + i] : p[i];
    const float send = u8 ? p[i] : p[4 + i];
    k4[i] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
  }
  float k2[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const float keep = u4 ? k4[2 + i] : k4[i];
    const float send = u4 ? k4[i] : k4[2 + i];
    k2[i] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
  }
  float sum = (u2 ? k2[1] : k2[0]) +
              __shfl_xor_sync(0xffffffffu, u2 ? k2[0] : k2[1], 2);
  sum += __shfl_xor_sync(0xffffffffu, sum, 1);
  sum += __shfl_xor_sync(0xffffffffu, sum, 2);
  return sum;
}

// One CTA per (request, value head). 4 CTAs/SM (64 registers).
template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate>
__global__ __launch_bounds__(kThreads, 4) void gdn_decode_16bit_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, StateT* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate,
    const void* __restrict__ norm_weight, __nv_bfloat16* __restrict__ out,
    int H, int HV, int state_indices_width, int dt_bias_type,
    bool norm_weight_is_bf16, float scale, float norm_eps,
    GdnDecodeStrides strides) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) {
    return;
  }

  const int accepted = num_accepted_tokens[request];
  const int source_slot =
      accepted > 0 && accepted <= state_indices_width
          ? state_indices[request * state_indices_width + accepted - 1]
          : 0;
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    for (int linear = tid; linear < num_tokens * kDim; linear += kThreads) {
      const int token = bos + linear / kDim;
      const int value = linear % kDim;
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDim + value;
      out[out_offset] = __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / ValueHeadsPerKeyHead;
  // Read as float4 below.
  __shared__ __align__(16) float shared_q[kMaxMtpTokens][kQkWidth];
  __shared__ __align__(16) float shared_k[kMaxMtpTokens][kQkWidth];
  __shared__ __nv_bfloat16 shared_v[kMaxMtpTokens][kDim];
  __shared__ __nv_bfloat16 shared_out[kMaxMtpTokens][kDim];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];

  // One warp per token: l2-normalized q/k, v, decay and beta.
#pragma unroll
  for (int pass = 0, t = warp; pass < kTokenPasses && t < num_tokens;
       ++pass, t += kWarps) {
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] =
          __bfloat162float(mixed_qkv[mixed_base + key_head * kDim + dim]);
      k_values[i] = __bfloat162float(
          mixed_qkv[mixed_base + H * kDim + key_head * kDim + dim]);
      shared_v[t][dim] =
          mixed_qkv[mixed_base + 2 * H * kDim + value_head * kDim + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      q_square += __shfl_xor_sync(0xffffffffu, q_square, offset);
      k_square += __shfl_xor_sync(0xffffffffu, k_square, offset);
    }
    const float q_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(q_square + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(k_square + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][qk_column(dim)] = q_values[i] * q_scale;
      shared_k[t][qk_column(dim)] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(
          a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(
          b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head,
                                                           dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
    }
  }
  __syncthreads();

  // Recurrence: 16 groups of 16 lanes, each lane owning 8 contiguous K values
  // of 4 rows, so state moves as 128-bit LDG/STG straight between global
  // memory and registers. Group g owns rows g*8 .. g*8+7, two passes of 4.
  // Each lane keeps the partial of each 4-value K half apart (lanes 2*sub and
  // 2*sub+1 in fused_gdn_decode_kernel.cu) and adds them last.
  const int64_t head_offset = static_cast<int64_t>(value_head) * kDim * kDim;
  const StateT* source_state =
      state + static_cast<int64_t>(source_slot) * strides.state_slot +
      head_offset;
  const int group = tid >> 4;
  const int sub = lane & 15;
  const int k_start = sub * 8;
  const int qk_col = sub * 12;
  // After reduce_scatter, a lane holds the total of row (lane >> 2) & 3.
  const int lane_row = (lane >> 2) & 3;
#pragma unroll 1
  for (int pass = 0; pass < 2; ++pass) {
    const int row_base = group * 8 + pass * 4;
    // h[r][i] = {K i, K 4+i} of row r: .x runs the a half, .y the b half.
    float2 h[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      float v[8];
      load_state8(source_state + (row_base + r) * kDim + k_start, v);
#pragma unroll
      for (int i = 0; i < 4; ++i) h[r][i] = make_float2(v[i], v[4 + i]);
    }
    const int lane_value = row_base + lane_row;
    for (int t = 0; t < num_tokens; ++t) {
      float2 kv[4], qv[4];
      {
        const float4 k0 =
            *reinterpret_cast<const float4*>(&shared_k[t][qk_col]);
        const float4 k1 =
            *reinterpret_cast<const float4*>(&shared_k[t][qk_col + 4]);
        kv[0] = make_float2(k0.x, k0.y);
        kv[1] = make_float2(k0.z, k0.w);
        kv[2] = make_float2(k1.x, k1.y);
        kv[3] = make_float2(k1.z, k1.w);
      }
      const float decay = shared_decay[t];
      const float2 decay2 = make_float2(decay, decay);
      float part[8];  // {row0 a, row0 b, row1 a, ...}
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        float2 dot = make_float2(0.0f, 0.0f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[r][i] = mul2(h[r][i], decay2);
          dot = fma2(h[r][i], kv[i], dot);
        }
        part[r * 2] = dot.x;
        part[r * 2 + 1] = dot.y;
      }
      const float lane_delta = (__bfloat162float(shared_v[t][lane_value]) -
                                reduce_scatter(part, lane)) *
                               shared_beta[t];
      {
        const float4 q0 =
            *reinterpret_cast<const float4*>(&shared_q[t][qk_col]);
        const float4 q1 =
            *reinterpret_cast<const float4*>(&shared_q[t][qk_col + 4]);
        qv[0] = make_float2(q0.x, q0.y);
        qv[1] = make_float2(q0.z, q0.w);
        qv[2] = make_float2(q1.x, q1.y);
        qv[3] = make_float2(q1.z, q1.w);
      }
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float d =
            __shfl_sync(0xffffffffu, lane_delta, (lane & 16) + r * 4);
        const float2 delta2 = make_float2(d, d);
        float2 dot = make_float2(0.0f, 0.0f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[r][i] = fma2(kv[i], delta2, h[r][i]);
          dot = fma2(h[r][i], qv[i], dot);
        }
        part[r * 2] = dot.x;
        part[r * 2 + 1] = dot.y;
      }
      const float reduced_hq = reduce_scatter(part, lane);
      if ((lane & 3) == 0) {
        shared_out[t][lane_value] = __float2bfloat16(reduced_hq);
      }
      const int destination_slot =
          state_indices[request * state_indices_width + t];
      if (destination_slot > 0) {
        StateT* destination_state =
            state +
            static_cast<int64_t>(destination_slot) * strides.state_slot +
            head_offset;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          float v[8];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            v[i] = h[r][i].x;
            v[4 + i] = h[r][i].y;
          }
          store_state8(destination_state + (row_base + r) * kDim + k_start, v);
        }
      }
    }
  }
  __syncthreads();

  // Gated RMSNorm over V, one warp per token.
#pragma unroll
  for (int pass = 0, t = warp; pass < kTokenPasses && t < num_tokens;
       ++pass, t += kWarps) {
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < kDim / 32; ++i) {
      const float value = __bfloat162float(shared_out[t][lane + i * 32]);
      sum_square += value * value;
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd = rsqrtf(sum_square / static_cast<float>(kDim) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < kDim / 32; ++i) {
      const int value = lane + i * 32;
      const float output_value = __bfloat162float(shared_out[t][value]);
      const float gate_input = __bfloat162float(
          output_gate[static_cast<int64_t>(token) * strides.gate_row +
                      value_head * kDim + value]);
      const float gate =
          SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight =
          norm_weight_is_bf16
              ? __bfloat162float(
                    static_cast<const __nv_bfloat16*>(norm_weight)[value])
              : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDim + value;
      out[out_offset] = __float2bfloat16(output_value * rstd * weight * gate);
    }
  }
}

}  // namespace

template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate>
cudaError_t launch_gdn_decode_post_conv_mtp_16bit(
    dim3 grid, cudaStream_t stream, const __nv_bfloat16* mixed_qkv,
    const __nv_bfloat16* a, const __nv_bfloat16* b, const float* a_log,
    const void* dt_bias, const int* state_indices, const int* cu_seqlens,
    const int* num_accepted_tokens, StateT* state,
    const __nv_bfloat16* output_gate, const void* norm_weight,
    __nv_bfloat16* out, int H, int HV, int state_indices_width,
    int dt_bias_type, bool norm_weight_is_bf16, float scale, float norm_eps,
    GdnDecodeStrides strides) {
  gdn_decode_16bit_kernel<StateT, ValueHeadsPerKeyHead, SigmoidGate>
      <<<grid, kThreads, 0, stream>>>(
          mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens,
          num_accepted_tokens, state, output_gate, norm_weight, out, H, HV,
          state_indices_width, dt_bias_type, norm_weight_is_bf16, scale,
          norm_eps, strides);
  return cudaGetLastError();
}

#define INSTANTIATE_GDN_DECODE_16BIT(StateT, Ratio)                            \
  template cudaError_t                                                         \
  launch_gdn_decode_post_conv_mtp_16bit<StateT, Ratio, true>(                  \
      dim3, cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*,          \
      const __nv_bfloat16*, const float*, const void*, const int*, const int*, \
      const int*, StateT*, const __nv_bfloat16*, const void*, __nv_bfloat16*,  \
      int, int, int, int, bool, float, float, GdnDecodeStrides);               \
  template cudaError_t                                                         \
  launch_gdn_decode_post_conv_mtp_16bit<StateT, Ratio, false>(                 \
      dim3, cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*,          \
      const __nv_bfloat16*, const float*, const void*, const int*, const int*, \
      const int*, StateT*, const __nv_bfloat16*, const void*, __nv_bfloat16*,  \
      int, int, int, int, bool, float, float, GdnDecodeStrides);

// HV / H in {1, 2, 3, 4, 8}, matching the dispatch in
// fused_gdn_decode_kernel.cu.
#define INSTANTIATE_GDN_DECODE_16BIT_RATIOS(StateT) \
  INSTANTIATE_GDN_DECODE_16BIT(StateT, 1)           \
  INSTANTIATE_GDN_DECODE_16BIT(StateT, 2)           \
  INSTANTIATE_GDN_DECODE_16BIT(StateT, 3)           \
  INSTANTIATE_GDN_DECODE_16BIT(StateT, 4)           \
  INSTANTIATE_GDN_DECODE_16BIT(StateT, 8)

INSTANTIATE_GDN_DECODE_16BIT_RATIOS(__nv_bfloat16)
INSTANTIATE_GDN_DECODE_16BIT_RATIOS(__half)
