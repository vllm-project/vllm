#pragma once

#include "libtorch_stable/quantization/vectorization.cuh"
#include "quantization/utils.cuh"
#include "quant_conversions.cuh"

#include "../../../cub_helpers.h"
#include "../../../cuda_compat.h"

#ifndef USE_ROCM

namespace vllm {

constexpr int32_t kSingleReadThreads = 256;
constexpr int32_t kSingleReadVecWidth = 8;
constexpr int32_t kSingleReadMaxVPT = 4;
constexpr int32_t kSingleReadWarps = kSingleReadThreads / 32;

template <typename Op>
__device__ __forceinline__ float block_reduce_single_read(float val,
                                                          float* s_warp,
                                                          Op op) {
  int32_t const lane = threadIdx.x & 31;
  int32_t const warp = threadIdx.x >> 5;
  #pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1) {
    val = op(val, VLLM_SHFL_XOR_SYNC(val, offset));
  }
  if (lane == 0) {
    s_warp[warp] = val;
  }
  __syncthreads();
  float total = s_warp[0];
  #pragma unroll
  for (int32_t i = 1; i < kSingleReadWarps; ++i) {
    total = op(total, s_warp[i]);
  }
  return total;
}

template <typename scalar_t, typename scalar_out_t, int32_t VPT>
__global__ void __launch_bounds__(kSingleReadThreads)
    rms_norm_dynamic_per_token_quant_single_read_kernel(
        scalar_out_t* __restrict__ out,       // [num_tokens, hidden_size]
        float* __restrict__ scales,           // [num_tokens]
        scalar_t const* __restrict__ input,   // [num_tokens, hidden_size]
        scalar_t const* __restrict__ weight,  // [hidden_size]
        float const* scale_ub, float const var_epsilon,
        int32_t const input_stride) {
  using vec8_t = vec_n_t<scalar_t, kSingleReadVecWidth>;
  using q8x8_t = q8_n_t<scalar_out_t, kSingleReadVecWidth>;
  constexpr int32_t kElems = VPT * kSingleReadVecWidth;
  constexpr int32_t kHidden = kSingleReadThreads * kElems;

  int64_t const token = blockIdx.x;
  int32_t const tid = threadIdx.x;

  vec8_t const* vec_input =
      reinterpret_cast<vec8_t const*>(input + token * input_stride);
  vec8_t const* vec_weight = reinterpret_cast<vec8_t const*>(weight);

  // Single global read of the row; x kept widened, weight kept narrow.
  float x[kElems];
  scalar_t w[kElems];
  float ss = 0.0f;
  #pragma unroll
  for (int32_t v = 0; v < VPT; ++v) {
    vec8_t const in = vec_input[v * kSingleReadThreads + tid];
    vec8_t const wt = vec_weight[v * kSingleReadThreads + tid];
  #pragma unroll
    for (int32_t j = 0; j < kSingleReadVecWidth; ++j) {
      float const xf = static_cast<float>(in.val[j]);
      x[v * kSingleReadVecWidth + j] = xf;
      w[v * kSingleReadVecWidth + j] = wt.val[j];
      ss += xf * xf;
    }
  }

  __shared__ float s_warp_sum[kSingleReadWarps];
  __shared__ float s_warp_max[kSingleReadWarps];

  ss = block_reduce_single_read(ss, s_warp_sum, CubAddOp{});
  float const rms = rsqrtf(ss / kHidden + var_epsilon);

  float p[kElems];
  float block_absmax_val_maybe = 0.0f;
  #pragma unroll
  for (int32_t i = 0; i < kElems; ++i) {
    p[i] = static_cast<scalar_t>(x[i] * rms) * w[i];
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(p[i]));
  }

  block_absmax_val_maybe =
      block_reduce_single_read(block_absmax_val_maybe, s_warp_max, CubMaxOp{});

  constexpr scalar_out_t qmax{quant_type_max_v<scalar_out_t>};
  float scale = scale_ub ? fminf(block_absmax_val_maybe, *scale_ub)
                         : block_absmax_val_maybe;
  scale = fmaxf(scale / qmax, min_scaling_factor<scalar_out_t>::val());
  if (tid == 0) {
    scales[token] = scale;
  }

  q8x8_t* vec_out = reinterpret_cast<q8x8_t*>(out + token * kHidden);
  #pragma unroll
  for (int32_t v = 0; v < VPT; ++v) {
    q8x8_t q;
  #pragma unroll
    for (int32_t j = 0; j < kSingleReadVecWidth; ++j) {
      q.val[j] = ScaledQuant<scalar_out_t, false>::quant_fn(
          p[v * kSingleReadVecWidth + j], scale);
    }
    vec_out[v * kSingleReadThreads + tid] = q;
  }
}

}  // namespace vllm

#endif  // !USE_ROCM
