#pragma once

/**
 * __device__ layernorm + symmetric quant utilities.
 */

#include "vectorization.cuh"

namespace vllm {

template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_symmetric_quant_qparams(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    float const min_scaling_factor, int const hidden_size,
    scalar_t const* __restrict__ residual = nullptr) {
  constexpr scalar_out_t qmax{std::numeric_limits<scalar_out_t>::max()};

  int const token_offset = blockIdx.x * hidden_size;

  float block_absmax_val_maybe{0.0f};
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }

    x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(x));
  }
  block_absmax_val_maybe = blockReduceMax(block_absmax_val_maybe);

  __shared__ float s_token_scale;
  if (threadIdx.x == 0) {
    float scale = 0.0f;
    if (scale_ub) {
      scale = min(block_absmax_val_maybe, *scale_ub);
    } else {
      scale = block_absmax_val_maybe;
    }
    // token scale computation
    scale = max(scale / qmax, min_scaling_factor);
    s_token_scale = scale;                 // Shared memory store
    all_token_scales[blockIdx.x] = scale;  // Global output store
  }
  __syncthreads();

  *token_scale = s_token_scale;
}

namespace vectorized {

// Vectorized version of vllm::compute_dynamic_per_token_symmetric_quant_qparams
// hidden_size must be a multiple of 4
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_symmetric_quant_qparams(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, float const* __restrict__ scale_ub,
    float const min_scaling_factor, int const hidden_size,
    scalar_t const* __restrict__ residual = nullptr) {
  int const token_offset = blockIdx.x * hidden_size;

  // Vectorized input/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  vec4_t<scalar_t> const* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual =
        reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
  }

  constexpr scalar_out_t qmax{std::numeric_limits<scalar_out_t>::max()};
  int const tid = threadIdx.x;

  int const num_vec_elems = hidden_size >> 2;
  float block_absmax_val_maybe = 0.0f;

#pragma unroll 4
  for (int i = tid; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    vec4_t<float> x;
    x.x = static_cast<float>(in.x);
    x.y = static_cast<float>(in.y);
    x.z = static_cast<float>(in.z);
    x.w = static_cast<float>(in.w);
    if constexpr (has_residual) {
      vec4_t<scalar_t> r = vec_residual[i];
      x.x += static_cast<float>(r.x);
      x.y += static_cast<float>(r.y);
      x.z += static_cast<float>(r.z);
      x.w += static_cast<float>(r.w);
    }

    block_absmax_val_maybe = fmaxf(
        block_absmax_val_maybe, fabs(static_cast<scalar_t>(x.x * rms) * w.x));
    block_absmax_val_maybe = fmaxf(
        block_absmax_val_maybe, fabs(static_cast<scalar_t>(x.y * rms) * w.y));
    block_absmax_val_maybe = fmaxf(
        block_absmax_val_maybe, fabs(static_cast<scalar_t>(x.z * rms) * w.z));
    block_absmax_val_maybe = fmaxf(
        block_absmax_val_maybe, fabs(static_cast<scalar_t>(x.w * rms) * w.w));
  }

  block_absmax_val_maybe = blockReduceMax(block_absmax_val_maybe);

  __shared__ float s_token_scale;
  if (threadIdx.x == 0) {
    float scale = 0.0f;
    if (scale_ub) {
      scale = min(block_absmax_val_maybe, *scale_ub);
    } else {
      scale = block_absmax_val_maybe;
    }
    // token scale computation
    scale = max(scale / qmax, min_scaling_factor);
    s_token_scale = scale;                 // shared memory store
    all_token_scales[blockIdx.x] = scale;  // global output store
  }
  __syncthreads();

  *token_scale = s_token_scale;
}

}  // namespace vectorized

}  // namespace vllm
