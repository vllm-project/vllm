#pragma once

/**
 * __device__ layernorm + asymmetric int8 quant utilities
 */

#include "vectorization.cuh"

namespace vllm {

template <typename scalar_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_asymmetric_int8_quant_qparams(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    int32_t* __restrict__ token_azp, int32_t* __restrict__ all_token_azps,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, int const hidden_size,
    scalar_t const* __restrict__ residual = nullptr) {
  constexpr float flt_max{std::numeric_limits<float>::max()};
  constexpr float flt_min{std::numeric_limits<float>::lowest()};

  int const token_offset = blockIdx.x * hidden_size;

  float block_max_maybe{flt_min};
  float block_min_maybe{flt_max};
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }

    x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    block_max_maybe = fmaxf(block_max_maybe, x);
    block_min_maybe = fminf(block_min_maybe, x);
  }
  block_max_maybe = blockReduceMax(block_max_maybe);
  __syncthreads();
  block_min_maybe = blockReduceMin(block_min_maybe);

  __shared__ float s_token_scale;
  __shared__ int32_t s_token_azp;

  if (threadIdx.x == 0) {
    float const scale = (block_max_maybe - block_min_maybe) / 255.0f;
    ;

    // Use rounding to even (same as torch.round)
    float const azp_float = std::nearbyint(-128.0f - block_min_maybe / scale);
    int32_t const azp = static_cast<int32_t>(azp_float);

    all_token_scales[blockIdx.x] = s_token_scale = scale;
    all_token_azps[blockIdx.x] = s_token_azp = azp;
  }
  __syncthreads();

  *token_scale = s_token_scale;
  *token_azp = s_token_azp;
}

namespace vectorized {

// Vectorized version of
// vllm::compute_dynamic_per_token_asymmetric_int8_quant_qparams hidden_size
// must be a multiple of 4
template <typename scalar_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_asymmetric_int8_quant_qparams(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    int32_t* __restrict__ token_azp, int32_t* __restrict__ all_token_azps,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    float const rms, int const hidden_size,
    scalar_t const* __restrict__ residual = nullptr) {
  constexpr float flt_max{std::numeric_limits<float>::max()};
  constexpr float flt_min{std::numeric_limits<float>::lowest()};

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

  float block_max_maybe{flt_min};
  float block_min_maybe{flt_max};

  int const tid = threadIdx.x;
  int const num_vec_elems = hidden_size >> 2;

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

    scalar_t normed_x = static_cast<scalar_t>(x.x * rms) * w.x;
    block_max_maybe = fmaxf(block_max_maybe, normed_x);
    block_min_maybe = fminf(block_min_maybe, normed_x);

    normed_x = static_cast<scalar_t>(x.y * rms) * w.y;
    block_max_maybe = fmaxf(block_max_maybe, normed_x);
    block_min_maybe = fminf(block_min_maybe, normed_x);

    normed_x = static_cast<scalar_t>(x.z * rms) * w.z;
    block_max_maybe = fmaxf(block_max_maybe, normed_x);
    block_min_maybe = fminf(block_min_maybe, normed_x);

    normed_x = static_cast<scalar_t>(x.w * rms) * w.w;
    block_max_maybe = fmaxf(block_max_maybe, normed_x);
    block_min_maybe = fminf(block_min_maybe, normed_x);
  }

  block_max_maybe = blockReduceMax(block_max_maybe);
  __syncthreads();
  block_min_maybe = blockReduceMin(block_min_maybe);

  __shared__ float s_token_scale;
  __shared__ int32_t s_token_azp;

  if (threadIdx.x == 0) {
    float const scale = (block_max_maybe - block_min_maybe) / 255.0f;
    ;

    // Use rounding to even (same as torch.round)
    float const azp_float = std::nearbyint(-128.0f - block_min_maybe / scale);
    int32_t const azp = static_cast<int32_t>(azp_float);

    all_token_scales[blockIdx.x] = s_token_scale = scale;
    all_token_azps[blockIdx.x] = s_token_azp = azp;
  }
  __syncthreads();

  *token_scale = s_token_scale;
  *token_azp = s_token_azp;
}
}  // namespace vectorized

}  // namespace vllm
