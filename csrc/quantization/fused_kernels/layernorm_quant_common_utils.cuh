#pragma once

/**
 * __device__ layernorm quant utils
 */

#include "quant_conversions.cuh"

namespace vllm {

// has_residual must be true, if residual is not a nullptr
template <typename scalar_t, bool has_residual = false>
__device__ void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int const hidden_size, float const epsilon,
                            scalar_t const* __restrict__ residual = nullptr) {
  int const token_offset = blockIdx.x * hidden_size;
  // sum of squares
  float ss = 0.0f;

  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }

    ss += x * x;
  }
  ss = blockReduceSum<float>(ss);
  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();

  *rms = s_rms;
}

template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false, bool has_azp = false>
__device__ void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int const hidden_size,
                               scalar_t* __restrict__ residual = nullptr,
                               int32_t const azp = 0) {
  int const token_offset = blockIdx.x * hidden_size;

  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
      residual[token_offset + i] = static_cast<scalar_t>(x);
    }
    // Norm
    x = static_cast<float>(static_cast<scalar_t>(x * rms) * weight[i]);
    // Quant
    output[token_offset + i] =
        ScaledQuant<scalar_out_t, is_scale_inverted, has_azp>::quant_fn(
            x, scale, azp);
  }
}

namespace vectorized {

// Compute rms(input)
// hidden_size must be a multiple of 4
template <typename scalar_t, bool has_residual = false>
__device__ void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int const hidden_size, float const epsilon,
                            scalar_t const* __restrict__ residual = nullptr) {
  int const token_offset = blockIdx.x * hidden_size;

  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual =
        reinterpret_cast<vec4_t<scalar_t> const*>(&residual[token_offset]);
  }

  // sum of squares
  float ss = 0.0f;

  int const tid = threadIdx.x;
  int const num_vec_elems = hidden_size >> 2;

#pragma unroll 4
  for (int i = tid; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> in = vec_input[i];

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

    ss += x.x * x.x;
    ss += x.y * x.y;
    ss += x.z * x.z;
    ss += x.w * x.w;
  }

  ss = blockReduceSum<float>(ss);
  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();

  *rms = s_rms;
}

// hidden_size must be a multiple of 4
template <typename scalar_t, typename scalar_out_t, bool is_scale_inverted,
          bool has_residual = false, bool has_azp = false>
__device__ void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_t const* __restrict__ weight,
                               float const rms, float const scale,
                               int const hidden_size,
                               scalar_t* __restrict__ residual = nullptr,
                               int32_t const azp = 0) {
  int const token_offset = blockIdx.x * hidden_size;

  // Vectorized input/output/weight/residual to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight =
      reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  q8x4_t<scalar_out_t>* vec_output =
      reinterpret_cast<q8x4_t<scalar_out_t>*>(&output[token_offset]);
  vec4_t<scalar_t>* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual = reinterpret_cast<vec4_t<scalar_t>*>(&residual[token_offset]);
  }

  int const tid = threadIdx.x;
  int const num_vec_elems = hidden_size >> 2;

#pragma unroll 4
  for (int i = tid; i < num_vec_elems; i += blockDim.x) {
    vec4_t<scalar_t> const in = vec_input[i];
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
      // Update residual
      r.x = static_cast<scalar_t>(x.x);
      r.y = static_cast<scalar_t>(x.y);
      r.z = static_cast<scalar_t>(x.z);
      r.w = static_cast<scalar_t>(x.w);
      vec_residual[i] = r;
    }

    q8x4_t<scalar_out_t> out;
    out.x = ScaledQuant<scalar_out_t, is_scale_inverted, has_azp>::quant_fn(
        static_cast<scalar_t>(x.x * rms) * w.x, scale, azp);
    out.y = ScaledQuant<scalar_out_t, is_scale_inverted, has_azp>::quant_fn(
        static_cast<scalar_t>(x.y * rms) * w.y, scale, azp);
    out.z = ScaledQuant<scalar_out_t, is_scale_inverted, has_azp>::quant_fn(
        static_cast<scalar_t>(x.z * rms) * w.z, scale, azp);
    out.w = ScaledQuant<scalar_out_t, is_scale_inverted, has_azp>::quant_fn(
        static_cast<scalar_t>(x.w * rms) * w.w, scale, azp);
    vec_output[i] = out;
  }
}

}  // namespace vectorized

}  // namespace vllm
