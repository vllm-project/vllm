#pragma once

/**
 * __device__ layernorm utilities.
 */

#include "vectorization.cuh"

namespace vllm {

// Compute 1.0/rms(input)
template <typename scalar_t>
__device__ void compute_rms(__shared__ float* s_rms,
                            scalar_t const* __restrict__ input,
                            int const hidden_size, float const epsilon) {
  // sum of squares
  float ss = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float const x = (float)input[blockIdx.x * hidden_size + idx];
    ss += x * x;
  }
  ss = blockReduceSum<float>(ss);
  if (threadIdx.x == 0) {
    *s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();
}

// Compute 1.0/rms(input + residual)
// Note that the kernel only performs the layernorm computation and DOES NOT
// update the residual.
template <typename scalar_t>
__device__ void compute_rms(__shared__ float* s_rms,
                            scalar_t* __restrict__ input,
                            scalar_t* __restrict__ residual,
                            int const hidden_size,
                            float const epsilon) {
  // sum of squares
  float ss = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float const r_val =
        static_cast<float>(residual[blockIdx.x * hidden_size + idx]);
    float const i_val =
        static_cast<float>(input[blockIdx.x * hidden_size + idx]);
    float const x = r_val + i_val;
    ss += x * x;
  }
  ss = blockReduceSum<float>(ss);
  if (threadIdx.x == 0) {
    *s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();
}

namespace vectorized {

// Compute 1.0/rms(input)
template <typename scalar_t>
__device__ void compute_rms(__shared__ float* s_rms,
                            scalar_t const* __restrict__ input,
                            int const hidden_size, float const epsilon) {
  // sum of squares
  float ss = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float const x = (float)input[blockIdx.x * hidden_size + idx];
    ss += x * x;
  }
  ss = blockReduceSum<float>(ss);
  if (threadIdx.x == 0) {
    *s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();
}

// Compute 1.0/rms(input + residual)
// Note that the kernel only performs the layernorm computation and DOES NOT
// update the residual.
template <typename scalar_t>
__device__ void compute_rms(__shared__ float* s_rms,
                            scalar_t* __restrict__ input,
                            scalar_t* __restrict__ residual,
                            float const epsilon,
                            int const tid, int const num_elems,
                            int const step) {
  // sum of squares
  float ss = 0.0f;

  int const num_vec_elems = num_elems >> 2;

  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vec_input =
      reinterpret_cast<vec4_t<scalar_t> const*>(input);
  vec4_t<scalar_t> const* vec_residual =
      reinterpret_cast<vec4_t<scalar_t> const*>(residual);

#pragma unroll 4
  for (int i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> const i_val = vec_input[i];
    vec4_t<scalar_t> const r_val = vec_residual[i]; 

    float x = 0.0f;
    x = static_cast<float>(i_val.x) + static_cast<float>(r_val.x);
    ss += x * x;
    x = static_cast<float>(i_val.y) + static_cast<float>(r_val.y);
    ss += x * x;
    x = static_cast<float>(i_val.z) + static_cast<float>(r_val.z);
    ss += x * x;
    x = static_cast<float>(i_val.w) + static_cast<float>(r_val.w);
    ss += x * x;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    float const i_val = static_cast<float>(input[i]);
    float const r_val = static_cast<float>(residual[i]);
    float const x = i_val + r_val;
    ss += x * x;
  }

  ss = blockReduceSum<float>(ss);
  if (threadIdx.x == 0) {
    *s_rms = rsqrtf(ss / num_elems  + epsilon);
  }
  __syncthreads();
}

} // namespace vectorized


} // namespace vllm

