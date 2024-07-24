#pragma once

/**
 * __device__ algorithms that perform vectorized loads/stores of input/output.
 */

#include "vectorization.cuh"

namespace vllm {
namespace vectorized {

template <typename scalar_t>
__device__ float thread_max_vec(scalar_t const* __restrict__ input,
                                int64_t const num_elems, int const tid,
                                int const step) {
  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vectorized_in =
      reinterpret_cast<vec4_t<scalar_t> const*>(input);

  int64_t const num_vec_elems = num_elems >> 2;
  float absmax_val = 0.0f;

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    absmax_val = max(absmax_val, fabs(in_vec.x));
    absmax_val = max(absmax_val, fabs(in_vec.y));
    absmax_val = max(absmax_val, fabs(in_vec.z));
    absmax_val = max(absmax_val, fabs(in_vec.w));
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    absmax_val = max(absmax_val, fabs(input[i]));
  }

  return absmax_val;
}

} // namespace vectorized
} // namespace vllm
