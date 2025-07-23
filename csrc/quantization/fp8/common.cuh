#pragma once

#include "quantization/vectorization.cuh"
#include "quantization/utils.cuh"

#include <cmath>

#ifdef USE_ROCM
  #include "amd/quant_utils.cuh"
#endif

// Determines the preferred FP8 type for the current platform.
// Note that for CUDA this just returns true,
// but on ROCm it will check device props.
static bool is_fp8_ocp() {
#ifndef USE_ROCM
  return true;
#else
  auto dprops = at::cuda::getCurrentDeviceProperties();
  std::string device_arch = dprops->gcnArchName;
  size_t substring = device_arch.find("gfx94");
  return substring == std::string::npos;
#endif
}

namespace vllm {

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int*)addr, __float_as_uint(value)));

  return old;
}

template <bool is_scale_inverted, typename fp8_type>
__device__ __forceinline__ fp8_type scaled_fp8_conversion(float const val,
                                                          float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r =
      fmaxf(-quant_type_max_v<fp8_type>, fminf(x, quant_type_max_v<fp8_type>));
#ifndef USE_ROCM
  return static_cast<fp8_type>(r);
#else
  // Use hardware cvt instruction for fp8 on rocm
  return fp8::cvt_c10<fp8_type>(r);
#endif
}

// Compute the absolute maximum m of the input tensor and store
// m / float8_e4m3::max() in *scale. Each thread block performs a
// reduction tree and the memory in scale is atomically updated.
// So to get the right answer, *scale needs to be initialized to
// a value <= 0.0 and we need to wait for all thread blocks to
// finish before consuming *scale.
template <typename scalar_t, typename fp8_type>
__global__ void segmented_max_reduction(float* __restrict__ scale,
                                        const scalar_t* __restrict__ input,
                                        int64_t num_elems) {
  __shared__ float cache[256];
  int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

  // First store maximum for all values processes by
  // the current thread in cache[threadIdx.x]
  scalar_t tmp = 0.0;
  while (i < num_elems) {
    float x = static_cast<float>(input[i]);
    tmp = fmaxf(tmp, fabsf(x));
    i += blockDim.x * gridDim.x;
  }
  cache[threadIdx.x] = tmp;

  __syncthreads();

  // Now perform parallel reduction within the thread block
  int ib = blockDim.x / 2;
  while (ib != 0) {
    if (threadIdx.x < ib && cache[threadIdx.x + ib] > cache[threadIdx.x]) {
      cache[threadIdx.x] = cache[threadIdx.x + ib];
    }
    __syncthreads();
    ib /= 2;
  }
  // Finally, since cache[0] contains the maximum for this thread block,
  // atomically write the max to the target location
  if (threadIdx.x == 0) {
    atomicMaxFloat(scale, cache[0] / quant_type_max_v<fp8_type>);
  }
}

template <typename scalar_t>
__device__ float thread_max_vec(scalar_t const* __restrict__ input,
                                int64_t const num_elems, int const tid,
                                int const step) {
  constexpr size_t VEC_SIZE = 16;
  using scalarxN_t = vec_n_t<scalar_t, VEC_SIZE>;
  // Vectorized input/output to better utilize memory bandwidth.
  auto const* vectorized_in = reinterpret_cast<scalarxN_t const*>(input);

  // num_elems / VEC_SIZE (which is 16)
  int64_t const num_vec_elems = num_elems >> 4;
  float absmax_val = 0.0f;

#pragma unroll
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    scalarxN_t in_vec = vectorized_in[i];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      absmax_val = fmaxf(absmax_val, fabsf(in_vec.val[j]));
    }
  }

  // Handle the remaining elements if num_elems is not divisible by VEC_SIZE
  for (int64_t i = num_vec_elems * VEC_SIZE + tid; i < num_elems; i += step) {
    absmax_val = fmaxf(absmax_val, fabsf(input[i]));
  }

  return absmax_val;
}

template <typename scalar_t, bool is_scale_inverted, typename fp8_type>
__device__ void scaled_fp8_conversion_vec(fp8_type* __restrict__ out,
                                          scalar_t const* __restrict__ input,
                                          float const scale,
                                          int64_t const num_elems,
                                          int const tid, int const step) {
  constexpr size_t VEC_SIZE = 16;
  using scalarxN_t = vec_n_t<scalar_t, VEC_SIZE>;
  using float8xN_t = q8_n_t<fp8_type, VEC_SIZE>;
  // Vectorized input/output to better utilize memory bandwidth.
  auto const* vectorized_in = reinterpret_cast<scalarxN_t const*>(input);
  auto* vectorized_out = reinterpret_cast<float8xN_t*>(out);

  // num_elems / VEC_SIZE (which is 16)
  int64_t const num_vec_elems = num_elems >> 4;

#pragma unroll
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    scalarxN_t in_vec = vectorized_in[i];
    float8xN_t out_vec;

#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      out_vec.val[j] = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
          static_cast<float>(in_vec.val[j]), scale);
    }
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by VEC_SIZE
  for (int64_t i = num_vec_elems * VEC_SIZE + tid; i < num_elems; i += step) {
    out[i] = scaled_fp8_conversion<is_scale_inverted, fp8_type>(
        static_cast<float>(input[i]), scale);
  }
}

}  // namespace vllm
