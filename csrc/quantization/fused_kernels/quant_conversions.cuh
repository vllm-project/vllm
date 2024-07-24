#pragma once

/**
 * __device__ helper functions to deal with float -> quant datatype conversion
 */

#include "vectorization.cuh"

namespace vllm {

static __device__ __forceinline__ int8_t float_to_int8_rn(float const x) {
#ifdef USE_ROCM
  static const float i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static const float i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  // round
  float dst = std::nearbyint(x);
  // saturate
  dst = std::clamp(dst, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

#define FP8_E4M3_MAX std::numeric_limits<c10::Float8_e4m3fn>::max()
static __device__ __forceinline__ c10::Float8_e4m3fn float_to_fp8(
    float const x) {
  float const r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
  return static_cast<c10::Float8_e4m3fn>(r);
}

template <typename quant_type_t, bool is_scale_inverted, typename enable = void>
struct ScaledQuant;

template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuant<
    quant_type_t, is_scale_inverted,
    typename std::enable_if_t<std::is_same_v<quant_type_t, int8_t>>> {
  static __device__ __forceinline__ quant_type_t quant_fn(float const x,
                                                          float const scale) {
    if constexpr (is_scale_inverted) {
      return float_to_int8_rn(x * scale);
    } else {
      return float_to_int8_rn(x / scale);
    }
  }
};

template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuant<quant_type_t, is_scale_inverted,
                   typename std::enable_if_t<
                       std::is_same_v<quant_type_t, c10::Float8_e4m3fn>>> {
  static __device__ __forceinline__ quant_type_t quant_fn(float const x,
                                                          float const scale) {
    if constexpr (is_scale_inverted) {
      return float_to_fp8(x * scale);
    } else {
      return float_to_fp8(x / scale);
    }
  }
};

template <typename scalar_t, typename quant_type_t, bool is_scale_inverted>
__device__ void scaled_quant_conversion(quant_type_t* __restrict__ output,
                                        scalar_t const* __restrict__ input,
                                        float const scale, int const tid,
                                        int const num_elements,
                                        int const step) {
  for (int i = tid; i < num_elements; i += step) {
    output[i] = ScaledQuant<quant_type_t, is_scale_inverted>(input[i], scale);
  }
}

namespace vectorized {

// Vectorized version of scaled_quant_conversion
template <typename scalar_t, typename quant_type_t, bool is_scale_inverted>
__device__ void scaled_quant_conversion(quant_type_t* __restrict__ out,
                                        scalar_t const* __restrict__ input,
                                        float const scale, int const tid,
                                        int const num_elems, int const step) {
  // Vectorized input/output to better utilize memory bandwidth.
  vec4_t<scalar_t> const* vectorized_in =
      reinterpret_cast<vec4_t<scalar_t> const*>(input);
  q8x4_t<quant_type_t>* vectorized_out =
      reinterpret_cast<q8x4_t<quant_type_t>*>(out);

  int const num_vec_elems = num_elems >> 2;

#pragma unroll 4
  for (int i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    q8x4_t<quant_type_t> out_vec;

    out_vec.x = ScaledQuant<quant_type_t, is_scale_inverted>(in_vec.x, scale);
    out_vec.y = ScaledQuant<quant_type_t, is_scale_inverted>(in_vec.y, scale);
    out_vec.z = ScaledQuant<quant_type_t, is_scale_inverted>(in_vec.z, scale);
    out_vec.w = ScaledQuant<quant_type_t, is_scale_inverted>(in_vec.w, scale);
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    out[i] = ScaledQuant<quant_type_t, is_scale_inverted>(input[i], scale);
  }
}

}  // namespace vectorized

}  // namespace vllm
