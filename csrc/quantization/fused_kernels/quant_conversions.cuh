#pragma once

/**
 * __device__ helper functions to deal with float -> quant datatype conversion
 */

#include "vectorization.cuh"

namespace vllm {

namespace detail {
__device__ __forceinline__ int8_t float_to_int8_rn(float const x) {
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
__device__ __forceinline__ c10::Float8_e4m3fn float_to_fp8(float const x) {
  float const r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
  return static_cast<c10::Float8_e4m3fn>(r);
}

__device__ __forceinline__ int32_t float_to_int32_rn(float x) {
#ifdef USE_ROCM
  static const float i32_min =
      static_cast<float>(std::numeric_limits<int32_t>::min());
  static const float i32_max =
      static_cast<float>(std::numeric_limits<int32_t>::max());
  // round
  float dst = std::nearbyint(x);
  // saturate
  dst = std::clamp(dst, i32_min, i32_max);
  return static_cast<int32_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int32_t&>(dst);
#endif
}

__device__ __forceinline__ int8_t int32_to_int8(int32_t x) {
#ifdef USE_ROCM
  static const float i8_min =
      static_cast<int32_t>(std::numeric_limits<int8_t>::min());
  static const float i8_max =
      static_cast<int32_t>(std::numeric_limits<int8_t>::max());

  // saturate
  int32_t dst = std::clamp(x, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.sat.s8.s32 %0, %1;" : "=r"(dst) : "r"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

}  // namespace detail

template <typename quant_type_t, bool is_scale_inverted, bool has_azp,
          typename enable = void>
struct ScaledQuant;

template <typename quant_type_t, bool is_scale_inverted, bool has_azp>
struct ScaledQuant<
    quant_type_t, is_scale_inverted, has_azp,
    typename std::enable_if_t<std::is_same_v<quant_type_t, int8_t>>> {
  static __device__ __forceinline__ quant_type_t
  quant_fn(float const x, float const scale, int32_t const azp = 0) {
    float scaled_x = 0.0f;
    if constexpr (is_scale_inverted) {
      scaled_x = x * scale;
    } else {
      scaled_x = x / scale;
    }

    if constexpr (has_azp) {
      return detail::int32_to_int8(detail::float_to_int32_rn(scaled_x) + azp);
    } else {
      return detail::float_to_int8_rn(scaled_x);
    }
  }
};

// fp8 doesn't support asymmetric quant
template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuant<quant_type_t, is_scale_inverted, false,
                   typename std::enable_if_t<
                       std::is_same_v<quant_type_t, c10::Float8_e4m3fn>>> {
  static __device__ __forceinline__ quant_type_t
  quant_fn(float const x, float const scale, int32_t const azp = 0) {
    if constexpr (is_scale_inverted) {
      return detail::float_to_fp8(x * scale);
    } else {
      return detail::float_to_fp8(x / scale);
    }
  }
};

template <typename scalar_t, typename quant_type_t, bool is_scale_inverted,
          bool has_azp>
__device__ void scaled_quant_conversion(quant_type_t* __restrict__ output,
                                        scalar_t const* __restrict__ input,
                                        float const scale, int32_t const azp,
                                        int const tid, int const num_elements,
                                        int const step) {
  for (int i = tid; i < num_elements; i += step) {
    output[i] = ScaledQuant<quant_type_t, is_scale_inverted, has_azp>(
        input[i], scale, azp);
  }
}

namespace vectorized {

// Vectorized version of scaled_quant_conversion
template <typename scalar_t, typename quant_type_t, bool is_scale_inverted,
          bool has_azp>
__device__ void scaled_quant_conversion(quant_type_t* __restrict__ out,
                                        scalar_t const* __restrict__ input,
                                        float const scale, int32_t const azp,
                                        int const tid, int const num_elems,
                                        int const step) {
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

    out_vec.x = ScaledQuant<quant_type_t, is_scale_inverted, has_azp>(
        in_vec.x, scale, azp);
    out_vec.y = ScaledQuant<quant_type_t, is_scale_inverted, has_azp>(
        in_vec.y, scale, azp);
    out_vec.z = ScaledQuant<quant_type_t, is_scale_inverted, has_azp>(
        in_vec.z, scale, azp);
    out_vec.w = ScaledQuant<quant_type_t, is_scale_inverted, has_azp>(
        in_vec.w, scale, azp);
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    out[i] = ScaledQuant<quant_type_t, is_scale_inverted, has_azp>(input[i],
                                                                   scale, azp);
  }
}

}  // namespace vectorized

}  // namespace vllm
