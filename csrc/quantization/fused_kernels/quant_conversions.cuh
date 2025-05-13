#pragma once

/**
 * __device__ helper functions to deal with float -> quant datatype conversion
 */

#include "quantization/vectorization.cuh"
// TODO(luka/varun):refactor common.cuh to use this file instead
#include "quantization/fp8/common.cuh"

namespace vllm {

// TODO(luka/varun): combine into common utilities for int8
//  (with int8_quant_kernels.cu)
static __device__ __forceinline__ int8_t float_to_int8_rn(float const x) {
#ifdef USE_ROCM
  static const float i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static const float i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  // round
  float dst = std::nearbyint(x);
  // saturate

  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // dst = std::clamp(dst, i8_min, i8_max);
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

template <typename fp8_type>
static __device__ __forceinline__ fp8_type float_to_fp8(float const x) {
  float const r =
      fmax(-quant_type_max_v<fp8_type>, fmin(x, quant_type_max_v<fp8_type>));
  return static_cast<fp8_type>(r);
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
                       std::is_same_v<quant_type_t, c10::Float8_e4m3fn> ||
                       std::is_same_v<quant_type_t, c10::Float8_e4m3fnuz>>> {
  static __device__ __forceinline__ quant_type_t quant_fn(float const x,
                                                          float const scale) {
    if constexpr (is_scale_inverted) {
      return float_to_fp8<quant_type_t>(x * scale);
    } else {
      return float_to_fp8<quant_type_t>(x / scale);
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

}  // namespace vllm
