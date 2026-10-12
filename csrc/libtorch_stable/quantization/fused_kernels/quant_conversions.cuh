#pragma once

/**
 * __device__ helper functions to deal with float -> quant datatype conversion
 */

#include "libtorch_stable/quantization/vectorization.cuh"
// TODO(luka/varun):refactor common.cuh to use this file instead
#include "../../../quantization/w8a8/fp8/common.cuh"

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

template <typename quant_type_t, bool is_scale_inverted, typename Enable = void>
struct ScaledQuantVec4;

template <bool is_scale_inverted>
struct ScaledQuantVec4<int8_t, is_scale_inverted> {
  static __device__ __forceinline__ q8x4_t<int8_t> quant_vec(
      vec4_t<float> const& val, float const scale) {
    q8x4_t<int8_t> out;
    const float eff_scale = is_scale_inverted ? scale : (1.0f / scale);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  // Efficient float -> int8 with saturation using PTX
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      float const scaled = val.val[j] * eff_scale;
      int32_t res;
      asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(res) : "f"(scaled));
      out.val[j] = static_cast<int8_t>(res);
    }
#else
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      out.val[j] = float_to_int8_rn(val.val[j] * eff_scale);
    }
#endif
    return out;
  }
};

template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuantVec4<
    quant_type_t, is_scale_inverted,
    std::enable_if_t<std::is_same_v<quant_type_t, c10::Float8_e4m3fn> ||
                     std::is_same_v<quant_type_t, c10::Float8_e4m3fnuz>>> {
  static __device__ __forceinline__ q8x4_t<quant_type_t> quant_vec(
      vec4_t<float> const& val, float const scale) {
    q8x4_t<quant_type_t> out;
    const float eff_scale = is_scale_inverted ? scale : (1.0f / scale);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && \
    defined(CUDA_PTX_FP8_CVT_SUPPORTED)
    // Convert 2 pairs of float into 2x uint16 (each holding 2 fp8 values)
    uint16_t raw_fp8_pair0, raw_fp8_pair1;
    float const v0 = val.val[0] * eff_scale;
    float const v1 = val.val[1] * eff_scale;
    float const v2 = val.val[2] * eff_scale;
    float const v3 = val.val[3] * eff_scale;

    if constexpr (std::is_same_v<quant_type_t, c10::Float8_e4m3fn>) {
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
          : "=h"(raw_fp8_pair0)
          : "f"(v1), "f"(v0));
      asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
          : "=h"(raw_fp8_pair1)
          : "f"(v3), "f"(v2));
    }

    // Copy the resulting 4 bytes into output vector
    uint32_t packed =
        (static_cast<uint32_t>(raw_fp8_pair1) << 16) | raw_fp8_pair0;
    *reinterpret_cast<uint32_t*>(&out) = packed;
#else
  // Generic fallback for pre-Ada GPUs or software emulation
  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      out.val[j] = float_to_fp8<quant_type_t>(val.val[j] * eff_scale);
    }
#endif
    return out;
  }
};

}  // namespace vllm
