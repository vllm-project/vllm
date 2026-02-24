#pragma once

/**
 * Quantization utilities including:
 *   Adjusted maximum values for qtypes.
 *   Minimum scaling factors for qtypes.
 */

#include <cmath>
#include <torch/headeronly/macros/Macros.h>

#ifndef USE_ROCM
  #include <torch/headeronly/util/Float8_e4m3fn.h>
  #define MAYBE_HOST_DEVICE C10_HOST_DEVICE
#else
  #include <torch/headeronly/util/Float8_e4m3fn.h>
  #include <torch/headeronly/util/Float8_e4m3fnuz.h>
  // ROCm doesn't seem to need C10_HOST_DEVICE for static constexpr
  #define MAYBE_HOST_DEVICE
#endif

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<T, int8_t>>>
struct quant_type_max {
  static constexpr T val() { return std::numeric_limits<T>::max(); }
};

// Using the default max value from pytorch (240.0 0x7F) will cause accuracy
// issues when running dynamic quantization. Here use 224.0 0x7E for rocm.
template <>
struct quant_type_max<c10::Float8_e4m3fnuz> {
  static constexpr c10::Float8_e4m3fnuz val() {
    return c10::Float8_e4m3fnuz(0x7E, c10::Float8_e4m3fnuz::from_bits());
  }
};

template <typename T>
MAYBE_HOST_DEVICE static constexpr T quant_type_max_v =
    quant_type_max<T>::val();

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<T, int8_t>>>
struct min_scaling_factor {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return 1.0f / (quant_type_max_v<T> * 512.0f);
  }
};

template <>
struct min_scaling_factor<int8_t> {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return std::numeric_limits<float>::epsilon();
  }
};
