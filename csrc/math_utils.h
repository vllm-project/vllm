#pragma once

#include "cuda_utils.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cmath>
#include <type_traits>

template <typename T1, typename T2>
HOST_DEVICE_INLINE constexpr auto div_ceil(T1 a, T2 b) {
  return (a + b - 1) / b;
}

template <typename T1, typename T2>
HOST_DEVICE_INLINE constexpr auto round_up(T1 a, T2 b) {
  return div_ceil(a, b) * b;
}

template <typename T1, typename T2>
HOST_DEVICE_INLINE constexpr auto round_down(T1 a, T2 b) {
  return (a / b) * b;
}

template <typename T>
inline std::enable_if_t<std::is_integral_v<T>, bool> not_zero(T value) {
  return value != 0;
}

template <typename T>
inline std::enable_if_t<std::is_floating_point_v<T> ||
                            std::is_same_v<T, c10::Half> ||
                            std::is_same_v<T, c10::BFloat16>,
                        bool>
not_zero(T value) {
  using std::fpclassify;
  return fpclassify(value) != FP_ZERO;
}

template <typename T>
bool is_zero(T value) {
  return !not_zero(value);
}
