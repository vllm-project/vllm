#pragma once

#include <climits>
#include <iostream>

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename T>
inline constexpr std::enable_if_t<std::is_integral_v<T>, T> ceil_div(T a, T b) {
  return (a + b - 1) / b;
}