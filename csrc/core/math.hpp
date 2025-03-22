#pragma once

#include <climits>
#include <iostream>

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
#ifdef _WIN32
  return 1 << (CHAR_BIT * sizeof(num) - __lzcnt(num - 1));
#else
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
#endif
}
