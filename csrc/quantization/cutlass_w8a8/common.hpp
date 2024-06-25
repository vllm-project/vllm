#pragma once

#include "cutlass/cutlass.h"
#include <climits>

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                        \
  {                                                  \
    TORCH_CHECK(status == cutlass::Status::kSuccess, \
                cutlassGetStatusString(status))      \
  }

inline uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

