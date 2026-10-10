// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace litetopk {

// Ordered unsigned key: a < b as floats (with -0 < +0) <=> ordered_key(a) <
// ordered_key(b)
__device__ __forceinline__ uint32_t ordered_key(float x) {
  const uint32_t bits = __float_as_uint(x);
  return bits ^ (static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) |
                 0x80000000u);
}

// Key of a threshold used with `x >= edge`: a zero edge also admits -0.0
__device__ __forceinline__ uint32_t edge_key(float edge) {
  return edge == 0.0f ? 0x7fffffffu : ordered_key(edge);
}

__device__ __forceinline__ float half_bits_to_float(uint32_t h) {
  return __half2float(__ushort_as_half(static_cast<unsigned short>(h)));
}

// Smallest score of DeepGEMM's coarse bins 0..t (bin 0 holds the largest
// scores). Positive bins t = 511 - c hold the FP16-RN magnitude code c = |h| >>
// 6 for c < 304, and [c - 288, c - 287) above; negative bins t = 512 + c hold
// the same codes, with lower-inclusive unit bins [-(c - 287), -(c - 288)).
__device__ __forceinline__ float coarse_lower_edge(uint32_t t) {
  if (t < 512) {
    const uint32_t c = 511 - t;
    if (c == 0) return 0.0f;
    if (c <= 304)  // first score rounding to FP16 code c << 6: ties go to the
                   // even code
      return 0.5f *
             (half_bits_to_float(c * 64 - 1) + half_bits_to_float(c * 64));
    return static_cast<float>(c - 288);
  }
  const uint32_t c = t - 512;
  if (c == 511) return -__int_as_float(0x7f800000);
  if (c >= 304) return -static_cast<float>(c - 287);
  // Largest magnitude rounding to at most code (c << 6) | 63: just below the
  // midpoint, whose tie rounds up
  const float mid = 0.5f * (half_bits_to_float(c * 64 + 63) +
                            half_bits_to_float(c * 64 + 64));
  return -__uint_as_float(__float_as_uint(mid) - 1);
}

// Bin of a live, non-NaN score: DeepGEMM's coarse_histogram_bin, whose lower
// edges coarse_lower_edge returns
__device__ __forceinline__ uint32_t coarse_bin(float score) {
  const uint32_t bits = __float_as_uint(score);
  uint32_t magnitude = bits & 0x7fffffffu;
  const bool negative = (bits >> 31) && magnitude != 0;
  int code = (__half_as_ushort(__float2half_rn(score)) & 0x7fffu) >> 6;
  if (code >= 304) {
    magnitude -= negative;  // lower-inclusive negative unit bins
    const float bounded = __uint_as_float(min(magnitude, 0x435f0000u));
    code = max(304, __float2int_rd(bounded) + 288);
  }
  return negative ? 512 + code : 511 - code;
}

}  // namespace litetopk
