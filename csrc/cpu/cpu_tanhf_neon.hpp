// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef CPU_TANHF_NEON_HPP
#define CPU_TANHF_NEON_HPP

#include <cstdint>
#include <arm_neon.h>

namespace vec_op {

namespace {

struct TanhfConstants {
  float32x4_t special_bound;
  float32x4_t two;
  float32x4_t c0;
  float32x4_t c2;
  int32x4_t exponent_bias;
  float c1;
  float c3;
  float two_over_ln2;
  float c4;
  float ln2_hi;
  float ln2_lo;
};

const TanhfConstants kTanhfConstants = {
    // 9.01, above which tanhf rounds to 1 (or -1 for  negative).
    .special_bound = vdupq_n_f32(0x1.205966p+3f),
    .two = vdupq_n_f32(0x1.0p+1f),
    .c0 = vdupq_n_f32(0x1.fffffep-2f),
    .c2 = vdupq_n_f32(0x1.555736p-5f),
    .exponent_bias = vdupq_n_s32(0x3f800000),
    .c1 = 0x1.5554aep-3f,
    .c3 = 0x1.12287cp-7f,
    .two_over_ln2 = 0x1.715476p+1f,
    .c4 = 0x1.6b55a2p-10f,
    .ln2_hi = 0x1.62e4p-1f,
    .ln2_lo = 0x1.7f7d1cp-20f,
};

// Return the ptr but hide it's value from the compiler so accesses
// through it can't be optimised based on contents.
template <typename T>
inline const T* ptr_barrier(const T* ptr) {
  const T* opaque_ptr = ptr;
  __asm__("" : "+r"(opaque_ptr));
  return opaque_ptr;
}

// Check whether any lanes in the mask are set
inline bool any_u32(uint32x4_t x) { return vmaxvq_u32(x) != 0; }

// e^2x - 1 inline helper
inline float32x4_t e2xm1f_inline(float32x4_t x, const TanhfConstants* d) {
  float32x2_t ln2 = vld1_f32(&d->ln2_hi);
  float32x4_t lane_consts = vld1q_f32(&d->c1);

  // Reduce argument: f in [-ln2/2, ln2/2], i is exact.
  float32x4_t j = vrndaq_f32(vmulq_laneq_f32(x, lane_consts, 2));
  int32x4_t i = vcvtq_s32_f32(j);
  float32x4_t f = vaddq_f32(x, x);
  f = vfmsq_lane_f32(f, j, ln2, 0);
  f = vfmsq_lane_f32(f, j, ln2, 1);

  // Approximate expm1(f) with polynomial P, expm1(f) ~= f + f^2 * P(f)
  float32x4_t f2 = vmulq_f32(f, f);
  float32x4_t f4 = vmulq_f32(f2, f2);
  float32x4_t p01 = vfmaq_laneq_f32(d->c0, f, lane_consts, 0);
  float32x4_t p23 = vfmaq_laneq_f32(d->c2, f, lane_consts, 1);
  float32x4_t poly = vfmaq_f32(p01, f2, p23);
  poly = vfmaq_laneq_f32(poly, f4, lane_consts, 3);
  poly = vfmaq_f32(f, f2, poly);

  // scale = 2^i
  int32x4_t u = vaddq_s32(vshlq_n_s32(i, 23), d->exponent_bias);
  float32x4_t scale = vreinterpretq_f32_s32(u);
  return vfmaq_f32(vsubq_f32(scale, vdupq_n_f32(1.0f)), poly, scale);
}

// Calculate the result tanh(x) = q / (q+2) and set special lanes to ±1
inline float32x4_t special_case(float32x4_t x, float32x4_t q,
                                uint32x4_t special) {
  const TanhfConstants* d = ptr_barrier(&kTanhfConstants);

  float32x4_t y = vdivq_f32(q, vaddq_f32(q, d->two));
  uint32x4_t ix = vreinterpretq_u32_f32(x);
  uint32x4_t one_bits = vreinterpretq_u32_s32(d->exponent_bias);
  uint32x4_t sign_mask = vdupq_n_u32(0x80000000u);
  uint32x4_t special_bits = vbslq_u32(sign_mask, ix, one_bits);
  float32x4_t special_y = vreinterpretq_f32_u32(special_bits);
  return vbslq_f32(special, special_y, y);
}

}  // namespace

// Implementation of tanhf adapted from Arm Optimized Routines (tanhf
// AdvSIMD)
// https://github.com/ARM-software/optimized-routines/blob/master/math/aarch64/advsimd/tanhf.c
//
// Approximation for single-precision vector tanh(x), using a simplified
// version of expm1f. The maximum error is 2.08 + 0.5 ULP:
// _ZGVnN4v_tanhf (0x1.fa5eep-5) got 0x1.f9ba02p-5 want 0x1.f9ba08p-5.
inline float32x4_t fast_tanhf_f32x4(float32x4_t x) {
  const TanhfConstants* d = ptr_barrier(&kTanhfConstants);

  // tanh(x) = (e^2x - 1) / (e^2x + 1)
  // q = e^2x -1
  float32x4_t q = e2xm1f_inline(x, d);

  // Check for special cases
  uint32x4_t special = vcagtq_f32(x, d->special_bound);

  // Fall back to vectorised special case for any lanes which would cause
  // expm1 to overflow
  if (any_u32(special)) {
    return special_case(x, q, special);
  }

  // Complete fast path if no special lanes
  // tanh(x) = q / (q+2)
  return vdivq_f32(q, vaddq_f32(q, d->two));
}

}  // namespace vec_op

#endif  // CPU_TANHF_NEON_HPP