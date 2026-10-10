// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef CPU_TANHF_SVE_HPP
#define CPU_TANHF_SVE_HPP

#include <arm_sve.h>

namespace vec_op {

namespace {

struct TanhfSVEConstants {
  float c2;
  float c4;
  float ln2_hi;
  float ln2_lo;
  float c0;
  float two_over_ln2;
  float c1;
  float c3;
  float special_bound;
};

const TanhfSVEConstants kTanhfSVEConstants = {
    .c2 = 0x1.555736p-5f,
    .c4 = 0x1.6b55a2p-10f,
    .ln2_hi = 0x1.62e4p-1f,
    .ln2_lo = 0x1.7f7d1cp-20f,
    .c0 = 0x1.fffffep-2f,
    .two_over_ln2 = 0x1.715476p+1f,
    .c1 = 0x1.5554aep-3f,
    .c3 = 0x1.12287cp-7f,
    .special_bound = 0x1.205966p+3f,
};

// Return the ptr but hide it's value from the compiler so accesses
// through it can't be optimised based on contents.
template <typename T>
inline const T* tanhf_sve_ptr_barrier(const T* ptr) {
  const T* opaque_ptr = ptr;
  __asm__("" : "+r"(opaque_ptr));
  return opaque_ptr;
}

inline svfloat32_t e2xm1f_sve_inline(svfloat32_t x, const svbool_t pg,
                                     const TanhfSVEConstants* d) {
  const svfloat32_t lane_constants = svld1rq_f32(svptrue_b32(), &d->c2);

  svfloat32_t j = svmul_x(svptrue_b32(), x, d->two_over_ln2);
  j = svrinta_x(pg, j);
  svfloat32_t f = svadd_x(pg, x, x);
  f = svmls_lane(f, j, lane_constants, 2);
  f = svmls_lane(f, j, lane_constants, 3);

  const svfloat32_t p12 = svmla_lane(svdup_n_f32(d->c1), f, lane_constants, 0);
  const svfloat32_t p34 = svmla_lane(svdup_n_f32(d->c3), f, lane_constants, 1);
  const svfloat32_t f2 = svmul_x(svptrue_b32(), f, f);
  svfloat32_t p = svmla_x(pg, p12, f2, p34);
  p = svmla_x(pg, svdup_n_f32(d->c0), f, p);
  p = svmla_x(pg, f, f2, p);

  const svfloat32_t scale =
      svscale_x(pg, svdup_n_f32(1.0f), svcvt_s32_x(pg, j));
  return svmla_x(pg, svsub_x(pg, scale, 1.0f), p, scale);
}

// Calculate the result tanh(x) = q / (q+2) and set special lanes to ±1
__attribute__((noinline)) inline svfloat32_t tanhf_sve_special_case(
    svfloat32_t x, const svbool_t pg, const svbool_t special, svfloat32_t q) {
  const svfloat32_t y = svdiv_x(pg, q, svadd_x(pg, q, 2.0f));
  const svfloat32_t abs_x = svabs_x(svptrue_b32(), x);
  const svuint32_t abs_bits = svreinterpret_u32(abs_x);
  const svuint32_t sign =
      sveor_x(svptrue_b32(), svreinterpret_u32(x), abs_bits);
  const svfloat32_t special_y =
      svreinterpret_f32(svorr_x(svptrue_b32(), sign, svdup_n_u32(0x3f800000u)));
  return svsel_f32(special, special_y, y);
}

}  // namespace

// Implementation adapted from Arm Optimized Routines (SVE tanhf):
// https://github.com/ARM-software/optimized-routines/blob/master/math/aarch64/sve/tanhf.c
//
// Approximation for single-precision SVE tanh(x), using a simplified
// version of expm1f. The maximum error is 2.06 + 0.5 ULP:
// _ZGVsMxv_tanhf (0x1.fc1832p-5) got 0x1.fb71a4p-5 want 0x1.fb71aap-5.
inline svfloat32_t fast_tanhf_f32x4(svfloat32_t x, const svbool_t pg) {
  const TanhfSVEConstants* d = tanhf_sve_ptr_barrier(&kTanhfSVEConstants);

  // tanh(x) = (e^2x - 1) / (e^2x + 1)
  // q = e^2x -1
  const svfloat32_t q = e2xm1f_sve_inline(x, pg, d);

  // Check for special cases
  const svbool_t special = svacgt(pg, x, d->special_bound);

  // Fall back to vectorised special case for any lanes which would cause
  // expm1 to overflow
  if (svptest_any(pg, special)) {
    return tanhf_sve_special_case(x, pg, special, q);
  }

  // Complete fast path if no special lanes
  // tanh(x) = q / (q+2)
  return svdiv_x(pg, q, svadd_x(pg, q, 2.0f));
}

}  // namespace vec_op

#endif  // CPU_TANHF_SVE_HPP
