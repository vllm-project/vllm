// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef CPU_ERF_SVE_HPP
#define CPU_ERF_SVE_HPP

#include <arm_sve.h>

#include "cpu_erf_data.hpp"

namespace vec_op {

namespace {

struct ErffSVEConstants {
  float min;
  float max;
  float scale;
  float shift;
  float third;
};

const ErffSVEConstants kErffSVEConstants = {
    .min = 0x1.cp-7f,
    .max = 3.9375f,
    .scale = 0x1.20dd76p+0f,
    .shift = 0x1p16f,
    .third = 0x1.555556p-2f,
};

template <typename T>
inline const T* erff_sve_ptr_barrier(const T* ptr) {
  const T* opaque_ptr = ptr;
  __asm__("" : "+r"(opaque_ptr));
  return opaque_ptr;
}

}  // namespace

// Implementation adapted from Arm Optimized Routines (SVE erff):
// https://github.com/ARM-software/optimized-routines/blob/master/math/aarch64/sve/erff.c
//
// Maximum error is 1.93 ULP near zero and 1.26 ULP elsewhere.
inline svfloat32_t fast_erff_f32xn(svfloat32_t x, const svbool_t pg) {
  const ErffSVEConstants* d = erff_sve_ptr_barrier(&kErffSVEConstants);

  const svbool_t a_gt_min = svacgt(pg, x, d->min);
  const svbool_t a_ge_max = svacge(pg, x, d->max);
  const svfloat32_t a = svabs_x(pg, x);

  const svfloat32_t shift = svdup_n_f32(d->shift);
  const svfloat32_t z = svadd_x(pg, a, shift);
  svuint32_t index = svand_x(pg, svreinterpret_u32(z), 0xfff);
  index = svadd_x(pg, index, index);

  const svfloat32_t r = svsub_z(a_gt_min, z, shift);
  const float* erf_table = &kErffData.tab[0].erf;
  const float* scale_table = &kErffData.tab[0].scale;
  const svfloat32_t erfr = svld1_gather_index(a_gt_min, erf_table, index);
  const svfloat32_t gathered_scale =
      svld1_gather_index(a_gt_min, scale_table, index);
  const svfloat32_t scale =
      svsel_f32(a_gt_min, gathered_scale, svdup_n_f32(d->scale));

  const svfloat32_t delta = svsub_x(pg, a, r);
  const svfloat32_t delta2 = svmul_x(pg, delta, delta);
  svfloat32_t y = svmla_x(pg, r, delta, d->third);
  y = svmla_x(pg, erfr, scale, svmls_x(pg, delta, delta2, y));

  y = svsel_f32(a_ge_max, svdup_n_f32(1.0f), y);
  const svuint32_t sign = svand_x(pg, svreinterpret_u32(x), 0x80000000u);
  return svreinterpret_f32(svorr_x(pg, sign, svreinterpret_u32(y)));
}

}  // namespace vec_op

#endif  // CPU_ERF_SVE_HPP
