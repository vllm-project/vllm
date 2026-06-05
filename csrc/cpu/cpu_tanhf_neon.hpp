// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef CPU_TANHF_NEON_HPP
#define CPU_TANHF_NEON_HPP

#include <arm_neon.h>

namespace vec_op {

// Implementation of tanhf adapted from Arm Optimized Routines (tanhf
// AdvSIMD)
// https://github.com/ARM-software/optimized-routines/blob/master/math/aarch64/advsimd/tanhf.c
float32x4_t fast_tanhf_f32x4(float32x4_t x);

}  // namespace vec_op

#endif  // CPU_TANHF_NEON_HPP