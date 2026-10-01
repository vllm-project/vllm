// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Per-architecture primitives of the SplitQ kernels. Every AMD target vLLM
// builds compiles the portable tier; faster tiers are selected at runtime by
// the host code (see docs/design/rocm_splitq.md, "Architecture contract").
//
//   int8 dot4: v_dot4_i32_iu8 on gfx11/gfx12, v_dot4_i32_i8 on gfx9/gfx10.3
//   decode, fast tier: splitq_decode_wmma_rdna3.cu (WMMA, gfx11)
//   prefill: splitq_prefill_rdna3.cu (WMMA, gfx11)
//   decode, portable: splitq_attn.cu decode_kernel (dot4, every target)
//
// The portable kernels are written for 32-lane logical waves: shuffles never
// cross 32 lanes, so they are correct on wave64 (CDNA) as well.

#pragma once

#include <cstdint>

namespace splitq {

// sum_i a.i8[i] * b.i8[i] + c, signed bytes on both sides.
__device__ __forceinline__ int dot4_i8(int a, int b, int c) {
#if defined(__GFX11__) || defined(__GFX12__)
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__) ||  \
    defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
    defined(__gfx1036__)
  return __builtin_amdgcn_sdot4(a, b, c, false);
#else
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    c += (int)(int8_t)(a >> (8 * i)) * (int)(int8_t)(b >> (8 * i));
  return c;
#endif
}

}  // namespace splitq
