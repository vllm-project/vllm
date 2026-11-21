// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Per-architecture primitives of the Octave kernels. Every AMD target vLLM
// builds compiles the portable tier; faster tiers are selected at runtime by
// the host code (see docs/design/rocm_octave.md, "Architecture contract").
//
//   int8 dot4: v_dot4_i32_iu8 on gfx11/gfx12, v_dot4_i32_i8 on gfx9/gfx10.3
//   decode, fast tier: octave_decode_wmma_rdna3.cu (WMMA, gfx1100, gfx1151)
//   prefill: octave_prefill_rdna3.cu (WMMA, gfx1100, gfx1151)
//   decode, portable: octave_attn.cu decode_kernel (dot4, every target)
//
// The portable kernels are written for 32-lane logical waves: shuffles never
// cross 32 lanes, so they are correct on wave64 (CDNA) as well.

#pragma once

#include <cstdint>
#include <string>

// Targets of the WMMA kernels: gfx1100 (RDNA3) and gfx1151 (RDNA3.5). Device
// passes for any other target compile those kernels as empty stubs.
#if defined(__gfx1100__) || defined(__gfx1151__)
  #define OCTAVE_WMMA_TARGET
#endif

namespace octave {

// Host twin of OCTAVE_WMMA_TARGET, on a gcnArchName. Keep the two lists equal:
// a chip matched here but not there would launch the empty stubs.
inline bool is_wmma_arch(const std::string& arch) {
  return arch.rfind("gfx1100", 0) == 0 || arch.rfind("gfx1151", 0) == 0;
}

// sum_i a.i8[i] * b.i8[i] + c, signed bytes on both sides.
__device__ __forceinline__ int dot4_i8(int a, int b, int c) {
#if defined(__GFX11__) || defined(__GFX12__)
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(__gfx942__)
  return __builtin_amdgcn_sdot4(a, b, c, false);
#else
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    c += (int)(int8_t)(a >> (8 * i)) * (int)(int8_t)(b >> (8 * i));
  return c;
#endif
}

}  // namespace octave
