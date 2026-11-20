// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MXFP4 dequant for RDNA4 (gfx12): E2M1 weight + E8M0 block scale (group 32,
// no zero point), decoded to **OCP FP8 E4M3** for the v_wmma_*_fp8_fp8
// pipeline.
//
// Differences from the RDNA3 header (qdq_mxfp4_rdna3.cuh), which targets bf16:
//
//  1. Target is 8-bit, so the sign shift is no longer a fixed `<<12`; it is a
//     parameter (4 for fp8, 12 for bf16/fp16).
//  2. The E8M0 scale can NOT be folded into the element. On bf16 the branch
//     applies it as an exponent add (`bits += (s8-127)<<7`), which works
//     because bf16 has an 8-bit exponent with the same 127 bias. E4M3 has a
//     *4-bit* exponent with bias 7 and a usable range of only 2^-6..2^8, so any
//     realistic E8M0 scale would overflow or flush to zero. Instead the scale
//     is returned as an fp32 multiplier and applied once per 32-group on the
//     WMMA accumulator (see mxfp4_e8m0_to_f32).
//
// E2M1 -> E4M3 is exact: the eight magnitudes {0,.5,1,1.5,2,3,4,6} are all
// representable in E4M3 (max normal 448), so the decode loses nothing.

#ifndef _qdq_mxfp4_rdna4_cuh
#define _qdq_mxfp4_rdna4_cuh

#include <cstdint>

#if defined(__HIPCC__)
  #define MXFP4_HD __host__ __device__ __forceinline__
  #define MXFP4_D __device__ __forceinline__
#else
  #define MXFP4_HD inline
#endif

namespace vllm {
namespace mxfp4_rdna4 {

// Pure-integer E2M1 -> target float bits. Generalises the RDNA3 helper by
// taking `sign_shift`, so the same arithmetic serves 8-bit and 16-bit targets.
//   fp8  E4M3: exp_bias=7,   mant_pos=2, sign_shift=4
//   bf16     : exp_bias=127, mant_pos=6, sign_shift=12
MXFP4_HD uint32_t mxfp4_e2m1_bits_g(uint32_t nib, uint32_t exp_bias,
                                    uint32_t mant_pos, uint32_t sign_shift) {
  const uint32_t sign = (nib & 0x8u) << sign_shift;
  const uint32_t em = nib & 0x7u;  // exp(2) | mant(1)
  const uint32_t e = em >> 1;
  const uint32_t m = em & 1u;
  // Normal: E = e-1+bias, M = m at the mantissa top. The 0.5 subnormal
  // (e==0,m==1) reuses that exponent with M forced to 0 -- for E4M3 that is
  // exponent field 6 (2^-1), still a normal number, so no subnormal handling
  // is needed. The zero code clears everything but the sign.
  const uint32_t out_exp = (e + exp_bias - 1u) << (mant_pos + 1u);
  const uint32_t out_mant = (e != 0u ? m : 0u) << mant_pos;
  return em == 0u ? sign : (sign | out_exp | out_mant);
}

// E2M1 nibble -> OCP E4M3 byte.
MXFP4_HD uint8_t mxfp4_e2m1_to_e4m3_bits(uint32_t nib) {
  return (uint8_t)mxfp4_e2m1_bits_g(nib, /*exp_bias=*/7u, /*mant_pos=*/2u,
                                    /*sign_shift=*/4u);
}

// E8M0 byte -> fp32 multiplier 2^(s8-127). An fp32 exponent field is 8 bits
// with bias 127, identical to E8M0, so this is a reinterpret, not a convert.
// s8==0 would denote 2^-127 (and 0xFF a NaN scale); neither is special-cased
// here, matching the RDNA3 path's documented limitation.
MXFP4_HD uint32_t mxfp4_e8m0_to_f32_bits(uint32_t s8) { return s8 << 23; }

#if defined(__HIPCC__)
MXFP4_D float mxfp4_e8m0_to_f32(uint32_t s8) {
  return __uint_as_float(mxfp4_e8m0_to_f32_bits(s8));
}

// Decode 4 E2M1 codes (low 16 bits of `x16`, nibble i = K position k0+i) into
// 4 packed E4M3 bytes via v_perm_b32. The 8 E2M1 magnitudes are exactly 8
// E4M3 bytes, so they fit v_perm's 8-byte {src0,src1} table and the 3-bit
// exp|mant field indexes it directly; the sign bit is OR'd in separately.
// This replaces ~16 ALU ops per nibble with ~9 per 4 nibbles; the scalar
// decode it replaces was ~60% of WMMA kernel time (measured by stubbing it).
// No scale is applied -- see the header note.
MXFP4_D uint32_t mxfp4_spread_nibbles(uint32_t x16) {
  // {n0,n1,n2,n3} at bits 0/4/8/12 -> one nibble per byte.
  const uint32_t y = (x16 | (x16 << 8)) & 0x00FF00FFu;
  return (y | (y << 4)) & 0x0F0F0F0Fu;
}

MXFP4_D uint32_t mxfp4_4nib_to_e4m3_perm(uint32_t x16) {
  constexpr uint32_t kLutLo = 0x3C383000u;  // em 0..3 -> src1 bytes 0..3
  constexpr uint32_t kLutHi = 0x4C484440u;  // em 4..7 -> src0 bytes 4..7
  const uint32_t z = mxfp4_spread_nibbles(x16);
  const uint32_t mag = __builtin_amdgcn_perm(kLutHi, kLutLo, z & 0x07070707u);
  return mag | ((z & 0x08080808u) << 4);
}

// E5M2 (bf8) weight operand with the E8M0 scale folded into the table: the
// eight E2M1 magnitudes scaled by 2^shift are exactly representable in E5M2
// while -13 <= shift <= 13 (all exponents stay normal), so one add shifts the
// whole 8-byte table (+4 per byte = +1 exponent step); code 0 is forced to 0.
// The caller takes shift = e8m0 - base[n] with a per-column base = max-13,
// and multiplies by 2^(base-127) once in the epilogue (FlyDSL's fold).
MXFP4_D void mxfp4_e5m2_tables(int shift, uint32_t& lo, uint32_t& hi) {
  const uint32_t off = (uint32_t)shift * 0x04040404u;
  lo = (0x3E3C3838u + off) & 0xFFFFFF00u;  // codes 0..3: 0, .5, 1, 1.5
  hi = 0x46444240u + off;                  // codes 4..7: 2, 3, 4, 6
}

MXFP4_D uint32_t mxfp4_4nib_perm(uint32_t x16, uint32_t lut_hi,
                                 uint32_t lut_lo) {
  const uint32_t z = mxfp4_spread_nibbles(x16);
  const uint32_t mag = __builtin_amdgcn_perm(lut_hi, lut_lo, z & 0x07070707u);
  return mag | ((z & 0x08080808u) << 4);
}

MXFP4_D void dequant_mxfp4_8_e5m2(uint32_t qa, uint32_t lut_lo, uint32_t lut_hi,
                                  uint32_t& lo, uint32_t& hi) {
  lo = mxfp4_4nib_perm(qa & 0xFFFFu, lut_hi, lut_lo);
  hi = mxfp4_4nib_perm(qa >> 16, lut_hi, lut_lo);
}

MXFP4_D uint32_t mxfp4_8nib_to_e4m3x4(uint32_t qa, int lo) {
  return mxfp4_4nib_to_e4m3_perm(lo ? (qa & 0xFFFFu) : (qa >> 16));
}

// Full 8-code unpack into the two 32-bit halves of a WMMA fp8 operand.
MXFP4_D void dequant_mxfp4_8_e4m3(uint32_t qa, uint32_t& lo, uint32_t& hi) {
  lo = mxfp4_8nib_to_e4m3x4(qa, 1);  // K+0..3
  hi = mxfp4_8nib_to_e4m3x4(qa, 0);  // K+4..7
}
#endif  // defined(__HIPCC__)

}  // namespace mxfp4_rdna4
}  // namespace vllm

#endif  // _qdq_mxfp4_rdna4_cuh
