// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MXFP4 dequant for RDNA3: E2M1 weight (mags {0,.5,1,1.5,2,3,4,6}) + E8M0
// block scale (2^(s8-127), group of 32), no zero-point. E2M1->{bf16,fp16}
// is a field copy (only 0 and the 0.5 subnormal are special); the E8M0
// scale folds in as an exponent add `bits += (s8-127)<<MANT_BITS`, no
// multiply. Pure-integer decode (host-testable without HIP) is gated below.

#ifndef _qdq_mxfp4_rdna3_cuh
#define _qdq_mxfp4_rdna3_cuh

#include <cstdint>

// Device-only HIP dependencies are gated so the pure-integer decode below can
// be #included and unit-tested by a plain host compiler.
#if defined(__HIPCC__)
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
  #define MXFP4_HD __host__ __device__ __forceinline__
  #define MXFP4_D __device__ __forceinline__
#else
  #define MXFP4_HD inline
#endif

namespace vllm {
namespace mxfp4_rdna3 {

#if defined(__HIPCC__)
using bf16_t = __hip_bfloat16;
using bf162_t = __hip_bfloat162;
#endif

// Pure-integer E2M1 -> float bits of the unscaled magnitude (no HIP types).
// bf16: bias 127, mant top bit 6. fp16: bias 15, mant top bit 9.
MXFP4_HD uint16_t mxfp4_e2m1_bits(uint32_t nib, uint32_t exp_bias,
                                  uint32_t mant_pos) {
  const uint32_t sign = (nib & 0x8u) << 12;
  const uint32_t em = nib & 0x7u;  // exp(2) | mant(1)
  const uint32_t e = em >> 1;
  const uint32_t m = em & 1u;
  // Normal: E = e-1+exp_bias, M = m at top. The 0.5 subnormal (e==0,m==1)
  // reuses that exponent with M forced to 0; the zero code clears to 0x0000.
  const uint32_t out_exp = (e + exp_bias - 1u) << (mant_pos + 1u);
  const uint32_t out_mant = (e != 0u ? m : 0u) << mant_pos;
  return (uint16_t)(em == 0u ? sign : (sign | out_exp | out_mant));
}

MXFP4_HD uint16_t mxfp4_e2m1_to_bf16_bits(uint32_t nib) {
  return mxfp4_e2m1_bits(nib, /*exp_bias=*/127u, /*mant_pos=*/6u);
}

MXFP4_HD uint16_t mxfp4_e2m1_to_fp16_bits(uint32_t nib) {
  return mxfp4_e2m1_bits(nib, /*exp_bias=*/15u, /*mant_pos=*/9u);
}

// Apply the E8M0 scale as an exponent add (zero stays zero).
MXFP4_HD uint16_t mxfp4_apply_e8m0_bits(uint16_t bits, int32_t bias_u16) {
  return (bits & 0x7FFFu) == 0u ? bits : (uint16_t)((int32_t)bits + bias_u16);
}

// E8M0 byte -> additive exponent bias. mant_bits = 7 (bf16) or 10 (fp16).
MXFP4_HD int32_t mxfp4_e8m0_bias(uint32_t s8, uint32_t mant_bits) {
  return ((int32_t)s8 - 127) << mant_bits;
}

#if defined(__HIPCC__)
MXFP4_D bf16_t mxfp4_nib_to_bf16(uint32_t nib, int32_t bias_u16) {
  union {
    uint16_t u;
    bf16_t b;
  } o;
  o.u = mxfp4_apply_e8m0_bits(mxfp4_e2m1_to_bf16_bits(nib), bias_u16);
  return o.b;
}

MXFP4_D half mxfp4_nib_to_fp16(uint32_t nib, int32_t bias_u16) {
  union {
    uint16_t u;
    half h;
  } o;
  o.u = mxfp4_apply_e8m0_bits(mxfp4_e2m1_to_fp16_bits(nib), bias_u16);
  return o.h;
}

// Unpack 8 codes from a packed uint32 (nibble i = K position k0+i, low nibble
// = even K). All 8 share one E8M0 block since K is tiled in multiples of 32.
MXFP4_D void dequant_mxfp4_8_bf16(uint32_t qa, int32_t bias_u16,
                                  bf16_t (&dq)[8]) {
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    dq[i] = mxfp4_nib_to_bf16((qa >> (4 * i)) & 0xFu, bias_u16);
  }
}

MXFP4_D void dequant_mxfp4_8_fp16(uint32_t qa, int32_t bias_u16,
                                  half (&dq)[8]) {
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    dq[i] = mxfp4_nib_to_fp16((qa >> (4 * i)) & 0xFu, bias_u16);
  }
}

// LUT decode: `lut` holds the 16 E2M1 magnitude bit-patterns for T (filled once
// from mxfp4_e2m1_to_{bf16,fp16}_bits), so per nibble we skip the arithmetic
// field construction and just apply the E8M0 exponent add. T = bf16_t or half.
template <typename T>
MXFP4_D void dequant_mxfp4_8_lut(uint32_t qa, int32_t bias_u16,
                                 const uint16_t* lut, T (&dq)[8]) {
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    uint16_t b = mxfp4_apply_e8m0_bits(lut[(qa >> (4 * i)) & 0xFu], bias_u16);
    __builtin_memcpy(&dq[i], &b, sizeof(T));
  }
}

// fp32 output (bf16 bits widened by a free <<16) for a scalar dot path.
MXFP4_D void dequant_mxfp4_8_f32(uint32_t qa, int32_t bias_u16,
                                 float (&dq)[8]) {
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    uint16_t b = mxfp4_apply_e8m0_bits(
        mxfp4_e2m1_to_bf16_bits((qa >> (4 * i)) & 0xFu), bias_u16);
    dq[i] = __uint_as_float((uint32_t)b << 16);
  }
}

#endif  // defined(__HIPCC__)

}  // namespace mxfp4_rdna3
}  // namespace vllm

#endif  // _qdq_mxfp4_rdna3_cuh
