// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 dequant primitives for RDNA3 (gfx1100/gfx1101/gfx1102), templated on
// the activation/scale dtype (half or __hip_bfloat16). The fp16 path reuses
// the classic exllamav2 bit-trick:
//
//   (qa & 0x000F000F) | 0x64006400  ->  half2(1024+q_lo, 1024+q_hi)
//   (qa & 0x00F000F0) | 0x64006400  ->  half2(1024+q_lo*16, 1024+q_hi*16)
//
// The "*16 then divide by 16 in the FMA" trick for the upper-nibble pairs
// works in fp16 because the mantissa (10 bits) is wide enough to hold a value
// shifted by 4 bits. In bf16 the mantissa is only 7 bits, so shifting an upper
// nibble into bits [7:4] would spill into the exponent. To avoid that, the
// bf16 path shifts each pair of nibbles down to bits [3:0]/[19:16] with a
// single right-shift before the OR with 0x43004300 (= bf162(128, 128)).

#ifndef _qdq_4_rdna3_cuh
#define _qdq_4_rdna3_cuh

#include <cstdint>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

namespace vllm {
namespace gptq_rdna3 {

using bf16_t = __hip_bfloat16;
using bf162_t = __hip_bfloat162;

// Bit-shuffle for an int32 holding 8 sequential 4-bit weights q[0..7]:
//   in:  q[7] q[6] q[5] q[4] q[3] q[2] q[1] q[0]   (LSB first)
//   out: q[7] q[5] q[3] q[1] q[6] q[4] q[2] q[0]   (even/odd interleaved)
//
// After shuffle, q[2k]   sits at bits [4k   : 4k+3]   (lower 16)
//                q[2k+1] sits at bits [16+4k: 16+4k+3] (upper 16)
// so a single mask 0x000F000F selects the matching even/odd pair, ready to
// bitcast to half2 / bfloat162 after OR-ing with the magic constant.
__forceinline__ __device__ void shuffle_4bit_8(uint32_t* q) {
  uint32_t qa = q[0];
  uint32_t qb = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t qa0 = qa & 0x0F;
    uint32_t qa1 = (qa & 0xF0) >> 4;
    qa >>= 8;
    qb |= (qa1 << (i * 4 + 16));
    qb |= (qa0 << (i * 4));
  }
  q[0] = qb;
}

// ---------------------------------------------------------------------------
// fp16 path: exact factored form.
// ---------------------------------------------------------------------------
// The pre-#54706-v2 kernels baked the zero/scale bias into every weight,
// which forced z = scale * (-1024 - zero) to be *stored* as fp16. At
// scale ~0.02 that costs ~0.008 abs per (group, column) constant and the
// error then accumulates along the K/groups axis (measured max abs
// error ~1.0 vs an FP32 dequantized reference at K=4096). Computing the
// product in fp32 does not help: the error is in the fp16 *representation*
// of z, not in the multiply.
//
// Instead use the identity, exact for any grouping:
//
//     sum_k a_k * w_k  =  y * sum_k a_k * (1024 + q_k)  +  z * sum_k a_k
//
// with one (y, z) pair per nibble slot (the high pairs carry a factor of
// 16): y_lo = s, z_lo = s*(-1024 - zero), y_hi = s/16, z_hi = s*(-64 -
// zero). The y/z constants live in fp32 and are never narrowed; the magic
// values (1024 + q) / (1024 + 16q) stay in fp16 where they are exact. The
// z * sum_a correction is applied once per 32-K round in the kernel (z is
// constant within a group, so per-round applications sum to the per-group
// application). sum_k a_k does not depend on the output column, so the
// extra v_dot2 against half2(1,1) is shared by all four columns a thread
// owns. This is the same structure the bf16 branches of q_gemm_rdna3.cu
// already use; fp16 needs two (y, z) pairs and two activation sums because
// its magic values carry the upper-nibble *16 trick.
// ---------------------------------------------------------------------------

// fp32 (y, z) pairs for the exact factored form above.
__forceinline__ __device__ void prep_zero_scale_fp16_f32(uint32_t zero,
                                                         half scale,
                                                         float (&y)[2],
                                                         float (&z)[2]) {
  const float s = __half2float(scale);
  const float zf = (float)(int)zero;
  y[0] = s;  // low pairs:  (1024 + q) * s
  z[0] = s * (-1024.0f - zf);
  y[1] = s * (1.0f / 16.0f);  // high pairs: (1024 + 16q) * s/16
  z[1] = s * (-64.0f - zf);
}

// Magic values only, with no y/z folded in: low pairs land in [0] and [2],
// high pairs in [1] and [3]. The lane mapping assumes the host-side
// gptq_shuffle interleave (see shuffle_4bit_8 above): even elements sit in
// the low nibbles, odd elements in the high nibbles.
__forceinline__ __device__ void magic_4bit_8_fp16(uint32_t qa, half2 (&qm)[4]) {
  const uint32_t c0 = 0x64006400;
  union {
    uint32_t u;
    half2 h2;
  } t;
  t.u = (qa & 0x000F000F) | c0;  // half2(1024 + q[0], 1024 + q[1])
  qm[0] = t.h2;
  t.u = (qa & 0x00F000F0) | c0;  // half2(1024 + q[2]*16, 1024 + q[3]*16)
  qm[1] = t.h2;
  const uint32_t qa_hi = qa >> 8;
  t.u = (qa_hi & 0x000F000F) | c0;  // half2(1024 + q[4], 1024 + q[5])
  qm[2] = t.h2;
  t.u = (qa_hi & 0x00F000F0) | c0;  // half2(1024 + q[6]*16, 1024 + q[7]*16)
  qm[3] = t.h2;
}

// half2(1.0, 1.0) for the activation-sum v_dot2 in the exact factored form.
__forceinline__ __device__ half2 ones_half2_fp16() {
  union {
    uint32_t u;
    half2 h2;
  } t;
  t.u = 0x3C003C00u;
  return t.h2;
}

// ---------------------------------------------------------------------------
// bf16 path
// ---------------------------------------------------------------------------

__forceinline__ __device__ void prep_zero_scale_bf16_f32(uint32_t zero,
                                                         bf16_t scale,
                                                         float& z_prep,
                                                         float& y_prep) {
  float scale_f = __bfloat162float(scale);
  z_prep = -(128.0f + (float)zero) * scale_f;
  y_prep = scale_f;
}

}  // namespace gptq_rdna3
}  // namespace vllm

#endif  // _qdq_4_rdna3_cuh
