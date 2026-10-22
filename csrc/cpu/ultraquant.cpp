// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// UltraQuant KV cache store for CPU. Slot layout matches
// vllm/v1/attention/ops/ultraquant/format.py byte for byte, so a cache written
// here is interchangeable with the GPU backend's.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "cpu_arch_macros.h"
#include "cpu_types.hpp"

namespace {

constexpr int kGroupSize = 32;
constexpr int kUE8M0Bias = 127;
constexpr float kDefaultConstantC = 0.156f;

// Midpoints between consecutive sorted FP4 levels, ascending. A value's sorted
// index is the count of midpoints below it; index 7 is zero, and the offset
// from 7 is already the E2M1 sign-magnitude code.
constexpr float kMidpoints[14] = {-5.0f,  -3.5f,  -2.5f, -1.75f, -1.25f,
                                  -0.75f, -0.25f, 0.25f, 0.75f,  1.25f,
                                  1.75f,  2.5f,   3.5f,  5.0f};

// FP4 E2M1 decode, indexed by the 4-bit code.
constexpr float kFp4Value[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                                 4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                                 -2.0f, -3.0f, -4.0f, -6.0f};

inline float ue8m0_scale(uint8_t byte) {
  return byte == 0 ? 0.0f : std::exp2(static_cast<float>(byte - kUE8M0Bias));
}

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512BF16__)
constexpr bool kPermutedNibbles = true;
#else
constexpr bool kPermutedNibbles = false;
#endif

// Element that decoded lane `i` of a group carries. See nibble_indices.
inline int64_t perm_elem(int64_t i) {
  if (!kPermutedNibbles) return i;
  return i < kGroupSize / 2 ? 2 * i : 2 * (i - kGroupSize / 2) + 1;
}

#if defined(__AVX512F__) && defined(__AVX512BW__) && \
    defined(__AVX512BF16__)

// The 16 FP4 levels as bf16 bit patterns, padded to a 32-entry VPERMW table.
alignas(64) constexpr uint16_t kFp4Bf16[32] = {
    0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
    0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0,
    0,      0,      0,      0,      0,      0,      0,      0,
    0,      0,      0,      0,      0,      0,      0,      0};

// 16 packed bytes -> 32 nibble indices, low nibbles then high nibbles. That
// yields the group's elements in the order 0,2,..,30,1,3,..,31 rather than
// 0..31, but costs four instructions fewer than interleaving them; Q and the
// accumulator are held in the same order so nothing has to be put back.
inline __m512i nibble_indices(const uint8_t* __restrict__ codes) {
  const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes));
  const __m128i m = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(b, m);
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m);
  return _mm512_cvtepu8_epi16(_mm256_set_m128i(hi, lo));
}

// A UE8M0 scale is a power of two, so applying it is an add on the bf16
// exponent field rather than a multiply. Zero codes are masked out so their
// exponent stays zero.
inline void decode_group(const uint8_t* __restrict__ codes, uint8_t byte,
                         __m512& lo, __m512& hi) {
  if (byte == 0) {
    lo = _mm512_setzero_ps();
    hi = _mm512_setzero_ps();
    return;
  }
  __m512i v = _mm512_permutexvar_epi16(
      nibble_indices(codes),
      _mm512_load_si512(reinterpret_cast<const void*>(kFp4Bf16)));
  const auto bump =
      static_cast<int16_t>((static_cast<int>(byte) - kUE8M0Bias) << 7);
  const __mmask32 nz = _mm512_test_epi16_mask(v, _mm512_set1_epi16(0x7FFF));
  v = _mm512_mask_add_epi16(v, nz, v, _mm512_set1_epi16(bump));

  // bf16 -> fp32 is a 16-bit left shift into the high half of each lane.
  lo = _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(v)), 16));
  hi = _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 1)), 16));
}

// Codes come out of the permute already in bf16 and every FP4 level times a
// power-of-two scale is exact there, so the QK dot uses them directly. The
// format casts Q through FP8 E4M3, whose 3 mantissa bits also survive bf16.
inline __m512i decode_group_bf16(const uint8_t* __restrict__ codes,
                                 uint8_t byte) {
  if (byte == 0) return _mm512_setzero_si512();
  const __m512i v = _mm512_permutexvar_epi16(
      nibble_indices(codes),
      _mm512_load_si512(reinterpret_cast<const void*>(kFp4Bf16)));
  const auto bump =
      static_cast<int16_t>((static_cast<int>(byte) - kUE8M0Bias) << 7);
  const __mmask32 nz = _mm512_test_epi16_mask(v, _mm512_set1_epi16(0x7FFF));
  return _mm512_mask_add_epi16(v, nz, v, _mm512_set1_epi16(bump));
}

// The lane count is a template parameter throughout: with a runtime trip count
// the compiler spills the accumulator arrays to the stack, which costs more
// than the arithmetic they hold.
constexpr int kMaxLanes = 8;

// HD is the head dim when it is known at compile time and 0 otherwise. Taking
// the group count from a runtime value costs more than twice what the loops
// below cost, so the common head dims are specialized, as in cpu_attn_impl.
constexpr int kSpecializedHeadDim = 128;

// Dequantized values are consumed straight out of registers; staging them in a
// buffer that the dot then reloads costs more than the dequant itself.
template <int NJ, int HD>
inline void qk_dots_lanes(const uint8_t* __restrict__ blk,
                          int64_t bsz_stride, int64_t codes_bytes,
                          const uint16_t* __restrict__ qbase, int64_t ntok,
                          int64_t head_dim_rt, float scale,
                          float* __restrict__ out_logits,
                          int64_t out_stride) {
  const int64_t head_dim = HD ? HD : head_dim_rt;
  const int64_t ngroup = head_dim / kGroupSize;
  for (int64_t n = 0; n < ntok; ++n) {
    const uint8_t* __restrict__ codes = blk + n * bsz_stride;
    const uint8_t* __restrict__ scales = codes + codes_bytes;
    // A horizontal reduction costs more than the FMAs feeding it, so the dot
    // stays in a vector accumulator and is reduced once per query head.
    __m512 dacc[NJ];
#pragma GCC unroll 8
    for (int j = 0; j < NJ; ++j) dacc[j] = _mm512_setzero_ps();
    for (int64_t g = 0; g < ngroup; ++g) {
      const __m512i k =
          decode_group_bf16(codes + g * (kGroupSize / 2), scales[g]);
      const uint16_t* qg = qbase + g * kGroupSize;
#pragma GCC unroll 8
      for (int j = 0; j < NJ; ++j) {
        const __m512i q = _mm512_loadu_si512(
            reinterpret_cast<const void*>(qg + j * head_dim));
        dacc[j] = _mm512_dpbf16_ps(dacc[j], reinterpret_cast<__m512bh>(k),
                                   reinterpret_cast<__m512bh>(q));
      }
    }
#pragma GCC unroll 8
    for (int j = 0; j < NJ; ++j)
      out_logits[j * out_stride + n] = _mm512_reduce_add_ps(dacc[j]) * scale;
  }
}

// The lane dispatch runs once per block, not once per token: a jump table and
// a call in the token loop cost more than the dot they guard.
template <int HD>
inline void qk_dots_block_hd(const uint8_t* __restrict__ blk,
                             int64_t bsz_stride, int64_t codes_bytes,
                             const uint16_t* __restrict__ qbase, int64_t ntok,
                             int64_t q_per_kv, int64_t head_dim, float scale,
                             float* __restrict__ out_logits,
                             int64_t out_stride) {
  int64_t j0 = 0;
  for (; j0 + kMaxLanes <= q_per_kv; j0 += kMaxLanes)
    qk_dots_lanes<kMaxLanes, HD>(blk, bsz_stride, codes_bytes,
                                 qbase + j0 * head_dim, ntok, head_dim, scale,
                                 out_logits + j0 * out_stride, out_stride);
#define UQ_QK_CASE(n)                                                        \
  case n:                                                                    \
    qk_dots_lanes<n, HD>(blk, bsz_stride, codes_bytes, qbase + j0 * head_dim, \
                         ntok, head_dim, scale,                              \
                         out_logits + j0 * out_stride, out_stride);          \
    break;
  switch (q_per_kv - j0) {
    case 0:
      break;
      UQ_QK_CASE(1)
      UQ_QK_CASE(2)
      UQ_QK_CASE(3)
      UQ_QK_CASE(4)
      UQ_QK_CASE(5)
      UQ_QK_CASE(6)
      UQ_QK_CASE(7)
  }
#undef UQ_QK_CASE
}

inline void qk_dots_block(const uint8_t* __restrict__ blk, int64_t bsz_stride,
                          int64_t codes_bytes,
                          const uint16_t* __restrict__ qbase, int64_t ntok,
                          int64_t q_per_kv, int64_t head_dim, float scale,
                          float* __restrict__ out_logits,
                          int64_t out_stride) {
  if (head_dim == kSpecializedHeadDim)
    qk_dots_block_hd<kSpecializedHeadDim>(blk, bsz_stride, codes_bytes, qbase,
                                          ntok, q_per_kv, head_dim, scale,
                                          out_logits, out_stride);
  else
    qk_dots_block_hd<0>(blk, bsz_stride, codes_bytes, qbase, ntok, q_per_kv,
                        head_dim, scale, out_logits, out_stride);
}

// The whole block runs under one set of accumulator registers, so the output
// is touched once per block instead of once per token. One KV head's block is
// block_size * slot_size bytes, already resident from the QK pass, which is
// what makes the strided walk over tokens cheap.
template <int NJ, int HD>
inline void pv_accum_lanes(const uint8_t* __restrict__ blk,
                           int64_t bsz_stride, int64_t v_codes_off,
                           int64_t v_scales_off,
                           const float* __restrict__ probs,
                           int64_t prob_stride, int64_t ntok,
                           int64_t head_dim_rt, float* __restrict__ acc) {
  const int64_t head_dim = HD ? HD : head_dim_rt;
  const int64_t ngroup = head_dim / kGroupSize;
  // Group outermost: the accumulator then lives in registers for the whole
  // block and is touched once, which beats reading each slot once even though
  // this walks the block once per group.
  for (int64_t g = 0; g < ngroup; ++g) {
    __m512 acc_lo[NJ], acc_hi[NJ];
#pragma GCC unroll 8
    for (int j = 0; j < NJ; ++j) {
      const float* a = acc + j * head_dim + g * kGroupSize;
      acc_lo[j] = _mm512_loadu_ps(a);
      acc_hi[j] = _mm512_loadu_ps(a + 16);
    }
    for (int64_t n = 0; n < ntok; ++n) {
      const uint8_t* slot = blk + n * bsz_stride;
      __m512 vlo, vhi;
      decode_group(slot + v_codes_off + g * (kGroupSize / 2),
                   slot[v_scales_off + g], vlo, vhi);
#pragma GCC unroll 8
      for (int j = 0; j < NJ; ++j) {
        const __m512 p = _mm512_set1_ps(probs[j * prob_stride + n]);
        acc_lo[j] = _mm512_fmadd_ps(p, vlo, acc_lo[j]);
        acc_hi[j] = _mm512_fmadd_ps(p, vhi, acc_hi[j]);
      }
    }
#pragma GCC unroll 8
    for (int j = 0; j < NJ; ++j) {
      float* a = acc + j * head_dim + g * kGroupSize;
      _mm512_storeu_ps(a, acc_lo[j]);
      _mm512_storeu_ps(a + 16, acc_hi[j]);
    }
  }
}

// The whole block runs under one set of accumulator registers, so the output
// is touched once per block instead of once per token. One KV head's block is
// block_size * slot_size bytes, already resident from the QK pass, which is
// what makes the strided walk over tokens cheap.
template <int HD>
inline void pv_accum_block_hd(const uint8_t* __restrict__ blk,
                              int64_t bsz_stride, int64_t v_codes_off,
                              int64_t v_scales_off,
                              const float* __restrict__ probs,
                              int64_t prob_stride, int64_t ntok,
                              int64_t q_per_kv, int64_t head_dim,
                              float* __restrict__ acc) {
  int64_t j0 = 0;
  for (; j0 + kMaxLanes <= q_per_kv; j0 += kMaxLanes)
    pv_accum_lanes<kMaxLanes, HD>(blk, bsz_stride, v_codes_off, v_scales_off,
                                  probs + j0 * prob_stride, prob_stride, ntok,
                                  head_dim, acc + j0 * head_dim);
#define UQ_PV_CASE(n)                                                     \
  case n:                                                                 \
    pv_accum_lanes<n, HD>(blk, bsz_stride, v_codes_off, v_scales_off,     \
                          probs + j0 * prob_stride, prob_stride, ntok,    \
                          head_dim, acc + j0 * head_dim);                 \
    break;
  switch (q_per_kv - j0) {
    case 0:
      break;
      UQ_PV_CASE(1)
      UQ_PV_CASE(2)
      UQ_PV_CASE(3)
      UQ_PV_CASE(4)
      UQ_PV_CASE(5)
      UQ_PV_CASE(6)
      UQ_PV_CASE(7)
  }
#undef UQ_PV_CASE
}

inline void pv_accum_block(const uint8_t* __restrict__ blk, int64_t bsz_stride,
                           int64_t v_codes_off, int64_t v_scales_off,
                           const float* __restrict__ probs,
                           int64_t prob_stride, int64_t ntok, int64_t q_per_kv,
                           int64_t head_dim, float* __restrict__ acc) {
  if (head_dim == kSpecializedHeadDim)
    pv_accum_block_hd<kSpecializedHeadDim>(blk, bsz_stride, v_codes_off,
                                           v_scales_off, probs, prob_stride,
                                           ntok, q_per_kv, head_dim, acc);
  else
    pv_accum_block_hd<0>(blk, bsz_stride, v_codes_off, v_scales_off, probs,
                         prob_stride, ntok, q_per_kv, head_dim, acc);
}

#else

inline float bf16_to_f32(uint16_t b) {
  const uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

inline void qk_dots_block(const uint8_t* __restrict__ blk, int64_t bsz_stride,
                          int64_t codes_bytes,
                          const uint16_t* __restrict__ qbase, int64_t ntok,
                          int64_t q_per_kv, int64_t head_dim, float scale,
                          float* __restrict__ out_logits,
                          int64_t out_stride) {
  for (int64_t n = 0; n < ntok; ++n) {
    const uint8_t* codes = blk + n * bsz_stride;
    const uint8_t* scales = codes + codes_bytes;
    for (int64_t j = 0; j < q_per_kv; ++j) {
      const uint16_t* q = qbase + j * head_dim;
      float dot = 0.0f;
      for (int64_t g = 0; g < head_dim / kGroupSize; ++g) {
        const float s = ue8m0_scale(scales[g]);
        const uint8_t* cp = codes + g * (kGroupSize / 2);
        const uint16_t* qg = q + g * kGroupSize;
        for (int i = 0; i < kGroupSize / 2; ++i) {
          dot += kFp4Value[cp[i] & 0x0F] * s * bf16_to_f32(qg[2 * i]);
          dot += kFp4Value[cp[i] >> 4] * s * bf16_to_f32(qg[2 * i + 1]);
        }
      }
      out_logits[j * out_stride + n] = dot * scale;
    }
  }
}

inline void pv_accum_block(const uint8_t* __restrict__ blk, int64_t bsz_stride,
                           int64_t v_codes_off, int64_t v_scales_off,
                           const float* __restrict__ probs,
                           int64_t prob_stride, int64_t ntok, int64_t q_per_kv,
                           int64_t head_dim, float* __restrict__ acc) {
  for (int64_t n = 0; n < ntok; ++n) {
    const uint8_t* slot = blk + n * bsz_stride;
    for (int64_t g = 0; g < head_dim / kGroupSize; ++g) {
      const float s = ue8m0_scale(slot[v_scales_off + g]);
      const uint8_t* cp = slot + v_codes_off + g * (kGroupSize / 2);
      for (int i = 0; i < kGroupSize / 2; ++i) {
        const float v0 = kFp4Value[cp[i] & 0x0F] * s;
        const float v1 = kFp4Value[cp[i] >> 4] * s;
        for (int64_t j = 0; j < q_per_kv; ++j) {
          const float p = probs[j * prob_stride + n];
          float* a = acc + j * head_dim + g * kGroupSize;
          a[2 * i] += p * v0;
          a[2 * i + 1] += p * v1;
        }
      }
    }
  }
}

#endif

// Round-half-up on the log, matching reference.py's floor(log2(s) + 0.5). A
// zero or non-finite group encodes as byte 0, which dequantizes to zero.
inline uint8_t ue8m0_byte(float s_raw, float& s_recip) {
  if (!(s_raw > 0.0f) || !std::isfinite(s_raw)) {
    s_recip = 0.0f;
    return 0;
  }
  int exp = static_cast<int>(std::floor(std::log2(s_raw) + 0.5f));
  exp = std::max(-126, std::min(127, exp));
  s_recip = std::exp2(static_cast<float>(-exp));
  return static_cast<uint8_t>(exp + kUE8M0Bias);
}

#if defined(__AVX512F__)

// Sorted position of each lane in the FP4 grid as the count of midpoints below
// it, re-expressed as the sign-magnitude E2M1 code. One compare per 16 lanes
// replaces the 14 the scalar path spends per element.
inline __m512i snap_fp4(__m512 xn) {
  const __m512i one = _mm512_set1_epi32(1);
  __m512i idx = _mm512_setzero_si512();
#pragma GCC unroll 14
  for (int i = 0; i < 14; ++i) {
    const __mmask16 gt =
        _mm512_cmp_ps_mask(xn, _mm512_set1_ps(kMidpoints[i]), _CMP_GT_OQ);
    idx = _mm512_mask_add_epi32(idx, gt, idx, one);
  }
  const __m512i zero = _mm512_setzero_si512();
  const __m512i d = _mm512_sub_epi32(idx, _mm512_set1_epi32(7));
  const __mmask16 neg = _mm512_cmplt_epi32_mask(d, zero);
  const __m512i mag = _mm512_mask_sub_epi32(d, neg, zero, d);
  return _mm512_and_si512(
      _mm512_mask_or_epi32(mag, neg, mag, _mm512_set1_epi32(8)),
      _mm512_set1_epi32(0x0F));
}

#endif

#if defined(__AVX512F__)

// A group of 32 as two fp32 vectors. Widening bf16 in register is exact, so
// the caller can hand over the model's own tensor instead of an fp32 copy.
inline void load_group(const float* __restrict__ x, __m512& a, __m512& b) {
  a = _mm512_loadu_ps(x);
  b = _mm512_loadu_ps(x + 16);
}

inline void load_group(const c10::BFloat16* __restrict__ x, __m512& a,
                       __m512& b) {
  const __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(x));
  a = _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(v)), 16));
  b = _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(v, 1)), 16));
}

#endif

// Encodes one group of 32 values into 16 packed bytes plus one scale byte.
// Even elements land in the low nibble, odd in the high nibble.
template <typename T>
inline uint8_t encode_group(const T* __restrict__ x,
                            uint8_t* __restrict__ codes, float constant_c) {
#if defined(__AVX512F__)
  alignas(64) static constexpr int32_t kEven[16] = {0,  2,  4,  6,  8,  10,
                                                    12, 14, 16, 18, 20, 22,
                                                    24, 26, 28, 30};
  alignas(64) static constexpr int32_t kOdd[16] = {1,  3,  5,  7,  9,  11,
                                                   13, 15, 17, 19, 21, 23,
                                                   25, 27, 29, 31};
  __m512 a, b;
  load_group(x, a, b);
  const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
  const float amax = _mm512_reduce_max_ps(
      _mm512_max_ps(_mm512_and_ps(a, absmask), _mm512_and_ps(b, absmask)));

  float s_recip;
  const uint8_t byte = ue8m0_byte(constant_c * amax, s_recip);

  // No early-out on a zero scale: the scalar path still runs the compares, and
  // a non-finite group has to reach the same 0xF codes it produces there.
  const __m512 r = _mm512_set1_ps(s_recip);
  const __m512i lo = snap_fp4(
      _mm512_mul_ps(_mm512_permutex2var_ps(a, _mm512_load_si512(kEven), b), r));
  const __m512i hi = snap_fp4(
      _mm512_mul_ps(_mm512_permutex2var_ps(a, _mm512_load_si512(kOdd), b), r));
  _mm_storeu_si128(
      reinterpret_cast<__m128i*>(codes),
      _mm512_cvtepi32_epi8(_mm512_or_si512(lo, _mm512_slli_epi32(hi, 4))));
  return byte;
#else
  float amax = 0.0f;
  for (int i = 0; i < kGroupSize; ++i)
    amax = std::max(amax, std::fabs(static_cast<float>(x[i])));

  float s_recip;
  const uint8_t byte = ue8m0_byte(constant_c * amax, s_recip);

  for (int i = 0; i < kGroupSize; i += 2) {
    uint8_t lo = 0, hi = 0;
    for (int half = 0; half < 2; ++half) {
      const float xn = static_cast<float>(x[i + half]) * s_recip;
      int idx = 0;
      for (const float m : kMidpoints) idx += (xn > m);
      const int d = idx - 7;
      const uint8_t code =
          static_cast<uint8_t>(d < 0 ? (8 | (-d)) : d) & uint8_t{0x0F};
      (half == 0 ? lo : hi) = code;
    }
    codes[i >> 1] = static_cast<uint8_t>(lo | (hi << 4));
  }
  return byte;
#endif
}

// Online softmax over one block, rescaling the running accumulator whenever
// the max moves. DEFINE_FAST_EXP lives here rather than around the caller:
// its constants would otherwise stay live across the dots and spill them.
inline void apply_block_softmax(const float* __restrict__ logits,
                                float* __restrict__ prob,
                                float* __restrict__ acc, float* __restrict__ m,
                                float* __restrict__ l, int64_t q_per_kv,
                                int64_t ntok, int64_t block_size,
                                int64_t head_dim) {
#if defined(DEFINE_FAST_EXP)
  DEFINE_FAST_EXP
#endif
  for (int64_t j = 0; j < q_per_kv; ++j) {
    const float* lg = logits + j * block_size;
    float bmax = -std::numeric_limits<float>::infinity();
    for (int64_t n = 0; n < ntok; ++n) bmax = std::max(bmax, lg[n]);

    const float m_new = std::max(m[j], bmax);
    const float corr = std::exp(m[j] - m_new);
    if (corr != 1.0f) {
      float* a = acc + j * head_dim;
#pragma omp simd
      for (int64_t d = 0; d < head_dim; ++d) a[d] *= corr;
      l[j] *= corr;
    }
    float* pr = prob + j * block_size;
    float sum = 0.0f;
    int64_t n = 0;
#if defined(DEFINE_FAST_EXP)
    const vec_op::FP32Vec16 mv(m_new);
    vec_op::FP32Vec16 sum_v(0.0f);
    for (; n + 16 <= ntok; n += 16) {
      const vec_op::FP32Vec16 e = fast_exp(vec_op::FP32Vec16(lg + n) - mv);
      e.save(pr + n);
      sum_v = sum_v + e;
    }
    sum = sum_v.reduce_sum();
#endif
    for (; n < ntok; ++n) {
      pr[n] = std::exp(lg[n] - m_new);
      sum += pr[n];
    }
    l[j] += sum;
    m[j] = m_new;
  }
}

template <typename T>
void store_slots(const T* __restrict__ kp, const T* __restrict__ vp,
                 uint8_t* __restrict__ cache,
                 const int64_t* __restrict__ slots, int64_t num_tokens,
                 int64_t num_kv_heads, int64_t head_dim, int64_t num_groups,
                 int64_t codes_bytes, int64_t block_size, int64_t block_stride,
                 int64_t head_stride, int64_t bsz_stride, float c) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t h = 0; h < num_kv_heads; ++h) {
      const int64_t slot = slots[t];
      if (slot < 0) continue;

      uint8_t* dst = cache + (slot / block_size) * block_stride +
                     (slot % block_size) * bsz_stride + h * head_stride;
      uint8_t* k_scales = dst + codes_bytes;
      uint8_t* v_codes = k_scales + num_groups;
      uint8_t* v_scales = v_codes + codes_bytes;

      const T* krow = kp + (t * num_kv_heads + h) * head_dim;
      const T* vrow = vp + (t * num_kv_heads + h) * head_dim;
      for (int64_t g = 0; g < num_groups; ++g) {
        const int64_t off = g * kGroupSize;
        k_scales[g] = encode_group(krow + off, dst + off / 2, c);
        v_scales[g] = encode_group(vrow + off, v_codes + off / 2, c);
      }
    }
  }
}

}  // namespace

void ultraquant_store(const torch::Tensor& key,    // [num_tokens, Hk, D]
                      const torch::Tensor& value,  // [num_tokens, Hk, D]
                      torch::Tensor& kv_cache,  // [blocks, Hk, bsz, slot_size]
                      const torch::Tensor& slot_mapping,
                      const torch::Tensor& rotation,  // [D, D], applied to K
                      const double constant_c) {
  TORCH_CHECK_EQ(key.dim(), 3);
  TORCH_CHECK_EQ(value.dim(), 3);
  TORCH_CHECK_EQ(kv_cache.dim(), 4);
  TORCH_CHECK(kv_cache.scalar_type() == at::ScalarType::Byte,
              "kv_cache must be uint8 for the UltraQuant path");

  TORCH_CHECK_EQ(slot_mapping.numel(), key.size(0));
  TORCH_CHECK(key.sizes() == value.sizes(), "key and value must agree in shape");
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long,
              "slot_mapping must be int64");

  const int64_t num_tokens = key.size(0);
  const int64_t num_kv_heads = key.size(1);
  const int64_t head_dim = key.size(2);
  TORCH_CHECK(head_dim % kGroupSize == 0, "head_dim must be a multiple of ",
              kGroupSize, ", got ", head_dim);

  const int64_t num_groups = head_dim / kGroupSize;
  const int64_t codes_bytes = head_dim / 2;
  const int64_t slot_size = 2 * (codes_bytes + num_groups);
  TORCH_CHECK_EQ(kv_cache.size(3), slot_size);
  TORCH_CHECK_EQ(kv_cache.size(1), num_kv_heads);

  // A caller slicing one request's slots out of a batch hands us a strided
  // view, and the loop below indexes raw.
  const torch::Tensor slot_map = slot_mapping.contiguous();

  uint8_t* __restrict__ cache = kv_cache.data_ptr<uint8_t>();
  const int64_t* __restrict__ slots = slot_map.data_ptr<int64_t>();

  const int64_t block_size = kv_cache.size(2);
  const int64_t block_stride = kv_cache.stride(0);
  const int64_t head_stride = kv_cache.stride(1);
  const int64_t bsz_stride = kv_cache.stride(2);
  const float c = static_cast<float>(constant_c);

  // One oneDNN GEMM beats a per-token butterfly here by ~4x at prefill shapes,
  // and the rotation is only defined on K. Staying in bf16 end to end keeps
  // the GEMM on the bf16 units and spares two full-size fp32 buffers; the
  // encoder widens in register. Rounding the rotation to bf16 costs 1.7e-3,
  // which moves the format's own 0.336 reconstruction error to 0.337.
  const bool bf16_path = key.scalar_type() == at::ScalarType::BFloat16 &&
                         value.scalar_type() == at::ScalarType::BFloat16;
  if (bf16_path) {
    const torch::Tensor k_rot =
        at::matmul(key, rotation.to(at::kBFloat16)).contiguous();
    const torch::Tensor v_src = value.contiguous();
    store_slots(k_rot.data_ptr<c10::BFloat16>(),
                v_src.data_ptr<c10::BFloat16>(), cache, slots, num_tokens,
                num_kv_heads, head_dim, num_groups, codes_bytes, block_size,
                block_stride, head_stride, bsz_stride, c);
  } else {
    const torch::Tensor k_rot =
        at::matmul(key.to(at::kFloat), rotation.to(at::kFloat)).contiguous();
    const torch::Tensor v_src = value.to(at::kFloat).contiguous();
    store_slots(k_rot.data_ptr<float>(), v_src.data_ptr<float>(), cache, slots,
                num_tokens, num_kv_heads, head_dim, num_groups, codes_bytes,
                block_size, block_stride, head_stride, bsz_stride, c);
  }
}

void ultraquant_decode_attention(
    const torch::Tensor& query,     // [num_seqs, Hq, D] fp32, pre-rotated
    // Logical [blocks, Hk, block_size, slot_size] u8, as the KV cache views
    // are shaped; the physical order is carried by the strides.
    const torch::Tensor& kv_cache,
    const torch::Tensor& block_table,  // [num_seqs, max_blocks] int32
    const torch::Tensor& seq_lens,     // [num_seqs] int32
    torch::Tensor& out,                // [num_seqs, Hq, D] fp32
    const double scale, const std::optional<torch::Tensor>& sinks) {
  TORCH_CHECK_EQ(query.dim(), 3);
  TORCH_CHECK_EQ(kv_cache.dim(), 4);
  TORCH_CHECK(kv_cache.scalar_type() == at::ScalarType::Byte,
              "kv_cache must be uint8 for the UltraQuant path");
  const bool q_is_bf16 = query.scalar_type() == at::ScalarType::BFloat16;
  TORCH_CHECK(q_is_bf16 || query.scalar_type() == at::ScalarType::Float,
              "query must be fp32 or bf16");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Float, "out must be fp32");

  const int64_t num_seqs = query.size(0);
  const int64_t num_q_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  const int64_t num_kv_heads = kv_cache.size(1);
  TORCH_CHECK(num_q_heads % num_kv_heads == 0,
              "num_q_heads must be a multiple of num_kv_heads");
  TORCH_CHECK(head_dim % kGroupSize == 0);

  const int64_t q_per_kv = num_q_heads / num_kv_heads;
  const int64_t num_groups = head_dim / kGroupSize;
  const int64_t codes_bytes = head_dim / 2;
  TORCH_CHECK_EQ(kv_cache.size(3), 2 * (codes_bytes + num_groups));

  const int64_t block_size = kv_cache.size(2);
  const int64_t block_stride = kv_cache.stride(0);
  const int64_t head_stride = kv_cache.stride(1);
  const int64_t bsz_stride = kv_cache.stride(2);

  const torch::Tensor q = query.contiguous();
  const torch::Tensor bt = block_table.to(at::kInt).contiguous();
  const torch::Tensor sl = seq_lens.to(at::kInt).contiguous();
  const int64_t bt_stride = bt.stride(0);

  // A bf16 query needs no conversion at all: the dots run in bf16, so the
  // kernel reads the caller's rows where it would otherwise round its own.
  const float* __restrict__ qp =
      q_is_bf16 ? nullptr : q.data_ptr<float>();
  const uint16_t* __restrict__ qbf =
      q_is_bf16 ? reinterpret_cast<const uint16_t*>(q.data_ptr<c10::BFloat16>())
                : nullptr;
  const uint8_t* __restrict__ cache = kv_cache.data_ptr<uint8_t>();
  const int32_t* __restrict__ btp = bt.data_ptr<int32_t>();
  const int32_t* __restrict__ slp = sl.data_ptr<int32_t>();
  float* __restrict__ outp = out.data_ptr<float>();
  // Named, because to()/contiguous() may allocate and the pointer has to
  // outlive the statement.
  const torch::Tensor sinks_f =
      sinks.has_value() ? sinks->to(at::kFloat).contiguous() : torch::Tensor();
  const float* __restrict__ sinkp =
      sinks.has_value() ? sinks_f.data_ptr<float>() : nullptr;
  const float sc = static_cast<float>(scale);

#pragma omp parallel
  {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<float> acc(q_per_kv * head_dim);
    std::vector<uint16_t> qbf16(q_per_kv * head_dim);
    std::vector<float> logits(q_per_kv * block_size);
    std::vector<float> prob(q_per_kv * block_size);
    std::vector<float> m(q_per_kv), l(q_per_kv);

    // Keyed on KV head, not query head: the GQA group shares one slot, so
    // dequantizing per query head would repeat the work q_per_kv times.
#pragma omp for collapse(2) schedule(static)
    for (int64_t s = 0; s < num_seqs; ++s) {
      for (int64_t kvh = 0; kvh < num_kv_heads; ++kvh) {
        const int64_t ctx = slp[s];
        const int64_t h0 = kvh * q_per_kv;
        float* dst = outp + (s * num_q_heads + h0) * head_dim;
        if (ctx <= 0) {
          std::fill(dst, dst + q_per_kv * head_dim, 0.0f);
          continue;
        }

        // Lay Q out in the order the dequant emits, once per KV head. fp32
        // rounds to bf16 on the way; the format feeds Q through FP8 E4M3, so
        // that conversion is lossless.
        const int64_t qoff = (s * num_q_heads + h0) * head_dim;
        for (int64_t r = 0; r < q_per_kv * num_groups; ++r) {
          const int64_t gb = r * kGroupSize;
          for (int64_t i = 0; i < kGroupSize; ++i) {
            const int64_t src = qoff + gb + perm_elem(i);
            if (q_is_bf16) {
              qbf16[gb + i] = qbf[src];
            } else {
              uint32_t bits;
              std::memcpy(&bits, &qp[src], sizeof(bits));
              qbf16[gb + i] = static_cast<uint16_t>(
                  (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
            }
          }
        }
        const uint16_t* qbase = qbf16.data();
        std::fill(acc.begin(), acc.end(), 0.0f);
        std::fill(m.begin(), m.end(), neg_inf);
        std::fill(l.begin(), l.end(), 0.0f);

        // Block at a time: one block for one KV head is block_size * slot
        // bytes, so the K pass leaves V in L1 for the second pass.
        for (int64_t base = 0; base < ctx; base += block_size) {
          const int64_t ntok = std::min(block_size, ctx - base);
          const uint8_t* blk =
              cache + btp[s * bt_stride + base / block_size] * block_stride +
              kvh * head_stride;

          qk_dots_block(blk, bsz_stride, codes_bytes, qbase, ntok, q_per_kv,
                        head_dim, sc, logits.data(), block_size);

          apply_block_softmax(logits.data(), prob.data(), acc.data(), m.data(),
                              l.data(), q_per_kv, ntok, block_size, head_dim);

          pv_accum_block(blk, bsz_stride, codes_bytes + num_groups,
                         2 * codes_bytes + num_groups, prob.data(), block_size,
                         ntok, q_per_kv, head_dim, acc.data());
        }

        for (int64_t j = 0; j < q_per_kv; ++j) {
          float* a = acc.data() + j * head_dim;
          // A sink joins the denominator and the max but contributes no value.
          if (sinkp != nullptr) {
            const float m_new = std::max(m[j], sinkp[h0 + j]);
            const float corr = std::exp(m[j] - m_new);
            for (int64_t d = 0; d < head_dim; ++d) a[d] *= corr;
            l[j] = l[j] * corr + std::exp(sinkp[h0 + j] - m_new);
          }
          const float inv = l[j] > 0.0f ? 1.0f / l[j] : 0.0f;
          float* o = dst + j * head_dim;
          for (int64_t g = 0; g < num_groups; ++g) {
            const int64_t gb = g * kGroupSize;
            for (int64_t i = 0; i < kGroupSize; ++i)
              o[gb + perm_elem(i)] = a[gb + i] * inv;
          }
        }
      }
    }
  }
}
