#ifndef FP8_UTILS_H
#define FP8_UTILS_H

#include <immintrin.h>
#include <cstdint>
#include <cmath>
#include <cstring>

static inline __m512i _mm512_cvte5m2_fp16(__m256i a) {
  return _mm512_slli_epi16(_mm512_cvtepi8_epi16(a), 8);
}

static inline __m256i _mm256_cvte5m2_fp16(__m128i a) {
  return _mm256_slli_epi16(_mm256_cvtepi8_epi16(a), 8);
}

static inline __m256i _mm256_cvt2fp16_e5m2(__m256i a, __m256i b) {
  const __m512i vnaninf = _mm512_set1_epi16(0x7c00),
                vrneadd = _mm512_set1_epi16(0x007f);
  const __m512i vfixup = _mm512_set1_epi16(0x0001),
                vfixupmask = _mm512_set1_epi16(0x0100);
  /* b: lower half, a : upper half */
  const __m512i a_ = _mm512_inserti64x4(
      _mm512_inserti64x4(_mm512_setzero_si512(), b, 0), a, 1);
  const __mmask32 maska1_ = _mm512_cmp_epi16_mask(_mm512_and_si512(a_, vnaninf),
                                                  vnaninf, _MM_CMPINT_NE);
  const __mmask32 maska2_ = _mm512_cmp_epi16_mask(
      _mm512_and_si512(a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
  __m512i a_rne_ = _mm512_mask_add_epi16(
      a_, maska1_, a_,
      _mm512_mask_add_epi16(vrneadd, maska2_, vrneadd, vfixup));
  return _mm512_cvtepi16_epi8(_mm512_srli_epi16(a_rne_, 8));
}

static inline __m256i _mm256_cvt2fp16_e5m2_noINF(__m256i a, __m256i b) {
  const __m512i vnaninf = _mm512_set1_epi16(0x7c00);
  const __m512i vrneadd = _mm512_set1_epi16(0x007f);
  const __m512i vfixup = _mm512_set1_epi16(0x0001);
  const __m512i vfixupmask = _mm512_set1_epi16(0x0100);
  /* use a non-standard exponent offset = 16, */
  const __m512i vExp_fp16 = _mm512_set1_epi16(0x000F);
  const __m512i vExp_e5m2 = _mm512_set1_epi16(0x0010);
  const __m512i vsMant = _mm512_set1_epi16(0x83FF);
  /* Exponent Offset = 16, reclaim inf/NaN */
  const __m512i vsatuval =
      _mm512_set1_epi16(0x7F00); /* 2^15*1.11 a.k.a 57344.0, largest value */
  const __m512i vinfval = _mm512_set1_epi16(0x8000); /* -0.0 as INF */
  const __m512i a_ = _mm512_inserti64x4(
      _mm512_inserti64x4(_mm512_setzero_si512(), b, 0), a, 1);
  const __mmask32 maska1_ = _mm512_cmp_epi16_mask(_mm512_and_si512(a_, vnaninf),
                                                  vnaninf, _MM_CMPINT_NE);
  const __mmask32 maska2_ = _mm512_cmp_epi16_mask(
      _mm512_and_si512(a_, vfixupmask), vfixupmask, _MM_CMPINT_EQ);
  const __mmask32 maska3_ =
      _mm512_cmp_epi16_mask(_mm512_and_si512(a_, _mm512_set1_epi16(0x7FFF)),
                            vsatuval, _MM_CMPINT_NLE);
  __m512i vExp_ = _mm512_sub_epi16(
      _mm512_srli_epi16(_mm512_and_si512(a_, vnaninf), 10), vExp_fp16);
  vExp_ = _mm512_slli_epi16(_mm512_add_epi16(vExp_, vExp_e5m2), 10);
  __m512i a_rne_ = _mm512_or_si512(vExp_, _mm512_and_si512(a_, vsMant));
  a_rne_ = _mm512_mask_add_epi16(
      a_rne_, maska1_, a_rne_,
      _mm512_mask_add_epi16(vrneadd, maska2_, vrneadd, vfixup));
  a_rne_ = _mm512_mask_mov_epi16(
      a_rne_, maska3_,
      _mm512_or_si512(_mm512_and_si512(a_rne_, vinfval), vsatuval));
  a_rne_ = _mm512_mask_mov_epi16(a_rne_, ~maska1_, vinfval);
  return _mm512_cvtepi16_epi8(_mm512_srli_epi16(a_rne_, 8));
}

static inline __m512i _mm512_cvte5m2_noinf_fp16(__m256i a) {
  const __m512i vExp_fp16 = _mm512_set1_epi16(0x000F);
  const __m512i vExp_e5m2 = _mm512_set1_epi16(0x0010);
  const __m512i vsMant = _mm512_set1_epi16(0x83FF);
  const __m512i vnaninf = _mm512_set1_epi16(0x8000); /* -0.0 as INF */
  const __m512i vinfval = _mm512_set1_epi16(0x7c00);
  __m512i a_ = _mm512_slli_epi16(_mm512_cvtepi8_epi16(a), 8);
  const __mmask32 mask1_ = _mm512_cmp_epi16_mask(a_, vnaninf, _MM_CMPINT_EQ);
  __m512i vExp_ = _mm512_sub_epi16(
      _mm512_srli_epi16(_mm512_and_si512(a_, vinfval), 10), vExp_e5m2);
  vExp_ = _mm512_slli_epi16(_mm512_add_epi16(vExp_, vExp_fp16), 10);
  a_ = _mm512_or_si512(vExp_, _mm512_and_si512(a_, vsMant));
  return _mm512_mask_mov_epi16(a_, mask1_, vinfval);
}

static inline void cvt_fp16_e5m2_noINF_rne_intrinsic(
    const short* __restrict__ in, unsigned char* out, int size) {
#pragma omp parallel for
  for (int i = 0; i < size; i += 32) {
    __m256i bh_ = _mm256_lddqu_si256((__m256i*)&in[i]);
    __m256i ah_ = _mm256_lddqu_si256((__m256i*)&in[i + 16]);
    _mm256_storeu_si256((__m256i*)&out[i], _mm256_cvt2fp16_e5m2(ah_, bh_));
  }
}

static inline void cvt_fp32_e5m2_noinf_rne_intrinsic(
    const float* __restrict__ in, float* out, int size, float scale) {
#pragma omp parallel for
  for (int i = 0; i < size; i += 32) {
    __m512 b = _mm512_loadu_ps(&in[i]);
    __m512 a = _mm512_loadu_ps(&in[i + 16]);
    __m256i ah_ =
        _mm512_cvtps_ph(a, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m256i bh_ =
        _mm512_cvtps_ph(b, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m512i a_rne_ =
        _mm512_cvte5m2_noinf_fp16(_mm256_cvt2fp16_e5m2_noINF(ah_, bh_));
    bh_ = _mm512_extracti64x4_epi64(a_rne_, 0);
    ah_ = _mm512_extracti64x4_epi64(a_rne_, 1);
    b = _mm512_cvtph_ps(bh_);
    a = _mm512_cvtph_ps(ah_);
    _mm512_storeu_ps(&out[i], b);
    _mm512_storeu_ps(&out[i + 16], a);
  }
}

static inline __m512i cast_fp8x32_to_fp16x32(__m256i a) {
  return _mm512_cvte5m2_fp16(a);
}

static inline __m256i cast_fp16x16x2_to_fp8x32(__m256i a, __m256i b) {
  return _mm256_cvt2fp16_e5m2(a, b);
}

static inline void cast_fp8xn_to_fp16xn(const char* __restrict__ in,
                                        unsigned short* out, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i += 32) {
    __m256i a = _mm256_loadu_si256((const __m256i*)&in[i]);
    __m512i b = cast_fp8x32_to_fp16x32(a);
    __m512i* out_p = (__m512i*)(&out[i]);
    *out_p = b;
  }
}

static inline void cast_fp16xn_to_fp8xn(const short* __restrict__ in,
                                        unsigned char* out, int n) {
  cvt_fp16_e5m2_noINF_rne_intrinsic(in, out, n);
}

static inline void cast_fp32xn_to_fp8xn(const float* __restrict__ in,
                                        float* out, int n) {
  cvt_fp32_e5m2_noinf_rne_intrinsic(in, out, n, 0);
}

static inline uint8_t cast_bf16x1_to_fp8x1(int16_t bf16bits) {
  // Define the FP32 bias and the target FP8 bias
  const int fp16Bias = 127;
  const int fp8Bias = 15;
  uint8_t sign = (bf16bits >> 15) & 0x01;
  int8_t shift = (bf16bits >> 7) & 0xFF;
  if (shift == (int8_t)0xFF) {
    return (sign << 7) | 0x7F;
  }
  if (shift <= (int8_t)0x70 && shift >= (int8_t)0x91) {
    return (sign << 7);
  }

  int8_t exponent = shift - fp16Bias;
  uint16_t mantissa = bf16bits & 0x007F;

  // Adjust the exponent and mantissa for FP8
  exponent += fp8Bias;

  // Handle special cases and rounding (not shown for brevity)
  // Assemble the FP8 value (manual bit manipulation)
  uint8_t fp8 =
      (sign << 7) | ((exponent & 0x1F) << 2) | ((mantissa >> 5) & 0x03);

  return fp8;
}

static inline uint8_t cast_fp32x1_to_fp8x1(float fp32) {
  // Define the FP32 bias and the target FP8 bias
  const int fp32Bias = 127;
  const int fp8Bias = 15;

  // Use intrinsics to extract the bits from the FP32 value
  __m128 fp32Vector = _mm_set_ss(fp32);
  int fp32Bits =
      _mm_extract_ps(fp32Vector, 0);  // Extract the bits into an integer

  // Extract sign, exponent, and mantissa from FP32
  uint8_t sign = (fp32Bits >> 31) & 0x01;
  int8_t shift = (fp32Bits >> 23) & 0xFF;
  if (shift == (int8_t)0xFF) {
    return (sign << 7) | 0x7F;
  }
  if (shift <= (int8_t)0x70 && shift >= (int8_t)0x91) {
    return (sign << 7);
  }
  int8_t exponent = shift - fp32Bias;
  uint32_t mantissa = fp32Bits & 0x007FFFFF;

  // Adjust the exponent and mantissa for FP8
  exponent += fp8Bias;

  // Handle special cases and rounding (not shown for brevity)
  // Assemble the FP8 value (manual bit manipulation)
  uint8_t fp8 =
      (sign << 7) | ((exponent & 0x1F) << 2) | ((mantissa >> 21) & 0x03);

  return fp8;
}

static inline uint32_t cast_fp8x1_to_fp32x1(uint8_t fp8) {
  uint8_t sign = (fp8 >> 7) & 0x01;
  // Handle special cases (e.g., zero, infinity)
  if ((fp8 & 0x7C) == 0) {
    // Zero or subnormal (treated as zero)
    return sign ? -0.0f : 0.0f;
  } else if ((fp8 & 0x7C) == 0x7C) {
    // Infinity
    return sign ? -INFINITY : INFINITY;
  }

  // Define the FP8 bias and the target FP32 bias
  const int fp8Bias = 15;
  const int fp32Bias = 127;

  // Extract sign, exponent, and mantissa from FP8
  int exponent = ((fp8 >> 2) & 0x1F) - fp8Bias;
  uint8_t mantissa = fp8 & 0x03;

  // Adjust the exponent and mantissa for FP32
  exponent += fp32Bias;

  // Normalize the mantissa (the implicit leading 1 is added)
  uint32_t mantissaFP32 = static_cast<uint32_t>(mantissa) << 21;

  // Assemble the FP32 value
  uint32_t fp32Bits = (static_cast<uint32_t>(sign) << 31) |
                      (static_cast<uint32_t>(exponent) << 23) | mantissaFP32;

  return fp32Bits;
}

static inline float cast_fp8x1_to_fp32x1_f(uint8_t fp8) {
  uint32_t fp32_i = cast_fp8x1_to_fp32x1(fp8);
  float fp32 = *(float*)(&fp32_i);
  return fp32;
}

static inline __m256 cast_fp8x16_to_fp16x16(__m128 fp8x16) {
  return (__m256)_mm256_cvte5m2_fp16((__m128i)fp8x16);
}

static inline __m512 cast_fp8x16_to_fp32x16(__m128 fp8x16) {
  __m512 res{0};
  // fp8x16 -> fp16x16 -> fp32x16
  __m256 fp16x16 = cast_fp8x16_to_fp16x16(fp8x16);
  res = _mm512_cvtph_ps((__m256i)fp16x16);
  return res;
}

#endif
