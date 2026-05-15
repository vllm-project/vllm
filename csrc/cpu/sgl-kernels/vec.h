// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#pragma once

#if defined(__AVX512F__) && defined(__AVX512BF16__) && defined(__AMX_BF16__)
#define CPU_CAPABILITY_AVX512
#endif

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

namespace {

using namespace at::vec;

template <typename scalar_t, typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 0>
inline Vectorized<scalar_t> convert_from_float_ext(const Vectorized<float>& a, const Vectorized<float>& b) {
  return at::vec::convert_from_float<scalar_t>(a, b);
}

// allow f16, bf16
template <typename scalar_t, typename std::enable_if_t<is_reduced_floating_point_v<scalar_t>, int> = 1>
inline std::tuple<Vectorized<float>, Vectorized<float>> load_float_vec2(const scalar_t* __restrict__ data) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  bVec x_vec = bVec::loadu(data);
  fVec x0, x1;
  std::tie(x0, x1) = at::vec::convert_to_float(x_vec);
  return std::make_tuple(x0, x1);
}

// allow  f32
inline std::tuple<Vectorized<float>, Vectorized<float>> load_float_vec2(const float* __restrict__ data) {
  using fVec = at::vec::Vectorized<float>;
  fVec x0 = fVec::loadu(data);
  fVec x1 = fVec::loadu(data + fVec::size());
  return std::make_tuple(x0, x1);
}

#if defined(CPU_CAPABILITY_AVX512)

// `at::vec::convert_from_float<>` from PyTorch doesn't have avx512-bf16 intrinsics
// use native instruction for bfloat16->float32 conversion
template <>
inline Vectorized<at::BFloat16>
convert_from_float_ext<at::BFloat16>(const Vectorized<float>& a, const Vectorized<float>& b) {
  return (__m512i)(_mm512_cvtne2ps_pbh(__m512(b), __m512(a)));
}

#define CVT_BF16_TO_FP32(a) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16))

#define CVT_FP16_TO_FP32(a) _mm512_cvtph_ps(a)

// this doesn't handle NaN.
inline __m512bh cvt_e4m3_bf16_intrinsic_no_nan(__m256i fp8_vec) {
  const __m512i x = _mm512_cvtepu8_epi16(fp8_vec);
  __m512i combined = _mm512_add_epi16(x, _mm512_set1_epi16(0x0780));
  combined = _mm512_slli_epi16(combined, 4);
  combined = _mm512_and_si512(combined, _mm512_set1_epi16(0x87f0));
  combined = _mm512_add_epi16(combined, _mm512_set1_epi16(0x3c00));

  const __mmask32 is_nonzero = _mm512_cmpneq_epi16_mask(x, _mm512_setzero_si512());
  return (__m512bh)_mm512_maskz_mov_epi16(is_nonzero, combined);
}

inline __m512bh cvt_e4m3_bf16_intrinsic_without_denorm(__m256i fp8_vec) {
  // The following conversion is without denorm behavior, that is to say,
  //   Max subnorm   : S.0000.111 = 0.875 ∗ 2**(−6)
  //   Min subnorm   : S.0000.001 = 2**(−9)
  // 0.0019 ~ 0.0137 cannot be converted correctly.
  __m512i x = _mm512_cvtepu8_epi16(fp8_vec);
  auto mask = _mm512_cmpneq_epi16_mask(
      _mm512_and_si512(x, _mm512_set1_epi16(127)),
      _mm512_setzero_si512());  // mask = x & 0x7f
  auto mask_nan = _mm512_cmpneq_epi16_mask(
      _mm512_and_si512(x, _mm512_set1_epi16(127)),
      _mm512_set1_epi16(127));                                                      // mask_nan = x & 0x7f
  auto mantissa = _mm512_slli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(7)), 4);  // mantissa = (x & 7) << 4
  auto exponent = _mm512_add_epi16(
      _mm512_srli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(120)), 3),
      _mm512_set1_epi16(120));  // exponent = (((x >> 3) & 15) + 120)
  auto nonsign = _mm512_maskz_mov_epi16(mask, _mm512_or_si512(mantissa, _mm512_slli_epi16(exponent, 7)));
  nonsign = _mm512_mask_mov_epi16(_mm512_set1_epi16(0x7fff), mask_nan, nonsign);  // deal with Nan
  return (__m512bh)(_mm512_or_si512(
      nonsign,
      _mm512_slli_epi16(
          _mm512_and_si512(x, _mm512_set1_epi16(128)),
          8)));  // add sign (x & 128) << 8
}

inline __m512bh cvt_e4m3_bf16_intrinsic_with_denorm(__m256i fp8_vec) {
  __m512i x = _mm512_cvtepu8_epi16(fp8_vec);
  __m512i lg2mant = _mm512_mask_mov_epi16(
      _mm512_mask_mov_epi16(
          _mm512_setzero_si512(), _mm512_test_epi16_mask(x, _mm512_set1_epi16(2)), _mm512_set1_epi16(1)),
      _mm512_test_epi16_mask(x, _mm512_set1_epi16(4)),
      _mm512_set1_epi16(2));
  return (__m512bh)(_mm512_or_si512(
      _mm512_maskz_mov_epi16(
          _mm512_cmpneq_epi16_mask(_mm512_and_si512(x, _mm512_set1_epi16(127)), _mm512_setzero_si512()),
          _mm512_mask_blend_epi16(
              _mm512_test_epi16_mask(x, _mm512_set1_epi16(120)),
              _mm512_or_si512(
                  _mm512_and_si512(
                      _mm512_sllv_epi16(
                          _mm512_and_si512(x, _mm512_set1_epi16(3)), _mm512_sub_epi16(_mm512_set1_epi16(7), lg2mant)),
                      _mm512_set1_epi16(0x007f)),
                  _mm512_slli_epi16(_mm512_add_epi16(lg2mant, _mm512_set1_epi16(118)), 7)),
              _mm512_or_si512(
                  _mm512_slli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(7)), 4),
                  _mm512_slli_epi16(
                      _mm512_add_epi16(
                          _mm512_srli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(120)), 3), _mm512_set1_epi16(120)),
                      7)))),
      _mm512_slli_epi16(_mm512_and_si512(x, _mm512_set1_epi16(128)), 8)));
}

inline __m512bh CVT_FP8_TO_BF16(__m256i a) {
#ifdef SGLANG_CPU_FP8_CVT_FTZ
  return cvt_e4m3_bf16_intrinsic_no_nan(a);
#else
  return cvt_e4m3_bf16_intrinsic_with_denorm(a);
#endif
}

// faster version of float8_e4m3fn conversion to bfloat16
//
// we mapped cuda implementation from below link and vectorized with avx512:
// https://github.com/thu-pacman/chitu/blob/1ed2078ec26581ebdca05b7306d4385f86edaa7c/csrc/cuda/marlin/marlin_gemm/dequant.h#L387
//
inline __attribute__((always_inline)) __m512bh CVT_FP8_TO_BF16_EXT(__m256i a) {
  const __m512i mask0 = _mm512_set1_epi16(0x80);  // sign bit
  const __m512i mask1 = _mm512_set1_epi16(0x7F);  // exponent and mantissa
  const __m512i mask2 = _mm512_set1_epi16(0x4000);

  __m512i x = _mm512_cvtepu8_epi16(a);
  __m512i vsign = _mm512_and_si512(x, mask0);
  vsign = _mm512_slli_epi16(vsign, 8);

  __m512i vexp_and_mant = _mm512_and_si512(x, mask1);
  vexp_and_mant = _mm512_slli_epi16(vexp_and_mant, 4);

  // _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C: 0b11111110
  return (__m512bh)(_mm512_ternarylogic_epi32(vsign, mask2, vexp_and_mant, 0b11111110));
}

// bias for conversion of fp8 to bf16 1/256 in float32
#define kFP8_BIAS 0x3b800000

// remove warning: ignoring attributes on template argument ‘__m512bh’ [-Wignored-attributes]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

#define MXFP4_VALUES \
  -6.0f, -4.0f, -3.0f, -2.0f, -1.5f, -1.0f, -0.5f, -0.0f, 6.0f, 4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f, 0.0f

// convert 64 mxfp4 to 2x bf16 vectors, expect input 32-way packing
inline std::tuple<__m512bh, __m512bh> cvt_mxfp4_e2m1_bf16_intrinsic_lut(__m256i a, __m512i s0, __m512i s1) {
  // LUT
  const __m512 values = _mm512_set_ps(MXFP4_VALUES);
  const __m512i lut = (__m512i)(_mm512_cvtne2ps_pbh(values, values));

  const __m512i abs_mask = _mm512_set1_epi16(0x7FFF);
  const __m512i zero = _mm512_setzero_si512();

  // expand values to 16-bit integers
  __m512i x0 = _mm512_cvtepu8_epi16(a);
  __m512i x1 = _mm512_srli_epi32(x0, 4);

  // LUT to convert mxfp4 values to bf16
  x0 = _mm512_permutexvar_epi16(x0, lut);
  x1 = _mm512_permutexvar_epi16(x1, lut);

  // check for zeros
  __mmask32 mask0 = _mm512_cmp_epi16_mask(_mm512_and_si512(x0, abs_mask), zero, _MM_CMPINT_EQ);
  __mmask32 mask1 = _mm512_cmp_epi16_mask(_mm512_and_si512(x1, abs_mask), zero, _MM_CMPINT_EQ);

  // emulate bf16 mul with scale factor
  x0 = _mm512_add_epi16(x0, s0);
  x1 = _mm512_add_epi16(x1, s1);

  // blend with zero
  x0 = _mm512_mask_blend_epi16(mask0, x0, zero);
  x1 = _mm512_mask_blend_epi16(mask1, x1, zero);

  return std::make_tuple(__m512bh(x0), __m512bh(x1));
}

#define CVT_MXFP4_TO_BF16(a, s0, s1) cvt_mxfp4_e2m1_bf16_intrinsic_lut(a, s0, s1)

#pragma GCC diagnostic pop

#endif

// vector to scalar reduction
#if defined(CPU_CAPABILITY_AVX512)
inline float vec_reduce_sum(const Vectorized<float>& a) {
  return _mm512_reduce_add_ps(__m512(a));
}

inline float vec_reduce_max(const Vectorized<float>& a) {
  return _mm512_reduce_max_ps(__m512(a));
}
#else
inline float vec_reduce_sum(const Vectorized<float>& a) {
  return vec_reduce_all([](Vectorized<float>& x, Vectorized<float>& y) { return x + y; }, a);
}

inline float vec_reduce_max(const Vectorized<float>& a) {
  return vec_reduce_all([](Vectorized<float>& x, Vectorized<float>& y) { return maximum(x, y); }, a);
}
#endif

// https://github.com/InternLM/lmdeploy/blob/086481ed84b59bee3b8e4274e5fc69620040c048/lmdeploy/pytorch/kernels/cuda/w8a8_triton_kernels.py#L282
template <typename scalar_t>
inline void
quantize_row_int8(uint8_t* __restrict__ Aq, float& As, const scalar_t* __restrict__ A, int64_t K, float eps = 1e-7) {
  float amax = 0.f;  // absolute max
  for (int64_t k = 0; k < K; ++k) {
    const float val = static_cast<float>(A[k]);
    amax = std::max(amax, std::abs(val));
  }

  amax = std::max(amax, eps);
  const float scale = amax / 127;
  const float inv_scale = 127 / amax;

  for (int64_t k = 0; k < K; ++k) {
    const float val = static_cast<float>(A[k]) * inv_scale;
    Aq[k] = (uint8_t)(std::round(val)) + 128;
  }
  As = scale;
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
inline void quantize_row_int8<at::BFloat16>(
    uint8_t* __restrict__ Aq, float& As, const at::BFloat16* __restrict__ A, int64_t K, float eps) {
  const __m512 signBit = _mm512_set1_ps(-0.0f);
  const __m512i off = _mm512_set1_epi32(128);

  // K is 32x, no remainder
  float amax = 0.f;
  __m512 vamax0 = _mm512_set1_ps(0.f);
  __m512 vamax1 = _mm512_set1_ps(0.f);
  for (int64_t k = 0; k < K; k += 32) {
    __m512i va = _mm512_loadu_si512((void*)(A + k));
    __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 0));
    __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 1));
    vamax0 = _mm512_max_ps(vamax0, _mm512_andnot_ps(signBit, va0));
    vamax1 = _mm512_max_ps(vamax1, _mm512_andnot_ps(signBit, va1));
  }
  amax = _mm512_reduce_max_ps(_mm512_max_ps(vamax0, vamax1));
  amax = std::max(amax, eps);
  const float scale = amax / 127;
  const float inv_scale = 127 / amax;
  const __m512 vd = _mm512_set1_ps(inv_scale);

  for (int64_t k = 0; k < K; k += 32) {
    __m512i va = _mm512_loadu_si512((void*)(A + k));
    __m512 va0 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 0));
    __m512 va1 = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(va, 1));
    va0 = _mm512_mul_ps(va0, vd);
    va1 = _mm512_mul_ps(va1, vd);
    va0 = _mm512_roundscale_ps(va0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    va1 = _mm512_roundscale_ps(va1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    __m128i i0 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(va0), off));
    __m128i i1 = _mm512_cvtepi32_epi8(_mm512_add_epi32(_mm512_cvtps_epi32(va1), off));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(Aq + k), _mm256_set_m128i(i1, i0));
  }
  As = scale;
}
#endif

// transpose utils
// taken from my PR in ggml: https://github.com/ggml-org/llama.cpp/pull/8998
#if defined(CPU_CAPABILITY_AVX512)
inline void transpose_16x16_32bit(__m512i* v) {
  __m512i v1[16];
  v1[0] = _mm512_unpacklo_epi32(v[0], v[1]);
  v1[1] = _mm512_unpackhi_epi32(v[0], v[1]);
  v1[2] = _mm512_unpacklo_epi32(v[2], v[3]);
  v1[3] = _mm512_unpackhi_epi32(v[2], v[3]);
  v1[4] = _mm512_unpacklo_epi32(v[4], v[5]);
  v1[5] = _mm512_unpackhi_epi32(v[4], v[5]);
  v1[6] = _mm512_unpacklo_epi32(v[6], v[7]);
  v1[7] = _mm512_unpackhi_epi32(v[6], v[7]);
  v1[8] = _mm512_unpacklo_epi32(v[8], v[9]);
  v1[9] = _mm512_unpackhi_epi32(v[8], v[9]);
  v1[10] = _mm512_unpacklo_epi32(v[10], v[11]);
  v1[11] = _mm512_unpackhi_epi32(v[10], v[11]);
  v1[12] = _mm512_unpacklo_epi32(v[12], v[13]);
  v1[13] = _mm512_unpackhi_epi32(v[12], v[13]);
  v1[14] = _mm512_unpacklo_epi32(v[14], v[15]);
  v1[15] = _mm512_unpackhi_epi32(v[14], v[15]);

  v[0] = _mm512_unpacklo_epi64(v1[0], v1[2]);
  v[1] = _mm512_unpackhi_epi64(v1[0], v1[2]);
  v[2] = _mm512_unpacklo_epi64(v1[1], v1[3]);
  v[3] = _mm512_unpackhi_epi64(v1[1], v1[3]);
  v[4] = _mm512_unpacklo_epi64(v1[4], v1[6]);
  v[5] = _mm512_unpackhi_epi64(v1[4], v1[6]);
  v[6] = _mm512_unpacklo_epi64(v1[5], v1[7]);
  v[7] = _mm512_unpackhi_epi64(v1[5], v1[7]);
  v[8] = _mm512_unpacklo_epi64(v1[8], v1[10]);
  v[9] = _mm512_unpackhi_epi64(v1[8], v1[10]);
  v[10] = _mm512_unpacklo_epi64(v1[9], v1[11]);
  v[11] = _mm512_unpackhi_epi64(v1[9], v1[11]);
  v[12] = _mm512_unpacklo_epi64(v1[12], v1[14]);
  v[13] = _mm512_unpackhi_epi64(v1[12], v1[14]);
  v[14] = _mm512_unpacklo_epi64(v1[13], v1[15]);
  v[15] = _mm512_unpackhi_epi64(v1[13], v1[15]);

  v1[0] = _mm512_shuffle_i32x4(v[0], v[4], 0x88);
  v1[1] = _mm512_shuffle_i32x4(v[1], v[5], 0x88);
  v1[2] = _mm512_shuffle_i32x4(v[2], v[6], 0x88);
  v1[3] = _mm512_shuffle_i32x4(v[3], v[7], 0x88);
  v1[4] = _mm512_shuffle_i32x4(v[0], v[4], 0xdd);
  v1[5] = _mm512_shuffle_i32x4(v[1], v[5], 0xdd);
  v1[6] = _mm512_shuffle_i32x4(v[2], v[6], 0xdd);
  v1[7] = _mm512_shuffle_i32x4(v[3], v[7], 0xdd);
  v1[8] = _mm512_shuffle_i32x4(v[8], v[12], 0x88);
  v1[9] = _mm512_shuffle_i32x4(v[9], v[13], 0x88);
  v1[10] = _mm512_shuffle_i32x4(v[10], v[14], 0x88);
  v1[11] = _mm512_shuffle_i32x4(v[11], v[15], 0x88);
  v1[12] = _mm512_shuffle_i32x4(v[8], v[12], 0xdd);
  v1[13] = _mm512_shuffle_i32x4(v[9], v[13], 0xdd);
  v1[14] = _mm512_shuffle_i32x4(v[10], v[14], 0xdd);
  v1[15] = _mm512_shuffle_i32x4(v[11], v[15], 0xdd);

  v[0] = _mm512_shuffle_i32x4(v1[0], v1[8], 0x88);
  v[1] = _mm512_shuffle_i32x4(v1[1], v1[9], 0x88);
  v[2] = _mm512_shuffle_i32x4(v1[2], v1[10], 0x88);
  v[3] = _mm512_shuffle_i32x4(v1[3], v1[11], 0x88);
  v[4] = _mm512_shuffle_i32x4(v1[4], v1[12], 0x88);
  v[5] = _mm512_shuffle_i32x4(v1[5], v1[13], 0x88);
  v[6] = _mm512_shuffle_i32x4(v1[6], v1[14], 0x88);
  v[7] = _mm512_shuffle_i32x4(v1[7], v1[15], 0x88);
  v[8] = _mm512_shuffle_i32x4(v1[0], v1[8], 0xdd);
  v[9] = _mm512_shuffle_i32x4(v1[1], v1[9], 0xdd);
  v[10] = _mm512_shuffle_i32x4(v1[2], v1[10], 0xdd);
  v[11] = _mm512_shuffle_i32x4(v1[3], v1[11], 0xdd);
  v[12] = _mm512_shuffle_i32x4(v1[4], v1[12], 0xdd);
  v[13] = _mm512_shuffle_i32x4(v1[5], v1[13], 0xdd);
  v[14] = _mm512_shuffle_i32x4(v1[6], v1[14], 0xdd);
  v[15] = _mm512_shuffle_i32x4(v1[7], v1[15], 0xdd);
}

// remove warning : ignoring attributes on template argument ‘__m512i’ [-Wignored-attributes]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

// transpose from [2, 32] to [32, 2]
inline std::tuple<__m512i, __m512i> transpose_2x32_16bit(__m512i r0, __m512i r1) {
  // r0: {a0, a1, ..., a31}
  // r1: {b0, b1, ..., b31}
  //
  // d0: {a0,   b0, ..., a15, b15}
  // d1: {a16, b16, ..., a31, b31}
  //
  __m512i d0 = _mm512_unpacklo_epi16(r0, r1);
  __m512i d1 = _mm512_unpackhi_epi16(r0, r1);
  r0 = _mm512_shuffle_i32x4(d0, d1, 0x88);
  r1 = _mm512_shuffle_i32x4(d0, d1, 0xdd);
  d0 = _mm512_shuffle_i32x4(r0, r1, 0x88);
  d1 = _mm512_shuffle_i32x4(r0, r1, 0xdd);
  return std::make_tuple(d0, d1);
}
#pragma GCC diagnostic pop

inline __attribute__((always_inline)) __m512 _mm512_fexp_u20_ps(const __m512 values) {
  const __m512 vec_c0 = _mm512_set1_ps(0.00010703434948458272f);
  const __m512 vec_c1 = _mm512_set1_ps(0.30354260500649682f);
  const __m512 vec_c2 = _mm512_set1_ps(-0.22433836478672356);
  const __m512 vec_c3 = _mm512_set1_ps(-0.079204240219773236);

  const __m512 vec_exp_log2ef = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));  // log2(e)

  const __m512 vec_a = _mm512_set1_ps(std::pow(2, 23) / std::log2(2));
  const __m512 vec_b = _mm512_set1_ps(std::pow(2, 23) * 127.f);

  const __m512 vec_ln_flt_min = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));
  const __m512 vec_ln_flt_max = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
  __m512i vec_infinity = _mm512_set1_epi32(0x7F800000);
  __m512i vec_zero = _mm512_setzero_epi32();

  // Fast Exponential Computation on SIMD Architectures
  // A. Cristiano I. Malossi, Yves Ineichen, Costas Bekas, and Alessandro
  // Curioni exp(x) = 2**(x * log2(e))
  //        = 2**xi * 2**xf   - TIPS we are using  the EEEE floating point
  //        representation with identification to the exponent and the
  //        mentissa
  //  2**xf will be approximated to a polynomial of degree 3 computed with
  //  Horner method
  // mask for the boundary condition
  auto min_mask = _mm512_cmp_ps_mask(values, vec_ln_flt_min, _CMP_LT_OS);
  auto max_mask = _mm512_cmp_ps_mask(values, vec_ln_flt_max, _CMP_GT_OS);

  // transformation with log2(e)
  auto vec_src = _mm512_mul_ps(values, vec_exp_log2ef);
  auto vec_fractional = _mm512_sub_ps(vec_src, _mm512_floor_ps(vec_src));

  // compute polynomial using Horner Scheme, for superscalar processor
  auto vec_res = _mm512_fmadd_ps(vec_fractional, vec_c3, vec_c2);
  vec_res = _mm512_fmadd_ps(vec_fractional, vec_res, vec_c1);
  vec_res = _mm512_fmadd_ps(vec_fractional, vec_res, vec_c0);

  vec_src = _mm512_sub_ps(vec_src, vec_res);
  // the tips is here, headache in perspective
  auto tmp = _mm512_fmadd_ps(vec_a, vec_src, vec_b);
  // headache bis - we loose precision with the cast but it "fits", but ok
  // after f32 -> f16 later
  __m512i casted_integer = _mm512_cvttps_epi32(tmp);
  // boundary condition, lower than the min -> 0
  casted_integer = _mm512_mask_mov_epi32(casted_integer, min_mask, vec_zero);
  // boundary condition, larger than the max -> +oo
  casted_integer = _mm512_mask_mov_epi32(casted_integer, max_mask, vec_infinity);
  // final interpretation to float
  return _mm512_castsi512_ps(casted_integer);
}
#endif

}  // anonymous namespace
