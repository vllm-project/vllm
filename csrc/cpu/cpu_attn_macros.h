#ifndef CPU_ATTN_MACROS_H
#define CPU_ATTN_MACROS_H

// x86_64
#ifdef __x86_64__
  #define FAST_SPINNING _mm_pause();

  #ifdef __AVX512F__
    #define DEFINE_FAST_EXP                                                    \
      const __m512 vec_factorial_1 = _mm512_set1_ps(0.999999701f);             \
      const __m512 vec_factorial_2 = _mm512_set1_ps(0.499991506f);             \
      const __m512 vec_factorial_3 = _mm512_set1_ps(0.166676521f);             \
      const __m512 vec_factorial_4 = _mm512_set1_ps(0.0418978221f);            \
      const __m512 vec_factorial_5 = _mm512_set1_ps(0.00828929059f);           \
      const __m512 vec_exp_log2ef =                                            \
          _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));                  \
      const __m512 vec_half = _mm512_set1_ps(0.5f);                            \
      const __m512 vec_one = _mm512_set1_ps(1.f);                              \
      const __m512 vec_zero = _mm512_set1_ps(0.f);                             \
      const __m512 vec_two = _mm512_set1_ps(2.f);                              \
      const __m512 vec_ln2f =                                                  \
          _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));                  \
      const __m512 vec_ln_flt_min =                                            \
          _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));                  \
      const __m512 vec_ln_flt_max =                                            \
          _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));                  \
      const __m512i vec_127 = _mm512_set1_epi32(0x0000007f);                   \
      const int n_mantissa_bits = 23;                                          \
      auto fast_exp = [&](vec_op::FP32Vec16& vec) __attribute__((              \
                          always_inline)) {                                    \
        __m512 values = vec.reg;                                               \
        auto less_ln_flt_min_mask =                                            \
            _mm512_cmp_ps_mask(values, vec_ln_flt_min, 1 /*_CMP_LT_OS*/);      \
        auto vec_src = _mm512_min_ps(values, vec_ln_flt_max);                  \
        vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);                      \
        auto vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);      \
        auto vec_fx_i = _mm512_cvt_roundps_epi32(                              \
            vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);                \
        vec_fx = _mm512_cvtepi32_ps(vec_fx_i);                                 \
        auto vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);       \
        auto vec_res =                                                         \
            _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);   \
        vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);     \
        vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);     \
        vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);     \
        vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);             \
        auto vec_exp_number = _mm512_sub_ps(vec_fx, vec_one);                  \
        auto vec_exp_number_i = _mm512_cvtps_epi32(vec_exp_number);            \
        auto vec_two_pow_n_i = _mm512_add_epi32(vec_exp_number_i, vec_127);    \
        vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits); \
        auto vec_two_pow_n = _mm512_castsi512_ps(vec_two_pow_n_i);             \
        vec_two_pow_n = _mm512_mask_blend_ps(less_ln_flt_min_mask,             \
                                             vec_two_pow_n, vec_zero);         \
        vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);                       \
        vec_res = _mm512_mul_ps(vec_res, vec_two);                             \
        vec_op::FP32Vec16 res(vec_res);                                        \
        return res;                                                            \
      };
  #endif

#endif

#ifdef __aarch64__
  // Implementation copied from Arm Optimized Routines (expf AdvSIMD)
  // https://github.com/ARM-software/optimized-routines/blob/master/math/aarch64/advsimd/expf.c
  #include <limits>
  #define DEFINE_FAST_EXP                                                      \
    const float32x4_t inv_ln2 = vdupq_n_f32(0x1.715476p+0f);                   \
    const float ln2_hi = 0x1.62e4p-1f;                                         \
    const float ln2_lo = 0x1.7f7d1cp-20f;                                      \
    const float c0 = 0x1.0e4020p-7f;                                           \
    const float c2 = 0x1.555e66p-3f;                                           \
    const float32x4_t ln2_c02 = {ln2_hi, ln2_lo, c0, c2};                      \
    const uint32x4_t exponent_bias = vdupq_n_u32(0x3f800000);                  \
    const float32x4_t c1 = vdupq_n_f32(0x1.573e2ep-5f);                        \
    const float32x4_t c3 = vdupq_n_f32(0x1.fffdb6p-2f);                        \
    const float32x4_t c4 = vdupq_n_f32(0x1.ffffecp-1f);                        \
    const float32x4_t pos_special_bound = vdupq_n_f32(0x1.5d5e2ap+6f);         \
    const float32x4_t neg_special_bound = vnegq_f32(pos_special_bound);        \
    const float32x4_t inf =                                                    \
        vdupq_n_f32(std::numeric_limits<float>::infinity());                   \
    const float32x4_t zero = vdupq_n_f32(0.0f);                                \
    auto neon_expf = [&](float32x4_t values) __attribute__((always_inline)) {  \
      float32x4_t n = vrndaq_f32(vmulq_f32(values, inv_ln2));                  \
      float32x4_t r = vfmsq_laneq_f32(values, n, ln2_c02, 0);                  \
      r = vfmsq_laneq_f32(r, n, ln2_c02, 1);                                   \
      uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_s32(vcvtq_s32_f32(n)), 23); \
      float32x4_t scale = vreinterpretq_f32_u32(vaddq_u32(e, exponent_bias));  \
      float32x4_t r2 = vmulq_f32(r, r);                                        \
      float32x4_t p = vfmaq_laneq_f32(c1, r, ln2_c02, 2);                      \
      float32x4_t q = vfmaq_laneq_f32(c3, r, ln2_c02, 3);                      \
      q = vfmaq_f32(q, p, r2);                                                 \
      p = vmulq_f32(c4, r);                                                    \
      float32x4_t poly = vfmaq_f32(p, q, r2);                                  \
      poly = vfmaq_f32(scale, poly, scale);                                    \
      const uint32x4_t hi_mask = vcgeq_f32(values, pos_special_bound);         \
      const uint32x4_t lo_mask = vcleq_f32(values, neg_special_bound);         \
      poly = vbslq_f32(hi_mask, inf, poly);                                    \
      return vbslq_f32(lo_mask, zero, poly);                                   \
    };                                                                         \
    auto fast_exp = [&](vec_op::FP32Vec16& vec)                                \
                        __attribute__((always_inline)) {                       \
                          float32x4x4_t result;                                \
                          result.val[0] = neon_expf(vec.reg.val[0]);           \
                          result.val[1] = neon_expf(vec.reg.val[1]);           \
                          result.val[2] = neon_expf(vec.reg.val[2]);           \
                          result.val[3] = neon_expf(vec.reg.val[3]);           \
                          return vec_op::FP32Vec16(result);                    \
                        };

#endif  // __aarch64__

#endif