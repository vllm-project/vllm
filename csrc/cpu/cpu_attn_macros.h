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

#endif