#ifndef CPU_ATTN_SVE_HPP
#define CPU_ATTN_SVE_HPP

#include <arm_sve.h>

#ifdef ARM_BF16_SUPPORT
  #include "cpu_attn_sve_bfmmla.hpp"
#endif

#include "cpu_attn_neon.hpp"

namespace cpu_attention {

#if defined(CPU_CAPABILITY_SVE256)
namespace {

// Raw SVE256 GEMM with named accumulators: vec_op helpers spilled registers.

FORCE_INLINE svfloat32_t rk_load8(const float* p, svbool_t pg8) {
  return svld1_f32(pg8, p);
}
FORCE_INLINE svfloat32_t rk_load8(const c10::Half* p, svbool_t pg8) {
  const svuint32_t u = svld1uh_u32(pg8, reinterpret_cast<const uint16_t*>(p));
  return svcvt_f32_f16_x(pg8, svreinterpret_f16_u32(u));
}
FORCE_INLINE svfloat32_t rk_load8(const c10::BFloat16* p, svbool_t pg8) {
  const svuint32_t w = svld1uh_u32(pg8, reinterpret_cast<const uint16_t*>(p));
#ifdef ARM_SVE_BF16_SUPPORT
  return svcvt_f32_bf16_x(pg8, svreinterpret_bf16_u32(w));
#else
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg8, w, 16));  // bf16<<16 == fp32
#endif
}

// fp32 is load-bound, so it loads A once per 4 k (ld1rq + lane FMLA);
// fp16/bf16 use svmla_n.
template <int32_t M, typename kv_cache_t>
FORCE_INLINE void rk_micro(const float* __restrict__ A,
                           const kv_cache_t* __restrict__ B,
                           float* __restrict__ C, const int64_t lda,
                           const int64_t ldb, const int64_t ldc,
                           const int32_t K, const bool accumulate) {
  static_assert(1 <= M && M <= 8);
  const svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)8);

  svfloat32_t c0, c1, c2, c3, c4, c5, c6, c7;
#define RK_INIT(I, CC)                                                    \
  if constexpr (M > (I)) {                                                \
    CC = accumulate ? svld1_f32(pg, C + (int64_t)(I) * ldc)               \
                    : svdup_n_f32(0.0f);                                  \
  }
  RK_INIT(0, c0) RK_INIT(1, c1) RK_INIT(2, c2) RK_INIT(3, c3)
  RK_INIT(4, c4) RK_INIT(5, c5) RK_INIT(6, c6) RK_INIT(7, c7)
#undef RK_INIT

  int32_t k = 0;
  for (; k + 3 < K; k += 4) {
    const svfloat32_t b0 = rk_load8(B + (int64_t)(k + 0) * ldb, pg);
    const svfloat32_t b1 = rk_load8(B + (int64_t)(k + 1) * ldb, pg);
    const svfloat32_t b2 = rk_load8(B + (int64_t)(k + 2) * ldb, pg);
    const svfloat32_t b3 = rk_load8(B + (int64_t)(k + 3) * ldb, pg);
#define RK_FMA4(I, CC)                                                     \
  if constexpr (M > (I)) {                                                  \
    const float* a = A + (int64_t)(I) * lda + k;                            \
    if constexpr (std::is_same_v<kv_cache_t, float>) {                      \
      const svfloat32_t aq = svld1rq_f32(svptrue_b32(), a);                 \
      CC = svmla_lane_f32(CC, b0, aq, 0);                                   \
      CC = svmla_lane_f32(CC, b1, aq, 1);                                   \
      CC = svmla_lane_f32(CC, b2, aq, 2);                                   \
      CC = svmla_lane_f32(CC, b3, aq, 3);                                   \
    } else {                                                                \
      CC = svmla_n_f32_x(pg, CC, b0, a[0]);                                 \
      CC = svmla_n_f32_x(pg, CC, b1, a[1]);                                 \
      CC = svmla_n_f32_x(pg, CC, b2, a[2]);                                 \
      CC = svmla_n_f32_x(pg, CC, b3, a[3]);                                 \
    }                                                                       \
  }
    RK_FMA4(0, c0) RK_FMA4(1, c1) RK_FMA4(2, c2) RK_FMA4(3, c3)
    RK_FMA4(4, c4) RK_FMA4(5, c5) RK_FMA4(6, c6) RK_FMA4(7, c7)
#undef RK_FMA4
  }

  for (; k < K; ++k) {
    const svfloat32_t b = rk_load8(B + (int64_t)k * ldb, pg);
#define RK_FMA1(I, CC)                                                    \
  if constexpr (M > (I)) CC = svmla_n_f32_x(pg, CC, b, A[(int64_t)(I) * lda + k]);
    RK_FMA1(0, c0) RK_FMA1(1, c1) RK_FMA1(2, c2) RK_FMA1(3, c3)
    RK_FMA1(4, c4) RK_FMA1(5, c5) RK_FMA1(6, c6) RK_FMA1(7, c7)
#undef RK_FMA1
  }

#define RK_STORE(I, CC) \
  if constexpr (M > (I)) svst1_f32(pg, C + (int64_t)(I) * ldc, CC);
  RK_STORE(0, c0) RK_STORE(1, c1) RK_STORE(2, c2) RK_STORE(3, c3)
  RK_STORE(4, c4) RK_STORE(5, c5) RK_STORE(6, c6) RK_STORE(7, c7)
#undef RK_STORE
}

template <int32_t N, typename kv_cache_t>
FORCE_INLINE void rk_macro(const float* __restrict__ A,
                           const kv_cache_t* __restrict__ B,
                           float* __restrict__ C, const int32_t M,
                           const int32_t K, const int64_t lda,
                           const int64_t ldb, const int64_t ldc,
                           const bool accumulate) {
  static_assert(N % 8 == 0);
  for (int32_t m = 0; m < M;) {
    const int32_t rem = M - m;
    const float* Ab = A + (int64_t)m * lda;
    float* Cb = C + (int64_t)m * ldc;
    for (int32_t n = 0; n < N; n += 8) {
      const kv_cache_t* Bn = B + n;
      float* Cn = Cb + n;
      if (rem >= 8)
        rk_micro<8>(Ab, Bn, Cn, lda, ldb, ldc, K, accumulate);
      else if (rem >= 4)
        rk_micro<4>(Ab, Bn, Cn, lda, ldb, ldc, K, accumulate);
      else if (rem >= 2)
        rk_micro<2>(Ab, Bn, Cn, lda, ldb, ldc, K, accumulate);
      else
        rk_micro<1>(Ab, Bn, Cn, lda, ldb, ldc, K, accumulate);
    }
    m += rem >= 8 ? 8 : rem >= 4 ? 4 : rem >= 2 ? 2 : 1;
  }
}

template <typename kv_cache_t>
class TileGemmSVE {
 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size,
                                float* __restrict__ a_tile,
                                kv_cache_t* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                [[maybe_unused]] const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    if constexpr (phase == AttentionGemmPhase::QK) {
      rk_macro<32>(a_tile, b_tile, c_tile, m_size, k_size, lda, ldb, ldc,
                   accum_c);
    } else {
      rk_macro<32>(a_tile, b_tile, c_tile, m_size, dynamic_k_size, lda, ldb,
                   ldc, accum_c);
    }
  }
};

}  // namespace
#endif  // CPU_CAPABILITY_SVE256

// SVE128 inherits NEON's GEMM; only SVE256 overrides it.
template <typename scalar_t, int64_t head_dim, typename kv_cache_scalar_t>
class AttentionImpl<ISA::SVE, scalar_t, head_dim, kv_cache_scalar_t>
    : public AttentionImpl<ISA::NEON, scalar_t, head_dim, kv_cache_scalar_t> {
 public:
  using base_t =
      AttentionImpl<ISA::NEON, scalar_t, head_dim, kv_cache_scalar_t>;

  using q_buffer_t = typename base_t::q_buffer_t;
  using kv_cache_t = typename base_t::kv_cache_t;
  using logits_buffer_t = typename base_t::logits_buffer_t;

  constexpr static ISA ISAType = ISA::SVE;

#if defined(CPU_CAPABILITY_SVE256)
  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    attention<TileGemmSVE<kv_cache_t>> attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }
#endif
};

#ifdef ARM_BF16_SUPPORT
template <int64_t head_dim>
class AttentionImpl<ISA::SVE, c10::BFloat16, head_dim, c10::BFloat16>
    : public AttentionImplSVEBFMMLA<32, ISA::SVE, head_dim> {};
#endif

}  // namespace cpu_attention

#endif  // CPU_ATTN_SVE_HPP