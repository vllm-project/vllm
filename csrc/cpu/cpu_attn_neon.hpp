#ifndef CPU_ATTN_NEON_HPP
#define CPU_ATTN_NEON_HPP

#include "cpu_attn_impl.hpp"
#include <arm_neon.h>
#include <type_traits>
namespace cpu_attention {

namespace {

#define BLOCK_SIZE_ALIGNMENT 32
#define HEAD_SIZE_ALIGNMENT 32
#define MAX_Q_HEAD_NUM_PER_ITER 16

// These do not use vectorized class for loading / converting
// because csrc/cpu/cpu_types_arm.hpp does not have fallback options
// for vec_op::BF16Vec* / vec_op::BF16Vec* on Arm HW that
// doesn't support BF16.
// We don't use vec_op::FP32Vec* or vec_op::FP16Vec* for consistency.
template <typename kv_cache_t>
FORCE_INLINE void load_row8_B_as_f32(const kv_cache_t* p, float32x4_t& b0,
                                     float32x4_t& b1);

template <>
FORCE_INLINE void load_row8_B_as_f32<float>(const float* p, float32x4_t& b0,
                                            float32x4_t& b1) {
  b0 = vld1q_f32(p + 0);
  b1 = vld1q_f32(p + 4);
}

template <>
FORCE_INLINE void load_row8_B_as_f32<c10::Half>(const c10::Half* p,
                                                float32x4_t& b0,
                                                float32x4_t& b1) {
  const float16_t* h = reinterpret_cast<const float16_t*>(p);
  float16x8_t v = vld1q_f16(h);
  b0 = vcvt_f32_f16(vget_low_f16(v));
  b1 = vcvt_f32_f16(vget_high_f16(v));
}

template <>
FORCE_INLINE void load_row8_B_as_f32<c10::BFloat16>(const c10::BFloat16* p,
                                                    float32x4_t& b0,
                                                    float32x4_t& b1) {
  const uint16_t* u = reinterpret_cast<const uint16_t*>(p);
#ifdef ARM_BF16_SUPPORT
  uint16x8_t u0 = vld1q_u16(u);
  bfloat16x8_t bf0 = vreinterpretq_bf16_u16(u0);
  b0 = vcvtq_low_f32_bf16(bf0);
  b1 = vcvtq_high_f32_bf16(bf0);
#else
  uint16x8_t x0 = vld1q_u16(u);
  uint32x4_t lo = vshlq_n_u32(vmovl_u16(vget_low_u16(x0)), 16);
  uint32x4_t hi = vshlq_n_u32(vmovl_u16(vget_high_u16(x0)), 16);
  b0 = vreinterpretq_f32_u32(lo);
  b1 = vreinterpretq_f32_u32(hi);
#endif
}

// Mx8, with 1 <= M <= 8 , K streamed, unroll-by-4 with NEON FMLAs
// #Loads = (K // 4) * (M + 4 * sizeof(kv_cache_t) / 2)
// #FMLAs = (K // 4) * (4 * 2 * M)
// We have (4 * 2 * M) FMLAs for (M + 4 * sizeof(kv_cache_t) / 2) loads
template <int32_t M, typename kv_cache_t>
FORCE_INLINE void gemm_micro_neon_fmla_Mx8_Ku4(
    const float* __restrict A,       // [M x K],
    const kv_cache_t* __restrict B,  // [K x 8],
    float* __restrict C,             // [M x 8],
    int64_t lda, int64_t ldb, int64_t ldc, int32_t K, bool accumulate) {
  // kernel supports max M of 8, as it'd spill for larger M
  static_assert(1 <= M && M <= 8, "M must be in [1,8]");

// helpers for per-M codegen
#define ROWS_APPLY(OP) OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6) OP(7)
#define IF_M(i) if constexpr (M > (i))

  // A row base pointers
#define DECL_A(i) const float* a##i = A + (i) * lda;
  ROWS_APPLY(DECL_A)
#undef DECL_A

  // declare 2 accumulators per row of M
#define DECL_ACC(i) float32x4_t acc##i##_0, acc##i##_1;
  ROWS_APPLY(DECL_ACC)
#undef DECL_ACC

  // initialize accumulators
#define INIT_ACC(i)                              \
  IF_M(i) {                                      \
    if (accumulate) {                            \
      acc##i##_0 = vld1q_f32(C + (i) * ldc + 0); \
      acc##i##_1 = vld1q_f32(C + (i) * ldc + 4); \
    } else {                                     \
      acc##i##_0 = vdupq_n_f32(0.f);             \
      acc##i##_1 = vdupq_n_f32(0.f);             \
    }                                            \
  }
  ROWS_APPLY(INIT_ACC)
#undef INIT_ACC

  int32_t k = 0;

  // K unrolled by 4
  for (; k + 3 < K; k += 4) {
    // load A[k..k+3] for each active row (M)
#define LOAD_A4(i)     \
  float32x4_t a##i##v; \
  IF_M(i) a##i##v = vld1q_f32(a##i + k);
    ROWS_APPLY(LOAD_A4)
#undef LOAD_A4

    // helper: FMA lane L from aiv
#define FMAS_LANE(i, aiv, L)                              \
  IF_M(i) {                                               \
    acc##i##_0 = vfmaq_laneq_f32(acc##i##_0, b0, aiv, L); \
    acc##i##_1 = vfmaq_laneq_f32(acc##i##_1, b1, aiv, L); \
  }

    // k + 0
    {
      float32x4_t b0, b1;
      load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 0) * ldb, b0, b1);
#define STEP_K0(i) FMAS_LANE(i, a##i##v, 0)
      ROWS_APPLY(STEP_K0)
#undef STEP_K0
    }
    // k + 1
    {
      float32x4_t b0, b1;
      load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 1) * ldb, b0, b1);
#define STEP_K1(i) FMAS_LANE(i, a##i##v, 1)
      ROWS_APPLY(STEP_K1)
#undef STEP_K1
    }
    // k + 2
    {
      float32x4_t b0, b1;
      load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 2) * ldb, b0, b1);
#define STEP_K2(i) FMAS_LANE(i, a##i##v, 2)
      ROWS_APPLY(STEP_K2)
#undef STEP_K2
    }
    // k + 3
    {
      float32x4_t b0, b1;
      load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 3) * ldb, b0, b1);
#define STEP_K3(i) FMAS_LANE(i, a##i##v, 3)
      ROWS_APPLY(STEP_K3)
#undef STEP_K3
    }
#undef FMAS_LANE
  }

  // K tail
  for (; k < K; ++k) {
    float32x4_t b0, b1;
    load_row8_B_as_f32<kv_cache_t>(B + (int64_t)k * ldb, b0, b1);
#define TAIL_ROW(i)                             \
  IF_M(i) {                                     \
    float32x4_t ai = vdupq_n_f32(*(a##i + k));  \
    acc##i##_0 = vfmaq_f32(acc##i##_0, b0, ai); \
    acc##i##_1 = vfmaq_f32(acc##i##_1, b1, ai); \
  }
    ROWS_APPLY(TAIL_ROW)
#undef TAIL_ROW
  }

  // store accumulators to C
#define STORE_ROW(i)                          \
  IF_M(i) {                                   \
    vst1q_f32(C + (i) * ldc + 0, acc##i##_0); \
    vst1q_f32(C + (i) * ldc + 4, acc##i##_1); \
  }
  ROWS_APPLY(STORE_ROW)
#undef STORE_ROW

#undef ROWS_APPLY
#undef IF_M
}

template <int32_t N, typename kv_cache_t>
FORCE_INLINE void gemm_macro_neon_fmla_Mx8_Ku4(const float* __restrict A,
                                               const kv_cache_t* __restrict B,
                                               float* __restrict C, int32_t M,
                                               int32_t K, int64_t lda,
                                               int64_t ldb, int64_t ldc,
                                               bool accumulate) {
  // micro kernel is Mx8
  static_assert(N % 8 == 0, "N must be a multiple of 8");
  for (int32_t m = 0; m < M;) {
    int32_t mb = (M - m >= 8) ? 8 : (M - m >= 4) ? 4 : (M - m >= 2) ? 2 : 1;
    const float* Ab = A + m * lda;
    float* Cb = C + m * ldc;

    for (int32_t n = 0; n < N; n += 8) {
      const kv_cache_t* Bn = B + n;
      float* Cn = Cb + n;
      switch (mb) {
        case 8:
          gemm_micro_neon_fmla_Mx8_Ku4<8, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                      K, accumulate);
          break;
        case 4:
          gemm_micro_neon_fmla_Mx8_Ku4<4, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                      K, accumulate);
          break;
        case 2:
          gemm_micro_neon_fmla_Mx8_Ku4<2, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                      K, accumulate);
          break;
        default:
          gemm_micro_neon_fmla_Mx8_Ku4<1, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                      K, accumulate);
          break;
      }
    }
    // no tail loop for N as it's guaranteed to be a multiple of 8
    m += mb;
  }
}

template <typename kv_cache_t>
class TileGemmNeonFMLA {
 public:
  template <AttentionGemmPhase phase, int32_t k_size>
  FORCE_INLINE static void gemm(const int32_t m_size,
                                float* __restrict__ a_tile,
                                kv_cache_t* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                const int64_t ldb, const int64_t ldc,
                                const int32_t block_size,
                                const int32_t dynamic_k_size,
                                const bool accum_c) {
    if constexpr (phase == AttentionGemmPhase::QK) {
      gemm_macro_neon_fmla_Mx8_Ku4<BLOCK_SIZE_ALIGNMENT, kv_cache_t>(
          a_tile, b_tile, c_tile, m_size, k_size, lda, ldb, ldc, accum_c);
    } else {
      gemm_macro_neon_fmla_Mx8_Ku4<HEAD_SIZE_ALIGNMENT, kv_cache_t>(
          a_tile, b_tile, c_tile, m_size, dynamic_k_size, lda, ldb, ldc,
          accum_c);
    }
  }
};

}  // namespace

// this is similar to "ISA::VEC" at the moment
template <typename scalar_t, int64_t head_dim>
class AttentionImpl<ISA::NEON, scalar_t, head_dim> {
 public:
  using query_t = scalar_t;
  using q_buffer_t = float;
  using kv_cache_t = scalar_t;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = float;

  constexpr static int64_t BlockSizeAlignment =
      BLOCK_SIZE_ALIGNMENT;  // KV token num unit of QK and PV phases
  constexpr static int64_t HeadDimAlignment =
      HEAD_SIZE_ALIGNMENT;  // headdim num unit of PV phase
  constexpr static int64_t MaxQHeadNumPerIteration = MAX_Q_HEAD_NUM_PER_ITER;
  constexpr static int64_t HeadDim = head_dim;
  constexpr static ISA ISAType = ISA::NEON;
  constexpr static bool scale_on_logits = false;  // apply scale on q_buffer

  //  static_assert(HeadDim % HeadDimAlignment == 0);
  // the gemm micro kernel is Mx8
  static_assert(HeadDimAlignment % 8 == 0);
  static_assert(BlockSizeAlignment % 8 == 0);

 public:
  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    attention<TileGemmNeonFMLA<kv_cache_t>> attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }

  // k_cache_token_group_stride: stride of K cache when move to next
  // BlockSizeAlignment tokens in a block
  constexpr static int64_t k_cache_token_group_stride(
      const int32_t block_size) {
    return BlockSizeAlignment;  // layout of k_cache block is [head_dim,
                                // block_size], row-major
  }

  // v_cache_token_group_stride: stride of V cache when move to next
  // BlockSizeAlignment tokens in a block
  constexpr static int64_t v_cache_token_group_stride(
      const int32_t block_size) {
    return head_dim * BlockSizeAlignment;  // layout of v_cache is [block_size,
                                           // head_dim], row-major
  }

  // v_cache_head_group_stride: stride of V cache when move to next
  // HeadDimAlignment head dims in a block
  constexpr static int64_t v_cache_head_group_stride(const int32_t block_size) {
    return HeadDimAlignment;  // layout of v_cache is [block_size, head_dim],
                              // row-major
  }

  // Copy q to q_buffer and cast it to fp32
  static void copy_q_heads_tile(
      scalar_t* __restrict__ src,  // [q_num, q_heads_per_kv, head_size]
      float* __restrict__ q_buffer, const int32_t q_num,
      const int32_t q_heads_per_kv, const int64_t q_num_stride,
      const int64_t q_head_stride, float scale) {
    static_assert(head_dim % 16 == 0);
    constexpr int32_t unroll_size = head_dim / 16;
    using load_vec_t = typename VecTypeTrait<scalar_t>::vec_t;

    vec_op::FP32Vec16 scale_vec(scale);
    for (int32_t q_num_idx = 0; q_num_idx < q_num; ++q_num_idx) {
      for (int32_t q_head_idx = 0; q_head_idx < q_heads_per_kv; ++q_head_idx) {
        scalar_t* __restrict__ curr_q =
            src + q_num_idx * q_num_stride + q_head_idx * q_head_stride;
        float* __restrict__ curr_q_buffer =
            q_buffer + q_num_idx * q_heads_per_kv * head_dim +
            q_head_idx * head_dim;

        vec_op::unroll_loop<int32_t, unroll_size>([&](int32_t i) {
          load_vec_t vec(curr_q);
          vec_op::FP32Vec16 fp32_vec(vec);
          fp32_vec = fp32_vec * scale_vec;
          fp32_vec.save(curr_q_buffer);

          curr_q += 16;
          curr_q_buffer += 16;
        });
      }
    }
  }

  // reshape K as column-major and V as row-major
  static void reshape_and_cache(
      const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
      scalar_t* __restrict__ key_cache, scalar_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride, const int64_t num_blocks,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size, const int64_t block_size_stride) {
#pragma omp parallel for collapse(2)
    for (int64_t token_idx = 0; token_idx < token_num; ++token_idx) {
      for (int64_t head_idx = 0; head_idx < head_num; ++head_idx) {
        const int64_t pos = slot_mapping[token_idx];
        if (pos < 0) {
          // skip
          continue;
        }

        const int64_t block_idx = pos / block_size;
        const int64_t block_offset = pos % block_size;
        {
          // Write Key
          const scalar_t* key_start_ptr = key +
                                          token_idx * key_token_num_stride +
                                          head_idx * key_head_num_stride;
          scalar_t* key_cache_start_ptr =
              key_cache + block_idx * num_blocks_stride +
              head_idx * cache_head_num_stride + block_offset;

#pragma GCC unroll 8
          for (int64_t i = 0, j = 0; i < head_dim; ++i, j += block_size) {
            key_cache_start_ptr[j] = key_start_ptr[i];
          }
        }
        {
          // Write Value
          const scalar_t* value_start_ptr = value +
                                            token_idx * value_token_num_stride +
                                            head_idx * value_head_num_stride;
          scalar_t* value_cache_start_ptr =
              value_cache + block_idx * num_blocks_stride +
              head_idx * cache_head_num_stride + block_offset * head_dim;
          std::memcpy(value_cache_start_ptr, value_start_ptr,
                      sizeof(scalar_t) * head_dim);
        }
      }
    }
  }
};
}  // namespace cpu_attention

#endif  // #ifndef CPU_ATTN_NEON_HPP
