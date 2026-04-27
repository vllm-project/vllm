#ifndef CPU_ATTN_VXE_HPP
#define CPU_ATTN_VXE_HPP

#include "cpu_attn_impl.hpp"
#include <vecintrin.h>
#include <type_traits>

namespace cpu_attention {

namespace {

// s390x Vector = 16 bytes (128 bits)
#define BLOCK_SIZE_ALIGNMENT 32
#define HEAD_SIZE_ALIGNMENT 32
#define MAX_Q_HEAD_NUM_PER_ITER 16

template <typename kv_cache_t>
FORCE_INLINE void load_row8_B_as_f32(const kv_cache_t* p, __vector float& b0,
                                     __vector float& b1);

// [1] Float Specialization
template <>
FORCE_INLINE void load_row8_B_as_f32<float>(const float* p, __vector float& b0,
                                            __vector float& b1) {
  // Explicitly cast to long long for offset, and float* for pointer
  b0 = vec_xl((long long)0, const_cast<float*>(p));
  b1 = vec_xl((long long)0, const_cast<float*>(p + 4));
}

// [2] BFloat16 Specialization (Big Endian Fix)
template <>
FORCE_INLINE void load_row8_B_as_f32<c10::BFloat16>(const c10::BFloat16* p,
                                                    __vector float& b0,
                                                    __vector float& b1) {
  // 1. Load 8 BF16s (16 bytes) into one vector
  // Explicit cast to unsigned short* for vec_xl to return vector unsigned short
  __vector unsigned short raw = vec_xl((long long)0, (unsigned short*)p);

  // 2. Prepare Zero vector
  __vector unsigned short zeros = vec_splat_u16(0);

  // 3. Merge High/Low to expand BF16 -> Float32
  // On Big Endian, a float is [BF16_bits | 16_zero_bits]
  b0 = (__vector float)vec_mergeh(raw, zeros);
  b1 = (__vector float)vec_mergel(raw, zeros);
}

template <>
FORCE_INLINE void load_row8_B_as_f32<c10::Half>(const c10::Half* p,
                                                __vector float& b0,
                                                __vector float& b1) {
  alignas(16) float tmp[8];

  // Manual unroll / conversion
  tmp[0] = static_cast<float>(p[0]);
  tmp[1] = static_cast<float>(p[1]);
  tmp[2] = static_cast<float>(p[2]);
  tmp[3] = static_cast<float>(p[3]);
  tmp[4] = static_cast<float>(p[4]);
  tmp[5] = static_cast<float>(p[5]);
  tmp[6] = static_cast<float>(p[6]);
  tmp[7] = static_cast<float>(p[7]);

  // Explicit arguments for intrinsic: (long long offset, float* ptr)
  b0 = vec_xl((long long)0, (float*)tmp);
  b1 = vec_xl((long long)0, (float*)(tmp + 4));
}

template <int32_t M, typename kv_cache_t>
FORCE_INLINE void gemm_micro_s390x_Mx8_Ku4(
    const float* __restrict A,       // [M x K]
    const kv_cache_t* __restrict B,  // [K x 8]
    float* __restrict C,             // [M x 8]
    int64_t lda, int64_t ldb, int64_t ldc, int32_t K, bool accumulate) {
  static_assert(1 <= M && M <= 8, "M must be in [1,8]");

// Helper macros to unroll codegen for M rows
#define ROWS_APPLY(OP) OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6) OP(7)
#define IF_M(i) if constexpr (M > (i))

  // 1. Define A pointers
#define DECL_A(i) const float* a##i = A + (i) * lda;
  ROWS_APPLY(DECL_A)
#undef DECL_A

  // 2. Define Accumulators (2 vectors covers 8 columns)
#define DECL_ACC(i) __vector float acc##i##_0, acc##i##_1;
  ROWS_APPLY(DECL_ACC)
#undef DECL_ACC

  // 3. Initialize Accumulators (Load C or Zero)
#define INIT_ACC(i)                                                    \
  IF_M(i) {                                                            \
    if (accumulate) {                                                  \
      acc##i##_0 =                                                     \
          vec_xl((long long)0, const_cast<float*>(C + (i) * ldc + 0)); \
      acc##i##_1 =                                                     \
          vec_xl((long long)0, const_cast<float*>(C + (i) * ldc + 4)); \
    } else {                                                           \
      acc##i##_0 = vec_splats(0.0f);                                   \
      acc##i##_1 = vec_splats(0.0f);                                   \
    }                                                                  \
  }
  ROWS_APPLY(INIT_ACC)
#undef INIT_ACC

  int32_t k = 0;

  for (; k + 3 < K; k += 4) {
    // Load 4 values of A for each Row M: A[k...k+3]
#define LOAD_A4(i)        \
  __vector float a##i##v; \
  IF_M(i) a##i##v = vec_xl((long long)0, const_cast<float*>(a##i + k));
    ROWS_APPLY(LOAD_A4)
#undef LOAD_A4

    // Helper: FMA for specific lane L of A
    // s390x: vec_madd(b, vec_splat(a, lane), acc)
#define FMAS_LANE(i, aiv, L)                        \
  IF_M(i) {                                         \
    __vector float a_broad = vec_splat(aiv, L);     \
    acc##i##_0 = vec_madd(b0, a_broad, acc##i##_0); \
    acc##i##_1 = vec_madd(b1, a_broad, acc##i##_1); \
  }

    // Unroll K=0..3
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 0) * ldb, b0, b1);
#define STEP_K0(i) FMAS_LANE(i, a##i##v, 0)
      ROWS_APPLY(STEP_K0)
#undef STEP_K0
    }
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 1) * ldb, b0, b1);
#define STEP_K1(i) FMAS_LANE(i, a##i##v, 1)
      ROWS_APPLY(STEP_K1)
#undef STEP_K1
    }
    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 2) * ldb, b0, b1);
#define STEP_K2(i) FMAS_LANE(i, a##i##v, 2)
      ROWS_APPLY(STEP_K2)
#undef STEP_K2
    }

    {
      __vector float b0, b1;
      load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 3) * ldb, b0, b1);
#define STEP_K3(i) FMAS_LANE(i, a##i##v, 3)
      ROWS_APPLY(STEP_K3)
#undef STEP_K3
    }
#undef FMAS_LANE
  }

  for (; k < K; ++k) {
    __vector float b0, b1;
    load_row8_B_as_f32<kv_cache_t>(B + (int64_t)k * ldb, b0, b1);
#define TAIL_ROW(i)                              \
  IF_M(i) {                                      \
    __vector float ai = vec_splats(*(a##i + k)); \
    acc##i##_0 = vec_madd(b0, ai, acc##i##_0);   \
    acc##i##_1 = vec_madd(b1, ai, acc##i##_1);   \
  }
    ROWS_APPLY(TAIL_ROW)
#undef TAIL_ROW
  }

#define STORE_ROW(i)                           \
  IF_M(i) {                                    \
    vec_xst(acc##i##_0, 0, C + (i) * ldc + 0); \
    vec_xst(acc##i##_1, 0, C + (i) * ldc + 4); \
  }
  ROWS_APPLY(STORE_ROW)
#undef STORE_ROW

#undef ROWS_APPLY
#undef IF_M
}

template <int32_t N, typename kv_cache_t>
FORCE_INLINE void gemm_macro_s390x_Mx8_Ku4(const float* __restrict A,
                                           const kv_cache_t* __restrict B,
                                           float* __restrict C, int32_t M,
                                           int32_t K, int64_t lda, int64_t ldb,
                                           int64_t ldc, bool accumulate) {
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
          gemm_micro_s390x_Mx8_Ku4<8, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc, K,
                                                  accumulate);
          break;
        case 4:
          gemm_micro_s390x_Mx8_Ku4<4, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc, K,
                                                  accumulate);
          break;
        case 2:
          gemm_micro_s390x_Mx8_Ku4<2, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc, K,
                                                  accumulate);
          break;
        default:
          gemm_micro_s390x_Mx8_Ku4<1, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc, K,
                                                  accumulate);
          break;
      }
    }
    m += mb;
  }
}

template <typename kv_cache_t>
class TileGemmS390X {
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
      gemm_macro_s390x_Mx8_Ku4<BLOCK_SIZE_ALIGNMENT, kv_cache_t>(
          a_tile, b_tile, c_tile, m_size, k_size, lda, ldb, ldc, accum_c);
    } else {
      gemm_macro_s390x_Mx8_Ku4<HEAD_SIZE_ALIGNMENT, kv_cache_t>(
          a_tile, b_tile, c_tile, m_size, dynamic_k_size, lda, ldb, ldc,
          accum_c);
    }
  }
};

}  // namespace

template <typename scalar_t, int64_t head_dim>
class AttentionImpl<ISA::VXE, scalar_t, head_dim> {
 public:
  using query_t = scalar_t;
  using q_buffer_t = float;
  using kv_cache_t = scalar_t;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = float;

  constexpr static int64_t BlockSizeAlignment = BLOCK_SIZE_ALIGNMENT;
  constexpr static int64_t HeadDimAlignment = HEAD_SIZE_ALIGNMENT;
  constexpr static int64_t MaxQHeadNumPerIteration = MAX_Q_HEAD_NUM_PER_ITER;
  constexpr static int64_t HeadDim = head_dim;
  constexpr static ISA ISAType = ISA::VXE;
  constexpr static bool scale_on_logits =
      false;  // Scale is applied to Q during copy

 public:
  AttentionImpl() {}

  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    attention<TileGemmS390X<kv_cache_t>> attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }

  // Strides for Memory Layout
  constexpr static int64_t k_cache_token_group_stride(
      const int32_t block_size) {
    return BlockSizeAlignment;  // [head_dim, block_size] layout
  }

  constexpr static int64_t v_cache_token_group_stride(
      const int32_t block_size) {
    return head_dim * BlockSizeAlignment;
  }

  constexpr static int64_t v_cache_head_group_stride(const int32_t block_size) {
    return HeadDimAlignment;
  }

  static void copy_q_heads_tile(scalar_t* __restrict__ src,
                                float* __restrict__ q_buffer,
                                const int32_t q_num,
                                const int32_t q_heads_per_kv,
                                const int64_t q_num_stride,
                                const int64_t q_head_stride, float scale) {
    __vector float scale_vec = vec_splats(scale);
    constexpr bool is_bf16 = std::is_same<scalar_t, c10::BFloat16>::value;

    // Process 8 elements at a time (32 bytes of float output)
    for (int32_t i = 0; i < q_num; ++i) {
      for (int32_t h = 0; h < q_heads_per_kv; ++h) {
        scalar_t* curr_src = src + i * q_num_stride + h * q_head_stride;
        float* curr_dst =
            q_buffer + i * q_heads_per_kv * head_dim + h * head_dim;

        int32_t d = 0;
        for (; d <= head_dim - 8; d += 8) {
          if constexpr (is_bf16) {
            __vector float v0, v1;
            // Reuse our Big-Endian-Safe loader
            load_row8_B_as_f32<scalar_t>(curr_src + d, v0, v1);

            v0 = vec_mul(v0, scale_vec);
            v1 = vec_mul(v1, scale_vec);

            vec_xst(v0, 0, curr_dst + d);
            vec_xst(v1, 0, curr_dst + d + 4);
          } else {
            __vector float v0 = vec_xl((long long)0, (float*)curr_src + d);
            __vector float v1 = vec_xl((long long)0, (float*)curr_src + d + 4);

            v0 = vec_mul(v0, scale_vec);
            v1 = vec_mul(v1, scale_vec);

            vec_xst(v0, 0, curr_dst + d);
            vec_xst(v1, 0, curr_dst + d + 4);
          }
        }

        for (; d < head_dim; ++d) {
          float val = static_cast<float>(curr_src[d]);
          curr_dst[d] = val * scale;
        }
      }
    }
  }

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
        if (pos < 0) continue;

        const int64_t block_idx = pos / block_size;
        const int64_t block_offset = pos % block_size;

        {
          const scalar_t* key_src = key + token_idx * key_token_num_stride +
                                    head_idx * key_head_num_stride;
          scalar_t* key_dst = key_cache + block_idx * num_blocks_stride +
                              head_idx * cache_head_num_stride + block_offset;

          for (int64_t i = 0, j = 0; i < head_dim; ++i, j += block_size) {
            key_dst[j] = key_src[i];
          }
        }

        {
          const scalar_t* val_src = value + token_idx * value_token_num_stride +
                                    head_idx * value_head_num_stride;
          scalar_t* val_dst = value_cache + block_idx * num_blocks_stride +
                              head_idx * cache_head_num_stride +
                              block_offset * head_dim;

          std::memcpy(val_dst, val_src, sizeof(scalar_t) * head_dim);
        }
      }
    }
  }
};

}  // namespace cpu_attention

#undef BLOCK_SIZE_ALIGNMENT
#undef HEAD_SIZE_ALIGNMENT
#undef MAX_Q_HEAD_NUM_PER_ITER

#endif