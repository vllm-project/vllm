// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef CPU_ATTN_RVV_HPP
#define CPU_ATTN_RVV_HPP

// RVV attention kernel using VLEN-agnostic RVVI() macros from
// cpu_types_riscv_defs.hpp.  The Mx8 tile GEMM uses 8 FP32 elements
// per vector (LMUL_256 bits of FP32 data), which maps to:
//   VLEN=128: m2 (256 bits = 8 x FP32)
//   VLEN=256: m1 (256 bits = 8 x FP32)
// Only VLEN=128 and VLEN=256 are supported; other VLENs (512, 1024)
// and scalar RISC-V builds fall back to VEC/VEC16.
#if defined(__riscv_v_min_vlen) && \
    (__riscv_v_min_vlen == 128 || __riscv_v_min_vlen == 256)

  #include "cpu_attn_impl.hpp"
  #include "cpu_types_riscv_defs.hpp"
  #include <riscv_vector.h>
  #include <type_traits>

namespace cpu_attention {

namespace {

  #define BLOCK_SIZE_ALIGNMENT 32
  #define HEAD_SIZE_ALIGNMENT 32
  #define MAX_Q_HEAD_NUM_PER_ITER 16

// ============================================================================
// B-matrix row loading: load 8 elements as FP32
// ============================================================================

template <typename kv_cache_t>
FORCE_INLINE fixed_fp32x8_t load_row8_B_as_f32(const kv_cache_t* p);

template <>
FORCE_INLINE fixed_fp32x8_t load_row8_B_as_f32<float>(const float* p) {
  return RVVI(__riscv_vle32_v_f32, LMUL_256)(p, 8);
}

template <>
FORCE_INLINE fixed_fp32x8_t load_row8_B_as_f32<c10::Half>(const c10::Half* p) {
  #ifdef __riscv_zvfh
  fixed_fp16x8_t h = RVVI(__riscv_vle16_v_f16, LMUL_128)(
      reinterpret_cast<const _Float16*>(p), 8);
  return RVVI(__riscv_vfwcvt_f_f_v_f32, LMUL_256)(h, 8);
  #else
  alignas(16) float tmp[8];
  for (int i = 0; i < 8; ++i) {
    tmp[i] = static_cast<float>(p[i]);
  }
  return RVVI(__riscv_vle32_v_f32, LMUL_256)(tmp, 8);
  #endif
}

template <>
FORCE_INLINE fixed_fp32x8_t
load_row8_B_as_f32<c10::BFloat16>(const c10::BFloat16* p) {
  #ifdef __riscv_zvfbfmin
  fixed_bf16x8_t bf = RVVI(__riscv_vle16_v_bf16, LMUL_128)(
      reinterpret_cast<const __bf16*>(p), 8);
  return RVVI(__riscv_vfwcvtbf16_f_f_v_f32, LMUL_256)(bf, 8);
  #else
  fixed_u16x8_t raw = RVVI(__riscv_vle16_v_u16, LMUL_128)(
      reinterpret_cast<const uint16_t*>(p), 8);
  fixed_u32x8_t wide = RVVI(__riscv_vzext_vf2_u32, LMUL_256)(raw, 8);
  fixed_u32x8_t shifted = RVVI(__riscv_vsll_vx_u32, LMUL_256)(wide, 16, 8);
  return RVVI4(__riscv_vreinterpret_v_u32, LMUL_256, _f32, LMUL_256)(shifted);
  #endif
}

// ============================================================================
// Micro kernel: Mx8 tile, K unrolled by 4, RVV scalar-broadcast FMA
// ============================================================================
//
// RVV has no lane-indexed FMA; instead we load A elements as scalars and
// use vfmacc_vf (scalar * vector + accumulator).
//
// The 8-column tile uses LMUL_256 bits of FP32 data:
//   VLEN=128: m2 (2 regs per accumulator), M=8 => 18 of 32 regs
//   VLEN=256: m1 (1 reg  per accumulator), M=8 =>  9 of 32 regs

template <int32_t M, typename kv_cache_t>
FORCE_INLINE void gemm_micro_rvv_fma_Mx8_Ku4(
    const float* __restrict A,       // [M x K]
    const kv_cache_t* __restrict B,  // [K x 8]
    float* __restrict C,             // [M x 8]
    int64_t lda, int64_t ldb, int64_t ldc, int32_t K, bool accumulate) {
  static_assert(1 <= M && M <= 8, "M must be in [1,8]");

  constexpr size_t vl = 8;

  #define ROWS_APPLY(OP) OP(0) OP(1) OP(2) OP(3) OP(4) OP(5) OP(6) OP(7)
  #define IF_M(i) if constexpr (M > (i))

  #define DECL_A(i) const float* a##i = A + (i) * lda;
  ROWS_APPLY(DECL_A)
  #undef DECL_A

  #define DECL_ACC(i) fixed_fp32x8_t acc##i;
  ROWS_APPLY(DECL_ACC)
  #undef DECL_ACC

  #define INIT_ACC(i)                                                    \
    IF_M(i) {                                                            \
      if (accumulate) {                                                  \
        acc##i = RVVI(__riscv_vle32_v_f32, LMUL_256)(C + (i) * ldc, vl); \
      } else {                                                           \
        acc##i = RVVI(__riscv_vfmv_v_f_f32, LMUL_256)(0.f, vl);          \
      }                                                                  \
    }
  ROWS_APPLY(INIT_ACC)
  #undef INIT_ACC

  int32_t k = 0;

  for (; k + 3 < K; k += 4) {
    {
      fixed_fp32x8_t b =
          load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 0) * ldb);
  #define STEP_K0(i)                                                          \
    IF_M(i) {                                                                 \
      acc##i = RVVI(__riscv_vfmacc_vf_f32, LMUL_256)(acc##i, *(a##i + k + 0), \
                                                     b, vl);                  \
    }
      ROWS_APPLY(STEP_K0)
  #undef STEP_K0
    }
    {
      fixed_fp32x8_t b =
          load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 1) * ldb);
  #define STEP_K1(i)                                                          \
    IF_M(i) {                                                                 \
      acc##i = RVVI(__riscv_vfmacc_vf_f32, LMUL_256)(acc##i, *(a##i + k + 1), \
                                                     b, vl);                  \
    }
      ROWS_APPLY(STEP_K1)
  #undef STEP_K1
    }
    {
      fixed_fp32x8_t b =
          load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 2) * ldb);
  #define STEP_K2(i)                                                          \
    IF_M(i) {                                                                 \
      acc##i = RVVI(__riscv_vfmacc_vf_f32, LMUL_256)(acc##i, *(a##i + k + 2), \
                                                     b, vl);                  \
    }
      ROWS_APPLY(STEP_K2)
  #undef STEP_K2
    }
    {
      fixed_fp32x8_t b =
          load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + 3) * ldb);
  #define STEP_K3(i)                                                          \
    IF_M(i) {                                                                 \
      acc##i = RVVI(__riscv_vfmacc_vf_f32, LMUL_256)(acc##i, *(a##i + k + 3), \
                                                     b, vl);                  \
    }
      ROWS_APPLY(STEP_K3)
  #undef STEP_K3
    }
  }

  for (; k < K; ++k) {
    fixed_fp32x8_t b = load_row8_B_as_f32<kv_cache_t>(B + (int64_t)k * ldb);
  #define TAIL_ROW(i)                                                        \
    IF_M(i) {                                                                \
      acc##i =                                                               \
          RVVI(__riscv_vfmacc_vf_f32, LMUL_256)(acc##i, *(a##i + k), b, vl); \
    }
    ROWS_APPLY(TAIL_ROW)
  #undef TAIL_ROW
  }

  #define STORE_ROW(i) \
    IF_M(i) { RVVI(__riscv_vse32_v_f32, LMUL_256)(C + (i) * ldc, acc##i, vl); }
  ROWS_APPLY(STORE_ROW)
  #undef STORE_ROW

  #undef ROWS_APPLY
  #undef IF_M
}

// ============================================================================
// Micro kernel: Mx16 tile (two adjacent 8-column groups), K unrolled by 4
// ============================================================================
//
// Same scalar-broadcast A form as the Mx8 kernel, but each row keeps TWO
// independent m2 accumulators, accl<i> over output columns [0,8) and acch<i>
// over [8,16).  For the narrow-M shapes that dominate decode and GQA
// (m_size <= 4) this:
//   * gives the out-of-order core 2 independent vfmacc.vf chains per row
//     instead of 1, which is what hides the loop-carried accumulator latency,
//     and
//   * amortizes each scalar A load over two broadcasts instead of one, halving
//     the flw -> vfmacc.vf pressure, and
//   * makes the tile consume a full 64-byte B row instead of half of it.
//
// The two independent chains are the operative effect, not the wider access:
// a variant that keeps the same accumulator shape but issues a single
// 16-element LMUL_512 B load instead of two m2 loads was measured to be no
// faster, and a variant with one LMUL_512 accumulator per row (i.e. one chain)
// was clearly slower.  Widening a register does not create an independent
// dependency chain -- it just makes one chain longer.  Measured on VLEN=128:
// M=1/K=256 1.49-1.56x, M=2/K=256 1.34-1.43x versus this Mx8 baseline.
//
// The per-output-element FP32 accumulation order is unchanged (k still ascends
// in steps of one), so results are bit-identical to the Mx8 kernel.
//
// Register budget (VLEN=128, LMUL_256 = m2 = 2 registers per accumulator):
//   M=4: 8 accumulators (16 regs) + 2 B vectors (4 regs) <= 32 regs
//   M=8 would need 32 accumulator registers, so it keeps using the Mx8 tile.

template <int32_t M, typename kv_cache_t>
FORCE_INLINE void gemm_micro_rvv_fma_Mx16_Ku4(
    const float* __restrict A,       // [M x K]
    const kv_cache_t* __restrict B,  // [K x 16]
    float* __restrict C,             // [M x 16]
    int64_t lda, int64_t ldb, int64_t ldc, int32_t K, bool accumulate) {
  static_assert(1 <= M && M <= 4, "M must be in [1,4] for the 16-column tile");

  constexpr size_t vl = 8;

  #define ROWS4_APPLY(OP) OP(0) OP(1) OP(2) OP(3)
  #define IF_M4(i) if constexpr (M > (i))

  #define DECL_A4(i) const float* a##i = A + (i) * lda;
  ROWS4_APPLY(DECL_A4)
  #undef DECL_A4

  // accl<i> accumulates output columns [0, 8) of row i, acch<i> the columns
  // [8, 16); both stay in the same k order as the Mx8 kernel.
  #define DECL_ACC4(i) fixed_fp32x8_t accl##i, acch##i;
  ROWS4_APPLY(DECL_ACC4)
  #undef DECL_ACC4

  #define INIT_ACC4(i)                                                        \
    IF_M4(i) {                                                                \
      if (accumulate) {                                                       \
        accl##i = RVVI(__riscv_vle32_v_f32, LMUL_256)(C + (i) * ldc, vl);     \
        acch##i = RVVI(__riscv_vle32_v_f32, LMUL_256)(C + (i) * ldc + 8, vl); \
      } else {                                                                \
        accl##i = RVVI(__riscv_vfmv_v_f_f32, LMUL_256)(0.f, vl);              \
        acch##i = RVVI(__riscv_vfmv_v_f_f32, LMUL_256)(0.f, vl);              \
      }                                                                       \
    }
  ROWS4_APPLY(INIT_ACC4)
  #undef INIT_ACC4

  int32_t k = 0;

  for (; k + 3 < K; k += 4) {
  #define STEP16_ROW(i, KOFF)                          \
    IF_M4(i) {                                         \
      accl##i = RVVI(__riscv_vfmacc_vf_f32, LMUL_256)( \
          accl##i, *(a##i + k + (KOFF)), blo, vl);     \
      acch##i = RVVI(__riscv_vfmacc_vf_f32, LMUL_256)( \
          acch##i, *(a##i + k + (KOFF)), bhi, vl);     \
    }
  #define STEP16(KOFF)                                                         \
    {                                                                          \
      const fixed_fp32x8_t blo =                                               \
          load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + (KOFF)) * ldb);     \
      const fixed_fp32x8_t bhi =                                               \
          load_row8_B_as_f32<kv_cache_t>(B + (int64_t)(k + (KOFF)) * ldb + 8); \
      STEP16_ROW(0, KOFF)                                                      \
      STEP16_ROW(1, KOFF)                                                      \
      STEP16_ROW(2, KOFF)                                                      \
      STEP16_ROW(3, KOFF)                                                      \
    }

    STEP16(0)
    STEP16(1)
    STEP16(2)
    STEP16(3)
  #undef STEP16
  #undef STEP16_ROW
  }

  for (; k < K; ++k) {
    const fixed_fp32x8_t blo =
        load_row8_B_as_f32<kv_cache_t>(B + (int64_t)k * ldb);
    const fixed_fp32x8_t bhi =
        load_row8_B_as_f32<kv_cache_t>(B + (int64_t)k * ldb + 8);
  #define TAIL16_ROW(i)                                                     \
    IF_M4(i) {                                                              \
      accl##i = RVVI(__riscv_vfmacc_vf_f32, LMUL_256)(accl##i, *(a##i + k), \
                                                      blo, vl);             \
      acch##i = RVVI(__riscv_vfmacc_vf_f32, LMUL_256)(acch##i, *(a##i + k), \
                                                      bhi, vl);             \
    }
    ROWS4_APPLY(TAIL16_ROW)
  #undef TAIL16_ROW
  }

  #define STORE16_ROW(i)                                                   \
    IF_M4(i) {                                                             \
      RVVI(__riscv_vse32_v_f32, LMUL_256)(C + (i) * ldc, accl##i, vl);     \
      RVVI(__riscv_vse32_v_f32, LMUL_256)(C + (i) * ldc + 8, acch##i, vl); \
    }
  ROWS4_APPLY(STORE16_ROW)
  #undef STORE16_ROW

  #undef ROWS4_APPLY
  #undef IF_M4
}

// ============================================================================
// Macro kernel: dispatch M tiles of {8,4,2,1}; narrow M tiles step N by 16
// ============================================================================

template <int32_t N, typename kv_cache_t>
FORCE_INLINE void gemm_macro_rvv_fma_Mx8_Ku4(const float* __restrict A,
                                             const kv_cache_t* __restrict B,
                                             float* __restrict C, int32_t M,
                                             int32_t K, int64_t lda,
                                             int64_t ldb, int64_t ldc,
                                             bool accumulate) {
  static_assert(N % 8 == 0, "N must be a multiple of 8");
  constexpr bool has_wide_tile = (N % 16 == 0);
  for (int32_t m = 0; m < M;) {
    int32_t mb = (M - m >= 8) ? 8 : (M - m >= 4) ? 4 : (M - m >= 2) ? 2 : 1;
    const float* Ab = A + m * lda;
    float* Cb = C + m * ldc;

    if constexpr (has_wide_tile) {
      // Narrow M tiles use the 16-column kernel: it needs at most 4 rows and
      // halves the A-slice scalar load count while doubling the independent
      // accumulator chains (see gemm_micro_rvv_fma_Mx16_Ku4).
      if (mb <= 4) {
        for (int32_t n = 0; n < N; n += 16) {
          const kv_cache_t* Bn = B + n;
          float* Cn = Cb + n;
          switch (mb) {
            case 4:
              gemm_micro_rvv_fma_Mx16_Ku4<4, kv_cache_t>(Ab, Bn, Cn, lda, ldb,
                                                         ldc, K, accumulate);
              break;
            case 2:
              gemm_micro_rvv_fma_Mx16_Ku4<2, kv_cache_t>(Ab, Bn, Cn, lda, ldb,
                                                         ldc, K, accumulate);
              break;
            default:
              gemm_micro_rvv_fma_Mx16_Ku4<1, kv_cache_t>(Ab, Bn, Cn, lda, ldb,
                                                         ldc, K, accumulate);
              break;
          }
        }
        m += mb;
        continue;
      }
    }

    for (int32_t n = 0; n < N; n += 8) {
      const kv_cache_t* Bn = B + n;
      float* Cn = Cb + n;
      switch (mb) {
        case 8:
          gemm_micro_rvv_fma_Mx8_Ku4<8, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                    K, accumulate);
          break;
        case 4:
          gemm_micro_rvv_fma_Mx8_Ku4<4, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                    K, accumulate);
          break;
        case 2:
          gemm_micro_rvv_fma_Mx8_Ku4<2, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                    K, accumulate);
          break;
        default:
          gemm_micro_rvv_fma_Mx8_Ku4<1, kv_cache_t>(Ab, Bn, Cn, lda, ldb, ldc,
                                                    K, accumulate);
          break;
      }
    }
    m += mb;
  }
}

// ============================================================================
// TileGemm wrapper — plugs into AttentionMainLoop
// ============================================================================

template <typename kv_cache_t>
class TileGemmRVV {
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
      gemm_macro_rvv_fma_Mx8_Ku4<BLOCK_SIZE_ALIGNMENT, kv_cache_t>(
          a_tile, b_tile, c_tile, m_size, k_size, lda, ldb, ldc, accum_c);
    } else {
      gemm_macro_rvv_fma_Mx8_Ku4<HEAD_SIZE_ALIGNMENT, kv_cache_t>(
          a_tile, b_tile, c_tile, m_size, dynamic_k_size, lda, ldb, ldc,
          accum_c);
    }
  }
};

}  // namespace

// ============================================================================
// AttentionImpl<ISA::RVV> — mirrors ISA::NEON specialization
// ============================================================================

template <typename scalar_t, int64_t head_dim, typename kv_cache_scalar_t>
class AttentionImpl<ISA::RVV, scalar_t, head_dim, kv_cache_scalar_t> {
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
  constexpr static ISA ISAType = ISA::RVV;
  constexpr static bool scale_on_logits = false;

  static_assert(HeadDim % HeadDimAlignment == 0);
  static_assert(HeadDimAlignment % 8 == 0);
  static_assert(BlockSizeAlignment % 8 == 0);

 public:
  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    attention<TileGemmRVV<kv_cache_t>> attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }

  constexpr static int64_t k_cache_token_group_stride(
      const int32_t block_size) {
    return BlockSizeAlignment;
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

  static void reshape_and_cache(
      const scalar_t* __restrict__ key, const scalar_t* __restrict__ value,
      scalar_t* __restrict__ key_cache, scalar_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride, const int64_t num_blocks,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size, const int64_t block_size_stride,
      const float /*k_inv*/ = 0.0f, const float /*v_inv*/ = 0.0f) {
  #pragma omp parallel for collapse(2)
    for (int64_t token_idx = 0; token_idx < token_num; ++token_idx) {
      for (int64_t head_idx = 0; head_idx < head_num; ++head_idx) {
        const int64_t pos = slot_mapping[token_idx];
        if (pos < 0) {
          continue;
        }

        const int64_t block_idx = pos / block_size;
        const int64_t block_offset = pos % block_size;
        {
          const scalar_t* key_start_ptr = key +
                                          token_idx * key_token_num_stride +
                                          head_idx * key_head_num_stride;
          scalar_t* key_cache_start_ptr =
              key_cache + block_idx * num_blocks_stride +
              head_idx * cache_head_num_stride + block_offset;

          {
            const ptrdiff_t byte_stride = block_size * sizeof(scalar_t);
            int64_t i = 0;
            for (; i < head_dim;) {
              size_t vl;
              if constexpr (std::is_same_v<scalar_t, float>) {
                vl = __riscv_vsetvl_e32m2(head_dim - i);
                vfloat32m2_t v = __riscv_vle32_v_f32m2(
                    reinterpret_cast<const float*>(key_start_ptr + i), vl);
                __riscv_vsse32_v_f32m2(
                    reinterpret_cast<float*>(key_cache_start_ptr +
                                             i * block_size),
                    byte_stride, v, vl);
              } else {
                vl = __riscv_vsetvl_e16m1(head_dim - i);
                vuint16m1_t v = __riscv_vle16_v_u16m1(
                    reinterpret_cast<const uint16_t*>(key_start_ptr + i), vl);
                __riscv_vsse16_v_u16m1(
                    reinterpret_cast<uint16_t*>(key_cache_start_ptr +
                                                i * block_size),
                    byte_stride, v, vl);
              }
              i += vl;
            }
          }
        }
        {
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

  #undef BLOCK_SIZE_ALIGNMENT
  #undef HEAD_SIZE_ALIGNMENT
  #undef MAX_Q_HEAD_NUM_PER_ITER

#endif  // __riscv_v_min_vlen == 128 || 256

#endif  // CPU_ATTN_RVV_HPP
