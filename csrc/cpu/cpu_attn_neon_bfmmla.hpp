// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef CPU_ATTN_NEON_BFMMLA_HPP
#define CPU_ATTN_NEON_BFMMLA_HPP

#include "cpu_attn_impl.hpp"

#include <arm_neon.h>

#include <cstdint>
#include <vector>

namespace cpu_attention {

namespace {

// BFMMLA tile dimensions
constexpr int32_t TILE_ROWS = 2;  // M dimension
constexpr int32_t TILE_K = 4;     // K reduction
constexpr int32_t TILE_COLS = 2;  // N dimension (column-pair)

// Derived constants
constexpr int32_t OUTPUT_COLS_PER_BLOCK = 8;   // 4 column-pairs
constexpr int32_t K_TOKENS_PER_GROUP = 8;      // Tokens grouped in K cache
constexpr int32_t V_TOKENS_PER_ROW_BLOCK = 4;  // Tokens per V cache row block
constexpr int32_t K_INNER_STRIDE = K_TOKENS_PER_GROUP * TILE_K;
constexpr int32_t V_INNER_STRIDE = V_TOKENS_PER_ROW_BLOCK * TILE_COLS;
constexpr int32_t PACK_ELEMENTS_PER_K_CHUNK = TILE_ROWS * TILE_K;  // A packing

// Matrix Packing and Accumulator
// Reshape two rows of Q into BFMMLA-friendly interleaved
// Input:  row0 = [a0,a1,a2,a3], row1 = [b0,b1,b2,b3]
// Output: [a0,a1,a2,a3,b0,b1,b2,b3, a4,a5,a6,a7,b4,b5,b6,b7]
// For K tail (K % TILE_K != 0): pads with zeros to complete the final chunk
FORCE_INLINE void reshape_Q_2xK_for_bfmmla(const c10::BFloat16* __restrict r0,
                                           const c10::BFloat16* __restrict r1,
                                           c10::BFloat16* __restrict dst,
                                           int32_t K) {
  const uint16_t* s0 = reinterpret_cast<const uint16_t*>(r0);
  const uint16_t* s1 = reinterpret_cast<const uint16_t*>(r1);
  uint16_t* d = reinterpret_cast<uint16_t*>(dst);

  // Process TILE_K elements at a time (PACK_ELEMENTS_PER_K_CHUNK output)
  int32_t k = 0;
  for (; k + TILE_K <= K; k += TILE_K, d += PACK_ELEMENTS_PER_K_CHUNK) {
    vst1q_u16(d, vcombine_u16(vld1_u16(s0 + k), vld1_u16(s1 + k)));
  }

  // Handle K tail: pack remaining elements with zero-padding
  const int32_t tail = K - k;
  if (tail > 0) {
    // Pack remaining tail elements: [r0[k..k+tail-1], pad, r1[k..k+tail-1],
    // pad]
    for (int32_t t = 0; t < tail; ++t) {
      d[t] = s0[k + t];
      d[t + TILE_K] = s1[k + t];
    }
    // Zero-pad the rest
    for (int32_t t = tail; t < TILE_K; ++t) {
      d[t] = 0;
      d[t + TILE_K] = 0;
    }
  }
}

// 2x2 accumulator load/store with compile-time row count
template <int32_t m_rows>
FORCE_INLINE float32x4_t load_acc_2x2(float* base, int64_t ldc, int col_off) {
  static_assert(m_rows == 1 || m_rows == 2);
  float32x2_t row0 = vld1_f32(base + col_off);
  float32x2_t row1 =
      (m_rows == 2) ? vld1_f32(base + ldc + col_off) : vdup_n_f32(0.f);
  return vcombine_f32(row0, row1);
}

template <int32_t m_rows>
FORCE_INLINE void store_acc_2x2(float32x4_t acc, float* base, int64_t ldc,
                                int col_off) {
  static_assert(m_rows == 1 || m_rows == 2);
  vst1_f32(base + col_off, vget_low_f32(acc));
  if constexpr (m_rows == 2) {
    vst1_f32(base + ldc + col_off, vget_high_f32(acc));
  }
}

// Initialize 4 column-pair accumulators for 2 rows (8 columns total)
#define INIT_ACC_ROWPAIR_4(a0, a1, a2, a3, Crow, ldc, m_rows, accum) \
  do {                                                               \
    if (accum) {                                                     \
      if (m_rows == 2) {                                             \
        a0 = load_acc_2x2<2>(Crow, ldc, 0);                          \
        a1 = load_acc_2x2<2>(Crow, ldc, 2);                          \
        a2 = load_acc_2x2<2>(Crow, ldc, 4);                          \
        a3 = load_acc_2x2<2>(Crow, ldc, 6);                          \
      } else {                                                       \
        a0 = load_acc_2x2<1>(Crow, ldc, 0);                          \
        a1 = load_acc_2x2<1>(Crow, ldc, 2);                          \
        a2 = load_acc_2x2<1>(Crow, ldc, 4);                          \
        a3 = load_acc_2x2<1>(Crow, ldc, 6);                          \
      }                                                              \
    } else {                                                         \
      a0 = a1 = a2 = a3 = vdupq_n_f32(0.f);                          \
    }                                                                \
  } while (0)

// Store 4 column-pair accumulators back to C matrix
#define STORE_ACC_ROWPAIR_4(a0, a1, a2, a3, Crow, ldc, m_rows) \
  do {                                                         \
    if (m_rows == 2) {                                         \
      store_acc_2x2<2>(a0, Crow, ldc, 0);                      \
      store_acc_2x2<2>(a1, Crow, ldc, 2);                      \
      store_acc_2x2<2>(a2, Crow, ldc, 4);                      \
      store_acc_2x2<2>(a3, Crow, ldc, 6);                      \
    } else {                                                   \
      store_acc_2x2<1>(a0, Crow, ldc, 0);                      \
      store_acc_2x2<1>(a1, Crow, ldc, 2);                      \
      store_acc_2x2<1>(a2, Crow, ldc, 4);                      \
      store_acc_2x2<1>(a3, Crow, ldc, 6);                      \
    }                                                          \
  } while (0)

// Perform 4 BFMMLA operations: acc += A @ B for 4 column-pairs
#define BFMMLA_COMPUTE_4(r0, r1, r2, r3, a, b0, b1, b2, b3) \
  do {                                                      \
    r0 = vbfmmlaq_f32(r0, a, b0);                           \
    r1 = vbfmmlaq_f32(r1, a, b1);                           \
    r2 = vbfmmlaq_f32(r2, a, b2);                           \
    r3 = vbfmmlaq_f32(r3, a, b3);                           \
  } while (0)

// Micro-kernel: updates a small fixed tile using BFMMLA.
// RP = number of row-pairs (1,2,4)
// Computes C[TILE_ROWS*RP, OUTPUT_COLS_PER_BLOCK] += A_packed @ B.
// A_packed interleaves RP row-pairs; B layout is driven by the attention phase:
// - AttentionGemmPhase::QK -> token-column layout (Q @ K^T)
// - AttentionGemmPhase::PV -> token-row layout (P @ V)
// K_static < 0 enables runtime K (PV only)
template <int32_t RP, int32_t K_static, AttentionGemmPhase phase>
FORCE_INLINE void gemm_rowpairs_x8_bfmmla_neon(
    const bfloat16_t* const* __restrict A_packed_rp,
    const int32_t* __restrict m_rows_rp, const bfloat16_t* __restrict B_blk,
    float* __restrict C, int64_t ldc, bool accumulate, int64_t b_stride,
    int32_t K_runtime = 0) {
  static_assert(RP == 1 || RP == 2 || RP == 4, "RP must be 1,2,4");
  static_assert(K_static < 0 || K_static % TILE_K == 0,
                "K must be divisible by TILE_K");
  static_assert(K_static >= 0 || phase == AttentionGemmPhase::PV,
                "Runtime K only supported for PV");

  constexpr bool runtime_k = (K_static < 0);
  const int32_t K_iters =
      runtime_k ? (K_runtime / TILE_K) : (K_static / TILE_K);
  const int32_t K_tail = runtime_k ? (K_runtime % TILE_K) : 0;

  if (!runtime_k) {
    // Help the compiler fold away unused K_runtime when K is compile-time
    (void)K_runtime;
  }

  auto* C_al = C;
  const auto* B_al = B_blk;

  // Setup A pointers
  const bfloat16_t* a_ptr[4] = {
      A_packed_rp[0],
      (RP >= 2) ? A_packed_rp[1] : nullptr,
      (RP >= 4) ? A_packed_rp[2] : nullptr,
      (RP >= 4) ? A_packed_rp[3] : nullptr,
  };

  // Setup B pointers based on layout
  const bfloat16_t* b_ptr[4];
  if constexpr (phase == AttentionGemmPhase::PV) {
    b_ptr[0] = B_blk + 0 * b_stride;
    b_ptr[1] = B_blk + 1 * b_stride;
    b_ptr[2] = B_blk + 2 * b_stride;
    b_ptr[3] = B_blk + 3 * b_stride;
  }

  float32x4_t acc[4][4];

// Initialize accumulators
#define INIT_RP(rp)                                                            \
  if constexpr (RP > rp) {                                                     \
    INIT_ACC_ROWPAIR_4(acc[rp][0], acc[rp][1], acc[rp][2], acc[rp][3],         \
                       C_al + (rp * 2) * ldc, ldc, m_rows_rp[rp], accumulate); \
  }
  INIT_RP(0);
  INIT_RP(1);
  INIT_RP(2);
  INIT_RP(3);
#undef INIT_RP

  // Main compute loop
  for (int32_t ki = 0; ki < K_iters; ++ki) {
    bfloat16x8_t b0, b1, b2, b3;
    if constexpr (phase == AttentionGemmPhase::PV) {
      b0 = vld1q_bf16(b_ptr[0] + ki * V_INNER_STRIDE);
      b1 = vld1q_bf16(b_ptr[1] + ki * V_INNER_STRIDE);
      b2 = vld1q_bf16(b_ptr[2] + ki * V_INNER_STRIDE);
      b3 = vld1q_bf16(b_ptr[3] + ki * V_INNER_STRIDE);
    } else {
      const bfloat16_t* b_base = B_al + ki * b_stride;
      b0 = vld1q_bf16(b_base + 0 * V_INNER_STRIDE);
      b1 = vld1q_bf16(b_base + 1 * V_INNER_STRIDE);
      b2 = vld1q_bf16(b_base + 2 * V_INNER_STRIDE);
      b3 = vld1q_bf16(b_base + 3 * V_INNER_STRIDE);
    }

#define COMPUTE_RP(rp)                                                       \
  if constexpr (RP > rp) {                                                   \
    bfloat16x8_t a = vld1q_bf16(a_ptr[rp] + ki * PACK_ELEMENTS_PER_K_CHUNK); \
    BFMMLA_COMPUTE_4(acc[rp][0], acc[rp][1], acc[rp][2], acc[rp][3], a, b0,  \
                     b1, b2, b3);                                            \
  }
    COMPUTE_RP(0);
    COMPUTE_RP(1);
    COMPUTE_RP(2);
    COMPUTE_RP(3);
#undef COMPUTE_RP
  }

  // K tail for runtime PV: fallback path
  if constexpr (runtime_k) {
    if (K_tail > 0) {
      const int32_t tail_offset = K_iters * V_INNER_STRIDE;
      const int32_t a_tail_offset = K_iters * PACK_ELEMENTS_PER_K_CHUNK;
      for (int32_t kt = 0; kt < K_tail; ++kt) {
        float32x4_t b_vecs[4];
        for (int32_t p = 0; p < 4; ++p) {
          const bfloat16_t* bp = b_ptr[p] + tail_offset + kt * TILE_COLS;
          const float b0 = vcvtah_f32_bf16(bp[0]);
          const float b1 = vcvtah_f32_bf16(bp[1]);
          const float32x2_t b_pair = vset_lane_f32(b1, vdup_n_f32(b0), 1);
          b_vecs[p] = vcombine_f32(b_pair, b_pair);
        }

#define TAIL_RP(rp)                                                     \
  if constexpr (RP > rp) {                                              \
    const bfloat16_t* ap = A_packed_rp[rp] + a_tail_offset;             \
    float a_row0 = vcvtah_f32_bf16(ap[kt]);                             \
    float a_row1 =                                                      \
        (m_rows_rp[rp] == 2) ? vcvtah_f32_bf16(ap[kt + TILE_K]) : 0.0f; \
    const float32x4_t a_vec =                                           \
        vcombine_f32(vdup_n_f32(a_row0), vdup_n_f32(a_row1));           \
    for (int32_t p = 0; p < 4; ++p) {                                   \
      acc[rp][p] = vmlaq_f32(acc[rp][p], a_vec, b_vecs[p]);             \
    }                                                                   \
  }
        TAIL_RP(0);
        TAIL_RP(1);
        TAIL_RP(2);
        TAIL_RP(3);
#undef TAIL_RP
      }
    }
  }

  // Store results
#define STORE_RP(rp)                                                    \
  if constexpr (RP > rp) {                                              \
    STORE_ACC_ROWPAIR_4(acc[rp][0], acc[rp][1], acc[rp][2], acc[rp][3], \
                        C_al + (rp * 2) * ldc, ldc, m_rows_rp[rp]);     \
  }
  STORE_RP(0);
  STORE_RP(1);
  STORE_RP(2);
  STORE_RP(3);
#undef STORE_RP
}

// Meso-kernel: packs a small MBxK slice of A, then tiles over N and calls the
// micro-kernel for each OUTPUT_COLS_PER_BLOCK chunk. K_static < 0 enables
// runtime K (PV only).
template <int32_t MB, int32_t N, int32_t K_static, AttentionGemmPhase phase>
FORCE_INLINE void gemm_packA_compute_MB_xN(
    const c10::BFloat16* __restrict A, const c10::BFloat16* __restrict B,
    float* __restrict C, int32_t K_runtime, int64_t lda, int64_t ldc,
    int64_t b_layout_stride, int64_t b_reduction_stride, bool accumulate) {
  static_assert(MB >= 1 && MB <= 8, "MB must be in [1,8]");
  static_assert(N % OUTPUT_COLS_PER_BLOCK == 0,
                "N must be a multiple of OUTPUT_COLS_PER_BLOCK");
  static_assert(K_static < 0 || K_static % TILE_K == 0,
                "K must be divisible by TILE_K");
  static_assert(K_static >= 0 || phase == AttentionGemmPhase::PV,
                "Runtime K only supported for PV");

  constexpr bool runtime_k = (K_static < 0);
  const int32_t K_val = runtime_k ? K_runtime : K_static;

  // Keep small packs on-stack to avoid heap churn
  constexpr int32_t STACK_PACK_STRIDE =
      (1024 / TILE_K) * PACK_ELEMENTS_PER_K_CHUNK;

  constexpr int32_t ROW_PAIRS = (MB + 1) / TILE_ROWS;
  const int32_t pack_stride =
      runtime_k ? ((K_val + TILE_K - 1) / TILE_K) * PACK_ELEMENTS_PER_K_CHUNK
                : (K_static / TILE_K) * PACK_ELEMENTS_PER_K_CHUNK;

  alignas(64) c10::BFloat16 A_packed_stack[ROW_PAIRS * STACK_PACK_STRIDE];
  std::vector<c10::BFloat16> A_packed_heap;
  c10::BFloat16* A_packed =
      (pack_stride <= STACK_PACK_STRIDE)
          ? A_packed_stack
          : (A_packed_heap.resize(ROW_PAIRS * pack_stride),
             A_packed_heap.data());

  for (int32_t rp = 0; rp < ROW_PAIRS; ++rp) {
    const int32_t m = rp * TILE_ROWS;
    const int32_t m_rows = (m + 1 < MB) ? TILE_ROWS : 1;
    const c10::BFloat16* A0 = A + m * lda;
    const c10::BFloat16* A1 = (m_rows == TILE_ROWS) ? (A + (m + 1) * lda) : A0;
    reshape_Q_2xK_for_bfmmla(A0, A1, A_packed + rp * pack_stride, K_val);
  }

  for (int32_t n = 0; n < N; n += OUTPUT_COLS_PER_BLOCK) {
    const c10::BFloat16* B_blk_c10 =
        (phase == AttentionGemmPhase::PV)
            ? (B + (n / TILE_COLS) * b_layout_stride)
            : (B + (n / OUTPUT_COLS_PER_BLOCK) * b_layout_stride);
    const bfloat16_t* B_blk = reinterpret_cast<const bfloat16_t*>(B_blk_c10);

    // Process row-pairs in groups of 4, 2, then 1
    int32_t row_pair_idx = 0;

#define PROCESS_RP_GROUP(group_size)                                       \
  for (; row_pair_idx + (group_size - 1) < ROW_PAIRS;                      \
       row_pair_idx += group_size) {                                       \
    const bfloat16_t* Ap[group_size];                                      \
    int32_t mr[group_size];                                                \
    for (int32_t i = 0; i < group_size; ++i) {                             \
      Ap[i] = reinterpret_cast<const bfloat16_t*>(                         \
          A_packed + (row_pair_idx + i) * pack_stride);                    \
      mr[i] = (((row_pair_idx + i) * TILE_ROWS + 1) < MB) ? TILE_ROWS : 1; \
    }                                                                      \
    float* C_blk = C + (row_pair_idx * TILE_ROWS) * ldc + n;               \
    if constexpr (runtime_k) {                                             \
      gemm_rowpairs_x8_bfmmla_neon<group_size, -1, phase>(                 \
          Ap, mr, B_blk, C_blk, ldc, accumulate, b_layout_stride, K_val);  \
    } else {                                                               \
      gemm_rowpairs_x8_bfmmla_neon<group_size, K_static, phase>(           \
          Ap, mr, B_blk, C_blk, ldc, accumulate,                           \
          (phase == AttentionGemmPhase::PV) ? b_layout_stride              \
                                            : b_reduction_stride);         \
    }                                                                      \
  }

    PROCESS_RP_GROUP(4);
    PROCESS_RP_GROUP(2);
    PROCESS_RP_GROUP(1);
#undef PROCESS_RP_GROUP
  }
}

// Macro-kernel: iterates over M in MB={8,4,2,1} chunks.
// Supports compile-time K specialization when K >= 0; otherwise uses runtime K
// (runtime K path is only supported for PV).
template <AttentionGemmPhase phase, int32_t N, int32_t K = -1>
FORCE_INLINE void gemm_macro_neon_bfmmla(
    const c10::BFloat16* __restrict A, const c10::BFloat16* __restrict B,
    float* __restrict C, int32_t M, int32_t K_runtime, int64_t lda, int64_t ldc,
    int64_t b_layout_stride, int64_t b_reduction_stride, bool accumulate) {
  static_assert(N % OUTPUT_COLS_PER_BLOCK == 0,
                "N must be a multiple of OUTPUT_COLS_PER_BLOCK");

  if constexpr (K >= 0) {
    static_assert(K % TILE_K == 0, "K must be divisible by TILE_K");
    for (int32_t m = 0; m < M;) {
      const int32_t rem = M - m;
      const c10::BFloat16* A_blk = A + m * lda;
      float* C_blk = C + m * ldc;

#define DISPATCH_MB(mb)                                                   \
  gemm_packA_compute_MB_xN<mb, N, K, phase>(A_blk, B, C_blk, 0, lda, ldc, \
                                            b_layout_stride,              \
                                            b_reduction_stride, accumulate)

      if (rem >= 8) {
        DISPATCH_MB(8);
        m += 8;
      } else if (rem >= 4) {
        DISPATCH_MB(4);
        m += 4;
      } else if (rem >= 2) {
        DISPATCH_MB(2);
        m += 2;
      } else {
        DISPATCH_MB(1);
        m += 1;
      }
#undef DISPATCH_MB
    }
  } else {
    static_assert(phase == AttentionGemmPhase::PV,
                  "Runtime K specialization only supported for PV.");
    const int32_t K_val = K_runtime;

    for (int32_t m = 0; m < M;) {
      const int32_t rem = M - m;
      const c10::BFloat16* A_blk = A + m * lda;
      float* C_blk = C + m * ldc;

#define DISPATCH_MB_RUNTIME(mb)                                                \
  gemm_packA_compute_MB_xN<mb, N, -1, phase>(A_blk, B, C_blk, K_val, lda, ldc, \
                                             b_layout_stride,                  \
                                             b_reduction_stride, accumulate)

      if (rem >= 8) {
        DISPATCH_MB_RUNTIME(8);
        m += 8;
      } else if (rem >= 4) {
        DISPATCH_MB_RUNTIME(4);
        m += 4;
      } else if (rem >= 2) {
        DISPATCH_MB_RUNTIME(2);
        m += 2;
      } else {
        DISPATCH_MB_RUNTIME(1);
        m += 1;
      }
#undef DISPATCH_MB_RUNTIME
    }
  }
}

#undef INIT_ACC_ROWPAIR_4
#undef STORE_ACC_ROWPAIR_4
#undef BFMMLA_COMPUTE_4

}  // namespace

// TileGemm Adapter for Attention

template <typename kv_cache_t, int32_t BlockTokens, int32_t HeadDim>
class TileGemmNEONBFMMLA {
 public:
  template <AttentionGemmPhase phase, int32_t head_dim_ct>
  FORCE_INLINE static void gemm(const int32_t m_size, void* __restrict__ a_tile,
                                kv_cache_t* __restrict__ b_tile,
                                float* __restrict__ c_tile, const int64_t lda,
                                [[maybe_unused]] const int64_t ldb,
                                const int64_t ldc,
                                [[maybe_unused]] const int32_t block_size,
                                [[maybe_unused]] const int32_t dynamic_k_size,
                                const bool accum_c) {
    static_assert(BlockTokens % OUTPUT_COLS_PER_BLOCK == 0);
    // BFMMLA kernels require compile-time head_dim; keep head_dim_ct only for
    // API parity with other tile_gemm implementations.
    if constexpr (head_dim_ct >= 0) {
      static_assert(head_dim_ct == HeadDim,
                    "BFMMLA expects head_dim_ct to match HeadDim; PV passes "
                    "-1 for API parity.");
    }

    if constexpr (phase == AttentionGemmPhase::QK) {
      const int64_t b_reduction_stride = K_INNER_STRIDE;
      const int64_t b_token_block_stride = (HeadDim / TILE_K) * K_INNER_STRIDE;

      gemm_macro_neon_bfmmla<AttentionGemmPhase::QK, BlockTokens, HeadDim>(
          reinterpret_cast<const c10::BFloat16*>(a_tile), b_tile, c_tile,
          m_size, 0, lda, ldc, b_token_block_stride, b_reduction_stride,
          accum_c);
    } else {
      const int64_t b_pair_stride =
          (block_size / V_TOKENS_PER_ROW_BLOCK) * V_INNER_STRIDE;

      // PV gemm with runtime K specialization
      switch (dynamic_k_size) {
        case 32:
          gemm_macro_neon_bfmmla<AttentionGemmPhase::PV, HeadDim, 32>(
              reinterpret_cast<const c10::BFloat16*>(a_tile), b_tile, c_tile,
              m_size, 32, lda, ldc, b_pair_stride, 0, accum_c);
          break;
        case 128:
          gemm_macro_neon_bfmmla<AttentionGemmPhase::PV, HeadDim, 128>(
              reinterpret_cast<const c10::BFloat16*>(a_tile), b_tile, c_tile,
              m_size, 128, lda, ldc, b_pair_stride, 0, accum_c);
          break;
        case 256:
          gemm_macro_neon_bfmmla<AttentionGemmPhase::PV, HeadDim, 256>(
              reinterpret_cast<const c10::BFloat16*>(a_tile), b_tile, c_tile,
              m_size, 256, lda, ldc, b_pair_stride, 0, accum_c);
          break;
        default:
          gemm_macro_neon_bfmmla<AttentionGemmPhase::PV, HeadDim>(
              reinterpret_cast<const c10::BFloat16*>(a_tile), b_tile, c_tile,
              m_size, dynamic_k_size, lda, ldc, b_pair_stride, 0, accum_c);
          break;
      }
    }
  }
};

// Shared ASIMD BFMMLA implementation (BF16 only). The block size alignment and
// ISA tag are template parameters so we can reuse the same kernels for
// different NEON configurations.
template <int64_t block_size_alignment, ISA isa_type, int64_t head_dim>
class AttentionImplNEONBFMMLA {
 public:
  using query_t = c10::BFloat16;
  using q_buffer_t = c10::BFloat16;
  using kv_cache_t = c10::BFloat16;
  using logits_buffer_t = float;
  using partial_output_buffer_t = float;
  using prob_buffer_t = c10::BFloat16;

  static constexpr int64_t BlockSizeAlignment = block_size_alignment;
  // HeadDimAlignment equals head_dim so that the PV phase processes
  // the full head dimension in a single gemm call.
  static constexpr int64_t HeadDimAlignment = head_dim;
  static constexpr int64_t MaxQHeadNumPerIteration = 16;
  static constexpr int64_t HeadDim = head_dim;
  static constexpr ISA ISAType = isa_type;
  static constexpr bool scale_on_logits = false;

  static_assert(HeadDim % OUTPUT_COLS_PER_BLOCK == 0);
  static_assert(BlockSizeAlignment % OUTPUT_COLS_PER_BLOCK == 0);
  static_assert(HeadDim % TILE_K == 0, "HeadDim must be a multiple of TILE_K");

 public:
  template <template <typename tile_gemm_t> typename attention>
  FORCE_INLINE void execute_attention(DEFINE_CPU_ATTENTION_PARAMS) {
    attention<
        TileGemmNEONBFMMLA<kv_cache_t, static_cast<int32_t>(BlockSizeAlignment),
                           static_cast<int32_t>(HeadDim)>>
        attention_iteration;
    attention_iteration(CPU_ATTENTION_PARAMS);
  }

  // Key cache stride per token group (TokenColumn layout; QK)
  static constexpr int64_t k_cache_token_group_stride(
      [[maybe_unused]] const int32_t block_size) {
    static_assert(BlockSizeAlignment % K_TOKENS_PER_GROUP == 0);
    return (BlockSizeAlignment / K_TOKENS_PER_GROUP) *
           ((head_dim / TILE_K) * K_INNER_STRIDE);
  }

  // Value cache stride per token group (TokenRow layout; PV)
  static constexpr int64_t v_cache_token_group_stride(
      [[maybe_unused]] const int32_t block_size) {
    static_assert(BlockSizeAlignment % V_TOKENS_PER_ROW_BLOCK == 0);
    return (BlockSizeAlignment / V_TOKENS_PER_ROW_BLOCK) * V_INNER_STRIDE;
  }

  // The stride to move to the "next" head_dim group
  // is the full V cache size per head, since HeadDimAlignment == head_dim.
  // Hence, the stride is not used in this case
  static constexpr int64_t v_cache_head_group_stride(
      [[maybe_unused]] const int32_t block_size) {
    return head_dim * block_size;
  }

  // Convert Q heads to BF16 and apply scale factor using native BF16 intrinsics
  static void copy_q_heads_tile(c10::BFloat16* __restrict__ src,
                                c10::BFloat16* __restrict__ q_buffer,
                                const int32_t q_num,
                                const int32_t q_heads_per_kv,
                                const int64_t q_num_stride,
                                const int64_t q_head_stride, float scale) {
    constexpr int32_t dim = static_cast<int32_t>(head_dim);
    const float32x4_t scale_vec = vdupq_n_f32(scale);

    for (int32_t qi = 0; qi < q_num; ++qi) {
      for (int32_t hi = 0; hi < q_heads_per_kv; ++hi) {
        c10::BFloat16* __restrict__ curr_q =
            src + qi * q_num_stride + hi * q_head_stride;
        c10::BFloat16* __restrict__ dst =
            q_buffer + qi * q_heads_per_kv * head_dim + hi * head_dim;

        for (int32_t i = 0; i < dim; i += OUTPUT_COLS_PER_BLOCK) {
          bfloat16x8_t in8 =
              vld1q_bf16(reinterpret_cast<const bfloat16_t*>(curr_q + i));
          float32x4_t lo = vmulq_f32(vcvtq_low_f32_bf16(in8), scale_vec);
          float32x4_t hi = vmulq_f32(vcvtq_high_f32_bf16(in8), scale_vec);

          bfloat16x4_t lo_b = vcvt_bf16_f32(lo);
          bfloat16x4_t hi_b = vcvt_bf16_f32(hi);
          bfloat16x8_t out = vcombine_bf16(lo_b, hi_b);
          vst1q_bf16(reinterpret_cast<bfloat16_t*>(dst + i), out);
        }
      }
    }
  }

 public:
  // Reshape and cache K/V into BFMMLA-optimized layouts
  // K cache:
  // [block_size/K_TOKENS_PER_GROUP][head_dim/TILE_K][K_INNER_STRIDE]
  // - TokenColumn
  // V cache:
  // [head_dim/TILE_COLS][block_size/V_TOKENS_PER_ROW_BLOCK][V_INNER_STRIDE]
  // - TokenRows
  static void reshape_and_cache(
      const c10::BFloat16* __restrict__ key,
      const c10::BFloat16* __restrict__ value,
      c10::BFloat16* __restrict__ key_cache,
      c10::BFloat16* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping, const int64_t token_num,
      const int64_t key_token_num_stride, const int64_t value_token_num_stride,
      const int64_t head_num, const int64_t key_head_num_stride,
      const int64_t value_head_num_stride,
      [[maybe_unused]] const int64_t num_blocks,
      const int64_t num_blocks_stride, const int64_t cache_head_num_stride,
      const int64_t block_size,
      [[maybe_unused]] const int64_t block_size_stride) {
    const int64_t k_block_stride = (head_dim / TILE_K) * K_INNER_STRIDE;
    const int64_t v_pair_stride =
        (block_size / V_TOKENS_PER_ROW_BLOCK) * V_INNER_STRIDE;

#pragma omp parallel for
    for (int64_t head_idx = 0; head_idx < head_num; ++head_idx) {
      for (int64_t token_idx = 0; token_idx < token_num; ++token_idx) {
        const int64_t pos = slot_mapping[token_idx];
        if (pos < 0) continue;

        const int64_t block_idx = pos / block_size;
        const int64_t block_offset = pos % block_size;

        // Key cache: TokenColumn QK
        {
          const c10::BFloat16* __restrict key_src =
              key + token_idx * key_token_num_stride +
              head_idx * key_head_num_stride;

          c10::BFloat16* __restrict key_base = key_cache +
                                               block_idx * num_blocks_stride +
                                               head_idx * cache_head_num_stride;

          const int64_t block_in_block = block_offset / K_TOKENS_PER_GROUP;
          const int64_t pair_in_block =
              (block_offset % K_TOKENS_PER_GROUP) / TILE_COLS;
          const int64_t lane_base = (block_offset & 1) ? TILE_K : 0;

          c10::BFloat16* __restrict block_base =
              key_base + block_in_block * k_block_stride;

          for (int64_t hd4 = 0; hd4 < head_dim / TILE_K; ++hd4) {
            uint16_t* dst_u16 = reinterpret_cast<uint16_t*>(
                block_base + hd4 * K_INNER_STRIDE +
                pair_in_block * V_INNER_STRIDE + lane_base);
            const uint16_t* src_u16 =
                reinterpret_cast<const uint16_t*>(key_src + hd4 * TILE_K);
            vst1_u16(dst_u16, vld1_u16(src_u16));
          }
        }

        // Value cache: TokenRow PV
        {
          const c10::BFloat16* __restrict value_src =
              value + token_idx * value_token_num_stride +
              head_idx * value_head_num_stride;

          c10::BFloat16* __restrict value_base =
              value_cache + block_idx * num_blocks_stride +
              head_idx * cache_head_num_stride;

          const int64_t row_block = block_offset / V_TOKENS_PER_ROW_BLOCK;
          const int64_t lane = block_offset & (V_TOKENS_PER_ROW_BLOCK - 1);

          c10::BFloat16* __restrict row_block_base =
              value_base + row_block * V_INNER_STRIDE;

          for (int64_t hd2 = 0; hd2 < head_dim / TILE_COLS; ++hd2) {
            c10::BFloat16* __restrict dst_val =
                row_block_base + hd2 * v_pair_stride;

            const uint16_t* src_u16 =
                reinterpret_cast<const uint16_t*>(value_src);
            uint16_t* dst_u16 = reinterpret_cast<uint16_t*>(dst_val);
            dst_u16[lane] = src_u16[hd2 * TILE_COLS + 0];
            dst_u16[lane + V_TOKENS_PER_ROW_BLOCK] =
                src_u16[hd2 * TILE_COLS + 1];
          }
        }
      }
    }
  }
};

}  // namespace cpu_attention

#endif  // CPU_ATTN_ASIMD_BFMMLA_HPP
