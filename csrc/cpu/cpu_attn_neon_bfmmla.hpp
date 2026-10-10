// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef CPU_ATTN_NEON_BFMMLA_HPP
#define CPU_ATTN_NEON_BFMMLA_HPP

#include "cpu_attn_impl.hpp"

#include <arm_bf16.h>
#include <arm_neon.h>
#include <c10/util/BFloat16.h>

#include <algorithm>
#include <cstdint>

namespace cpu_attention {

class BfmmlaGemm {
 public:
  static constexpr int32_t KTile = 4;
  static constexpr int32_t NTile = 8;
  static constexpr int32_t MaxRows = 8;

  FORCE_INLINE static void gemm(const c10::BFloat16* __restrict__ a,
                                const c10::BFloat16* __restrict__ b,
                                float* __restrict__ c, const int32_t m,
                                const int32_t n, const int32_t k,
                                const int64_t a_pair_stride,
                                const int64_t b_n_group_stride,
                                const int64_t b_k_group_stride,
                                const int64_t ldc, const bool accumulate) {
    const auto* a_ptr = reinterpret_cast<const bfloat16_t*>(a);
    const auto* b_ptr = reinterpret_cast<const bfloat16_t*>(b);

    for (int32_t n_idx = 0; n_idx < n; n_idx += 16) {
      const auto* b_panel = b_ptr + (n_idx / NTile) * b_n_group_stride;
      float* c_panel = c + n_idx;

      // Preserve this range so the inactive row pair is optimized away.
      if (m <= 2) {
        gemm_4x16(a_ptr, b_panel, c_panel, m, k, a_pair_stride,
                  b_n_group_stride, b_k_group_stride, ldc, accumulate);
      } else if (m <= 4) {
        gemm_4x16(a_ptr, b_panel, c_panel, m, k, a_pair_stride,
                  b_n_group_stride, b_k_group_stride, ldc, accumulate);
      } else {
        gemm_8x8(a_ptr, b_panel, c_panel, m, k, a_pair_stride, b_k_group_stride,
                 ldc, accumulate);
        gemm_8x8(a_ptr, b_panel + b_n_group_stride, c_panel + NTile, m, k,
                 a_pair_stride, b_k_group_stride, ldc, accumulate);
      }
    }
  }

 private:
  FORCE_INLINE static float32x4_t zip_low_pairs(const float32x4_t a,
                                                const float32x4_t b) {
    return vreinterpretq_f32_f64(
        vzip1q_f64(vreinterpretq_f64_f32(a), vreinterpretq_f64_f32(b)));
  }

  FORCE_INLINE static float32x4_t zip_high_pairs(const float32x4_t a,
                                                 const float32x4_t b) {
    return vreinterpretq_f32_f64(
        vzip2q_f64(vreinterpretq_f64_f32(a), vreinterpretq_f64_f32(b)));
  }

  FORCE_INLINE static void init_accumulators(
      float32x4_t& acc01, float32x4_t& acc23, float32x4_t& acc45,
      float32x4_t& acc67, const float* __restrict__ c, const int64_t ldc,
      const int32_t rows, const bool accumulate) {
    if (!accumulate || rows == 0) {
      acc01 = vdupq_n_f32(0.0f);
      acc23 = vdupq_n_f32(0.0f);
      acc45 = vdupq_n_f32(0.0f);
      acc67 = vdupq_n_f32(0.0f);
      return;
    }

    const float32x4_t row0_0123 = vld1q_f32(c);
    const float32x4_t row0_4567 = vld1q_f32(c + 4);
    const float32x4_t row1_0123 =
        (rows == 2) ? vld1q_f32(c + ldc) : vdupq_n_f32(0.0f);
    const float32x4_t row1_4567 =
        (rows == 2) ? vld1q_f32(c + ldc + 4) : vdupq_n_f32(0.0f);

    acc01 = zip_low_pairs(row0_0123, row1_0123);
    acc23 = zip_high_pairs(row0_0123, row1_0123);
    acc45 = zip_low_pairs(row0_4567, row1_4567);
    acc67 = zip_high_pairs(row0_4567, row1_4567);
  }

  FORCE_INLINE static void store_accumulators(
      const float32x4_t acc01, const float32x4_t acc23, const float32x4_t acc45,
      const float32x4_t acc67, float* __restrict__ c, const int64_t ldc,
      const int32_t rows) {
    if (rows == 0) {
      return;
    }

    vst1q_f32(c, zip_low_pairs(acc01, acc23));
    vst1q_f32(c + 4, zip_low_pairs(acc45, acc67));
    if (rows == 2) {
      vst1q_f32(c + ldc, zip_high_pairs(acc01, acc23));
      vst1q_f32(c + ldc + 4, zip_high_pairs(acc45, acc67));
    }
  }

  FORCE_INLINE static bfloat16x8_t load_a_pair(const bfloat16_t* __restrict__ a,
                                               const int32_t rows) {
    if (rows == 0) {
      return vdupq_n_bf16(bfloat16_t{});
    }
    // Packed A reserves both rows for an M tail.
    return vld1q_bf16(a);
  }

  FORCE_INLINE static void gemm_4x16(const bfloat16_t* __restrict__ a,
                                     const bfloat16_t* __restrict__ b,
                                     float* __restrict__ c, const int32_t m,
                                     const int32_t k,
                                     const int64_t a_pair_stride,
                                     const int64_t b_n_group_stride,
                                     const int64_t b_k_group_stride,
                                     const int64_t ldc, const bool accumulate) {
    const int32_t rows01 = std::min(2, std::max(0, m));
    const int32_t rows23 = std::min(2, std::max(0, m - 2));
    float32x4_t acc0101, acc0123, acc0145, acc0167;
    float32x4_t acc2301, acc2323, acc2345, acc2367;
    float32x4_t acc0189, acc011011, acc011213, acc011415;
    float32x4_t acc2389, acc231011, acc231213, acc231415;
    init_accumulators(acc0101, acc0123, acc0145, acc0167, c, ldc, rows01,
                      accumulate);
    init_accumulators(acc2301, acc2323, acc2345, acc2367, c + 2 * ldc, ldc,
                      rows23, accumulate);
    init_accumulators(acc0189, acc011011, acc011213, acc011415, c + 8, ldc,
                      rows01, accumulate);
    init_accumulators(acc2389, acc231011, acc231213, acc231415, c + 2 * ldc + 8,
                      ldc, rows23, accumulate);

    const bfloat16_t* a01 = a;
    const bfloat16_t* a23 = a + a_pair_stride;
    const bfloat16_t* b0 = b;
    const bfloat16_t* b1 = b + b_n_group_stride;

#pragma GCC unroll 4
    for (int32_t k_idx = 0; k_idx < k; k_idx += KTile) {
      const bfloat16x8_t av01 = load_a_pair(a01, rows01);
      const bfloat16x8_t av23 = load_a_pair(a23, rows23);
      const bfloat16x8_t b01 = vld1q_bf16(b0);
      const bfloat16x8_t b23 = vld1q_bf16(b0 + NTile);
      const bfloat16x8_t b45 = vld1q_bf16(b0 + 2 * NTile);
      const bfloat16x8_t b67 = vld1q_bf16(b0 + 3 * NTile);
      const bfloat16x8_t b89 = vld1q_bf16(b1);
      const bfloat16x8_t b1011 = vld1q_bf16(b1 + NTile);
      const bfloat16x8_t b1213 = vld1q_bf16(b1 + 2 * NTile);
      const bfloat16x8_t b1415 = vld1q_bf16(b1 + 3 * NTile);

      acc0101 = vbfmmlaq_f32(acc0101, av01, b01);
      acc2301 = vbfmmlaq_f32(acc2301, av23, b01);
      acc0123 = vbfmmlaq_f32(acc0123, av01, b23);
      acc2323 = vbfmmlaq_f32(acc2323, av23, b23);
      acc0145 = vbfmmlaq_f32(acc0145, av01, b45);
      acc2345 = vbfmmlaq_f32(acc2345, av23, b45);
      acc0167 = vbfmmlaq_f32(acc0167, av01, b67);
      acc2367 = vbfmmlaq_f32(acc2367, av23, b67);
      acc0189 = vbfmmlaq_f32(acc0189, av01, b89);
      acc2389 = vbfmmlaq_f32(acc2389, av23, b89);
      acc011011 = vbfmmlaq_f32(acc011011, av01, b1011);
      acc231011 = vbfmmlaq_f32(acc231011, av23, b1011);
      acc011213 = vbfmmlaq_f32(acc011213, av01, b1213);
      acc231213 = vbfmmlaq_f32(acc231213, av23, b1213);
      acc011415 = vbfmmlaq_f32(acc011415, av01, b1415);
      acc231415 = vbfmmlaq_f32(acc231415, av23, b1415);

      a01 += 2 * KTile;
      a23 += 2 * KTile;
      b0 += b_k_group_stride;
      b1 += b_k_group_stride;
    }

    store_accumulators(acc0101, acc0123, acc0145, acc0167, c, ldc, rows01);
    store_accumulators(acc2301, acc2323, acc2345, acc2367, c + 2 * ldc, ldc,
                       rows23);
    store_accumulators(acc0189, acc011011, acc011213, acc011415, c + 8, ldc,
                       rows01);
    store_accumulators(acc2389, acc231011, acc231213, acc231415,
                       c + 2 * ldc + 8, ldc, rows23);
  }

  FORCE_INLINE static void gemm_8x8(const bfloat16_t* __restrict__ a,
                                    const bfloat16_t* __restrict__ b,
                                    float* __restrict__ c, const int32_t m,
                                    const int32_t k,
                                    const int64_t a_pair_stride,
                                    const int64_t b_k_group_stride,
                                    const int64_t ldc, const bool accumulate) {
    const int32_t rows01 = std::min(2, std::max(0, m));
    const int32_t rows23 = std::min(2, std::max(0, m - 2));
    const int32_t rows45 = std::min(2, std::max(0, m - 4));
    const int32_t rows67 = std::min(2, std::max(0, m - 6));
    float32x4_t acc0101, acc0123, acc0145, acc0167;
    float32x4_t acc2301, acc2323, acc2345, acc2367;
    float32x4_t acc4501, acc4523, acc4545, acc4567;
    float32x4_t acc6701, acc6723, acc6745, acc6767;
    init_accumulators(acc0101, acc0123, acc0145, acc0167, c, ldc, rows01,
                      accumulate);
    init_accumulators(acc2301, acc2323, acc2345, acc2367, c + 2 * ldc, ldc,
                      rows23, accumulate);
    init_accumulators(acc4501, acc4523, acc4545, acc4567, c + 4 * ldc, ldc,
                      rows45, accumulate);
    init_accumulators(acc6701, acc6723, acc6745, acc6767, c + 6 * ldc, ldc,
                      rows67, accumulate);

    const bfloat16_t* a01 = a;
    const bfloat16_t* a23 = a + a_pair_stride;
    const bfloat16_t* a45 = a + 2 * a_pair_stride;
    const bfloat16_t* a67 = a + 3 * a_pair_stride;
    const bfloat16_t* b_ptr = b;

#pragma GCC unroll 4
    for (int32_t k_idx = 0; k_idx < k; k_idx += KTile) {
      const bfloat16x8_t av01 = load_a_pair(a01, rows01);
      const bfloat16x8_t av23 = load_a_pair(a23, rows23);
      const bfloat16x8_t av45 = load_a_pair(a45, rows45);
      const bfloat16x8_t av67 = load_a_pair(a67, rows67);
      const bfloat16x8_t b01 = vld1q_bf16(b_ptr);
      const bfloat16x8_t b23 = vld1q_bf16(b_ptr + NTile);
      const bfloat16x8_t b45 = vld1q_bf16(b_ptr + 2 * NTile);
      const bfloat16x8_t b67 = vld1q_bf16(b_ptr + 3 * NTile);

      acc0101 = vbfmmlaq_f32(acc0101, av01, b01);
      acc2301 = vbfmmlaq_f32(acc2301, av23, b01);
      acc4501 = vbfmmlaq_f32(acc4501, av45, b01);
      acc6701 = vbfmmlaq_f32(acc6701, av67, b01);
      acc0123 = vbfmmlaq_f32(acc0123, av01, b23);
      acc2323 = vbfmmlaq_f32(acc2323, av23, b23);
      acc4523 = vbfmmlaq_f32(acc4523, av45, b23);
      acc6723 = vbfmmlaq_f32(acc6723, av67, b23);
      acc0145 = vbfmmlaq_f32(acc0145, av01, b45);
      acc2345 = vbfmmlaq_f32(acc2345, av23, b45);
      acc4545 = vbfmmlaq_f32(acc4545, av45, b45);
      acc6745 = vbfmmlaq_f32(acc6745, av67, b45);
      acc0167 = vbfmmlaq_f32(acc0167, av01, b67);
      acc2367 = vbfmmlaq_f32(acc2367, av23, b67);
      acc4567 = vbfmmlaq_f32(acc4567, av45, b67);
      acc6767 = vbfmmlaq_f32(acc6767, av67, b67);

      a01 += 2 * KTile;
      a23 += 2 * KTile;
      a45 += 2 * KTile;
      a67 += 2 * KTile;
      b_ptr += b_k_group_stride;
    }

    store_accumulators(acc0101, acc0123, acc0145, acc0167, c, ldc, rows01);
    store_accumulators(acc2301, acc2323, acc2345, acc2367, c + 2 * ldc, ldc,
                       rows23);
    store_accumulators(acc4501, acc4523, acc4545, acc4567, c + 4 * ldc, ldc,
                       rows45);
    store_accumulators(acc6701, acc6723, acc6745, acc6767, c + 6 * ldc, ldc,
                       rows67);
  }
};

namespace {

constexpr int32_t TILE_K = BfmmlaGemm::KTile;
constexpr int32_t TILE_COLS = 2;
constexpr int32_t OUTPUT_COLS_PER_BLOCK = BfmmlaGemm::NTile;
constexpr int32_t K_TOKENS_PER_GROUP = 8;
constexpr int32_t V_TOKENS_PER_ROW_BLOCK = 4;
constexpr int32_t K_CACHE_K_GROUP_STRIDE = K_TOKENS_PER_GROUP * TILE_K;
constexpr int32_t B_COL_PAIR_STRIDE = V_TOKENS_PER_ROW_BLOCK * TILE_COLS;

}  // namespace

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
    static_assert(BlockTokens % 16 == 0);
    if constexpr (head_dim_ct >= 0) {
      static_assert(head_dim_ct == HeadDim);
    }

    const auto* a = reinterpret_cast<const c10::BFloat16*>(a_tile);
    if constexpr (phase == AttentionGemmPhase::QK) {
      constexpr int64_t b_n_group_stride =
          (HeadDim / BfmmlaGemm::KTile) * K_CACHE_K_GROUP_STRIDE;

      for (int32_t row = 0; row < m_size; row += BfmmlaGemm::MaxRows) {
        const int32_t panel_m = std::min(BfmmlaGemm::MaxRows, m_size - row);
        BfmmlaGemm::gemm(a + row * HeadDim, b_tile, c_tile + row * ldc, panel_m,
                         BlockTokens, HeadDim, 2 * HeadDim, b_n_group_stride,
                         K_CACHE_K_GROUP_STRIDE, ldc, accum_c);
      }
    } else {
      const int64_t b_n_group_stride =
          (block_size / V_TOKENS_PER_ROW_BLOCK) * K_CACHE_K_GROUP_STRIDE;

      for (int32_t row = 0; row < m_size; row += BfmmlaGemm::MaxRows) {
        const int32_t panel_m = std::min(BfmmlaGemm::MaxRows, m_size - row);
        BfmmlaGemm::gemm(a + row * lda, b_tile, c_tile + row * ldc, panel_m,
                         HeadDim, dynamic_k_size, 2 * lda, b_n_group_stride,
                         K_CACHE_K_GROUP_STRIDE, ldc, accum_c);
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
  static constexpr int64_t VCacheNGroup = OUTPUT_COLS_PER_BLOCK;
  static constexpr int64_t VCacheKGroupStride = VCacheNGroup * TILE_K;

  static_assert(HeadDim % (2 * OUTPUT_COLS_PER_BLOCK) == 0);
  static_assert(BlockSizeAlignment % K_TOKENS_PER_GROUP == 0);
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

  struct ProbabilityTokenStore {
    static constexpr int32_t TokenStride = 2;

    FORCE_INLINE static void store_probabilities(
        c10::BFloat16* __restrict__ probability,
        const vec_op::FP32Vec16& values, const int32_t row,
        const int64_t row_stride) {
      const int32_t row_in_pair = row & 1;
      auto* dst = reinterpret_cast<bfloat16_t*>(
          probability + row_in_pair * (BfmmlaGemm::KTile - row_stride));
      vst1_bf16(dst, vcvt_bf16_f32(values.reg.val[0]));
      vst1_bf16(dst + 2 * BfmmlaGemm::KTile, vcvt_bf16_f32(values.reg.val[1]));
      vst1_bf16(dst + 4 * BfmmlaGemm::KTile, vcvt_bf16_f32(values.reg.val[2]));
      vst1_bf16(dst + 6 * BfmmlaGemm::KTile, vcvt_bf16_f32(values.reg.val[3]));
    }
  };

  // Key cache stride per token group (TokenColumn layout; QK)
  static constexpr int64_t k_cache_token_group_stride(
      [[maybe_unused]] const int32_t block_size) {
    static_assert(BlockSizeAlignment % K_TOKENS_PER_GROUP == 0);
    return (BlockSizeAlignment / K_TOKENS_PER_GROUP) *
           ((head_dim / TILE_K) * K_CACHE_K_GROUP_STRIDE);
  }

  // Value cache stride per token group (TokenRow layout; PV)
  static constexpr int64_t v_cache_token_group_stride(
      [[maybe_unused]] const int32_t block_size) {
    static_assert(BlockSizeAlignment % V_TOKENS_PER_ROW_BLOCK == 0);
    return (BlockSizeAlignment / V_TOKENS_PER_ROW_BLOCK) * VCacheKGroupStride;
  }

  // The stride to move to the "next" head_dim group
  // is the full V cache size per head, since HeadDimAlignment == head_dim.
  // Hence, the stride is not used in this case
  static constexpr int64_t v_cache_head_group_stride(
      [[maybe_unused]] const int32_t block_size) {
    return head_dim * block_size;
  }

  // Scale Q and write row pairs in BFMMLA reduction order.
  static void copy_q_heads_tile(c10::BFloat16* __restrict__ src,
                                c10::BFloat16* __restrict__ q_buffer,
                                const int32_t q_num,
                                const int32_t q_heads_per_kv,
                                const int64_t q_num_stride,
                                const int64_t q_head_stride, float scale) {
    constexpr int32_t dim = static_cast<int32_t>(head_dim);
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    const bfloat16x4_t zero = vdup_n_bf16(bfloat16_t{});
    const int32_t row_num = q_num * q_heads_per_kv;

    for (int32_t row = 0; row < row_num; row += 2) {
      const int32_t q0 = row / q_heads_per_kv;
      const int32_t h0 = row % q_heads_per_kv;
      const auto* row0 = reinterpret_cast<const bfloat16_t*>(
          src + q0 * q_num_stride + h0 * q_head_stride);
      const bool has_row1 = row + 1 < row_num;
      const int32_t q1 = (row + 1) / q_heads_per_kv;
      const int32_t h1 = (row + 1) % q_heads_per_kv;
      const auto* row1 = has_row1
                             ? reinterpret_cast<const bfloat16_t*>(
                                   src + q1 * q_num_stride + h1 * q_head_stride)
                             : nullptr;
      auto* dst = reinterpret_cast<bfloat16_t*>(q_buffer + row * head_dim);

      for (int32_t k = 0; k < dim; k += OUTPUT_COLS_PER_BLOCK) {
        const bfloat16x8_t in0 = vld1q_bf16(row0 + k);
        const bfloat16x4_t out0_lo =
            vcvt_bf16_f32(vmulq_f32(vcvtq_low_f32_bf16(in0), scale_vec));
        const bfloat16x4_t out0_hi =
            vcvt_bf16_f32(vmulq_f32(vcvtq_high_f32_bf16(in0), scale_vec));

        bfloat16x4_t out1_lo = zero;
        bfloat16x4_t out1_hi = zero;
        if (has_row1) {
          const bfloat16x8_t in1 = vld1q_bf16(row1 + k);
          out1_lo =
              vcvt_bf16_f32(vmulq_f32(vcvtq_low_f32_bf16(in1), scale_vec));
          out1_hi =
              vcvt_bf16_f32(vmulq_f32(vcvtq_high_f32_bf16(in1), scale_vec));
        }

        vst1q_bf16(dst + 2 * k, vcombine_bf16(out0_lo, out1_lo));
        vst1q_bf16(dst + 2 * k + 8, vcombine_bf16(out0_hi, out1_hi));
      }
    }
  }

 public:
  // Reshape and cache K/V into BFMMLA-optimized layouts
  // K cache:
  // [block_size/K_TOKENS_PER_GROUP][head_dim/TILE_K]
  // [K_CACHE_K_GROUP_STRIDE]
  // - TokenColumn
  // V cache:
  // [head_dim/VCacheNGroup][block_size/V_TOKENS_PER_ROW_BLOCK]
  // [VCacheKGroupStride]
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
      [[maybe_unused]] const int64_t block_size_stride,
      const float /*k_inv*/ = 0.0f, const float /*v_inv*/ = 0.0f) {
    const int64_t k_block_stride = (head_dim / TILE_K) * K_CACHE_K_GROUP_STRIDE;
    const int64_t v_n_group_stride =
        (block_size / V_TOKENS_PER_ROW_BLOCK) * VCacheKGroupStride;

#pragma omp parallel for collapse(2)
    for (int64_t token_idx = 0; token_idx < token_num; ++token_idx) {
      for (int64_t head_idx = 0; head_idx < head_num; ++head_idx) {
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
            bfloat16_t* dst = reinterpret_cast<bfloat16_t*>(
                block_base + hd4 * K_CACHE_K_GROUP_STRIDE +
                pair_in_block * B_COL_PAIR_STRIDE + lane_base);
            const bfloat16_t* src =
                reinterpret_cast<const bfloat16_t*>(key_src + hd4 * TILE_K);
            vst1_bf16(dst, vld1_bf16(src));
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

          const int64_t token_group = block_offset / V_TOKENS_PER_ROW_BLOCK;
          const int64_t lane = block_offset & (V_TOKENS_PER_ROW_BLOCK - 1);

          const auto* src = reinterpret_cast<const bfloat16_t*>(value_src);
          auto* dst = reinterpret_cast<bfloat16_t*>(value_base);
          for (int64_t hd8 = 0; hd8 < head_dim / OUTPUT_COLS_PER_BLOCK; ++hd8) {
            const bfloat16x8_t values =
                vld1q_bf16(src + hd8 * OUTPUT_COLS_PER_BLOCK);
            const bfloat16x4_t low = vget_low_bf16(values);
            const bfloat16x4_t high = vget_high_bf16(values);
            bfloat16_t* group =
                dst + hd8 * v_n_group_stride + token_group * VCacheKGroupStride;

            vst1_lane_bf16(group + lane, low, 0);
            vst1_lane_bf16(group + V_TOKENS_PER_ROW_BLOCK + lane, low, 1);
            vst1_lane_bf16(group + B_COL_PAIR_STRIDE + lane, low, 2);
            vst1_lane_bf16(
                group + B_COL_PAIR_STRIDE + V_TOKENS_PER_ROW_BLOCK + lane, low,
                3);
            vst1_lane_bf16(group + 2 * B_COL_PAIR_STRIDE + lane, high, 0);
            vst1_lane_bf16(
                group + 2 * B_COL_PAIR_STRIDE + V_TOKENS_PER_ROW_BLOCK + lane,
                high, 1);
            vst1_lane_bf16(group + 3 * B_COL_PAIR_STRIDE + lane, high, 2);
            vst1_lane_bf16(
                group + 3 * B_COL_PAIR_STRIDE + V_TOKENS_PER_ROW_BLOCK + lane,
                high, 3);
          }
        }
      }
    }
  }
};

}  // namespace cpu_attention

#endif  // CPU_ATTN_NEON_BFMMLA_HPP
