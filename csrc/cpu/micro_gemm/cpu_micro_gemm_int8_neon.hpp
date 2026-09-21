// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef CPU_MICRO_GEMM_INT8_NEON_HPP
#define CPU_MICRO_GEMM_INT8_NEON_HPP

#include <algorithm>
#include <cstdint>

#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"

#include <arm_bf16.h>
#include <arm_neon.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>

namespace cpu_micro_gemm {

namespace neon_smmla {

constexpr int32_t K = 8;
constexpr int32_t Cols = 2;
constexpr int32_t TileSize = K * Cols;

FORCE_INLINE float32x4x2_t load_as_f32(const float* input) {
  float32x4x2_t result;
  result.val[0] = vld1q_f32(input);
  result.val[1] = vld1q_f32(input + 4);
  return result;
}

FORCE_INLINE float32x4x2_t load_as_f32(const c10::Half* input) {
  const auto input_vec = vld1q_f16(reinterpret_cast<const float16_t*>(input));
  float32x4x2_t result;
  result.val[0] = vcvt_f32_f16(vget_low_f16(input_vec));
  result.val[1] = vcvt_f32_f16(vget_high_f16(input_vec));
  return result;
}

FORCE_INLINE float32x4x2_t load_as_f32(const c10::BFloat16* input) {
  const auto input_vec = vld1q_bf16(reinterpret_cast<const bfloat16_t*>(input));
  float32x4x2_t result;
  result.val[0] = vcvt_f32_bf16(vget_low_bf16(input_vec));
  result.val[1] = vcvt_f32_bf16(vget_high_bf16(input_vec));
  return result;
}

FORCE_INLINE void store_acc_rowpair(const int32x4_t acc01,
                                    const int32x4_t acc23,
                                    const int32x4_t acc45,
                                    const int32x4_t acc67,
                                    int32_t* __restrict__ c_ptr,
                                    const int64_t ldc, const int32_t m_rows) {
  if (m_rows == 0) {
    return;
  }

  vst1q_s32(c_ptr, vcombine_s32(vget_low_s32(acc01), vget_low_s32(acc23)));
  vst1q_s32(c_ptr + 4, vcombine_s32(vget_low_s32(acc45), vget_low_s32(acc67)));

  if (m_rows == 2) {
    vst1q_s32(c_ptr + ldc,
              vcombine_s32(vget_high_s32(acc01), vget_high_s32(acc23)));
    vst1q_s32(c_ptr + ldc + 4,
              vcombine_s32(vget_high_s32(acc45), vget_high_s32(acc67)));
  }
}

FORCE_INLINE void gemm_micro_smmla_8x8_packed_a(
    const int8_t* __restrict__ a_packed, const int8_t* __restrict__ b_packed,
    int32_t* __restrict__ c_ptr, const int32_t m, const int32_t k_size,
    const int64_t ldc) {
  const int32x4_t zero = vdupq_n_s32(0);
  int32x4_t acc0101 = zero, acc0123 = zero, acc0145 = zero, acc0167 = zero;
  int32x4_t acc2301 = zero, acc2323 = zero, acc2345 = zero, acc2367 = zero;
  int32x4_t acc4501 = zero, acc4523 = zero, acc4545 = zero, acc4567 = zero;
  int32x4_t acc6701 = zero, acc6723 = zero, acc6745 = zero, acc6767 = zero;

  const int8_t* __restrict__ a_tile = a_packed;
  const int8_t* __restrict__ b_tile = b_packed;

#pragma GCC unroll 8
  for (int32_t k_idx = 0; k_idx < k_size; k_idx += K) {
    const int8x16_t a_tile01 = vld1q_s8(a_tile);
    const int8x16_t a_tile23 = vld1q_s8(a_tile + TileSize);
    const int8x16_t a_tile45 = vld1q_s8(a_tile + 2 * TileSize);
    const int8x16_t a_tile67 = vld1q_s8(a_tile + 3 * TileSize);

    const int8x16_t b_tile01 = vld1q_s8(b_tile);
    const int8x16_t b_tile23 = vld1q_s8(b_tile + TileSize);
    const int8x16_t b_tile45 = vld1q_s8(b_tile + 2 * TileSize);
    const int8x16_t b_tile67 = vld1q_s8(b_tile + 3 * TileSize);

    acc0101 = vmmlaq_s32(acc0101, a_tile01, b_tile01);
    acc2301 = vmmlaq_s32(acc2301, a_tile23, b_tile01);
    acc4501 = vmmlaq_s32(acc4501, a_tile45, b_tile01);
    acc6701 = vmmlaq_s32(acc6701, a_tile67, b_tile01);

    acc0123 = vmmlaq_s32(acc0123, a_tile01, b_tile23);
    acc2323 = vmmlaq_s32(acc2323, a_tile23, b_tile23);
    acc4523 = vmmlaq_s32(acc4523, a_tile45, b_tile23);
    acc6723 = vmmlaq_s32(acc6723, a_tile67, b_tile23);

    acc0145 = vmmlaq_s32(acc0145, a_tile01, b_tile45);
    acc2345 = vmmlaq_s32(acc2345, a_tile23, b_tile45);
    acc4545 = vmmlaq_s32(acc4545, a_tile45, b_tile45);
    acc6745 = vmmlaq_s32(acc6745, a_tile67, b_tile45);

    acc0167 = vmmlaq_s32(acc0167, a_tile01, b_tile67);
    acc2367 = vmmlaq_s32(acc2367, a_tile23, b_tile67);
    acc4567 = vmmlaq_s32(acc4567, a_tile45, b_tile67);
    acc6767 = vmmlaq_s32(acc6767, a_tile67, b_tile67);

    a_tile += 4 * TileSize;
    b_tile += 4 * TileSize;
  }

  store_acc_rowpair(acc0101, acc0123, acc0145, acc0167, c_ptr, ldc,
                    std::min(2, m));
  store_acc_rowpair(acc2301, acc2323, acc2345, acc2367, c_ptr + 2 * ldc, ldc,
                    std::min(2, std::max(0, m - 2)));
  store_acc_rowpair(acc4501, acc4523, acc4545, acc4567, c_ptr + 4 * ldc, ldc,
                    std::min(2, std::max(0, m - 4)));
  store_acc_rowpair(acc6701, acc6723, acc6745, acc6767, c_ptr + 6 * ldc, ldc,
                    std::min(2, std::max(0, m - 6)));
}

FORCE_INLINE void gemm_micro_smmla_4x16_packed_a(
    const int8_t* __restrict__ a_packed, const int8_t* __restrict__ b_packed,
    int32_t* __restrict__ c_ptr, const int32_t m, const int32_t k_size,
    const int64_t b_n_group_stride, const int64_t ldc) {
  const int32_t m_rows_01 = std::min(2, m);
  const int32_t m_rows_23 = std::min(2, std::max(0, m - 2));
  const int32x4_t zero = vdupq_n_s32(0);

  int32x4_t acc0101 = zero, acc0123 = zero, acc0145 = zero, acc0167 = zero;
  int32x4_t acc2301 = zero, acc2323 = zero, acc2345 = zero, acc2367 = zero;
  int32x4_t acc0189 = zero, acc011011 = zero, acc011213 = zero,
            acc011415 = zero;
  int32x4_t acc2389 = zero, acc231011 = zero, acc231213 = zero,
            acc231415 = zero;

  const int8_t* __restrict__ a_tile = a_packed;
  // note: b packs 8 panels contiguously, so we need 2 b_tile ptrs
  // for the 4x16 microkernel
  const int8_t* __restrict__ b_tile0 = b_packed;
  const int8_t* __restrict__ b_tile1 = b_packed + b_n_group_stride;

#pragma GCC unroll 8
  for (int32_t k_idx = 0; k_idx < k_size; k_idx += K) {
    const int8x16_t a_tile01 = vld1q_s8(a_tile);
    const int8x16_t a_tile23 = vld1q_s8(a_tile + TileSize);
    const int8x16_t b_tile01 = vld1q_s8(b_tile0);
    const int8x16_t b_tile23 = vld1q_s8(b_tile0 + TileSize);
    const int8x16_t b_tile45 = vld1q_s8(b_tile0 + 2 * TileSize);
    const int8x16_t b_tile67 = vld1q_s8(b_tile0 + 3 * TileSize);
    const int8x16_t b_tile89 = vld1q_s8(b_tile1);
    const int8x16_t b_tile1011 = vld1q_s8(b_tile1 + TileSize);
    const int8x16_t b_tile1213 = vld1q_s8(b_tile1 + 2 * TileSize);
    const int8x16_t b_tile1415 = vld1q_s8(b_tile1 + 3 * TileSize);

    acc0101 = vmmlaq_s32(acc0101, a_tile01, b_tile01);
    acc2301 = vmmlaq_s32(acc2301, a_tile23, b_tile01);
    acc0123 = vmmlaq_s32(acc0123, a_tile01, b_tile23);
    acc2323 = vmmlaq_s32(acc2323, a_tile23, b_tile23);

    acc0145 = vmmlaq_s32(acc0145, a_tile01, b_tile45);
    acc2345 = vmmlaq_s32(acc2345, a_tile23, b_tile45);
    acc0167 = vmmlaq_s32(acc0167, a_tile01, b_tile67);
    acc2367 = vmmlaq_s32(acc2367, a_tile23, b_tile67);

    acc0189 = vmmlaq_s32(acc0189, a_tile01, b_tile89);
    acc2389 = vmmlaq_s32(acc2389, a_tile23, b_tile89);
    acc011011 = vmmlaq_s32(acc011011, a_tile01, b_tile1011);
    acc231011 = vmmlaq_s32(acc231011, a_tile23, b_tile1011);

    acc011213 = vmmlaq_s32(acc011213, a_tile01, b_tile1213);
    acc231213 = vmmlaq_s32(acc231213, a_tile23, b_tile1213);
    acc011415 = vmmlaq_s32(acc011415, a_tile01, b_tile1415);
    acc231415 = vmmlaq_s32(acc231415, a_tile23, b_tile1415);

    a_tile += 2 * TileSize;
    b_tile0 += 4 * TileSize;
    b_tile1 += 4 * TileSize;
  }

  // rows 0-1, columns 0-7
  store_acc_rowpair(acc0101, acc0123, acc0145, acc0167, c_ptr, ldc, m_rows_01);
  // rows 0-1, columns 8-15
  store_acc_rowpair(acc0189, acc011011, acc011213, acc011415, c_ptr + 8, ldc,
                    m_rows_01);
  // rows 2-3, columns 0-7
  store_acc_rowpair(acc2301, acc2323, acc2345, acc2367, c_ptr + 2 * ldc, ldc,
                    m_rows_23);
  // rows 2-3, columns 8-15
  store_acc_rowpair(acc2389, acc231011, acc231213, acc231415,
                    c_ptr + 2 * ldc + 8, ldc, m_rows_23);
}

}  // namespace neon_smmla

template <typename scalar_t>
class MicroGemmINT8<cpu_utils::ISA::NEON, scalar_t> {
 public:
  static constexpr int32_t K = neon_smmla::K;
  static constexpr int32_t Mr = 8;
  static constexpr int32_t Nr = 8;
  static constexpr int32_t NrGemv = 16;
  static constexpr int32_t MaxMSize = 8;
  static constexpr int32_t NSize = 32;
  static constexpr int32_t WeightOCGroupSize = Nr;
  static_assert(MaxMSize % Mr == 0);

  static FORCE_INLINE void quantize_row(const scalar_t* input, int8_t* output,
                                        float& scale, const int32_t size) {
    TORCH_CHECK_EQ(size % K, 0);
    float32x4_t max_vec = vdupq_n_f32(0.0f);

    for (int32_t i = 0; i < size; i += K) {
      const float32x4x2_t input_vec = neon_smmla::load_as_f32(input + i);
      max_vec = vmaxq_f32(max_vec, vabsq_f32(input_vec.val[0]));
      max_vec = vmaxq_f32(max_vec, vabsq_f32(input_vec.val[1]));
    }

    const float abs_max = std::max(vmaxvq_f32(max_vec), 1.0e-7f);
    scale = abs_max / 127.0f;
    const float32x4_t inv_scale_vec = vdupq_n_f32(127.0f / abs_max);

    for (int32_t i = 0; i < size; i += K) {
      const float32x4x2_t input_vec = neon_smmla::load_as_f32(input + i);
      const int32x4_t output_low =
          vcvtnq_s32_f32(vmulq_f32(input_vec.val[0], inv_scale_vec));
      const int32x4_t output_high =
          vcvtnq_s32_f32(vmulq_f32(input_vec.val[1], inv_scale_vec));
      const int16x8_t output_s16 =
          vcombine_s16(vqmovn_s32(output_low), vqmovn_s32(output_high));
      vst1_s8(output + i, vqmovn_s16(output_s16));
    }
  }

  // with current code, fusing this into the gemm micro kernel didn't move the
  // needle
  static FORCE_INLINE void dequantize_tile(
      int32_t* input, float* output, const float* __restrict__ input_scales,
      const float* __restrict__ weight_scales, const int32_t m, const int32_t n,
      const int32_t stride) {
    TORCH_CHECK_EQ(n % 4, 0);
    for (int32_t m_idx = 0; m_idx < m; ++m_idx) {
      const float32x4_t input_scale_vec = vdupq_n_f32(input_scales[m_idx]);
      for (int32_t n_idx = 0; n_idx < n; n_idx += 4) {
        const int32x4_t input_vec = vld1q_s32(input + m_idx * stride + n_idx);
        const float32x4_t weight_scale_vec = vld1q_f32(weight_scales + n_idx);
        const float32x4_t output_vec =
            vmulq_f32(vcvtq_f32_s32(input_vec),
                      vmulq_f32(input_scale_vec, weight_scale_vec));
        vst1q_f32(output + m_idx * stride + n_idx, output_vec);
      }
    }
  }

  // physical layout [
  //  M / (8 or 4); Mr is 8 or 4
  //  K / 8; K for smmla is 8
  //  4,   ; 4 row-pairs for each 8 rows
  //  2,   ; row-pair is 2 rows
  //  4    ; 4 elements per row
  // ]
  static void pack_input_from_rows(const int8_t* const* __restrict__ rows,
                                   int8_t* __restrict__ a_packed,
                                   const int32_t m, const int32_t k) {
    TORCH_CHECK(m > 0 && m <= MaxMSize);
    TORCH_CHECK(k % K == 0);
    const int8x8_t zero = vdup_n_s8(0);

    for (int32_t row_base = 0; row_base < m; row_base += Mr) {
      const int32_t panel_m = std::min(Mr, m - row_base);
      const int8_t* const* panel_rows = rows + row_base;
      int8_t* __restrict__ out = a_packed + row_base * k;

      // fast path for full 8-row panels (fast path for 4-row panels didn't move
      // the needle)
      if (panel_m == Mr) {
        const int8_t* __restrict__ row0 = panel_rows[0];
        const int8_t* __restrict__ row1 = panel_rows[1];
        const int8_t* __restrict__ row2 = panel_rows[2];
        const int8_t* __restrict__ row3 = panel_rows[3];
        const int8_t* __restrict__ row4 = panel_rows[4];
        const int8_t* __restrict__ row5 = panel_rows[5];
        const int8_t* __restrict__ row6 = panel_rows[6];
        const int8_t* __restrict__ row7 = panel_rows[7];
        int32_t k_idx = 0;
        for (; k_idx + 2 * K <= k; k_idx += 2 * K) {
          int8_t* __restrict__ block0 = out;
          int8_t* __restrict__ block1 = out + 4 * neon_smmla::TileSize;

          int8x16_t a0 = vld1q_s8(row0 + k_idx);
          int8x16_t a1 = vld1q_s8(row1 + k_idx);
          vst1q_s8(block0, vcombine_s8(vget_low_s8(a0), vget_low_s8(a1)));
          vst1q_s8(block1, vcombine_s8(vget_high_s8(a0), vget_high_s8(a1)));

          a0 = vld1q_s8(row2 + k_idx);
          a1 = vld1q_s8(row3 + k_idx);
          vst1q_s8(block0 + neon_smmla::TileSize,
                   vcombine_s8(vget_low_s8(a0), vget_low_s8(a1)));
          vst1q_s8(block1 + neon_smmla::TileSize,
                   vcombine_s8(vget_high_s8(a0), vget_high_s8(a1)));

          a0 = vld1q_s8(row4 + k_idx);
          a1 = vld1q_s8(row5 + k_idx);
          vst1q_s8(block0 + 2 * neon_smmla::TileSize,
                   vcombine_s8(vget_low_s8(a0), vget_low_s8(a1)));
          vst1q_s8(block1 + 2 * neon_smmla::TileSize,
                   vcombine_s8(vget_high_s8(a0), vget_high_s8(a1)));

          a0 = vld1q_s8(row6 + k_idx);
          a1 = vld1q_s8(row7 + k_idx);
          vst1q_s8(block0 + 3 * neon_smmla::TileSize,
                   vcombine_s8(vget_low_s8(a0), vget_low_s8(a1)));
          vst1q_s8(block1 + 3 * neon_smmla::TileSize,
                   vcombine_s8(vget_high_s8(a0), vget_high_s8(a1)));

          out += 8 * neon_smmla::TileSize;
        }

        for (; k_idx < k; k_idx += K) {
          int8x8_t a0 = vld1_s8(row0 + k_idx);
          int8x8_t a1 = vld1_s8(row1 + k_idx);
          vst1q_s8(out, vcombine_s8(a0, a1));

          a0 = vld1_s8(row2 + k_idx);
          a1 = vld1_s8(row3 + k_idx);
          vst1q_s8(out + neon_smmla::TileSize, vcombine_s8(a0, a1));

          a0 = vld1_s8(row4 + k_idx);
          a1 = vld1_s8(row5 + k_idx);
          vst1q_s8(out + 2 * neon_smmla::TileSize, vcombine_s8(a0, a1));

          a0 = vld1_s8(row6 + k_idx);
          a1 = vld1_s8(row7 + k_idx);
          vst1q_s8(out + 3 * neon_smmla::TileSize, vcombine_s8(a0, a1));

          out += 4 * neon_smmla::TileSize;
        }
        continue;
      }

      const int32_t row_pairs = (panel_m <= 4) ? 2 : Mr / 2;
      for (int32_t k_idx = 0; k_idx < k; k_idx += K) {
        for (int32_t pair_idx = 0; pair_idx < row_pairs; ++pair_idx) {
          const int32_t row_idx = pair_idx * 2;
          const int8x8_t row0 =
              (row_idx < panel_m) ? vld1_s8(panel_rows[row_idx] + k_idx) : zero;
          const int8x8_t row1 = (row_idx + 1 < panel_m)
                                    ? vld1_s8(panel_rows[row_idx + 1] + k_idx)
                                    : zero;
          vst1q_s8(out, vcombine_s8(row0, row1));
          out += neon_smmla::TileSize;
        }
      }
    }
  }

  // physical layout [
  //  N / 8; Nr is 8
  //  K / 8; K for smmla is 8
  //  4,   ; 4 col-pairs for each 8 cols
  //  2,   ; col-pair is 2 cols
  //  4    ; 4 elements per col
  // ]
  static void pack_weight(const int8_t* __restrict__ weight,
                          int8_t* __restrict__ packed_weight,
                          const int32_t output_size, const int32_t input_size) {
    TORCH_CHECK(output_size % NSize == 0);
    TORCH_CHECK(input_size % K == 0);

    for (int32_t o_idx = 0; o_idx < output_size; o_idx += Nr) {
      int8_t* __restrict__ dst = packed_weight + o_idx * input_size;
      for (int32_t k_idx = 0; k_idx < input_size; k_idx += K) {
        for (int32_t pair_idx = 0; pair_idx < Nr;
             pair_idx += neon_smmla::Cols) {
          const int8_t* __restrict__ row0 =
              weight + (o_idx + pair_idx) * input_size + k_idx;
          const int8_t* __restrict__ row1 = row0 + input_size;
          vst1q_s8(dst, vcombine_s8(vld1_s8(row0), vld1_s8(row1)));
          dst += neon_smmla::TileSize;
        }
      }
    }
  }

  void gemm(const int8_t* __restrict__ a_packed,
            const int8_t* __restrict__ b_packed, int32_t* __restrict__ c,
            const int32_t m, const int32_t k, const int64_t b_n_group_stride,
            const int64_t ldc) const {
    TORCH_CHECK(m > 0 && m <= MaxMSize);
    TORCH_CHECK(k % K == 0);

    for (int32_t n_idx = 0; n_idx < NSize; n_idx += NrGemv) {
      const int8_t* __restrict__ b_panel = b_packed + n_idx * k;

      for (int32_t row_base = 0; row_base < m; row_base += Mr) {
        const int32_t panel_m = std::min(Mr, m - row_base);
        const int8_t* __restrict__ a_panel = a_packed + row_base * k;
        int32_t* __restrict__ c_panel = c + row_base * ldc + n_idx;

        if (panel_m <= 4) {
          neon_smmla::gemm_micro_smmla_4x16_packed_a(
              a_panel, b_panel, c_panel, panel_m, k, b_n_group_stride, ldc);
        } else {
          neon_smmla::gemm_micro_smmla_8x8_packed_a(a_panel, b_panel, c_panel,
                                                    panel_m, k, ldc);
          neon_smmla::gemm_micro_smmla_8x8_packed_a(
              a_panel, b_panel + b_n_group_stride, c_panel + Nr, panel_m, k,
              ldc);
        }
      }
    }
  }
};

}  // namespace cpu_micro_gemm

#endif
