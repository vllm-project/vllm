#ifndef CPU_MICRO_GEMM_NEON_HPP
#define CPU_MICRO_GEMM_NEON_HPP

#include <algorithm>
#include <cstdint>

#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"

#include <arm_bf16.h>
#include <arm_neon.h>

namespace cpu_micro_gemm {

namespace {

constexpr int32_t K = 4;
constexpr int32_t Cols = 2;
constexpr int32_t TileSize = K * Cols;
constexpr int32_t Mr = 8;
constexpr int32_t Nr = 8;
constexpr int32_t Nr_gemv = 16;

// a = [a0, a1, a2, a3], b = [b0, b1, b2, b3] -> [a0, a1, b0, b1]
FORCE_INLINE float32x4_t zip1_f32x4(const float32x4_t a, const float32x4_t b) {
  return vreinterpretq_f32_f64(
      vzip1q_f64(vreinterpretq_f64_f32(a), vreinterpretq_f64_f32(b)));
}

// a = [a0, a1, a2, a3], b = [b0, b1, b2, b3] -> [a2, a3, b2, b3]
FORCE_INLINE float32x4_t zip2_f32x4(const float32x4_t a, const float32x4_t b) {
  return vreinterpretq_f32_f64(
      vzip2q_f64(vreinterpretq_f64_f32(a), vreinterpretq_f64_f32(b)));
}

FORCE_INLINE void init_acc_rowpair(float32x4_t& acc01, float32x4_t& acc23,
                                   float32x4_t& acc45, float32x4_t& acc67,
                                   const float* __restrict__ c_ptr,
                                   const int64_t ldc, const int32_t m_rows,
                                   const bool accum_c) {
  if (!accum_c || m_rows == 0) {
    acc01 = vdupq_n_f32(0.0f);
    acc23 = vdupq_n_f32(0.0f);
    acc45 = vdupq_n_f32(0.0f);
    acc67 = vdupq_n_f32(0.0f);
    return;
  }

  const float32x4_t row0_0123 = vld1q_f32(c_ptr);
  const float32x4_t row0_4567 = vld1q_f32(c_ptr + 4);
  const float32x4_t row1_0123 =
      (m_rows == 2) ? vld1q_f32(c_ptr + ldc) : vdupq_n_f32(0.0f);
  const float32x4_t row1_4567 =
      (m_rows == 2) ? vld1q_f32(c_ptr + ldc + 4) : vdupq_n_f32(0.0f);

  acc01 = zip1_f32x4(row0_0123, row1_0123);
  acc23 = zip2_f32x4(row0_0123, row1_0123);
  acc45 = zip1_f32x4(row0_4567, row1_4567);
  acc67 = zip2_f32x4(row0_4567, row1_4567);
}

FORCE_INLINE void store_acc_rowpair(const float32x4_t acc01,
                                    const float32x4_t acc23,
                                    const float32x4_t acc45,
                                    const float32x4_t acc67,
                                    float* __restrict__ c_ptr,
                                    const int64_t ldc, const int32_t m_rows) {
  if (m_rows == 0) {
    return;
  }

  vst1q_f32(c_ptr, zip1_f32x4(acc01, acc23));
  vst1q_f32(c_ptr + 4, zip1_f32x4(acc45, acc67));

  if (m_rows == 2) {
    vst1q_f32(c_ptr + ldc, zip2_f32x4(acc01, acc23));
    vst1q_f32(c_ptr + ldc + 4, zip2_f32x4(acc45, acc67));
  }
}

FORCE_INLINE void gemm_micro_bfmmla_8x8_packed_a(
    const bfloat16_t* __restrict__ a_packed,
    const bfloat16_t* __restrict__ b_packed, float* __restrict__ c_ptr,
    const int32_t m, const int32_t k_size, const int64_t ldc,
    const bool accum_c) {
  float32x4_t acc0101, acc0123, acc0145, acc0167;
  float32x4_t acc2301, acc2323, acc2345, acc2367;
  float32x4_t acc4501, acc4523, acc4545, acc4567;
  float32x4_t acc6701, acc6723, acc6745, acc6767;

  init_acc_rowpair(acc0101, acc0123, acc0145, acc0167, c_ptr, ldc,
                   std::min(2, m), accum_c);
  init_acc_rowpair(acc2301, acc2323, acc2345, acc2367, c_ptr + 2 * ldc, ldc,
                   std::min(2, std::max(0, m - 2)), accum_c);
  init_acc_rowpair(acc4501, acc4523, acc4545, acc4567, c_ptr + 4 * ldc, ldc,
                   std::min(2, std::max(0, m - 4)), accum_c);
  init_acc_rowpair(acc6701, acc6723, acc6745, acc6767, c_ptr + 6 * ldc, ldc,
                   std::min(2, std::max(0, m - 6)), accum_c);

  const bfloat16_t* __restrict__ a_tile = a_packed;
  const bfloat16_t* __restrict__ b_tile = b_packed;

#pragma GCC unroll 8
  for (int32_t k_idx = 0; k_idx < k_size; k_idx += K) {
    const bfloat16x8_t a_tile01 = vld1q_bf16(a_tile);
    const bfloat16x8_t a_tile23 = vld1q_bf16(a_tile + TileSize);
    const bfloat16x8_t a_tile45 = vld1q_bf16(a_tile + 2 * TileSize);
    const bfloat16x8_t a_tile67 = vld1q_bf16(a_tile + 3 * TileSize);

    const bfloat16x8_t b_tile01 = vld1q_bf16(b_tile);
    const bfloat16x8_t b_tile23 = vld1q_bf16(b_tile + TileSize);
    const bfloat16x8_t b_tile45 = vld1q_bf16(b_tile + 2 * TileSize);
    const bfloat16x8_t b_tile67 = vld1q_bf16(b_tile + 3 * TileSize);

    acc0101 = vbfmmlaq_f32(acc0101, a_tile01, b_tile01);
    acc2301 = vbfmmlaq_f32(acc2301, a_tile23, b_tile01);
    acc4501 = vbfmmlaq_f32(acc4501, a_tile45, b_tile01);
    acc6701 = vbfmmlaq_f32(acc6701, a_tile67, b_tile01);

    acc0123 = vbfmmlaq_f32(acc0123, a_tile01, b_tile23);
    acc2323 = vbfmmlaq_f32(acc2323, a_tile23, b_tile23);
    acc4523 = vbfmmlaq_f32(acc4523, a_tile45, b_tile23);
    acc6723 = vbfmmlaq_f32(acc6723, a_tile67, b_tile23);

    acc0145 = vbfmmlaq_f32(acc0145, a_tile01, b_tile45);
    acc2345 = vbfmmlaq_f32(acc2345, a_tile23, b_tile45);
    acc4545 = vbfmmlaq_f32(acc4545, a_tile45, b_tile45);
    acc6745 = vbfmmlaq_f32(acc6745, a_tile67, b_tile45);

    acc0167 = vbfmmlaq_f32(acc0167, a_tile01, b_tile67);
    acc2367 = vbfmmlaq_f32(acc2367, a_tile23, b_tile67);
    acc4567 = vbfmmlaq_f32(acc4567, a_tile45, b_tile67);
    acc6767 = vbfmmlaq_f32(acc6767, a_tile67, b_tile67);

    a_tile += 4 * TileSize;
    b_tile += Nr * K;
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

FORCE_INLINE void gemm_micro_bfmmla_4x16_packed_a(
    const bfloat16_t* __restrict__ a_packed,
    const bfloat16_t* __restrict__ b_packed, float* __restrict__ c_ptr,
    const int32_t m, const int32_t k_size, const int64_t b_n_group_stride,
    const int64_t ldc, const bool accum_c) {
  const int32_t m_rows_01 = std::min(2, m);
  const int32_t m_rows_23 = std::min(2, std::max(0, m - 2));

  float32x4_t acc0101, acc0123, acc0145, acc0167;
  float32x4_t acc2301, acc2323, acc2345, acc2367;
  float32x4_t acc0189, acc011011, acc011213, acc011415;
  float32x4_t acc2389, acc231011, acc231213, acc231415;

  init_acc_rowpair(acc0101, acc0123, acc0145, acc0167, c_ptr, ldc, m_rows_01,
                   accum_c);
  init_acc_rowpair(acc2301, acc2323, acc2345, acc2367, c_ptr + 2 * ldc, ldc,
                   m_rows_23, accum_c);
  init_acc_rowpair(acc0189, acc011011, acc011213, acc011415, c_ptr + 8, ldc,
                   m_rows_01, accum_c);
  init_acc_rowpair(acc2389, acc231011, acc231213, acc231415,
                   c_ptr + 2 * ldc + 8, ldc, m_rows_23, accum_c);

  const bfloat16_t* __restrict__ a_tile = a_packed;
  const bfloat16_t* __restrict__ b_tile0 = b_packed;
  const bfloat16_t* __restrict__ b_tile1 = b_packed + b_n_group_stride;

#pragma GCC unroll 8
  for (int32_t k_idx = 0; k_idx < k_size; k_idx += K) {
    const bfloat16x8_t a_tile01 = vld1q_bf16(a_tile);
    const bfloat16x8_t a_tile23 = vld1q_bf16(a_tile + TileSize);
    const bfloat16x8_t b_tile01 = vld1q_bf16(b_tile0);
    const bfloat16x8_t b_tile23 = vld1q_bf16(b_tile0 + TileSize);
    const bfloat16x8_t b_tile45 = vld1q_bf16(b_tile0 + 2 * TileSize);
    const bfloat16x8_t b_tile67 = vld1q_bf16(b_tile0 + 3 * TileSize);
    const bfloat16x8_t b_tile89 = vld1q_bf16(b_tile1);
    const bfloat16x8_t b_tile1011 = vld1q_bf16(b_tile1 + TileSize);
    const bfloat16x8_t b_tile1213 = vld1q_bf16(b_tile1 + 2 * TileSize);
    const bfloat16x8_t b_tile1415 = vld1q_bf16(b_tile1 + 3 * TileSize);

    acc0101 = vbfmmlaq_f32(acc0101, a_tile01, b_tile01);
    acc2301 = vbfmmlaq_f32(acc2301, a_tile23, b_tile01);
    acc0123 = vbfmmlaq_f32(acc0123, a_tile01, b_tile23);
    acc2323 = vbfmmlaq_f32(acc2323, a_tile23, b_tile23);

    acc0145 = vbfmmlaq_f32(acc0145, a_tile01, b_tile45);
    acc2345 = vbfmmlaq_f32(acc2345, a_tile23, b_tile45);
    acc0167 = vbfmmlaq_f32(acc0167, a_tile01, b_tile67);
    acc2367 = vbfmmlaq_f32(acc2367, a_tile23, b_tile67);

    acc0189 = vbfmmlaq_f32(acc0189, a_tile01, b_tile89);
    acc2389 = vbfmmlaq_f32(acc2389, a_tile23, b_tile89);
    acc011011 = vbfmmlaq_f32(acc011011, a_tile01, b_tile1011);
    acc231011 = vbfmmlaq_f32(acc231011, a_tile23, b_tile1011);

    acc011213 = vbfmmlaq_f32(acc011213, a_tile01, b_tile1213);
    acc231213 = vbfmmlaq_f32(acc231213, a_tile23, b_tile1213);
    acc011415 = vbfmmlaq_f32(acc011415, a_tile01, b_tile1415);
    acc231415 = vbfmmlaq_f32(acc231415, a_tile23, b_tile1415);

    a_tile += 2 * TileSize;
    b_tile0 += Nr * K;
    b_tile1 += Nr * K;
  }

  store_acc_rowpair(acc0101, acc0123, acc0145, acc0167, c_ptr, ldc, m_rows_01);
  store_acc_rowpair(acc2301, acc2323, acc2345, acc2367, c_ptr + 2 * ldc, ldc,
                    m_rows_23);
  store_acc_rowpair(acc0189, acc011011, acc011213, acc011415, c_ptr + 8, ldc,
                    m_rows_01);
  store_acc_rowpair(acc2389, acc231011, acc231213, acc231415,
                    c_ptr + 2 * ldc + 8, ldc, m_rows_23);
}

}  // namespace

template <typename scalar_t>
class MicroGemm<cpu_utils::ISA::NEON, scalar_t> {
 public:
  static constexpr int32_t MaxMSize = 8;
  static constexpr int32_t NSize = 32;
  static constexpr int32_t WeightOCGroupSize = Nr;
  static constexpr bool PackA = false;

 public:
  void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    TORCH_CHECK(false, "NEON BFMMLA MicroGemm only supports bfloat16.");
  }

  static void pack_weight(const scalar_t* __restrict__ /*weight*/,
                          scalar_t* __restrict__ /*packed_weight*/,
                          const int32_t /*output_size*/,
                          const int32_t /*input_size*/) {
    TORCH_CHECK(false, "NEON BFMMLA MicroGemm only supports bfloat16.");
  }
};

template <>
class MicroGemm<cpu_utils::ISA::NEON, c10::BFloat16> {
 public:
  using scalar_t = c10::BFloat16;

  static constexpr int32_t MaxMSize = 8;
  static constexpr int32_t NSize = 32;
  static constexpr int32_t WeightOCGroupSize = Nr;
  static constexpr bool PackA = true;

 public:
  // physical layout [
  //  M / 8; Mr is 8
  //  K / 4; K for bfmmla is 4
  //  4,   ; 4 row-pairs for each 8 rows
  //  2,   ; row-pair is 2 rows
  //  4    ; 4 elements per row
  // ]

  static void pack_input_from_rows(const scalar_t* const* __restrict__ rows,
                                   scalar_t* __restrict__ a_packed,
                                   const int32_t m, const int32_t k) {
    TORCH_CHECK(m > 0 && m <= MaxMSize);
    TORCH_CHECK_EQ(k % K, 0);

    auto* __restrict__ out = reinterpret_cast<bfloat16_t*>(a_packed);
    const bfloat16x8_t zero_q = vdupq_n_bf16(bfloat16_t{});
    const bfloat16x4_t zero = vget_low_bf16(zero_q);

    for (int32_t row_base = 0; row_base < m; row_base += Mr) {
      const int32_t actual_m = std::min(Mr, m - row_base);
      const bfloat16_t* __restrict__ row[Mr];
      for (int32_t i = 0; i < actual_m; ++i) {
        row[i] = reinterpret_cast<const bfloat16_t*>(rows[row_base + i]);
      }

      if (actual_m == 8) {
        int32_t k_idx = 0;
        for (; k_idx + 8 <= k; k_idx += 8) {
          bfloat16_t* __restrict__ block0 = out;
          bfloat16_t* __restrict__ block1 = out + 4 * TileSize;

          bfloat16x8_t a0 = vld1q_bf16(row[0] + k_idx);
          bfloat16x8_t a1 = vld1q_bf16(row[1] + k_idx);
          vst1q_bf16(block0,
                     vcombine_bf16(vget_low_bf16(a0), vget_low_bf16(a1)));
          vst1q_bf16(block1,
                     vcombine_bf16(vget_high_bf16(a0), vget_high_bf16(a1)));

          a0 = vld1q_bf16(row[2] + k_idx);
          a1 = vld1q_bf16(row[3] + k_idx);
          vst1q_bf16(block0 + TileSize,
                     vcombine_bf16(vget_low_bf16(a0), vget_low_bf16(a1)));
          vst1q_bf16(block1 + TileSize,
                     vcombine_bf16(vget_high_bf16(a0), vget_high_bf16(a1)));

          a0 = vld1q_bf16(row[4] + k_idx);
          a1 = vld1q_bf16(row[5] + k_idx);
          vst1q_bf16(block0 + 2 * TileSize,
                     vcombine_bf16(vget_low_bf16(a0), vget_low_bf16(a1)));
          vst1q_bf16(block1 + 2 * TileSize,
                     vcombine_bf16(vget_high_bf16(a0), vget_high_bf16(a1)));

          a0 = vld1q_bf16(row[6] + k_idx);
          a1 = vld1q_bf16(row[7] + k_idx);
          vst1q_bf16(block0 + 3 * TileSize,
                     vcombine_bf16(vget_low_bf16(a0), vget_low_bf16(a1)));
          vst1q_bf16(block1 + 3 * TileSize,
                     vcombine_bf16(vget_high_bf16(a0), vget_high_bf16(a1)));

          out += 8 * TileSize;
        }

        for (; k_idx < k; k_idx += K) {
          bfloat16x4_t a0 = vld1_bf16(row[0] + k_idx);
          bfloat16x4_t a1 = vld1_bf16(row[1] + k_idx);
          vst1q_bf16(out, vcombine_bf16(a0, a1));

          a0 = vld1_bf16(row[2] + k_idx);
          a1 = vld1_bf16(row[3] + k_idx);
          vst1q_bf16(out + TileSize, vcombine_bf16(a0, a1));

          a0 = vld1_bf16(row[4] + k_idx);
          a1 = vld1_bf16(row[5] + k_idx);
          vst1q_bf16(out + 2 * TileSize, vcombine_bf16(a0, a1));

          a0 = vld1_bf16(row[6] + k_idx);
          a1 = vld1_bf16(row[7] + k_idx);
          vst1q_bf16(out + 3 * TileSize, vcombine_bf16(a0, a1));

          out += 4 * TileSize;
        }
        continue;
      }

      if (actual_m == 4) {
        int32_t k_idx = 0;
        for (; k_idx + 8 <= k; k_idx += 8) {
          bfloat16_t* __restrict__ block0 = out;
          bfloat16_t* __restrict__ block1 = out + 2 * TileSize;

          bfloat16x8_t a0 = vld1q_bf16(row[0] + k_idx);
          bfloat16x8_t a1 = vld1q_bf16(row[1] + k_idx);
          vst1q_bf16(block0,
                     vcombine_bf16(vget_low_bf16(a0), vget_low_bf16(a1)));
          vst1q_bf16(block1,
                     vcombine_bf16(vget_high_bf16(a0), vget_high_bf16(a1)));

          a0 = vld1q_bf16(row[2] + k_idx);
          a1 = vld1q_bf16(row[3] + k_idx);
          vst1q_bf16(block0 + TileSize,
                     vcombine_bf16(vget_low_bf16(a0), vget_low_bf16(a1)));
          vst1q_bf16(block1 + TileSize,
                     vcombine_bf16(vget_high_bf16(a0), vget_high_bf16(a1)));

          out += 4 * TileSize;
        }

        for (; k_idx < k; k_idx += K) {
          bfloat16x4_t a0 = vld1_bf16(row[0] + k_idx);
          bfloat16x4_t a1 = vld1_bf16(row[1] + k_idx);
          vst1q_bf16(out, vcombine_bf16(a0, a1));

          a0 = vld1_bf16(row[2] + k_idx);
          a1 = vld1_bf16(row[3] + k_idx);
          vst1q_bf16(out + TileSize, vcombine_bf16(a0, a1));

          out += 2 * TileSize;
        }
        continue;
      }

      const int32_t row_pair_count = (actual_m <= 4) ? 2 : Mr / 2;

      int32_t k_idx = 0;
      for (; k_idx + 8 <= k; k_idx += 8) {
        bfloat16_t* __restrict__ block0 = out;
        bfloat16_t* __restrict__ block1 = out + row_pair_count * TileSize;

        bfloat16x8_t a0 = vld1q_bf16(row[0] + k_idx);
        bfloat16x8_t a1 = (actual_m > 1) ? vld1q_bf16(row[1] + k_idx) : zero_q;
        vst1q_bf16(block0, vcombine_bf16(vget_low_bf16(a0), vget_low_bf16(a1)));
        vst1q_bf16(block1,
                   vcombine_bf16(vget_high_bf16(a0), vget_high_bf16(a1)));

        a0 = (actual_m > 2) ? vld1q_bf16(row[2] + k_idx) : zero_q;
        a1 = (actual_m > 3) ? vld1q_bf16(row[3] + k_idx) : zero_q;
        vst1q_bf16(block0 + TileSize,
                   vcombine_bf16(vget_low_bf16(a0), vget_low_bf16(a1)));
        vst1q_bf16(block1 + TileSize,
                   vcombine_bf16(vget_high_bf16(a0), vget_high_bf16(a1)));

        if (actual_m > 4) {
          a0 = vld1q_bf16(row[4] + k_idx);
          a1 = (actual_m > 5) ? vld1q_bf16(row[5] + k_idx) : zero_q;
          vst1q_bf16(block0 + 2 * TileSize,
                     vcombine_bf16(vget_low_bf16(a0), vget_low_bf16(a1)));
          vst1q_bf16(block1 + 2 * TileSize,
                     vcombine_bf16(vget_high_bf16(a0), vget_high_bf16(a1)));

          a0 = (actual_m > 6) ? vld1q_bf16(row[6] + k_idx) : zero_q;
          a1 = (actual_m > 7) ? vld1q_bf16(row[7] + k_idx) : zero_q;
          vst1q_bf16(block0 + 3 * TileSize,
                     vcombine_bf16(vget_low_bf16(a0), vget_low_bf16(a1)));
          vst1q_bf16(block1 + 3 * TileSize,
                     vcombine_bf16(vget_high_bf16(a0), vget_high_bf16(a1)));
        }

        out += 2 * row_pair_count * TileSize;
      }

      for (; k_idx < k; k_idx += K) {
        bfloat16x4_t a0 = vld1_bf16(row[0] + k_idx);
        bfloat16x4_t a1 = (actual_m > 1) ? vld1_bf16(row[1] + k_idx) : zero;
        vst1q_bf16(out, vcombine_bf16(a0, a1));

        a0 = (actual_m > 2) ? vld1_bf16(row[2] + k_idx) : zero;
        a1 = (actual_m > 3) ? vld1_bf16(row[3] + k_idx) : zero;
        vst1q_bf16(out + TileSize, vcombine_bf16(a0, a1));

        if (actual_m > 4) {
          a0 = vld1_bf16(row[4] + k_idx);
          a1 = (actual_m > 5) ? vld1_bf16(row[5] + k_idx) : zero;
          vst1q_bf16(out + 2 * TileSize, vcombine_bf16(a0, a1));

          a0 = (actual_m > 6) ? vld1_bf16(row[6] + k_idx) : zero;
          a1 = (actual_m > 7) ? vld1_bf16(row[7] + k_idx) : zero;
          vst1q_bf16(out + 3 * TileSize, vcombine_bf16(a0, a1));
        }
        out += row_pair_count * TileSize;
      }
    }
  }

  void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    (void)lda;  // A is packed, so lda is not needed
    TORCH_CHECK_EQ(k % K, 0);

    for (int32_t n_idx = 0; n_idx < NSize; n_idx += Nr_gemv) {
      const bfloat16_t* __restrict__ b_panel =
          reinterpret_cast<const bfloat16_t*>(b_ptr) + n_idx * k;

      for (int32_t row_base = 0; row_base < m; row_base += Mr) {
        const int32_t panel_m = std::min(Mr, m - row_base);
        const bfloat16_t* __restrict__ a_panel =
            reinterpret_cast<const bfloat16_t*>(a_ptr) + row_base * k;
        float* __restrict__ c_panel = c_ptr + row_base * ldc + n_idx;

        if (panel_m <= 4) {
          gemm_micro_bfmmla_4x16_packed_a(a_panel, b_panel, c_panel, panel_m, k,
                                          b_n_group_stride, ldc, accum_c);
        } else {
          gemm_micro_bfmmla_8x8_packed_a(a_panel, b_panel, c_panel, panel_m, k,
                                         ldc, accum_c);
          gemm_micro_bfmmla_8x8_packed_a(a_panel, b_panel + b_n_group_stride,
                                         c_panel + Nr, panel_m, k, ldc,
                                         accum_c);
        }
      }
    }
  }

  // physical layout [
  //  N / 8; Nr is 8
  //  K / 4; K for bfmmla is 4
  //  4,   ; 4 col-pairs for each 8 cols
  //  2,   ; col-pair is 2 cols
  //  4    ; 4 elements per col
  // ]
  static void pack_weight(const c10::BFloat16* __restrict__ weight,
                          c10::BFloat16* __restrict__ packed_weight,
                          const int32_t output_size, const int32_t input_size) {
    TORCH_CHECK_EQ(output_size % NSize, 0);
    TORCH_CHECK_EQ(input_size % K, 0);

    for (int32_t o_idx = 0; o_idx < output_size; o_idx += Nr) {
      c10::BFloat16* __restrict__ dst = packed_weight + o_idx * input_size;
      for (int32_t k_idx = 0; k_idx < input_size; k_idx += K) {
        for (int32_t pair_idx = 0; pair_idx < Nr; pair_idx += Cols) {
          const c10::BFloat16* __restrict__ row0 =
              weight + (o_idx + pair_idx) * input_size;
          const c10::BFloat16* __restrict__ row1 = row0 + input_size;
          dst[0] = row0[k_idx + 0];
          dst[1] = row0[k_idx + 1];
          dst[2] = row0[k_idx + 2];
          dst[3] = row0[k_idx + 3];
          dst[4] = row1[k_idx + 0];
          dst[5] = row1[k_idx + 1];
          dst[6] = row1[k_idx + 2];
          dst[7] = row1[k_idx + 3];
          dst += TileSize;
        }
      }
    }
  }
};

}  // namespace cpu_micro_gemm

#endif
