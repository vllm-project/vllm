#ifndef CPU_MICRO_GEMM_INT8_VSX_HPP
#define CPU_MICRO_GEMM_INT8_VSX_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"

#include <altivec.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>

namespace cpu_micro_gemm {

namespace vsx_int8 {

constexpr int32_t K = 4;
constexpr int32_t Mr = 4;
constexpr int32_t Nr = 32;

// ---------------------------------------------------------------------------
// 8-Accumulator MMA 4x32 micro-GEMM (POWER10 xvi8ger4pp).
//
// Computes 4 rows x 32 columns (128 outputs) using all 8 hardware accumulators.
// Operates on pre-biased uint8 weights (B + 128) and subtracts 128 * a_sums.
// Works for all m in [1, 4] with zero scalar loop overhead.
// ---------------------------------------------------------------------------
__attribute__((target("cpu=power10"))) FORCE_INLINE void gemm_mma_4x32(
    const int8_t* __restrict__ a_packed, const uint8_t* __restrict__ b_panel,
    const int32_t* __restrict__ a_sums, int32_t* __restrict__ c_ptr,
    const int32_t m, const int32_t k_size, const int64_t ldc) {
  TORCH_CHECK(m > 0 && m <= Mr);
  TORCH_CHECK(k_size % K == 0);

  __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
  __builtin_mma_xxsetaccz(&acc0);
  __builtin_mma_xxsetaccz(&acc1);
  __builtin_mma_xxsetaccz(&acc2);
  __builtin_mma_xxsetaccz(&acc3);
  __builtin_mma_xxsetaccz(&acc4);
  __builtin_mma_xxsetaccz(&acc5);
  __builtin_mma_xxsetaccz(&acc6);
  __builtin_mma_xxsetaccz(&acc7);

  const int8_t* __restrict__ ap = a_packed;
  const uint8_t* __restrict__ bp = b_panel;

#pragma GCC unroll 4
  for (int32_t k = 0; k < k_size; k += 4) {
    const __vector unsigned char avec =
        vec_xl(0, reinterpret_cast<const unsigned char*>(ap));
    __builtin_mma_xvi8ger4pp(&acc0, avec, vec_xl(0, bp));
    __builtin_mma_xvi8ger4pp(&acc1, avec, vec_xl(16, bp));
    __builtin_mma_xvi8ger4pp(&acc2, avec, vec_xl(32, bp));
    __builtin_mma_xvi8ger4pp(&acc3, avec, vec_xl(48, bp));
    __builtin_mma_xvi8ger4pp(&acc4, avec, vec_xl(64, bp));
    __builtin_mma_xvi8ger4pp(&acc5, avec, vec_xl(80, bp));
    __builtin_mma_xvi8ger4pp(&acc6, avec, vec_xl(96, bp));
    __builtin_mma_xvi8ger4pp(&acc7, avec, vec_xl(112, bp));
    ap += 16;
    bp += 128;
  }

  __vector signed int res[4];
#define STORE_ACC(ACC, CG_IDX)                                                \
  {                                                                           \
    __builtin_mma_disassemble_acc(res, &ACC);                                 \
    for (int r = 0; r < m; ++r) {                                             \
      const __vector signed int vcorr = vec_splats(128 * a_sums[r]);          \
      const __vector signed int val = vec_sub(res[r], vcorr);                 \
      vec_xst(val, 0,                                                         \
              reinterpret_cast<signed int*>(c_ptr + r * ldc + (CG_IDX) * 4)); \
    }                                                                         \
  }
  STORE_ACC(acc0, 0);
  STORE_ACC(acc1, 1);
  STORE_ACC(acc2, 2);
  STORE_ACC(acc3, 3);
  STORE_ACC(acc4, 4);
  STORE_ACC(acc5, 5);
  STORE_ACC(acc6, 6);
  STORE_ACC(acc7, 7);
#undef STORE_ACC
}

}  // namespace vsx_int8

// =============================================================================
// MicroGemmINT8 specialisation for VSX (POWER10+)
// =============================================================================
template <typename scalar_t>
class MicroGemmINT8<cpu_utils::ISA::VSX, scalar_t> {
 public:
  static constexpr int32_t K = vsx_int8::K;    // 4
  static constexpr int32_t Mr = vsx_int8::Mr;  // 4
  static constexpr int32_t Nr = vsx_int8::Nr;  // 32
  static constexpr int32_t NrGemv = Nr;
  static constexpr int32_t MaxMSize = 8;
  static constexpr int32_t NSize = Nr;
  static constexpr int32_t WeightOCGroupSize = Nr;
  static_assert(MaxMSize % Mr == 0);

  // -------------------------------------------------------------------------
  // quantize_row: scalar_t -> INT8 + per-token scale
  // Single-pass 16-element vector conversion with aligned 16-byte stores.
  // -------------------------------------------------------------------------
  static FORCE_INLINE void quantize_row(const scalar_t* input, int8_t* output,
                                        float& scale, const int32_t size) {
    TORCH_CHECK_EQ(size % 16, 0);

    const __vector signed short z = vec_splats((signed short)0);

    auto load_lo_hi =
        [&](int32_t i) -> std::pair<__vector float, __vector float> {
      if constexpr (std::is_same_v<scalar_t, float>) {
        const float* p = reinterpret_cast<const float*>(input) + i;
        return {vec_xl(0, p), vec_xl(16, p)};
      } else {
        const signed short* p =
            reinterpret_cast<const signed short*>(input + i);
        const __vector signed short raw = (__vector signed short)vec_xl(0, p);
        return {(__vector float)vec_mergeh(z, raw),
                (__vector float)vec_mergel(z, raw)};
      }
    };

    __vector float max_v = vec_splats(0.0f);
    for (int32_t i = 0; i < size; i += 8) {
      auto [lo, hi] = load_lo_hi(i);
      max_v = vec_max(max_v, vec_abs(lo));
      max_v = vec_max(max_v, vec_abs(hi));
    }
    __vector float t = vec_max(max_v, vec_sld(max_v, max_v, 8));
    t = vec_max(t, vec_sld(t, t, 4));
    const float abs_max = std::max(vec_extract(t, 0), 1.0e-7f);
    scale = abs_max / 127.0f;
    const __vector float inv_scale = vec_splats(127.0f / abs_max);

    for (int32_t i = 0; i < size; i += 16) {
      auto [lo0, hi0] = load_lo_hi(i);
      auto [lo1, hi1] = load_lo_hi(i + 8);

      const __vector signed int i0 =
          vec_cts(vec_round(vec_mul(lo0, inv_scale)), 0);
      const __vector signed int i1 =
          vec_cts(vec_round(vec_mul(hi0, inv_scale)), 0);
      const __vector signed int i2 =
          vec_cts(vec_round(vec_mul(lo1, inv_scale)), 0);
      const __vector signed int i3 =
          vec_cts(vec_round(vec_mul(hi1, inv_scale)), 0);

      const __vector signed short s0 = vec_packs(i0, i1);
      const __vector signed short s1 = vec_packs(i2, i3);
      const __vector signed char s8 = vec_packs(s0, s1);

      vec_xst(s8, 0, reinterpret_cast<signed char*>(output + i));
    }
  }

  // -------------------------------------------------------------------------
  // dequantize_tile: INT32 accumulator -> FP32 scaled output
  // -------------------------------------------------------------------------
  static FORCE_INLINE void dequantize_tile(
      int32_t* input, float* output, const float* __restrict__ input_scales,
      const float* __restrict__ weight_scales, const int32_t m, const int32_t n,
      const int32_t stride) {
    TORCH_CHECK_EQ(n % 4, 0);
    for (int32_t mi = 0; mi < m; ++mi) {
      const __vector float a_scale = vec_splats(input_scales[mi]);
      for (int32_t ni = 0; ni < n; ni += 4) {
        const __vector signed int i32 = vec_xl(0, input + mi * stride + ni);
        const __vector float wscale = vec_xl(0, weight_scales + ni);
        vec_xst(vec_mul(vec_ctf(i32, 0), vec_mul(a_scale, wscale)), 0,
                output + mi * stride + ni);
      }
    }
  }

  // -------------------------------------------------------------------------
  // pack_input_from_rows: pack rows into 4-row MMA tiles and compute row sums
  // Header: 64 bytes (contains int32 a_sums[0..7])
  // Data:   a_packed + 64 (contains 4x4 tiles of A)
  // -------------------------------------------------------------------------
  static void pack_input_from_rows(const int8_t* const* __restrict__ rows,
                                   int8_t* __restrict__ a_packed,
                                   const int32_t m, const int32_t k) {
    TORCH_CHECK(m > 0 && m <= MaxMSize);
    TORCH_CHECK_EQ(k % K, 0);

    int32_t* a_sums = reinterpret_cast<int32_t*>(a_packed);
    for (int32_t r = 0; r < 8; ++r) a_sums[r] = 0;

    int8_t* a_data = a_packed + 64;

    for (int32_t row_base = 0; row_base < m; row_base += Mr) {
      const int32_t panel_m = std::min(Mr, m - row_base);
      const int8_t* const* panel_rows = rows + row_base;
      int32_t* panel_sums = a_sums + row_base;
      int8_t* dst_panel = a_data + (row_base / Mr) * (k * Mr);

      for (int32_t ki = 0; ki < k; ki += 4) {
        int8_t* dst = dst_panel + (ki / 4) * 16;
        for (int32_t r = 0; r < 4; ++r) {
          if (r < panel_m) {
            const int8_t* src = panel_rows[r] + ki;
            dst[r * 4 + 0] = src[0];
            dst[r * 4 + 1] = src[1];
            dst[r * 4 + 2] = src[2];
            dst[r * 4 + 3] = src[3];
            panel_sums[r] +=
                static_cast<int32_t>(src[0]) + src[1] + src[2] + src[3];
          } else {
            dst[r * 4 + 0] = 0;
            dst[r * 4 + 1] = 0;
            dst[r * 4 + 2] = 0;
            dst[r * 4 + 3] = 0;
          }
        }
      }
    }
  }

  // -------------------------------------------------------------------------
  // pack_weight: [output_size, input_size] INT8 -> packed layout
  //   Pre-biases with +128 in [N/32][K/4][8 col-groups][4 cols x 4 k] layout
  // -------------------------------------------------------------------------
  static void pack_weight(const int8_t* __restrict__ weight,
                          int8_t* __restrict__ packed_weight,
                          const int32_t output_size, const int32_t input_size) {
    TORCH_CHECK_EQ(output_size % NSize, 0);
    TORCH_CHECK_EQ(input_size % K, 0);

    uint8_t* dst_base = reinterpret_cast<uint8_t*>(packed_weight);

    for (int32_t o = 0; o < output_size; o += Nr) {
      uint8_t* dst_panel = dst_base + o * input_size;
      for (int32_t k = 0; k < input_size; k += 4) {
        for (int32_t cg = 0; cg < 8; ++cg) {
          uint8_t* dst = dst_panel + (k / 4) * 128 + cg * 16;
          for (int32_t col = 0; col < 4; ++col) {
            for (int32_t ki = 0; ki < 4; ++ki) {
              const int8_t w = weight[(o + cg * 4 + col) * input_size + k + ki];
              dst[col * 4 + ki] =
                  static_cast<uint8_t>(static_cast<int>(w) + 128);
            }
          }
        }
      }
    }
  }

  // -------------------------------------------------------------------------
  // gemm: 8-accumulator MMA GEMM over 32-column tiles
  // -------------------------------------------------------------------------
  void gemm(const int8_t* __restrict__ a_packed,
            const int8_t* __restrict__ b_packed, int32_t* __restrict__ c,
            const int32_t m, const int32_t k, const int64_t b_n_group_stride,
            const int64_t ldc) const {
    TORCH_CHECK(m > 0 && m <= MaxMSize);
    TORCH_CHECK_EQ(k % K, 0);
    (void)b_n_group_stride;

    const int32_t* a_sums = reinterpret_cast<const int32_t*>(a_packed);
    const int8_t* a_data = a_packed + 64;
    const uint8_t* b_u8 = reinterpret_cast<const uint8_t*>(b_packed);

    for (int32_t n_idx = 0; n_idx < NSize; n_idx += Nr) {
      const uint8_t* b_panel = b_u8 + n_idx * k;

      for (int32_t row_base = 0; row_base < m; row_base += Mr) {
        const int32_t panel_m = std::min(Mr, m - row_base);
        const int8_t* a_panel = a_data + (row_base / Mr) * (k * Mr);
        const int32_t* panel_sums = a_sums + row_base;

        vsx_int8::gemm_mma_4x32(a_panel, b_panel, panel_sums,
                                c + row_base * ldc + n_idx, panel_m, k, ldc);
      }
    }
  }
};

}  // namespace cpu_micro_gemm

#endif  // CPU_MICRO_GEMM_INT8_VSX_HPP
