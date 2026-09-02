#ifndef CPU_MICRO_GEMM_INT8_VSX_HPP
#define CPU_MICRO_GEMM_INT8_VSX_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"

#include <altivec.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>

namespace cpu_micro_gemm {

namespace vsx_int8 {

constexpr int32_t K = 8;
constexpr int32_t Mr = 4;
constexpr int32_t Nr = 16;

// ---------------------------------------------------------------------------
// Load 4 lanes of FP32 from scalar_t, widening BF16 as needed.
// ---------------------------------------------------------------------------
FORCE_INLINE __vector float load4_f32_lo(const float* p) { return vec_xl(0, p); }
FORCE_INLINE __vector float load4_f32_hi(const float* p) { return vec_xl(16, p); }

FORCE_INLINE __vector float load4_bf16_lo(const c10::BFloat16* p) {
  const __vector signed short raw =
      (__vector signed short)vec_xl(0, (const signed short*)p);
  const __vector signed short z = vec_splats((signed short)0);
  return (__vector float)vec_mergeh(z, raw);
}

FORCE_INLINE __vector float load4_bf16_hi(const c10::BFloat16* p) {
  const __vector signed short raw =
      (__vector signed short)vec_xl(0, (const signed short*)p);
  const __vector signed short z = vec_splats((signed short)0);
  return (__vector float)vec_mergel(z, raw);
}

// ---------------------------------------------------------------------------
// VSX scalar 4x16 micro-GEMM: used for m=1,2.
//
// A: [m][k_size] row-major int8
// B: [Nr/4 col-groups][k_size][4 interleaved columns]
// C: [m][ldc] int32 output (overwrite)
// ---------------------------------------------------------------------------
FORCE_INLINE void gemm_vsx_4x16(
    const int8_t* __restrict__ a_packed,
    const int8_t* __restrict__ b_packed,
    int32_t* __restrict__ c_ptr,
    const int32_t m,
    const int32_t k_size,
    const int64_t ldc) {
  __vector signed int acc[Mr][Nr / 4];
  for (int r = 0; r < Mr; ++r)
    for (int c = 0; c < Nr / 4; ++c)
      acc[r][c] = vec_splats((signed int)0);

  for (int32_t k = 0; k < k_size; k += K) {
    for (int cg = 0; cg < Nr / 4; ++cg) {
      const int8_t* bp = b_packed + cg * k_size * 4 + k * 4;
      for (int r = 0; r < m; ++r) {
        const signed char* ap =
            reinterpret_cast<const signed char*>(a_packed + r * k_size + k);
        __vector signed int partial = vec_splats((signed int)0);
        for (int ki = 0; ki < K; ++ki) {
          const signed char* bki = reinterpret_cast<const signed char*>(bp) + ki * 4;
          const int32_t a_val = static_cast<int32_t>(ap[ki]);
          __vector signed int contrib = {
              a_val * static_cast<int32_t>(bki[0]),
              a_val * static_cast<int32_t>(bki[1]),
              a_val * static_cast<int32_t>(bki[2]),
              a_val * static_cast<int32_t>(bki[3])};
          partial = vec_add(partial, contrib);
        }
        acc[r][cg] = vec_add(acc[r][cg], partial);
      }
    }
  }

  for (int r = 0; r < m; ++r)
    for (int cg = 0; cg < Nr / 4; ++cg)
      vec_xst(acc[r][cg], 0, reinterpret_cast<signed int*>(c_ptr + r * ldc + cg * 4));
}

// ---------------------------------------------------------------------------
// MMA 4x16 micro-GEMM (POWER10 pmxvi8ger4pp), used for m=3..8 panels.
//
// xvi8ger4pp treats both operands as unsigned INT8, so B is biased by +128
// before the GER call and the bias is subtracted afterward:
//   A * B_s8 = A * (B_s8 + 128) - 128 * A = MMA(A, B_u8) - 128 * sum_k(A)
// ---------------------------------------------------------------------------
__attribute__((target("cpu=power10")))
FORCE_INLINE void gemm_mma_4x16(
    const int8_t* __restrict__ a_packed,
    const int8_t* __restrict__ b_packed,
    int32_t* __restrict__ c_ptr,
    const int32_t m,
    const int32_t k_size,
    const int64_t ldc) {
  TORCH_CHECK(m > 0 && m <= Mr);
  TORCH_CHECK(k_size % K == 0);

  // One accumulator per 4-column group: acc[0]->cols 0-3, acc[1]->cols 4-7, ...
  __vector_quad acc[Nr / 4];
  for (int cg = 0; cg < Nr / 4; ++cg)
    __builtin_mma_xxsetaccz(&acc[cg]);

  int32_t a_sum[Mr] = {0, 0, 0, 0};

  for (int32_t kbase = 0; kbase < k_size; kbase += 4) {
    alignas(16) unsigned char avec_data[16] = {};
    for (int row = 0; row < m; ++row) {
      const int8_t* ap = a_packed + row * k_size + kbase;
      for (int ki = 0; ki < 4; ++ki) {
        const int8_t value = ap[ki];
        avec_data[row * 4 + ki] = static_cast<unsigned char>(value);
        a_sum[row] += static_cast<int32_t>(value);
      }
    }
    const __vector unsigned char avec = vec_xl(0, avec_data);

    for (int cg = 0; cg < Nr / 4; ++cg) {
      alignas(16) unsigned char bvec_data[16] = {};
      const int8_t* bp = b_packed + cg * k_size * 4 + kbase * 4;
      // Packed B is [k][4 cols]; transpose to [col][4 k] and bias to unsigned.
      for (int col = 0; col < 4; ++col) {
        for (int ki = 0; ki < 4; ++ki) {
          const int unsigned_value = static_cast<int>(bp[ki * 4 + col]) + 128;
          bvec_data[col * 4 + ki] = static_cast<unsigned char>(unsigned_value);
        }
      }
      const __vector unsigned char bvec = vec_xl(0, bvec_data);
      __builtin_mma_xvi8ger4pp(&acc[cg], avec, bvec);
    }
  }

  for (int cg = 0; cg < Nr / 4; ++cg) {
    __vector signed int result[4];
    __builtin_mma_disassemble_acc(result, &acc[cg]);
    alignas(16) int32_t mma_result[16];
    for (int row = 0; row < 4; ++row)
      vec_xst(result[row], 0, mma_result + row * 4);

    for (int row = 0; row < m; ++row) {
      const int32_t correction = 128 * a_sum[row];
      for (int col = 0; col < 4; ++col)
        c_ptr[row * ldc + cg * 4 + col] = mma_result[row * 4 + col] - correction;
    }
  }
}

}  // namespace vsx_int8

// =============================================================================
// MicroGemmINT8 specialisation for VSX (POWER10+)
// =============================================================================
template <typename scalar_t>
class MicroGemmINT8<cpu_utils::ISA::VSX, scalar_t> {
 public:
  static constexpr int32_t K                 = vsx_int8::K;   // 8
  static constexpr int32_t Mr                = vsx_int8::Mr;  // 4
  static constexpr int32_t Nr                = vsx_int8::Nr;  // 16
  static constexpr int32_t NrGemv            = Nr;
  static constexpr int32_t MaxMSize          = 8;
  static constexpr int32_t NSize             = Nr;
  static constexpr int32_t WeightOCGroupSize = Nr;
  static_assert(MaxMSize % Mr == 0);

  // -------------------------------------------------------------------------
  // quantize_row: scalar_t -> INT8 + per-token scale
  // -------------------------------------------------------------------------
  static FORCE_INLINE void quantize_row(
      const scalar_t* input, int8_t* output, float& scale, const int32_t size) {
    TORCH_CHECK_EQ(size % K, 0);

    auto load_lo = [&](int i) -> __vector float {
      if constexpr (std::is_same_v<scalar_t, float>)
        return vsx_int8::load4_f32_lo(reinterpret_cast<const float*>(input) + i);
      else
        return vsx_int8::load4_bf16_lo(input + i);
    };
    auto load_hi = [&](int i) -> __vector float {
      if constexpr (std::is_same_v<scalar_t, float>)
        return vsx_int8::load4_f32_hi(reinterpret_cast<const float*>(input) + i);
      else
        return vsx_int8::load4_bf16_hi(input + i);
    };

    __vector float max_vec = vec_splats(0.0f);
    for (int32_t i = 0; i < size; i += K) {
      max_vec = vec_max(max_vec, vec_abs(load_lo(i)));
      max_vec = vec_max(max_vec, vec_abs(load_hi(i)));
    }
    __vector float t = vec_max(max_vec, vec_sld(max_vec, max_vec, 8));
    t = vec_max(t, vec_sld(t, t, 4));
    const float abs_max = std::max(vec_extract(t, 0), 1.0e-7f);
    scale = abs_max / 127.0f;
    const __vector float inv_scale = vec_splats(127.0f / abs_max);

    for (int32_t i = 0; i < size; i += K) {
      __vector signed int lo_i = vec_cts(vec_mul(load_lo(i), inv_scale), 0);
      __vector signed int hi_i = vec_cts(vec_mul(load_hi(i), inv_scale), 0);
      __vector signed short s16 = vec_packs(lo_i, hi_i);
      __vector signed char  s8  = vec_packs(s16, s16);
      vec_xst_len(s8, reinterpret_cast<signed char*>(output + i), K);
    }
  }

  // -------------------------------------------------------------------------
  // dequantize_tile: INT32 accumulator -> FP32 scaled output
  // -------------------------------------------------------------------------
  static FORCE_INLINE void dequantize_tile(
      int32_t* input, float* output,
      const float* __restrict__ input_scales,
      const float* __restrict__ weight_scales,
      const int32_t m, const int32_t n, const int32_t stride) {
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
  // pack_input_from_rows: gather INT8 quantized rows -> contiguous [m][k]
  // -------------------------------------------------------------------------
  static void pack_input_from_rows(
      const int8_t* const* __restrict__ rows,
      int8_t* __restrict__ a_packed,
      const int32_t m, const int32_t k) {
    TORCH_CHECK(m > 0 && m <= MaxMSize);
    TORCH_CHECK_EQ(k % K, 0);
    for (int32_t r = 0; r < m; ++r)
      std::memcpy(a_packed + r * k, rows[r], k);
  }

  // -------------------------------------------------------------------------
  // pack_weight: [output_size, input_size] INT8 -> packed layout
  //   [Nr/4 col-groups][input_size][4 columns interleaved per k]
  // -------------------------------------------------------------------------
  static void pack_weight(
      const int8_t* __restrict__ weight,
      int8_t* __restrict__ packed_weight,
      const int32_t output_size,
      const int32_t input_size) {
    TORCH_CHECK_EQ(output_size % NSize, 0);
    TORCH_CHECK_EQ(input_size % K, 0);

    for (int32_t o = 0; o < output_size; o += Nr) {
      for (int32_t k = 0; k < input_size; k += K) {
        for (int32_t cg = 0; cg < Nr; cg += 4) {
          const int32_t cg_idx = cg / 4;
          int8_t* dst = packed_weight
                        + o * input_size
                        + cg_idx * input_size * 4
                        + k * 4;
          for (int32_t ki = 0; ki < K; ++ki) {
            dst[ki * 4 + 0] = weight[(o + cg + 0) * input_size + k + ki];
            dst[ki * 4 + 1] = weight[(o + cg + 1) * input_size + k + ki];
            dst[ki * 4 + 2] = weight[(o + cg + 2) * input_size + k + ki];
            dst[ki * 4 + 3] = weight[(o + cg + 3) * input_size + k + ki];
          }
        }
      }
    }
  }

  // -------------------------------------------------------------------------
  // gemm: outer N loop; m<=2 -> VSX scalar, m>2 -> MMA in Mr-row panels
  // -------------------------------------------------------------------------
  void gemm(
      const int8_t* __restrict__ a_packed,
      const int8_t* __restrict__ b_packed,
      int32_t* __restrict__ c,
      const int32_t m, const int32_t k,
      const int64_t b_n_group_stride,
      const int64_t ldc) const {
    TORCH_CHECK(m > 0 && m <= MaxMSize);
    TORCH_CHECK_EQ(k % K, 0);
    (void)b_n_group_stride;

    for (int32_t n_idx = 0; n_idx < NSize; n_idx += Nr) {
      const int8_t* b_panel = b_packed + n_idx * k;

      if (m <= 2) {
        vsx_int8::gemm_vsx_4x16(a_packed, b_panel, c + n_idx, m, k, ldc);
        continue;
      }

      for (int32_t row_base = 0; row_base < m; row_base += Mr) {
        const int32_t panel_m = std::min(Mr, m - row_base);
        vsx_int8::gemm_mma_4x16(
            a_packed + row_base * k,
            b_panel,
            c + row_base * ldc + n_idx,
            panel_m, k, ldc);
      }
    }
  }
};

}  // namespace cpu_micro_gemm

#endif  // CPU_MICRO_GEMM_INT8_VSX_HPP
