// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef CPU_MICRO_GEMM_INT8_VSX_HPP
#define CPU_MICRO_GEMM_INT8_VSX_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"

#include <altivec.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>

namespace cpu_micro_gemm {

namespace vsx_int8 {

constexpr int32_t K   = 8;
constexpr int32_t Mr  = 4;
constexpr int32_t Nr  = 16;

// -----------------------------------------------------------------------
// load 4 floats from various scalar_t pointers
// -----------------------------------------------------------------------
FORCE_INLINE __vector float load4_f32(const float* p) {
  return vec_xl(0, p);
}

FORCE_INLINE __vector float load4_f32_lo(const float* p) {
  return vec_xl(0, p);
}

FORCE_INLINE __vector float load4_f32_hi(const float* p) {
  return vec_xl(16, p);
}

FORCE_INLINE __vector float load4_bf16_lo(const c10::BFloat16* p) {
  // Load 8 bf16 values, expand lower 4 to float32
  const __vector signed short raw =
      (__vector signed short)vec_xl(0, (const signed short*)p);
  const __vector signed short z = vec_splats((signed short)0);
  return (__vector float)vec_mergeh(z, raw);
}

FORCE_INLINE __vector float load4_bf16_hi(const c10::BFloat16* p) {
  // Load 8 bf16 values, expand upper 4 to float32
  const __vector signed short raw =
      (__vector signed short)vec_xl(0, (const signed short*)p);
  const __vector signed short z = vec_splats((signed short)0);
  return (__vector float)vec_mergel(z, raw);
}

FORCE_INLINE __vector float load4_f16_lo(const c10::Half* p) {
  const __vector unsigned short raw =
      (__vector unsigned short)vec_xl(0, (const unsigned short*)p);
  const __vector unsigned int wide =
      (__vector unsigned int)vec_unpackh((__vector signed short)raw);
  const __vector unsigned int ms = {0x8000u,0x8000u,0x8000u,0x8000u};
  const __vector unsigned int me = {0x7C00u,0x7C00u,0x7C00u,0x7C00u};
  const __vector unsigned int mm = {0x03FFu,0x03FFu,0x03FFu,0x03FFu};
  const __vector unsigned int ba = {112u,112u,112u,112u};
  __vector unsigned int s = (wide & ms) << 16;
  __vector unsigned int e = ((wide & me) >> 10) + ba;
  __vector unsigned int m = (wide & mm) << 13;
  return (__vector float)(s | (e << 23) | m);
}

FORCE_INLINE __vector float load4_f16_hi(const c10::Half* p) {
  const __vector unsigned short raw =
      (__vector unsigned short)vec_xl(0, (const unsigned short*)p);
  const __vector unsigned int wide =
      (__vector unsigned int)vec_unpackl((__vector signed short)raw);
  const __vector unsigned int ms = {0x8000u,0x8000u,0x8000u,0x8000u};
  const __vector unsigned int me = {0x7C00u,0x7C00u,0x7C00u,0x7C00u};
  const __vector unsigned int mm = {0x03FFu,0x03FFu,0x03FFu,0x03FFu};
  const __vector unsigned int ba = {112u,112u,112u,112u};
  __vector unsigned int s = (wide & ms) << 16;
  __vector unsigned int e = ((wide & me) >> 10) + ba;
  __vector unsigned int m = (wide & mm) << 13;
  return (__vector float)(s | (e << 23) | m);
}

// -----------------------------------------------------------------------
// INT8 dot-product of 16 elements: a[16] . b[16] -> int32 vector (4 lanes)
// Uses vec_mule/vec_mulo to multiply pairs, then vec_sum4s to reduce.
// -----------------------------------------------------------------------
FORCE_INLINE __vector int32_t int8_dot16(
    __vector signed char a,
    __vector signed char b) {
  __vector signed short lo = vec_mule(a, b);
  __vector signed short hi = vec_mulo(a, b);
  __vector signed int zero32 = vec_splats((signed int)0);
  __vector signed int s_lo = vec_sum4s(lo, zero32);
  __vector signed int s_hi = vec_sum4s(hi, zero32);
  return vec_add(s_lo, s_hi);
}

// -----------------------------------------------------------------------
// Core 4xNr micro-GEMM (POWER9 VSX path, always available)
// a_packed : [m][K] row-major, rows packed consecutively
// b_packed : [Nr/4 col-groups][K/4 quads][4 cols][4 elems] = Nr*K bytes
// c_ptr    : [m][ldc] int32 output (accumulate)
// -----------------------------------------------------------------------
FORCE_INLINE void gemm_vsx_4x16(
    const int8_t* __restrict__ a_packed,
    const int8_t* __restrict__ b_packed,
    int32_t* __restrict__ c_ptr,
    const int32_t m,
    const int32_t k_size,
    const int64_t ldc) {

  // accumulators: acc[row][col_group_of_4]
  __vector signed int acc[Mr][Nr/4];
  for (int r = 0; r < Mr; ++r)
    for (int c = 0; c < Nr/4; ++c)
      acc[r][c] = vec_splats((signed int)0);

  for (int32_t k = 0; k < k_size; k += K) {
    // Load K=8 bytes per row for up to Mr=4 rows
    // a_packed layout: rows stored consecutively, K bytes each
    __vector signed char a_rows[Mr];
    for (int r = 0; r < m; ++r) {
      // load 8 bytes, zero-extend to 16-byte vector
      a_rows[r] = (__vector signed char)vec_xl_len(
          (const signed char*)(a_packed + r * k_size + k), K);
    }
    for (int r = m; r < Mr; ++r)
      a_rows[r] = vec_splats((signed char)0);

    // b_packed layout: [Nr col-groups-of-4][K bytes per col]
    // For each col group of 4, the K weights for those 4 cols are interleaved:
    // [col0_k0, col1_k0, col2_k0, col3_k0, col0_k1, col1_k1, ...]
    for (int cg = 0; cg < Nr/4; ++cg) {
      // 4 cols × K=8 bytes = 32 bytes for this col group at this k block
      const int8_t* bp = b_packed + cg * k_size * 4 + k * 4;

      // We process K=8 elements split into two 4-element quads
      // Each quad: 4 consecutive col values for one k position
      // Load 16 bytes covering col0..3 × k0..3
      __vector signed char b0 = (__vector signed char)vec_xl(0,  (const signed char*)bp);
      // Load next 16 bytes covering col0..3 × k4..7
      __vector signed char b1 = (__vector signed char)vec_xl(16, (const signed char*)bp);

      for (int r = 0; r < Mr; ++r) {
        // Broadcast each of the K=8 activation bytes across the col dimension
        // and multiply with the 4 corresponding weight values
        // We use a scalar loop over K since VSX lacks a native int8 broadcast-madd
        const signed char* ap = (const signed char*)(a_packed + r * k_size + k);
        __vector signed int partial = vec_splats((signed int)0);
        for (int ki = 0; ki < K; ++ki) {
          // Load 4 weight bytes for the 4 cols at this k position
          const signed char* bki = (const signed char*)bp + ki * 4;
          // Widen 4 int8 weights to int32 and multiply by scalar activation
          int32_t a_val = (int32_t)ap[ki];
          __vector signed int contrib = {
            a_val * (int32_t)bki[0],
            a_val * (int32_t)bki[1],
            a_val * (int32_t)bki[2],
            a_val * (int32_t)bki[3]
          };
          partial = vec_add(partial, contrib);
        }
        if (r < m)
          acc[r][cg] = vec_add(acc[r][cg], partial);
      }
    }
  }

  // Write accumulators to c_ptr (overwrite, do not accumulate from existing contents)
  for (int r = 0; r < m; ++r) {
    for (int cg = 0; cg < Nr/4; ++cg) {
      vec_xst(acc[r][cg], 0, (signed int*)(c_ptr + r * ldc + cg * 4));
    }
  }
}

}  // namespace vsx_int8


// =========================================================================
// MicroGemmINT8 specialisation for VSX
// =========================================================================
template <typename scalar_t>
class MicroGemmINT8<cpu_utils::ISA::VSX, scalar_t> {
 public:
  static constexpr int32_t K              = vsx_int8::K;   // 8
  static constexpr int32_t Mr             = vsx_int8::Mr;  // 4
  static constexpr int32_t Nr             = vsx_int8::Nr;  // 16
  static constexpr int32_t NrGemv         = Nr;
  static constexpr int32_t MaxMSize       = 8;
  static constexpr int32_t NSize          = Nr;
  static constexpr int32_t WeightOCGroupSize = Nr;
  static_assert(MaxMSize % Mr == 0);

  // -----------------------------------------------------------------------
  // quantize_row: scalar_t row -> INT8 + per-token scale
  // -----------------------------------------------------------------------
  static FORCE_INLINE void quantize_row(
      const scalar_t* input, int8_t* output,
      float& scale, const int32_t size) {
    TORCH_CHECK_EQ(size % K, 0);

    __vector float max_vec = vec_splats(0.0f);
    for (int32_t i = 0; i < size; i += K) {
      __vector float lo, hi;
      if constexpr (std::is_same_v<scalar_t, float>) {
        lo = vsx_int8::load4_f32_lo(reinterpret_cast<const float*>(input) + i);
        hi = vsx_int8::load4_f32_hi(reinterpret_cast<const float*>(input) + i);
      } else if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
        lo = vsx_int8::load4_bf16_lo(input + i);
        hi = vsx_int8::load4_bf16_hi(input + i);
      } else {
        lo = vsx_int8::load4_f16_lo(input + i);
        hi = vsx_int8::load4_f16_hi(input + i);
      }
      max_vec = vec_max(max_vec, vec_abs(lo));
      max_vec = vec_max(max_vec, vec_abs(hi));
    }
    __vector float t = vec_max(max_vec, vec_sld(max_vec, max_vec, 8));
    t = vec_max(t, vec_sld(t, t, 4));
    const float abs_max = std::max(vec_extract(t, 0), 1.0e-7f);
    scale = abs_max / 127.0f;
    const __vector float inv_scale = vec_splats(127.0f / abs_max);

    for (int32_t i = 0; i < size; i += K) {
      __vector float lo, hi;
      if constexpr (std::is_same_v<scalar_t, float>) {
        lo = vsx_int8::load4_f32_lo(reinterpret_cast<const float*>(input) + i);
        hi = vsx_int8::load4_f32_hi(reinterpret_cast<const float*>(input) + i);
      } else if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
        lo = vsx_int8::load4_bf16_lo(input + i);
        hi = vsx_int8::load4_bf16_hi(input + i);
      } else {
        lo = vsx_int8::load4_f16_lo(input + i);
        hi = vsx_int8::load4_f16_hi(input + i);
      }
      __vector signed int lo_i = vec_cts(vec_mul(lo, inv_scale), 0);
      __vector signed int hi_i = vec_cts(vec_mul(hi, inv_scale), 0);
      __vector signed short s16 = vec_packs(lo_i, hi_i);
      __vector signed char  s8  = vec_packs(s16, s16);
      vec_xst_len(s8, (signed char*)(output + i), K);
    }
  }

  // -----------------------------------------------------------------------
  // dequantize_tile: INT32 -> FP32 with act_scale * weight_scale
  // -----------------------------------------------------------------------
  static FORCE_INLINE void dequantize_tile(
      int32_t* input, float* output,
      const float* __restrict__ input_scales,
      const float* __restrict__ weight_scales,
      const int32_t m, const int32_t n, const int32_t stride) {
    TORCH_CHECK_EQ(n % 4, 0);
    for (int32_t mi = 0; mi < m; ++mi) {
      const __vector float a_scale = vec_splats(input_scales[mi]);
      for (int32_t ni = 0; ni < n; ni += 4) {
        const __vector signed int i32 =
            vec_xl(0, (const signed int*)(input + mi * stride + ni));
        const __vector float f32   = vec_ctf(i32, 0);
        const __vector float wscale = vec_xl(0, weight_scales + ni);
        vec_xst(vec_mul(f32, vec_mul(a_scale, wscale)), 0,
                output + mi * stride + ni);
      }
    }
  }

  // -----------------------------------------------------------------------
  // pack_input_from_rows: gather INT8 rows into contiguous buffer
  // Layout: for each K-block, for each row: K bytes consecutively
  // -----------------------------------------------------------------------
  static void pack_input_from_rows(
      const int8_t* const* __restrict__ rows,
      int8_t* __restrict__ a_packed,
      const int32_t m, const int32_t k) {
    TORCH_CHECK(m > 0 && m <= MaxMSize);
    TORCH_CHECK(k % K == 0);
    // Store each row's full K bytes consecutively
    // a_packed shape: [m][k]
    for (int32_t r = 0; r < m; ++r) {
      std::memcpy(a_packed + r * k, rows[r], k);
    }
  }

  // -----------------------------------------------------------------------
  // pack_weight: reorder [output_size, input_size] INT8 weights
  // Layout: [Nr col-groups][input_size][4 cols interleaved per K-quad]
  // -----------------------------------------------------------------------
  static void pack_weight(
      const int8_t* __restrict__ weight,
      int8_t* __restrict__ packed_weight,
      const int32_t output_size,
      const int32_t input_size) {
    TORCH_CHECK(output_size % NSize == 0);
    TORCH_CHECK(input_size % K == 0);

    for (int32_t o = 0; o < output_size; o += Nr) {
      for (int32_t k = 0; k < input_size; k += K) {
        for (int32_t cg = 0; cg < Nr; cg += 4) {
          // cg_idx: 0,1,2,3 — index of this col-group within the Nr tile
          // gemm_vsx_4x16 reads col-group cg_idx at: b_packed + cg_idx * k_size * 4
          const int32_t cg_idx = cg / 4;
          int8_t* dst = packed_weight
              + (o * input_size)              // offset for this Nr group
              + cg_idx * input_size * 4       // offset for this col-group (matches gemm stride)
              + k * 4;                        // offset for this K-block
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

  // -----------------------------------------------------------------------
  // gemm: top-level dispatcher, loops Mr panels and calls gemm_vsx_4x16
  // -----------------------------------------------------------------------
  void gemm(
      const int8_t* __restrict__ a_packed,
      const int8_t* __restrict__ b_packed,
      int32_t* __restrict__ c,
      const int32_t m, const int32_t k,
      const int64_t b_n_group_stride,
      const int64_t ldc) const {
    TORCH_CHECK(m > 0 && m <= MaxMSize);
    TORCH_CHECK(k % K == 0);

    for (int32_t n_idx = 0; n_idx < NSize; n_idx += Nr) {
      const int8_t* b_panel = b_packed + n_idx * k;
      for (int32_t row_base = 0; row_base < m; row_base += Mr) {
        const int32_t panel_m = std::min(Mr, m - row_base);
        vsx_int8::gemm_vsx_4x16(
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
