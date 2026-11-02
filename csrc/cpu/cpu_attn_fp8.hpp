// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "cpu/utils.hpp"

typedef uint32_t __attribute__((__may_alias__)) u32_alias_t;
typedef uint16_t __attribute__((__may_alias__)) u16_alias_t;
typedef float __attribute__((__may_alias__)) f32_alias_t;

// Reference scalar dequant — used to verify vectorized AMX dequant.
inline float fp8e4m3_to_float_scalar(uint8_t b, float scale) noexcept {
  // NaN encoding in E4M3
  if ((b & 0x7F) == 0x7F) return std::numeric_limits<float>::quiet_NaN();
  uint32_t b_u32 = static_cast<uint32_t>(b);
  uint32_t sign = (b_u32 & 0x80) << 24;
  uint32_t payload = (b_u32 & 0x7F) << 20;
  uint32_t bits = sign | payload;
  float b_f32_unscaled = *reinterpret_cast<const f32_alias_t*>(&bits);
  float b_f32_scaled = b_f32_unscaled * scale * 0x1p120f;
  return b_f32_scaled;
}

template <int mantissa_bits>
inline uint8_t float_to_fp8_scalar(float v, float inv_scale) noexcept {
  constexpr int bias = mantissa_bits == 3 ? 7 : 15;
  constexpr uint32_t max_bits = mantissa_bits == 3 ? 0x43e00000u : 0x47600000u;
  constexpr uint8_t max_fp8 = mantissa_bits == 3 ? 0x7e : 0x7b;
  constexpr uint32_t half_min_subnormal = (127 - bias - mantissa_bits) << 23;
  constexpr uint32_t min_normal = (128 - bias) << 23;
  v *= inv_scale;
  const uint32_t bits = std::bit_cast<uint32_t>(v);
  const uint8_t sign = (bits >> 24) & 0x80;
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > max_bits) {
    // Preserve the existing cache writer's saturation of NaN inputs.
    return magnitude > 0x7f800000u ? max_fp8 : sign | max_fp8;
  }
  if (magnitude <= half_min_subnormal) return sign;
  if (magnitude < min_normal) {
    const uint32_t significand = (magnitude & 0x7fffffu) | 0x800000u;
    const uint32_t shift = 151 - bias - mantissa_bits - (magnitude >> 23);
    const uint32_t rounded =
        significand + (1u << (shift - 1)) - 1 + ((significand >> shift) & 1);
    return sign | static_cast<uint8_t>(rounded >> shift);
  }
  constexpr int shift = 23 - mantissa_bits;
  const uint32_t rounded = magnitude +
                           ((1u << (shift - 1)) - 1 - ((127 - bias) << 23)) +
                           ((magnitude >> shift) & 1);
  return sign | static_cast<uint8_t>(rounded >> shift);
}

// ---------------------------------------------------------------------------
// AMX reshape impl — parameterised on the quantisation function.
// Writes key/value into uint8 FP8 KV cache using the AMX tile-friendly layout.
// K: halfword-packed (2 FP8 per uint16, token_num_per_group=16).
// V: sub-group packing (token_num_per_sub_group=2, head_elems_per_group=16).
// block_size must be divisible by 32.
// ---------------------------------------------------------------------------
template <typename scalar_t, uint8_t (*quant_fn)(float, float)>
inline void reshape_and_cache_fp8_amx_impl(
    const scalar_t* key_ptr, const scalar_t* value_ptr, uint8_t* key_cache_ptr,
    uint8_t* value_cache_ptr, const int64_t* slot_ptr, int64_t token_num,
    int64_t head_num, int64_t head_dim, int64_t block_size, int64_t k_stride0,
    int64_t k_stride1, int64_t v_stride0, int64_t v_stride1, int64_t kc_stride0,
    int64_t kc_stride1, int64_t vc_stride0, int64_t vc_stride1, float k_inv,
    float v_inv) {
  constexpr int64_t token_num_per_group = 16;  // AMX_TILE_ROW_NUM
  const int64_t halfword_num = head_dim / 2;   // 2 FP8 per uint16
  const int64_t halfword_num_per_group = token_num_per_group * halfword_num;
  constexpr int64_t head_elems_per_group = 16;
  constexpr int64_t token_num_per_sub_group = 2;  // = 4 / sizeof(BF16)
  const int64_t group_num = head_dim / head_elems_per_group;
  const int64_t group_size = block_size * head_elems_per_group;

#pragma omp parallel for collapse(2) schedule(static)
  for (int64_t tok = 0; tok < token_num; ++tok) {
    for (int64_t h = 0; h < head_num; ++h) {
      const int64_t slot = slot_ptr[tok];
      if (slot < 0) continue;
      const int64_t block_idx = slot / block_size;
      const int64_t block_offset = slot % block_size;

      // Key: halfword-packed, 2 FP8 per uint16
      {
        const scalar_t* ksrc = key_ptr + tok * k_stride0 + h * k_stride1;
        const int64_t group_idx = block_offset / token_num_per_group;
        const int64_t group_offset = block_offset % token_num_per_group;
        uint16_t* kdst =
            reinterpret_cast<uint16_t*>(key_cache_ptr + block_idx * kc_stride0 +
                                        h * kc_stride1) +
            group_idx * halfword_num_per_group + group_offset;
        for (int64_t j = 0; j < halfword_num; ++j) {
          uint8_t fp8_0 = quant_fn(static_cast<float>(ksrc[j * 2]), k_inv);
          uint8_t fp8_1 = quant_fn(static_cast<float>(ksrc[j * 2 + 1]), k_inv);
          uint8_t bytes[2] = {fp8_0, fp8_1};
          uint16_t hw = *reinterpret_cast<const u16_alias_t*>(bytes);
          kdst[j * token_num_per_group] = hw;
        }
      }

      // Value: sub-group packing (token_num_per_sub_group = 2)
      {
        const scalar_t* vsrc = value_ptr + tok * v_stride0 + h * v_stride1;
        const int64_t sub_group_idx = block_offset / token_num_per_sub_group;
        const int64_t sub_group_offset = block_offset % token_num_per_sub_group;
        uint8_t* vdst =
            value_cache_ptr + block_idx * vc_stride0 + h * vc_stride1 +
            sub_group_idx * token_num_per_sub_group * head_elems_per_group +
            sub_group_offset;
        for (int64_t i = 0; i < group_num; ++i) {
          for (int64_t j = 0; j < head_elems_per_group; ++j)
            vdst[j * token_num_per_sub_group] =
                quant_fn(static_cast<float>(vsrc[j]), v_inv);
          vsrc += head_elems_per_group;
          vdst += group_size;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Select the FP8 quant function at compile time based on kv_cache_t.
// ---------------------------------------------------------------------------
template <typename kv_cache_t>
constexpr auto select_fp8_quant_fn() {
  if constexpr (std::is_same_v<kv_cache_t, c10::Float8_e5m2>)
    return float_to_fp8_scalar<2>;
  else
    return float_to_fp8_scalar<3>;
}

// ---------------------------------------------------------------------------
// VEC reshape impl — parameterised on the quantisation function.
// Writes key (column-major) and value (row-major) into uint8 FP8 KV cache.
// The pragma omp must live outside VLLM_DISPATCH_FLOATING_TYPES because
// #pragma cannot appear inside variadic macro arguments.
// ---------------------------------------------------------------------------
template <typename scalar_t, uint8_t (*quant_fn)(float, float)>
inline void reshape_and_cache_fp8_vec_impl(
    const scalar_t* key_ptr, const scalar_t* value_ptr, uint8_t* key_cache_ptr,
    uint8_t* value_cache_ptr, const int64_t* slot_ptr, int64_t token_num,
    int64_t head_num, int64_t head_dim, int64_t block_size, int64_t k_stride0,
    int64_t k_stride1, int64_t v_stride0, int64_t v_stride1, int64_t kc_stride0,
    int64_t kc_stride1, int64_t vc_stride0, int64_t vc_stride1, float k_inv,
    float v_inv) {
#pragma omp parallel for collapse(2) schedule(static)
  for (int64_t tok = 0; tok < token_num; ++tok) {
    for (int64_t h = 0; h < head_num; ++h) {
      const int64_t slot = slot_ptr[tok];
      if (slot < 0) continue;
      const int64_t block_idx = slot / block_size;
      const int64_t block_offset = slot % block_size;

      // Key layout: column-major within block
      const scalar_t* ksrc = key_ptr + tok * k_stride0 + h * k_stride1;
      uint8_t* kdst = key_cache_ptr + block_idx * kc_stride0 + h * kc_stride1 +
                      block_offset;
      for (int64_t i = 0; i < head_dim; ++i)
        kdst[i * block_size] = quant_fn(static_cast<float>(ksrc[i]), k_inv);

      // Value layout: row-major within block (contiguous head_dim bytes)
      const scalar_t* vsrc = value_ptr + tok * v_stride0 + h * v_stride1;
      uint8_t* vdst = value_cache_ptr + block_idx * vc_stride0 +
                      h * vc_stride1 + block_offset * head_dim;
      for (int64_t i = 0; i < head_dim; ++i)
        vdst[i] = quant_fn(static_cast<float>(vsrc[i]), v_inv);
    }
  }
}
