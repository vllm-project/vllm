// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

inline float fp8e4m3_to_float_scalar(uint8_t b, float scale) noexcept {
  // NaN encoding in E4M3
  if ((b & 0x7F) == 0x7F) return std::numeric_limits<float>::quiet_NaN();
  uint32_t b_u32 = static_cast<uint32_t>(b);
  uint32_t sign = (b_u32 & 0x80) << 24;
  uint32_t payload = (b_u32 & 0x7F) << 20;
  uint32_t bits = sign | payload;
  float b_f32_unscaled;
  std::memcpy(&b_f32_unscaled, &bits, sizeof(float));
  float b_f32_scaled = b_f32_unscaled * scale * 0x1p120f;
  return b_f32_scaled;
}

inline uint8_t float_to_fp8e4m3_scalar(float v, float inv_scale) noexcept {
  v *= inv_scale;
  constexpr float fp8_max = 448.0f;
  v = std::max(-fp8_max, std::min(fp8_max, v));
  if (v == 0.0f) return 0;

  // Inverse mapping of fp8e4m3_to_float_scalar: shift the effective exponent
  // bias from fp32 (127) back to fp8 e4m3 (7), then pack sign|payload.
  float v_f32_unscaled = v * 0x1p-120f;
  uint32_t bits;
  std::memcpy(&bits, &v_f32_unscaled, sizeof(float));
  uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80);
  uint8_t payload = static_cast<uint8_t>((bits >> 20) & 0x7F);
  if (payload == 0) return sign;
  payload = std::min<uint8_t>(payload, 0x7E);  // keep 0x7F as NaN encoding
  return static_cast<uint8_t>(sign | payload);
}

template <typename scalar_t>
inline void quant_to_fp8(const scalar_t* src, uint8_t* dst, int n,
                         float inv_scale) {
  for (int i = 0; i < n; ++i)
    dst[i] = float_to_fp8e4m3_scalar(static_cast<float>(src[i]), inv_scale);
}

inline void deq_fp8_to_fp32(const uint8_t* src, float* dst, int n,
                            float scale) {
  for (int i = 0; i < n; ++i) dst[i] = fp8e4m3_to_float_scalar(src[i], scale);
}

// Writes key/value into uint8 FP8 KV cache using the AMX tile-friendly layout.
// K: halfword-packed layout (token_num_per_group=16, halfword_num=head_dim/2).
//    Mirrors the BF16 AMX K layout (2 BF16 per int32 → 2 FP8 per uint16).
//    Each K tile row = 16 tokens × halfword, stride = token_num_per_group
//    uint16s. After dequantisation the resulting 32-BF16 row matches the BF16 K
//    tile row.
// V: sub-group packing (token_num_per_sub_group=2, head_elems_per_group=16).
//    Mirrors the BF16 AMX V layout: token_num_per_sub_group = 4/sizeof(BF16)
//    = 2. FP8 bytes are dequantised to BF16 before loading into AMX tiles, so
//    the effective element size is BF16 (2 bytes), not FP8 (1 byte).
// block_size must be divisible by 32.
template <typename scalar_t>
inline void reshape_and_cache_fp8_amx_typed(
    const scalar_t* key_ptr, const scalar_t* value_ptr, uint8_t* key_cache_ptr,
    uint8_t* value_cache_ptr, const int64_t* slot_ptr, int64_t token_num,
    int64_t head_num, int64_t head_dim, int64_t block_size, int64_t k_stride0,
    int64_t k_stride1, int64_t v_stride0, int64_t v_stride1, int64_t kc_stride0,
    int64_t kc_stride1, int64_t vc_stride0, int64_t vc_stride1, float k_inv,
    float v_inv) {
  // K layout constants: 2 FP8 per uint16 halfword (= BF16 AMX layout scaled
  // to 1-byte FP8 elements).  The TileGemm deq loop reads with stride 32
  // bytes/row, which matches halfword_num * token_num_per_group * 2 bytes.
  constexpr int64_t token_num_per_group = 16;  // AMX_TILE_ROW_NUM
  const int64_t halfword_num = head_dim / 2;   // 2 FP8 per uint16
  const int64_t halfword_num_per_group = token_num_per_group * halfword_num;
  // V layout constants: token_num_per_sub_group = 4/sizeof(BF16) = 2.
  // FP8 values are dequantised to BF16 before entering AMX tiles, so this
  // must equal the BF16 AMX value (2), not the FP8-derived value (4).
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

      // Key: halfword-packed AMX Layout A — 2 FP8 per uint16, stride =
      // token_num_per_group uint16s between consecutive halfwords.
      // This mirrors the BF16 K layout (2 BF16 per int32, stride 16 int32s).
      {
        const scalar_t* ksrc = key_ptr + tok * k_stride0 + h * k_stride1;
        const int64_t group_idx = block_offset / token_num_per_group;
        const int64_t group_offset = block_offset % token_num_per_group;
        uint16_t* kdst =
            reinterpret_cast<uint16_t*>(key_cache_ptr + block_idx * kc_stride0 +
                                        h * kc_stride1) +
            group_idx * halfword_num_per_group + group_offset;
        for (int64_t j = 0; j < halfword_num; ++j) {
          uint8_t fp8_0 =
              float_to_fp8e4m3_scalar(static_cast<float>(ksrc[j * 2]), k_inv);
          uint8_t fp8_1 = float_to_fp8e4m3_scalar(
              static_cast<float>(ksrc[j * 2 + 1]), k_inv);
          uint8_t bytes[2] = {fp8_0, fp8_1};
          uint16_t hw;
          std::memcpy(&hw, bytes, 2);
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
                float_to_fp8e4m3_scalar(static_cast<float>(vsrc[j]), v_inv);
          vsrc += head_elems_per_group;
          vdst += group_size;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// FP8 E5M2 scalar helpers
// ---------------------------------------------------------------------------

// FP8 E5M2: s[7] e[6:2] m[1:0], exponent bias = 15 (same as FP16).
// Byte b → FP16 bits = b << 8 (no bias correction needed).
inline float fp8e5m2_to_float_scalar(uint8_t b, float scale) noexcept {
  const uint8_t exp_bits = (b >> 2) & 0x1F;
  const uint8_t mant_bits = b & 0x03;
  // NaN: exp=11111, mant!=00
  if (exp_bits == 0x1F && mant_bits != 0)
    return std::numeric_limits<float>::quiet_NaN();
  const uint32_t sign = static_cast<uint32_t>(b & 0x80) << 24;
  if (exp_bits == 0x1F)
    return sign ? -std::numeric_limits<float>::infinity()
                : std::numeric_limits<float>::infinity();
  if (exp_bits == 0) {  // subnormal: (-1)^s * 2^-14 * mant/4
    if (mant_bits == 0) return 0.0f;
    float v = mant_bits * 0x1p-16f;
    return (sign ? -v : v) * scale;
  }
  // Normal: FP32 exp = exp5 - 15 + 127, mantissa top 2 bits
  uint32_t fp32_bits = sign |
                       ((static_cast<uint32_t>(exp_bits) - 15 + 127) << 23) |
                       (static_cast<uint32_t>(mant_bits) << 21);
  float val;
  std::memcpy(&val, &fp32_bits, sizeof(float));
  return val * scale;
}

inline uint8_t float_to_fp8e5m2_scalar(float v, float inv_scale) noexcept {
  v *= inv_scale;
  constexpr float fp8_e5m2_max = 57344.0f;
  v = std::max(-fp8_e5m2_max, std::min(fp8_e5m2_max, v));
  if (v == 0.0f) return 0;
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(float));
  const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80);
  const int32_t exp_fp32 = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
  const uint8_t mant2 = static_cast<uint8_t>((bits >> 21) & 0x03);
  if (exp_fp32 < -14) {  // subnormal in E5M2
    const int shift = -14 - exp_fp32;
    const uint32_t m = (0x800000u | (bits & 0x7FFFFFu)) >> (shift + 21);
    return sign | static_cast<uint8_t>(std::min<uint32_t>(m, 3u));
  }
  const uint8_t exp5 = static_cast<uint8_t>(exp_fp32 + 15);
  return sign | (exp5 << 2) | mant2;
}

template <typename scalar_t>
inline void quant_to_fp8e5m2(const scalar_t* src, uint8_t* dst, int n,
                             float inv_scale) {
  for (int i = 0; i < n; ++i)
    dst[i] = float_to_fp8e5m2_scalar(static_cast<float>(src[i]), inv_scale);
}

inline void deq_fp8e5m2_to_fp32(const uint8_t* src, float* dst, int n,
                                float scale) {
  for (int i = 0; i < n; ++i) dst[i] = fp8e5m2_to_float_scalar(src[i], scale);
}

// ---------------------------------------------------------------------------
// E5M2 reshape helpers (VEC layout — identical structure to E4M3 variants)
// ---------------------------------------------------------------------------

template <typename scalar_t>
inline void reshape_and_cache_fp8_e5m2_typed(
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
        kdst[i * block_size] =
            float_to_fp8e5m2_scalar(static_cast<float>(ksrc[i]), k_inv);

      // Value layout: row-major within block (contiguous head_dim bytes)
      const scalar_t* vsrc = value_ptr + tok * v_stride0 + h * v_stride1;
      uint8_t* vdst = value_cache_ptr + block_idx * vc_stride0 +
                      h * vc_stride1 + block_offset * head_dim;
      quant_to_fp8e5m2<scalar_t>(vsrc, vdst, static_cast<int>(head_dim), v_inv);
    }
  }
}

// AMX-layout E5M2 reshape — same halfword-packed K / sub-group V layout as
// E4M3; only the quantization function differs.
template <typename scalar_t>
inline void reshape_and_cache_fp8_amx_e5m2_typed(
    const scalar_t* key_ptr, const scalar_t* value_ptr, uint8_t* key_cache_ptr,
    uint8_t* value_cache_ptr, const int64_t* slot_ptr, int64_t token_num,
    int64_t head_num, int64_t head_dim, int64_t block_size, int64_t k_stride0,
    int64_t k_stride1, int64_t v_stride0, int64_t v_stride1, int64_t kc_stride0,
    int64_t kc_stride1, int64_t vc_stride0, int64_t vc_stride1, float k_inv,
    float v_inv) {
  constexpr int64_t token_num_per_group = 16;
  const int64_t halfword_num = head_dim / 2;
  const int64_t halfword_num_per_group = token_num_per_group * halfword_num;
  constexpr int64_t head_elems_per_group = 16;
  constexpr int64_t token_num_per_sub_group = 2;
  const int64_t group_num = head_dim / head_elems_per_group;
  const int64_t group_size = block_size * head_elems_per_group;

#pragma omp parallel for collapse(2) schedule(static)
  for (int64_t tok = 0; tok < token_num; ++tok) {
    for (int64_t h = 0; h < head_num; ++h) {
      const int64_t slot = slot_ptr[tok];
      if (slot < 0) continue;
      const int64_t block_idx = slot / block_size;
      const int64_t block_offset = slot % block_size;

      // Key: halfword-packed AMX layout
      {
        const scalar_t* ksrc = key_ptr + tok * k_stride0 + h * k_stride1;
        const int64_t group_idx = block_offset / token_num_per_group;
        const int64_t group_offset = block_offset % token_num_per_group;
        uint16_t* kdst =
            reinterpret_cast<uint16_t*>(key_cache_ptr + block_idx * kc_stride0 +
                                        h * kc_stride1) +
            group_idx * halfword_num_per_group + group_offset;
        for (int64_t j = 0; j < halfword_num; ++j) {
          uint8_t fp8_0 =
              float_to_fp8e5m2_scalar(static_cast<float>(ksrc[j * 2]), k_inv);
          uint8_t fp8_1 = float_to_fp8e5m2_scalar(
              static_cast<float>(ksrc[j * 2 + 1]), k_inv);
          uint8_t bytes[2] = {fp8_0, fp8_1};
          uint16_t hw;
          std::memcpy(&hw, bytes, 2);
          kdst[j * token_num_per_group] = hw;
        }
      }

      // Value: sub-group packing
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
                float_to_fp8e5m2_scalar(static_cast<float>(vsrc[j]), v_inv);
          vsrc += head_elems_per_group;
          vdst += group_size;
        }
      }
    }
  }
}

// Writes key (column-major) and value (row-major) into uint8 FP8 KV cache.
// The pragma omp must live outside VLLM_DISPATCH_FLOATING_TYPES because
// #pragma cannot appear inside variadic macro arguments.
template <typename scalar_t>
inline void reshape_and_cache_fp8_typed(
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

      const scalar_t* ksrc = key_ptr + tok * k_stride0 + h * k_stride1;
      uint8_t* kdst = key_cache_ptr + block_idx * kc_stride0 + h * kc_stride1 +
                      block_offset;
      for (int64_t i = 0; i < head_dim; ++i)
        kdst[i * block_size] =
            float_to_fp8e4m3_scalar(static_cast<float>(ksrc[i]), k_inv);

      const scalar_t* vsrc = value_ptr + tok * v_stride0 + h * v_stride1;
      uint8_t* vdst = value_cache_ptr + block_idx * vc_stride0 +
                      h * vc_stride1 + block_offset * head_dim;
      quant_to_fp8<scalar_t>(vsrc, vdst, static_cast<int>(head_dim), v_inv);
    }
  }
}
