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
  if (payload == 0) return sign;               // underflow -> +-0
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

      // Key layout: column-major within block
      const scalar_t* ksrc = key_ptr + tok * k_stride0 + h * k_stride1;
      uint8_t* kdst = key_cache_ptr + block_idx * kc_stride0 + h * kc_stride1 +
                      block_offset;
      for (int64_t i = 0; i < head_dim; ++i)
        kdst[i * block_size] =
            float_to_fp8e4m3_scalar(static_cast<float>(ksrc[i]), k_inv);

      // Value layout: row-major within block (contiguous head_dim bytes)
      const scalar_t* vsrc = value_ptr + tok * v_stride0 + h * v_stride1;
      uint8_t* vdst = value_cache_ptr + block_idx * vc_stride0 +
                      h * vc_stride1 + block_offset * head_dim;
      quant_to_fp8<scalar_t>(vsrc, vdst, static_cast<int>(head_dim), v_inv);
    }
  }
}
