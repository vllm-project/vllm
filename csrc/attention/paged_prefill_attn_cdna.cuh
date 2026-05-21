// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared MFMA helpers for CDNA (gfx942 / gfx950 / gfx90a) paged-prefill
// attention kernels with INT8 / INT4 per-token-head KV cache.
//
// Wave64 + MFMA 16x16x16 fragment layout (per AMD ISA):
//   A operand (16 rows x 16 K): each lane holds 4 fp16/bf16, distributed as
//     row_id  = lane / 16  (rows 0..3 across 4 lane-groups of 16)
//     k_id    = lane % 16 * 4 + k_local (k_local in [0..3] per lane)
//   B operand (16 K x 16 cols): each lane holds 4 fp16/bf16, distributed as
//     col_id  = lane / 16
//     k_id    = lane % 16 * 4 + k_local
//   C accumulator (16 rows x 16 cols): floatx4 per lane, 4 elements own
//     row_id = lane / 16 + 4 * acc_idx (acc_idx in [0..3])
//     col_id = lane % 16
//
// This file deliberately exposes the same surface as
// paged_prefill_attn_rdna3.cuh (wmma_mma, v8fp32, etc.) so that callers can
// be ported with a header swap, but the underlying layouts are different.

#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

namespace vllm {
namespace prefill_attn_cdna {

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__CDNA__
#endif

// ---------------------------------------------------------------------------
// Vector types
// ---------------------------------------------------------------------------

using bf16_t = __hip_bfloat16;

using floatx4  = __attribute__((__vector_size__(4 * sizeof(float))))  float;
using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using halfx4   = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using bf16x4   = __attribute__((__vector_size__(4 * sizeof(__bf16))))   __bf16;
using int8x4   = __attribute__((__vector_size__(4 * sizeof(int8_t))))   int8_t;
using int8x8   = __attribute__((__vector_size__(8 * sizeof(int8_t))))   int8_t;
using int32x4  = __attribute__((__vector_size__(4 * sizeof(int32_t))))  int32_t;

// Bit-equivalent 16-bit-x-4 (used as the storage container for both fp16
// and bf16 fragments fed to MFMA).
using _B16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;

// Helper "vector of N elements with type T" alias used by the kernels.
template <typename T>
struct WmmaNative;  // name kept for API compatibility with the RDNA header.

template <>
struct WmmaNative<_Float16> {
  using elem = _Float16;
  using v4   = halfx4;     // 4-element fp16 fragment (per-lane storage)
};

template <>
struct WmmaNative<bf16_t> {
  using elem = bf16_t;
  using v4   = bf16x4;
};

// 8-fp32 result vector kept for source-level compatibility with the RDNA
// kernel's S accumulator. On CDNA we map a 16x16 result to two floatx4 (the
// underlying MFMA returns floatx4 per call). The kernel code below uses two
// floatx4 per query-tile half.
struct v8fp32 {
  floatx4 lo;
  floatx4 hi;
};

// ---------------------------------------------------------------------------
// Bit-cast helpers
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ _B16x4 to_b16x4(typename WmmaNative<T>::v4 v) {
  _B16x4 r;
  __builtin_memcpy(&r, &v, sizeof(_B16x4));
  return r;
}

template <typename T>
__device__ __forceinline__ typename WmmaNative<T>::v4 from_b16x4(_B16x4 v) {
  typename WmmaNative<T>::v4 r;
  __builtin_memcpy(&r, &v, sizeof(r));
  return r;
}

// ---------------------------------------------------------------------------
// MFMA wrappers (16x16x16, fp16/bf16 inputs, fp32 accumulator)
// ---------------------------------------------------------------------------

#if defined(__HIP__CDNA__)

template <typename T>
__device__ __forceinline__ floatx4 mfma_16x16x16(typename WmmaNative<T>::v4 a,
                                                 typename WmmaNative<T>::v4 b,
                                                 floatx4 c) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
  } else {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
  }
}

// Signed-signed int8 -> int32 16x16x16.
// `a` and `b` are 4 int8s per lane packed into one int32.
__device__ __forceinline__ int32x4 mfma_i32_16x16x16_i8(int32_t a, int32_t b,
                                                        int32x4 c) {
  return __builtin_amdgcn_mfma_i32_16x16x16i8(a, b, c, 0, 0, 0);
}

// ---------------------------------------------------------------------------
// Scalar conversions
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T from_float_rn(float v);

template <>
__device__ __forceinline__ _Float16 from_float_rn<_Float16>(float v) {
  return (_Float16)v;
}

template <>
__device__ __forceinline__ bf16_t from_float_rn<bf16_t>(float v) {
  // Round-to-nearest-even bf16 cast that skips the NaN canonicalisation
  // path (we never see NaNs in dequantised int8/int4 weights).
  union { uint32_t u; float f; } u;
  u.f = v;
  uint32_t rounding_bias = 0x7fff + ((u.u >> 16) & 1);
  uint16_t bf = (uint16_t)((u.u + rounding_bias) >> 16);
  bf16_t r;
  __builtin_memcpy(&r, &bf, 2);
  return r;
}

template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<_Float16>(_Float16 v) {
  return (float)v;
}

template <>
__device__ __forceinline__ float to_float<bf16_t>(bf16_t v) {
  uint16_t bf;
  __builtin_memcpy(&bf, &v, 2);
  union { uint32_t u; float f; } u;
  u.u = (uint32_t)bf << 16;
  return u.f;
}

// int8 -> T conversion that, for bf16, skips the NaN-canonicalisation path
// that __float2bfloat16 emits unconditionally. Values in [-128, 127] are
// exactly representable in bf16's 7-bit mantissa.
template <typename T>
__device__ __forceinline__ T cvt_T_from_int8(int8_t v) {
  return from_float_rn<T>((float)v);
}

// ---------------------------------------------------------------------------
// Wave-wide reductions (wave64, 32-lane half-waves)
//
// On CDNA wave64, the MFMA fragment layout gives each lane four
// independent (row, col) accumulator entries. Per-row reductions over the
// 16-column axis use a butterfly across lanes whose lane%16 differs.
// Since the wave is 64 lanes and the row group is 16 lanes wide (lane/16
// identifies one of 4 row groups), we reduce within each 16-lane group.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float wave_group16_max(float v) {
  v = fmaxf(v, __shfl_xor(v, 1));
  v = fmaxf(v, __shfl_xor(v, 2));
  v = fmaxf(v, __shfl_xor(v, 4));
  v = fmaxf(v, __shfl_xor(v, 8));
  return v;
}

__device__ __forceinline__ float wave_group16_sum(float v) {
  v += __shfl_xor(v, 1);
  v += __shfl_xor(v, 2);
  v += __shfl_xor(v, 4);
  v += __shfl_xor(v, 8);
  return v;
}

#endif  // __HIP__CDNA__

}  // namespace prefill_attn_cdna
}  // namespace vllm
