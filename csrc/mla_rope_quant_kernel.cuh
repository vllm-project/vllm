// SPDX-License-Identifier: Apache-2.0
// Yanked from FlashInfer (flashinfer/vec_dtypes.cuh, flashinfer/pos_enc.cuh,
// flashinfer/layout.cuh) so the kernel can be modified in-tree without
// rebuilding FlashInfer. Original copyright:
//   Copyright (c) 2023-2025 by FlashInfer team.
//   Licensed under the Apache License, Version 2.0
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>

namespace vllm {
namespace mla_rope {

// ============================================================================
// vec_t infrastructure (from flashinfer/vec_dtypes.cuh)
// ============================================================================

#define MLA_ROPE_INLINE inline __attribute__((always_inline)) __device__

// ---- vec_cast generic ----
template <typename dst_t, typename src_t>
struct vec_cast {
  template <size_t vec_size>
  MLA_ROPE_INLINE static void cast(dst_t* dst, const src_t* src) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      dst[i] = (dst_t)src[i];
    }
  }
};

template <>
struct vec_cast<__nv_fp8_e4m3, float> {
  template <size_t vec_size>
  MLA_ROPE_INLINE static void cast(__nv_fp8_e4m3* dst, const float* src) {
    if constexpr (vec_size == 1) {
      dst[0] = __nv_fp8_e4m3(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((__nv_fp8x2_storage_t*)dst)[i] = __nv_cvt_float2_to_fp8x2(
            ((float2*)src)[i], __NV_SATFINITE, __NV_E4M3);
      }
    }
  }
};

template <>
struct vec_cast<float, nv_bfloat16> {
  template <size_t vec_size>
  MLA_ROPE_INLINE static void cast(float* dst, const nv_bfloat16* src) {
    if constexpr (vec_size == 1) {
      dst[0] = __bfloat162float(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((float2*)dst)[i] = __bfloat1622float2(((nv_bfloat162*)src)[i]);
      }
    }
  }
};

template <>
struct vec_cast<nv_bfloat16, float> {
  template <size_t vec_size>
  MLA_ROPE_INLINE static void cast(nv_bfloat16* dst, const float* src) {
    if constexpr (vec_size == 1) {
      dst[0] = __float2bfloat16(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((nv_bfloat162*)dst)[i] = __float22bfloat162_rn(((float2*)src)[i]);
      }
    }
  }
};

template <>
struct vec_cast<float, half> {
  template <size_t vec_size>
  MLA_ROPE_INLINE static void cast(float* dst, const half* src) {
    if constexpr (vec_size == 1) {
      dst[0] = __half2float(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((float2*)dst)[i] = __half22float2(((half2*)src)[i]);
      }
    }
  }
};

template <>
struct vec_cast<half, float> {
  template <size_t vec_size>
  MLA_ROPE_INLINE static void cast(half* dst, const float* src) {
    if constexpr (vec_size == 1) {
      dst[0] = __float2half(src[0]);
    } else {
#pragma unroll
      for (size_t i = 0; i < vec_size / 2; ++i) {
        ((half2*)dst)[i] = __float22half2_rn(((float2*)src)[i]);
      }
    }
  }
};

// ---- Forward declarations ----
template <typename float_t, size_t vec_size>
struct vec_t;

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
MLA_ROPE_INLINE void cast_from_impl(vec_t<tgt_float_t, vec_size>& dst,
                                    const vec_t<src_float_t, vec_size>& src) {
  vec_cast<tgt_float_t, src_float_t>::template cast<vec_size>(
      dst.ptr(), const_cast<vec_t<src_float_t, vec_size>*>(&src)->ptr());
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
MLA_ROPE_INLINE void cast_load_impl(vec_t<tgt_float_t, vec_size>& dst,
                                    const src_float_t* src_ptr) {
  if constexpr (std::is_same_v<src_float_t, tgt_float_t>) {
    dst.load(src_ptr);
  } else {
    vec_t<src_float_t, vec_size> tmp;
    tmp.load(src_ptr);
    dst.cast_from(tmp);
  }
}

template <typename src_float_t, typename tgt_float_t, size_t vec_size>
MLA_ROPE_INLINE void cast_store_impl(tgt_float_t* dst_ptr,
                                     const vec_t<src_float_t, vec_size>& src) {
  if constexpr (std::is_same_v<src_float_t, tgt_float_t>) {
    src.store(dst_ptr);
  } else {
    vec_t<tgt_float_t, vec_size> tmp;
    tmp.cast_from(src);
    tmp.store(dst_ptr);
  }
}

// ---- Primary template (never instantiated) ----
template <typename float_t, size_t vec_size>
struct vec_t {};

// ============================================================================
// vec_t<float, N> specializations
// ============================================================================

template <>
struct vec_t<float, 1> {
  float data;
  MLA_ROPE_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  MLA_ROPE_INLINE const float& operator[](size_t i) const {
    return ((const float*)(&data))[i];
  }
  MLA_ROPE_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  MLA_ROPE_INLINE void fill(float val) { data = val; }
  MLA_ROPE_INLINE void load(const float* ptr) { data = *ptr; }
  MLA_ROPE_INLINE void store(float* ptr) const { *ptr = data; }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <>
struct vec_t<float, 2> {
  float2 data;
  MLA_ROPE_INLINE float& operator[](size_t i) { return ((float*)(&data))[i]; }
  MLA_ROPE_INLINE const float& operator[](size_t i) const {
    return ((const float*)(&data))[i];
  }
  MLA_ROPE_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  MLA_ROPE_INLINE void fill(float val) { data = make_float2(val, val); }
  MLA_ROPE_INLINE void load(const float* ptr) { data = *((float2*)ptr); }
  MLA_ROPE_INLINE void store(float* ptr) const { *((float2*)ptr) = data; }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <size_t vec_size>
struct vec_t<float, vec_size> {
  static_assert(vec_size % 4 == 0, "Invalid vector size");
  float4 data[vec_size / 4];
  MLA_ROPE_INLINE float& operator[](size_t i) { return ((float*)(data))[i]; }
  MLA_ROPE_INLINE const float& operator[](size_t i) const {
    return ((const float*)(data))[i];
  }
  MLA_ROPE_INLINE float* ptr() { return reinterpret_cast<float*>(&data); }
  MLA_ROPE_INLINE void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4(val, val, val, val);
    }
  }
  MLA_ROPE_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  MLA_ROPE_INLINE void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

// ============================================================================
// vec_t<nv_bfloat16, N> specializations
// ============================================================================

template <>
struct vec_t<nv_bfloat16, 1> {
  nv_bfloat16 data;
  MLA_ROPE_INLINE nv_bfloat16& operator[](size_t i) {
    return ((nv_bfloat16*)(&data))[i];
  }
  MLA_ROPE_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  MLA_ROPE_INLINE nv_bfloat16* ptr() {
    return reinterpret_cast<nv_bfloat16*>(&data);
  }
  MLA_ROPE_INLINE void fill(nv_bfloat16 val) { data = val; }
  MLA_ROPE_INLINE void load(const nv_bfloat16* ptr) { data = *ptr; }
  MLA_ROPE_INLINE void store(nv_bfloat16* ptr) const { *ptr = data; }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <>
struct vec_t<nv_bfloat16, 2> {
  nv_bfloat162 data;
  MLA_ROPE_INLINE nv_bfloat16& operator[](size_t i) {
    return ((nv_bfloat16*)(&data))[i];
  }
  MLA_ROPE_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  MLA_ROPE_INLINE nv_bfloat16* ptr() {
    return reinterpret_cast<nv_bfloat16*>(&data);
  }
  MLA_ROPE_INLINE void fill(nv_bfloat16 val) {
    data = make_bfloat162(val, val);
  }
  MLA_ROPE_INLINE void load(const nv_bfloat16* ptr) {
    data = *((nv_bfloat162*)ptr);
  }
  MLA_ROPE_INLINE void store(nv_bfloat16* ptr) const {
    *((nv_bfloat162*)ptr) = data;
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <>
struct vec_t<nv_bfloat16, 4> {
  uint2 data;
  MLA_ROPE_INLINE nv_bfloat16& operator[](size_t i) {
    return ((nv_bfloat16*)(&data))[i];
  }
  MLA_ROPE_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)(&data))[i];
  }
  MLA_ROPE_INLINE nv_bfloat16* ptr() {
    return reinterpret_cast<nv_bfloat16*>(&data);
  }
  MLA_ROPE_INLINE void fill(nv_bfloat16 val) {
    *(nv_bfloat162*)(&data.x) = make_bfloat162(val, val);
    *(nv_bfloat162*)(&data.y) = make_bfloat162(val, val);
  }
  MLA_ROPE_INLINE void load(const nv_bfloat16* ptr) { data = *((uint2*)ptr); }
  MLA_ROPE_INLINE void store(nv_bfloat16* ptr) const { *((uint2*)ptr) = data; }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <size_t vec_size>
struct vec_t<nv_bfloat16, vec_size> {
  static_assert(vec_size % 8 == 0, "Invalid vector size");
  int4 data[vec_size / 8];
  MLA_ROPE_INLINE nv_bfloat16& operator[](size_t i) {
    return ((nv_bfloat16*)data)[i];
  }
  MLA_ROPE_INLINE const nv_bfloat16& operator[](size_t i) const {
    return ((const nv_bfloat16*)data)[i];
  }
  MLA_ROPE_INLINE nv_bfloat16* ptr() {
    return reinterpret_cast<nv_bfloat16*>(&data);
  }
  MLA_ROPE_INLINE void fill(nv_bfloat16 val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(nv_bfloat162*)(&(data[i].x)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].y)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].z)) = make_bfloat162(val, val);
      *(nv_bfloat162*)(&(data[i].w)) = make_bfloat162(val, val);
    }
  }
  MLA_ROPE_INLINE void load(const nv_bfloat16* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((int4*)ptr)[i];
    }
  }
  MLA_ROPE_INLINE void store(nv_bfloat16* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((int4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

// ============================================================================
// vec_t<half, N> specializations
// ============================================================================

template <>
struct vec_t<half, 1> {
  half data;
  MLA_ROPE_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  MLA_ROPE_INLINE const half& operator[](size_t i) const {
    return ((const half*)(&data))[i];
  }
  MLA_ROPE_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  MLA_ROPE_INLINE void fill(half val) { data = val; }
  MLA_ROPE_INLINE void load(const half* ptr) { data = *ptr; }
  MLA_ROPE_INLINE void store(half* ptr) const { *ptr = data; }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <>
struct vec_t<half, 2> {
  half2 data;
  MLA_ROPE_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  MLA_ROPE_INLINE const half& operator[](size_t i) const {
    return ((const half*)(&data))[i];
  }
  MLA_ROPE_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  MLA_ROPE_INLINE void fill(half val) { data = make_half2(val, val); }
  MLA_ROPE_INLINE void load(const half* ptr) { data = *((half2*)ptr); }
  MLA_ROPE_INLINE void store(half* ptr) const { *((half2*)ptr) = data; }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <>
struct vec_t<half, 4> {
  uint2 data;
  MLA_ROPE_INLINE half& operator[](size_t i) { return ((half*)(&data))[i]; }
  MLA_ROPE_INLINE const half& operator[](size_t i) const {
    return ((const half*)(&data))[i];
  }
  MLA_ROPE_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  MLA_ROPE_INLINE void fill(half val) {
    *(half2*)(&data.x) = make_half2(val, val);
    *(half2*)(&data.y) = make_half2(val, val);
  }
  MLA_ROPE_INLINE void load(const half* ptr) { data = *((uint2*)ptr); }
  MLA_ROPE_INLINE void store(half* ptr) const { *((uint2*)ptr) = data; }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <size_t vec_size>
struct vec_t<half, vec_size> {
  static_assert(vec_size % 8 == 0, "Invalid vector size");
  int4 data[vec_size / 8];
  MLA_ROPE_INLINE half& operator[](size_t i) { return ((half*)data)[i]; }
  MLA_ROPE_INLINE const half& operator[](size_t i) const {
    return ((const half*)data)[i];
  }
  MLA_ROPE_INLINE half* ptr() { return reinterpret_cast<half*>(&data); }
  MLA_ROPE_INLINE void fill(half val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      *(half2*)(&(data[i].x)) = make_half2(val, val);
      *(half2*)(&(data[i].y)) = make_half2(val, val);
      *(half2*)(&(data[i].z)) = make_half2(val, val);
      *(half2*)(&(data[i].w)) = make_half2(val, val);
    }
  }
  MLA_ROPE_INLINE void load(const half* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      data[i] = ((int4*)ptr)[i];
    }
  }
  MLA_ROPE_INLINE void store(half* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 8; ++i) {
      ((int4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

// ============================================================================
// vec_t<__nv_fp8_e4m3, N> specializations
// ============================================================================

template <>
struct vec_t<__nv_fp8_e4m3, 1> {
  __nv_fp8_e4m3 data;
  MLA_ROPE_INLINE __nv_fp8_e4m3& operator[](size_t i) {
    return ((__nv_fp8_e4m3*)(&data))[i];
  }
  MLA_ROPE_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)(&data))[i];
  }
  MLA_ROPE_INLINE __nv_fp8_e4m3* ptr() {
    return reinterpret_cast<__nv_fp8_e4m3*>(&data);
  }
  MLA_ROPE_INLINE void fill(__nv_fp8_e4m3 val) { data = val; }
  MLA_ROPE_INLINE void load(const __nv_fp8_e4m3* ptr) { data = *ptr; }
  MLA_ROPE_INLINE void store(__nv_fp8_e4m3* ptr) const { *ptr = data; }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, 1>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <>
struct vec_t<__nv_fp8_e4m3, 2> {
  __nv_fp8x2_e4m3 data;
  MLA_ROPE_INLINE __nv_fp8_e4m3& operator[](size_t i) {
    return ((__nv_fp8_e4m3*)(&data))[i];
  }
  MLA_ROPE_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)(&data))[i];
  }
  MLA_ROPE_INLINE __nv_fp8_e4m3* ptr() {
    return reinterpret_cast<__nv_fp8_e4m3*>(&data);
  }
  MLA_ROPE_INLINE void fill(__nv_fp8_e4m3 val) {
    data.__x =
        (__nv_fp8x2_storage_t(val.__x) << 8) | __nv_fp8x2_storage_t(val.__x);
  }
  MLA_ROPE_INLINE void load(const __nv_fp8_e4m3* ptr) {
    data = *((__nv_fp8x2_e4m3*)ptr);
  }
  MLA_ROPE_INLINE void store(__nv_fp8_e4m3* ptr) const {
    *((__nv_fp8x2_e4m3*)ptr) = data;
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, 2>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <>
struct vec_t<__nv_fp8_e4m3, 4> {
  __nv_fp8x4_e4m3 data;
  MLA_ROPE_INLINE __nv_fp8_e4m3& operator[](size_t i) {
    return ((__nv_fp8_e4m3*)(&data))[i];
  }
  MLA_ROPE_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)(&data))[i];
  }
  MLA_ROPE_INLINE __nv_fp8_e4m3* ptr() {
    return reinterpret_cast<__nv_fp8_e4m3*>(&data);
  }
  MLA_ROPE_INLINE void fill(__nv_fp8_e4m3 val) {
    data.__x = (__nv_fp8x4_storage_t(val.__x) << 24) |
               (__nv_fp8x4_storage_t(val.__x) << 16) |
               (__nv_fp8x4_storage_t(val.__x) << 8) |
               __nv_fp8x4_storage_t(val.__x);
  }
  MLA_ROPE_INLINE void load(const __nv_fp8_e4m3* ptr) {
    data = *((__nv_fp8x4_e4m3*)ptr);
  }
  MLA_ROPE_INLINE void store(__nv_fp8_e4m3* ptr) const {
    *((__nv_fp8x4_e4m3*)ptr) = data;
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, 4>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <>
struct vec_t<__nv_fp8_e4m3, 8> {
  uint2 data;
  MLA_ROPE_INLINE __nv_fp8_e4m3& operator[](size_t i) {
    return ((__nv_fp8_e4m3*)(&data))[i];
  }
  MLA_ROPE_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)(&data))[i];
  }
  MLA_ROPE_INLINE __nv_fp8_e4m3* ptr() {
    return reinterpret_cast<__nv_fp8_e4m3*>(&data);
  }
  MLA_ROPE_INLINE void fill(__nv_fp8_e4m3 val) {
    ((__nv_fp8x4_e4m3*)(&data.x))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) |
                                         (__nv_fp8x4_storage_t(val.__x) << 16) |
                                         (__nv_fp8x4_storage_t(val.__x) << 8) |
                                         __nv_fp8x4_storage_t(val.__x);
    ((__nv_fp8x4_e4m3*)(&data.y))->__x = (__nv_fp8x4_storage_t(val.__x) << 24) |
                                         (__nv_fp8x4_storage_t(val.__x) << 16) |
                                         (__nv_fp8x4_storage_t(val.__x) << 8) |
                                         __nv_fp8x4_storage_t(val.__x);
  }
  MLA_ROPE_INLINE void load(const __nv_fp8_e4m3* ptr) { data = *((uint2*)ptr); }
  MLA_ROPE_INLINE void store(__nv_fp8_e4m3* ptr) const {
    *((uint2*)ptr) = data;
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, 8>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

template <size_t vec_size>
struct vec_t<__nv_fp8_e4m3, vec_size> {
  static_assert(vec_size % 16 == 0, "Invalid vector size");
  int4 data[vec_size / 16];
  MLA_ROPE_INLINE __nv_fp8_e4m3& operator[](size_t i) {
    return ((__nv_fp8_e4m3*)data)[i];
  }
  MLA_ROPE_INLINE const __nv_fp8_e4m3& operator[](size_t i) const {
    return ((const __nv_fp8_e4m3*)data)[i];
  }
  MLA_ROPE_INLINE __nv_fp8_e4m3* ptr() {
    return reinterpret_cast<__nv_fp8_e4m3*>(&data);
  }
  MLA_ROPE_INLINE void fill(__nv_fp8_e4m3 val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((__nv_fp8x4_e4m3*)(&(data[i].x)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) |
          (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e4m3*)(&(data[i].y)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) |
          (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e4m3*)(&(data[i].z)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) |
          (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
      ((__nv_fp8x4_e4m3*)(&(data[i].w)))->__x =
          (__nv_fp8x4_storage_t(val.__x) << 24) |
          (__nv_fp8x4_storage_t(val.__x) << 16) |
          (__nv_fp8x4_storage_t(val.__x) << 8) | __nv_fp8x4_storage_t(val.__x);
    }
  }
  MLA_ROPE_INLINE void load(const __nv_fp8_e4m3* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      data[i] = ((int4*)ptr)[i];
    }
  }
  MLA_ROPE_INLINE void store(__nv_fp8_e4m3* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 16; ++i) {
      ((int4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_from(const vec_t<T, vec_size>& src) {
    cast_from_impl(*this, src);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_load(const T* ptr) {
    cast_load_impl(*this, ptr);
  }
  template <typename T>
  MLA_ROPE_INLINE void cast_store(T* ptr) const {
    cast_store_impl(ptr, *this);
  }
};

// ============================================================================
// Layout helper (from flashinfer/layout.cuh)
// ============================================================================

__host__ __device__ __forceinline__ size_t
get_elem_offset_impl(size_t elem_idx, size_t head_idx, size_t feat_idx,
                     size_t stride_n, size_t stride_h) {
  return elem_idx * stride_n + head_idx * stride_h + feat_idx;
}

// ============================================================================
// RoPE helper functions (from flashinfer/pos_enc.cuh)
// ============================================================================

// Non-interleaved RoPE with pre-computed cos/sin cache
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size> vec_apply_llama_rope_cos_sin(
    const T* x, const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    permuted_vec.cast_load(x + ((threadIdx.x * vec_size < rotary_dim / 2)
                                    ? threadIdx.x * vec_size + rotary_dim / 2
                                    : threadIdx.x * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] = vec[i] * cos[i] + ((threadIdx.x * vec_size < rotary_dim / 2)
                                      ? -permuted_vec[i]
                                      : permuted_vec[i]) *
                                     sin[i];
    }
  }
  return vec;
}

// Interleaved RoPE with pre-computed cos/sin cache (reuses first half)
template <uint32_t vec_size, uint32_t bdx, typename T>
__device__ __forceinline__ vec_t<float, vec_size>
vec_apply_llama_rope_cos_sin_interleave_reuse_half(
    const T* x, const vec_t<float, vec_size>& cos,
    const vec_t<float, vec_size>& sin,
    const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> vec, vec_before;
  vec.cast_load(x + threadIdx.x * vec_size);

  if (threadIdx.x * vec_size < rotary_dim) {
    vec_before = vec;
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      vec[i] =
          vec[i] * cos[i / 2] +
          ((i % 2 == 0) ? -vec_before[i ^ 1] : vec_before[i ^ 1]) * sin[i / 2];
    }
  }
  return vec;
}

// Scale + quantize helper for non-RoPE dimensions (handles partial chunks)
template <typename DType, typename QuantType, uint32_t vec_size>
__device__ __forceinline__ void scale_store_partial_chunk(
    const DType* in_ptr, QuantType* out_ptr, uint32_t lane_elem_offset,
    uint32_t chunk_valid, float scale) {
  if (chunk_valid == 0 || lane_elem_offset >= chunk_valid) {
    return;
  }
  vec_t<float, vec_size> vec;
  if (lane_elem_offset + vec_size <= chunk_valid) {
    vec.cast_load(in_ptr + lane_elem_offset);
  } else {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      uint32_t elem_idx = lane_elem_offset + i;
      if (elem_idx < chunk_valid) {
        vec_t<float, 1> tmp;
        tmp.cast_load(in_ptr + elem_idx);
        vec[i] = tmp[0];
      } else {
        vec[i] = 0.f;
      }
    }
  }
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    vec[i] = vec[i] * scale;
  }
  if (lane_elem_offset + vec_size <= chunk_valid) {
    vec.cast_store(out_ptr + lane_elem_offset);
  } else {
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      uint32_t elem_idx = lane_elem_offset + i;
      if (elem_idx < chunk_valid) {
        vec_t<float, 1> tmp;
        tmp[0] = vec[i];
        tmp.cast_store(out_ptr + elem_idx);
      }
    }
  }
}

// ============================================================================
// Main kernel (from flashinfer/pos_enc.cuh RopeQuantizeKernel)
// ============================================================================

template <bool interleave, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType, typename QuantType, typename CacheType = float>
__global__ void RopeQuantizeKernel(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in,
    QuantType* q_rope_out, QuantType* k_rope_out, QuantType* q_nope_out,
    QuantType* k_nope_out, CacheType* __restrict__ cos_sin_cache,
    IdType* __restrict__ pos_ids, uint32_t nnz, uint32_t num_qo_heads,
    uint32_t num_kv_heads, uint32_t rope_dim, uint32_t no_rope_dim,
    size_t q_rope_in_stride_n, size_t q_rope_in_stride_h,
    size_t q_nope_in_stride_n, size_t q_nope_in_stride_h,
    size_t q_rope_out_stride_n, size_t q_rope_out_stride_h,
    size_t q_nope_out_stride_n, size_t q_nope_out_stride_h,
    size_t k_rope_in_stride, size_t k_rope_in_stride_h, size_t k_nope_in_stride,
    size_t k_nope_in_stride_h, size_t k_rope_out_stride,
    size_t k_rope_out_stride_h, size_t k_nope_out_stride,
    size_t k_nope_out_stride_h, float quant_scale_q, float quant_scale_kv) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && \
     (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  uint32_t bdy = blockDim.y;

  uint32_t rope_chunk_size = rope_dim;
  uint32_t rope_chunks = (rope_dim + rope_chunk_size - 1) / rope_chunk_size;
  uint32_t no_rope_chunks =
      (no_rope_dim + rope_chunk_size - 1) / rope_chunk_size;

  uint32_t q_rope_end = num_qo_heads * rope_chunks;
  uint32_t k_rope_end = q_rope_end + num_kv_heads * rope_chunks;
  uint32_t k_nope_end = k_rope_end + num_kv_heads * no_rope_chunks;

  vec_t<float, vec_size> cos, sin;
  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];

    const int half_rope_dim = rope_dim / 2;
    if ((tx * vec_size < rope_dim) && (by < k_rope_end)) {
      int sin_offset = rope_dim / 2;
      int vec_idx;
      if constexpr (interleave) {
        vec_idx = (tx * vec_size) / 2;
      } else {
        vec_idx = (tx * vec_size) % half_rope_dim;
      }
      cos.cast_load(cos_sin_cache + (pos * rope_dim) + vec_idx);
      sin.cast_load(cos_sin_cache + (pos * rope_dim) + (sin_offset + vec_idx));
    }

    if (by < q_rope_end) {
      // Q RoPE processing
      uint32_t q_head_idx = by / rope_chunks;
      uint32_t rope_chunk_idx = by % rope_chunks;
      uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

      DType* q_rope_in_ptr =
          q_rope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset,
                                           q_rope_in_stride_n,
                                           q_rope_in_stride_h);
      QuantType* q_rope_out_ptr =
          q_rope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset,
                                            q_rope_out_stride_n,
                                            q_rope_out_stride_h);

      vec_t<float, vec_size> q_rope_vec;
      if constexpr (interleave) {
        q_rope_vec =
            vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
                q_rope_in_ptr, cos, sin, rope_dim);
      } else {
        q_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(
            q_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        q_rope_vec[i] = q_rope_vec[i] * quant_scale_q;
      }
      q_rope_vec.cast_store(q_rope_out_ptr + tx * vec_size);

    } else if (by < k_rope_end) {
      // K RoPE processing
      uint32_t k_head_idx = (by - q_rope_end) / rope_chunks;
      uint32_t rope_chunk_idx = (by - q_rope_end) % rope_chunks;
      uint32_t elem_offset = rope_chunk_idx * rope_chunk_size;

      DType* k_rope_in_ptr =
          k_rope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                           k_rope_in_stride,
                                           k_rope_in_stride_h);
      QuantType* k_rope_out_ptr =
          k_rope_out + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                            k_rope_out_stride,
                                            k_rope_out_stride_h);

      vec_t<float, vec_size> k_rope_vec;
      if constexpr (interleave) {
        k_rope_vec =
            vec_apply_llama_rope_cos_sin_interleave_reuse_half<vec_size, bdx>(
                k_rope_in_ptr, cos, sin, rope_dim);
      } else {
        k_rope_vec = vec_apply_llama_rope_cos_sin<vec_size, bdx>(
            k_rope_in_ptr, cos, sin, rope_dim);
      }
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        k_rope_vec[i] = k_rope_vec[i] * quant_scale_kv;
      }
      k_rope_vec.cast_store(k_rope_out_ptr + tx * vec_size);

    } else if (by < k_nope_end) {
      // K non-RoPE processing (quantize only)
      uint32_t k_head_idx = (by - k_rope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_rope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

      DType* k_nope_in_ptr =
          k_nope_in + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                           k_nope_in_stride,
                                           k_nope_in_stride_h);
      QuantType* k_nope_out_ptr =
          k_nope_out + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                            k_nope_out_stride,
                                            k_nope_out_stride_h);

      uint32_t chunk_valid =
          (elem_offset < no_rope_dim)
              ? min(rope_chunk_size, no_rope_dim - elem_offset)
              : 0u;
      uint32_t lane_elem_offset = tx * vec_size;
      scale_store_partial_chunk<DType, QuantType, vec_size>(
          k_nope_in_ptr, k_nope_out_ptr, lane_elem_offset, chunk_valid,
          quant_scale_kv);

    } else {
      // Q non-RoPE processing (quantize only)
      uint32_t q_head_idx = (by - k_nope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_nope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

      DType* q_nope_in_ptr =
          q_nope_in + get_elem_offset_impl(idx, q_head_idx, elem_offset,
                                           q_nope_in_stride_n,
                                           q_nope_in_stride_h);
      QuantType* q_nope_out_ptr =
          q_nope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset,
                                            q_nope_out_stride_n,
                                            q_nope_out_stride_h);

      uint32_t chunk_valid =
          (elem_offset < no_rope_dim)
              ? min(rope_chunk_size, no_rope_dim - elem_offset)
              : 0u;
      uint32_t lane_elem_offset = tx * vec_size;
      scale_store_partial_chunk<DType, QuantType, vec_size>(
          q_nope_in_ptr, q_nope_out_ptr, lane_elem_offset, chunk_valid,
          quant_scale_q);
    }
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && \
     (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace mla_rope
}  // namespace vllm

// ============================================================
// Tiled dynamic RoPE kernel: computes cos/sin on-the-fly from inv_freq
// instead of looking up from a precomputed cos_sin_cache.
// Processes vec_size in tiles of 4 to reduce register pressure.
// Uses __launch_bounds__(128, 16) to target 32 registers/thread.
// ============================================================

namespace vllm {
namespace mla_rope {

// Tiled interleave rotation: process 4 elements at a time
template <typename DType, typename QuantType, uint32_t vec_size>
__device__ __forceinline__ void tiled_rope_interleave_store(
    const DType* x_ptr, QuantType* out_ptr, const float* inv_freq,
    uint32_t vec_idx, float fpos, uint32_t rope_dim, float scale, uint32_t tx) {
  constexpr uint32_t TILE = 4;
  uint32_t base_offset = tx * vec_size;

  if (base_offset >= rope_dim) {
    vec_t<float, vec_size> vec;
    vec.cast_load(x_ptr + base_offset);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) vec[i] *= scale;
    vec.cast_store(out_ptr + base_offset);
    return;
  }

#pragma unroll
  for (uint32_t tile = 0; tile < vec_size / TILE; tile++) {
    uint32_t tile_offset = tile * TILE;
    uint32_t elem_offset = base_offset + tile_offset;

    float in4[TILE];
    {
      vec_t<float, TILE> tmp;
      tmp.cast_load(x_ptr + elem_offset);
#pragma unroll
      for (uint32_t i = 0; i < TILE; i++) in4[i] = tmp[i];
    }

    float out4[TILE];
#pragma unroll
    for (uint32_t i = 0; i < TILE; i += 2) {
      uint32_t freq_idx = vec_idx + (tile_offset + i) / 2;
      float cos_val, sin_val;
      if (freq_idx < rope_dim / 2) {
        __sincosf(fpos * inv_freq[freq_idx], &sin_val, &cos_val);
      } else {
        cos_val = 1.f;
        sin_val = 0.f;
      }
      out4[i] = (in4[i] * cos_val - in4[i + 1] * sin_val) * scale;
      out4[i + 1] = (in4[i] * sin_val + in4[i + 1] * cos_val) * scale;
    }

    {
      vec_t<float, TILE> tmp;
#pragma unroll
      for (uint32_t i = 0; i < TILE; i++) tmp[i] = out4[i];
      tmp.cast_store(out_ptr + elem_offset);
    }
  }
}

// Tiled non-interleave (NeoX) rotation
template <typename DType, typename QuantType, uint32_t vec_size>
__device__ __forceinline__ void tiled_rope_neox_store(
    const DType* x_ptr, QuantType* out_ptr, const float* inv_freq,
    uint32_t vec_idx, float fpos, uint32_t rope_dim, float scale, uint32_t tx) {
  constexpr uint32_t TILE = 4;
  uint32_t base_offset = tx * vec_size;
  uint32_t half_rope = rope_dim / 2;

  if (base_offset >= rope_dim) {
    vec_t<float, vec_size> vec;
    vec.cast_load(x_ptr + base_offset);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) vec[i] *= scale;
    vec.cast_store(out_ptr + base_offset);
    return;
  }

#pragma unroll
  for (uint32_t tile = 0; tile < vec_size / TILE; tile++) {
    uint32_t tile_offset = tile * TILE;
    uint32_t elem_offset = base_offset + tile_offset;

    float in4[TILE];
    {
      vec_t<float, TILE> tmp;
      tmp.cast_load(x_ptr + elem_offset);
#pragma unroll
      for (uint32_t i = 0; i < TILE; i++) in4[i] = tmp[i];
    }

    float paired4[TILE];
    {
      uint32_t pair_offset = (elem_offset < half_rope)
                                 ? elem_offset + half_rope
                                 : elem_offset - half_rope;
      vec_t<float, TILE> tmp;
      tmp.cast_load(x_ptr + pair_offset);
#pragma unroll
      for (uint32_t i = 0; i < TILE; i++) paired4[i] = tmp[i];
    }

    float out4[TILE];
#pragma unroll
    for (uint32_t i = 0; i < TILE; i++) {
      uint32_t freq_idx = (elem_offset + i) % half_rope;
      float cos_val, sin_val;
      if (freq_idx < half_rope) {
        __sincosf(fpos * inv_freq[freq_idx], &sin_val, &cos_val);
      } else {
        cos_val = 1.f;
        sin_val = 0.f;
      }
      float sign = (elem_offset + i < half_rope) ? -1.f : 1.f;
      out4[i] = (in4[i] * cos_val + sign * paired4[i] * sin_val) * scale;
    }

    {
      vec_t<float, TILE> tmp;
#pragma unroll
      for (uint32_t i = 0; i < TILE; i++) tmp[i] = out4[i];
      tmp.cast_store(out_ptr + elem_offset);
    }
  }
}

template <bool interleave, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType, typename QuantType>
__global__ void __launch_bounds__(128, 16) RopeQuantizeTiledKernel(
    DType* q_rope_in, DType* k_rope_in, DType* q_nope_in, DType* k_nope_in,
    QuantType* q_rope_out, QuantType* k_rope_out, QuantType* q_nope_out,
    QuantType* k_nope_out, const float* __restrict__ inv_freq,
    IdType* __restrict__ pos_ids, uint32_t nnz, uint32_t num_qo_heads,
    uint32_t num_kv_heads, uint32_t rope_dim, uint32_t no_rope_dim,
    size_t q_rope_in_stride_n, size_t q_rope_in_stride_h,
    size_t q_nope_in_stride_n, size_t q_nope_in_stride_h,
    size_t q_rope_out_stride_n, size_t q_rope_out_stride_h,
    size_t q_nope_out_stride_n, size_t q_nope_out_stride_h,
    size_t k_rope_in_stride, size_t k_rope_in_stride_h, size_t k_nope_in_stride,
    size_t k_nope_in_stride_h, size_t k_rope_out_stride,
    size_t k_rope_out_stride_h, size_t k_nope_out_stride,
    size_t k_nope_out_stride_h, float quant_scale_q, float quant_scale_kv) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  uint32_t bdy = blockDim.y;

  uint32_t rope_chunk_size = rope_dim;
  uint32_t rope_chunks = (rope_dim + rope_chunk_size - 1) / rope_chunk_size;
  uint32_t no_rope_chunks =
      (no_rope_dim + rope_chunk_size - 1) / rope_chunk_size;

  uint32_t q_rope_end = num_qo_heads * rope_chunks;
  uint32_t k_rope_end = q_rope_end + num_kv_heads * rope_chunks;
  uint32_t k_nope_end = k_rope_end + num_kv_heads * no_rope_chunks;

  if (bx * bdy + ty < nnz) {
    const uint32_t idx = bx * bdy + ty;
    const IdType pos = pos_ids[idx];
    const float fpos = static_cast<float>(pos);

    uint32_t vec_idx;
    if constexpr (interleave) {
      vec_idx = (tx * vec_size) / 2;
    } else {
      vec_idx = (tx * vec_size) % (rope_dim / 2);
    }

    if (by < q_rope_end) {
      uint32_t q_head_idx = by / rope_chunks;
      uint32_t elem_offset = (by % rope_chunks) * rope_chunk_size;

      DType* in_ptr = q_rope_in + get_elem_offset_impl(
                                      idx, q_head_idx, elem_offset,
                                      q_rope_in_stride_n, q_rope_in_stride_h);
      QuantType* out_ptr =
          q_rope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset,
                                            q_rope_out_stride_n,
                                            q_rope_out_stride_h);

      if constexpr (interleave) {
        tiled_rope_interleave_store<DType, QuantType, vec_size>(
            in_ptr, out_ptr, inv_freq, vec_idx, fpos, rope_dim, quant_scale_q,
            tx);
      } else {
        tiled_rope_neox_store<DType, QuantType, vec_size>(
            in_ptr, out_ptr, inv_freq, vec_idx, fpos, rope_dim, quant_scale_q,
            tx);
      }

    } else if (by < k_rope_end) {
      uint32_t k_head_idx = (by - q_rope_end) / rope_chunks;
      uint32_t elem_offset =
          ((by - q_rope_end) % rope_chunks) * rope_chunk_size;

      DType* in_ptr = k_rope_in + get_elem_offset_impl(
                                      idx, k_head_idx, elem_offset,
                                      k_rope_in_stride, k_rope_in_stride_h);
      QuantType* out_ptr =
          k_rope_out + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                            k_rope_out_stride,
                                            k_rope_out_stride_h);

      if constexpr (interleave) {
        tiled_rope_interleave_store<DType, QuantType, vec_size>(
            in_ptr, out_ptr, inv_freq, vec_idx, fpos, rope_dim, quant_scale_kv,
            tx);
      } else {
        tiled_rope_neox_store<DType, QuantType, vec_size>(
            in_ptr, out_ptr, inv_freq, vec_idx, fpos, rope_dim, quant_scale_kv,
            tx);
      }

    } else if (by < k_nope_end) {
      uint32_t k_head_idx = (by - k_rope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_rope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

      DType* in_ptr = k_nope_in + get_elem_offset_impl(
                                      idx, k_head_idx, elem_offset,
                                      k_nope_in_stride, k_nope_in_stride_h);
      QuantType* out_ptr =
          k_nope_out + get_elem_offset_impl(idx, k_head_idx, elem_offset,
                                            k_nope_out_stride,
                                            k_nope_out_stride_h);

      uint32_t chunk_valid =
          (elem_offset < no_rope_dim)
              ? min(rope_chunk_size, no_rope_dim - elem_offset)
              : 0u;
      scale_store_partial_chunk<DType, QuantType, vec_size>(
          in_ptr, out_ptr, tx * vec_size, chunk_valid, quant_scale_kv);

    } else {
      uint32_t q_head_idx = (by - k_nope_end) / no_rope_chunks;
      uint32_t nope_chunk_idx = (by - k_nope_end) % no_rope_chunks;
      uint32_t elem_offset = nope_chunk_idx * rope_chunk_size;

      DType* in_ptr = q_nope_in + get_elem_offset_impl(
                                      idx, q_head_idx, elem_offset,
                                      q_nope_in_stride_n, q_nope_in_stride_h);
      QuantType* out_ptr =
          q_nope_out + get_elem_offset_impl(idx, q_head_idx, elem_offset,
                                            q_nope_out_stride_n,
                                            q_nope_out_stride_h);

      uint32_t chunk_valid =
          (elem_offset < no_rope_dim)
              ? min(rope_chunk_size, no_rope_dim - elem_offset)
              : 0u;
      scale_store_partial_chunk<DType, QuantType, vec_size>(
          in_ptr, out_ptr, tx * vec_size, chunk_valid, quant_scale_q);
    }
  }
}

// ============================================================================
// Split-fused cache kernels: RoPE + KV cache scatter-write
//
// Kernel A: RopeOnlyFusedCacheKernel (bdx=4, bdy=32) for rope_dim=64
//   Q rope -> q_rope_out,  K rope -> cache[slot, no_rope_dim:]
//
// Kernel B: NopeScaleQuantFusedCacheKernel (bdx=32, bdy=4) for nope_dim=512
//   Q nope -> q_nope_out,  K nope -> cache[slot, 0:no_rope_dim]
// ============================================================================

template <bool interleave, uint32_t vec_size, uint32_t bdx, typename DType,
          typename IdType, typename QuantType>
__global__ void __launch_bounds__(128, 16) RopeOnlyFusedCacheKernel(
    DType* q_rope_in, QuantType* q_rope_out, DType* k_rope_in,
    QuantType* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ inv_freq, IdType* __restrict__ pos_ids,
    uint32_t nnz, uint32_t num_actual_tokens, uint32_t num_qo_heads,
    uint32_t num_kv_heads, uint32_t rope_dim, uint32_t no_rope_dim,
    size_t q_rope_in_stride_n, size_t q_rope_in_stride_h,
    size_t q_rope_out_stride_n, size_t q_rope_out_stride_h,
    size_t k_rope_in_stride, size_t k_rope_in_stride_h,
    int64_t cache_stride_block, int64_t cache_stride_entry, int block_size,
    float quant_scale_q, float quant_scale_kv) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  uint32_t bdy = blockDim.y;

  if (bx * bdy + ty >= nnz) return;
  const uint32_t idx = bx * bdy + ty;
  const IdType pos = pos_ids[idx];
  const float fpos = static_cast<float>(pos);

  uint32_t vec_idx;
  if constexpr (interleave) {
    vec_idx = (tx * vec_size) / 2;
  } else {
    vec_idx = (tx * vec_size) % (rope_dim / 2);
  }

  if (by < num_qo_heads) {
    // Q rope -> q_rope_out
    DType* in_ptr =
        q_rope_in + get_elem_offset_impl(idx, by, 0u, q_rope_in_stride_n,
                                         q_rope_in_stride_h);
    QuantType* out_ptr =
        q_rope_out + get_elem_offset_impl(idx, by, 0u, q_rope_out_stride_n,
                                          q_rope_out_stride_h);
    if constexpr (interleave) {
      tiled_rope_interleave_store<DType, QuantType, vec_size>(
          in_ptr, out_ptr, inv_freq, vec_idx, fpos, rope_dim, quant_scale_q,
          tx);
    } else {
      tiled_rope_neox_store<DType, QuantType, vec_size>(
          in_ptr, out_ptr, inv_freq, vec_idx, fpos, rope_dim, quant_scale_q,
          tx);
    }
  } else {
    // K rope -> cache[slot, no_rope_dim:]
    uint32_t k_head_idx = by - num_qo_heads;
    DType* in_ptr =
        k_rope_in + get_elem_offset_impl(idx, k_head_idx, 0u, k_rope_in_stride,
                                         k_rope_in_stride_h);
    if (idx < num_actual_tokens) {
      int64_t slot = slot_mapping[idx];
      if (slot >= 0) {
        QuantType* dst = kv_cache + (slot / block_size) * cache_stride_block +
                         (slot % block_size) * cache_stride_entry + no_rope_dim;
        if constexpr (interleave) {
          tiled_rope_interleave_store<DType, QuantType, vec_size>(
              in_ptr, dst, inv_freq, vec_idx, fpos, rope_dim, quant_scale_kv,
              tx);
        } else {
          tiled_rope_neox_store<DType, QuantType, vec_size>(
              in_ptr, dst, inv_freq, vec_idx, fpos, rope_dim, quant_scale_kv,
              tx);
        }
      }
    }
  }
}

template <uint32_t vec_size, typename DType, typename QuantType>
__global__ void __launch_bounds__(128, 16) NopeScaleQuantFusedCacheKernel(
    const DType* __restrict__ q_nope_in, QuantType* __restrict__ q_nope_out,
    const DType* __restrict__ k_nope_in, QuantType* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping, uint32_t nnz,
    uint32_t num_actual_tokens, uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t dim, size_t q_stride_n, size_t q_stride_h, size_t q_out_stride_n,
    size_t q_out_stride_h, size_t k_stride_n, size_t k_stride_h,
    int64_t cache_stride_block, int64_t cache_stride_entry, int block_size,
    float scale_q, float scale_kv) {
  uint32_t bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  uint32_t by = blockIdx.y;
  uint32_t bdy = blockDim.y;

  if (bx * bdy + ty >= nnz) return;
  const uint32_t idx = bx * bdy + ty;
  const uint32_t offset = tx * vec_size;
  if (offset >= dim) return;

  if (by < num_qo_heads) {
    // Q nope -> q_nope_out
    const DType* in_ptr = q_nope_in + idx * q_stride_n + by * q_stride_h;
    QuantType* out_ptr =
        q_nope_out + idx * q_out_stride_n + by * q_out_stride_h;
    vec_t<float, vec_size> v;
    v.cast_load(in_ptr + offset);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) v[i] *= scale_q;
    v.cast_store(out_ptr + offset);
  } else {
    // K nope -> cache[slot, 0:dim]
    uint32_t k_head_idx = by - num_qo_heads;
    const DType* in_ptr =
        k_nope_in + idx * k_stride_n + k_head_idx * k_stride_h;
    vec_t<float, vec_size> v;
    v.cast_load(in_ptr + offset);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) v[i] *= scale_kv;
    if (idx < num_actual_tokens) {
      int64_t slot = slot_mapping[idx];
      if (slot >= 0) {
        QuantType* dst = kv_cache + (slot / block_size) * cache_stride_block +
                         (slot % block_size) * cache_stride_entry;
        v.cast_store(dst + offset);
      }
    }
  }
}

}  // namespace mla_rope
}  // namespace vllm
