/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * and https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "attention_generic.dp.hpp"
#include "dtype_float32.dp.hpp"

#include <stdint.h>
#include <oneapi/mkl/bfloat16.hpp>

namespace vllm {

// Define custom BF16 vector data types.
struct bf16_4_t {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
};

struct bf16_8_t {
  __nv_bfloat162 x;
  __nv_bfloat162 y;
  __nv_bfloat162 z;
  __nv_bfloat162 w;
};

// BF16 vector types for Q, K, V.
template <> struct Vec<oneapi::mkl::bfloat16, 1> {
  using Type = oneapi::mkl::bfloat16;
};
template <> struct Vec<oneapi::mkl::bfloat16, 2> {
  using Type = __nv_bfloat162;
};
template <> struct Vec<oneapi::mkl::bfloat16, 4> {
  using Type = bf16_4_t;
};
template <> struct Vec<oneapi::mkl::bfloat16, 8> {
  using Type = bf16_8_t;
};

// FP32 accumulator vector types corresponding to Vec.
template <> struct FloatVec<oneapi::mkl::bfloat16> {
  using Type = float;
};
template<>
struct FloatVec<__nv_bfloat162> {
  using Type = sycl::float2;
};
template<>
struct FloatVec<bf16_4_t> {
  using Type = Float4_;
};
template<>
struct FloatVec<bf16_8_t> {
  using Type = Float8_;
};

// Utility functions for type conversions.
inline sycl::float2 bf1622float2(const __nv_bfloat162 val) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  assert(false);
#else
  return __bfloat1622float2(val);
#endif
}

inline __nv_bfloat162 bf162bf162(const oneapi::mkl::bfloat16 val) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  assert(false);
#else
  return __bfloat162bfloat162(val);
#endif
}

// Vector addition.
inline oneapi::mkl::bfloat16 add(oneapi::mkl::bfloat16 a,
                                 oneapi::mkl::bfloat16 b) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  assert(false);
#else
  return a + b;
#endif
}

inline __nv_bfloat162 add(__nv_bfloat162 a, __nv_bfloat162 b) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  assert(false);
#else
  return __hadd2(a, b);
#endif
}

inline bf16_4_t add(bf16_4_t a, bf16_4_t b) {
  bf16_4_t c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline bf16_8_t add(bf16_8_t a, bf16_8_t b) {
  bf16_8_t c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

inline sycl::float2 add(__nv_bfloat162 a, sycl::float2 fb) {
  sycl::float2 fa = bf1622float2(a);
  return add(fa, fb);
}

inline Float4_ add(bf16_4_t a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  return fc;
}

inline Float8_ add(bf16_8_t a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
  return fc;
}

// Vector multiplication.
template <>
inline oneapi::mkl::bfloat16 mul(oneapi::mkl::bfloat16 a,
                                 oneapi::mkl::bfloat16 b) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  assert(false);
#else
  return __hmul(a, b);
#endif
}

template<>
inline __nv_bfloat162 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  assert(false);
#else
  return __hmul2(a, b);
#endif
}

template <>
inline __nv_bfloat162 mul(oneapi::mkl::bfloat16 a, __nv_bfloat162 b) {
  return mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(bf162bf162(a), b);
}

template<>
inline bf16_4_t mul(bf16_4_t a, bf16_4_t b) {
  bf16_4_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  return c;
}

template <> inline bf16_4_t mul(oneapi::mkl::bfloat16 a, bf16_4_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_4_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  return c;
}

template<>
inline bf16_8_t mul(bf16_8_t a, bf16_8_t b) {
  bf16_8_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.z, b.z);
  c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(a.w, b.w);
  return c;
}

template <> inline bf16_8_t mul(oneapi::mkl::bfloat16 a, bf16_8_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_8_t c;
  c.x = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  c.y = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  c.z = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.z);
  c.w = mul<__nv_bfloat162, __nv_bfloat162, __nv_bfloat162>(s, b.w);
  return c;
}

template <> inline float mul(oneapi::mkl::bfloat16 a, oneapi::mkl::bfloat16 b) {
  float fa = static_cast<float>(a);
  float fb = static_cast<float>(b);
  return fa * fb;
}

template <> inline sycl::float2 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
  sycl::float2 fa = bf1622float2(a);
  sycl::float2 fb = bf1622float2(b);
  return mul<sycl::float2, sycl::float2, sycl::float2>(fa, fb);
}

template <> inline sycl::float2 mul(oneapi::mkl::bfloat16 a, __nv_bfloat162 b) {
  return mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(bf162bf162(a), b);
}

template<>
inline Float4_ mul(bf16_4_t a, bf16_4_t b) {
  Float4_ fc;
  fc.x = mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  fc.y = mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  return fc;
}

template <> inline Float4_ mul(oneapi::mkl::bfloat16 a, bf16_4_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  Float4_ fc;
  fc.x = mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  fc.y = mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  return fc;
}

template<>
inline Float8_ mul(bf16_8_t a, bf16_8_t b) {
  Float8_ fc;
  fc.x = mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(a.x, b.x);
  fc.y = mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(a.y, b.y);
  fc.z = mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(a.z, b.z);
  fc.w = mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(a.w, b.w);
  return fc;
}

template <> inline Float8_ mul(oneapi::mkl::bfloat16 a, bf16_8_t b) {
  __nv_bfloat162 s = bf162bf162(a);
  Float8_ fc;
  fc.x = mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(s, b.x);
  fc.y = mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(s, b.y);
  fc.z = mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(s, b.z);
  fc.w = mul<sycl::float2, __nv_bfloat162, __nv_bfloat162>(s, b.w);
  return fc;
}

// Vector fused multiply-add.
inline __nv_bfloat162 fma(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  assert(false);
#else
  return __hfma2(a, b, c);
#endif
}

inline __nv_bfloat162 fma(oneapi::mkl::bfloat16 a, __nv_bfloat162 b,
                          __nv_bfloat162 c) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  assert(false);
#else
  return __hfma2(bf162bf162(a), b, c);
#endif
}

inline bf16_4_t fma(bf16_4_t a, bf16_4_t b, bf16_4_t c) {
  bf16_4_t d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline bf16_4_t fma(oneapi::mkl::bfloat16 a, bf16_4_t b, bf16_4_t c) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_4_t d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  return d;
}

inline bf16_8_t fma(bf16_8_t a, bf16_8_t b, bf16_8_t c) {
  bf16_8_t d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline bf16_8_t fma(oneapi::mkl::bfloat16 a, bf16_8_t b, bf16_8_t c) {
  __nv_bfloat162 s = bf162bf162(a);
  bf16_8_t d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  d.z = fma(s, b.z, c.z);
  d.w = fma(s, b.w, c.w);
  return d;
}

inline float fma(oneapi::mkl::bfloat16 a, oneapi::mkl::bfloat16 b, float fc) {
  return static_cast<float>(a) * static_cast<float>(b) + fc;
}

inline sycl::float2 fma(__nv_bfloat162 a, __nv_bfloat162 b, sycl::float2 fc) {
  sycl::float2 fa = bf1622float2(a);
  sycl::float2 fb = bf1622float2(b);
  return fma(fa, fb, fc);
}

inline sycl::float2 fma(oneapi::mkl::bfloat16 a, __nv_bfloat162 b,
                        sycl::float2 fc) {
  return fma(bf162bf162(a), b, fc);
}

inline Float4_ fma(bf16_4_t a, bf16_4_t b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  return fd;
}

inline Float4_ fma(oneapi::mkl::bfloat16 a, bf16_4_t b, Float4_ fc) {
  __nv_bfloat162 s = bf162bf162(a);
  Float4_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  return fd;
}

inline Float8_ fma(bf16_8_t a, bf16_8_t b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  fd.z = fma(a.z, b.z, fc.z);
  fd.w = fma(a.w, b.w, fc.w);
  return fd;
}

inline Float8_ fma(oneapi::mkl::bfloat16 a, bf16_8_t b, Float8_ fc) {
  __nv_bfloat162 s = bf162bf162(a);
  Float8_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  fd.z = fma(s, b.z, fc.z);
  fd.w = fma(s, b.w, fc.w);
  return fd;
}

// Vector sum.
template <> inline float sum(oneapi::mkl::bfloat16 v) {
  return static_cast<float>(v);
}

template<>
inline float sum(__nv_bfloat162 v) {
  sycl::float2 vf = bf1622float2(v);
  return vf.x() + vf.y();
}

template<>
inline float sum(bf16_4_t v) {
  return sum(v.x) + sum(v.y);
}

template<>
inline float sum(bf16_8_t v) {
  return sum(v.x) + sum(v.y) + sum(v.z) + sum(v.w);
}

// From float32 to bfloat16.
inline void from_float(oneapi::mkl::bfloat16 &dst, float src) {
  dst = oneapi::mkl::bfloat16(src);
}

inline void from_float(__nv_bfloat162 &dst, sycl::float2 src) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  assert(false);
#else
  dst = __float22bfloat162_rn(src);
#endif
}

inline void from_float(bf16_4_t& dst, Float4_ src) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  assert(false);
#else
  dst.x = __float22bfloat162_rn(src.x);
  dst.y = __float22bfloat162_rn(src.y);
#endif
}

inline void from_float(bf16_8_t& dst, Float8_ src) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  assert(false);
#else
  dst.x = __float22bfloat162_rn(src.x);
  dst.y = __float22bfloat162_rn(src.y);
  dst.z = __float22bfloat162_rn(src.z);
  dst.w = __float22bfloat162_rn(src.w);
#endif
}

// From bfloat16 to float32.
inline float to_float(oneapi::mkl::bfloat16 u) {
  return static_cast<float>(u);
}

// Zero-out a variable.
inline void zero(oneapi::mkl::bfloat16 &dst) {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 800
  assert(false);
#else
  // Same as CUDART_ZERO_BF16 introduced in CUDA 12.2.
  dst = __ushort_as_bfloat16((unsigned short)0x0000U);
#endif
}

} // namespace vllm
