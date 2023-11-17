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

namespace vllm {

// FP16 vector types for Q, K, V.
template<>
struct Vec<uint16_t, 1> {
  using Type = uint16_t;
};
template<>
struct Vec<uint16_t, 2> {
  using Type = uint32_t;
};
template<>
struct Vec<uint16_t, 4> {
  using Type = sycl::uint2;
};
template<>
struct Vec<uint16_t, 8> {
  using Type = sycl::uint4;
};

// FP32 accumulator vector types corresponding to Vec.
template<>
struct FloatVec<uint16_t> {
  using Type = float;
};
template<>
struct FloatVec<uint32_t> {
  using Type = sycl::float2;
};
template <> struct FloatVec<sycl::uint2> {
  using Type = Float4_;
};
template <> struct FloatVec<sycl::uint4> {
  using Type = Float8_;
};

// Utility functions for type conversions.
inline uint32_t h0_h0(uint16_t a) {
  uint32_t b;
  /*
  DPCT1053:27: Migration of device assembly code is not supported.
  */
  asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));
  return b;
}

inline float half_to_float(uint16_t h) {
  float f;
  /*
  DPCT1053:28: Migration of device assembly code is not supported.
  */
  asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
  return f;
}

inline sycl::float2 half2_to_float2(uint32_t v) {
  uint16_t lo, hi;
  /*
  DPCT1053:29: Migration of device assembly code is not supported.
  */
  asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
  return sycl::float2(half_to_float(lo), half_to_float(hi));
}

inline uint16_t float_to_half(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  /*
  DPCT1053:30: Migration of device assembly code is not supported.
  */
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
  return tmp.u16[0];
}

inline uint32_t float2_to_half2(sycl::float2 f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP >= 800
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
#else
  /*
  DPCT1053:31: Migration of device assembly code is not supported.
  */
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x()));
  /*
  DPCT1053:32: Migration of device assembly code is not supported.
  */
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y()));
#endif
  return tmp.u32;
}

// Vector addition.
inline uint16_t add(uint16_t a, uint16_t b) {
  uint16_t c;
  /*
  DPCT1053:33: Migration of device assembly code is not supported.
  */
  asm volatile("add.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
  return c;
}

inline uint32_t add(uint32_t a, uint32_t b) {
  uint32_t c;
  /*
  DPCT1053:34: Migration of device assembly code is not supported.
  */
  asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

inline sycl::uint2 add(sycl::uint2 a, sycl::uint2 b) {
  sycl::uint2 c;
  c.x() = add(a.x(), b.x());
  c.y() = add(a.y(), b.y());
  return c;
}

inline sycl::uint4 add(sycl::uint4 a, sycl::uint4 b) {
  sycl::uint4 c;
  c.x() = add(a.x(), b.x());
  c.y() = add(a.y(), b.y());
  c.z() = add(a.z(), b.z());
  c.w() = add(a.w(), b.w());
  return c;
}

inline sycl::float2 add(uint32_t a, sycl::float2 fb) {
  sycl::float2 fa = half2_to_float2(a);
  return add(fa, fb);
}

inline Float4_ add(sycl::uint2 a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x(), fb.x);
  fc.y = add(a.y(), fb.y);
  return fc;
}

inline Float8_ add(sycl::uint4 a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x(), fb.x);
  fc.y = add(a.y(), fb.y);
  fc.z = add(a.z(), fb.z);
  fc.w = add(a.w(), fb.w);
  return fc;
}

// Vector multiplication.
template<>
inline uint16_t mul(uint16_t a, uint16_t b) {
  uint16_t c;
  /*
  DPCT1053:35: Migration of device assembly code is not supported.
  */
  asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
  return c;
}

template<>
inline uint32_t mul(uint32_t a, uint32_t b) {
  uint32_t c;
  /*
  DPCT1053:36: Migration of device assembly code is not supported.
  */
  asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

template<>
inline uint32_t mul(uint16_t a, uint32_t b) {
  return mul<uint32_t, uint32_t, uint32_t>(h0_h0(a), b);
}

template <> inline sycl::uint2 mul(sycl::uint2 a, sycl::uint2 b) {
  sycl::uint2 c;
  c.x() = mul<uint32_t, uint32_t, uint32_t>(a.x(), b.x());
  c.y() = mul<uint32_t, uint32_t, uint32_t>(a.y(), b.y());
  return c;
}

template <> inline sycl::uint2 mul(uint16_t a, sycl::uint2 b) {
  uint32_t s = h0_h0(a);
  sycl::uint2 c;
  c.x() = mul<uint32_t, uint32_t, uint32_t>(s, b.x());
  c.y() = mul<uint32_t, uint32_t, uint32_t>(s, b.y());
  return c;
}

template <> inline sycl::uint4 mul(sycl::uint4 a, sycl::uint4 b) {
  sycl::uint4 c;
  c.x() = mul<uint32_t, uint32_t, uint32_t>(a.x(), b.x());
  c.y() = mul<uint32_t, uint32_t, uint32_t>(a.y(), b.y());
  c.z() = mul<uint32_t, uint32_t, uint32_t>(a.z(), b.z());
  c.w() = mul<uint32_t, uint32_t, uint32_t>(a.w(), b.w());
  return c;
}

template <> inline sycl::uint4 mul(uint16_t a, sycl::uint4 b) {
  uint32_t s = h0_h0(a);
  sycl::uint4 c;
  c.x() = mul<uint32_t, uint32_t, uint32_t>(s, b.x());
  c.y() = mul<uint32_t, uint32_t, uint32_t>(s, b.y());
  c.z() = mul<uint32_t, uint32_t, uint32_t>(s, b.z());
  c.w() = mul<uint32_t, uint32_t, uint32_t>(s, b.w());
  return c;
}

template<>
inline float mul(uint16_t a, uint16_t b) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb;
}

template <> inline sycl::float2 mul(uint32_t a, uint32_t b) {
  sycl::float2 fa = half2_to_float2(a);
  sycl::float2 fb = half2_to_float2(b);
  return mul<sycl::float2, sycl::float2, sycl::float2>(fa, fb);
}

template <> inline sycl::float2 mul(uint16_t a, uint32_t b) {
  return mul<sycl::float2, uint32_t, uint32_t>(h0_h0(a), b);
}

template <> inline Float4_ mul(sycl::uint2 a, sycl::uint2 b) {
  Float4_ fc;
  fc.x = mul<sycl::float2, uint32_t, uint32_t>(a.x(), b.x());
  fc.y = mul<sycl::float2, uint32_t, uint32_t>(a.y(), b.y());
  return fc;
}

template <> inline Float4_ mul(uint16_t a, sycl::uint2 b) {
  uint32_t s = h0_h0(a);
  Float4_ fc;
  fc.x = mul<sycl::float2, uint32_t, uint32_t>(s, b.x());
  fc.y = mul<sycl::float2, uint32_t, uint32_t>(s, b.y());
  return fc;
}

template <> inline Float8_ mul(sycl::uint4 a, sycl::uint4 b) {
  Float8_ fc;
  fc.x = mul<sycl::float2, uint32_t, uint32_t>(a.x(), b.x());
  fc.y = mul<sycl::float2, uint32_t, uint32_t>(a.y(), b.y());
  fc.z = mul<sycl::float2, uint32_t, uint32_t>(a.z(), b.z());
  fc.w = mul<sycl::float2, uint32_t, uint32_t>(a.w(), b.w());
  return fc;
}

template <> inline Float8_ mul(uint16_t a, sycl::uint4 b) {
  uint32_t s = h0_h0(a);
  Float8_ fc;
  fc.x = mul<sycl::float2, uint32_t, uint32_t>(s, b.x());
  fc.y = mul<sycl::float2, uint32_t, uint32_t>(s, b.y());
  fc.z = mul<sycl::float2, uint32_t, uint32_t>(s, b.z());
  fc.w = mul<sycl::float2, uint32_t, uint32_t>(s, b.w());
  return fc;
}

// Vector fused multiply-add.
inline uint32_t fma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  /*
  DPCT1053:37: Migration of device assembly code is not supported.
  */
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
               : "=r"(d)
               : "r"(a), "r"(b), "r"(c));
  return d;
}

inline uint32_t fma(uint16_t a, uint32_t b, uint32_t c) {
  return fma(h0_h0(a), b, c);
}

inline sycl::uint2 fma(sycl::uint2 a, sycl::uint2 b, sycl::uint2 c) {
  sycl::uint2 d;
  d.x() = fma(a.x(), b.x(), c.x());
  d.y() = fma(a.y(), b.y(), c.y());
  return d;
}

inline sycl::uint2 fma(uint16_t a, sycl::uint2 b, sycl::uint2 c) {
  uint32_t s = h0_h0(a);
  sycl::uint2 d;
  d.x() = fma(s, b.x(), c.x());
  d.y() = fma(s, b.y(), c.y());
  return d;
}

inline sycl::uint4 fma(sycl::uint4 a, sycl::uint4 b, sycl::uint4 c) {
  sycl::uint4 d;
  d.x() = fma(a.x(), b.x(), c.x());
  d.y() = fma(a.y(), b.y(), c.y());
  d.z() = fma(a.z(), b.z(), c.z());
  d.w() = fma(a.w(), b.w(), c.w());
  return d;
}

inline sycl::uint4 fma(uint16_t a, sycl::uint4 b, sycl::uint4 c) {
  uint32_t s = h0_h0(a);
  sycl::uint4 d;
  d.x() = fma(s, b.x(), c.x());
  d.y() = fma(s, b.y(), c.y());
  d.z() = fma(s, b.z(), c.z());
  d.w() = fma(s, b.w(), c.w());
  return d;
}

inline float fma(uint16_t a, uint16_t b, float fc) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb + fc;
}

inline sycl::float2 fma(uint32_t a, uint32_t b, sycl::float2 fc) {
  sycl::float2 fa = half2_to_float2(a);
  sycl::float2 fb = half2_to_float2(b);
  return fma(fa, fb, fc);
}

inline sycl::float2 fma(uint16_t a, uint32_t b, sycl::float2 fc) {
  return fma(h0_h0(a), b, fc);
}

inline Float4_ fma(sycl::uint2 a, sycl::uint2 b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(a.x(), b.x(), fc.x);
  fd.y = fma(a.y(), b.y(), fc.y);
  return fd;
}

inline Float4_ fma(uint16_t a, sycl::uint2 b, Float4_ fc) {
  uint32_t s = h0_h0(a);
  Float4_ fd;
  fd.x = fma(s, b.x(), fc.x);
  fd.y = fma(s, b.y(), fc.y);
  return fd;
}

inline Float8_ fma(sycl::uint4 a, sycl::uint4 b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(a.x(), b.x(), fc.x);
  fd.y = fma(a.y(), b.y(), fc.y);
  fd.z = fma(a.z(), b.z(), fc.z);
  fd.w = fma(a.w(), b.w(), fc.w);
  return fd;
}

inline Float8_ fma(uint16_t a, sycl::uint4 b, Float8_ fc) {
  uint32_t s = h0_h0(a);
  Float8_ fd;
  fd.x = fma(s, b.x(), fc.x);
  fd.y = fma(s, b.y(), fc.y);
  fd.z = fma(s, b.z(), fc.z);
  fd.w = fma(s, b.w(), fc.w);
  return fd;
}

// Vector sum.
template<>
inline float sum(uint16_t v) {
  return half_to_float(v);
}

template<>
inline float sum(uint32_t v) {
  sycl::float2 tmp = half2_to_float2(v);
  return tmp.x() + tmp.y();
}

template <> inline float sum(sycl::uint2 v) {
  uint32_t c = add(v.x(), v.y());
  return sum(c);
}

template <> inline float sum(sycl::uint4 v) {
  uint32_t c = add(v.x(), v.y());
  c = add(c, v.z());
  c = add(c, v.w());
  return sum(c);
}

// From float32 to float16.
inline void from_float(uint16_t& dst, float src) {
  dst = float_to_half(src);
}

inline void from_float(uint32_t &dst, sycl::float2 src) {
  dst = float2_to_half2(src);
}

inline void from_float(sycl::uint2 &dst, Float4_ src) {
  dst.x() = float2_to_half2(src.x);
  dst.y() = float2_to_half2(src.y);
}

inline void from_float(sycl::uint4 &dst, Float8_ src) {
  dst.x() = float2_to_half2(src.x);
  dst.y() = float2_to_half2(src.y);
  dst.z() = float2_to_half2(src.z);
  dst.w() = float2_to_half2(src.w);
}

// From float16 to float32.
inline float to_float(uint16_t u) {
  return half_to_float(u);
}

inline sycl::float2 to_float(uint32_t u) {
  return half2_to_float2(u);
}

inline Float4_ to_float(sycl::uint2 u) {
  Float4_ tmp;
  tmp.x = half2_to_float2(u.x());
  tmp.y = half2_to_float2(u.y());
  return tmp;
}

inline Float8_ to_float(sycl::uint4 u) {
  Float8_ tmp;
  tmp.x = half2_to_float2(u.x());
  tmp.y = half2_to_float2(u.y());
  tmp.z = half2_to_float2(u.z());
  tmp.w = half2_to_float2(u.w());
  return tmp;
}

// Zero-out a variable.
inline void zero(uint16_t& dst) {
  dst = uint16_t(0);
}

} // namespace vllm
