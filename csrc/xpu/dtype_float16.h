/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * and
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
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

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include "attention_generic.h"
#include "dtype_float32.h"
#include "utils.h"

#include <stdint.h>

namespace vllm {

// FP16 vector types for Q, K, V.
template <>
struct Vec<sycl::half, 1> {
  using Type = sycl::half;
};
template <>
struct Vec<sycl::half, 2> {
  using Type = sycl::half2;
};
template <>
struct Vec<sycl::half, 4> {
  using Type = sycl::half4;
};
template <>
struct Vec<sycl::half, 8> {
  using Type = sycl::half8;
};

template <>
struct FloatVec<sycl::half> {
  using Type = float;
};
template <>
struct FloatVec<sycl::half2> {
  using Type = sycl::float2;
};

template <>
struct FloatVec<sycl::half4> {
  using Type = Float4_;
};
template <>
struct FloatVec<sycl::half8> {
  using Type = Float8_;
};

// Utility functions for type conversions.
inline sycl::half2 h0_h0(sycl::half a) {
  return sycl::half2{a, a};
}

inline float half_to_float(sycl::half h) {
  return float(h);
}

inline sycl::float2 half2_to_float2(sycl::half2 v) {

  return sycl::float2(half_to_float(v.x()), half_to_float(v.y()));
}

inline sycl::half float_to_half(float f) {
  return sycl::half(f);
}

inline sycl::half2 float2_to_half2(sycl::float2 f) {
  return sycl::half2{float_to_half(f.x()), float_to_half(f.y())};
}

// Vector addition.
inline sycl::half add(sycl::half a, sycl::half b) {
  return sycl_half_add(a,b);
}

inline sycl::half2 add(sycl::half2 a, sycl::half2 b) {
  auto val = sycl_half_add2(a, b);
  return (val);
}

inline sycl::half4 add(sycl::half4 a, sycl::half4 b) {
  sycl::half4 c;
  c.x() = add(a.x(), b.x());
  c.y() = add(a.y(), b.y());
  c.z() = add(a.z(), b.z());
  c.w() = add(a.w(), b.w());
  return c;
}

inline sycl::half8 add(sycl::half8 a, sycl::half8 b) {
  sycl::half8 c;
  c.s0() = add(a.s0(), b.s0());
  c.s1() = add(a.s1(), b.s1());
  c.s2() = add(a.s2(), b.s2());
  c.s3() = add(a.s3(), b.s3());
  c.s4() = add(a.s4(), b.s4());
  c.s5() = add(a.s5(), b.s5());
  c.s6() = add(a.s6(), b.s6());
  c.s7() = add(a.s7(), b.s7());
  return c;
}

inline sycl::float2 add(sycl::half2 a, sycl::float2 fb) {
  sycl::float2 fa = half2_to_float2(a);
  return add(fa, fb);
}

inline Float4_ add(sycl::half4 a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(sycl::half2{a.x(), a.y()}, fb.x);
  fc.y = add(sycl::half2{a.z(), a.w()}, fb.y);
  return fc;
}

inline Float8_ add(sycl::half8 a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(sycl::half2{a.s0(), a.s1()}, fb.x);
  fc.y = add(sycl::half2{a.s2(), a.s3()}, fb.y);
  fc.z = add(sycl::half2{a.s4(), a.s5()}, fb.z);
  fc.w = add(sycl::half2{a.s6(), a.s7()}, fb.w);
  return fc;
}

// Vector multiplication.
template <>
inline sycl::half mul(sycl::half a, sycl::half b) {
  auto val = sycl_half_mul((a), (b));
  return (val);
}

template <>
inline sycl::half2 mul(sycl::half2 a, sycl::half2 b) {
  auto val = sycl_half_mul2((a), (b));
  return (val);
}

template <>
inline sycl::half2 mul(sycl::half a, sycl::half2 b) {
  return mul<sycl::half2, sycl::half2, sycl::half2>(h0_h0(a), b);
}


template <>
inline sycl::half4 mul(sycl::half4 a, sycl::half4 b) {
  sycl::half4 c;
  c.x() = mul<sycl::half, sycl::half, sycl::half>(a.x(), b.x());
  c.y() = mul<sycl::half, sycl::half, sycl::half>(a.y(), b.y());
  c.z() = mul<sycl::half, sycl::half, sycl::half>(a.z(), b.z());
  c.w() = mul<sycl::half, sycl::half, sycl::half>(a.w(), b.w());
  return c;
}

template <>
inline sycl::half4 mul(sycl::half a, sycl::half4 b) {
  sycl::half4 c;
  c.x() = mul<sycl::half, sycl::half, sycl::half>(a, b.x());
  c.y() = mul<sycl::half, sycl::half, sycl::half>(a, b.y());
  c.z() = mul<sycl::half, sycl::half, sycl::half>(a, b.z());
  c.w() = mul<sycl::half, sycl::half, sycl::half>(a, b.w());
  return c;
}

template <>
inline sycl::half8 mul(sycl::half8 a, sycl::half8 b) {
  sycl::half8 c;
  c.s0() = mul<sycl::half, sycl::half, sycl::half>(a.s0(), b.s0());
  c.s1() = mul<sycl::half, sycl::half, sycl::half>(a.s1(), b.s1());
  c.s2() = mul<sycl::half, sycl::half, sycl::half>(a.s2(), b.s2());
  c.s3() = mul<sycl::half, sycl::half, sycl::half>(a.s3(), b.s3());
  c.s4() = mul<sycl::half, sycl::half, sycl::half>(a.s4(), b.s4());
  c.s5() = mul<sycl::half, sycl::half, sycl::half>(a.s5(), b.s5());
  c.s6() = mul<sycl::half, sycl::half, sycl::half>(a.s6(), b.s6());
  c.s7() = mul<sycl::half, sycl::half, sycl::half>(a.s7(), b.s7());
  return c;
}

template <>
inline sycl::half8 mul(sycl::half a, sycl::half8 b) {
  sycl::half8 c;
  c.s0() = mul<sycl::half, sycl::half, sycl::half>(a, b.s0());
  c.s1() = mul<sycl::half, sycl::half, sycl::half>(a, b.s1());
  c.s2() = mul<sycl::half, sycl::half, sycl::half>(a, b.s2());
  c.s3() = mul<sycl::half, sycl::half, sycl::half>(a, b.s3());
  c.s4() = mul<sycl::half, sycl::half, sycl::half>(a, b.s4());
  c.s5() = mul<sycl::half, sycl::half, sycl::half>(a, b.s5());
  c.s6() = mul<sycl::half, sycl::half, sycl::half>(a, b.s6());
  c.s7() = mul<sycl::half, sycl::half, sycl::half>(a, b.s7());
  return c;
}

template <>
inline float mul(sycl::half a, sycl::half b) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb;
}

template <>
inline sycl::float2 mul(sycl::half2 a, sycl::half2 b) {
  sycl::float2 fa = half2_to_float2(a);
  sycl::float2 fb = half2_to_float2(b);
  return mul<sycl::float2, sycl::float2, sycl::float2>(fa, fb);
}

template <>
inline sycl::float2 mul(sycl::half a, sycl::half2 b) {
  return mul<sycl::float2, sycl::half2, sycl::half2>(h0_h0(a), b);
}

template <>
inline Float4_ mul(sycl::half4 a, sycl::half4 b) {
  Float4_ fc;
  fc.x = mul<sycl::float2, sycl::half2, sycl::half2>(
      sycl::half2{a.x(), a.y()}, sycl::half2{b.x(), b.y()});
  fc.y = mul<sycl::float2, sycl::half2, sycl::half2>(
      sycl::half2{a.z(), a.w()}, sycl::half2{b.z(), b.w()});
  return fc;
}

template <>
inline Float4_ mul(sycl::half a, sycl::half4 b) {
  sycl::half2 s = h0_h0(a);
  Float4_ fc;

  fc.x =
      mul<sycl::float2, sycl::half2, sycl::half2>(s, sycl::half2{b.x(), b.y()});
  fc.y =
      mul<sycl::float2, sycl::half2, sycl::half2>(s, sycl::half2{b.z(), b.w()});
  return fc;
}

template <>
inline Float8_ mul(sycl::half8 a, sycl::half8 b) {
  Float8_ fc;
  fc.x = mul<sycl::float2, sycl::half2, sycl::half2>(
      sycl::half2{a.s0(), a.s1()}, sycl::half2{b.s0(), b.s1()});
  fc.y = mul<sycl::float2, sycl::half2, sycl::half2>(
      sycl::half2{a.s2(), a.s3()}, sycl::half2{b.s2(), b.s3()});
  fc.z = mul<sycl::float2, sycl::half2, sycl::half2>(
      sycl::half2{a.s4(), a.s5()}, sycl::half2{b.s4(), b.s5()});
  fc.w = mul<sycl::float2, sycl::half2, sycl::half2>(
      sycl::half2{a.s6(), a.s7()}, sycl::half2{b.s6(), b.s7()});
  return fc;
}

template <>
inline Float8_ mul(sycl::half a, sycl::half8 b) {
  sycl::half2 s = h0_h0(a);
  Float8_ fc;
  fc.x = mul<sycl::float2, sycl::half2, sycl::half2>(
      s, sycl::half2{b.s0(), b.s1()});
  fc.y = mul<sycl::float2, sycl::half2, sycl::half2>(
      s, sycl::half2{b.s2(), b.s3()});
  fc.z = mul<sycl::float2, sycl::half2, sycl::half2>(
      s, sycl::half2{b.s4(), b.s5()});
  fc.w = mul<sycl::float2, sycl::half2, sycl::half2>(
      s, sycl::half2{b.s6(), b.s7()});
  return fc;
}

// Vector fused multiply-add.
inline sycl::half2 fma(sycl::half2 a, sycl::half2 b, sycl::half2 c) {
  auto val = sycl_half_fma2((a), (b), (c));
  return (val);
}

inline sycl::half2 fma(sycl::half a, sycl::half2 b, sycl::half2 c) {
  return fma(h0_h0(a), b, c);
}

inline sycl::half4 fma(sycl::half4 a, sycl::half4 b, sycl::half4 c) {
  sycl::half4 d;
  d.x() = fma(a.x(), b.x(), c.x());
  d.y() = fma(a.y(), b.y(), c.y());
  d.z() = fma(a.z(), b.z(), c.z());
  d.w() = fma(a.w(), b.w(), c.w());
  return d;
}

inline sycl::half4 fma(sycl::half a, sycl::half4 b, sycl::half4 c) {
  sycl::half4 s = sycl::half4{a, a, a, a};
  return fma(s, b, c);
}

inline sycl::half8 fma(sycl::half8 a, sycl::half8 b, sycl::half8 c) {
  sycl::half8 d;
  d.s0() = fma(a.s0(), b.s0(), c.s0());
  d.s1() = fma(a.s1(), b.s1(), c.s1());
  d.s2() = fma(a.s2(), b.s2(), c.s2());
  d.s3() = fma(a.s3(), b.s3(), c.s3());
  d.s4() = fma(a.s4(), b.s4(), c.s4());
  d.s5() = fma(a.s5(), b.s5(), c.s5());
  d.s6() = fma(a.s6(), b.s6(), c.s6());
  d.s7() = fma(a.s7(), b.s7(), c.s7());
  return d;
}

inline sycl::half8 fma(sycl::half a, sycl::half8 b, sycl::half8 c) {
  sycl::half8 d;
  d.s0() = fma(a, b.s0(), c.s0());
  d.s1() = fma(a, b.s1(), c.s1());
  d.s2() = fma(a, b.s2(), c.s2());
  d.s3() = fma(a, b.s3(), c.s3());
  d.s4() = fma(a, b.s4(), c.s4());
  d.s5() = fma(a, b.s5(), c.s5());
  d.s6() = fma(a, b.s6(), c.s6());
  d.s7() = fma(a, b.s7(), c.s7());
  return d;
}

inline float fma(sycl::half a, sycl::half b, float fc) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return sycl::fma(fa, fb, fc);
}

inline sycl::float2 fma(sycl::half2 a, sycl::half2 b, sycl::float2 fc) {
  sycl::float2 fa = half2_to_float2(a);
  sycl::float2 fb = half2_to_float2(b);
  return fma(fa, fb, fc);
}

inline sycl::float2 fma(sycl::half a, sycl::half2 b, sycl::float2 fc) {
  return fma(h0_h0(a), b, fc);
}

inline Float4_ fma(sycl::half4 a, sycl::half4 b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(sycl::half2{a.x(), a.y()}, sycl::half2{b.x(), b.y()}, fc.x);
  fd.y = fma(sycl::half2{a.z(), a.w()}, sycl::half2{b.z(), b.w()}, fc.y);
  return fd;
}

inline Float4_ fma(sycl::half a, sycl::half4 b, Float4_ fc) {
  sycl::half4 s = sycl::half4{a, a, a, a};

  return fma(s, b, fc);
}

inline Float8_ fma(sycl::half8 a, sycl::half8 b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(sycl::half2{a.s0(), a.s1()}, sycl::half2{b.s0(), b.s1()}, fc.x);
  fd.y = fma(sycl::half2{a.s2(), a.s3()}, sycl::half2{b.s2(), b.s3()}, fc.y);
  fd.z = fma(sycl::half2{a.s4(), a.s5()}, sycl::half2{b.s4(), b.s5()}, fc.z);
  fd.w = fma(sycl::half2{a.s6(), a.s7()}, sycl::half2{b.s6(), b.s7()}, fc.w);
  return fd;
}

inline Float8_ fma(sycl::half a, sycl::half8 b, Float8_ fc) {
  sycl::half8 s = sycl::half8{a, a, a, a, a, a, a, a};

  return fma(s, b, fc);
}

// Vector sum.
template <>
inline float sum(sycl::half v) {
  return half_to_float(v);
}

template <>
inline float sum(sycl::half2 v) {
  sycl::float2 tmp = half2_to_float2(v);
  return tmp.x() + tmp.y();
}

template <>
inline float sum(sycl::half4 v) {
  sycl::half2 c = add(sycl::half2{v.x(), v.y()}, sycl::half2{v.z(), v.w()});
  return sum(c);
}

template <>
inline float sum(sycl::half8 v) {
  return add(
      sum(sycl::half4{v.s0(), v.s1(), v.s2(), v.s3()}),
      sum(sycl::half4{v.s4(), v.s5(), v.s6(), v.s7()}));
}

inline void from_float(sycl::half& dst, float src) {
  dst = sycl::half(src);
}

inline void from_float(sycl::half2& dst, sycl::float2 src) {
  dst = float2_to_half2(src);
}

inline void from_float(sycl::half4& dst, Float4_ src) {
  sycl::half2 h0 = float2_to_half2(src.x);
  sycl::half2 h1 = float2_to_half2(src.y);
  dst.x() = h0.x();
  dst.y() = h0.y();
  dst.z() = h1.x();
  dst.w() = h1.y();
}

inline void from_float(sycl::half8& dst, Float8_ src) {
  dst.s0() = float2_to_half2(src.x).x();
  dst.s1() = float2_to_half2(src.x).y();
  dst.s2() = float2_to_half2(src.y).x();
  dst.s3() = float2_to_half2(src.y).y();
  dst.s4() = float2_to_half2(src.z).x();
  dst.s5() = float2_to_half2(src.z).y();
  dst.s6() = float2_to_half2(src.w).x();
  dst.s7() = float2_to_half2(src.w).y();
}

// From float16 to float32.
inline float to_float(sycl::half u) {
  return half_to_float(u);
}

inline sycl::float2 to_float(sycl::half2 u) {
  return half2_to_float2(u);
}

inline Float4_ to_float(sycl::half4 u) {
  Float4_ tmp;
  tmp.x = half2_to_float2(sycl::half2{u.x(), u.y()});
  tmp.y = half2_to_float2(sycl::half2{u.z(), u.w()});
  return tmp;
}

inline Float8_ to_float(sycl::half8 u) {
  Float8_ tmp;
  tmp.x = half2_to_float2(sycl::half2{u.s0(), u.s1()});
  tmp.y = half2_to_float2(sycl::half2{u.s2(), u.s3()});
  tmp.z = half2_to_float2(sycl::half2{u.s4(), u.s5()});
  tmp.w = half2_to_float2(sycl::half2{u.s6(), u.s7()});
  return tmp;
}

// Zero-out a variable.
inline void zero(sycl::half& dst) {
  dst = sycl::half(0);
}

} // namespace vllm