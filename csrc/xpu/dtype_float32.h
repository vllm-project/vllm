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
#include "attention_generic.h"

#include <stdint.h>

namespace vllm {

// Define custom FP32 vector data types.
struct Float4_ {
  sycl::float2 x;
  sycl::float2 y;
};

struct Float8_ {
  sycl::float2 x;
  sycl::float2 y;
  sycl::float2 z;
  sycl::float2 w;
};

// FP32 vector types for Q, K, V.
template<>
struct Vec<float, 1> {
  using Type = float;
};
template<>
struct Vec<float, 2> {
  using Type = sycl::float2;
};
template<>
struct Vec<float, 4> {
  using Type = sycl::float4;
};

// FP32 accumulator vector types corresponding to Vec.
template<>
struct FloatVec<float> {
  using Type = float;
};
template <> struct FloatVec<sycl::float2> {
  using Type = sycl::float2;
};
template <> struct FloatVec<sycl::float4> {
  using Type = sycl::float4;
};

// Vector addition.
inline float add(float a, float b) {
  return a + b;
}

inline sycl::float2 add(sycl::float2 a, sycl::float2 b) {
  sycl::float2 c;
  c.x() = add(a.x(), b.x());
  c.y() = add(a.y(), b.y());
  return c;
}

inline sycl::float4 add(sycl::float4 a, sycl::float4 b) {
  sycl::float4 c;
  c.x() = add(a.x(), b.x());
  c.y() = add(a.y(), b.y());
  c.z() = add(a.z(), b.z());
  c.w() = add(a.w(), b.w());
  return c;
}

// Vector multiplication.
template<>
inline float mul<float, float>(float a, float b) {
  return a * b;
}

template <> inline sycl::float2 mul(sycl::float2 a, sycl::float2 b) {
  sycl::float2 c;
  c.x() = a.x() * b.x();
  c.y() = a.y() * b.y();
  return c;
}

template <> inline sycl::float2 mul(float a, sycl::float2 b) {
  sycl::float2 c;
  c.x() = a * b.x();
  c.y() = a * b.y();
  return c;
}

template <> inline sycl::float4 mul(sycl::float4 a, sycl::float4 b) {
  sycl::float4 c;
  c.x() = a.x() * b.x();
  c.y() = a.y() * b.y();
  c.z() = a.z() * b.z();
  c.w() = a.w() * b.w();
  return c;
}

template <> inline sycl::float4 mul(float a, sycl::float4 b) {
  sycl::float4 c;
  c.x() = a * b.x();
  c.y() = a * b.y();
  c.z() = a * b.z();
  c.w() = a * b.w();
  return c;
}

// Vector fused multiply-add.
inline float fma(float a, float b, float c) {
  return a * b + c;
}

inline sycl::float2 fma(sycl::float2 a, sycl::float2 b, sycl::float2 c) {
  sycl::float2 d;
  d.x() = fma(a.x(), b.x(), c.x());
  d.y() = fma(a.y(), b.y(), c.y());
  return d;
}

inline sycl::float2 fma(float a, sycl::float2 b, sycl::float2 c) {
  sycl::float2 d;
  d.x() = fma(a, b.x(), c.x());
  d.y() = fma(a, b.y(), c.y());
  return d;
}

inline sycl::float4 fma(sycl::float4 a, sycl::float4 b, sycl::float4 c) {
  sycl::float4 d;
  d.x() = fma(a.x(), b.x(), c.x());
  d.y() = fma(a.y(), b.y(), c.y());
  d.z() = fma(a.z(), b.z(), c.z());
  d.w() = fma(a.w(), b.w(), c.w());
  return d;
}

inline sycl::float4 fma(float a, sycl::float4 b, sycl::float4 c) {
  sycl::float4 d;
  d.x() = fma(a, b.x(), c.x());
  d.y() = fma(a, b.y(), c.y());
  d.z() = fma(a, b.z(), c.z());
  d.w() = fma(a, b.w(), c.w());
  return d;
}

inline Float4_ fma(float a, Float4_ b, Float4_ c) {
  Float4_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

inline Float8_ fma(float a, Float8_ b, Float8_ c) {
  Float8_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

// Vector sum.
template<>
inline float sum(float v) {
  return v;
}

template <> inline float sum(sycl::float2 v) {
  return v.x() + v.y();
}

template <> inline float sum(sycl::float4 v) {
  return v.x() + v.y() + v.z() + v.w();
}

template<>
inline float sum(Float4_ v) {
  return v.x.x() + v.x.y() + v.y.x() + v.y.y();
}

template<>
inline float sum(Float8_ v) {
  return v.x.x() + v.x.y() + v.y.x() + v.y.y() + v.z.x() + v.z.y() + v.w.x() +
         v.w.y();
}

// Vector dot product.
inline float dot(float a, float b) {
  return a * b;
}

inline float dot(sycl::float2 a, sycl::float2 b) {
  sycl::float2 c = mul<sycl::float2, sycl::float2, sycl::float2>(a, b);
  return c.x() + c.y();
}

inline float dot(Float4_ a, Float4_ b) {
  sycl::float2 acc = mul<sycl::float2, sycl::float2, sycl::float2>(a.x, b.x);
  acc = fma(a.y, b.y, acc);
  return acc.x() + acc.y();
}

inline float dot(Float8_ a, Float8_ b) {
  sycl::float2 acc = mul<sycl::float2, sycl::float2, sycl::float2>(a.x, b.x);
  acc = fma(a.y, b.y, acc);
  acc = fma(a.z, b.z, acc);
  acc = fma(a.w, b.w, acc);
  return acc.x() + acc.y();
}

// From float to float.
inline void from_float(float& dst, float src) {
  dst = src;
}

inline void from_float(sycl::float2 &dst, sycl::float2 src) {
  dst = src;
}

inline void from_float(sycl::float4 &dst, sycl::float4 src) {
  dst = src;
}

// From float to float.
inline float to_float(float u) {
  return u;
}

inline sycl::float2 to_float(sycl::float2 u) {
  return u;
}

inline sycl::float4 to_float(sycl::float4 u) {
  return u;
}

inline Float4_ to_float(Float4_ u) {
  return u;
}

inline Float8_ to_float(Float8_ u) {
  return u;
}

// Zero-out a variable.
inline void zero(float& dst) {
  dst = 0.f;
}

} // namespace vllm