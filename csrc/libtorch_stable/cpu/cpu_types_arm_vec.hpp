// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

// Local NEON stand-in for the ATen Vectorized<> API used by cpu_types_arm.hpp.
// Header-only: no ATen/cpu/vec includes.

#include <cmath>
#include <cstring>
#include <tuple>

#include <arm_neon.h>

#ifdef ARM_BF16_SUPPORT
  #include <arm_bf16.h>
#endif

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

namespace vec_op::arm_vec {

#ifdef ARM_BF16_SUPPORT
using neon_bfloat16x8_t = bfloat16x8_t;
using neon_bfloat16x4_t = bfloat16x4_t;
#else
using neon_bfloat16x8_t = uint16x8_t;
using neon_bfloat16x4_t = uint16x4_t;
#endif

template <typename T>
struct Vectorized;

namespace {

template <typename T>
inline void copy_partial(T* dst, const T* src, int count) {
  std::memcpy(dst, src, static_cast<size_t>(count) * sizeof(T));
}

inline float32x4_t bf16x4_to_f32(neon_bfloat16x4_t v) {
#ifdef ARM_BF16_SUPPORT
  return vcvt_f32_bf16(v);
#else
  return vreinterpretq_f32_u32(vshll_n_u16(v, 16));
#endif
}

inline neon_bfloat16x4_t f32_to_bf16x4(float32x4_t v) {
#ifdef ARM_BF16_SUPPORT
  return vcvt_bf16_f32(v);
#else
  return vshrn_n_u32(vreinterpretq_u32_f32(v), 16);
#endif
}

inline neon_bfloat16x8_t load_bf16x8(const c10::BFloat16* ptr) {
#ifdef ARM_BF16_SUPPORT
  return vld1q_bf16(reinterpret_cast<const bfloat16_t*>(ptr));
#else
  return vld1q_u16(reinterpret_cast<const uint16_t*>(ptr));
#endif
}

inline void store_bf16x8(c10::BFloat16* ptr, neon_bfloat16x8_t v) {
#ifdef ARM_BF16_SUPPORT
  vst1q_bf16(reinterpret_cast<bfloat16_t*>(ptr), v);
#else
  vst1q_u16(reinterpret_cast<uint16_t*>(ptr), v);
#endif
}

inline neon_bfloat16x4_t bf16_low(neon_bfloat16x8_t v) {
#ifdef ARM_BF16_SUPPORT
  return vget_low_bf16(v);
#else
  return vget_low_u16(v);
#endif
}

inline neon_bfloat16x4_t bf16_high(neon_bfloat16x8_t v) {
#ifdef ARM_BF16_SUPPORT
  return vget_high_bf16(v);
#else
  return vget_high_u16(v);
#endif
}

inline neon_bfloat16x8_t bf16_combine(neon_bfloat16x4_t lo,
                                      neon_bfloat16x4_t hi) {
#ifdef ARM_BF16_SUPPORT
  return vcombine_bf16(lo, hi);
#else
  return vcombine_u16(lo, hi);
#endif
}

inline neon_bfloat16x8_t bf16_dup(c10::BFloat16 v) {
  uint16_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
#ifdef ARM_BF16_SUPPORT
  bfloat16_t h;
  std::memcpy(&h, &bits, sizeof(h));
  return vdupq_n_bf16(h);
#else
  return vdupq_n_u16(bits);
#endif
}

}  // namespace

template <>
struct Vectorized<float> {
  float32x4_t values;
  static constexpr int size_ = 4;

  Vectorized() : values(vdupq_n_f32(0.f)) {}
  Vectorized(float32x4_t v) : values(v) {}
  explicit Vectorized(float v) : values(vdupq_n_f32(v)) {}

  static constexpr int size() { return size_; }

  operator float32x4_t() const { return values; }

  static Vectorized loadu(const float* ptr) { return vld1q_f32(ptr); }
  static Vectorized loadu(const float* ptr, int count) {
    alignas(16) float buf[size_]{};
    copy_partial(buf, ptr, count);
    return loadu(buf);
  }

  void store(float* ptr) const { vst1q_f32(ptr, values); }
  void store(float* ptr, int count) const {
    alignas(16) float buf[size_];
    store(buf);
    copy_partial(ptr, buf, count);
  }

  Vectorized neg() const { return vnegq_f32(values); }
  Vectorized abs() const { return vabsq_f32(values); }
  Vectorized sqrt() const { return vsqrtq_f32(values); }
  Vectorized tanh() const { return vec_op::fast_tanhf_f32x4(values); }

  template <typename Fn>
  Vectorized map_scalar(Fn fn) const {
    alignas(16) float buf[size_];
    store(buf);
    for (int i = 0; i < size_; ++i) {
      buf[i] = fn(buf[i]);
    }
    return loadu(buf);
  }

  Vectorized erf() const {
    return map_scalar([](float x) { return std::erf(x); });
  }
  Vectorized fexp_u20() const {
    return map_scalar([](float x) { return std::exp(x); });
  }
  Vectorized exp_u20() const { return fexp_u20(); }
};

inline Vectorized<float> operator+(const Vectorized<float>& a,
                                   const Vectorized<float>& b) {
  return vaddq_f32(a, b);
}
inline Vectorized<float> operator-(const Vectorized<float>& a,
                                   const Vectorized<float>& b) {
  return vsubq_f32(a, b);
}
inline Vectorized<float> operator*(const Vectorized<float>& a,
                                   const Vectorized<float>& b) {
  return vmulq_f32(a, b);
}
inline Vectorized<float> operator/(const Vectorized<float>& a,
                                   const Vectorized<float>& b) {
  return vdivq_f32(a, b);
}

inline Vectorized<float> maximum(const Vectorized<float>& a,
                                 const Vectorized<float>& b) {
  return vmaxq_f32(a, b);
}
inline Vectorized<float> minimum(const Vectorized<float>& a,
                                 const Vectorized<float>& b) {
  return vminq_f32(a, b);
}
inline Vectorized<float> clamp(const Vectorized<float>& v,
                               const Vectorized<float>& min_v,
                               const Vectorized<float>& max_v) {
  return vminq_f32(vmaxq_f32(v, min_v), max_v);
}
inline Vectorized<float> fmadd(const Vectorized<float>& a,
                               const Vectorized<float>& b,
                               const Vectorized<float>& c) {
  return vfmaq_f32(c, a, b);
}

template <typename AccT>
inline AccT vec_reduce_add(const Vectorized<float>& v) {
  return static_cast<AccT>(vaddvq_f32(v));
}

template <>
struct Vectorized<c10::Half> {
  float16x8_t values;
  static constexpr int size_ = 8;

  Vectorized() : values(vdupq_n_f16(static_cast<float16_t>(0))) {}
  Vectorized(float16x8_t v) : values(v) {}
  explicit Vectorized(c10::Half v) {
    values = vdupq_n_f16(static_cast<float16_t>(static_cast<float>(v)));
  }

  static constexpr int size() { return size_; }

  operator float16x8_t() const { return values; }

  static Vectorized loadu(const c10::Half* ptr) {
    return vld1q_f16(reinterpret_cast<const float16_t*>(ptr));
  }
  static Vectorized loadu(const c10::Half* ptr, int count) {
    alignas(16) c10::Half buf[size_]{};
    copy_partial(buf, ptr, count);
    return loadu(buf);
  }

  void store(c10::Half* ptr) const {
    vst1q_f16(reinterpret_cast<float16_t*>(ptr), values);
  }
  void store(c10::Half* ptr, int count) const {
    alignas(16) c10::Half buf[size_];
    store(buf);
    copy_partial(ptr, buf, count);
  }

  template <typename Fn>
  Vectorized map_via_float(Fn fn) const {
    float32x4_t lo = vcvt_f32_f16(vget_low_f16(values));
    float32x4_t hi = vcvt_f32_f16(vget_high_f16(values));
    alignas(16) float buf[8];
    vst1q_f32(buf, lo);
    vst1q_f32(buf + 4, hi);
    for (int i = 0; i < 8; ++i) {
      buf[i] = fn(buf[i]);
    }
    lo = vld1q_f32(buf);
    hi = vld1q_f32(buf + 4);
    return vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
  }

  Vectorized abs() const {
    return map_via_float([](float x) { return std::fabs(x); });
  }
  Vectorized sqrt() const {
    return map_via_float([](float x) { return std::sqrt(x); });
  }
};

inline Vectorized<c10::Half> convert_float_half(const Vectorized<float>& a,
                                                const Vectorized<float>& b) {
  return vcombine_f16(vcvt_f16_f32(a), vcvt_f16_f32(b));
}

template <>
struct Vectorized<c10::BFloat16> {
  neon_bfloat16x8_t values;
  static constexpr int size_ = 8;

  Vectorized() : values{} {}
  Vectorized(neon_bfloat16x8_t v) : values(v) {}
  explicit Vectorized(c10::BFloat16 v) : values(bf16_dup(v)) {}

  static constexpr int size() { return size_; }

  operator neon_bfloat16x8_t() const { return values; }

  static Vectorized loadu(const c10::BFloat16* ptr) { return load_bf16x8(ptr); }
  static Vectorized loadu(const c10::BFloat16* ptr, int count) {
    alignas(16) c10::BFloat16 buf[size_]{};
    copy_partial(buf, ptr, count);
    return loadu(buf);
  }

  void store(c10::BFloat16* ptr) const { store_bf16x8(ptr, values); }
  void store(c10::BFloat16* ptr, int count) const {
    alignas(16) c10::BFloat16 buf[size_];
    store(buf);
    copy_partial(ptr, buf, count);
  }
};

inline std::tuple<Vectorized<float>, Vectorized<float>> convert_bfloat16_float(
    const Vectorized<c10::BFloat16>& a) {
  neon_bfloat16x8_t x = a;
  return {Vectorized<float>(bf16x4_to_f32(bf16_low(x))),
          Vectorized<float>(bf16x4_to_f32(bf16_high(x)))};
}

inline Vectorized<c10::BFloat16> convert_float_bfloat16(
    const Vectorized<float>& a, const Vectorized<float>& b) {
  return bf16_combine(f32_to_bf16x4(a), f32_to_bf16x4(b));
}

}  // namespace vec_op::arm_vec
