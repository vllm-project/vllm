// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <arm_sve.h>

#include "cpu_erf_sve.hpp"
#include "cpu_tanhf_sve.hpp"
#include "cpu_types_arm.hpp"

using namespace at::vec;

namespace vec_op {

static_assert(Vectorized<float>::size() == 4);
static_assert(Vectorized<c10::Half>::size() == 8);
static_assert(Vectorized<c10::BFloat16>::size() == 8);
static_assert(Vectorized<int8_t>::size() == 16);
static_assert(Vectorized<int32_t>::size() == 4);

namespace {

FORCE_INLINE svfloat16_t load_half(const Vectorized<c10::Half>& value) {
  alignas(16) c10::Half data[Vectorized<c10::Half>::size()];
  value.store(data);
  return svld1_f16(svptrue_b16(), reinterpret_cast<const float16_t*>(data));
}

FORCE_INLINE Vectorized<c10::Half> store_half(const svfloat16_t value) {
  alignas(16) c10::Half data[Vectorized<c10::Half>::size()];
  svst1_f16(svptrue_b16(), reinterpret_cast<float16_t*>(data), value);
  return Vectorized<c10::Half>::loadu(data);
}

}  // namespace

struct FP16Vec8 : public VectorizedRegWrapper<FP16Vec8, 1, c10::Half> {
  using Base = VectorizedRegWrapper<FP16Vec8, 1, c10::Half>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  explicit FP16Vec8(const FP32Vec8&);
};

struct FP16Vec16 : public VectorizedRegWrapper<FP16Vec16, 2, c10::Half> {
  using Base = VectorizedRegWrapper<FP16Vec16, 2, c10::Half>;
  using VectorizedT = typename Base::VectorizedT;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  explicit FP16Vec16(bool, const void* ptr) {
    constexpr int lanes = VectorizedT::size();
    const svbool_t pg = svptrue_b16();
    const auto* source = reinterpret_cast<const float16_t*>(ptr);
    for (int i = 0; i < Base::VEC_REG_NUM; ++i) {
      reg.val[i] = store_half(svldnt1_f16(pg, source + i * lanes));
    }
  }

  explicit FP16Vec16(const FP32Vec16& vec);
};

struct BF16Vec8 : public VectorizedRegWrapper<BF16Vec8, 1, c10::BFloat16> {
  using Base = VectorizedRegWrapper<BF16Vec8, 1, c10::BFloat16>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  explicit BF16Vec8(const FP32Vec8&);
};

struct BF16Vec16 : public VectorizedRegWrapper<BF16Vec16, 2, c10::BFloat16> {
  using Base = VectorizedRegWrapper<BF16Vec16, 2, c10::BFloat16>;
  using VectorizedT = typename Base::VectorizedT;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  explicit BF16Vec16(bool, const void* ptr) {
    constexpr int lanes = VectorizedT::size();
    const svbool_t pg = svptrue_b16();
    const auto* source = reinterpret_cast<const bfloat16_t*>(ptr);
    reg.val[0] = VectorizedT(svldnt1_bf16(pg, source));
    reg.val[1] = VectorizedT(svldnt1_bf16(pg, source + lanes));
  }

  explicit BF16Vec16(const FP32Vec16&);
};

struct BF16Vec32 : public VectorizedRegWrapper<BF16Vec32, 4, c10::BFloat16> {
  using Base = VectorizedRegWrapper<BF16Vec32, 4, c10::BFloat16>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  explicit BF16Vec32(const BF16Vec8& vec8_data) {
    reg.val[0] = vec8_data.reg.val[0];
    reg.val[1] = vec8_data.reg.val[0];
    reg.val[2] = vec8_data.reg.val[0];
    reg.val[3] = vec8_data.reg.val[0];
  }

  explicit BF16Vec32(const uint8_t*, fp8_e4m3_tag) : Base() {}
  explicit BF16Vec32(const uint8_t*, fp8_e5m2_tag) : Base() {}
};

struct FP32Vec4 : public VectorizedRegWrapper<FP32Vec4, 1, float> {
  using Base = VectorizedRegWrapper<FP32Vec4, 1, float>;
  using VectorizedT = typename Base::VectorizedT;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  FORCE_INLINE FP32Vec4 er() const {
    FP32Vec4 result(uninit);
    result.reg.val[0] = VectorizedT(
        fast_erff_f32xn(static_cast<svfloat32_t>(reg.val[0]), svptrue_b32()));
    return result;
  }

  FORCE_INLINE FP32Vec4 tanh() const {
    FP32Vec4 result(uninit);
    result.reg.val[0] = VectorizedT(
        fast_tanhf_f32x4(static_cast<svfloat32_t>(reg.val[0]), svptrue_b32()));
    return result;
  }
};

struct FP32Vec8 : public VectorizedRegWrapper<FP32Vec8, 2, float> {
  using Base = VectorizedRegWrapper<FP32Vec8, 2, float>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;
  using Base::VEC_REG_NUM;

  using VectorizedT = typename Base::VectorizedT;
  using Vectorized2x4f = typename Base::NxVectorizedTArray;

  FP32Vec8() : Base() {}
  FP32Vec8(const FP32Vec8& data) : Base(data) {}

  explicit FP32Vec8(float v) : Base(v) {}
  explicit FP32Vec8(const float* ptr)
      : Base(reinterpret_cast<const void*>(ptr)) {}
  explicit FP32Vec8(const float* ptr, const int elem_num)
      : Base(reinterpret_cast<const void*>(ptr), elem_num) {}

  explicit FP32Vec8(const Vectorized2x4f& data) : Base(data) {}

  explicit FP32Vec8(const BF16Vec8& v) {
    std::tie(reg.val[0], reg.val[1]) = convert_bfloat16_float(v.reg.val[0]);
  }

  explicit FP32Vec8(const FP16Vec8& v) {
    std::tie(reg.val[0], reg.val[1]) = convert_half_float(v.reg.val[0]);
  }

  FORCE_INLINE FP32Vec8 er() const {
    FP32Vec8 result(uninit);
    result.reg.val[0] = VectorizedT(
        fast_erff_f32xn(static_cast<svfloat32_t>(reg.val[0]), svptrue_b32()));
    result.reg.val[1] = VectorizedT(
        fast_erff_f32xn(static_cast<svfloat32_t>(reg.val[1]), svptrue_b32()));

    return result;
  }

  FORCE_INLINE FP32Vec8 tanh() const {
    FP32Vec8 result(uninit);
    result.reg.val[0] = VectorizedT(
        fast_tanhf_f32x4(static_cast<svfloat32_t>(reg.val[0]), svptrue_b32()));
    result.reg.val[1] = VectorizedT(
        fast_tanhf_f32x4(static_cast<svfloat32_t>(reg.val[1]), svptrue_b32()));

    return result;
  }

  FORCE_INLINE float reduce_sum() const noexcept {
    std::plus<VectorizedT> add;
    float answer =
        vec_reduce_all<float, std::plus<VectorizedT>>(add, reg.val[0]);
    answer += vec_reduce_all<float, std::plus<VectorizedT>>(add, reg.val[1]);
    return answer;
  }

  FORCE_INLINE FP32Vec8 operator+(const FP32Vec8& b) const noexcept {
    FP32Vec8 result(uninit);
    result.reg.val[0] = reg.val[0] + b.reg.val[0];
    result.reg.val[1] = reg.val[1] + b.reg.val[1];
    return result;
  }

  FORCE_INLINE FP32Vec8 operator-(const FP32Vec8& b) const noexcept {
    FP32Vec8 result(uninit);
    result.reg.val[0] = reg.val[0] - b.reg.val[0];
    result.reg.val[1] = reg.val[1] - b.reg.val[1];
    return result;
  }

  FORCE_INLINE FP32Vec8 operator*(const FP32Vec8& b) const noexcept {
    FP32Vec8 result(uninit);
    result.reg.val[0] = reg.val[0] * b.reg.val[0];
    result.reg.val[1] = reg.val[1] * b.reg.val[1];
    return result;
  }

  FORCE_INLINE FP32Vec8 operator/(const FP32Vec8& b) const noexcept {
    FP32Vec8 result(uninit);
    result.reg.val[0] = reg.val[0] / b.reg.val[0];
    result.reg.val[1] = reg.val[1] / b.reg.val[1];
    return result;
  }
};

struct FP32Vec16 : public VectorizedRegWrapper<FP32Vec16, 4, float> {
  using Base = VectorizedRegWrapper<FP32Vec16, 4, float>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;
  using Base::VEC_REG_NUM;

  using ScalarT = typename Base::ScalarT;
  using VectorizedT = typename Base::VectorizedT;
  using Vectorized4x4f = typename Base::NxVectorizedTArray;

  FP32Vec16() : Base() {}
  FP32Vec16(const FP32Vec16& data) : Base(data) {}
  FORCE_INLINE explicit FP32Vec16(float v) : Base(v) {}
  explicit FP32Vec16(const float* ptr)
      : Base(reinterpret_cast<const void*>(ptr)) {}
  explicit FP32Vec16(const float* ptr, const int elem_num)
      : Base(reinterpret_cast<const void*>(ptr), elem_num) {}
  explicit FP32Vec16(const Vectorized4x4f& data) : Base(data) {}

  explicit FP32Vec16(bool, const float* ptr) {
    constexpr int lanes = VectorizedT::size();
    const svbool_t pg = svptrue_b32();
    reg.val[0] = VectorizedT(svldnt1_f32(pg, ptr));
    reg.val[1] = VectorizedT(svldnt1_f32(pg, ptr + lanes));
    reg.val[2] = VectorizedT(svldnt1_f32(pg, ptr + 2 * lanes));
    reg.val[3] = VectorizedT(svldnt1_f32(pg, ptr + 3 * lanes));
  }

  explicit FP32Vec16(const FP32Vec4& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[0];
    reg.val[2] = data.reg.val[0];
    reg.val[3] = data.reg.val[0];
  }

  explicit FP32Vec16(const FP32Vec8& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[1];
    reg.val[2] = data.reg.val[0];
    reg.val[3] = data.reg.val[1];
  }

  explicit FP32Vec16(const BF16Vec16& v) {
    std::tie(reg.val[0], reg.val[1]) = convert_bfloat16_float(v.reg.val[0]);
    std::tie(reg.val[2], reg.val[3]) = convert_bfloat16_float(v.reg.val[1]);
  }

  explicit FP32Vec16(const BF16Vec8& v) : FP32Vec16(FP32Vec8(v)) {}

  explicit FP32Vec16(const BF16Vec32&, int) : Base() {}

  explicit FP32Vec16(const FP16Vec16& v) {
    std::tie(reg.val[0], reg.val[1]) = convert_half_float(v.reg.val[0]);
    std::tie(reg.val[2], reg.val[3]) = convert_half_float(v.reg.val[1]);
  }

  FORCE_INLINE FP32Vec16 er() const {
    FP32Vec16 result(uninit);
    result.reg.val[0] = VectorizedT(
        fast_erff_f32xn(static_cast<svfloat32_t>(reg.val[0]), svptrue_b32()));
    result.reg.val[1] = VectorizedT(
        fast_erff_f32xn(static_cast<svfloat32_t>(reg.val[1]), svptrue_b32()));
    result.reg.val[2] = VectorizedT(
        fast_erff_f32xn(static_cast<svfloat32_t>(reg.val[2]), svptrue_b32()));
    result.reg.val[3] = VectorizedT(
        fast_erff_f32xn(static_cast<svfloat32_t>(reg.val[3]), svptrue_b32()));
    return result;
  }

  FORCE_INLINE FP32Vec16 tanh() const {
    FP32Vec16 result(uninit);
    result.reg.val[0] = VectorizedT(
        fast_tanhf_f32x4(static_cast<svfloat32_t>(reg.val[0]), svptrue_b32()));
    result.reg.val[1] = VectorizedT(
        fast_tanhf_f32x4(static_cast<svfloat32_t>(reg.val[1]), svptrue_b32()));
    result.reg.val[2] = VectorizedT(
        fast_tanhf_f32x4(static_cast<svfloat32_t>(reg.val[2]), svptrue_b32()));
    result.reg.val[3] = VectorizedT(
        fast_tanhf_f32x4(static_cast<svfloat32_t>(reg.val[3]), svptrue_b32()));
    return result;
  }

  static FORCE_INLINE void load_even_odd(const float* ptr, FP32Vec16& even,
                                         FP32Vec16& odd) noexcept {
    const svbool_t pg = svptrue_b32();
    constexpr int lanes = VectorizedT::size();
    const svfloat32_t low0 = svld1_f32(pg, ptr);
    const svfloat32_t high0 = svld1_f32(pg, ptr + lanes);
    even.reg.val[0] = VectorizedT(svuzp1_f32(low0, high0));
    odd.reg.val[0] = VectorizedT(svuzp2_f32(low0, high0));

    const svfloat32_t low1 = svld1_f32(pg, ptr + 2 * lanes);
    const svfloat32_t high1 = svld1_f32(pg, ptr + 3 * lanes);
    even.reg.val[1] = VectorizedT(svuzp1_f32(low1, high1));
    odd.reg.val[1] = VectorizedT(svuzp2_f32(low1, high1));

    const svfloat32_t low2 = svld1_f32(pg, ptr + 4 * lanes);
    const svfloat32_t high2 = svld1_f32(pg, ptr + 5 * lanes);
    even.reg.val[2] = VectorizedT(svuzp1_f32(low2, high2));
    odd.reg.val[2] = VectorizedT(svuzp2_f32(low2, high2));

    const svfloat32_t low3 = svld1_f32(pg, ptr + 6 * lanes);
    const svfloat32_t high3 = svld1_f32(pg, ptr + 7 * lanes);
    even.reg.val[3] = VectorizedT(svuzp1_f32(low3, high3));
    odd.reg.val[3] = VectorizedT(svuzp2_f32(low3, high3));
  }

  FORCE_INLINE FP32Vec16 operator+(const FP32Vec16& b) const noexcept {
    FP32Vec16 result(uninit);
    result.reg.val[0] = reg.val[0] + b.reg.val[0];
    result.reg.val[1] = reg.val[1] + b.reg.val[1];
    result.reg.val[2] = reg.val[2] + b.reg.val[2];
    result.reg.val[3] = reg.val[3] + b.reg.val[3];
    return result;
  }

  FORCE_INLINE FP32Vec16 operator-(const FP32Vec16& b) const noexcept {
    FP32Vec16 result(uninit);
    result.reg.val[0] = reg.val[0] - b.reg.val[0];
    result.reg.val[1] = reg.val[1] - b.reg.val[1];
    result.reg.val[2] = reg.val[2] - b.reg.val[2];
    result.reg.val[3] = reg.val[3] - b.reg.val[3];
    return result;
  }

  FORCE_INLINE FP32Vec16 operator-() const noexcept {
    FP32Vec16 result(uninit);
    result.reg.val[0] = reg.val[0].neg();
    result.reg.val[1] = reg.val[1].neg();
    result.reg.val[2] = reg.val[2].neg();
    result.reg.val[3] = reg.val[3].neg();
    return result;
  }

  FORCE_INLINE FP32Vec16 operator*(const FP32Vec16& b) const noexcept {
    FP32Vec16 result(uninit);
    result.reg.val[0] = reg.val[0] * b.reg.val[0];
    result.reg.val[1] = reg.val[1] * b.reg.val[1];
    result.reg.val[2] = reg.val[2] * b.reg.val[2];
    result.reg.val[3] = reg.val[3] * b.reg.val[3];
    return result;
  }

  FORCE_INLINE FP32Vec16 operator/(const FP32Vec16& b) const noexcept {
    FP32Vec16 result(uninit);
    result.reg.val[0] = reg.val[0] / b.reg.val[0];
    result.reg.val[1] = reg.val[1] / b.reg.val[1];
    result.reg.val[2] = reg.val[2] / b.reg.val[2];
    result.reg.val[3] = reg.val[3] / b.reg.val[3];
    return result;
  }

  FORCE_INLINE FP32Vec16 clamp(const FP32Vec16& min,
                               const FP32Vec16& max) const {
    FP32Vec16 result(uninit);
    result.reg.val[0] =
        at::vec::clamp(reg.val[0], min.reg.val[0], max.reg.val[0]);
    result.reg.val[1] =
        at::vec::clamp(reg.val[1], min.reg.val[1], max.reg.val[1]);
    result.reg.val[2] =
        at::vec::clamp(reg.val[2], min.reg.val[2], max.reg.val[2]);
    result.reg.val[3] =
        at::vec::clamp(reg.val[3], min.reg.val[3], max.reg.val[3]);
    return result;
  }

  FORCE_INLINE FP32Vec16 min(const FP32Vec16& b) const {
    FP32Vec16 result(uninit);
    result.reg.val[0] = minimum(reg.val[0], b.reg.val[0]);
    result.reg.val[1] = minimum(reg.val[1], b.reg.val[1]);
    result.reg.val[2] = minimum(reg.val[2], b.reg.val[2]);
    result.reg.val[3] = minimum(reg.val[3], b.reg.val[3]);
    return result;
  }

  FORCE_INLINE FP32Vec16 max(const FP32Vec16& b) const {
    FP32Vec16 result(uninit);
    result.reg.val[0] = maximum(reg.val[0], b.reg.val[0]);
    result.reg.val[1] = maximum(reg.val[1], b.reg.val[1]);
    result.reg.val[2] = maximum(reg.val[2], b.reg.val[2]);
    result.reg.val[3] = maximum(reg.val[3], b.reg.val[3]);
    return result;
  }

  FP32Vec16 min(const FP32Vec16& b, const int elem_num) const {
    TORCH_INTERNAL_ASSERT(elem_num >= 0 && elem_num <= VEC_ELEM_NUM);
    if (elem_num == VEC_ELEM_NUM) {
      return min(b);
    }

    constexpr int lanes = VectorizedT::size();
    const int full = elem_num / lanes;
    const int remainder = elem_num % lanes;
    FP32Vec16 result(*this);
    for (int i = 0; i < full; ++i) {
      result.reg.val[i] = minimum(reg.val[i], b.reg.val[i]);
    }
    if (remainder > 0) {
      const VectorizedT values = minimum(reg.val[full], b.reg.val[full]);
      result.reg.val[full] = VectorizedT::set(reg.val[full], values, remainder);
    }
    return result;
  }

  FP32Vec16 max(const FP32Vec16& b, const int elem_num) const {
    TORCH_INTERNAL_ASSERT(elem_num >= 0 && elem_num <= VEC_ELEM_NUM);
    if (elem_num == VEC_ELEM_NUM) {
      return max(b);
    }

    constexpr int lanes = VectorizedT::size();
    const int full = elem_num / lanes;
    const int remainder = elem_num % lanes;
    FP32Vec16 result(*this);
    for (int i = 0; i < full; ++i) {
      result.reg.val[i] = maximum(reg.val[i], b.reg.val[i]);
    }
    if (remainder > 0) {
      const VectorizedT values = maximum(reg.val[full], b.reg.val[full]);
      result.reg.val[full] = VectorizedT::set(reg.val[full], values, remainder);
    }
    return result;
  }

  float reduce_max() const {
    VectorizedT value = reg.val[0];
    for (int i = 1; i < VEC_REG_NUM; ++i) {
      value = maximum(value, reg.val[i]);
    }
    return svmaxv_f32(svptrue_b32(), value);
  }

  float reduce_min() const {
    VectorizedT value = reg.val[0];
    for (int i = 1; i < VEC_REG_NUM; ++i) {
      value = minimum(value, reg.val[i]);
    }
    return svminv_f32(svptrue_b32(), value);
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);
    TORCH_INTERNAL_ASSERT(idx >= 0 && idx < VEC_ELEM_NUM / group_size);

    alignas(16) float values[VEC_ELEM_NUM];
    reg.save(values);
    float answer = 0;
    const int start = idx * group_size;
    for (int i = 0; i < group_size; ++i) {
      answer += values[start + i];
    }
    return answer;
  }

  float reduce_sum() const {
    std::plus<VectorizedT> add;
    float answer = vec_reduce_all<float>(add, reg.val[0]);
    answer += vec_reduce_all<float>(add, reg.val[1]);
    answer += vec_reduce_all<float>(add, reg.val[2]);
    answer += vec_reduce_all<float>(add, reg.val[3]);
    return answer;
  }
};

struct INT8Vec16 : public Vec<INT8Vec16> {
  using VectorizedT = Vectorized<int8_t>;
  using Reg = NxVectorizedTVecReg<1, int8_t>;
  constexpr static int VEC_ELEM_NUM = 16;

  Reg reg;

  explicit INT8Vec16(const FP32Vec16& vec) {
    const svbool_t pg = svptrue_b32();
    alignas(16) int32_t values32[VEC_ELEM_NUM];
    alignas(16) int8_t values8[VEC_ELEM_NUM];

    svfloat32_t rounded0 = svrintn_f32_x(pg, vec.reg.val[0]);
    svint32_t value0 = svcvt_s32_f32_z(pg, rounded0);
    value0 = svmax_n_s32_x(pg, value0, -128);
    value0 = svmin_n_s32_x(pg, value0, 127);
    svst1_s32(pg, values32, value0);

    svfloat32_t rounded1 = svrintn_f32_x(pg, vec.reg.val[1]);
    svint32_t value1 = svcvt_s32_f32_z(pg, rounded1);
    value1 = svmax_n_s32_x(pg, value1, -128);
    value1 = svmin_n_s32_x(pg, value1, 127);
    svst1_s32(pg, values32 + Vectorized<int32_t>::size(), value1);

    svfloat32_t rounded2 = svrintn_f32_x(pg, vec.reg.val[2]);
    svint32_t value2 = svcvt_s32_f32_z(pg, rounded2);
    value2 = svmax_n_s32_x(pg, value2, -128);
    value2 = svmin_n_s32_x(pg, value2, 127);
    svst1_s32(pg, values32 + 2 * Vectorized<int32_t>::size(), value2);

    svfloat32_t rounded3 = svrintn_f32_x(pg, vec.reg.val[3]);
    svint32_t value3 = svcvt_s32_f32_z(pg, rounded3);
    value3 = svmax_n_s32_x(pg, value3, -128);
    value3 = svmin_n_s32_x(pg, value3, 127);
    svst1_s32(pg, values32 + 3 * Vectorized<int32_t>::size(), value3);

    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      values8[i] = static_cast<int8_t>(values32[i]);
    }
    reg.val[0] = VectorizedT::loadu(values8);
  }

  void save(int8_t* ptr) const { reg.val[0].store(ptr); }

  void save(int8_t* ptr, const int elem_num) const {
    TORCH_CHECK(elem_num >= 0 && elem_num <= VEC_ELEM_NUM);
    reg.val[0].store(ptr, elem_num);
  }
};

struct INT8Vec64 : public Vec<INT8Vec64> {
  using VectorizedT = Vectorized<int8_t>;
  using Reg = NxVectorizedTVecReg<4, int8_t>;
  constexpr static int VEC_ELEM_NUM = 64;

  Reg reg;

  explicit INT8Vec64(const int8_t* ptr) : reg(ptr) {}

  explicit INT8Vec64(bool, const int8_t* ptr) {
    constexpr int lanes = VectorizedT::size();
    const svbool_t pg = svptrue_b8();
    reg.val[0] = VectorizedT(svldnt1_s8(pg, ptr));
    reg.val[1] = VectorizedT(svldnt1_s8(pg, ptr + lanes));
    reg.val[2] = VectorizedT(svldnt1_s8(pg, ptr + 2 * lanes));
    reg.val[3] = VectorizedT(svldnt1_s8(pg, ptr + 3 * lanes));
  }

  void save(int8_t* ptr) const { reg.save(ptr); }

  void save(int8_t* ptr, int elem_num) const {
    TORCH_CHECK(elem_num > 0 && elem_num <= VEC_ELEM_NUM);
    reg.save(ptr, elem_num);
  }

  void nt_save(int8_t* ptr) const {
    constexpr int lanes = VectorizedT::size();
    const svbool_t pg = svptrue_b8();
    svstnt1_s8(pg, ptr, static_cast<svint8_t>(reg.val[0]));
    svstnt1_s8(pg, ptr + lanes, static_cast<svint8_t>(reg.val[1]));
    svstnt1_s8(pg, ptr + 2 * lanes, static_cast<svint8_t>(reg.val[2]));
    svstnt1_s8(pg, ptr + 3 * lanes, static_cast<svint8_t>(reg.val[3]));
  }
};

struct INT32Vec16 : public Vec<INT32Vec16> {
  using Reg = NxVectorizedTVecReg<4, int32_t>;
  constexpr static int VEC_ELEM_NUM = 16;

  Reg reg;

  explicit INT32Vec16(const void* ptr) : reg(ptr) {}

  void save(int32_t* ptr) const { reg.save(ptr); }

  void save(int32_t* ptr, const int elem_num) const {
    TORCH_CHECK(elem_num > 0 && elem_num <= VEC_ELEM_NUM);
    reg.save(ptr, elem_num);
  }
};

inline FP16Vec8::FP16Vec8(const FP32Vec8& v) {
  reg.val[0] = convert_float_half(v.reg.val[0], v.reg.val[1]);
}

inline FP16Vec16::FP16Vec16(const FP32Vec16& v) {
  reg.val[0] = convert_float_half(v.reg.val[0], v.reg.val[1]);
  reg.val[1] = convert_float_half(v.reg.val[2], v.reg.val[3]);
}

inline BF16Vec8::BF16Vec8(const FP32Vec8& v) {
  reg.val[0] = convert_float_bfloat16(v.reg.val[0], v.reg.val[1]);
}

inline BF16Vec16::BF16Vec16(const FP32Vec16& v) {
  reg.val[0] = convert_float_bfloat16(v.reg.val[0], v.reg.val[1]);
  reg.val[1] = convert_float_bfloat16(v.reg.val[2], v.reg.val[3]);
}

inline void fma(FP32Vec16& acc, FP32Vec16& a, FP32Vec16& b) {
  acc.reg.val[0] = fmadd(a.reg.val[0], b.reg.val[0], acc.reg.val[0]);
  acc.reg.val[1] = fmadd(a.reg.val[1], b.reg.val[1], acc.reg.val[1]);
  acc.reg.val[2] = fmadd(a.reg.val[2], b.reg.val[2], acc.reg.val[2]);
  acc.reg.val[3] = fmadd(a.reg.val[3], b.reg.val[3], acc.reg.val[3]);
}

inline void fma(FP32Vec16& acc, BF16Vec32& a, BF16Vec32& b) {
  using VectorizedT = FP32Vec16::VectorizedT;
  acc.reg.val[0] =
      VectorizedT(svbfdot_f32(static_cast<svfloat32_t>(acc.reg.val[0]),
                              static_cast<svbfloat16_t>(a.reg.val[0]),
                              static_cast<svbfloat16_t>(b.reg.val[0])));
  acc.reg.val[1] =
      VectorizedT(svbfdot_f32(static_cast<svfloat32_t>(acc.reg.val[1]),
                              static_cast<svbfloat16_t>(a.reg.val[1]),
                              static_cast<svbfloat16_t>(b.reg.val[1])));
  acc.reg.val[2] =
      VectorizedT(svbfdot_f32(static_cast<svfloat32_t>(acc.reg.val[2]),
                              static_cast<svbfloat16_t>(a.reg.val[2]),
                              static_cast<svbfloat16_t>(b.reg.val[2])));
  acc.reg.val[3] =
      VectorizedT(svbfdot_f32(static_cast<svfloat32_t>(acc.reg.val[3]),
                              static_cast<svbfloat16_t>(a.reg.val[3]),
                              static_cast<svbfloat16_t>(b.reg.val[3])));
}

template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
#ifdef ARM_BF16_SUPPORT
  *reinterpret_cast<__bf16*>(ptr) = vcvth_bf16_f32(v);
#else
  *ptr = static_cast<c10::BFloat16>(v);
#endif
}

}  // namespace vec_op
