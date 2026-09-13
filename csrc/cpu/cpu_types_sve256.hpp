// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cstring>
#include <arm_sve.h>

#include "cpu_erf_sve.hpp"
#include "cpu_types_arm.hpp"

using namespace at::vec;

namespace vec_op {

static_assert(Vectorized<float>::size() == 8);
static_assert(Vectorized<c10::Half>::size() == 16);
static_assert(Vectorized<c10::BFloat16>::size() == 16);
static_assert(Vectorized<int8_t>::size() == 32);
static_assert(Vectorized<int32_t>::size() == 8);

struct FP16Vec8 : public PartialVectorizedRegWrapper<FP16Vec8, c10::Half, 8> {
  using Base = PartialVectorizedRegWrapper<FP16Vec8, c10::Half, 8>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  explicit FP16Vec8(const FP32Vec8&);

  // PyTorch does not have sve256 Vectorized<Half> implementations for abs and
  // sqrt, so they fall back to std implementations. Add SVE abs and sqrt
  // overrides instead.
  FORCE_INLINE FP16Vec8 abs() const {
    const svbool_t pg = svwhilelt_b16(uint64_t{0}, uint64_t{VEC_ELEM_NUM});
    const c10::Half* source = reg.val[0];
    FP16Vec8 result(uninit);
    c10::Half* destination = result.reg.val[0];
    svst1_f16(
        pg, reinterpret_cast<float16_t*>(destination),
        svabs_f16_x(pg,
                    svld1_f16(pg, reinterpret_cast<const float16_t*>(source))));
    return result;
  }

  FORCE_INLINE FP16Vec8 sqrt() const {
    const svbool_t pg = svwhilelt_b16(uint64_t{0}, uint64_t{VEC_ELEM_NUM});
    const c10::Half* source = reg.val[0];
    FP16Vec8 result(uninit);
    c10::Half* destination = result.reg.val[0];
    svst1_f16(
        pg, reinterpret_cast<float16_t*>(destination),
        svsqrt_f16_x(
            pg, svld1_f16(pg, reinterpret_cast<const float16_t*>(source))));
    return result;
  }
};

struct FP16Vec16 : public VectorizedRegWrapper<FP16Vec16, 1, c10::Half> {
  using Base = VectorizedRegWrapper<FP16Vec16, 1, c10::Half>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  explicit FP16Vec16(bool, const void* ptr) {
    const svbool_t pg = svwhilelt_b16(uint64_t{0}, uint64_t{VEC_ELEM_NUM});

    const svfloat16_t values =
        svldnt1_f16(pg, reinterpret_cast<const float16_t*>(ptr));

    c10::Half* destination = reg.val[0];
    svst1_f16(pg, reinterpret_cast<float16_t*>(destination), values);
  }

  explicit FP16Vec16(const FP32Vec16& vec);

  FORCE_INLINE FP16Vec16 abs() const {
    const svbool_t pg = svptrue_b16();
    const c10::Half* source = reg.val[0];
    FP16Vec16 result(uninit);
    c10::Half* destination = result.reg.val[0];

    const svfloat16_t values =
        svld1_f16(pg, reinterpret_cast<const float16_t*>(source));
    svst1_f16(pg, reinterpret_cast<float16_t*>(destination),
              svabs_f16_x(pg, values));

    return result;
  }

  FORCE_INLINE FP16Vec16 sqrt() const {
    const svbool_t pg = svptrue_b16();
    const c10::Half* source = reg.val[0];
    FP16Vec16 result(uninit);
    c10::Half* destination = result.reg.val[0];

    const svfloat16_t values =
        svld1_f16(pg, reinterpret_cast<const float16_t*>(source));
    svst1_f16(pg, reinterpret_cast<float16_t*>(destination),
              svsqrt_f16_x(pg, values));

    return result;
  }

  FORCE_INLINE FP16Vec16 er() const;
  FORCE_INLINE FP16Vec16 cos() const;
  FORCE_INLINE FP16Vec16 sin() const;
  FORCE_INLINE FP16Vec16 tan() const;
  FORCE_INLINE FP16Vec16 tanh() const;
};

struct BF16Vec8
    : public PartialVectorizedRegWrapper<BF16Vec8, c10::BFloat16, 8> {
  using Base = PartialVectorizedRegWrapper<BF16Vec8, c10::BFloat16, 8>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  explicit BF16Vec8(const FP32Vec8&);
};

struct BF16Vec16 : public VectorizedRegWrapper<BF16Vec16, 1, c10::BFloat16> {
  using Base = VectorizedRegWrapper<BF16Vec16, 1, c10::BFloat16>;
  using VectorizedT = typename Base::VectorizedT;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  explicit BF16Vec16(bool, const void* ptr) {
    const svbool_t pg = svwhilelt_b16(uint64_t{0}, uint64_t{VEC_ELEM_NUM});

    reg.val[0] =
        VectorizedT(svldnt1_bf16(pg, reinterpret_cast<const bfloat16_t*>(ptr)));
  }

  explicit BF16Vec16(const FP32Vec16&);

  // SVE provides no mathematical vectorization here. It adds packing,
  // conversion, temporary-buffer and load/store overhead around the same scalar
  // libm calls. Call the pytorch implementation directly instead, as we don't
  // currently enable pytorch for bf16/fp16, and benefit from more efficient
  // widening and SVE/sleef.
  FORCE_INLINE BF16Vec16 er() const {
    BF16Vec16 result(uninit);
    result.reg.val[0] = reg.val[0].erf();
    return result;
  }

  FORCE_INLINE BF16Vec16 sin() const {
    BF16Vec16 result(uninit);
    result.reg.val[0] = reg.val[0].sin();
    return result;
  }

  FORCE_INLINE BF16Vec16 cos() const {
    BF16Vec16 result(uninit);
    result.reg.val[0] = reg.val[0].cos();
    return result;
  }

  FORCE_INLINE BF16Vec16 tan() const {
    BF16Vec16 result(uninit);
    result.reg.val[0] = reg.val[0].tan();
    return result;
  }

  FORCE_INLINE BF16Vec16 tanh() const {
    BF16Vec16 result(uninit);
    result.reg.val[0] = reg.val[0].tanh();
    return result;
  }
};

struct BF16Vec32 : public VectorizedRegWrapper<BF16Vec32, 2, c10::BFloat16> {
  using Base = VectorizedRegWrapper<BF16Vec32, 2, c10::BFloat16>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  explicit BF16Vec32(const BF16Vec8& vec8_data) {
    // We only have enough data for 8 of the 16 lanes, so we need to duplicate:
    // 2 x [1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8]
    // instead of 2 x [1 2 3 4 5 6 7 8 0 0 0 0 0 0 0 0]
    const svbool_t low_half =
        svwhilelt_b16(uint64_t{0}, uint64_t{BF16Vec8::VEC_ELEM_NUM});
    const svbfloat16_t source = vec8_data.reg.val[0];

    const Vectorized<c10::BFloat16> repeated(
        svsplice_bf16(low_half, source, source));

    reg.val[0] = repeated;
    reg.val[1] = repeated;
  }

  explicit BF16Vec32(const uint8_t*, fp8_e4m3_tag) : Base() {}
  explicit BF16Vec32(const uint8_t*, fp8_e5m2_tag) : Base() {}

  // SVE provides no mathematical vectorization here. It adds packing,
  // conversion, temporary-buffer and load/store overhead around the same scalar
  // libm calls. Call the pytorch implementation directly instead, as we don't
  // currently enable pytorch for bf16/fp16, and benefit from more efficient
  // widening and SVE/sleef.
  FORCE_INLINE BF16Vec32 er() const {
    BF16Vec32 result(uninit);
    result.reg.val[0] = reg.val[0].erf();
    result.reg.val[1] = reg.val[1].erf();
    return result;
  }

  FORCE_INLINE BF16Vec32 sin() const {
    BF16Vec32 result(uninit);
    result.reg.val[0] = reg.val[0].sin();
    result.reg.val[1] = reg.val[1].sin();
    return result;
  }

  FORCE_INLINE BF16Vec32 cos() const {
    BF16Vec32 result(uninit);
    result.reg.val[0] = reg.val[0].cos();
    result.reg.val[1] = reg.val[1].cos();
    return result;
  }

  FORCE_INLINE BF16Vec32 tan() const {
    BF16Vec32 result(uninit);
    result.reg.val[0] = reg.val[0].tan();
    result.reg.val[1] = reg.val[1].tan();
    return result;
  }

  FORCE_INLINE BF16Vec32 tanh() const {
    BF16Vec32 result(uninit);
    result.reg.val[0] = reg.val[0].tanh();
    result.reg.val[1] = reg.val[1].tanh();
    return result;
  }
};

struct FP32Vec4 : public PartialVectorizedRegWrapper<FP32Vec4, float, 4> {
  using Base = PartialVectorizedRegWrapper<FP32Vec4, float, 4>;
  using VectorizedT = typename Base::VectorizedT;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  FP32Vec4() : Base(0.0f) {}

  explicit FP32Vec4(const float* ptr)
      : Base(reinterpret_cast<const void*>(ptr)) {}

  FP32Vec4(const FP32Vec4&) = default;

  explicit FP32Vec4(float value) : Base(value) {}
};

struct FP32Vec8 : public VectorizedRegWrapper<FP32Vec8, 1, float> {
  using Base = VectorizedRegWrapper<FP32Vec8, 1, float>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;
  using Base::VEC_REG_NUM;

  using VectorizedT = typename Base::VectorizedT;
  using Vectorized8f = typename Base::NxVectorizedTArray;

  FP32Vec8() : Base() {};
  FP32Vec8(const FP32Vec8& data) : Base(data) {};

  explicit FP32Vec8(float v) : Base(v) {};
  explicit FP32Vec8(const float* ptr)
      : Base(reinterpret_cast<const void*>(ptr)) {};
  explicit FP32Vec8(const float* ptr, const int elem_num)
      : Base(reinterpret_cast<const void*>(ptr), elem_num) {};

  explicit FP32Vec8(const Vectorized8f& data) { reg.val[0] = data.val[0]; };

  explicit FP32Vec8(const BF16Vec8& v) {
    reg.val[0] = std::get<0>(convert_bfloat16_float(v.reg.val[0]));
  };
  explicit FP32Vec8(const FP16Vec8& v) {
    const c10::Half* source = v.reg.val[0];
    const svbool_t pg = svwhilelt_b16(uint64_t{0}, uint64_t{8});

    const svfloat16_t packed =
        svld1_f16(pg, reinterpret_cast<const float16_t*>(source));
    const svfloat16_t unpacked = svzip1_f16(packed, svdup_n_f16(0.0f));

    reg.val[0] = Vectorized<float>(svcvt_f32_f16_x(svptrue_b32(), unpacked));
  };

  FORCE_INLINE FP32Vec8 er() const {
    FP32Vec8 result(uninit);
    result.reg.val[0] = VectorizedT(
        fast_erff_f32xn(static_cast<svfloat32_t>(reg.val[0]), svptrue_b32()));
    return result;
  }

  FORCE_INLINE float reduce_sum() const noexcept {
    float answer = 0;
    std::plus<VectorizedT> add;

    answer =
        at::vec::vec_reduce_all<float, std::plus<VectorizedT>>(add, reg.val[0]);

    return answer;
  }

  FORCE_INLINE FP32Vec8 operator+(const FP32Vec8& b) const noexcept {
    FP32Vec8 r(uninit);
    r.reg.val[0] = reg.val[0] + b.reg.val[0];
    return r;
  }

  FORCE_INLINE FP32Vec8 operator-(const FP32Vec8& b) const noexcept {
    FP32Vec8 r(uninit);
    r.reg.val[0] = reg.val[0] - b.reg.val[0];
    return r;
  }

  FORCE_INLINE FP32Vec8 operator*(const FP32Vec8& b) const noexcept {
    FP32Vec8 r(uninit);
    r.reg.val[0] = reg.val[0] * b.reg.val[0];
    return r;
  }

  FORCE_INLINE FP32Vec8 operator/(const FP32Vec8& b) const noexcept {
    FP32Vec8 r(uninit);
    r.reg.val[0] = reg.val[0] / b.reg.val[0];
    return r;
  }
};

struct FP32Vec16 : public VectorizedRegWrapper<FP32Vec16, 2, float> {
  using Base = VectorizedRegWrapper<FP32Vec16, 2, float>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  using ScalarT = typename Base::ScalarT;
  using VectorizedT = typename Base::VectorizedT;
  using Vectorized2x8f = typename Base::NxVectorizedTArray;

  FP32Vec16() : Base() {};
  FP32Vec16(const FP32Vec16& data) : Base(data) {};
  FORCE_INLINE explicit FP32Vec16(float v) : Base(v){};
  explicit FP32Vec16(const float* ptr)
      : Base(reinterpret_cast<const void*>(ptr)) {};
  explicit FP32Vec16(const float* ptr, const int elem_num)
      : Base(reinterpret_cast<const void*>(ptr), elem_num) {};
  explicit FP32Vec16(const Vectorized2x8f& data) {
    reg.val[0] = data.val[0];
    reg.val[1] = data.val[1];
  };

  explicit FP32Vec16(bool, const float* ptr) {
    constexpr int lanes = VectorizedT::size();  // 8
    const svbool_t pg = svwhilelt_b32(uint64_t{0}, uint64_t{lanes});

    reg.val[0] = VectorizedT(svldnt1_f32(pg, ptr));
    reg.val[1] = VectorizedT(svldnt1_f32(pg, ptr + lanes));
  }

  explicit FP32Vec16(const FP32Vec4& data) {
    // We only have enough data for half the vector, so we need to duplicate:
    // 2 x [1 2 3 4 1 2 3 4] instead of 2 x [1 2 3 4 0 0 0 0]
    const svbool_t low_half =
        svwhilelt_b32(uint64_t{0}, uint64_t{FP32Vec4::VEC_ELEM_NUM});
    const svfloat32_t source = data.reg.val[0];

    const Vectorized<float> repeated(svsplice_f32(low_half, source, source));

    reg.val[0] = repeated;
    reg.val[1] = repeated;
  };

  explicit FP32Vec16(const FP32Vec8& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[0];
  };

  explicit FP32Vec16(const BF16Vec16& v) {
    std::tie(reg.val[0], reg.val[1]) = convert_bfloat16_float(v.reg.val[0]);
  };

  explicit FP32Vec16(const BF16Vec8& v) : FP32Vec16(FP32Vec8(v)) {};

  // FP8 stub: dead code on ARM (fp8 KV cache is x86-only), needed for
  // load_b_pair_vec template to compile on all platforms.
  explicit FP32Vec16(const BF16Vec32&, int) : Base() {}

  explicit FP32Vec16(const FP16Vec16& v) {
    const c10::Half* source = v.reg.val[0];
    const svfloat16_t packed =
        svld1_f16(svptrue_b16(), reinterpret_cast<const float16_t*>(source));
    const svfloat16_t zero = svdup_n_f16(0.0f);

    reg.val[0] = Vectorized<float>(
        svcvt_f32_f16_x(svptrue_b32(), svzip1_f16(packed, zero)));
    reg.val[1] = Vectorized<float>(
        svcvt_f32_f16_x(svptrue_b32(), svzip2_f16(packed, zero)));
  };

  FORCE_INLINE FP32Vec16 er() const {
    FP32Vec16 result(uninit);
    const svbool_t pg = svptrue_b32();
    result.reg.val[0] =
        VectorizedT(fast_erff_f32xn(static_cast<svfloat32_t>(reg.val[0]), pg));
    result.reg.val[1] =
        VectorizedT(fast_erff_f32xn(static_cast<svfloat32_t>(reg.val[1]), pg));
    return result;
  }

  static FORCE_INLINE void load_even_odd(const float* ptr, FP32Vec16& even,
                                         FP32Vec16& odd) noexcept {
    auto const pg = svptrue_b32();

    // Deinterleaving loads
    const svfloat32x2_t pair0 = svld2_f32(pg, ptr);
    const svfloat32x2_t pair1 = svld2_f32(pg, ptr + 16);

    even.reg.val[0] = VectorizedT(svget2_f32(pair0, 0));
    odd.reg.val[0] = VectorizedT(svget2_f32(pair0, 1));
    even.reg.val[1] = VectorizedT(svget2_f32(pair1, 0));
    odd.reg.val[1] = VectorizedT(svget2_f32(pair1, 1));
  }

  FORCE_INLINE FP32Vec16 operator+(const FP32Vec16& b) const noexcept {
    FP32Vec16 r(uninit);
    r.reg.val[0] = reg.val[0] + b.reg.val[0];
    r.reg.val[1] = reg.val[1] + b.reg.val[1];
    return r;
  }

  FORCE_INLINE FP32Vec16 operator-(const FP32Vec16& b) const noexcept {
    FP32Vec16 r(uninit);
    r.reg.val[0] = reg.val[0] - b.reg.val[0];
    r.reg.val[1] = reg.val[1] - b.reg.val[1];
    return r;
  }

  FORCE_INLINE FP32Vec16 operator-() const noexcept {
    FP32Vec16 r(uninit);
    r.reg.val[0] = reg.val[0].neg();
    r.reg.val[1] = reg.val[1].neg();
    return r;
  }

  FORCE_INLINE FP32Vec16 operator*(const FP32Vec16& b) const noexcept {
    FP32Vec16 r(uninit);
    r.reg.val[0] = reg.val[0] * b.reg.val[0];
    r.reg.val[1] = reg.val[1] * b.reg.val[1];
    return r;
  }

  FORCE_INLINE FP32Vec16 operator/(const FP32Vec16& b) const noexcept {
    FP32Vec16 r(uninit);
    r.reg.val[0] = reg.val[0] / b.reg.val[0];
    r.reg.val[1] = reg.val[1] / b.reg.val[1];
    return r;
  }

  FORCE_INLINE FP32Vec16 clamp(const FP32Vec16& min,
                               const FP32Vec16& max) const {
    FP32Vec16 r(uninit);
    r.reg.val[0] = at::vec::clamp(reg.val[0], min.reg.val[0], max.reg.val[0]);
    r.reg.val[1] = at::vec::clamp(reg.val[1], min.reg.val[1], max.reg.val[1]);
    return r;
  };

  FORCE_INLINE FP32Vec16 min(const FP32Vec16& b) const {
    FP32Vec16 r(uninit);
    r.reg.val[0] = minimum(b.reg.val[0], reg.val[0]);
    r.reg.val[1] = minimum(b.reg.val[1], reg.val[1]);
    return r;
  };

  FORCE_INLINE FP32Vec16 max(const FP32Vec16& b) const {
    FP32Vec16 r(uninit);
    r.reg.val[0] = maximum(b.reg.val[0], reg.val[0]);
    r.reg.val[1] = maximum(b.reg.val[1], reg.val[1]);
    return r;
  };

  FP32Vec16 min(const FP32Vec16& b, const int elem_num) const {
    TORCH_INTERNAL_ASSERT(elem_num >= 0 && elem_num <= VEC_ELEM_NUM);
    constexpr size_t num_elements = reg.val[0].size();

    if (elem_num == VEC_ELEM_NUM) {
      return FP32Vec16::min(b);
    }

    int full_blocks = elem_num / num_elements;
    const int remainder = elem_num % num_elements;

    FP32Vec16 res(*this);
    for (int i = 0; i < full_blocks; i++)
      res.reg.val[i] = minimum(b.reg.val[i], reg.val[i]);

    if (remainder > 0) {
      auto min_v = minimum(reg.val[full_blocks], b.reg.val[full_blocks]);
      res.reg.val[full_blocks] =
          VectorizedT::set(reg.val[full_blocks], min_v, remainder);
    }

    return res;
  };

  FP32Vec16 max(const FP32Vec16& b, const int elem_num) const {
    TORCH_INTERNAL_ASSERT(elem_num >= 0 && elem_num <= VEC_ELEM_NUM);
    constexpr size_t num_elements = reg.val[0].size();

    if (elem_num == VEC_ELEM_NUM) {
      return FP32Vec16::max(b);
    }

    int full_blocks = elem_num / num_elements;
    int remainder = elem_num % num_elements;

    FP32Vec16 res(*this);

    for (int i = 0; i < full_blocks; i++)
      res.reg.val[i] = maximum(b.reg.val[i], reg.val[i]);

    if (remainder > 0) {
      auto max_v = maximum(reg.val[full_blocks], b.reg.val[full_blocks]);
      res.reg.val[full_blocks] =
          VectorizedT::set(reg.val[full_blocks], max_v, remainder);
    }

    return res;
  };

  float reduce_max() const {
    VectorizedT max_vec = reg.val[0];
    unroll_loop<int, VEC_REG_NUM>([&](int i) {
      if (i > 0) max_vec = maximum(max_vec, reg.val[i]);
    });

    return svmaxv_f32(svptrue_b32(), max_vec);
  }

  float reduce_min() const {
    VectorizedT min_vec = reg.val[0];
    unroll_loop<int, VEC_REG_NUM>([&](int i) {
      if (i > 0) min_vec = minimum(min_vec, reg.val[i]);
    });

    return svminv_f32(svptrue_b32(), min_vec);
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);
    TORCH_INTERNAL_ASSERT(idx >= 0 && idx < VEC_ELEM_NUM / group_size);

    alignas(64) float values[VEC_ELEM_NUM];
    reg.save(values);

    float answer = 0;
    const int start = idx * group_size;
    unroll_loop<int, group_size>([&](int i) { answer += values[start + i]; });

    return answer;
  };

  float reduce_sum() const {
    float answer = 0;
    std::plus<VectorizedT> add;
    unroll_loop<int, VEC_REG_NUM>([&](int i) {
      answer += at::vec::vec_reduce_all<float>(add, reg.val[i]);
    });

    return answer;
  }
};

// We have to implement vectorized widening ourselves, as pytorch doesn't have
// this implemented for FP16, falling back to the generic scalar std functions.
// Then we can access the FP32 SVE/sleef pytorch implementations.
FORCE_INLINE FP16Vec16 FP16Vec16::er() const {
  const FP32Vec16 wide(*this);
  return FP16Vec16(wide.er());
}

FORCE_INLINE FP16Vec16 FP16Vec16::sin() const {
  const FP32Vec16 wide(*this);
  return FP16Vec16(wide.sin());
}

FORCE_INLINE FP16Vec16 FP16Vec16::cos() const {
  const FP32Vec16 wide(*this);
  return FP16Vec16(wide.cos());
}

FORCE_INLINE FP16Vec16 FP16Vec16::tan() const {
  const FP32Vec16 wide(*this);
  return FP16Vec16(wide.tan());
}

FORCE_INLINE FP16Vec16 FP16Vec16::tanh() const {
  const FP32Vec16 wide(*this);
  return FP16Vec16(wide.tanh());
}

struct INT8Vec16 : public Vec<INT8Vec16> {
  using VectorizedT = Vectorized<int8_t>;
  using Reg = NxVectorizedTVecReg<1, int8_t>;

  constexpr static int VEC_ELEM_NUM = 16;

  Reg reg;

  void save(int8_t* ptr) const { reg.val[0].store(ptr, VEC_ELEM_NUM); }

  explicit INT8Vec16(const FP32Vec16& vec) {
    // Convert each 256-bit float32 vector to int32
    auto const pg = svptrue_b32();
    auto rounded0 = svrintn_f32_x(pg, vec.reg.val[0]);
    auto rounded1 = svrintn_f32_x(pg, vec.reg.val[1]);
    svint32_t part0 =
        svcvt_s32_f32_z(pg, rounded0);  // Convert first 256-bit block
    svint32_t part1 =
        svcvt_s32_f32_z(pg, rounded1);  // Convert second 256-bit block

    // Clamp values to the expected range
    part0 = svmax_n_s32_x(pg, part0, -128);
    part0 = svmin_n_s32_x(pg, part0, 127);
    part1 = svmax_n_s32_x(pg, part1, -128);
    part1 = svmin_n_s32_x(pg, part1, 127);

    alignas(16) int8_t values8[16];
    svst1b_s32(pg, values8, part0);
    svst1b_s32(pg, values8 + 8, part1);

    reg.val[0] = VectorizedT::loadu(values8, VEC_ELEM_NUM);
  }

  void save(int8_t* ptr, const int elem_num) const {
    TORCH_CHECK(elem_num >= 0 && elem_num <= VEC_ELEM_NUM);
    reg.val[0].store(ptr, elem_num);
  }
};

struct INT8Vec64 : public Vec<INT8Vec64> {
  using VectorizedT = Vectorized<int8_t>;
  using Reg = NxVectorizedTVecReg<2, int8_t>;
  constexpr static int VEC_ELEM_NUM = 64;

  Reg reg;

  explicit INT8Vec64(const int8_t* ptr) : reg(ptr) {}

  explicit INT8Vec64(bool, const int8_t* ptr) {
    constexpr int lanes = VectorizedT::size();  // 32
    const svbool_t pg = svwhilelt_b8(uint64_t{0}, uint64_t{lanes});

    reg.val[0] = VectorizedT(svldnt1_s8(pg, ptr));
    reg.val[1] = VectorizedT(svldnt1_s8(pg, ptr + lanes));
  }

  void save(int8_t* ptr) const { reg.save(ptr); }

  // masked store
  void save(int8_t* p, int elem_num) const {
    TORCH_CHECK(elem_num <= VEC_ELEM_NUM && elem_num > 0);
    reg.save(p, elem_num);
  }

  void nt_save(int8_t* ptr) const {
    constexpr int lanes = VectorizedT::size();
    const svbool_t pg = svwhilelt_b8(uint64_t{0}, static_cast<uint64_t>(lanes));

    svstnt1_s8(pg, ptr, static_cast<svint8_t>(reg.val[0]));
    svstnt1_s8(pg, ptr + lanes, static_cast<svint8_t>(reg.val[1]));
  }
};  // INT8Vec64

struct INT32Vec16 : public Vec<INT32Vec16> {
  using VectorizedT = Vectorized<int32_t>;
  using Reg = NxVectorizedTVecReg<2, int32_t>;
  constexpr static int VEC_ELEM_NUM = 16;

  Reg reg;

  explicit INT32Vec16(const void* ptr) : reg(ptr) {}

  void save(int32_t* ptr) const { reg.save(ptr); };

  void save(int32_t* ptr, const int elem_num) const {
    TORCH_CHECK(elem_num <= VEC_ELEM_NUM && elem_num > 0);
    reg.save(ptr, elem_num);
  }
};

inline FP16Vec8::FP16Vec8(const FP32Vec8& v) : Base(c10::Half(0)) {
  const svbool_t convert_pg =
      svwhilelt_b32(uint64_t{0}, uint64_t{VEC_ELEM_NUM});

  const svfloat16_t converted =
      svcvt_f16_f32_x(convert_pg, static_cast<svfloat32_t>(v.reg.val[0]));

  // Narrowing places results in alternating halfword lanes.
  const svfloat16_t packed = svuzp1_f16(converted, svdup_n_f16(0.0f));

  const svbool_t store_pg = svwhilelt_b16(uint64_t{0}, uint64_t{VEC_ELEM_NUM});

  c10::Half* destination = reg.val[0];
  svst1_f16(store_pg, reinterpret_cast<float16_t*>(destination), packed);
};

inline FP16Vec16::FP16Vec16(const FP32Vec16& v) {
  const svbool_t pg = svwhilelt_b32(uint64_t{0}, uint64_t{VEC_ELEM_NUM});

  auto lo = svcvt_f16_f32_x(pg, v.reg.val[0]);
  auto hi = svcvt_f16_f32_x(pg, v.reg.val[1]);
  auto packed = svuzp1_f16(lo, hi);

  const svbool_t store_pg = svwhilelt_b16(uint64_t{0}, uint64_t{VEC_ELEM_NUM});

  c10::Half* destination = reg.val[0];
  svst1_f16(store_pg, reinterpret_cast<float16_t*>(destination), packed);
};

FORCE_INLINE void fma(FP32Vec16& acc, const FP32Vec16& a, const FP32Vec16& b) {
  acc.reg.val[0] = fmadd(a.reg.val[0], b.reg.val[0], acc.reg.val[0]);
  acc.reg.val[1] = fmadd(a.reg.val[1], b.reg.val[1], acc.reg.val[1]);
};

inline BF16Vec8::BF16Vec8(const FP32Vec8& v) {
  reg.val[0] = convert_float_bfloat16(v.reg.val[0], Vectorized<float>(0.0f));
};

inline BF16Vec16::BF16Vec16(const FP32Vec16& v) {
  reg.val[0] = convert_float_bfloat16(v.reg.val[0], v.reg.val[1]);
};

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
};

template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
#ifdef ARM_BF16_SUPPORT
  *reinterpret_cast<__bf16*>(ptr) = vcvth_bf16_f32(v);
#else
  *ptr = static_cast<c10::BFloat16>(v);
#endif
};

};  // namespace vec_op
