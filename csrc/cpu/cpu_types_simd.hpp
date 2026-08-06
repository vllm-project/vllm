// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <arm_neon.h>

#include "cpu_tanhf_neon.hpp"
#include "cpu_types_arm.hpp"

using namespace at::vec;

namespace vec_op {

struct FP16Vec8 : public VectorizedRegWrapper<FP16Vec8, 1, c10::Half> {
  using Base = VectorizedRegWrapper<FP16Vec8, 1, c10::Half>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  explicit FP16Vec8(const FP32Vec8&);
};

struct FP16Vec16 : public VectorizedRegWrapper<FP16Vec16, 2, c10::Half> {
  using Base = VectorizedRegWrapper<FP16Vec16, 2, c10::Half>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  // ASIMD does not support non-temporal loads
  explicit FP16Vec16(bool, const void* ptr) : Base(ptr) {}

  explicit FP16Vec16(const FP32Vec16& vec);
};

struct BF16Vec8 : public VectorizedRegWrapper<BF16Vec8, 1, c10::BFloat16> {
  using Base = VectorizedRegWrapper<BF16Vec8, 1, c10::BFloat16>;
  using VectorizedT = typename Base::VectorizedT;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  explicit BF16Vec8(at_bfloat16x8_t data) : Base(VectorizedT(data)) {};

  explicit BF16Vec8(float32x4x2_t v) {
    reg.val[0] = convert_float_bfloat16(v.val[0], v.val[1]);
  };

  explicit BF16Vec8(const FP32Vec8&);
};

struct BF16Vec16 : public VectorizedRegWrapper<BF16Vec16, 2, c10::BFloat16> {
  using Base = VectorizedRegWrapper<BF16Vec16, 2, c10::BFloat16>;
  using VectorizedT = typename Base::VectorizedT;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  // ASIMD does not support non-temporal loads
  explicit BF16Vec16(bool, const void* ptr) : Base(ptr) {}

  explicit BF16Vec16(float32x4x4_t v) {
    reg.val[0] = convert_float_bfloat16(v.val[0], v.val[1]);
    reg.val[1] = convert_float_bfloat16(v.val[2], v.val[3]);
  };

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
  };

  explicit BF16Vec32(const uint8_t*, fp8_e4m3_tag) : Base() {}
  explicit BF16Vec32(const uint8_t*, fp8_e5m2_tag) : Base() {}
};

struct FP32Vec4 : public VectorizedRegWrapper<FP32Vec4, 1, float> {
  using Base = VectorizedRegWrapper<FP32Vec4, 1, float>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  using VectorizedT = typename Base::VectorizedT;
  using Vectorized1x4f = typename Base::NxVectorizedTArray;

  FP32Vec4() : Base() {};
  explicit FP32Vec4(float v) : Base(v) {};

  explicit FP32Vec4(float32x4_t data) : Base(VectorizedT(data)) {};

  explicit FP32Vec4(const FP32Vec4& data) : Base(data) {};

  FORCE_INLINE FP32Vec4 tanh() const {
    return FP32Vec4(fast_tanhf_f32x4(reg.val[0]));
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

  FP32Vec8() : Base() {};
  FP32Vec8(const FP32Vec8& data) : Base(data) {};

  explicit FP32Vec8(float v) : Base(v) {};
  explicit FP32Vec8(const float* ptr)
      : Base(reinterpret_cast<const void*>(ptr)) {};
  explicit FP32Vec8(const float* ptr, const int elem_num)
      : Base(reinterpret_cast<const void*>(ptr), elem_num) {};

  explicit FP32Vec8(const Vectorized2x4f& data) {
    reg.val[0] = data.val[0];
    reg.val[1] = data.val[1];
  };

  explicit FP32Vec8(const BF16Vec8& v) {
    std::tie(reg.val[0], reg.val[1]) = convert_bfloat16_float(v.reg.val[0]);
  };
  explicit FP32Vec8(const FP16Vec8& v) {
    reg.val[0] = Vectorized<float>(vcvt_f32_f16(vget_low_f16(v.reg.val[0])));
    reg.val[1] = Vectorized<float>(vcvt_f32_f16(vget_high_f16(v.reg.val[0])));
  };
  explicit FP32Vec8(float16x8_t v) {
    reg.val[0] = Vectorized<float>(vcvt_f32_f16(vget_low_f16(v)));
    reg.val[1] = Vectorized<float>(vcvt_f32_f16(vget_high_f16(v)));
  };
  explicit FP32Vec8(at_bfloat16x8_t v) {
    std::tie(reg.val[0], reg.val[1]) =
        convert_bfloat16_float(Vectorized<c10::BFloat16>(v));
  };
  explicit FP32Vec8(float32x4x2_t data) {
    reg.val[0] = Vectorized<float>(data.val[0]);
    reg.val[1] = Vectorized<float>(data.val[1]);
  }

  FORCE_INLINE FP32Vec8 tanh() const {
    FP32Vec8 r(uninit);
    r.reg.val[0] = Vectorized<float>(fast_tanhf_f32x4(reg.val[0]));
    r.reg.val[1] = Vectorized<float>(fast_tanhf_f32x4(reg.val[1]));
    return r;
  }

  FORCE_INLINE float reduce_sum() const noexcept {
    float answer = 0;
    std::plus<VectorizedT> add;

    unroll_loop<int, VEC_REG_NUM>([&](int i) {
      answer += at::vec::vec_reduce_all<float, std::plus<VectorizedT>>(
          add, reg.val[i]);
    });
    return answer;
  }

  FORCE_INLINE FP32Vec8 operator+(const FP32Vec8& b) const noexcept {
    FP32Vec8 r(uninit);
    r.reg.val[0] = reg.val[0] + b.reg.val[0];
    r.reg.val[1] = reg.val[1] + b.reg.val[1];
    return r;
  }

  FORCE_INLINE FP32Vec8 operator-(const FP32Vec8& b) const noexcept {
    FP32Vec8 r(uninit);
    r.reg.val[0] = reg.val[0] - b.reg.val[0];
    r.reg.val[1] = reg.val[1] - b.reg.val[1];
    return r;
  }

  FORCE_INLINE FP32Vec8 operator*(const FP32Vec8& b) const noexcept {
    FP32Vec8 r(uninit);
    r.reg.val[0] = reg.val[0] * b.reg.val[0];
    r.reg.val[1] = reg.val[1] * b.reg.val[1];
    return r;
  }

  FORCE_INLINE FP32Vec8 operator/(const FP32Vec8& b) const noexcept {
    FP32Vec8 r(uninit);
    r.reg.val[0] = reg.val[0] / b.reg.val[0];
    r.reg.val[1] = reg.val[1] / b.reg.val[1];
    return r;
  }
};

struct FP32Vec16 : public VectorizedRegWrapper<FP32Vec16, 4, float> {
  using Base = VectorizedRegWrapper<FP32Vec16, 4, float>;
  using Base::Base;
  using Base::get_elem_num;
  using Base::VEC_ELEM_NUM;

  using ScalarT = typename Base::ScalarT;
  using VectorizedT = typename Base::VectorizedT;
  using Vectorized4x4f = typename Base::NxVectorizedTArray;

  FP32Vec16() : Base() {};
  FP32Vec16(const FP32Vec16& data) : Base(data) {};
  FORCE_INLINE explicit FP32Vec16(float v) : Base(v){};
  explicit FP32Vec16(const float* ptr)
      : Base(reinterpret_cast<const void*>(ptr)) {};
  explicit FP32Vec16(const float* ptr, const int elem_num)
      : Base(reinterpret_cast<const void*>(ptr), elem_num) {};
  explicit FP32Vec16(const Vectorized4x4f& data) {
    reg.val[0] = data.val[0];
    reg.val[1] = data.val[1];
    reg.val[2] = data.val[2];
    reg.val[3] = data.val[3];
  };

  // ASIMD does not support non-temporal loads
  explicit FP32Vec16(bool, const float* ptr) : Base(ptr) {}

  explicit FP32Vec16(float32x4x4_t data) {
    reg.val[0] = data.val[0];
    reg.val[1] = data.val[1];
    reg.val[2] = data.val[2];
    reg.val[3] = data.val[3];
  };

  explicit FP32Vec16(const FP32Vec4& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[0];
    reg.val[2] = data.reg.val[0];
    reg.val[3] = data.reg.val[0];
  };

  explicit FP32Vec16(const FP32Vec8& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[1];
    reg.val[2] = data.reg.val[0];
    reg.val[3] = data.reg.val[1];
  };

  explicit FP32Vec16(const BF16Vec16& v) {
    std::tie(reg.val[0], reg.val[1]) = convert_bfloat16_float(v.reg.val[0]);
    std::tie(reg.val[2], reg.val[3]) = convert_bfloat16_float(v.reg.val[1]);
  };

  explicit FP32Vec16(const BF16Vec8& v) : FP32Vec16(FP32Vec8(v)) {};

  // FP8 stub: dead code on ARM (fp8 KV cache is x86-only), needed for
  // load_b_pair_vec template to compile on all platforms.
  explicit FP32Vec16(const BF16Vec32&, int) : Base() {}

  explicit FP32Vec16(const FP16Vec16& v) {
    reg.val[0] = Vectorized<float>(vcvt_f32_f16(vget_low_f16(v.reg.val[0])));
    reg.val[1] = Vectorized<float>(vcvt_f32_f16(vget_high_f16(v.reg.val[0])));
    reg.val[2] = Vectorized<float>(vcvt_f32_f16(vget_low_f16(v.reg.val[1])));
    reg.val[3] = Vectorized<float>(vcvt_f32_f16(vget_high_f16(v.reg.val[1])));
  };

  FORCE_INLINE FP32Vec16 tanh() const {
    FP32Vec16 r(uninit);
    r.reg.val[0] = Vectorized<float>(fast_tanhf_f32x4(reg.val[0]));
    r.reg.val[1] = Vectorized<float>(fast_tanhf_f32x4(reg.val[1]));
    r.reg.val[2] = Vectorized<float>(fast_tanhf_f32x4(reg.val[2]));
    r.reg.val[3] = Vectorized<float>(fast_tanhf_f32x4(reg.val[3]));
    return r;
  }

  static FORCE_INLINE void load_even_odd(const float* ptr, FP32Vec16& even,
                                         FP32Vec16& odd) noexcept {
    const float32x4x2_t x01 = vuzpq_f32(vld1q_f32(ptr), vld1q_f32(ptr + 4));
    const float32x4x2_t x23 =
        vuzpq_f32(vld1q_f32(ptr + 8), vld1q_f32(ptr + 12));
    const float32x4x2_t x45 =
        vuzpq_f32(vld1q_f32(ptr + 16), vld1q_f32(ptr + 20));
    const float32x4x2_t x67 =
        vuzpq_f32(vld1q_f32(ptr + 24), vld1q_f32(ptr + 28));

    even.reg.val[0] = VectorizedT(x01.val[0]);
    even.reg.val[1] = VectorizedT(x23.val[0]);
    even.reg.val[2] = VectorizedT(x45.val[0]);
    even.reg.val[3] = VectorizedT(x67.val[0]);
    odd.reg.val[0] = VectorizedT(x01.val[1]);
    odd.reg.val[1] = VectorizedT(x23.val[1]);
    odd.reg.val[2] = VectorizedT(x45.val[1]);
    odd.reg.val[3] = VectorizedT(x67.val[1]);
  }

  FORCE_INLINE FP32Vec16 operator+(const FP32Vec16& b) const noexcept {
    FP32Vec16 r(uninit);
    r.reg.val[0] = reg.val[0] + b.reg.val[0];
    r.reg.val[1] = reg.val[1] + b.reg.val[1];
    r.reg.val[2] = reg.val[2] + b.reg.val[2];
    r.reg.val[3] = reg.val[3] + b.reg.val[3];
    return r;
  }

  FORCE_INLINE FP32Vec16 operator-(const FP32Vec16& b) const noexcept {
    FP32Vec16 r(uninit);
    r.reg.val[0] = reg.val[0] - b.reg.val[0];
    r.reg.val[1] = reg.val[1] - b.reg.val[1];
    r.reg.val[2] = reg.val[2] - b.reg.val[2];
    r.reg.val[3] = reg.val[3] - b.reg.val[3];
    return r;
  }

  FORCE_INLINE FP32Vec16 operator-() const noexcept {
    FP32Vec16 r(uninit);
    r.reg.val[0] = reg.val[0].neg();
    r.reg.val[1] = reg.val[1].neg();
    r.reg.val[2] = reg.val[2].neg();
    r.reg.val[3] = reg.val[3].neg();
    return r;
  }

  FORCE_INLINE FP32Vec16 operator*(const FP32Vec16& b) const noexcept {
    FP32Vec16 r(uninit);
    r.reg.val[0] = reg.val[0] * b.reg.val[0];
    r.reg.val[1] = reg.val[1] * b.reg.val[1];
    r.reg.val[2] = reg.val[2] * b.reg.val[2];
    r.reg.val[3] = reg.val[3] * b.reg.val[3];
    return r;
  }

  FORCE_INLINE FP32Vec16 operator/(const FP32Vec16& b) const noexcept {
    FP32Vec16 r(uninit);
    r.reg.val[0] = reg.val[0] / b.reg.val[0];
    r.reg.val[1] = reg.val[1] / b.reg.val[1];
    r.reg.val[2] = reg.val[2] / b.reg.val[2];
    r.reg.val[3] = reg.val[3] / b.reg.val[3];
    return r;
  }

  FORCE_INLINE FP32Vec16 clamp(const FP32Vec16& min,
                               const FP32Vec16& max) const {
    FP32Vec16 r(uninit);
    r.reg.val[0] = at::vec::clamp(reg.val[0], min.reg.val[0], max.reg.val[0]);
    r.reg.val[1] = at::vec::clamp(reg.val[1], min.reg.val[1], max.reg.val[1]);
    r.reg.val[2] = at::vec::clamp(reg.val[2], min.reg.val[2], max.reg.val[2]);
    r.reg.val[3] = at::vec::clamp(reg.val[3], min.reg.val[3], max.reg.val[3]);
    return r;
  };

  FORCE_INLINE FP32Vec16 min(const FP32Vec16& b) const {
    FP32Vec16 r(uninit);
    r.reg.val[0] = minimum(b.reg.val[0], reg.val[0]),
    r.reg.val[1] = minimum(b.reg.val[1], reg.val[1]);
    r.reg.val[2] = minimum(b.reg.val[2], reg.val[2]);
    r.reg.val[3] = minimum(b.reg.val[3], reg.val[3]);
    return r;
  };

  FORCE_INLINE FP32Vec16 max(const FP32Vec16& b) const {
    FP32Vec16 r(uninit);
    r.reg.val[0] = maximum(b.reg.val[0], reg.val[0]);
    r.reg.val[1] = maximum(b.reg.val[1], reg.val[1]);
    r.reg.val[2] = maximum(b.reg.val[2], reg.val[2]);
    r.reg.val[3] = maximum(b.reg.val[3], reg.val[3]);
    return r;
  };

  FP32Vec16 min(const FP32Vec16& b, const int elem_num) const {
    size_t num_elements = reg.val[0].size();

    if (elem_num == VEC_ELEM_NUM) {
      return FP32Vec16::min(b);
    }

    int full_blocks = elem_num / num_elements;
    const int remainder = elem_num % num_elements;

    FP32Vec16 res(uninit);
    for (int i = 0; i < full_blocks; i++)
      res.reg.val[i] = minimum(b.reg.val[i], reg.val[i]);

    if (remainder > 0) {
      float min_v = std::min(vgetq_lane_f32(reg.val[full_blocks], 0),
                             vgetq_lane_f32(b.reg.val[full_blocks], 0));
      res.reg.val[full_blocks] =
          vsetq_lane_f32(min_v, res.reg.val[full_blocks], 0);
    }
    if (remainder > 1) {
      float min_v = std::min(vgetq_lane_f32(reg.val[full_blocks], 1),
                             vgetq_lane_f32(b.reg.val[full_blocks], 1));
      res.reg.val[full_blocks] =
          vsetq_lane_f32(min_v, res.reg.val[full_blocks], 1);
    }
    if (remainder > 2) {
      float min_v = std::min(vgetq_lane_f32(reg.val[full_blocks], 2),
                             vgetq_lane_f32(b.reg.val[full_blocks], 2));
      res.reg.val[full_blocks] =
          vsetq_lane_f32(min_v, res.reg.val[full_blocks], 2);
    }

    return res;
  };

  FP32Vec16 max(const FP32Vec16& b, const int elem_num) const {
    size_t num_elements = reg.val[0].size();

    if (elem_num == VEC_ELEM_NUM) {
      return FP32Vec16::max(b);
    }

    int full_blocks = elem_num / num_elements;
    int remainder = elem_num % num_elements;

    FP32Vec16 res(uninit);

    for (int i = 0; i < full_blocks; i++)
      res.reg.val[i] = maximum(b.reg.val[i], reg.val[i]);

    if (remainder > 0) {
      float max_v = std::max(vgetq_lane_f32(reg.val[full_blocks], 0),
                             vgetq_lane_f32(b.reg.val[full_blocks], 0));
      res.reg.val[full_blocks] =
          vsetq_lane_f32(max_v, res.reg.val[full_blocks], 0);
    }
    if (remainder > 1) {
      float max_v = std::max(vgetq_lane_f32(reg.val[full_blocks], 1),
                             vgetq_lane_f32(b.reg.val[full_blocks], 1));
      res.reg.val[full_blocks] =
          vsetq_lane_f32(max_v, res.reg.val[full_blocks], 1);
    }
    if (remainder > 2) {
      float max_v = std::max(vgetq_lane_f32(reg.val[full_blocks], 2),
                             vgetq_lane_f32(b.reg.val[full_blocks], 2));
      res.reg.val[full_blocks] =
          vsetq_lane_f32(max_v, res.reg.val[full_blocks], 2);
    }
    return res;
  };

  float reduce_max() const {
    VectorizedT max_vec = reg.val[0];
    unroll_loop<int, VEC_REG_NUM>([&](int i) {
      if (i > 0) max_vec = maximum(max_vec, reg.val[i]);
    });

    return vmaxvq_f32(max_vec);
  }

  float reduce_min() const {
    VectorizedT min_vec = reg.val[0];
    unroll_loop<int, VEC_REG_NUM>([&](int i) {
      if (i > 0) min_vec = minimum(min_vec, reg.val[i]);
    });

    return vminvq_f32(min_vec);
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);

    AliasReg<NxVectorizedTArray, ScalarT, VEC_ELEM_NUM> ar{reg};
    float answer = 0;
    const int start = idx * group_size;
    unroll_loop<int, group_size>(
        [&](int i) { answer += ar.values[start + i]; });

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

struct INT8Vec16 : public Vec<INT8Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    int8x16_t reg;
    int8_t values[VEC_ELEM_NUM];
  };
  int8x16_t reg;

  explicit INT8Vec16(const FP32Vec16& vec) {
    // Convert each 128-bit float32 vector to int32
    int32x4_t part0 =
        vcvtq_s32_f32(vec.reg.val[0]);  // Convert first 128-bit block
    int32x4_t part1 =
        vcvtq_s32_f32(vec.reg.val[1]);  // Convert second 128-bit block
    int32x4_t part2 =
        vcvtq_s32_f32(vec.reg.val[2]);  // Convert third 128-bit block
    int32x4_t part3 =
        vcvtq_s32_f32(vec.reg.val[3]);  // Convert fourth 128-bit block

    // Narrow each 32-bit vector to 8 bits and combine
    int8x8_t lower =
        vqmovn_s16(vcombine_s16(vqmovn_s32(part0), vqmovn_s32(part1)));
    int8x8_t upper =
        vqmovn_s16(vcombine_s16(vqmovn_s32(part2), vqmovn_s32(part3)));
    reg = vcombine_s8(lower, upper);  // Combine to form a single 128-bit vector
  }

  void save(int8_t* ptr) const { vst1q_s8(ptr, reg); };

  void save(int8_t* ptr, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg);

    for (int i = 0; i < full_blocks; i++)
      vst1q_s8(reinterpret_cast<int8_t*>(ptr) + NUM_ELEMENTS_REG(reg) * i, reg);
    if (remainder > 0) {
      int8x16_t temp = reg;
      int8_t* base =
          reinterpret_cast<int8_t*>(ptr) + full_blocks * NUM_ELEMENTS_REG(reg);
      if (remainder > 0) base[0] = vgetq_lane_s8(temp, 0);
      if (remainder > 1) base[1] = vgetq_lane_s8(temp, 1);
      if (remainder > 2) base[2] = vgetq_lane_s8(temp, 2);
      if (remainder > 3) base[3] = vgetq_lane_s8(temp, 3);
      if (remainder > 4) base[4] = vgetq_lane_s8(temp, 4);
      if (remainder > 5) base[5] = vgetq_lane_s8(temp, 5);
      if (remainder > 6) base[6] = vgetq_lane_s8(temp, 6);
      if (remainder > 7) base[7] = vgetq_lane_s8(temp, 7);
      if (remainder > 8) base[8] = vgetq_lane_s8(temp, 8);
      if (remainder > 9) base[9] = vgetq_lane_s8(temp, 9);
      if (remainder > 10) base[10] = vgetq_lane_s8(temp, 10);
      if (remainder > 11) base[11] = vgetq_lane_s8(temp, 11);
      if (remainder > 12) base[12] = vgetq_lane_s8(temp, 12);
      if (remainder > 13) base[13] = vgetq_lane_s8(temp, 13);
      if (remainder > 14) base[14] = vgetq_lane_s8(temp, 14);
    }
  };
};

struct INT8Vec64 : public Vec<INT8Vec64> {
  constexpr static int VEC_ELEM_NUM = 64;
  union AliasReg {
    int8x16x4_t reg;
    int8_t values[VEC_ELEM_NUM];
  };
  int8x16x4_t reg;

  explicit INT8Vec64(const int8_t* ptr) { reg = vld1q_s8_x4(ptr); }

  // ASIMD does not support non-temporal loads
  explicit INT8Vec64(bool, const int8_t* ptr) : INT8Vec64(ptr) {}

  void save(int8_t* ptr) const { vst1q_s8_x4(ptr, reg); }

  // masked store
  void save(int8_t* p, int elem_num) const {
    TORCH_CHECK(elem_num <= VEC_ELEM_NUM && elem_num > 0);

    if (elem_num == VEC_ELEM_NUM) {
      vst1q_s8_x4(p, reg);
      return;
    }

    const int full_quadwords = elem_num / 16;
    const int remaining_bytes = elem_num % 16;

    for (int i = 0; i < full_quadwords; ++i) {
      vst1q_s8(p + 16 * i, reg.val[i]);
    }

    if (remaining_bytes) {
      const int8x16_t v = reg.val[full_quadwords];
      int8_t* tail = p + 16 * full_quadwords;
      switch (remaining_bytes) {
        case 15:
          tail[14] = vgetq_lane_s8(v, 14);
          [[fallthrough]];
        case 14:
          tail[13] = vgetq_lane_s8(v, 13);
          [[fallthrough]];
        case 13:
          tail[12] = vgetq_lane_s8(v, 12);
          [[fallthrough]];
        case 12:
          tail[11] = vgetq_lane_s8(v, 11);
          [[fallthrough]];
        case 11:
          tail[10] = vgetq_lane_s8(v, 10);
          [[fallthrough]];
        case 10:
          tail[9] = vgetq_lane_s8(v, 9);
          [[fallthrough]];
        case 9:
          tail[8] = vgetq_lane_s8(v, 8);
          [[fallthrough]];
        case 8:
          tail[7] = vgetq_lane_s8(v, 7);
          [[fallthrough]];
        case 7:
          tail[6] = vgetq_lane_s8(v, 6);
          [[fallthrough]];
        case 6:
          tail[5] = vgetq_lane_s8(v, 5);
          [[fallthrough]];
        case 5:
          tail[4] = vgetq_lane_s8(v, 4);
          [[fallthrough]];
        case 4:
          tail[3] = vgetq_lane_s8(v, 3);
          [[fallthrough]];
        case 3:
          tail[2] = vgetq_lane_s8(v, 2);
          [[fallthrough]];
        case 2:
          tail[1] = vgetq_lane_s8(v, 1);
          [[fallthrough]];
        case 1:
          tail[0] = vgetq_lane_s8(v, 0);
          break;
        default:
          break;
      }
    }
  }

  // ASIMD does not support non-temporal stores
  void nt_save(int8_t* ptr) const { save(ptr); }
};  // INT8Vec64

struct INT32Vec16 : public Vec<INT32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    int32x4x4_t reg;
    int32_t values[VEC_ELEM_NUM];
  };
  int32x4x4_t reg;

  explicit INT32Vec16(const void* ptr) {
    reg.val[0] = vld1q_s32(reinterpret_cast<const int32_t*>(ptr));
    reg.val[1] = vld1q_s32(reinterpret_cast<const int32_t*>(ptr) + 4);
    reg.val[2] = vld1q_s32(reinterpret_cast<const int32_t*>(ptr) + 8);
    reg.val[3] = vld1q_s32(reinterpret_cast<const int32_t*>(ptr) + 12);
  }

  void save(int32_t* ptr) const {
    vst1q_s32(ptr, reg.val[0]);
    vst1q_s32(ptr + 4, reg.val[1]);
    vst1q_s32(ptr + 8, reg.val[2]);
    vst1q_s32(ptr + 12, reg.val[3]);
  };

  void save(int32_t* ptr, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);

    for (int i = 0; i < full_blocks; i++)
      vst1q_s32(
          reinterpret_cast<__int32_t*>(ptr) + NUM_ELEMENTS_REG(reg.val[0]) * i,
          reg.val[i]);

    if (remainder > 0) {
      int32x4_t temp = reg.val[full_blocks];
      int32_t* base = reinterpret_cast<int32_t*>(ptr) + full_blocks * 4;
      if (remainder > 0) base[0] = vgetq_lane_s32(temp, 0);
      if (remainder > 1) base[1] = vgetq_lane_s32(temp, 1);
      if (remainder > 2) base[2] = vgetq_lane_s32(temp, 2);
      if (remainder > 3) base[3] = vgetq_lane_s32(temp, 3);
    }
  }
};

inline FP16Vec8::FP16Vec8(const FP32Vec8& v) {
  reg.val[0] = convert_float_half(v.reg.val[0], v.reg.val[1]);
};

inline FP16Vec16::FP16Vec16(const FP32Vec16& v) {
  reg.val[0] = convert_float_half(v.reg.val[0], v.reg.val[1]);
  reg.val[1] = convert_float_half(v.reg.val[2], v.reg.val[3]);
};

inline void fma(FP32Vec16& acc, FP32Vec16& a, FP32Vec16& b) {
  fmadd(acc.reg.val[0], a.reg.val[0], b.reg.val[0]);
  fmadd(acc.reg.val[1], a.reg.val[1], b.reg.val[1]);
  fmadd(acc.reg.val[2], a.reg.val[2], b.reg.val[2]);
  fmadd(acc.reg.val[3], a.reg.val[3], b.reg.val[3]);
};

inline BF16Vec8::BF16Vec8(const FP32Vec8& v) {
  reg.val[0] = convert_float_bfloat16(v.reg.val[0], v.reg.val[1]);
};

inline BF16Vec16::BF16Vec16(const FP32Vec16& v) {
  reg.val[0] = convert_float_bfloat16(v.reg.val[0], v.reg.val[1]);
  reg.val[1] = convert_float_bfloat16(v.reg.val[2], v.reg.val[3]);
};

inline void fma(FP32Vec16& acc, BF16Vec32& a, BF16Vec32& b) {
  Vectorized<float> a0_low, a0_high, a1_low, a1_high, b0_low, b0_high, b1_low,
      b1_high;

  std::tie(a0_low, a0_high) = convert_bfloat16_float(a.reg.val[0]);
  std::tie(a1_low, a1_high) = convert_bfloat16_float(a.reg.val[1]);
  std::tie(b0_low, b0_high) = convert_bfloat16_float(b.reg.val[0]);
  std::tie(b1_low, b1_high) = convert_bfloat16_float(b.reg.val[1]);

  fmadd(acc.reg.val[0], a0_low, b0_low);
  fmadd(acc.reg.val[1], a0_high, b0_high);
  fmadd(acc.reg.val[2], a1_low, b1_low);
  fmadd(acc.reg.val[3], a1_high, b1_high);
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
