#ifndef CPU_TYPES_ARM_HPP
#define CPU_TYPES_ARM_HPP

#include <arm_neon.h>
#include <cmath>
#include <iostream>
#include <torch/torch.h>

#include "cpu_types_base.hpp"

#ifndef __ARM_NEON
static_assert(false, "This CPU backend implementation requires ARM Neon.");
#endif

namespace vec_op {

struct FP32Vec8Impl;
struct FP32Vec16Impl;
struct INT32Vec16Impl;

/****************************************/
/*               FP16Vec8               */
/****************************************/
struct FP16Vec8Impl : public FP16Vec8Base {
  float16x8_t reg;

  explicit FP16Vec8Impl(const void* ptr)
      : reg(vld1q_f16(reinterpret_cast<const float16_t*>(ptr))) {}

  explicit FP16Vec8Impl(const FP32Vec8Impl& v);

  void save(void* ptr) const override {
    vst1q_f16(reinterpret_cast<float16_t*>(ptr), reg);
  }
};

/****************************************/
/*               FP16Vec16              */
/****************************************/
struct FP16Vec16Impl : public FP16Vec16Base {
  float16x8x2_t reg;

  constexpr static int VEC_ELEM_NUM = 16;

  union AliasReg {
    float16x8x2_t reg;
    float16_t values[VEC_ELEM_NUM];
  };

  explicit FP16Vec16Impl(const void* ptr)
      : reg{vld1q_f16(reinterpret_cast<const float16_t*>(ptr)),
            vld1q_f16(reinterpret_cast<const float16_t*>(ptr)+8)} {}

  explicit FP16Vec16Impl(const FP32Vec16Impl& v);

  void save(void* ptr) const override {
    vst1q_f16(reinterpret_cast<float16_t*>(ptr), reg.val[0]);
    vst1q_f16(reinterpret_cast<float16_t*>(ptr)+8, reg.val[1]);
  }

  void save(void* ptr, const int elem_num) const override {
    float16_t* ptr2 = reinterpret_cast<float16_t*>(ptr);
    AliasReg ar;

    if (elem_num <= 8) {
      ar.reg.val[0] = reg.val[0];
      for (int i = 0; i < elem_num; ++i) {
        ptr2[i] = ar.values[i];
      }
    } else {
      vst1q_f16(ptr2, reg.val[0]);
      ar.reg.val[1] = reg.val[1];
      for (int i = 8; i < elem_num; ++i) {
        ptr2[i] = ar.values[i];
      }
    }
  }
};

/****************************************/
/*               BF16Vec8               */
/****************************************/
struct BF16Vec8Impl : public BF16Vec8Base {
  uint16x8_t reg;

  explicit BF16Vec8Impl(const void* ptr)
      : reg(vld1q_u16(reinterpret_cast<const uint16_t*>(ptr))) {}

  explicit BF16Vec8Impl(const FP32Vec8Impl&);

  void save(void* ptr) const override {
    vst1q_u16(reinterpret_cast<uint16_t*>(ptr), reg);
  }
};

/****************************************/
/*               BF16Vec16              */
/****************************************/
struct BF16Vec16Impl : public BF16Vec16Base {
  uint16x8x2_t reg;

  constexpr static int VEC_ELEM_NUM = 16;

  union AliasReg {
    uint16x8x2_t reg;
    uint16_t values[VEC_ELEM_NUM];
  };

  explicit BF16Vec16Impl(const void* ptr)
      : reg{vld1q_u16(reinterpret_cast<const uint16_t*>(ptr)),
            vld1q_u16(reinterpret_cast<const uint16_t*>(ptr)+8)} {}

  explicit BF16Vec16Impl(const FP32Vec16Impl& v);

  void save(void* ptr) const override {
    vst1q_u16(reinterpret_cast<uint16_t*>(ptr), reg.val[0]);
    vst1q_u16(reinterpret_cast<uint16_t*>(ptr)+8, reg.val[1]);
  }

  void save(void* ptr, const int elem_num) const override {
    uint16_t* ptr2 = reinterpret_cast<uint16_t*>(ptr);
    AliasReg ar;

    // Masked saving depending on elem_num
    if (elem_num <= 8) {
      ar.reg.val[0] = reg.val[0];
      for (int i = 0; i < elem_num; ++i) {
        ptr2[i] = ar.values[i];
      }
    } else {
      // Save the first 8 elements
      vst1q_u16(ptr2, reg.val[0]);
      ar.reg.val[1] = reg.val[1];
      for (int i = 8; i < elem_num; ++i) {
        ptr2[i] = ar.values[i];
      }
    }
  }
};

/****************************************/
/*               BF16Vec32              */
/****************************************/
struct BF16Vec32Impl : public BF16Vec32Base {
  uint16x8x4_t reg;

  explicit BF16Vec32Impl(const void* ptr)
      : reg{vld1q_u16(reinterpret_cast<const uint16_t*>(ptr)),
            vld1q_u16(reinterpret_cast<const uint16_t*>(ptr)+8),
            vld1q_u16(reinterpret_cast<const uint16_t*>(ptr)+16),
            vld1q_u16(reinterpret_cast<const uint16_t*>(ptr)+24)} {}

  explicit BF16Vec32Impl(uint16x8_t r0, uint16x8_t r1,
                         uint16x8_t r2, uint16x8_t r3)
      : reg{r0, r1, r2, r3} {}

  explicit BF16Vec32Impl(const BF16Vec8Impl& v)
      : reg{v.reg, v.reg, v.reg, v.reg} {}

  void save(void* ptr) const override {
    uint16_t* ptr2 = reinterpret_cast<uint16_t*>(ptr);
    vst1q_u16(ptr2, reg.val[0]);
    vst1q_u16(ptr2+8, reg.val[1]);
    vst1q_u16(ptr2+16, reg.val[2]);
    vst1q_u16(ptr2+24, reg.val[3]);
  }
};

/****************************************/
/*               FP32Vec4               */
/****************************************/
struct FP32Vec4Impl : public FP32Vec4Base {
  float32x4_t reg;

  explicit FP32Vec4Impl(const float* ptr) : reg(vld1q_f32(ptr)) {}

  explicit FP32Vec4Impl(float v) : reg(vdupq_n_f32(v)) {}

  explicit FP32Vec4Impl() : reg(vdupq_n_f32(0.0f)) {}

  explicit FP32Vec4Impl(float32x4_t reg) : reg(reg) {}

  explicit FP32Vec4Impl(const FP32Vec4Impl& v) : reg(v.reg) {}

  void save(float* ptr) const override {
    vst1q_f32(ptr, reg);
  }
};

namespace {
inline void bf16vec8_to_fp32vec8(const uint16x8_t& reg,
                                 float32x4_t& reg_low,
                                 float32x4_t& reg_high) {
  // Shift left by 16 bits to align BF16 with float32
  uint32x4_t ext_low = vshll_n_u16(vget_low_u16(reg), 16);
  uint32x4_t ext_high = vshll_high_n_u16(reg, 16);

  // Reinterpret the extended results as float32
  reg_low = vreinterpretq_f32_u32(ext_low);
  reg_high = vreinterpretq_f32_u32(ext_high);
}
}; // namespace

/****************************************/
/*               FP32Vec8               */
/****************************************/
struct FP32Vec8Impl : public FP32Vec8Base<FP32Vec8Impl> {
  constexpr static int VEC_ELEM_NUM = 8;

  float32x4x2_t reg;

  union AliasReg {
    float32x4x2_t reg;
    float values[VEC_ELEM_NUM];
  };

  explicit FP32Vec8Impl(const void* ptr)
      : reg{vld1q_f32(reinterpret_cast<const float32_t*>(ptr)),
            vld1q_f32(reinterpret_cast<const float32_t*>(ptr)+4)} {}

  explicit FP32Vec8Impl(float v) : reg{vdupq_n_f32(v), vdupq_n_f32(v)} {}

  explicit FP32Vec8Impl() : reg{vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)} {}

  explicit FP32Vec8Impl(const FP32Vec8Impl& v) : reg(v.reg) {}

  explicit FP32Vec8Impl(const float32x4_t& low, const float32x4_t& high)
      : reg{low, high} {}

  explicit FP32Vec8Impl(const float32x4x2_t& reg) : reg(reg) {}

  explicit FP32Vec8Impl(const FP32Vec4Impl& low, const FP32Vec4Impl& high)
      : reg{low.reg, high.reg} {}

  explicit FP32Vec8Impl(const FP16Vec8Impl& v)
      : reg{vcvt_f32_f16(vget_low_f16(v.reg)), vcvt_high_f32_f16(v.reg)} {}

  explicit FP32Vec8Impl(const BF16Vec8Impl& v) {
    bf16vec8_to_fp32vec8(v.reg, reg.val[0], reg.val[1]);
  }

  float reduce_sum() const override {
    float32x4_t sum_vec = vaddq_f32(reg.val[0], reg.val[1]);
    return vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
           vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
  }

  FP32Vec8Impl exp() const override {
    AliasReg ar;
    ar.reg = reg;
    unroll_loop<int, VEC_ELEM_NUM>([&ar](int i) { ar.values[i] = expf(ar.values[i]); });
    return FP32Vec8Impl(ar.reg);
  }

  FP32Vec8Impl tanh() const override {
    AliasReg ar;
    ar.reg = reg;
    unroll_loop<int, VEC_ELEM_NUM>([&ar](int i) { ar.values[i] = tanhf(ar.values[i]); });
    return FP32Vec8Impl(ar.reg);
  }

  FP32Vec8Impl er() const override {
    AliasReg ar;
    ar.reg = reg;
    unroll_loop<int, VEC_ELEM_NUM>([&ar](int i) { ar.values[i] = erf(ar.values[i]); });
    return FP32Vec8Impl(ar.reg);
  }

  FP32Vec8Impl operator*(const FP32Vec8Impl& b) const override {
    return FP32Vec8Impl(vmulq_f32(reg.val[0], b.reg.val[0]), vmulq_f32(reg.val[1], b.reg.val[1]));
  }

  FP32Vec8Impl operator+(const FP32Vec8Impl& b) const override {
    return FP32Vec8Impl(vaddq_f32(reg.val[0], b.reg.val[0]), vaddq_f32(reg.val[1], b.reg.val[1]));
  }

  FP32Vec8Impl operator-(const FP32Vec8Impl& b) const override {
    return FP32Vec8Impl(vsubq_f32(reg.val[0], b.reg.val[0]), vsubq_f32(reg.val[1], b.reg.val[1]));
  }

  FP32Vec8Impl operator/(const FP32Vec8Impl& b) const override {
    return FP32Vec8Impl(vdivq_f32(reg.val[0], b.reg.val[0]), vdivq_f32(reg.val[1], b.reg.val[1]));
  }

  void save(float* ptr) const override {
    vst1q_f32(ptr, reg.val[0]);
    vst1q_f32(ptr+4, reg.val[1]);
  }
};

/****************************************/
/*               FP32Vec16              */
/****************************************/
struct FP32Vec16Impl : public FP32Vec16Base<FP32Vec16Impl> {
  constexpr static int VEC_ELEM_NUM = 16;

  float32x4x4_t reg;

  union AliasReg {
    float32x4x2_t reg;
    float values[VEC_ELEM_NUM/2];
  };

  explicit FP32Vec16Impl(const void* ptr)
      : reg{vld1q_f32(reinterpret_cast<const float32_t*>(ptr)),
            vld1q_f32(reinterpret_cast<const float32_t*>(ptr)+4),
            vld1q_f32(reinterpret_cast<const float32_t*>(ptr)+8),
            vld1q_f32(reinterpret_cast<const float32_t*>(ptr)+12)} {}

  explicit FP32Vec16Impl(const float& v)
      : reg{vdupq_n_f32(v), vdupq_n_f32(v), vdupq_n_f32(v), vdupq_n_f32(v)} {}

  explicit FP32Vec16Impl()
      : reg{vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)} {}

  explicit FP32Vec16Impl(const float32x4_t& r0, const float32x4_t& r1,
                         const float32x4_t& r2, const float32x4_t& r3)
      : reg{r0, r1, r2, r3} {}

  explicit FP32Vec16Impl(const FP32Vec4Impl& v0, const FP32Vec4Impl& v1,
                         const FP32Vec4Impl& v2, const FP32Vec4Impl& v3)
      : reg{v0.reg, v1.reg, v2.reg, v3.reg} {}

  explicit FP32Vec16Impl(const FP32Vec4Impl& v)
      : reg{v.reg, v.reg, v.reg, v.reg} {}

  explicit FP32Vec16Impl(const FP32Vec8Impl& v)
      : reg{v.reg.val[0], v.reg.val[1], v.reg.val[0], v.reg.val[1]} {}

  explicit FP32Vec16Impl(const FP32Vec16Impl& v)
      : reg{v.reg.val[0], v.reg.val[1], v.reg.val[2], v.reg.val[3]} {}

  explicit FP32Vec16Impl(const BF16Vec16Impl& v) {
    bf16vec8_to_fp32vec8(v.reg.val[0], reg.val[0], reg.val[1]);
    bf16vec8_to_fp32vec8(v.reg.val[1], reg.val[2], reg.val[3]);
  }

  explicit FP32Vec16Impl(const FP16Vec16Impl& v)
      : reg{vcvt_f32_f16(vget_low_f16(v.reg.val[0])),
            vcvt_high_f32_f16(v.reg.val[0]),
            vcvt_f32_f16(vget_low_f16(v.reg.val[1])),
            vcvt_high_f32_f16(v.reg.val[1])} {}

  explicit FP32Vec16Impl(const FP16Vec8Impl& v) : FP32Vec16Impl(FP32Vec8Impl(v)) {}

  explicit FP32Vec16Impl(const BF16Vec8Impl& v) : FP32Vec16Impl(FP32Vec8Impl(v)) {}

  explicit FP32Vec16Impl(const INT32Vec16Impl& v) {
    TORCH_CHECK(false, "Not implemented yet.");
  }

  FP32Vec16Impl operator*(const FP32Vec16Impl& b) const override {
    return FP32Vec16Impl(vmulq_f32(reg.val[0], b.reg.val[0]),
                         vmulq_f32(reg.val[1], b.reg.val[1]),
                         vmulq_f32(reg.val[2], b.reg.val[2]),
                         vmulq_f32(reg.val[3], b.reg.val[3]));
  }

  FP32Vec16Impl operator+(const FP32Vec16Impl& b) const override {
    return FP32Vec16Impl(vaddq_f32(reg.val[0], b.reg.val[0]),
                         vaddq_f32(reg.val[1], b.reg.val[1]),
                         vaddq_f32(reg.val[2], b.reg.val[2]),
                         vaddq_f32(reg.val[3], b.reg.val[3]));
  }

  FP32Vec16Impl operator-(const FP32Vec16Impl& b) const override {
    return FP32Vec16Impl(vsubq_f32(reg.val[0], b.reg.val[0]),
                         vsubq_f32(reg.val[1], b.reg.val[1]),
                         vsubq_f32(reg.val[2], b.reg.val[2]),
                         vsubq_f32(reg.val[3], b.reg.val[3]));
  }

  FP32Vec16Impl operator/(const FP32Vec16Impl& b) const override {
    return FP32Vec16Impl(vdivq_f32(reg.val[0], b.reg.val[0]),
                         vdivq_f32(reg.val[1], b.reg.val[1]),
                         vdivq_f32(reg.val[2], b.reg.val[2]),
                         vdivq_f32(reg.val[3], b.reg.val[3]));
  }

  FP32Vec16Impl clamp(const FP32Vec16Impl& min, const FP32Vec16Impl& max) const override {
    TORCH_CHECK(false, "Not implemented yet.");
  }

  FP32Vec16Impl max(const FP32Vec16Impl& b) const override {
    TORCH_CHECK(false, "Not implemented yet.");
  }

  FP32Vec16Impl max(const FP32Vec16Impl& b, const int elem_num) const override {
    TORCH_CHECK(false, "Not implemented yet.");
  }

  FP32Vec16Impl min(const FP32Vec16Impl& b) const override {
    TORCH_CHECK(false, "Not implemented yet.");
  }

  FP32Vec16Impl min(const FP32Vec16Impl& b, const int elem_num) const override {
    TORCH_CHECK(false, "Not implemented yet.");
  }

  FP32Vec16Impl abs() const override {
    TORCH_CHECK(false, "Not implemented yet.");
  }

  float reduce_max() const override {
    TORCH_CHECK(false, "Not implemented yet.");
  }

  float reduce_min() const override {
    TORCH_CHECK(false, "Not implemented yet.");
  }

  float reduce_sum() const override {
    float32x4_t sum_low = vaddq_f32(reg.val[0], reg.val[1]);
    float32x4_t sum_high = vaddq_f32(reg.val[2], reg.val[3]);
    float32x4_t sum_vec = vaddq_f32(sum_low, sum_high);
    return vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
           vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
  }

  float reduce_sub_sum(int idx, int group_size) const override {
    float sum = 0.0;
    uint32_t base_mask = (0xFFFF >> (16 - group_size));
    uint32_t mask = base_mask << (idx * group_size);

    AliasReg ar;

    auto func = [&sum, &mask, &ar](int i) {
      int flag = mask & 0x1;
      mask = mask >> 1;
      if (flag != 0) sum += ar.values[i];
    };

    ar.reg.val[0] = reg.val[0];
    ar.reg.val[1] = reg.val[1];
    unroll_loop<int, 8>(func);

    ar.reg.val[0] = reg.val[2];
    ar.reg.val[1] = reg.val[3];
    unroll_loop<int, 8>(func);

    return sum;
  }

  void save(float* ptr) const override {
    vst1q_f32(ptr, reg.val[0]);
    vst1q_f32(ptr+4, reg.val[1]);
    vst1q_f32(ptr+8, reg.val[2]);
    vst1q_f32(ptr+12, reg.val[3]);
  }

  void save(float* ptr, const int elem_num) const override {
    TORCH_CHECK(false, "Not implemented yet.");
  }
};

/****************************************/
/*               INT32Vec16             */
/****************************************/
struct INT32Vec16Impl : public INT32Vec16Base {
  constexpr static int VEC_ELEM_NUM = 16;

  int32x4x4_t reg;

  union AliasReg {
    int32x4x4_t reg;
    int32_t values[VEC_ELEM_NUM];
  };

  explicit INT32Vec16Impl(const void* ptr)
      : reg{vld1q_s32(reinterpret_cast<const int32_t*>(ptr)),
            vld1q_s32(reinterpret_cast<const int32_t*>(ptr)+4),
            vld1q_s32(reinterpret_cast<const int32_t*>(ptr)+8),
            vld1q_s32(reinterpret_cast<const int32_t*>(ptr)+12)} {}

  void save(int32_t* ptr) const override {
    vst1q_s32(ptr, reg.val[0]);
    vst1q_s32(ptr+4, reg.val[1]);
    vst1q_s32(ptr+8, reg.val[2]);
    vst1q_s32(ptr+12, reg.val[3]);
  }

  void save(int32_t* ptr, const int elem_num) const override {
    AliasReg ar;

    if (elem_num <= 4) {
      ar.reg.val[0] = reg.val[0];
      for (int i = 0; i < elem_num; ++i) {
        ptr[i] = ar.values[i];
      }
    } else if (elem_num <= 8) {
      vst1q_s32(ptr, reg.val[0]);
      ar.reg.val[1] = reg.val[1];
      for (int i = 4; i < elem_num; ++i) {
        ptr[i] = ar.values[i];
      }
    } else if (elem_num <= 12) {
      vst1q_s32(ptr, reg.val[0]);
      vst1q_s32(ptr+4, reg.val[1]);
      ar.reg.val[2] = reg.val[2];
      for (int i = 8; i < elem_num; ++i) {
        ptr[i] = ar.values[i];
      }
    } else {
      vst1q_s32(ptr, reg.val[0]);
      vst1q_s32(ptr+4, reg.val[1]);
      vst1q_s32(ptr+8, reg.val[2]);
      ar.reg.val[3] = reg.val[3];
      for (int i = 12; i < elem_num; ++i) {
        ptr[i] = ar.values[i];
      }
    }
  }
};

/****************************************/
/*               INT8Vec16              */
/****************************************/
struct INT8Vec16Impl : public INT8Vec16Base {
  constexpr static int VEC_ELEM_NUM = 16;

  int8x16_t reg;

  union AliasReg {
    int8x16_t reg;
    int8_t values[VEC_ELEM_NUM];
  };

  explicit INT8Vec16Impl(const void* ptr)
      : reg(vld1q_s8(reinterpret_cast<const int8_t*>(ptr))) {}

  explicit INT8Vec16Impl(const FP32Vec16Impl& v) {
    // Convert from float32x4 to int32x4 with rounding
    int32x4_t int_vec0 = vcvtq_s32_f32(v.reg.val[0]);
    int32x4_t int_vec1 = vcvtq_s32_f32(v.reg.val[1]);
    int32x4_t int_vec2 = vcvtq_s32_f32(v.reg.val[2]);
    int32x4_t int_vec3 = vcvtq_s32_f32(v.reg.val[3]);

    // Narrow each int32x4 down to int16x4 with saturation
    int16x4_t narrow_vec0 = vqmovn_s32(int_vec0);
    int16x4_t narrow_vec1 = vqmovn_s32(int_vec1);
    int16x4_t narrow_vec2 = vqmovn_s32(int_vec2);
    int16x4_t narrow_vec3 = vqmovn_s32(int_vec3);

    // Combine into int16x8_t vectors
    int16x8_t combined_low = vcombine_s16(narrow_vec0, narrow_vec1);
    int16x8_t combined_high = vcombine_s16(narrow_vec2, narrow_vec3);

    // Narrow each int16x8 to int8x8 with saturation and then combine them
    reg = vcombine_s8(vqmovn_s16(combined_low), vqmovn_s16(combined_high));
  }

  void save(int8_t* ptr) const override {
    vst1q_s8(ptr, reg);
  }

  void save(int8_t* ptr, const int elem_num) const override {
    AliasReg ar;

    if (elem_num <= 8) {
      ar.reg = reg;
      for (int i = 0; i < elem_num; ++i) {
        ptr[i] = ar.values[i];
      }
    } else {
      ar.reg = reg;
      int8x8_t reg_low = vld1_s8(ar.values);
      vst1_s8(ptr, reg_low);
      for (int i = 8; i < elem_num; ++i) {
        ptr[i] = ar.values[i];
      }
    }
  }
};

namespace {
inline uint16x8_t fp32vec8_to_bf16vec8(const FP32Vec8Impl& v) {
  uint32x4_t low_u32 = vreinterpretq_u32_f32(v.reg.val[0]);
  uint32x4_t high_u32 = vreinterpretq_u32_f32(v.reg.val[1]);

  // Shift each 32-bit float to the right by 16 bits
  high_u32 = vshrq_n_u32(high_u32, 16);
  low_u32 = vshrq_n_u32(low_u32, 16);

  // Narrow down to 16 bits, packing into bfloat16 (uint16)
  uint16x4_t high_u16 = vmovn_u32(high_u32);
  uint16x4_t low_u16 = vmovn_u32(low_u32);

  // Return packed bf16 values in a structure holding two 64-bit vectors
  return vcombine_u16(low_u16, high_u16);
}
}; // namespace

inline FP16Vec8Impl::FP16Vec8Impl(const FP32Vec8Impl& v)
    : reg(vcombine_f16(vcvt_f16_f32(v.reg.val[0]),
          vcvt_f16_f32(v.reg.val[1]))) {}

inline FP16Vec16Impl::FP16Vec16Impl(const FP32Vec16Impl& v)
    : reg{vcombine_f16(vcvt_f16_f32(v.reg.val[0]),
                       vcvt_f16_f32(v.reg.val[1])),
          vcombine_f16(vcvt_f16_f32(v.reg.val[2]),
                       vcvt_f16_f32(v.reg.val[3]))} {}

inline BF16Vec16Impl::BF16Vec16Impl(const FP32Vec16Impl& v)
    : reg{BF16Vec8Impl(FP32Vec8Impl(v.reg.val[0], v.reg.val[1])).reg,
          BF16Vec8Impl(FP32Vec8Impl(v.reg.val[2], v.reg.val[3])).reg} {}

inline BF16Vec8Impl::BF16Vec8Impl(const FP32Vec8Impl& v)
    : reg(fp32vec8_to_bf16vec8(v)) {}

inline void fma_impl(FP32Vec16Impl& acc, FP32Vec16Impl& a, FP32Vec16Impl& b) {
  acc.reg.val[0] = vfmaq_f32(acc.reg.val[0], a.reg.val[0], b.reg.val[0]);
  acc.reg.val[1] = vfmaq_f32(acc.reg.val[1], a.reg.val[1], b.reg.val[1]);
  acc.reg.val[2] = vfmaq_f32(acc.reg.val[2], a.reg.val[2], b.reg.val[2]);
  acc.reg.val[3] = vfmaq_f32(acc.reg.val[3], a.reg.val[3], b.reg.val[3]);
}

inline void fma_impl(FP32Vec16Impl& acc, BF16Vec32Impl& a, BF16Vec32Impl& b) {
  float32x4x4_t a_f32, b_f32;

  bf16vec8_to_fp32vec8(a.reg.val[0], a_f32.val[0], a_f32.val[1]);
  bf16vec8_to_fp32vec8(a.reg.val[1], a_f32.val[2], a_f32.val[3]);
  bf16vec8_to_fp32vec8(b.reg.val[0], b_f32.val[0], b_f32.val[1]);
  bf16vec8_to_fp32vec8(b.reg.val[1], b_f32.val[2], b_f32.val[3]);

  acc.reg.val[0] = vfmaq_f32(acc.reg.val[0], a_f32.val[0], b_f32.val[0]);
  acc.reg.val[1] = vfmaq_f32(acc.reg.val[1], a_f32.val[1], b_f32.val[1]);
  acc.reg.val[2] = vfmaq_f32(acc.reg.val[2], a_f32.val[2], b_f32.val[2]);
  acc.reg.val[3] = vfmaq_f32(acc.reg.val[3], a_f32.val[3], b_f32.val[3]);
}

inline void storeFP32_impl(float v, c10::Half* ptr) {
  *reinterpret_cast<__fp16 *>(ptr) = v;
}

inline void storeFP32_impl(float v, c10::BFloat16* ptr) {
  c10::BFloat16* __attribute__((__may_alias__)) v_ptr =
    reinterpret_cast<c10::BFloat16*>(&v);
  *ptr = *(v_ptr + 1);
}

inline void prefetch_impl(const void* addr) { __builtin_prefetch(addr, 0, 1); }

}; // namespace vec_op

#endif
