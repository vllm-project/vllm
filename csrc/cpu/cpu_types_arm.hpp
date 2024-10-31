#ifndef CPU_TYPES_ARM_HPP
#define CPU_TYPES_ARM_HPP

#include <arm_neon.h>
#include <torch/torch.h>
#include <iostream>
#include <cmath>

#ifndef __ARM_NEON
static_assert(false, "ARM64 and Neon must be supported for the current implementation.");
#endif

namespace vec_op {

// FIXME: FP16 is not fully supported in Torch-CPU
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)                                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef CPU_OP_GUARD
#define CPU_KERNEL_GUARD_IN(NAME)
#define CPU_KERNEL_GUARD_OUT(NAME)
#else
#define CPU_KERNEL_GUARD_IN(NAME)                                              \
  RECORD_FUNCTION(#NAME, c10::ArrayRef<c10::IValue>({}));
#define CPU_KERNEL_GUARD_OUT(NAME)
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F &&f) {
  (f(std::integral_constant<T, indexes>{}), ...);
}
}; // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F &&f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T> struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; }
};

struct FP32Vec8;
struct FP32Vec16;

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  uint16x8_t reg;

  explicit BF16Vec8(const void *ptr)
    : reg(vld1q_u16(reinterpret_cast<const uint16_t *>(ptr))) {}

  explicit BF16Vec8(const FP32Vec8 &);

  void save(void *ptr) const {
    vst1q_u16(reinterpret_cast<uint16_t *>(ptr), reg);
  }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  union AliasReg {
    uint16x8_t reg[2];
    uint16_t values[VEC_ELEM_NUM];
  };

  uint16x8_t reg_low, reg_high;

  explicit BF16Vec16(const void *ptr)
    : reg_low(vld1q_u16(reinterpret_cast<const uint16_t *>(ptr))),
      reg_high(vld1q_u16(reinterpret_cast<const uint16_t *>(ptr) + 8)) {}

  explicit BF16Vec16(const FP32Vec16 &);

  void save(void *ptr) const {
    uint16_t *data = reinterpret_cast<uint16_t *>(ptr);
    vst1q_u16(data, reg_low);
    vst1q_u16(data + 8, reg_high);
  }

  // Save function with element number control
  void save(void *ptr, const int elem_num) const {
    uint16_t *data = reinterpret_cast<uint16_t *>(ptr);
    AliasReg ar;

    // Masked saving depending on elem_num
    if (elem_num <= 8) {
      ar.reg[0] = reg_low;
      for (int i = 0; i < elem_num; ++i) {
        data[i] = ar.values[i];
      }
    } else {
      // Save the first 8 elements in reg_low
      vst1q_u16(data, reg_low);
      ar.reg[1] = reg_high;
      for (int i = 8; i < elem_num; ++i) {
        data[i] = ar.values[i];
      }
    }
  }
};

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;

  uint16x8_t reg[4];

  explicit BF16Vec32(const void *ptr)
    : reg{vld1q_u16(reinterpret_cast<const uint16_t *>(ptr)),
          vld1q_u16(reinterpret_cast<const uint16_t *>(ptr) + 8),
          vld1q_u16(reinterpret_cast<const uint16_t *>(ptr) + 16),
          vld1q_u16(reinterpret_cast<const uint16_t *>(ptr) + 24)} {}

  // Constructor from two 256-bit segments, each represented by two 128-bit registers
  BF16Vec32(uint16x8_t reg0, uint16x8_t reg1, uint16x8_t reg2, uint16x8_t reg3)
    : reg{reg0, reg1, reg2, reg3} {}

  // Constructor from BF16Vec8 to BF16Vec32 by replicating BF16Vec8's data
  explicit BF16Vec32(const BF16Vec8 &vec8_data)
    : reg{vec8_data.reg, vec8_data.reg, vec8_data.reg, vec8_data.reg} {}

  void save(void *ptr) const {
    uint16_t *data = reinterpret_cast<uint16_t *>(ptr);
    vst1q_u16(data, reg[0]);
    vst1q_u16(data + 8, reg[1]);
    vst1q_u16(data + 16, reg[2]);
    vst1q_u16(data + 24, reg[3]);
  }
};

struct FP32Vec4 : public Vec<FP32Vec4> {
  constexpr static int VEC_ELEM_NUM = 4;

  union AliasReg {
    float32x4_t reg;
    float values[VEC_ELEM_NUM];
  };

  float32x4_t reg;

  // Constructor that sets all elements to a specified float value
  explicit FP32Vec4(float v) : reg(vdupq_n_f32(v)) {}

  // Default constructor that sets all elements to 0.0
  explicit FP32Vec4() : reg(vdupq_n_f32(0.0f)) {}

  // Constructor that loads values from a float pointer
  explicit FP32Vec4(const float *ptr) : reg(vld1q_f32(ptr)) {}

  // Constructor that initializes from an existing float32x4_t
  explicit FP32Vec4(float32x4_t data) : reg(data) {}

  // Copy constructor
  explicit FP32Vec4(const FP32Vec4 &data) : reg(data.reg) {}

  void save(void *ptr) const {
    float32_t *data = reinterpret_cast<float32_t *>(ptr);
    vst1q_f32(data, reg);
  }
};

namespace {
inline void _bf16vec8_to_fp32vec8(const uint16x8_t &reg,
                                  float32x4_t &reg_low,
                                  float32x4_t &reg_high) {
  // Split bf16 vector into low and high parts
  uint16x4_t bf16_low = vget_low_u16(reg);
  uint16x4_t bf16_high = vget_high_u16(reg);

  // Shift left by 16 bits to align BF16 with float32
  uint32x4_t ext_low = vshll_n_u16(bf16_low, 16);
  uint32x4_t ext_high = vshll_n_u16(bf16_high, 16);

  // Reinterpret the extended results as float32
  reg_low = vreinterpretq_f32_u32(ext_low);
  reg_high = vreinterpretq_f32_u32(ext_high);
}
}; // namespace

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  union AliasReg {
    float32x4_t reg[2];
    float values[VEC_ELEM_NUM];
  };

  float32x4_t reg_low, reg_high;

  // Constructors
  explicit FP32Vec8(float v) : reg_low(vdupq_n_f32(v)), reg_high(vdupq_n_f32(v)) {}

  explicit FP32Vec8() : reg_low(vdupq_n_f32(0.0f)), reg_high(vdupq_n_f32(0.0f)) {}

  explicit FP32Vec8(const float *ptr) : reg_low(vld1q_f32(ptr)), reg_high(vld1q_f32(ptr + 4)) {}

  explicit FP32Vec8(const float32x4_t &low, const float32x4_t &high) : reg_low(low), reg_high(high) {}

  explicit FP32Vec8(const float32x4_t &&low, const float32x4_t &&high) : reg_low(low), reg_high(high) {}

  explicit FP32Vec8(const FP32Vec4 &low, const FP32Vec4 &high) : reg_low(low.reg), reg_high(high.reg) {}

  explicit FP32Vec8(const FP32Vec8 &data) : reg_low(data.reg_low), reg_high(data.reg_high) {}

  explicit FP32Vec8(const BF16Vec8 &v) {
    _bf16vec8_to_fp32vec8(v.reg, reg_low, reg_high);
  }

  float reduce_sum() const {
    float32x4_t sum_vec = vaddq_f32(reg_low, reg_high);
    return vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
           vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
  }

  // Element-wise exponent
  FP32Vec8 exp() const {
    AliasReg ar;
    vst1q_f32(ar.values, reg_low);
    vst1q_f32(ar.values + 4, reg_high);
    unroll_loop<int, VEC_ELEM_NUM>([&ar](int i) { ar.values[i] = expf(ar.values[i]); });
    return FP32Vec8(vld1q_f32(ar.values), vld1q_f32(ar.values + 4));
  }

  // Element-wise tanh
  FP32Vec8 tanh() const {
    AliasReg ar;
    vst1q_f32(ar.values, reg_low);
    vst1q_f32(ar.values + 4, reg_high);
    unroll_loop<int, VEC_ELEM_NUM>([&ar](int i) { ar.values[i] = tanhf(ar.values[i]); });
    return FP32Vec8(vld1q_f32(ar.values), vld1q_f32(ar.values + 4));
  }

  // Element-wise error function
  FP32Vec8 er() const {
    AliasReg ar;
    vst1q_f32(ar.values, reg_low);
    vst1q_f32(ar.values + 4, reg_high);
    unroll_loop<int, VEC_ELEM_NUM>([&ar](int i) { ar.values[i] = erf(ar.values[i]); });
    return FP32Vec8(vld1q_f32(ar.values), vld1q_f32(ar.values + 4));
  }

  // Arithmetic operations
  FP32Vec8 operator*(const FP32Vec8 &b) const {
    return FP32Vec8(vmulq_f32(reg_low, b.reg_low), vmulq_f32(reg_high, b.reg_high));
  }

  FP32Vec8 operator+(const FP32Vec8 &b) const {
    return FP32Vec8(vaddq_f32(reg_low, b.reg_low), vaddq_f32(reg_high, b.reg_high));
  }

  FP32Vec8 operator-(const FP32Vec8 &b) const {
    return FP32Vec8(vsubq_f32(reg_low, b.reg_low), vsubq_f32(reg_high, b.reg_high));
  }

  FP32Vec8 operator/(const FP32Vec8 &b) const {
    return FP32Vec8(vdivq_f32(reg_low, b.reg_low), vdivq_f32(reg_high, b.reg_high));
  }

  // Save function
  void save(float *ptr) const {
    vst1q_f32(ptr, reg_low);
    vst1q_f32(ptr + 4, reg_high);
  }
};

struct INT32Vec16 : public Vec<INT32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    int32x4_t reg[4];
    int32_t values[VEC_ELEM_NUM];
  };

  int32x4_t reg[4];

  explicit INT32Vec16(const void *ptr)
    : reg{vld1q_s32(reinterpret_cast<const int32_t *>(ptr)),
          vld1q_s32(reinterpret_cast<const int32_t *>(ptr) + 4),
          vld1q_s32(reinterpret_cast<const int32_t *>(ptr) + 8),
          vld1q_s32(reinterpret_cast<const int32_t *>(ptr) + 12)} {}

  void save(int32_t *ptr) const {
    vst1q_s32(ptr, reg[0]);
    vst1q_s32(ptr + 4, reg[1]);
    vst1q_s32(ptr + 8, reg[2]);
    vst1q_s32(ptr + 12, reg[3]);
  }

  void save(int32_t *ptr, const int elem_num) const {
    AliasReg ar;

    // Masked saving depending on elem_num
    if (elem_num <= 4) {
      ar.reg[0] = reg[0];
      for (int i = 0; i < elem_num; ++i) {
        ptr[i] = ar.values[i];
      }
    } else if (elem_num <= 8) {
      vst1q_s32(ptr, reg[0]);
      ar.reg[1] = reg[1];
      for (int i = 4; i < elem_num; ++i) {
        ptr[i] = ar.values[i];
      }
    } else if (elem_num <= 12) {
      vst1q_s32(ptr, reg[0]);
      vst1q_s32(ptr+4, reg[1]);
      ar.reg[2] = reg[2];
      for (int i = 8; i < elem_num; ++i) {
        ptr[i] = ar.values[i];
      }
    } else {
      vst1q_s32(ptr, reg[0]);
      vst1q_s32(ptr+4, reg[1]);
      vst1q_s32(ptr+8, reg[2]);
      ar.reg[3] = reg[3];
      for (int i = 12; i < elem_num; ++i) {
        ptr[i] = ar.values[i];
      }
    }
  }
};

struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  union AliasReg {
    float32x4_t reg[2];
    float values[VEC_ELEM_NUM/2];
  };

  float32x4_t reg[4];

  // Constructor that sets all values to the same float
  explicit FP32Vec16(const float &v)
    : reg{vdupq_n_f32(v), vdupq_n_f32(v), vdupq_n_f32(v), vdupq_n_f32(v)} {}

  // Constructor that sets all values to 0.0
  explicit FP32Vec16()
    : reg{vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)} {}

  // Constructor that loads values from a float array
  explicit FP32Vec16(const float *ptr)
    : reg{vld1q_f32(ptr), vld1q_f32(ptr + 4), vld1q_f32(ptr + 8), vld1q_f32(ptr + 12)} {}

  // Constructor that takes four registers
  explicit FP32Vec16(const float32x4_t &reg0, const float32x4_t &reg1,
                     const float32x4_t &reg2, const float32x4_t &reg3)
    : reg{reg0, reg1, reg2, reg3} {}

  explicit FP32Vec16(const float32x4_t &&reg0, const float32x4_t &&reg1,
                     const float32x4_t &&reg2, const float32x4_t &&reg3)
    : reg{reg0, reg1, reg2, reg3} {}

  explicit FP32Vec16(const FP32Vec4 &vec0, const FP32Vec4 &vec1,
                     const FP32Vec4 &vec2, const FP32Vec4 &vec3)
    : reg{vec0.reg, vec1.reg, vec2.reg, vec3.reg} {}

  // Copy constructor
  explicit FP32Vec16(const FP32Vec16 &data)
    : reg{data.reg[0], data.reg[1], data.reg[2], data.reg[3]} {}

  // Convert from FP32Vec4
  explicit FP32Vec16(const FP32Vec4 &data)
    : reg{data.reg, data.reg, data.reg, data.reg} {}

  // Convert from FP32Vec8
  explicit FP32Vec16(const FP32Vec8 &data)
    : reg{data.reg_low, data.reg_high, data.reg_low, data.reg_high} {}

  explicit FP32Vec16(const BF16Vec16 &v) {
    _bf16vec8_to_fp32vec8(v.reg_low, reg[0], reg[1]);
    _bf16vec8_to_fp32vec8(v.reg_high, reg[2], reg[3]);
  }

  explicit FP32Vec16(const BF16Vec8 &v) : FP32Vec16(FP32Vec8(v)) {}

  // Multiplication operator
  FP32Vec16 operator*(const FP32Vec16 &b) const {
    return FP32Vec16(vmulq_f32(reg[0], b.reg[0]),
                     vmulq_f32(reg[1], b.reg[1]),
                     vmulq_f32(reg[2], b.reg[2]),
                     vmulq_f32(reg[3], b.reg[3]));
  }

  // Addition operator
  FP32Vec16 operator+(const FP32Vec16 &b) const {
    return FP32Vec16(vaddq_f32(reg[0], b.reg[0]),
                     vaddq_f32(reg[1], b.reg[1]),
                     vaddq_f32(reg[2], b.reg[2]),
                     vaddq_f32(reg[3], b.reg[3]));
  }

  // Subtraction operator
  FP32Vec16 operator-(const FP32Vec16 &b) const {
    return FP32Vec16(vsubq_f32(reg[0], b.reg[0]),
                     vsubq_f32(reg[1], b.reg[1]),
                     vsubq_f32(reg[2], b.reg[2]),
                     vsubq_f32(reg[3], b.reg[3]));
  }

  // Division operator
  FP32Vec16 operator/(const FP32Vec16 &b) const {
    return FP32Vec16(vdivq_f32(reg[0], b.reg[0]),
                     vdivq_f32(reg[1], b.reg[1]),
                     vdivq_f32(reg[2], b.reg[2]),
                     vdivq_f32(reg[3], b.reg[3]));
  }

  // Reduce sum function
  float reduce_sum() const {
    float32x4_t sum_low = vaddq_f32(reg[0], reg[1]);
    float32x4_t sum_high = vaddq_f32(reg[2], reg[3]);
    float32x4_t sum_vec = vaddq_f32(sum_low, sum_high);
    return vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
           vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
  }

  template <int group_size> float reduce_sub_sum(int idx) {
    float sum = 0.0;
    static_assert(VEC_ELEM_NUM % group_size == 0);
    constexpr uint32_t base_mask = (0xFFFF >> (16 - group_size));
    uint32_t mask = base_mask << (idx * group_size);

    AliasReg ar;

    auto func = [&sum, &mask, &ar](int i) {
      int flag = mask & 0x1;
      mask = mask >> 1;
      if (flag != 0) sum += ar.values[i];
    };

    ar.reg[0] = reg[0];
    ar.reg[1] = reg[1];
    unroll_loop<int, 8>(func);

    ar.reg[0] = reg[2];
    ar.reg[1] = reg[3];
    unroll_loop<int, 8>(func);

    return sum;
  }

  // Save function
  void save(float *ptr) const {
    vst1q_f32(ptr, reg[0]);
    vst1q_f32(ptr + 4, reg[1]);
    vst1q_f32(ptr + 8, reg[2]);
    vst1q_f32(ptr + 12, reg[3]);
  }
};

struct INT8Vec16 : public Vec<INT8Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  union AliasReg {
    int8x16_t reg;
    int8_t values[VEC_ELEM_NUM];
  };

  int8x16_t reg;

  explicit INT8Vec16(const void *ptr)
    : reg(vld1q_s8(reinterpret_cast<const int8_t *>(ptr))) {}

  // Constructor that converts from FP32Vec16
  explicit INT8Vec16(const FP32Vec16 &vec) {
    // Convert from float32x4 to int32x4 with rounding
    int32x4_t int_vec0 = vcvtq_s32_f32(vec.reg[0]);
    int32x4_t int_vec1 = vcvtq_s32_f32(vec.reg[1]);
    int32x4_t int_vec2 = vcvtq_s32_f32(vec.reg[2]);
    int32x4_t int_vec3 = vcvtq_s32_f32(vec.reg[3]);

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

  // Save function
  void save(int8_t *ptr) const {
    vst1q_s8(ptr, reg);
  }

  // Save function with element count
  void save(int8_t *ptr, const int elem_num) const {
    AliasReg ar;

    // Masked saving depending on elem_num
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

template <typename T> struct VecType { using vec_type = void; };

template <typename T> using vec_t = typename VecType<T>::vec_type;

template <> struct VecType<float> { using vec_type = FP32Vec8; };

template <> struct VecType<c10::BFloat16> { using vec_type = BF16Vec8; };

template <typename T> void storeFP32(float v, T *ptr) { *ptr = v; }

inline void fma(FP32Vec16 &acc, FP32Vec16 &a, FP32Vec16 &b) {
  acc = acc + a * b;
}

template <> inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16 *ptr) {
  c10::BFloat16 __attribute__((__may_alias__)) *v_ptr =
    reinterpret_cast<c10::BFloat16 *>(&v);
  *ptr = *(v_ptr + 1);
}

namespace {
uint16x8_t FP32Vec8_to_BF16Vec8(const FP32Vec8 &v) {
  uint32x4_t low_u32 = vreinterpretq_u32_f32(v.reg_low);
  uint32x4_t high_u32 = vreinterpretq_u32_f32(v.reg_high);

  // Shift each 32-bit float to the right by 16 bits
  high_u32 = vshrq_n_u32(high_u32, 16);
  low_u32 = vshrq_n_u32(low_u32, 16);

  // Narrow down to 16 bits, packing into bfloat16 (uint16)
  uint16x4_t high_u16 = vmovn_u32(high_u32);
  uint16x4_t low_u16 = vmovn_u32(low_u32);

  // Return packed bf16 values in a structure holding two 64-bit vectors
  return vcombine_u16(low_u16, high_u16);
}
}

inline BF16Vec8::BF16Vec8(const FP32Vec8 &v)
  : reg(FP32Vec8_to_BF16Vec8(v)) {}

inline BF16Vec16::BF16Vec16(const FP32Vec16 &v)
  : reg_low(BF16Vec8(FP32Vec8(v.reg[0], v.reg[1])).reg),
    reg_high(BF16Vec8(FP32Vec8(v.reg[2], v.reg[3])).reg) {}

inline void prefetch(const void *addr) { __builtin_prefetch(addr, 0, 1); }

}; // namespace vec_op

#endif
