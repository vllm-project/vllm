#include <arm_neon.h>
#include <torch/all.h>
#include <cmath>

#if defined(__APPLE__)
  #include "omp.h"
#endif

namespace vec_op {

#ifdef ARM_BF16_SUPPORT
  #define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#else
  #define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)
#endif

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef CPU_OP_GUARD
  #define CPU_KERNEL_GUARD_IN(NAME)
  #define CPU_KERNEL_GUARD_OUT(NAME)
#else
  #define CPU_KERNEL_GUARD_IN(NAME) \
    std::cout << #NAME << " invoked." << std::endl;
  #define CPU_KERNEL_GUARD_OUT(NAME) \
    std::cout << #NAME << " exit." << std::endl;
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline
// Number of elements in single ASIMD vector of given Datatype
#define NUM_ELEMENTS_REG(vec) (sizeof(vec) / sizeof(vec[0]))

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F&& f) {
  (f(std::integral_constant<T, indexes>{}), ...);
};
};  // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F&& f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T>
struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; };
};

struct FP32Vec8;
struct FP32Vec16;

struct FP16Vec8 : public Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  float16x8_t reg;

  explicit FP16Vec8(const void* ptr)
      : reg(vld1q_f16(static_cast<const __fp16*>(ptr))) {};

  explicit FP16Vec8(const FP32Vec8&);

  void save(void* ptr) const { vst1q_f16(static_cast<__fp16*>(ptr), reg); }
};

struct FP16Vec16 : public Vec<FP16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  float16x8x2_t reg;

  explicit FP16Vec16(const void* ptr) {
    reg.val[0] = vld1q_f16(reinterpret_cast<const __fp16*>(ptr));
    reg.val[1] = vld1q_f16(reinterpret_cast<const __fp16*>(ptr) + 8);
  }

  explicit FP16Vec16(const FP32Vec16& vec);

  void save(void* ptr) const {
    vst1q_f16(reinterpret_cast<__fp16*>(ptr), reg.val[0]);
    vst1q_f16(reinterpret_cast<__fp16*>(ptr) + 8, reg.val[1]);
  }

  void save(void* ptr, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);

    if (full_blocks > 0) {
      vst1q_f16(reinterpret_cast<__fp16*>(ptr), reg.val[0]);
      if (full_blocks > 1) {
        vst1q_f16(reinterpret_cast<__fp16*>(ptr) + 8, reg.val[1]);
      }
    }

    // Note: below is the unrolled version of the following code:
    //
    // for (int i = 0; i < remainder; ++i) {
    //     reinterpret_cast<__fp16*>(ptr)[full_blocks * 8 + i] =
    //          vgetq_lane_f16(temp, i);
    // }
    //
    // For macOS build (Clang), the arm/neon intrinsics function
    // `vgetq_lane_f16` needs the parameter `i` to be constant at compile
    // time.

    if (remainder > 0) {
      float16x8_t temp = reg.val[full_blocks];
      __fp16* fp16_ptr = reinterpret_cast<__fp16*>(ptr);
      switch (remainder) {
        case 1:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          break;
        case 2:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          fp16_ptr[full_blocks * 8 + 1] = vgetq_lane_f16(temp, 1);
          break;
        case 3:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          fp16_ptr[full_blocks * 8 + 1] = vgetq_lane_f16(temp, 1);
          fp16_ptr[full_blocks * 8 + 2] = vgetq_lane_f16(temp, 2);
          break;
        case 4:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          fp16_ptr[full_blocks * 8 + 1] = vgetq_lane_f16(temp, 1);
          fp16_ptr[full_blocks * 8 + 2] = vgetq_lane_f16(temp, 2);
          fp16_ptr[full_blocks * 8 + 3] = vgetq_lane_f16(temp, 3);
          break;
        case 5:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          fp16_ptr[full_blocks * 8 + 1] = vgetq_lane_f16(temp, 1);
          fp16_ptr[full_blocks * 8 + 2] = vgetq_lane_f16(temp, 2);
          fp16_ptr[full_blocks * 8 + 3] = vgetq_lane_f16(temp, 3);
          fp16_ptr[full_blocks * 8 + 4] = vgetq_lane_f16(temp, 4);
          break;
        case 6:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          fp16_ptr[full_blocks * 8 + 1] = vgetq_lane_f16(temp, 1);
          fp16_ptr[full_blocks * 8 + 2] = vgetq_lane_f16(temp, 2);
          fp16_ptr[full_blocks * 8 + 3] = vgetq_lane_f16(temp, 3);
          fp16_ptr[full_blocks * 8 + 4] = vgetq_lane_f16(temp, 4);
          fp16_ptr[full_blocks * 8 + 5] = vgetq_lane_f16(temp, 5);
          break;
        case 7:
          fp16_ptr[full_blocks * 8 + 0] = vgetq_lane_f16(temp, 0);
          fp16_ptr[full_blocks * 8 + 1] = vgetq_lane_f16(temp, 1);
          fp16_ptr[full_blocks * 8 + 2] = vgetq_lane_f16(temp, 2);
          fp16_ptr[full_blocks * 8 + 3] = vgetq_lane_f16(temp, 3);
          fp16_ptr[full_blocks * 8 + 4] = vgetq_lane_f16(temp, 4);
          fp16_ptr[full_blocks * 8 + 5] = vgetq_lane_f16(temp, 5);
          fp16_ptr[full_blocks * 8 + 6] = vgetq_lane_f16(temp, 6);
          break;

        default:
          break;
      }
    }
  }
};

#ifdef ARM_BF16_SUPPORT
struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  bfloat16x8_t reg;

  explicit BF16Vec8(const void* ptr)
      : reg(*reinterpret_cast<const bfloat16x8_t*>(ptr)) {};

  explicit BF16Vec8(bfloat16x8_t data) : reg(data) {};

  explicit BF16Vec8(const FP32Vec8&);

  explicit BF16Vec8(float32x4x2_t v)
      : reg(vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.val[0]), v.val[1])) {};

  void save(void* ptr) const { *reinterpret_cast<bfloat16x8_t*>(ptr) = reg; }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  bfloat16x8x2_t reg;

  explicit BF16Vec16(const void* ptr)
      : reg(*reinterpret_cast<const bfloat16x8x2_t*>(ptr)) {};

  explicit BF16Vec16(bfloat16x8x2_t data) : reg(data) {};

  explicit BF16Vec16(const FP32Vec16&);

  explicit BF16Vec16(float32x4x4_t v)
      : reg({vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.val[0]), v.val[1]),
             vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.val[2]), v.val[3])}) {};

  void save(void* ptr) const { *reinterpret_cast<bfloat16x8x2_t*>(ptr) = reg; };
  void save(void* ptr, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);
    for (int i = 0; i < full_blocks; i++)
      vst1q_bf16(
          reinterpret_cast<__bf16*>(ptr) + NUM_ELEMENTS_REG(reg.val[0]) * i,
          reg.val[i]);
    if (remainder > 0) {
      bfloat16x8_t temp = reg.val[full_blocks];
      bfloat16_t* base = reinterpret_cast<bfloat16_t*>(ptr) + full_blocks * 8;
      if (remainder > 0) base[0] = vgetq_lane_bf16(temp, 0);
      if (remainder > 1) base[1] = vgetq_lane_bf16(temp, 1);
      if (remainder > 2) base[2] = vgetq_lane_bf16(temp, 2);
      if (remainder > 3) base[3] = vgetq_lane_bf16(temp, 3);
      if (remainder > 4) base[4] = vgetq_lane_bf16(temp, 4);
      if (remainder > 5) base[5] = vgetq_lane_bf16(temp, 5);
      if (remainder > 6) base[6] = vgetq_lane_bf16(temp, 6);
    }
  };
};

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;

  bfloat16x8x4_t reg;

  explicit BF16Vec32(const void* ptr)
      : reg(*reinterpret_cast<const bfloat16x8x4_t*>(ptr)) {};

  explicit BF16Vec32(bfloat16x8x4_t data) : reg(data) {};

  explicit BF16Vec32(const BF16Vec8& vec8_data)
      : reg({vec8_data.reg, vec8_data.reg, vec8_data.reg, vec8_data.reg}) {};

  void save(void* ptr) const { *reinterpret_cast<bfloat16x8x4_t*>(ptr) = reg; };
  void save(void* ptr, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);
    for (int i = 0; i < full_blocks; i++)
      vst1q_bf16(
          reinterpret_cast<__bf16*>(ptr) + NUM_ELEMENTS_REG(reg.val[0]) * i,
          reg.val[i]);
    if (remainder > 0) {
      bfloat16x8_t temp = reg.val[full_blocks];
      bfloat16_t* base = reinterpret_cast<bfloat16_t*>(ptr) + full_blocks * 8;
      base[0] = vgetq_lane_bf16(temp, 0);
      if (remainder > 1) base[1] = vgetq_lane_bf16(temp, 1);
      if (remainder > 2) base[2] = vgetq_lane_bf16(temp, 2);
      if (remainder > 3) base[3] = vgetq_lane_bf16(temp, 3);
      if (remainder > 4) base[4] = vgetq_lane_bf16(temp, 4);
      if (remainder > 5) base[5] = vgetq_lane_bf16(temp, 5);
      if (remainder > 6) base[6] = vgetq_lane_bf16(temp, 6);
    }
  };
};
#endif

struct FP32Vec4 : public Vec<FP32Vec4> {
  constexpr static int VEC_ELEM_NUM = 4;

  union AliasReg {
    float32x4_t reg;
    float values[VEC_ELEM_NUM];
  };

  float32x4_t reg;

  explicit FP32Vec4(float v) : reg(vdupq_n_f32(v)) {};

  explicit FP32Vec4() : reg(vdupq_n_f32(0.0f)) {};

  explicit FP32Vec4(const float* ptr) : reg(vld1q_f32(ptr)) {};

  explicit FP32Vec4(float32x4_t data) : reg(data) {};

  explicit FP32Vec4(const FP32Vec4& data) : reg(data.reg) {};
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  union AliasReg {
    float32x4x2_t reg;
    float values[VEC_ELEM_NUM];
  };

  float32x4x2_t reg;

  explicit FP32Vec8(float v) : reg({vmovq_n_f32(v), vmovq_n_f32(v)}) {};

  explicit FP32Vec8() : reg({vmovq_n_f32(0.0), vmovq_n_f32(0.0)}) {};

  explicit FP32Vec8(const float* ptr)
      : reg({vld1q_f32(ptr), vld1q_f32(ptr + 4)}) {};

  explicit FP32Vec8(float32x4x2_t data) : reg(data) {};

  explicit FP32Vec8(const FP32Vec8& data) : reg(data.reg) {};

  explicit FP32Vec8(const FP16Vec8& v) {
    reg.val[0] = vcvt_f32_f16(vget_low_f16(v.reg));
    reg.val[1] = vcvt_f32_f16(vget_high_f16(v.reg));
  };

  explicit FP32Vec8(float16x8_t v)
      : reg({vcvt_f32_f16(vget_low_f16(v)), vcvt_f32_f16(vget_high_f16(v))}) {};

#ifdef ARM_BF16_SUPPORT

  explicit FP32Vec8(bfloat16x8_t v)
      : reg({vcvtq_low_f32_bf16(v), vcvtq_high_f32_bf16(v)}) {};

  explicit FP32Vec8(const BF16Vec8& v)
      : reg({vcvtq_low_f32_bf16(v.reg), vcvtq_high_f32_bf16(v.reg)}) {};

#endif

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float answer = 0;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&answer, &ar](int i) { answer += ar.values[i]; });

    return answer;
  }

  FP32Vec8 exp() const {
    AliasReg ar;
    ar.reg = reg;

    float32x2_t exp_vec0 = {expf(ar.values[0]), expf(ar.values[1])};
    float32x2_t exp_vec1 = {expf(ar.values[2]), expf(ar.values[3])};
    float32x2_t exp_vec2 = {expf(ar.values[4]), expf(ar.values[5])};
    float32x2_t exp_vec3 = {expf(ar.values[6]), expf(ar.values[7])};

    float32x4_t result0 = vcombine_f32(exp_vec0, exp_vec1);
    float32x4_t result1 = vcombine_f32(exp_vec2, exp_vec3);

    float32x4x2_t result;
    result.val[0] = result0;
    result.val[1] = result1;

    return FP32Vec8(result);
  }

  FP32Vec8 tanh() const {
    AliasReg ar;
    ar.reg = reg;

    float32x2_t tanh_vec0 = {tanhf(ar.values[0]), tanhf(ar.values[1])};
    float32x2_t tanh_vec1 = {tanhf(ar.values[2]), tanhf(ar.values[3])};
    float32x2_t tanh_vec2 = {tanhf(ar.values[4]), tanhf(ar.values[5])};
    float32x2_t tanh_vec3 = {tanhf(ar.values[6]), tanhf(ar.values[7])};

    float32x4_t result0 = vcombine_f32(tanh_vec0, tanh_vec1);
    float32x4_t result1 = vcombine_f32(tanh_vec2, tanh_vec3);

    float32x4x2_t result;
    result.val[0] = result0;
    result.val[1] = result1;

    return FP32Vec8(result);
  }

  FP32Vec8 er() const {
    AliasReg ar;
    ar.reg = reg;

    float32x2_t er_vec0 = {static_cast<float32_t>(erf(ar.values[0])),
                           static_cast<float32_t>(erf(ar.values[1]))};
    float32x2_t er_vec1 = {static_cast<float32_t>(erf(ar.values[2])),
                           static_cast<float32_t>(erf(ar.values[3]))};
    float32x2_t er_vec2 = {static_cast<float32_t>(erf(ar.values[4])),
                           static_cast<float32_t>(erf(ar.values[5]))};
    float32x2_t er_vec3 = {static_cast<float32_t>(erf(ar.values[6])),
                           static_cast<float32_t>(erf(ar.values[7]))};

    float32x4_t result0 = vcombine_f32(er_vec0, er_vec1);
    float32x4_t result1 = vcombine_f32(er_vec2, er_vec3);

    float32x4x2_t result;
    result.val[0] = result0;
    result.val[1] = result1;

    return FP32Vec8(result);
  }

  FP32Vec8 operator*(const FP32Vec8& b) const {
    return FP32Vec8(float32x4x2_t({vmulq_f32(reg.val[0], b.reg.val[0]),
                                   vmulq_f32(reg.val[1], b.reg.val[1])}));
  }

  FP32Vec8 operator+(const FP32Vec8& b) const {
    return FP32Vec8(float32x4x2_t({vaddq_f32(reg.val[0], b.reg.val[0]),
                                   vaddq_f32(reg.val[1], b.reg.val[1])}));
  }

  FP32Vec8 operator-(const FP32Vec8& b) const {
    return FP32Vec8(float32x4x2_t({vsubq_f32(reg.val[0], b.reg.val[0]),
                                   vsubq_f32(reg.val[1], b.reg.val[1])}));
  }

  FP32Vec8 operator/(const FP32Vec8& b) const {
    return FP32Vec8(float32x4x2_t({vdivq_f32(reg.val[0], b.reg.val[0]),
                                   vdivq_f32(reg.val[1], b.reg.val[1])}));
  }

  void save(float* ptr) const {
    vst1q_f32(ptr, reg.val[0]);
    vst1q_f32(ptr + 4, reg.val[1]);
  }
};

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

struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    float32x4x4_t reg;
    float values[VEC_ELEM_NUM];
  };

  float32x4x4_t reg;

  explicit FP32Vec16(float v)
      : reg({vmovq_n_f32(v), vmovq_n_f32(v), vmovq_n_f32(v), vmovq_n_f32(v)}) {}

  explicit FP32Vec16()
      : reg({vmovq_n_f32(0.0), vmovq_n_f32(0.0), vmovq_n_f32(0.0),
             vmovq_n_f32(0.0)}) {}

  explicit FP32Vec16(const float* ptr)
      : reg({vld1q_f32(ptr), vld1q_f32(ptr + 4), vld1q_f32(ptr + 8),
             vld1q_f32(ptr + 12)}) {}

  explicit FP32Vec16(float32x4x4_t data) : reg(data) {}

  explicit FP32Vec16(const FP32Vec8& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[1];
    reg.val[2] = data.reg.val[0];
    reg.val[3] = data.reg.val[1];
  }

  explicit FP32Vec16(const FP32Vec16& data) : reg(data.reg) {}

  explicit FP32Vec16(const FP16Vec8& v) : FP32Vec16(FP32Vec8(v.reg)) {}

#ifdef ARM_BF16_SUPPORT
  explicit FP32Vec16(bfloat16x8x2_t v)
      : reg({vcvtq_low_f32_bf16(v.val[0]), vcvtq_high_f32_bf16(v.val[0]),
             vcvtq_low_f32_bf16(v.val[1]), vcvtq_high_f32_bf16(v.val[1])}) {};
#endif

  explicit FP32Vec16(const FP32Vec4& data) {
    reg.val[0] = data.reg;
    reg.val[1] = data.reg;
    reg.val[2] = data.reg;
    reg.val[3] = data.reg;
  };

#ifdef ARM_BF16_SUPPORT
  explicit FP32Vec16(const BF16Vec16& v)
      : reg({vcvtq_low_f32_bf16(v.reg.val[0]),
             vcvtq_high_f32_bf16(v.reg.val[0]),
             vcvtq_low_f32_bf16(v.reg.val[1]),
             vcvtq_high_f32_bf16(v.reg.val[1])}) {};

  explicit FP32Vec16(const BF16Vec8& v) : FP32Vec16(FP32Vec8(v)) {};
#endif

  explicit FP32Vec16(const FP16Vec16& v) {
    reg.val[0] = vcvt_f32_f16(vget_low_f16(v.reg.val[0]));
    reg.val[1] = vcvt_f32_f16(vget_high_f16(v.reg.val[0]));
    reg.val[2] = vcvt_f32_f16(vget_low_f16(v.reg.val[1]));
    reg.val[3] = vcvt_f32_f16(vget_high_f16(v.reg.val[1]));
  };
  explicit FP32Vec16(const INT32Vec16& v) {
    reg.val[0] = vcvtq_f32_s32(v.reg.val[0]);
    reg.val[1] = vcvtq_f32_s32(v.reg.val[1]);
    reg.val[2] = vcvtq_f32_s32(v.reg.val[2]);
    reg.val[3] = vcvtq_f32_s32(v.reg.val[3]);
  };
  FP32Vec16 operator+(const FP32Vec16& b) const {
    return FP32Vec16(float32x4x4_t({vaddq_f32(reg.val[0], b.reg.val[0]),
                                    vaddq_f32(reg.val[1], b.reg.val[1]),
                                    vaddq_f32(reg.val[2], b.reg.val[2]),
                                    vaddq_f32(reg.val[3], b.reg.val[3])}));
  };

  FP32Vec16 operator*(const FP32Vec16& b) const {
    return FP32Vec16(float32x4x4_t({vmulq_f32(reg.val[0], b.reg.val[0]),
                                    vmulq_f32(reg.val[1], b.reg.val[1]),
                                    vmulq_f32(reg.val[2], b.reg.val[2]),
                                    vmulq_f32(reg.val[3], b.reg.val[3])}));
  };

  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(float32x4x4_t({vsubq_f32(reg.val[0], b.reg.val[0]),
                                    vsubq_f32(reg.val[1], b.reg.val[1]),
                                    vsubq_f32(reg.val[2], b.reg.val[2]),
                                    vsubq_f32(reg.val[3], b.reg.val[3])}));
  };

  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(float32x4x4_t({vdivq_f32(reg.val[0], b.reg.val[0]),
                                    vdivq_f32(reg.val[1], b.reg.val[1]),
                                    vdivq_f32(reg.val[2], b.reg.val[2]),
                                    vdivq_f32(reg.val[3], b.reg.val[3])}));
  };

  FP32Vec16 clamp(const FP32Vec16& min, const FP32Vec16& max) const {
    return FP32Vec16(float32x4x4_t(
        {vminq_f32(max.reg.val[0], vmaxq_f32(min.reg.val[0], reg.val[0])),
         vminq_f32(max.reg.val[1], vmaxq_f32(min.reg.val[1], reg.val[1])),
         vminq_f32(max.reg.val[2], vmaxq_f32(min.reg.val[2], reg.val[2])),
         vminq_f32(max.reg.val[3], vmaxq_f32(min.reg.val[3], reg.val[3]))}));
  };

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(float32x4x4_t({vmaxq_f32(b.reg.val[0], reg.val[0]),
                                    vmaxq_f32(b.reg.val[1], reg.val[1]),
                                    vmaxq_f32(b.reg.val[2], reg.val[2]),
                                    vmaxq_f32(b.reg.val[3], reg.val[3])}));
  };

  FP32Vec16 max(const FP32Vec16& b, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);
    float32x4x4_t temp;

    for (int i = 0; i < full_blocks; i++)
      temp.val[i] = vmaxq_f32(b.reg.val[i], reg.val[i]);

    if (remainder > 0) {
      float max_v = std::max(vgetq_lane_f32(reg.val[full_blocks], 0),
                             vgetq_lane_f32(b.reg.val[full_blocks], 0));
      temp.val[full_blocks] = vsetq_lane_f32(max_v, temp.val[full_blocks], 0);
    }
    if (remainder > 1) {
      float max_v = std::max(vgetq_lane_f32(reg.val[full_blocks], 1),
                             vgetq_lane_f32(b.reg.val[full_blocks], 1));
      temp.val[full_blocks] = vsetq_lane_f32(max_v, temp.val[full_blocks], 1);
    }
    if (remainder > 2) {
      float max_v = std::max(vgetq_lane_f32(reg.val[full_blocks], 2),
                             vgetq_lane_f32(b.reg.val[full_blocks], 2));
      temp.val[full_blocks] = vsetq_lane_f32(max_v, temp.val[full_blocks], 2);
    }
    return FP32Vec16(temp);
  };

  FP32Vec16 min(const FP32Vec16& b) const {
    return FP32Vec16(float32x4x4_t({
        vminq_f32(b.reg.val[0], reg.val[0]),
        vminq_f32(b.reg.val[1], reg.val[1]),
        vminq_f32(b.reg.val[2], reg.val[2]),
        vminq_f32(b.reg.val[3], reg.val[3]),
    }));
  };
  FP32Vec16 min(const FP32Vec16& b, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    const int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);
    float32x4x4_t temp;
    for (int i = 0; i < full_blocks; i++)
      temp.val[i] = vminq_f32(b.reg.val[i], reg.val[i]);

    if (remainder > 0) {
      float min_v = std::min(vgetq_lane_f32(reg.val[full_blocks], 0),
                             vgetq_lane_f32(b.reg.val[full_blocks], 0));
      temp.val[full_blocks] = vsetq_lane_f32(min_v, temp.val[full_blocks], 0);
    }
    if (remainder > 1) {
      float min_v = std::min(vgetq_lane_f32(reg.val[full_blocks], 1),
                             vgetq_lane_f32(b.reg.val[full_blocks], 1));
      temp.val[full_blocks] = vsetq_lane_f32(min_v, temp.val[full_blocks], 1);
    }
    if (remainder > 2) {
      float min_v = std::min(vgetq_lane_f32(reg.val[full_blocks], 2),
                             vgetq_lane_f32(b.reg.val[full_blocks], 2));
      temp.val[full_blocks] = vsetq_lane_f32(min_v, temp.val[full_blocks], 2);
    }

    return FP32Vec16(temp);
  };
  FP32Vec16 abs() const {
    return FP32Vec16(
        float32x4x4_t({vabsq_f32(reg.val[0]), vabsq_f32(reg.val[1]),
                       vabsq_f32(reg.val[2]), vabsq_f32(reg.val[3])}));
  }
  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float answer = 0;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&answer, &ar](int i) { answer += ar.values[i]; });

    return answer;
  };

  float reduce_max() const {
    AliasReg ar;
    ar.reg = reg;
    float max_v = std::numeric_limits<float>::lowest();
    unroll_loop<int, VEC_ELEM_NUM>(
        [&max_v, &ar](int i) { max_v = std::max(max_v, ar.values[i]); });
    return max_v;
  }

  float reduce_min() const {
    AliasReg ar;
    ar.reg = reg;
    float min_v = std::numeric_limits<float>::max();
    unroll_loop<int, VEC_ELEM_NUM>(
        [&min_v, &ar](int i) { min_v = std::min(min_v, ar.values[i]); });
    return min_v;
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);

    AliasReg ar;
    ar.reg = reg;
    float answer = 0;
    const int start = idx * group_size;
    unroll_loop<int, group_size>(
        [&answer, &start, ar](int i) { answer += ar.values[start + i]; });

    return answer;
  };

  void save(float* ptr) const {
    vst1q_f32(ptr, reg.val[0]);
    vst1q_f32(ptr + 4, reg.val[1]);
    vst1q_f32(ptr + 8, reg.val[2]);
    vst1q_f32(ptr + 12, reg.val[3]);
  };

  void save(float* ptr, const int elem_num) const {
    int full_blocks = elem_num / NUM_ELEMENTS_REG(reg.val[0]);
    int remainder = elem_num % NUM_ELEMENTS_REG(reg.val[0]);

    for (int i = 0; i < full_blocks; i++)
      vst1q_f32(
          reinterpret_cast<float32_t*>(ptr) + NUM_ELEMENTS_REG(reg.val[0]) * i,
          reg.val[i]);

    if (remainder > 0) {
      float32x4_t temp = reg.val[full_blocks];
      float* base = reinterpret_cast<float32_t*>(ptr) +
                    full_blocks * NUM_ELEMENTS_REG(reg.val[0]);
      if (remainder > 0) base[0] = vgetq_lane_f32(temp, 0);
      if (remainder > 1) base[1] = vgetq_lane_f32(temp, 1);
      if (remainder > 2) base[2] = vgetq_lane_f32(temp, 2);
    }
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

template <typename T>
struct VecType {
  using vec_type = void;
};

template <typename T>
using vec_t = typename VecType<T>::vec_type;

template <>
struct VecType<float> {
  using vec_type = FP32Vec8;
};

template <>
struct VecType<c10::Half> {
  using vec_type = FP16Vec8;
};

#ifdef ARM_BF16_SUPPORT
template <>
struct VecType<c10::BFloat16> {
  using vec_type = BF16Vec8;
};
#endif

template <typename T>
void storeFP32(float v, T* ptr) {
  *ptr = v;
}

template <>
inline void storeFP32<c10::Half>(float v, c10::Half* ptr) {
  *reinterpret_cast<__fp16*>(ptr) = v;
}

inline FP16Vec16::FP16Vec16(const FP32Vec16& v) {
  float16x4_t low_0 = vcvt_f16_f32(v.reg.val[0]);
  float16x4_t high_0 = vcvt_f16_f32(v.reg.val[1]);
  float16x4_t low_1 = vcvt_f16_f32(v.reg.val[2]);
  float16x4_t high_1 = vcvt_f16_f32(v.reg.val[3]);

  reg.val[0] = vcombine_f16(low_0, high_0);
  reg.val[1] = vcombine_f16(low_1, high_1);
};

inline FP16Vec8 ::FP16Vec8(const FP32Vec8& v) {
  float16x4_t lower_half = vcvt_f16_f32(v.reg.val[0]);
  float16x4_t upper_half = vcvt_f16_f32(v.reg.val[1]);

  reg = vcombine_f16(lower_half, upper_half);
};

inline void fma(FP32Vec16& acc, FP32Vec16& a, FP32Vec16& b) {
  acc.reg.val[0] = vfmaq_f32(acc.reg.val[0], a.reg.val[0], b.reg.val[0]);
  acc.reg.val[1] = vfmaq_f32(acc.reg.val[1], a.reg.val[1], b.reg.val[1]);
  acc.reg.val[2] = vfmaq_f32(acc.reg.val[2], a.reg.val[2], b.reg.val[2]);
  acc.reg.val[3] = vfmaq_f32(acc.reg.val[3], a.reg.val[3], b.reg.val[3]);
};

#ifdef ARM_BF16_SUPPORT
inline void fma(FP32Vec16& acc, BF16Vec32& a, BF16Vec32& b) {
  float32x4_t a0_low = vcvt_f32_bf16(vget_low_bf16(a.reg.val[0]));
  float32x4_t a0_high = vcvt_f32_bf16(vget_high_bf16(a.reg.val[0]));
  float32x4_t a1_low = vcvt_f32_bf16(vget_low_bf16(a.reg.val[1]));
  float32x4_t a1_high = vcvt_f32_bf16(vget_high_bf16(a.reg.val[1]));

  float32x4_t b0_low = vcvt_f32_bf16(vget_low_bf16(b.reg.val[0]));
  float32x4_t b0_high = vcvt_f32_bf16(vget_high_bf16(b.reg.val[0]));
  float32x4_t b1_low = vcvt_f32_bf16(vget_low_bf16(b.reg.val[1]));
  float32x4_t b1_high = vcvt_f32_bf16(vget_high_bf16(b.reg.val[1]));

  acc.reg.val[0] = vfmaq_f32(acc.reg.val[0], a0_low, b0_low);
  acc.reg.val[1] = vfmaq_f32(acc.reg.val[1], a0_high, b0_high);
  acc.reg.val[2] = vfmaq_f32(acc.reg.val[2], a1_low, b1_low);
  acc.reg.val[3] = vfmaq_f32(acc.reg.val[3], a1_high, b1_high);
};
#endif

#ifdef ARM_BF16_SUPPORT
inline BF16Vec8::BF16Vec8(const FP32Vec8& v)
    : reg(vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.reg.val[0]), v.reg.val[1])) {
      };

inline BF16Vec16::BF16Vec16(const FP32Vec16& v)
    : reg({vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.reg.val[0]), v.reg.val[1]),
           vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.reg.val[2]),
                               v.reg.val[3])}) {};
#endif

inline void prefetch(const void* addr) { __builtin_prefetch(addr, 0, 1); };

#ifdef ARM_BF16_SUPPORT
template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  *reinterpret_cast<__bf16*>(ptr) = vcvth_bf16_f32(v);
};
#endif
};  // namespace vec_op