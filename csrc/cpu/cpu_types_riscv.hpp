#ifndef CPU_TYPES_RISCV_HPP
#define CPU_TYPES_RISCV_HPP

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <riscv_vector.h>
#include <torch/all.h>

// ============================================================================
// Vector Register Type Definitions (VLEN=128 bits)
// ============================================================================

typedef vfloat16m1_t fixed_vfloat16m1_t
    __attribute__((riscv_rvv_vector_bits(128)));
typedef vfloat16m2_t fixed_vfloat16m2_t
    __attribute__((riscv_rvv_vector_bits(256)));

typedef vfloat32m1_t fixed_vfloat32m1_t
    __attribute__((riscv_rvv_vector_bits(128)));
typedef vfloat32m2_t fixed_vfloat32m2_t
    __attribute__((riscv_rvv_vector_bits(256)));
typedef vfloat32m4_t fixed_vfloat32m4_t
    __attribute__((riscv_rvv_vector_bits(512)));
typedef vfloat32m8_t fixed_vfloat32m8_t
    __attribute__((riscv_rvv_vector_bits(1024)));

typedef vint32m2_t fixed_vint32m2_t __attribute__((riscv_rvv_vector_bits(256)));
typedef vint32m4_t fixed_vint32m4_t __attribute__((riscv_rvv_vector_bits(512)));

typedef vuint16m1_t fixed_vuint16m1_t
    __attribute__((riscv_rvv_vector_bits(128)));
typedef vuint16m2_t fixed_vuint16m2_t
    __attribute__((riscv_rvv_vector_bits(256)));
typedef vuint16m4_t fixed_vuint16m4_t
    __attribute__((riscv_rvv_vector_bits(512)));

#ifdef RISCV_BF16_SUPPORT
typedef vbfloat16m1_t fixed_vbfloat16m1_t
    __attribute__((riscv_rvv_vector_bits(128)));
typedef vbfloat16m2_t fixed_vbfloat16m2_t
    __attribute__((riscv_rvv_vector_bits(256)));
typedef vbfloat16m4_t fixed_vbfloat16m4_t
    __attribute__((riscv_rvv_vector_bits(512)));
#endif

namespace vec_op {

#ifdef RISCV_BF16_SUPPORT
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

#define FORCE_INLINE __attribute__((always_inline)) inline

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F&& f) {
  (f(std::integral_constant<T, indexes>{}), ...);
};
}  // namespace

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

// ============================================================================
// FP16 Implementation
// ============================================================================

struct FP16Vec8 : public Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  fixed_vfloat16m1_t reg;

  explicit FP16Vec8(const void* ptr)
      : reg(__riscv_vle16_v_f16m1(static_cast<const _Float16*>(ptr),
                                  VEC_ELEM_NUM)) {};

  explicit FP16Vec8(const FP32Vec8&);

  void save(void* ptr) const {
    __riscv_vse16_v_f16m1(static_cast<_Float16*>(ptr), reg, VEC_ELEM_NUM);
  }
  void save(void* ptr, int elem_num) const {
    __riscv_vse16_v_f16m1(static_cast<_Float16*>(ptr), reg, elem_num);
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(_Float16);
    __riscv_vsse16_v_f16m1(static_cast<_Float16*>(ptr), byte_stride, reg,
                           VEC_ELEM_NUM);
  }
};

struct FP16Vec16 : public Vec<FP16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  fixed_vfloat16m2_t reg;

  explicit FP16Vec16(const void* ptr)
      : reg(__riscv_vle16_v_f16m2(static_cast<const _Float16*>(ptr),
                                  VEC_ELEM_NUM)) {};

  explicit FP16Vec16(const FP32Vec16& vec);

  void save(void* ptr) const {
    __riscv_vse16_v_f16m2(static_cast<_Float16*>(ptr), reg, VEC_ELEM_NUM);
  }
  void save(void* ptr, int elem_num) const {
    __riscv_vse16_v_f16m2(static_cast<_Float16*>(ptr), reg, elem_num);
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(_Float16);
    __riscv_vsse16_v_f16m2(static_cast<_Float16*>(ptr), byte_stride, reg,
                           VEC_ELEM_NUM);
  }
};

// ============================================================================
// BF16 Implementation
// ============================================================================

#ifdef RISCV_BF16_SUPPORT

FORCE_INLINE fixed_vuint16m1_t bf16_to_u16(fixed_vbfloat16m1_t v) {
  return __riscv_vreinterpret_v_bf16m1_u16m1(v);
}
FORCE_INLINE fixed_vuint16m2_t bf16_to_u16(fixed_vbfloat16m2_t v) {
  return __riscv_vreinterpret_v_bf16m2_u16m2(v);
}
FORCE_INLINE fixed_vuint16m4_t bf16_to_u16(fixed_vbfloat16m4_t v) {
  return __riscv_vreinterpret_v_bf16m4_u16m4(v);
}

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  fixed_vbfloat16m1_t reg;

  explicit BF16Vec8(const void* ptr)
      : reg(__riscv_vreinterpret_v_u16m1_bf16m1(__riscv_vle16_v_u16m1(
            reinterpret_cast<const uint16_t*>(ptr), VEC_ELEM_NUM))) {};

  explicit BF16Vec8(fixed_vbfloat16m1_t data) : reg(data) {};
  explicit BF16Vec8(const FP32Vec8&);

  void save(void* ptr) const {
    __riscv_vse16_v_u16m1(reinterpret_cast<uint16_t*>(ptr), bf16_to_u16(reg),
                          VEC_ELEM_NUM);
  }
  void save(void* ptr, int elem_num) const {
    __riscv_vse16_v_u16m1(reinterpret_cast<uint16_t*>(ptr), bf16_to_u16(reg),
                          elem_num);
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(uint16_t);
    __riscv_vsse16_v_u16m1(reinterpret_cast<uint16_t*>(ptr), byte_stride,
                           bf16_to_u16(reg), VEC_ELEM_NUM);
  }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  fixed_vbfloat16m2_t reg;

  explicit BF16Vec16(const void* ptr)
      : reg(__riscv_vreinterpret_v_u16m2_bf16m2(__riscv_vle16_v_u16m2(
            reinterpret_cast<const uint16_t*>(ptr), VEC_ELEM_NUM))) {};

  explicit BF16Vec16(fixed_vbfloat16m2_t data) : reg(data) {};
  explicit BF16Vec16(const FP32Vec16&);

  void save(void* ptr) const {
    __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(ptr), bf16_to_u16(reg),
                          VEC_ELEM_NUM);
  }
  void save(void* ptr, int elem_num) const {
    __riscv_vse16_v_u16m2(reinterpret_cast<uint16_t*>(ptr), bf16_to_u16(reg),
                          elem_num);
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(uint16_t);
    __riscv_vsse16_v_u16m2(reinterpret_cast<uint16_t*>(ptr), byte_stride,
                           bf16_to_u16(reg), VEC_ELEM_NUM);
  }
};

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;
  fixed_vbfloat16m4_t reg;

  explicit BF16Vec32(const void* ptr)
      : reg(__riscv_vreinterpret_v_u16m4_bf16m4(__riscv_vle16_v_u16m4(
            reinterpret_cast<const uint16_t*>(ptr), VEC_ELEM_NUM))) {};

  explicit BF16Vec32(fixed_vbfloat16m4_t data) : reg(data) {};

  explicit BF16Vec32(const BF16Vec8& v) {
    fixed_vuint16m1_t u16_val = bf16_to_u16(v.reg);
    fixed_vuint16m4_t u16_combined =
        __riscv_vcreate_v_u16m1_u16m4(u16_val, u16_val, u16_val, u16_val);
    reg = __riscv_vreinterpret_v_u16m4_bf16m4(u16_combined);
  };

  void save(void* ptr) const {
    __riscv_vse16_v_u16m4(reinterpret_cast<uint16_t*>(ptr), bf16_to_u16(reg),
                          VEC_ELEM_NUM);
  }
  void save(void* ptr, int elem_num) const {
    __riscv_vse16_v_u16m4(reinterpret_cast<uint16_t*>(ptr), bf16_to_u16(reg),
                          elem_num);
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(uint16_t);
    __riscv_vsse16_v_u16m4(reinterpret_cast<uint16_t*>(ptr), byte_stride,
                           bf16_to_u16(reg), VEC_ELEM_NUM);
  }
};

#else
// ============================================================================
// BF16 Fallback Implementation (FP32 Simulation)
// ============================================================================

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  fixed_vfloat32m2_t reg_fp32;
  explicit BF16Vec8(const void* ptr) {
    const uint16_t* u16 = static_cast<const uint16_t*>(ptr);
    float tmp[8];
    for (int i = 0; i < 8; ++i) {
      uint32_t v = static_cast<uint32_t>(u16[i]) << 16;
      std::memcpy(&tmp[i], &v, 4);
    }
    reg_fp32 = __riscv_vle32_v_f32m2(tmp, 8);
  }
  explicit BF16Vec8(const FP32Vec8&);
  void save(void* ptr) const {
    float tmp[8];
    __riscv_vse32_v_f32m2(tmp, reg_fp32, 8);
    uint16_t* u16 = static_cast<uint16_t*>(ptr);
    for (int i = 0; i < 8; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      u16[i] = static_cast<uint16_t>(v >> 16);
    }
  }
  void save(void* ptr, int elem_num) const {
    float tmp[8];
    __riscv_vse32_v_f32m2(tmp, reg_fp32, 8);
    uint16_t* u16 = static_cast<uint16_t*>(ptr);
    for (int i = 0; i < elem_num; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      u16[i] = static_cast<uint16_t>(v >> 16);
    }
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    float tmp[8];
    __riscv_vse32_v_f32m2(tmp, reg_fp32, 8);
    uint8_t* u8 = static_cast<uint8_t*>(ptr);
    ptrdiff_t byte_stride = stride * sizeof(uint16_t);
    for (int i = 0; i < 8; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      uint16_t val = static_cast<uint16_t>(v >> 16);
      *reinterpret_cast<uint16_t*>(u8 + i * byte_stride) = val;
    }
  }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  fixed_vfloat32m4_t reg_fp32;
  explicit BF16Vec16(const void* ptr) {
    const uint16_t* u16 = static_cast<const uint16_t*>(ptr);
    float tmp[16];
    for (int i = 0; i < 16; ++i) {
      uint32_t v = static_cast<uint32_t>(u16[i]) << 16;
      std::memcpy(&tmp[i], &v, 4);
    }
    reg_fp32 = __riscv_vle32_v_f32m4(tmp, 16);
  }
  explicit BF16Vec16(const FP32Vec16&);
  void save(void* ptr) const {
    float tmp[16];
    __riscv_vse32_v_f32m4(tmp, reg_fp32, 16);
    uint16_t* u16 = static_cast<uint16_t*>(ptr);
    for (int i = 0; i < 16; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      u16[i] = static_cast<uint16_t>(v >> 16);
    }
  }
  void save(void* ptr, int elem_num) const {
    float tmp[16];
    __riscv_vse32_v_f32m4(tmp, reg_fp32, 16);
    uint16_t* u16 = static_cast<uint16_t*>(ptr);
    for (int i = 0; i < elem_num; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      u16[i] = static_cast<uint16_t>(v >> 16);
    }
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    float tmp[16];
    __riscv_vse32_v_f32m4(tmp, reg_fp32, 16);
    uint8_t* u8 = static_cast<uint8_t*>(ptr);
    ptrdiff_t byte_stride = stride * sizeof(uint16_t);
    for (int i = 0; i < 16; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      uint16_t val = static_cast<uint16_t>(v >> 16);
      *reinterpret_cast<uint16_t*>(u8 + i * byte_stride) = val;
    }
  }
};

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;
  fixed_vfloat32m8_t reg_fp32;

  explicit BF16Vec32(const void* ptr) {
    const uint16_t* u16 = static_cast<const uint16_t*>(ptr);
    float tmp[32];
    for (int i = 0; i < 32; ++i) {
      uint32_t v = static_cast<uint32_t>(u16[i]) << 16;
      std::memcpy(&tmp[i], &v, 4);
    }
    reg_fp32 = __riscv_vle32_v_f32m8(tmp, 32);
  }

  explicit BF16Vec32(const BF16Vec8& v) {
    float tmp_small[8];
    __riscv_vse32_v_f32m2(tmp_small, v.reg_fp32, 8);
    float tmp_large[32];
    for (int i = 0; i < 4; ++i) {
      std::memcpy(tmp_large + (i * 8), tmp_small, 8 * sizeof(float));
    }
    reg_fp32 = __riscv_vle32_v_f32m8(tmp_large, 32);
  }

  void save(void* ptr) const {
    float tmp[32];
    __riscv_vse32_v_f32m8(tmp, reg_fp32, 32);
    uint16_t* u16 = static_cast<uint16_t*>(ptr);
    for (int i = 0; i < 32; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      u16[i] = static_cast<uint16_t>(v >> 16);
    }
  }

  void save(void* ptr, int elem_num) const {
    float tmp[32];
    __riscv_vse32_v_f32m8(tmp, reg_fp32, 32);
    uint16_t* u16 = static_cast<uint16_t*>(ptr);
    for (int i = 0; i < elem_num; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      u16[i] = static_cast<uint16_t>(v >> 16);
    }
  }

  void save_strided(void* ptr, ptrdiff_t stride) const {
    float tmp[32];
    __riscv_vse32_v_f32m8(tmp, reg_fp32, 32);
    uint8_t* u8 = static_cast<uint8_t*>(ptr);
    ptrdiff_t byte_stride = stride * sizeof(uint16_t);
    for (int i = 0; i < 32; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      uint16_t val = static_cast<uint16_t>(v >> 16);
      *reinterpret_cast<uint16_t*>(u8 + i * byte_stride) = val;
    }
  }
};
#endif

// ============================================================================
// FP32 Implementation
// ============================================================================

struct FP32Vec4 : public Vec<FP32Vec4> {
  constexpr static int VEC_ELEM_NUM = 4;
  fixed_vfloat32m1_t reg;
  explicit FP32Vec4(float v) : reg(__riscv_vfmv_v_f_f32m1(v, VEC_ELEM_NUM)) {};
  explicit FP32Vec4() : reg(__riscv_vfmv_v_f_f32m1(0.0f, VEC_ELEM_NUM)) {};
  explicit FP32Vec4(const float* ptr)
      : reg(__riscv_vle32_v_f32m1(ptr, VEC_ELEM_NUM)) {};
  explicit FP32Vec4(fixed_vfloat32m1_t data) : reg(data) {};
  explicit FP32Vec4(const FP32Vec4& data) : reg(data.reg) {};
  void save(float* ptr) const { __riscv_vse32_v_f32m1(ptr, reg, VEC_ELEM_NUM); }
  void save(float* ptr, int elem_num) const {
    __riscv_vse32_v_f32m1(ptr, reg, elem_num);
  }
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  fixed_vfloat32m2_t reg;

  explicit FP32Vec8(float v) : reg(__riscv_vfmv_v_f_f32m2(v, VEC_ELEM_NUM)) {};
  explicit FP32Vec8() : reg(__riscv_vfmv_v_f_f32m2(0.0f, VEC_ELEM_NUM)) {};
  explicit FP32Vec8(const float* ptr)
      : reg(__riscv_vle32_v_f32m2(ptr, VEC_ELEM_NUM)) {};
  explicit FP32Vec8(fixed_vfloat32m2_t data) : reg(data) {};
  explicit FP32Vec8(const FP32Vec8& data) : reg(data.reg) {};
  explicit FP32Vec8(const FP16Vec8& v)
      : reg(__riscv_vfwcvt_f_f_v_f32m2(v.reg, VEC_ELEM_NUM)) {};
  explicit FP32Vec8(fixed_vfloat16m1_t v)
      : reg(__riscv_vfwcvt_f_f_v_f32m2(v, VEC_ELEM_NUM)) {};

#ifdef RISCV_BF16_SUPPORT
  explicit FP32Vec8(fixed_vbfloat16m1_t v)
      : reg(__riscv_vfwcvtbf16_f_f_v_f32m2(v, VEC_ELEM_NUM)) {};
  explicit FP32Vec8(const BF16Vec8& v)
      : reg(__riscv_vfwcvtbf16_f_f_v_f32m2(v.reg, VEC_ELEM_NUM)) {};
#else
  explicit FP32Vec8(const BF16Vec8& v) : reg(v.reg_fp32) {};
#endif

  float reduce_sum() const {
    fixed_vfloat32m1_t scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    scalar = __riscv_vfredusum_vs_f32m2_f32m1(reg, scalar, VEC_ELEM_NUM);
    return __riscv_vfmv_f_s_f32m1_f32(scalar);
  }

  FP32Vec8 operator*(const FP32Vec8& b) const {
    return FP32Vec8(__riscv_vfmul_vv_f32m2(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec8 operator+(const FP32Vec8& b) const {
    return FP32Vec8(__riscv_vfadd_vv_f32m2(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec8 operator-(const FP32Vec8& b) const {
    return FP32Vec8(__riscv_vfsub_vv_f32m2(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec8 operator/(const FP32Vec8& b) const {
    return FP32Vec8(__riscv_vfdiv_vv_f32m2(reg, b.reg, VEC_ELEM_NUM));
  }

  FP32Vec8 min(const FP32Vec8& b) const {
    return FP32Vec8(__riscv_vfmin_vv_f32m2(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec8 max(const FP32Vec8& b) const {
    return FP32Vec8(__riscv_vfmax_vv_f32m2(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec8 abs() const {
    return FP32Vec8(__riscv_vfabs_v_f32m2(reg, VEC_ELEM_NUM));
  }

  FP32Vec8 min(const FP32Vec8& b, int elem_num) const {
    return FP32Vec8(__riscv_vfmin_vv_f32m2(reg, b.reg, elem_num));
  }
  FP32Vec8 max(const FP32Vec8& b, int elem_num) const {
    return FP32Vec8(__riscv_vfmax_vv_f32m2(reg, b.reg, elem_num));
  }

  FP32Vec8 clamp(const FP32Vec8& min_v, const FP32Vec8& max_v) const {
    fixed_vfloat32m2_t temp =
        __riscv_vfmax_vv_f32m2(min_v.reg, reg, VEC_ELEM_NUM);
    return FP32Vec8(__riscv_vfmin_vv_f32m2(max_v.reg, temp, VEC_ELEM_NUM));
  }

  void save(float* ptr) const { __riscv_vse32_v_f32m2(ptr, reg, VEC_ELEM_NUM); }
  void save(float* ptr, int elem_num) const {
    __riscv_vse32_v_f32m2(ptr, reg, elem_num);
  }
  void save_strided(float* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(float);
    __riscv_vsse32_v_f32m2(ptr, byte_stride, reg, VEC_ELEM_NUM);
  }

  FP32Vec8 exp() const {
    const float inv_ln2 = 1.44269504088896341f;
    fixed_vfloat32m2_t x_scaled =
        __riscv_vfmul_vf_f32m2(reg, inv_ln2, VEC_ELEM_NUM);
    fixed_vint32m2_t n_int = __riscv_vfcvt_x_f_v_i32m2(x_scaled, VEC_ELEM_NUM);
    fixed_vfloat32m2_t n_float = __riscv_vfcvt_f_x_v_f32m2(n_int, VEC_ELEM_NUM);

    fixed_vfloat32m2_t r =
        __riscv_vfsub_vv_f32m2(x_scaled, n_float, VEC_ELEM_NUM);

    fixed_vfloat32m2_t poly =
        __riscv_vfmv_v_f_f32m2(0.001333355810164f, VEC_ELEM_NUM);
    poly = __riscv_vfmul_vv_f32m2(poly, r, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m2(poly, 0.009618129107628f, VEC_ELEM_NUM);
    poly = __riscv_vfmul_vv_f32m2(poly, r, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m2(poly, 0.055504108664821f, VEC_ELEM_NUM);
    poly = __riscv_vfmul_vv_f32m2(poly, r, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m2(poly, 0.240226506959101f, VEC_ELEM_NUM);
    poly = __riscv_vfmul_vv_f32m2(poly, r, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m2(poly, 0.693147180559945f, VEC_ELEM_NUM);
    poly = __riscv_vfmul_vv_f32m2(poly, r, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m2(poly, 1.0f, VEC_ELEM_NUM);

    fixed_vint32m2_t biased_exp =
        __riscv_vadd_vx_i32m2(n_int, 127, VEC_ELEM_NUM);
    biased_exp = __riscv_vmax_vx_i32m2(biased_exp, 0, VEC_ELEM_NUM);
    fixed_vint32m2_t exponent_bits =
        __riscv_vsll_vx_i32m2(biased_exp, 23, VEC_ELEM_NUM);
    fixed_vfloat32m2_t scale =
        __riscv_vreinterpret_v_i32m2_f32m2(exponent_bits);

    return FP32Vec8(__riscv_vfmul_vv_f32m2(poly, scale, VEC_ELEM_NUM));
  }

  FP32Vec8 tanh() const {
    fixed_vfloat32m2_t x_clamped = __riscv_vfmin_vf_f32m2(
        __riscv_vfmax_vf_f32m2(reg, -9.0f, VEC_ELEM_NUM), 9.0f, VEC_ELEM_NUM);
    fixed_vfloat32m2_t x2 =
        __riscv_vfmul_vf_f32m2(x_clamped, 2.0f, VEC_ELEM_NUM);
    FP32Vec8 exp_val = FP32Vec8(x2).exp();
    fixed_vfloat32m2_t num =
        __riscv_vfsub_vf_f32m2(exp_val.reg, 1.0f, VEC_ELEM_NUM);
    fixed_vfloat32m2_t den =
        __riscv_vfadd_vf_f32m2(exp_val.reg, 1.0f, VEC_ELEM_NUM);
    return FP32Vec8(__riscv_vfdiv_vv_f32m2(num, den, VEC_ELEM_NUM));
  }

  FP32Vec8 er() const {
    const float p = 0.3275911f, a1 = 0.254829592f, a2 = -0.284496736f,
                a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
    fixed_vfloat32m2_t abs_x = __riscv_vfabs_v_f32m2(reg, VEC_ELEM_NUM);

    fixed_vfloat32m2_t t = __riscv_vfadd_vf_f32m2(
        __riscv_vfmul_vf_f32m2(abs_x, p, VEC_ELEM_NUM), 1.0f, VEC_ELEM_NUM);
    t = __riscv_vfrdiv_vf_f32m2(t, 1.0f, VEC_ELEM_NUM);

    fixed_vfloat32m2_t poly = __riscv_vfmv_v_f_f32m2(a5, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(poly, t, VEC_ELEM_NUM),
                                  a4, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(poly, t, VEC_ELEM_NUM),
                                  a3, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(poly, t, VEC_ELEM_NUM),
                                  a2, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(poly, t, VEC_ELEM_NUM),
                                  a1, VEC_ELEM_NUM);
    poly = __riscv_vfmul_vv_f32m2(poly, t, VEC_ELEM_NUM);

    fixed_vfloat32m2_t exp_val =
        FP32Vec8(__riscv_vfneg_v_f32m2(
                     __riscv_vfmul_vv_f32m2(abs_x, abs_x, VEC_ELEM_NUM),
                     VEC_ELEM_NUM))
            .exp()
            .reg;
    fixed_vfloat32m2_t res = __riscv_vfrsub_vf_f32m2(
        __riscv_vfmul_vv_f32m2(poly, exp_val, VEC_ELEM_NUM), 1.0f,
        VEC_ELEM_NUM);

    vbool16_t mask = __riscv_vmflt_vf_f32m2_b16(reg, 0.0f, VEC_ELEM_NUM);
    return FP32Vec8(__riscv_vfneg_v_f32m2_m(mask, res, VEC_ELEM_NUM));
  }
};

struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  fixed_vfloat32m4_t reg;

  explicit FP32Vec16(float v) : reg(__riscv_vfmv_v_f_f32m4(v, VEC_ELEM_NUM)) {};
  explicit FP32Vec16() : reg(__riscv_vfmv_v_f_f32m4(0.0f, VEC_ELEM_NUM)) {};
  explicit FP32Vec16(const float* ptr)
      : reg(__riscv_vle32_v_f32m4(ptr, VEC_ELEM_NUM)) {};
  explicit FP32Vec16(fixed_vfloat32m4_t data) : reg(data) {};
  explicit FP32Vec16(const FP32Vec8& data)
      : reg(__riscv_vcreate_v_f32m2_f32m4(data.reg, data.reg)) {};
  explicit FP32Vec16(const FP32Vec16& data) : reg(data.reg) {};
  explicit FP32Vec16(const FP16Vec16& v);

#ifdef RISCV_BF16_SUPPORT
  explicit FP32Vec16(fixed_vbfloat16m2_t v)
      : reg(__riscv_vfwcvtbf16_f_f_v_f32m4(v, VEC_ELEM_NUM)) {};
  explicit FP32Vec16(const BF16Vec16& v)
      : reg(__riscv_vfwcvtbf16_f_f_v_f32m4(v.reg, VEC_ELEM_NUM)) {};
#else
  explicit FP32Vec16(const BF16Vec16& v) : reg(v.reg_fp32) {};
#endif

  FP32Vec16 operator+(const FP32Vec16& b) const {
    return FP32Vec16(__riscv_vfadd_vv_f32m4(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(__riscv_vfsub_vv_f32m4(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec16 operator*(const FP32Vec16& b) const {
    return FP32Vec16(__riscv_vfmul_vv_f32m4(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(__riscv_vfdiv_vv_f32m4(reg, b.reg, VEC_ELEM_NUM));
  }

  FP32Vec16 fma(const FP32Vec16& a, const FP32Vec16& b) const {
    return FP32Vec16(__riscv_vfmacc_vv_f32m4(reg, a.reg, b.reg, VEC_ELEM_NUM));
  }

  float reduce_sum() const {
    fixed_vfloat32m1_t scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    scalar = __riscv_vfredusum_vs_f32m4_f32m1(reg, scalar, VEC_ELEM_NUM);
    return __riscv_vfmv_f_s_f32m1_f32(scalar);
  }

  float reduce_max() const {
    fixed_vfloat32m1_t scalar =
        __riscv_vfmv_s_f_f32m1(std::numeric_limits<float>::lowest(), 1);
    scalar = __riscv_vfredmax_vs_f32m4_f32m1(reg, scalar, VEC_ELEM_NUM);
    return __riscv_vfmv_f_s_f32m1_f32(scalar);
  }

  float reduce_min() const {
    fixed_vfloat32m1_t scalar =
        __riscv_vfmv_s_f_f32m1(std::numeric_limits<float>::max(), 1);
    scalar = __riscv_vfredmin_vs_f32m4_f32m1(reg, scalar, VEC_ELEM_NUM);
    return __riscv_vfmv_f_s_f32m1_f32(scalar);
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);
    const int start = idx * group_size;
    vuint32m4_t indices = __riscv_vid_v_u32m4(VEC_ELEM_NUM);
    vbool8_t mask = __riscv_vmand_mm_b8(
        __riscv_vmsgeu_vx_u32m4_b8(indices, start, VEC_ELEM_NUM),
        __riscv_vmsltu_vx_u32m4_b8(indices, start + group_size, VEC_ELEM_NUM),
        VEC_ELEM_NUM);
    fixed_vfloat32m1_t scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    scalar =
        __riscv_vfredusum_vs_f32m4_f32m1_m(mask, reg, scalar, VEC_ELEM_NUM);
    return __riscv_vfmv_f_s_f32m1_f32(scalar);
  };

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(__riscv_vfmax_vv_f32m4(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec16 min(const FP32Vec16& b) const {
    return FP32Vec16(__riscv_vfmin_vv_f32m4(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec16 abs() const {
    return FP32Vec16(__riscv_vfabs_v_f32m4(reg, VEC_ELEM_NUM));
  }

  FP32Vec16 clamp(const FP32Vec16& min_v, const FP32Vec16& max_v) const {
    return FP32Vec16(__riscv_vfmin_vv_f32m4(
        max_v.reg, __riscv_vfmax_vv_f32m4(min_v.reg, reg, VEC_ELEM_NUM),
        VEC_ELEM_NUM));
  }

  void save(float* ptr) const { __riscv_vse32_v_f32m4(ptr, reg, VEC_ELEM_NUM); }
  void save(float* ptr, int elem_num) const {
    __riscv_vse32_v_f32m4(ptr, reg, elem_num);
  }
  void save_strided(float* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(float);
    __riscv_vsse32_v_f32m4(ptr, byte_stride, reg, VEC_ELEM_NUM);
  }

  FP32Vec16 exp() const {
    const float inv_ln2 = 1.44269504088896341f;
    fixed_vfloat32m4_t x_scaled =
        __riscv_vfmul_vf_f32m4(reg, inv_ln2, VEC_ELEM_NUM);
    fixed_vint32m4_t n_int = __riscv_vfcvt_x_f_v_i32m4(x_scaled, VEC_ELEM_NUM);
    fixed_vfloat32m4_t n_float = __riscv_vfcvt_f_x_v_f32m4(n_int, VEC_ELEM_NUM);
    fixed_vfloat32m4_t r =
        __riscv_vfsub_vv_f32m4(x_scaled, n_float, VEC_ELEM_NUM);

    fixed_vfloat32m4_t poly =
        __riscv_vfmv_v_f_f32m4(0.001333355810164f, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, r, VEC_ELEM_NUM),
                                  0.009618129107628f, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, r, VEC_ELEM_NUM),
                                  0.055504108664821f, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, r, VEC_ELEM_NUM),
                                  0.240226506959101f, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, r, VEC_ELEM_NUM),
                                  0.693147180559945f, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, r, VEC_ELEM_NUM),
                                  1.0f, VEC_ELEM_NUM);

    fixed_vint32m4_t biased_exp = __riscv_vmax_vx_i32m4(
        __riscv_vadd_vx_i32m4(n_int, 127, VEC_ELEM_NUM), 0, VEC_ELEM_NUM);
    fixed_vfloat32m4_t scale = __riscv_vreinterpret_v_i32m4_f32m4(
        __riscv_vsll_vx_i32m4(biased_exp, 23, VEC_ELEM_NUM));

    return FP32Vec16(__riscv_vfmul_vv_f32m4(poly, scale, VEC_ELEM_NUM));
  }

  FP32Vec16 tanh() const {
    fixed_vfloat32m4_t x_clamped = __riscv_vfmin_vf_f32m4(
        __riscv_vfmax_vf_f32m4(reg, -9.0f, VEC_ELEM_NUM), 9.0f, VEC_ELEM_NUM);
    FP32Vec16 exp_val =
        FP32Vec16(__riscv_vfmul_vf_f32m4(x_clamped, 2.0f, VEC_ELEM_NUM)).exp();
    return FP32Vec16(__riscv_vfdiv_vv_f32m4(
        __riscv_vfsub_vf_f32m4(exp_val.reg, 1.0f, VEC_ELEM_NUM),
        __riscv_vfadd_vf_f32m4(exp_val.reg, 1.0f, VEC_ELEM_NUM), VEC_ELEM_NUM));
  }

  FP32Vec16 er() const {
    const float p = 0.3275911f, a1 = 0.254829592f, a2 = -0.284496736f,
                a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
    fixed_vfloat32m4_t abs_x = __riscv_vfabs_v_f32m4(reg, VEC_ELEM_NUM);
    fixed_vfloat32m4_t t = __riscv_vfrdiv_vf_f32m4(
        __riscv_vfadd_vf_f32m4(__riscv_vfmul_vf_f32m4(abs_x, p, VEC_ELEM_NUM),
                               1.0f, VEC_ELEM_NUM),
        1.0f, VEC_ELEM_NUM);

    fixed_vfloat32m4_t poly = __riscv_vfmv_v_f_f32m4(a5, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, t, VEC_ELEM_NUM),
                                  a4, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, t, VEC_ELEM_NUM),
                                  a3, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, t, VEC_ELEM_NUM),
                                  a2, VEC_ELEM_NUM);
    poly = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(poly, t, VEC_ELEM_NUM),
                                  a1, VEC_ELEM_NUM);
    poly = __riscv_vfmul_vv_f32m4(poly, t, VEC_ELEM_NUM);

    fixed_vfloat32m4_t exp_val =
        FP32Vec16(__riscv_vfneg_v_f32m4(
                      __riscv_vfmul_vv_f32m4(abs_x, abs_x, VEC_ELEM_NUM),
                      VEC_ELEM_NUM))
            .exp()
            .reg;
    fixed_vfloat32m4_t res = __riscv_vfrsub_vf_f32m4(
        __riscv_vfmul_vv_f32m4(poly, exp_val, VEC_ELEM_NUM), 1.0f,
        VEC_ELEM_NUM);

    vbool8_t mask = __riscv_vmflt_vf_f32m4_b8(reg, 0.0f, VEC_ELEM_NUM);
    return FP32Vec16(__riscv_vfneg_v_f32m4_m(mask, res, VEC_ELEM_NUM));
  }
};

// ============================================================================
// Type Traits & Global Helpers
// ============================================================================

template <typename T>
struct VecType {
  using vec_type = void;
  using vec_t = void;
};

template <typename T>
using vec_t = typename VecType<T>::vec_type;

template <>
struct VecType<float> {
  using vec_type = FP32Vec8;
  using vec_t = FP32Vec8;
};
template <>
struct VecType<c10::Half> {
  using vec_type = FP16Vec8;
  using vec_t = FP16Vec8;
};
template <>
struct VecType<c10::BFloat16> {
  using vec_type = BF16Vec8;
  using vec_t = BF16Vec8;
};

template <typename T>
void storeFP32(float v, T* ptr) {
  *ptr = v;
}
template <>
inline void storeFP32<c10::Half>(float v, c10::Half* ptr) {
  *reinterpret_cast<_Float16*>(ptr) = static_cast<_Float16>(v);
}

inline FP16Vec16::FP16Vec16(const FP32Vec16& v) {
  reg = __riscv_vfncvt_f_f_w_f16m2(v.reg, VEC_ELEM_NUM);
}
inline FP16Vec8::FP16Vec8(const FP32Vec8& v) {
  reg = __riscv_vfncvt_f_f_w_f16m1(v.reg, VEC_ELEM_NUM);
}
inline FP32Vec16::FP32Vec16(const FP16Vec16& v) {
  reg = __riscv_vfwcvt_f_f_v_f32m4(v.reg, VEC_ELEM_NUM);
}
inline void fma(FP32Vec16& acc, const FP32Vec16& a, const FP32Vec16& b) {
  acc = acc.fma(a, b);
}

#ifdef RISCV_BF16_SUPPORT
template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  *ptr = static_cast<__bf16>(v);
};
inline BF16Vec8::BF16Vec8(const FP32Vec8& v)
    : reg(__riscv_vfncvtbf16_f_f_w_bf16m1(v.reg, VEC_ELEM_NUM)) {};
inline BF16Vec16::BF16Vec16(const FP32Vec16& v)
    : reg(__riscv_vfncvtbf16_f_f_w_bf16m2(v.reg, VEC_ELEM_NUM)) {};
#else
template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  uint32_t val;
  std::memcpy(&val, &v, 4);
  *reinterpret_cast<uint16_t*>(ptr) = static_cast<uint16_t>(val >> 16);
}
inline BF16Vec8::BF16Vec8(const FP32Vec8& v) : reg_fp32(v.reg) {}
inline BF16Vec16::BF16Vec16(const FP32Vec16& v) : reg_fp32(v.reg) {}
#endif

inline void prefetch(const void* addr) { __builtin_prefetch(addr, 0, 1); }

}  // namespace vec_op

#ifndef CPU_KERNEL_GUARD_IN
  #define CPU_KERNEL_GUARD_IN(NAME)
#endif

#ifndef CPU_KERNEL_GUARD_OUT
  #define CPU_KERNEL_GUARD_OUT(NAME)
#endif

#endif  // CPU_TYPES_RISCV_HPP