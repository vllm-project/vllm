#ifndef CPU_TYPES_RISCV_IMPL_HPP
#define CPU_TYPES_RISCV_IMPL_HPP

// Shared implementation of RVV vector-type wrapper classes.
// This file is VLEN-independent: it uses the semantic type names and
// RVVI() intrinsic macros from cpu_types_riscv_defs.hpp.
//
// DO NOT include this file directly; include cpu_types_riscv.hpp instead.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <torch/all.h>
namespace vec_op {

// FP8 KV cache is not supported on RISC-V. These tag types and the
// corresponding BF16Vec32 stub constructors below exist solely so that
// templates referencing vec_op::fp8_*_tag in their bodies (e.g. in
// cpu_attn_vec.hpp) compile under GCC's -Wtemplate-body lookup. The
// stubs are never instantiated by CPU_ATTN_DISPATCH on __riscv.
struct fp8_e4m3_tag {};
struct fp8_e5m2_tag {};

// BFloat16 is always supported on RISC-V: natively when __riscv_zvfbfmin
// is defined (compiler-provided when -march includes zvfbfmin), otherwise
// via the FP32-simulation fallback path.
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

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
  fixed_fp16x8_t reg;

  explicit FP16Vec8(const void* ptr)
      : reg(RVVI(__riscv_vle16_v_f16, LMUL_128)(
            static_cast<const _Float16*>(ptr), VEC_ELEM_NUM)) {};

  explicit FP16Vec8(const FP32Vec8&);

  void save(void* ptr) const {
    RVVI(__riscv_vse16_v_f16, LMUL_128)(static_cast<_Float16*>(ptr), reg,
                                        VEC_ELEM_NUM);
  }
  void save(void* ptr, int elem_num) const {
    RVVI(__riscv_vse16_v_f16, LMUL_128)(static_cast<_Float16*>(ptr), reg,
                                        elem_num);
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(_Float16);
    RVVI(__riscv_vsse16_v_f16, LMUL_128)(static_cast<_Float16*>(ptr),
                                         byte_stride, reg, VEC_ELEM_NUM);
  }
};

struct FP16Vec16 : public Vec<FP16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  fixed_fp16x16_t reg;

  explicit FP16Vec16(const void* ptr)
      : reg(RVVI(__riscv_vle16_v_f16, LMUL_256)(
            static_cast<const _Float16*>(ptr), VEC_ELEM_NUM)) {};

  explicit FP16Vec16(const FP32Vec16& vec);

  void save(void* ptr) const {
    RVVI(__riscv_vse16_v_f16, LMUL_256)(static_cast<_Float16*>(ptr), reg,
                                        VEC_ELEM_NUM);
  }
  void save(void* ptr, int elem_num) const {
    RVVI(__riscv_vse16_v_f16, LMUL_256)(static_cast<_Float16*>(ptr), reg,
                                        elem_num);
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(_Float16);
    RVVI(__riscv_vsse16_v_f16, LMUL_256)(static_cast<_Float16*>(ptr),
                                         byte_stride, reg, VEC_ELEM_NUM);
  }
};

// ============================================================================
// BF16 Implementation
// ============================================================================

#ifdef __riscv_zvfbfmin

FORCE_INLINE fixed_u16x8_t bf16_to_u16(fixed_bf16x8_t v) {
  return RVVI4(__riscv_vreinterpret_v_bf16, LMUL_128, _u16, LMUL_128)(v);
}
FORCE_INLINE fixed_u16x16_t bf16_to_u16(fixed_bf16x16_t v) {
  return RVVI4(__riscv_vreinterpret_v_bf16, LMUL_256, _u16, LMUL_256)(v);
}
FORCE_INLINE fixed_u16x32_t bf16_to_u16(fixed_bf16x32_t v) {
  return RVVI4(__riscv_vreinterpret_v_bf16, LMUL_512, _u16, LMUL_512)(v);
}

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  fixed_bf16x8_t reg;

  explicit BF16Vec8(const void* ptr)
      : reg(RVVI4(__riscv_vreinterpret_v_u16, LMUL_128, _bf16,
                  LMUL_128)(RVVI(__riscv_vle16_v_u16, LMUL_128)(
            reinterpret_cast<const uint16_t*>(ptr), VEC_ELEM_NUM))) {};

  explicit BF16Vec8(fixed_bf16x8_t data) : reg(data) {};
  explicit BF16Vec8(const FP32Vec8&);

  void save(void* ptr) const {
    RVVI(__riscv_vse16_v_u16, LMUL_128)(reinterpret_cast<uint16_t*>(ptr),
                                        bf16_to_u16(reg), VEC_ELEM_NUM);
  }
  void save(void* ptr, int elem_num) const {
    RVVI(__riscv_vse16_v_u16, LMUL_128)(reinterpret_cast<uint16_t*>(ptr),
                                        bf16_to_u16(reg), elem_num);
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(uint16_t);
    RVVI(__riscv_vsse16_v_u16, LMUL_128)(reinterpret_cast<uint16_t*>(ptr),
                                         byte_stride, bf16_to_u16(reg),
                                         VEC_ELEM_NUM);
  }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  fixed_bf16x16_t reg;

  explicit BF16Vec16(const void* ptr)
      : reg(RVVI4(__riscv_vreinterpret_v_u16, LMUL_256, _bf16,
                  LMUL_256)(RVVI(__riscv_vle16_v_u16, LMUL_256)(
            reinterpret_cast<const uint16_t*>(ptr), VEC_ELEM_NUM))) {};

  explicit BF16Vec16(fixed_bf16x16_t data) : reg(data) {};
  explicit BF16Vec16(const FP32Vec16&);

  void save(void* ptr) const {
    RVVI(__riscv_vse16_v_u16, LMUL_256)(reinterpret_cast<uint16_t*>(ptr),
                                        bf16_to_u16(reg), VEC_ELEM_NUM);
  }
  void save(void* ptr, int elem_num) const {
    RVVI(__riscv_vse16_v_u16, LMUL_256)(reinterpret_cast<uint16_t*>(ptr),
                                        bf16_to_u16(reg), elem_num);
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(uint16_t);
    RVVI(__riscv_vsse16_v_u16, LMUL_256)(reinterpret_cast<uint16_t*>(ptr),
                                         byte_stride, bf16_to_u16(reg),
                                         VEC_ELEM_NUM);
  }
};

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;
  fixed_bf16x32_t reg;

  explicit BF16Vec32(const void* ptr)
      : reg(RVVI4(__riscv_vreinterpret_v_u16, LMUL_512, _bf16,
                  LMUL_512)(RVVI(__riscv_vle16_v_u16, LMUL_512)(
            reinterpret_cast<const uint16_t*>(ptr), VEC_ELEM_NUM))) {};

  explicit BF16Vec32(fixed_bf16x32_t data) : reg(data) {};

  // FP8 KV cache stubs: never instantiated on RISC-V (CPU_ATTN_DISPATCH
  // omits FP8 cases on __riscv); exist only so name lookup succeeds.
  explicit BF16Vec32(const uint8_t* ptr, fp8_e4m3_tag)
      : BF16Vec32(static_cast<const void*>(ptr)) {}
  explicit BF16Vec32(const uint8_t* ptr, fp8_e5m2_tag)
      : BF16Vec32(static_cast<const void*>(ptr)) {}

  explicit BF16Vec32(const BF16Vec8& v) {
    fixed_u16x8_t u16_val = bf16_to_u16(v.reg);
    fixed_u16x32_t u16_combined =
        RVVI4(__riscv_vcreate_v_u16, LMUL_128, _u16, LMUL_512)(
            u16_val, u16_val, u16_val, u16_val);
    reg = RVVI4(__riscv_vreinterpret_v_u16, LMUL_512, _bf16,
                LMUL_512)(u16_combined);
  };

  void save(void* ptr) const {
    RVVI(__riscv_vse16_v_u16, LMUL_512)(reinterpret_cast<uint16_t*>(ptr),
                                        bf16_to_u16(reg), VEC_ELEM_NUM);
  }
  void save(void* ptr, int elem_num) const {
    RVVI(__riscv_vse16_v_u16, LMUL_512)(reinterpret_cast<uint16_t*>(ptr),
                                        bf16_to_u16(reg), elem_num);
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(uint16_t);
    RVVI(__riscv_vsse16_v_u16, LMUL_512)(reinterpret_cast<uint16_t*>(ptr),
                                         byte_stride, bf16_to_u16(reg),
                                         VEC_ELEM_NUM);
  }
};

#else
// ============================================================================
// BF16 Fallback Implementation (FP32 Simulation)
// ============================================================================

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  fixed_fp32x8_t reg_fp32;
  explicit BF16Vec8(const void* ptr) {
    const uint16_t* u16 = static_cast<const uint16_t*>(ptr);
    float tmp[8];
    for (int i = 0; i < 8; ++i) {
      uint32_t v = static_cast<uint32_t>(u16[i]) << 16;
      std::memcpy(&tmp[i], &v, 4);
    }
    reg_fp32 = RVVI(__riscv_vle32_v_f32, LMUL_256)(tmp, 8);
  }
  explicit BF16Vec8(const FP32Vec8&);
  void save(void* ptr) const {
    float tmp[8];
    RVVI(__riscv_vse32_v_f32, LMUL_256)(tmp, reg_fp32, 8);
    uint16_t* u16 = static_cast<uint16_t*>(ptr);
    for (int i = 0; i < 8; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      u16[i] = static_cast<uint16_t>(v >> 16);
    }
  }
  void save(void* ptr, int elem_num) const {
    float tmp[8];
    RVVI(__riscv_vse32_v_f32, LMUL_256)(tmp, reg_fp32, 8);
    uint16_t* u16 = static_cast<uint16_t*>(ptr);
    for (int i = 0; i < elem_num; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      u16[i] = static_cast<uint16_t>(v >> 16);
    }
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    float tmp[8];
    RVVI(__riscv_vse32_v_f32, LMUL_256)(tmp, reg_fp32, 8);
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
  fixed_fp32x16_t reg_fp32;
  explicit BF16Vec16(const void* ptr) {
    const uint16_t* u16 = static_cast<const uint16_t*>(ptr);
    float tmp[16];
    for (int i = 0; i < 16; ++i) {
      uint32_t v = static_cast<uint32_t>(u16[i]) << 16;
      std::memcpy(&tmp[i], &v, 4);
    }
    reg_fp32 = RVVI(__riscv_vle32_v_f32, LMUL_512)(tmp, 16);
  }
  explicit BF16Vec16(const FP32Vec16&);
  void save(void* ptr) const {
    float tmp[16];
    RVVI(__riscv_vse32_v_f32, LMUL_512)(tmp, reg_fp32, 16);
    uint16_t* u16 = static_cast<uint16_t*>(ptr);
    for (int i = 0; i < 16; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      u16[i] = static_cast<uint16_t>(v >> 16);
    }
  }
  void save(void* ptr, int elem_num) const {
    float tmp[16];
    RVVI(__riscv_vse32_v_f32, LMUL_512)(tmp, reg_fp32, 16);
    uint16_t* u16 = static_cast<uint16_t*>(ptr);
    for (int i = 0; i < elem_num; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      u16[i] = static_cast<uint16_t>(v >> 16);
    }
  }
  void save_strided(void* ptr, ptrdiff_t stride) const {
    float tmp[16];
    RVVI(__riscv_vse32_v_f32, LMUL_512)(tmp, reg_fp32, 16);
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
  fixed_fp32x32_t reg_fp32;

  explicit BF16Vec32(const void* ptr) {
    const uint16_t* u16 = static_cast<const uint16_t*>(ptr);
    float tmp[32];
    for (int i = 0; i < 32; ++i) {
      uint32_t v = static_cast<uint32_t>(u16[i]) << 16;
      std::memcpy(&tmp[i], &v, 4);
    }
    reg_fp32 = RVVI(__riscv_vle32_v_f32, LMUL_1024)(tmp, 32);
  }

  // FP8 KV cache stubs: never instantiated on RISC-V (CPU_ATTN_DISPATCH
  // omits FP8 cases on __riscv); exist only so name lookup succeeds.
  explicit BF16Vec32(const uint8_t* ptr, fp8_e4m3_tag)
      : BF16Vec32(static_cast<const void*>(ptr)) {}
  explicit BF16Vec32(const uint8_t* ptr, fp8_e5m2_tag)
      : BF16Vec32(static_cast<const void*>(ptr)) {}

  explicit BF16Vec32(const BF16Vec8& v) {
    float tmp_small[8];
    RVVI(__riscv_vse32_v_f32, LMUL_256)(tmp_small, v.reg_fp32, 8);
    float tmp_large[32];
    for (int i = 0; i < 4; ++i) {
      std::memcpy(tmp_large + (i * 8), tmp_small, 8 * sizeof(float));
    }
    reg_fp32 = RVVI(__riscv_vle32_v_f32, LMUL_1024)(tmp_large, 32);
  }

  void save(void* ptr) const {
    float tmp[32];
    RVVI(__riscv_vse32_v_f32, LMUL_1024)(tmp, reg_fp32, 32);
    uint16_t* u16 = static_cast<uint16_t*>(ptr);
    for (int i = 0; i < 32; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      u16[i] = static_cast<uint16_t>(v >> 16);
    }
  }

  void save(void* ptr, int elem_num) const {
    float tmp[32];
    RVVI(__riscv_vse32_v_f32, LMUL_1024)(tmp, reg_fp32, 32);
    uint16_t* u16 = static_cast<uint16_t*>(ptr);
    for (int i = 0; i < elem_num; ++i) {
      uint32_t v;
      std::memcpy(&v, &tmp[i], 4);
      u16[i] = static_cast<uint16_t>(v >> 16);
    }
  }

  void save_strided(void* ptr, ptrdiff_t stride) const {
    float tmp[32];
    RVVI(__riscv_vse32_v_f32, LMUL_1024)(tmp, reg_fp32, 32);
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
  fixed_fp32x4_t reg;
  explicit FP32Vec4(float v)
      : reg(RVVI(__riscv_vfmv_v_f_f32, LMUL_128)(v, VEC_ELEM_NUM)) {};
  explicit FP32Vec4()
      : reg(RVVI(__riscv_vfmv_v_f_f32, LMUL_128)(0.0f, VEC_ELEM_NUM)) {};
  explicit FP32Vec4(const float* ptr)
      : reg(RVVI(__riscv_vle32_v_f32, LMUL_128)(ptr, VEC_ELEM_NUM)) {};
  explicit FP32Vec4(fixed_fp32x4_t data) : reg(data) {};
  explicit FP32Vec4(const FP32Vec4& data) : reg(data.reg) {};
  void save(float* ptr) const {
    RVVI(__riscv_vse32_v_f32, LMUL_128)(ptr, reg, VEC_ELEM_NUM);
  }
  void save(float* ptr, int elem_num) const {
    RVVI(__riscv_vse32_v_f32, LMUL_128)(ptr, reg, elem_num);
  }
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  fixed_fp32x8_t reg;

  explicit FP32Vec8(float v)
      : reg(RVVI(__riscv_vfmv_v_f_f32, LMUL_256)(v, VEC_ELEM_NUM)) {};
  explicit FP32Vec8()
      : reg(RVVI(__riscv_vfmv_v_f_f32, LMUL_256)(0.0f, VEC_ELEM_NUM)) {};
  explicit FP32Vec8(const float* ptr)
      : reg(RVVI(__riscv_vle32_v_f32, LMUL_256)(ptr, VEC_ELEM_NUM)) {};
  explicit FP32Vec8(fixed_fp32x8_t data) : reg(data) {};
  explicit FP32Vec8(const FP32Vec8& data) : reg(data.reg) {};
  explicit FP32Vec8(const FP16Vec8& v)
      : reg(RVVI(__riscv_vfwcvt_f_f_v_f32, LMUL_256)(v.reg, VEC_ELEM_NUM)) {};
  explicit FP32Vec8(fixed_fp16x8_t v)
      : reg(RVVI(__riscv_vfwcvt_f_f_v_f32, LMUL_256)(v, VEC_ELEM_NUM)) {};

#ifdef __riscv_zvfbfmin
  explicit FP32Vec8(fixed_bf16x8_t v)
      : reg(RVVI(__riscv_vfwcvtbf16_f_f_v_f32, LMUL_256)(v, VEC_ELEM_NUM)) {};
  explicit FP32Vec8(const BF16Vec8& v)
      : reg(RVVI(__riscv_vfwcvtbf16_f_f_v_f32, LMUL_256)(v.reg, VEC_ELEM_NUM)) {
        };
#else
  explicit FP32Vec8(const BF16Vec8& v) : reg(v.reg_fp32) {};
#endif

  float reduce_sum() const {
    rvv_f32_accum_t scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    scalar = RVVI3(__riscv_vfredusum_vs_f32, LMUL_256, _f32m1)(reg, scalar,
                                                               VEC_ELEM_NUM);
    return __riscv_vfmv_f_s_f32m1_f32(scalar);
  }

  FP32Vec8 operator*(const FP32Vec8& b) const {
    return FP32Vec8(
        RVVI(__riscv_vfmul_vv_f32, LMUL_256)(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec8 operator+(const FP32Vec8& b) const {
    return FP32Vec8(
        RVVI(__riscv_vfadd_vv_f32, LMUL_256)(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec8 operator-(const FP32Vec8& b) const {
    return FP32Vec8(
        RVVI(__riscv_vfsub_vv_f32, LMUL_256)(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec8 operator/(const FP32Vec8& b) const {
    return FP32Vec8(
        RVVI(__riscv_vfdiv_vv_f32, LMUL_256)(reg, b.reg, VEC_ELEM_NUM));
  }

  FP32Vec8 min(const FP32Vec8& b) const {
    return FP32Vec8(
        RVVI(__riscv_vfmin_vv_f32, LMUL_256)(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec8 max(const FP32Vec8& b) const {
    return FP32Vec8(
        RVVI(__riscv_vfmax_vv_f32, LMUL_256)(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec8 abs() const {
    return FP32Vec8(RVVI(__riscv_vfabs_v_f32, LMUL_256)(reg, VEC_ELEM_NUM));
  }

  FP32Vec8 min(const FP32Vec8& b, int elem_num) const {
    return FP32Vec8(RVVI(__riscv_vfmin_vv_f32, LMUL_256)(reg, b.reg, elem_num));
  }
  FP32Vec8 max(const FP32Vec8& b, int elem_num) const {
    return FP32Vec8(RVVI(__riscv_vfmax_vv_f32, LMUL_256)(reg, b.reg, elem_num));
  }

  FP32Vec8 clamp(const FP32Vec8& min_v, const FP32Vec8& max_v) const {
    fixed_fp32x8_t temp =
        RVVI(__riscv_vfmax_vv_f32, LMUL_256)(min_v.reg, reg, VEC_ELEM_NUM);
    return FP32Vec8(
        RVVI(__riscv_vfmin_vv_f32, LMUL_256)(max_v.reg, temp, VEC_ELEM_NUM));
  }

  void save(float* ptr) const {
    RVVI(__riscv_vse32_v_f32, LMUL_256)(ptr, reg, VEC_ELEM_NUM);
  }
  void save(float* ptr, int elem_num) const {
    RVVI(__riscv_vse32_v_f32, LMUL_256)(ptr, reg, elem_num);
  }
  void save_strided(float* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(float);
    RVVI(__riscv_vsse32_v_f32, LMUL_256)(ptr, byte_stride, reg, VEC_ELEM_NUM);
  }

  FP32Vec8 exp() const {
    // Clamp input to prevent NaN: exp(-inf) must return 0, not NaN.
    // Without clamping, -inf * 0.0 = NaN in the final poly * scale step.
    // Matches the clamping strategy used by x86 AVX-512 and ARM NEON.
    constexpr float exp_lo = -87.3365447505f;  // ln(FLT_MIN)
    constexpr float exp_hi = 88.7228391117f;   // ln(FLT_MAX)
    fixed_fp32x8_t x = RVVI(__riscv_vfmin_vf_f32, LMUL_256)(
        RVVI(__riscv_vfmax_vf_f32, LMUL_256)(reg, exp_lo, VEC_ELEM_NUM), exp_hi,
        VEC_ELEM_NUM);

    const float inv_ln2 = 1.44269504088896341f;
    fixed_fp32x8_t x_scaled =
        RVVI(__riscv_vfmul_vf_f32, LMUL_256)(x, inv_ln2, VEC_ELEM_NUM);
    fixed_i32x8_t n_int =
        RVVI(__riscv_vfcvt_x_f_v_i32, LMUL_256)(x_scaled, VEC_ELEM_NUM);
    fixed_fp32x8_t n_float =
        RVVI(__riscv_vfcvt_f_x_v_f32, LMUL_256)(n_int, VEC_ELEM_NUM);

    fixed_fp32x8_t r =
        RVVI(__riscv_vfsub_vv_f32, LMUL_256)(x_scaled, n_float, VEC_ELEM_NUM);

    fixed_fp32x8_t poly =
        RVVI(__riscv_vfmv_v_f_f32, LMUL_256)(0.001333355810164f, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfmul_vv_f32, LMUL_256)(poly, r, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_256)(poly, 0.009618129107628f,
                                                VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfmul_vv_f32, LMUL_256)(poly, r, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_256)(poly, 0.055504108664821f,
                                                VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfmul_vv_f32, LMUL_256)(poly, r, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_256)(poly, 0.240226506959101f,
                                                VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfmul_vv_f32, LMUL_256)(poly, r, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_256)(poly, 0.693147180559945f,
                                                VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfmul_vv_f32, LMUL_256)(poly, r, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_256)(poly, 1.0f, VEC_ELEM_NUM);

    fixed_i32x8_t biased_exp =
        RVVI(__riscv_vadd_vx_i32, LMUL_256)(n_int, 127, VEC_ELEM_NUM);
    biased_exp =
        RVVI(__riscv_vmax_vx_i32, LMUL_256)(biased_exp, 0, VEC_ELEM_NUM);
    fixed_i32x8_t exponent_bits =
        RVVI(__riscv_vsll_vx_i32, LMUL_256)(biased_exp, 23, VEC_ELEM_NUM);
    fixed_fp32x8_t scale = RVVI4(__riscv_vreinterpret_v_i32, LMUL_256, _f32,
                                 LMUL_256)(exponent_bits);

    return FP32Vec8(
        RVVI(__riscv_vfmul_vv_f32, LMUL_256)(poly, scale, VEC_ELEM_NUM));
  }

  FP32Vec8 tanh() const {
    fixed_fp32x8_t x_clamped = RVVI(__riscv_vfmin_vf_f32, LMUL_256)(
        RVVI(__riscv_vfmax_vf_f32, LMUL_256)(reg, -9.0f, VEC_ELEM_NUM), 9.0f,
        VEC_ELEM_NUM);
    fixed_fp32x8_t x2 =
        RVVI(__riscv_vfmul_vf_f32, LMUL_256)(x_clamped, 2.0f, VEC_ELEM_NUM);
    FP32Vec8 exp_val = FP32Vec8(x2).exp();
    fixed_fp32x8_t num =
        RVVI(__riscv_vfsub_vf_f32, LMUL_256)(exp_val.reg, 1.0f, VEC_ELEM_NUM);
    fixed_fp32x8_t den =
        RVVI(__riscv_vfadd_vf_f32, LMUL_256)(exp_val.reg, 1.0f, VEC_ELEM_NUM);
    return FP32Vec8(
        RVVI(__riscv_vfdiv_vv_f32, LMUL_256)(num, den, VEC_ELEM_NUM));
  }

  FP32Vec8 er() const {
    const float p = 0.3275911f, a1 = 0.254829592f, a2 = -0.284496736f,
                a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
    fixed_fp32x8_t abs_x =
        RVVI(__riscv_vfabs_v_f32, LMUL_256)(reg, VEC_ELEM_NUM);

    fixed_fp32x8_t t = RVVI(__riscv_vfadd_vf_f32, LMUL_256)(
        RVVI(__riscv_vfmul_vf_f32, LMUL_256)(abs_x, p, VEC_ELEM_NUM), 1.0f,
        VEC_ELEM_NUM);
    t = RVVI(__riscv_vfrdiv_vf_f32, LMUL_256)(t, 1.0f, VEC_ELEM_NUM);

    fixed_fp32x8_t poly =
        RVVI(__riscv_vfmv_v_f_f32, LMUL_256)(a5, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_256)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_256)(poly, t, VEC_ELEM_NUM), a4,
        VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_256)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_256)(poly, t, VEC_ELEM_NUM), a3,
        VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_256)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_256)(poly, t, VEC_ELEM_NUM), a2,
        VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_256)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_256)(poly, t, VEC_ELEM_NUM), a1,
        VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfmul_vv_f32, LMUL_256)(poly, t, VEC_ELEM_NUM);

    fixed_fp32x8_t exp_val = FP32Vec8(RVVI(__riscv_vfneg_v_f32, LMUL_256)(
                                          RVVI(__riscv_vfmul_vv_f32, LMUL_256)(
                                              abs_x, abs_x, VEC_ELEM_NUM),
                                          VEC_ELEM_NUM))
                                 .exp()
                                 .reg;
    fixed_fp32x8_t res = RVVI(__riscv_vfrsub_vf_f32, LMUL_256)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_256)(poly, exp_val, VEC_ELEM_NUM), 1.0f,
        VEC_ELEM_NUM);

    rvv_mask_f32x8_t mask = RVVIB(__riscv_vmflt_vf_f32, LMUL_256, BOOL_256)(
        reg, 0.0f, VEC_ELEM_NUM);
    return FP32Vec8(
        RVVI3(__riscv_vfneg_v_f32, LMUL_256, _m)(mask, res, VEC_ELEM_NUM));
  }
};

struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  fixed_fp32x16_t reg;

  explicit FP32Vec16(float v)
      : reg(RVVI(__riscv_vfmv_v_f_f32, LMUL_512)(v, VEC_ELEM_NUM)) {};
  explicit FP32Vec16()
      : reg(RVVI(__riscv_vfmv_v_f_f32, LMUL_512)(0.0f, VEC_ELEM_NUM)) {};
  explicit FP32Vec16(const float* ptr)
      : reg(RVVI(__riscv_vle32_v_f32, LMUL_512)(ptr, VEC_ELEM_NUM)) {};
  explicit FP32Vec16(fixed_fp32x16_t data) : reg(data) {};
  explicit FP32Vec16(const FP32Vec8& data)
      : reg(RVVI4(__riscv_vcreate_v_f32, LMUL_256, _f32, LMUL_512)(
            data.reg, data.reg)) {};
  explicit FP32Vec16(const FP32Vec16& data) : reg(data.reg) {};
  explicit FP32Vec16(const FP16Vec16& v);

#ifdef __riscv_zvfbfmin
  explicit FP32Vec16(fixed_bf16x16_t v)
      : reg(RVVI(__riscv_vfwcvtbf16_f_f_v_f32, LMUL_512)(v, VEC_ELEM_NUM)) {};
  explicit FP32Vec16(const BF16Vec16& v)
      : reg(RVVI(__riscv_vfwcvtbf16_f_f_v_f32, LMUL_512)(v.reg, VEC_ELEM_NUM)) {
        };
#else
  explicit FP32Vec16(const BF16Vec16& v) : reg(v.reg_fp32) {};
#endif

  FP32Vec16 operator+(const FP32Vec16& b) const {
    return FP32Vec16(
        RVVI(__riscv_vfadd_vv_f32, LMUL_512)(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(
        RVVI(__riscv_vfsub_vv_f32, LMUL_512)(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec16 operator*(const FP32Vec16& b) const {
    return FP32Vec16(
        RVVI(__riscv_vfmul_vv_f32, LMUL_512)(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(
        RVVI(__riscv_vfdiv_vv_f32, LMUL_512)(reg, b.reg, VEC_ELEM_NUM));
  }

  FP32Vec16 fma(const FP32Vec16& a, const FP32Vec16& b) const {
    return FP32Vec16(
        RVVI(__riscv_vfmacc_vv_f32, LMUL_512)(reg, a.reg, b.reg, VEC_ELEM_NUM));
  }

  float reduce_sum() const {
    rvv_f32_accum_t scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    scalar = RVVI3(__riscv_vfredusum_vs_f32, LMUL_512, _f32m1)(reg, scalar,
                                                               VEC_ELEM_NUM);
    return __riscv_vfmv_f_s_f32m1_f32(scalar);
  }

  float reduce_max() const {
    rvv_f32_accum_t scalar =
        __riscv_vfmv_s_f_f32m1(std::numeric_limits<float>::lowest(), 1);
    scalar = RVVI3(__riscv_vfredmax_vs_f32, LMUL_512, _f32m1)(reg, scalar,
                                                              VEC_ELEM_NUM);
    return __riscv_vfmv_f_s_f32m1_f32(scalar);
  }

  float reduce_min() const {
    rvv_f32_accum_t scalar =
        __riscv_vfmv_s_f_f32m1(std::numeric_limits<float>::max(), 1);
    scalar = RVVI3(__riscv_vfredmin_vs_f32, LMUL_512, _f32m1)(reg, scalar,
                                                              VEC_ELEM_NUM);
    return __riscv_vfmv_f_s_f32m1_f32(scalar);
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);
    const int start = idx * group_size;
    auto indices = RVVI(__riscv_vid_v_u32, LMUL_512)(VEC_ELEM_NUM);
    rvv_mask_f32x16_t mask = RVVI(__riscv_vmand_mm_, BOOL_512)(
        RVVIB(__riscv_vmsgeu_vx_u32, LMUL_512, BOOL_512)(indices, start,
                                                         VEC_ELEM_NUM),
        RVVIB(__riscv_vmsltu_vx_u32, LMUL_512, BOOL_512)(
            indices, start + group_size, VEC_ELEM_NUM),
        VEC_ELEM_NUM);
    rvv_f32_accum_t scalar = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    scalar = RVVI3(__riscv_vfredusum_vs_f32, LMUL_512, _f32m1_m)(
        mask, reg, scalar, VEC_ELEM_NUM);
    return __riscv_vfmv_f_s_f32m1_f32(scalar);
  };

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(
        RVVI(__riscv_vfmax_vv_f32, LMUL_512)(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec16 min(const FP32Vec16& b) const {
    return FP32Vec16(
        RVVI(__riscv_vfmin_vv_f32, LMUL_512)(reg, b.reg, VEC_ELEM_NUM));
  }
  FP32Vec16 abs() const {
    return FP32Vec16(RVVI(__riscv_vfabs_v_f32, LMUL_512)(reg, VEC_ELEM_NUM));
  }

  FP32Vec16 clamp(const FP32Vec16& min_v, const FP32Vec16& max_v) const {
    return FP32Vec16(RVVI(__riscv_vfmin_vv_f32, LMUL_512)(
        max_v.reg,
        RVVI(__riscv_vfmax_vv_f32, LMUL_512)(min_v.reg, reg, VEC_ELEM_NUM),
        VEC_ELEM_NUM));
  }

  void save(float* ptr) const {
    RVVI(__riscv_vse32_v_f32, LMUL_512)(ptr, reg, VEC_ELEM_NUM);
  }
  void save(float* ptr, int elem_num) const {
    RVVI(__riscv_vse32_v_f32, LMUL_512)(ptr, reg, elem_num);
  }
  void save_strided(float* ptr, ptrdiff_t stride) const {
    ptrdiff_t byte_stride = stride * sizeof(float);
    RVVI(__riscv_vsse32_v_f32, LMUL_512)(ptr, byte_stride, reg, VEC_ELEM_NUM);
  }

  FP32Vec16 exp() const {
    // Clamp input to prevent NaN: exp(-inf) must return 0, not NaN.
    // Without clamping, -inf * 0.0 = NaN in the final poly * scale step.
    // Matches the clamping strategy used by x86 AVX-512 and ARM NEON.
    constexpr float exp_lo = -87.3365447505f;  // ln(FLT_MIN)
    constexpr float exp_hi = 88.7228391117f;   // ln(FLT_MAX)
    fixed_fp32x16_t x = RVVI(__riscv_vfmin_vf_f32, LMUL_512)(
        RVVI(__riscv_vfmax_vf_f32, LMUL_512)(reg, exp_lo, VEC_ELEM_NUM), exp_hi,
        VEC_ELEM_NUM);

    const float inv_ln2 = 1.44269504088896341f;
    fixed_fp32x16_t x_scaled =
        RVVI(__riscv_vfmul_vf_f32, LMUL_512)(x, inv_ln2, VEC_ELEM_NUM);
    fixed_i32x16_t n_int =
        RVVI(__riscv_vfcvt_x_f_v_i32, LMUL_512)(x_scaled, VEC_ELEM_NUM);
    fixed_fp32x16_t n_float =
        RVVI(__riscv_vfcvt_f_x_v_f32, LMUL_512)(n_int, VEC_ELEM_NUM);
    fixed_fp32x16_t r =
        RVVI(__riscv_vfsub_vv_f32, LMUL_512)(x_scaled, n_float, VEC_ELEM_NUM);

    fixed_fp32x16_t poly =
        RVVI(__riscv_vfmv_v_f_f32, LMUL_512)(0.001333355810164f, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_512)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_512)(poly, r, VEC_ELEM_NUM),
        0.009618129107628f, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_512)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_512)(poly, r, VEC_ELEM_NUM),
        0.055504108664821f, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_512)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_512)(poly, r, VEC_ELEM_NUM),
        0.240226506959101f, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_512)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_512)(poly, r, VEC_ELEM_NUM),
        0.693147180559945f, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_512)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_512)(poly, r, VEC_ELEM_NUM), 1.0f,
        VEC_ELEM_NUM);

    fixed_i32x16_t biased_exp = RVVI(__riscv_vmax_vx_i32, LMUL_512)(
        RVVI(__riscv_vadd_vx_i32, LMUL_512)(n_int, 127, VEC_ELEM_NUM), 0,
        VEC_ELEM_NUM);
    fixed_fp32x16_t scale =
        RVVI4(__riscv_vreinterpret_v_i32, LMUL_512, _f32, LMUL_512)(
            RVVI(__riscv_vsll_vx_i32, LMUL_512)(biased_exp, 23, VEC_ELEM_NUM));

    return FP32Vec16(
        RVVI(__riscv_vfmul_vv_f32, LMUL_512)(poly, scale, VEC_ELEM_NUM));
  }

  FP32Vec16 tanh() const {
    fixed_fp32x16_t x_clamped = RVVI(__riscv_vfmin_vf_f32, LMUL_512)(
        RVVI(__riscv_vfmax_vf_f32, LMUL_512)(reg, -9.0f, VEC_ELEM_NUM), 9.0f,
        VEC_ELEM_NUM);
    FP32Vec16 exp_val = FP32Vec16(RVVI(__riscv_vfmul_vf_f32, LMUL_512)(
                                      x_clamped, 2.0f, VEC_ELEM_NUM))
                            .exp();
    return FP32Vec16(RVVI(__riscv_vfdiv_vv_f32, LMUL_512)(
        RVVI(__riscv_vfsub_vf_f32, LMUL_512)(exp_val.reg, 1.0f, VEC_ELEM_NUM),
        RVVI(__riscv_vfadd_vf_f32, LMUL_512)(exp_val.reg, 1.0f, VEC_ELEM_NUM),
        VEC_ELEM_NUM));
  }

  FP32Vec16 er() const {
    const float p = 0.3275911f, a1 = 0.254829592f, a2 = -0.284496736f,
                a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
    fixed_fp32x16_t abs_x =
        RVVI(__riscv_vfabs_v_f32, LMUL_512)(reg, VEC_ELEM_NUM);
    fixed_fp32x16_t t = RVVI(__riscv_vfrdiv_vf_f32, LMUL_512)(
        RVVI(__riscv_vfadd_vf_f32, LMUL_512)(
            RVVI(__riscv_vfmul_vf_f32, LMUL_512)(abs_x, p, VEC_ELEM_NUM), 1.0f,
            VEC_ELEM_NUM),
        1.0f, VEC_ELEM_NUM);

    fixed_fp32x16_t poly =
        RVVI(__riscv_vfmv_v_f_f32, LMUL_512)(a5, VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_512)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_512)(poly, t, VEC_ELEM_NUM), a4,
        VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_512)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_512)(poly, t, VEC_ELEM_NUM), a3,
        VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_512)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_512)(poly, t, VEC_ELEM_NUM), a2,
        VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfadd_vf_f32, LMUL_512)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_512)(poly, t, VEC_ELEM_NUM), a1,
        VEC_ELEM_NUM);
    poly = RVVI(__riscv_vfmul_vv_f32, LMUL_512)(poly, t, VEC_ELEM_NUM);

    fixed_fp32x16_t exp_val =
        FP32Vec16(RVVI(__riscv_vfneg_v_f32, LMUL_512)(
                      RVVI(__riscv_vfmul_vv_f32, LMUL_512)(abs_x, abs_x,
                                                           VEC_ELEM_NUM),
                      VEC_ELEM_NUM))
            .exp()
            .reg;
    fixed_fp32x16_t res = RVVI(__riscv_vfrsub_vf_f32, LMUL_512)(
        RVVI(__riscv_vfmul_vv_f32, LMUL_512)(poly, exp_val, VEC_ELEM_NUM), 1.0f,
        VEC_ELEM_NUM);

    rvv_mask_f32x16_t mask = RVVIB(__riscv_vmflt_vf_f32, LMUL_512, BOOL_512)(
        reg, 0.0f, VEC_ELEM_NUM);
    return FP32Vec16(
        RVVI3(__riscv_vfneg_v_f32, LMUL_512, _m)(mask, res, VEC_ELEM_NUM));
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
  reg = RVVI(__riscv_vfncvt_f_f_w_f16, LMUL_256)(v.reg, VEC_ELEM_NUM);
}
inline FP16Vec8::FP16Vec8(const FP32Vec8& v) {
  reg = RVVI(__riscv_vfncvt_f_f_w_f16, LMUL_128)(v.reg, VEC_ELEM_NUM);
}
inline FP32Vec16::FP32Vec16(const FP16Vec16& v) {
  reg = RVVI(__riscv_vfwcvt_f_f_v_f32, LMUL_512)(v.reg, VEC_ELEM_NUM);
}
inline void fma(FP32Vec16& acc, const FP32Vec16& a, const FP32Vec16& b) {
  acc = acc.fma(a, b);
}

#ifdef __riscv_zvfbfmin
template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  *ptr = static_cast<__bf16>(v);
};
inline BF16Vec8::BF16Vec8(const FP32Vec8& v)
    : reg(RVVI(__riscv_vfncvtbf16_f_f_w_bf16, LMUL_128)(v.reg, VEC_ELEM_NUM)) {
      };
inline BF16Vec16::BF16Vec16(const FP32Vec16& v)
    : reg(RVVI(__riscv_vfncvtbf16_f_f_w_bf16, LMUL_256)(v.reg, VEC_ELEM_NUM)) {
      };
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

#endif  // CPU_TYPES_RISCV_IMPL_HPP
