#include <riscv_vector.h>
#include <torch/all.h>
#include <cmath>

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

  fixed_vfloat16m1_t reg;

  explicit FP16Vec8(const void* ptr)
      : reg(__riscv_vle16_v_f16m1(static_cast<const _Float16*>(ptr),
                                  VEC_ELEM_NUM)) {};

  explicit FP16Vec8(const FP32Vec8&);

  void save(void* ptr) const {
    __riscv_vse16_v_f16m1(static_cast<_Float16*>(ptr), reg, VEC_ELEM_NUM);
  }
};

struct FP16Vec16 : public Vec<FP16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  fixed_vfloat16m2_t reg;

  explicit FP16Vec16(const void* ptr)
      : reg(__riscv_vle16_v_f16m2(reinterpret_cast<const _Float16*>(ptr),
                                  VEC_ELEM_NUM)) {};

  explicit FP16Vec16(const FP32Vec16& vec);

  void save(void* ptr) const {
    __riscv_vse16_v_f16m2(reinterpret_cast<_Float16*>(ptr), reg, VEC_ELEM_NUM);
  }

  void save(void* ptr, const int elem_num) const {
    vuint16m2_t index = __riscv_vid_v_u16m2(elem_num);
    vbool8_t mask = __riscv_vmsltu_vx_u16m2_b8(index, elem_num, VEC_ELEM_NUM);
    __riscv_vse16_v_f16m2_m(mask, reinterpret_cast<_Float16*>(ptr), reg,
                            VEC_ELEM_NUM);
  }
};

#ifdef RISCV_BF16_SUPPORT
typedef vbfloat16m1_t fixed_vbfloat16m1_t
    __attribute__((riscv_rvv_vector_bits(128)));

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  fixed_vbfloat16m1_t reg;

  explicit BF16Vec8(const void* ptr)
      : reg(*reinterpret_cast<const fixed_vbfloat16m1_t*>(ptr)) {};

  explicit BF16Vec8(fixed_vbfloat16m1_t data) : reg(data) {};

  explicit BF16Vec8(const FP32Vec8&);

  explicit BF16Vec8(fixed_vfloat32m2_t v)
      : reg(__riscv_vfncvtbf16_f_f_w_bf16m1(v, VEC_ELEM_NUM)) {};

  void save(void* ptr) const {
    *reinterpret_cast<fixed_vbfloat16m1_t*>(ptr) = reg;
  }
};

typedef vbfloat16m2_t fixed_vbfloat16m2_t
    __attribute__((riscv_rvv_vector_bits(256)));

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  fixed_vbfloat16m2_t reg;

  explicit BF16Vec16(const void* ptr)
      : reg(*reinterpret_cast<const fixed_vbfloat16m2_t*>(ptr)) {};

  explicit BF16Vec16(fixed_vbfloat16m2_t data) : reg(data) {};

  explicit BF16Vec16(const FP32Vec16&);

  explicit BF16Vec16(fixed_vfloat32m4_t v)
      : reg(__riscv_vfncvtbf16_f_f_w_bf16m2(v, VEC_ELEM_NUM)) {};

  void save(void* ptr) const {
    *reinterpret_cast<fixed_vbfloat16m2_t*>(ptr) = reg;
  };
};

typedef vbfloat16m4_t fixed_vbfloat16m4_t
    __attribute__((riscv_rvv_vector_bits(512)));

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;

  fixed_vbfloat16m4_t reg;

  explicit BF16Vec32(const void* ptr)
      : reg(*reinterpret_cast<const fixed_vbfloat16m4_t*>(ptr)) {};

  explicit BF16Vec32(fixed_vbfloat16m4_t data) : reg(data) {};

  explicit BF16Vec32(const BF16Vec8& vec8_data)
      : reg(__riscv_vcreate_v_bf16m1_bf16m4(vec8_data.reg, vec8_data.reg,
                                            vec8_data.reg, vec8_data.reg)) {};

  void save(void* ptr) const {
    *reinterpret_cast<fixed_vbfloat16m4_t*>(ptr) = reg;
  };
};
#endif

struct FP32Vec4 : public Vec<FP32Vec4> {
  constexpr static int VEC_ELEM_NUM = 4;

  union AliasReg {
    fixed_vfloat32m1_t reg;
    float values[VEC_ELEM_NUM];
  };

  fixed_vfloat32m1_t reg;

  explicit FP32Vec4(float v) : reg(__riscv_vfmv_v_f_f32m1(v, VEC_ELEM_NUM)) {};

  explicit FP32Vec4() : reg(__riscv_vfmv_v_f_f32m1(0.0f, VEC_ELEM_NUM)) {};

  explicit FP32Vec4(const float* ptr)
      : reg(__riscv_vle32_v_f32m1(ptr, VEC_ELEM_NUM)) {};

  explicit FP32Vec4(fixed_vfloat32m1_t data) : reg(data) {};

  explicit FP32Vec4(const FP32Vec4& data) : reg(data.reg) {};
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  union AliasReg {
    fixed_vfloat32m2_t reg;
    float values[VEC_ELEM_NUM];
  };

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

    float exp_vals[VEC_ELEM_NUM];
    unroll_loop<int, VEC_ELEM_NUM>(
        [&exp_vals, &ar](int i) { exp_vals[i] = expf(ar.values[i]); });
    fixed_vfloat32m2_t result = __riscv_vle32_v_f32m2(exp_vals, VEC_ELEM_NUM);

    return FP32Vec8(result);
  }

  FP32Vec8 tanh() const {
    AliasReg ar;
    ar.reg = reg;

    float tanh_vals[VEC_ELEM_NUM];
    unroll_loop<int, VEC_ELEM_NUM>(
        [&tanh_vals, &ar](int i) { tanh_vals[i] = tanhf(ar.values[i]); });
    fixed_vfloat32m2_t result = __riscv_vle32_v_f32m2(tanh_vals, VEC_ELEM_NUM);

    return FP32Vec8(result);
  }

  FP32Vec8 er() const {
    AliasReg ar;
    ar.reg = reg;

    float er_vals[VEC_ELEM_NUM];
    unroll_loop<int, VEC_ELEM_NUM>(
        [&er_vals, &ar](int i) { er_vals[i] = erf(ar.values[i]); });
    fixed_vfloat32m2_t result = __riscv_vle32_v_f32m2(er_vals, VEC_ELEM_NUM);

    return FP32Vec8(result);
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

  void save(float* ptr) const { __riscv_vse32_v_f32m2(ptr, reg, VEC_ELEM_NUM); }
};

struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  union AliasReg {
    fixed_vfloat32m4_t reg;
    float values[VEC_ELEM_NUM];
  };

  fixed_vfloat32m4_t reg;

  explicit FP32Vec16(float v) : reg(__riscv_vfmv_v_f_f32m4(v, VEC_ELEM_NUM)) {};

  explicit FP32Vec16() : reg(__riscv_vfmv_v_f_f32m4(0.0f, VEC_ELEM_NUM)) {};

  explicit FP32Vec16(const float* ptr)
      : reg(__riscv_vle32_v_f32m4(ptr, VEC_ELEM_NUM)) {};

  explicit FP32Vec16(fixed_vfloat32m4_t data) : reg(data) {};

  explicit FP32Vec16(const FP32Vec8& data)
      : reg(__riscv_vcreate_v_f32m2_f32m4(data.reg, data.reg)) {};

  explicit FP32Vec16(const FP32Vec16& data) : reg(data.reg) {};

  explicit FP32Vec16(const FP16Vec8& v) : FP32Vec16(FP32Vec8(v.reg)) {};

#ifdef RISCV_BF16_SUPPORT
  explicit FP32Vec16(fixed_vbfloat16m2_t v)
      : reg(__riscv_vfwcvtbf16_f_f_v_f32m4(v, VEC_ELEM_NUM)) {};
#endif

  explicit FP32Vec16(const FP32Vec4& data)
      : reg(__riscv_vcreate_v_f32m1_f32m4(data.reg, data.reg, data.reg,
                                          data.reg)) {};

#ifdef RISCV_BF16_SUPPORT
  explicit FP32Vec16(const BF16Vec16& v)
      : reg(__riscv_vfwcvtbf16_f_f_v_f32m4(v.reg, VEC_ELEM_NUM)) {};

  explicit FP32Vec16(const BF16Vec8& v) : FP32Vec16(FP32Vec8(v)) {};
#endif

  explicit FP32Vec16(const FP16Vec16& v)
      : reg(reg = __riscv_vfwcvt_f_f_v_f32m4(v.reg, VEC_ELEM_NUM)) {};

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

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float answer = 0;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&answer, &ar](int i) { answer += ar.values[i]; });

    return answer;
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

  void save(float* ptr) const { __riscv_vse32_v_f32m4(ptr, reg, VEC_ELEM_NUM); }
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

#ifdef RISCV_BF16_SUPPORT
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
  *reinterpret_cast<_Float16*>(ptr) = v;
}

inline FP16Vec16::FP16Vec16(const FP32Vec16& v) {
  reg = __riscv_vfncvt_f_f_w_f16m2(v.reg, VEC_ELEM_NUM);
};

inline FP16Vec8 ::FP16Vec8(const FP32Vec8& v) {
  reg = __riscv_vfncvt_f_f_w_f16m1(v.reg, VEC_ELEM_NUM);
};

inline void fma(FP32Vec16& acc, FP32Vec16& a, FP32Vec16& b) {
  acc.reg = __riscv_vfmacc_vv_f32m4(acc.reg, a.reg, b.reg, 16);
};

#ifdef RISCV_BF16_SUPPORT
inline BF16Vec8::BF16Vec8(const FP32Vec8& v)
    : reg(__riscv_vfncvtbf16_f_f_w_bf16m1(v.reg, VEC_ELEM_NUM)) {};

inline BF16Vec16::BF16Vec16(const FP32Vec16& v)
    : reg(__riscv_vfncvtbf16_f_f_w_bf16m2(v.reg, VEC_ELEM_NUM)) {};
#endif

inline void prefetch(const void* addr) { __builtin_prefetch(addr, 0, 1); }

#ifdef RISCV_BF16_SUPPORT
template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  *ptr = static_cast<__bf16>(v);
};
#endif
}  // namespace vec_op