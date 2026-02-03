#include <cmath>
#include <cstdint>
#include <cstring>
#include <torch/all.h>
#include "float_convert.hpp"

namespace vec_op {

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)    \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

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

typedef struct f16x8_t {
  uint16_t val[8];
} f16x8_t;

typedef struct f16x16_t {
  uint16_t val[16];
} f16x16_t;

typedef struct f16x32_t {
  uint16_t val[32];
} f16x32_t;

typedef struct f32x4_t {
  float val[4];
} f32x4_t;

typedef struct f32x8_t {
  float val[8];
} f32x8_t;

typedef struct f32x16_t {
  float val[16];
} f32x16_t;

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F&& f) {
  (f(std::integral_constant<T, indexes>{}), ...);
};
};  // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T> > >
constexpr void unroll_loop(F&& f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T>
struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; }
};

struct FP32Vec8;
struct FP32Vec16;

struct FP16Vec8 : public Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  f16x8_t reg;

  explicit FP16Vec8(const void* ptr)
      : reg(*reinterpret_cast<const f16x8_t*>(ptr)) {};

  explicit FP16Vec8(const FP32Vec8&);

  void save(void* ptr) const { *reinterpret_cast<f16x8_t*>(ptr) = reg; }
};

struct FP16Vec16 : public Vec<FP16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  f16x16_t reg;

  explicit FP16Vec16(const void* ptr)
      : reg(*reinterpret_cast<const f16x16_t*>(ptr)) {};

  explicit FP16Vec16(const FP32Vec16&);

  void save(void* ptr) const { *reinterpret_cast<f16x16_t*>(ptr) = reg; }

  void save(void* ptr, const int elem_num) const {
    int num = std::min(elem_num, VEC_ELEM_NUM);
    std::memcpy(ptr, &(reg.val[0]), num * sizeof(uint16_t));
  }
};

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  f16x8_t reg;

  explicit BF16Vec8(const void* ptr)
      : reg(*reinterpret_cast<const f16x8_t*>(ptr)) {};

  explicit BF16Vec8(const FP32Vec8&);

  void save(void* ptr) const { *reinterpret_cast<f16x8_t*>(ptr) = reg; }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  f16x16_t reg;

  explicit BF16Vec16(const void* ptr)
      : reg(*reinterpret_cast<const f16x16_t*>(ptr)) {};

  explicit BF16Vec16(const FP32Vec16&);

  void save(void* ptr) const { *reinterpret_cast<f16x16_t*>(ptr) = reg; }

  void save(void* ptr, const int elem_num) const {
    int num = std::min(elem_num, VEC_ELEM_NUM);
    std::memcpy(ptr, &(reg.val[0]), num * sizeof(uint16_t));
  }
};

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;
  f16x32_t reg;

  explicit BF16Vec32(const void* ptr)
      : reg(*reinterpret_cast<const f16x32_t*>(ptr)) {};

  explicit BF16Vec32(f16x32_t data) : reg(data) {};

  explicit BF16Vec32(BF16Vec8& vec8_data) {
    unroll_loop<int, VEC_ELEM_NUM>([&vec8_data, this](int i) {
      reg.val[i] = vec8_data.reg.val[i % BF16Vec8::VEC_ELEM_NUM];
    });
  }

  void save(void* ptr) const { *reinterpret_cast<f16x32_t*>(ptr) = reg; }
};

struct FP32Vec4 : public Vec<FP32Vec4> {
  constexpr static int VEC_ELEM_NUM = 4;

  f32x4_t reg;

  explicit FP32Vec4(float v) {
    unroll_loop<int, VEC_ELEM_NUM>([&v, this](int i) { reg.val[i] = v; });
  }

  explicit FP32Vec4() {
    unroll_loop<int, VEC_ELEM_NUM>([this](int i) { reg.val[i] = 0.0f; });
  }

  explicit FP32Vec4(const float* ptr)
      : reg(*reinterpret_cast<const f32x4_t*>(ptr)) {};

  explicit FP32Vec4(f32x4_t data) : reg(data) {};

  explicit FP32Vec4(const FP32Vec4& data) : reg(data.reg) {};
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  f32x8_t reg;

  explicit FP32Vec8(float v) {
    unroll_loop<int, VEC_ELEM_NUM>([&v, this](int i) { reg.val[i] = v; });
  }

  explicit FP32Vec8() {
    unroll_loop<int, VEC_ELEM_NUM>([this](int i) { reg.val[i] = 0.0f; });
  }

  explicit FP32Vec8(const float* ptr)
      : reg(*reinterpret_cast<const f32x8_t*>(ptr)) {};

  explicit FP32Vec8(f32x8_t data) : reg(data) {};

  explicit FP32Vec8(const FP32Vec8& data) : reg(data.reg) {};

  explicit FP32Vec8(const FP16Vec8& v) {
    unroll_loop<int, VEC_ELEM_NUM>(
        [&v, this](int i) { reg.val[i] = fp16_to_float(v.reg.val[i]); });
  }

  FP32Vec8(const BF16Vec8& v) {
    unroll_loop<int, VEC_ELEM_NUM>(
        [&v, this](int i) { reg.val[i] = bf16_to_float(v.reg.val[i]); });
  }

  float reduce_sum() const {
    float result = 0;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&result, this](int i) { result += reg.val[i]; });
    return result;
  }

  FP32Vec8 exp() const {
    f32x8_t ret;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&ret, this](int i) { ret.val[i] = expf(reg.val[i]); });
    return FP32Vec8(ret);
  }

  FP32Vec8 tanh() const {
    f32x8_t ret;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&ret, this](int i) { ret.val[i] = tanhf(reg.val[i]); });
    return FP32Vec8(ret);
  }

  FP32Vec8 er() const {
    f32x8_t ret;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&ret, this](int i) { ret.val[i] = erf(reg.val[i]); });
    return FP32Vec8(ret);
  }

  FP32Vec8 operator*(const FP32Vec8& b) const {
    f32x8_t ret;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&ret, &b, this](int i) { ret.val[i] = reg.val[i] * b.reg.val[i]; });
    return FP32Vec8(ret);
  }

  FP32Vec8 operator+(const FP32Vec8& b) const {
    f32x8_t ret;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&ret, &b, this](int i) { ret.val[i] = reg.val[i] + b.reg.val[i]; });
    return FP32Vec8(ret);
  }

  FP32Vec8 operator-(const FP32Vec8& b) const {
    f32x8_t ret;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&ret, &b, this](int i) { ret.val[i] = reg.val[i] - b.reg.val[i]; });
    return FP32Vec8(ret);
  }

  FP32Vec8 operator/(const FP32Vec8& b) const {
    f32x8_t ret;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&ret, &b, this](int i) { ret.val[i] = reg.val[i] / b.reg.val[i]; });
    return FP32Vec8(ret);
  }

  void save(void* ptr) const { *reinterpret_cast<f32x8_t*>(ptr) = reg; }
};

struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  f32x16_t reg;

  explicit FP32Vec16(float v) {
    unroll_loop<int, VEC_ELEM_NUM>([&v, this](int i) { reg.val[i] = v; });
  }

  explicit FP32Vec16() {
    unroll_loop<int, VEC_ELEM_NUM>([this](int i) { reg.val[i] = 0.0f; });
  }

  explicit FP32Vec16(const float* ptr)
      : reg(*reinterpret_cast<const f32x16_t*>(ptr)) {};

  explicit FP32Vec16(f32x16_t data) : reg(data) {};

  FP32Vec16(const FP32Vec4& data) {
    unroll_loop<int, VEC_ELEM_NUM>([&data, this](int i) {
      reg.val[i] = data.reg.val[i % FP32Vec4::VEC_ELEM_NUM];
    });
  }

  FP32Vec16(const FP32Vec8& data) {
    unroll_loop<int, VEC_ELEM_NUM>([&data, this](int i) {
      reg.val[i] = data.reg.val[i % FP32Vec8::VEC_ELEM_NUM];
    });
  }

  FP32Vec16(const FP32Vec16& data) : reg(data.reg) {};

  explicit FP32Vec16(const FP16Vec16& v) {
    unroll_loop<int, VEC_ELEM_NUM>(
        [&v, this](int i) { reg.val[i] = fp16_to_float(v.reg.val[i]); });
  }

  explicit FP32Vec16(const BF16Vec16& v) {
    unroll_loop<int, VEC_ELEM_NUM>(
        [&v, this](int i) { reg.val[i] = bf16_to_float(v.reg.val[i]); });
  }

  explicit FP32Vec16(const FP16Vec8& v) : FP32Vec16(FP32Vec8(v)) {};

  FP32Vec16(const BF16Vec8& v) : FP32Vec16(FP32Vec8(v)) {};

  FP32Vec16 operator*(const FP32Vec16& b) const {
    f32x16_t ret;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&ret, &b, this](int i) { ret.val[i] = reg.val[i] * b.reg.val[i]; });
    return FP32Vec16(ret);
  }

  FP32Vec16 operator+(const FP32Vec16& b) const {
    f32x16_t ret;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&ret, &b, this](int i) { ret.val[i] = reg.val[i] + b.reg.val[i]; });
    return FP32Vec16(ret);
  }

  FP32Vec16 operator-(const FP32Vec16& b) const {
    f32x16_t ret;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&ret, &b, this](int i) { ret.val[i] = reg.val[i] - b.reg.val[i]; });
    return FP32Vec16(ret);
  }

  FP32Vec16 operator/(const FP32Vec16& b) const {
    f32x16_t ret;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&ret, &b, this](int i) { ret.val[i] = reg.val[i] / b.reg.val[i]; });
    return FP32Vec16(ret);
  }

  FP32Vec16 max(const FP32Vec16& b) const {
    f32x16_t ret;
    unroll_loop<int, VEC_ELEM_NUM>([&ret, &b, this](int i) {
      ret.val[i] = std::max(reg.val[i], b.reg.val[i]);
    });
    return FP32Vec16(ret);
  }

  FP32Vec16 min(const FP32Vec16& b) const {
    f32x16_t ret;
    unroll_loop<int, VEC_ELEM_NUM>([&ret, &b, this](int i) {
      ret.val[i] = std::min(reg.val[i], b.reg.val[i]);
    });
    return FP32Vec16(ret);
  }

  FP32Vec16 abs() const {
    f32x16_t ret;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&ret, this](int i) { ret.val[i] = std::abs(reg.val[i]); });
    return FP32Vec16(ret);
  }

  float reduce_sum() const {
    float result = 0.0f;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&result, this](int i) { result += reg.val[i]; });
    return result;
  }

  float reduce_max() const {
    float result = std::numeric_limits<float>::lowest();
    unroll_loop<int, VEC_ELEM_NUM>(
        [&result, this](int i) { result = std::max(reg.val[i], result); });
    return result;
  }

  float reduce_min() const {
    float result = std::numeric_limits<float>::max();
    unroll_loop<int, VEC_ELEM_NUM>(
        [&result, this](int i) { result = std::min(reg.val[i], result); });
    return result;
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);
    float sum = 0.0;
    const int start = idx * group_size;
    unroll_loop<int, group_size>(
        [&sum, &start, this](int i) { sum += reg.val[start + i]; });
    return sum;
  }

  void save(void* ptr) const { *reinterpret_cast<f32x16_t*>(ptr) = reg; }
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

template <>
struct VecType<c10::BFloat16> {
  using vec_type = BF16Vec8;
};

template <typename T>
void storeFP32(float v, T* ptr) {
  *ptr = v;
}

/*
template <> inline void storeFP32<c10::Half>(float v, c10::Half *ptr) {
  c10::Half __attribute__((__may_alias__)) *v_ptr =
      reinterpret_cast<c10::Half *>(&v);
  *ptr = *(v_ptr + 1);
}
*/

template <>
inline void storeFP32<c10::Half>(float v, c10::Half* ptr) {
  uint16_t fp16 = float_to_fp16(v);
  *reinterpret_cast<uint16_t*>(ptr) = fp16;
}

template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  c10::BFloat16 __attribute__((__may_alias__))* v_ptr =
      reinterpret_cast<c10::BFloat16*>(&v);
  *ptr = *(v_ptr + 1);
}

inline FP16Vec16::FP16Vec16(const FP32Vec16& v) {
  unroll_loop<int, FP16Vec16::VEC_ELEM_NUM>(
      [&v, this](int i) { reg.val[i] = float_to_fp16(v.reg.val[i]); });
}

inline FP16Vec8 ::FP16Vec8(const FP32Vec8& v) {
  unroll_loop<int, FP16Vec8::VEC_ELEM_NUM>(
      [&v, this](int i) { reg.val[i] = float_to_fp16(v.reg.val[i]); });
}

inline void fma(FP32Vec16& acc, FP32Vec16& a, FP32Vec16& b) {
  acc = acc + a * b;
}

inline BF16Vec8::BF16Vec8(const FP32Vec8& v) {
  unroll_loop<int, BF16Vec8::VEC_ELEM_NUM>(
      [&v, this](int i) { reg.val[i] = float_to_bf16(v.reg.val[i]); });
}

inline BF16Vec16::BF16Vec16(const FP32Vec16& v) {
  unroll_loop<int, BF16Vec16::VEC_ELEM_NUM>(
      [&v, this](int i) { reg.val[i] = float_to_bf16(v.reg.val[i]); });
}

inline void prefetch(const void* addr) { __builtin_prefetch(addr, 0, 3); }

};  // namespace vec_op
