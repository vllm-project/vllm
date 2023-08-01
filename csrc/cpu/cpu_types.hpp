
#ifndef CPU_TYPES_HPP
#define CPU_TYPES_HPP

#include <immintrin.h>
#include <torch/extension.h>

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
  std::cout << #NAME << " invoked." << std::endl;
#define CPU_KERNEL_GUARD_OUT(NAME) std::cout << #NAME << " exit." << std::endl;
#endif

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

#ifdef __AVX512FP16__
struct FP16Vec8 : public Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  __m128h reg;

  explicit FP16Vec8(_Float16 v) : reg(_mm_set1_ph(v)) {}

  explicit FP16Vec8(const void *ptr) : reg(_mm_loadu_ph(ptr)) {}

  explicit FP16Vec8(__m128h data) : reg(data) {}

  explicit FP16Vec8(__m256 v)
      : reg(_mm_castsi128_ph(_mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT))){};

  FP16Vec8 operator*(const FP16Vec8 &b) const {
    return FP16Vec8(_mm_mul_ph(reg, b.reg));
  }

  FP16Vec8 operator+(const FP16Vec8 &b) const {
    return FP16Vec8(_mm_add_ph(reg, b.reg));
  }

  FP16Vec8 operator-(const FP16Vec8 &b) const {
    return FP16Vec8(_mm_sub_ph(reg, b.reg));
  }

  FP16Vec8 operator/(const FP16Vec8 &b) const {
    return FP16Vec8(_mm_div_ph(reg, b.reg));
  }

  void save(void *ptr) const { _mm_storeu_ph(ptr, reg); }
};
#endif

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  __m128bh reg;

  explicit BF16Vec8(const void *ptr)
      : reg(*reinterpret_cast<const __m128bh *>(ptr)) {}

  explicit BF16Vec8(__m128bh data) : reg(data) {}

  explicit BF16Vec8(__m256 v) : reg(_mm256_cvtneps_pbh(v)){};

  void save(void *ptr) const { *reinterpret_cast<__m128bh *>(ptr) = reg; }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  __m256bh reg;

  explicit BF16Vec16(const void *ptr)
      : reg(*reinterpret_cast<const __m256bh *>(ptr)) {}

  explicit BF16Vec16(__m256bh data) : reg(data) {}

  explicit BF16Vec16(__m512 v) : reg(_mm512_cvtneps_pbh(v)){};

  void save(void *ptr) const { *reinterpret_cast<__m256bh *>(ptr) = reg; }
};

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;

  __m512bh reg;

  explicit BF16Vec32(const void *ptr)
      : reg(*reinterpret_cast<const __m512bh *>(ptr)) {}

  explicit BF16Vec32(__m512bh data) : reg(data) {}

  explicit BF16Vec32(BF16Vec8 &vec8_data)
      : reg((__m512bh)_mm512_inserti32x4(
            _mm512_inserti32x4(_mm512_inserti32x4(_mm512_castsi128_si512(
                                                      (__m128i)vec8_data.reg),
                                                  (__m128i)vec8_data.reg, 1),
                               (__m128i)vec8_data.reg, 2),
            (__m128i)vec8_data.reg, 3)) {}

  void save(void *ptr) const { *reinterpret_cast<__m512bh *>(ptr) = reg; }
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  union AliasReg {
    __m256 reg;
    float values[VEC_ELEM_NUM];
  };

  __m256 reg;

  explicit FP32Vec8(float v) : reg(_mm256_set1_ps(v)) {}

  explicit FP32Vec8() : reg(_mm256_set1_ps(0.0)) {}

  explicit FP32Vec8(const float *ptr) : reg(_mm256_loadu_ps(ptr)) {}

  explicit FP32Vec8(__m256 data) : reg(data) {}

#ifdef __AVX512FP16__
  explicit FP32Vec8(__m128h v) : reg(_mm256_cvtph_ps(_mm_castph_si128(v))) {}
#endif

  explicit FP32Vec8(__m128bh v) : reg(_mm256_cvtpbh_ps(v)) {}

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float ans = 0;
    unroll_loop<int, VEC_ELEM_NUM>([&ans, &ar](int i) { ans += ar.values[i]; });

    return ans;
  }

  FP32Vec8 exp() const {
    AliasReg ar;
    ar.reg = reg;
    return FP32Vec8(_mm256_set_ps(expf(ar.values[7]), expf(ar.values[6]),
                                  expf(ar.values[5]), expf(ar.values[4]),
                                  expf(ar.values[3]), expf(ar.values[2]),
                                  expf(ar.values[1]), expf(ar.values[0])));
  }

  FP32Vec8 operator*(const FP32Vec8 &b) const {
    return FP32Vec8(_mm256_mul_ps(reg, b.reg));
  }

  FP32Vec8 operator+(const FP32Vec8 &b) const {
    return FP32Vec8(_mm256_add_ps(reg, b.reg));
  }

  FP32Vec8 operator-(const FP32Vec8 &b) const {
    return FP32Vec8(_mm256_sub_ps(reg, b.reg));
  }

  FP32Vec8 operator/(const FP32Vec8 &b) const {
    return FP32Vec8(_mm256_div_ps(reg, b.reg));
  }

  void save(float *ptr) const { _mm256_storeu_ps(ptr, reg); }
};

struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    __m512 reg;
    float values[VEC_ELEM_NUM];
  };

  __m512 reg;

  explicit FP32Vec16(float v) : reg(_mm512_set1_ps(v)) {}

  explicit FP32Vec16() : reg(_mm512_set1_ps(0.0)) {}

  explicit FP32Vec16(const float *ptr) : reg(_mm512_loadu_ps(ptr)) {}

  explicit FP32Vec16(__m512 data) : reg(data) {}

  explicit FP32Vec16(__m256bh v) : reg(_mm512_cvtpbh_ps(v)) {}

  FP32Vec16 operator+(const FP32Vec16 &b) const {
    return FP32Vec16(_mm512_add_ps(reg, b.reg));
  }

  FP32Vec16 operator*(const FP32Vec16 &b) const {
    return FP32Vec16(_mm512_mul_ps(reg, b.reg));
  }

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float ans = 0;
    unroll_loop<int, VEC_ELEM_NUM>([&ans, &ar](int i) { ans += ar.values[i]; });

    return ans;
  }

  template <int group_size> float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);

    AliasReg ar;
    ar.reg = reg;
    float ans = 0;
    const int start = idx * group_size;
    unroll_loop<int, group_size>(
        [&ans, &start, ar](int i) { ans += ar.values[start + i]; });

    return ans;
  }

  void save(float *ptr) const { _mm512_storeu_ps(ptr, reg); }
};

template <typename T> struct VecType { using vec_type = void; };

template <typename T> using vec_t = typename VecType<T>::vec_type;

template <> struct VecType<float> { using vec_type = FP32Vec8; };

#ifdef __AVX512FP16__
template <> struct VecType<c10::Half> { using vec_type = FP16Vec8; };
#endif

template <> struct VecType<c10::BFloat16> { using vec_type = BF16Vec8; };

template <typename T> void storeFP32ToT(float v, T *ptr) { *ptr = v; }

#ifdef __AVX512FP16__
template <> inline void storeFP32ToT<c10::Half>(float v, c10::Half *ptr) {
  *reinterpret_cast<_Float16 *>(ptr) = v;
}
#endif

inline FP32Vec16 fma(BF16Vec32 &a, BF16Vec32 &b, FP32Vec16 &c) {
  return FP32Vec16(_mm512_dpbf16_ps(c.reg, a.reg, b.reg));
}

template <>
inline void storeFP32ToT<c10::BFloat16>(float v, c10::BFloat16 *ptr) {
  *reinterpret_cast<__bfloat16 *>(ptr) = _mm_cvtness_sbh(v);
}

}; // namespace vec_op

#endif