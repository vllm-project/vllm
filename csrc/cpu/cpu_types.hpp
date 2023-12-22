
#ifndef CPU_TYPES_HPP
#define CPU_TYPES_HPP
#include <type_traits>
#include <utility>
#include <torch/extension.h>

#ifdef __aarch64__
#define _Float16 __fp16
#include <arm_neon.h>

/* NEON implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/

/* Copyright (C) 2011  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/
#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

typedef float32x4_t v4sf;  // vector of 4 float
typedef uint32x4_t v4su;  // vector of 4 uint32
typedef int32x4_t v4si;  // vector of 4 uint32
/* exp() computed for 4 float at once */
static v4sf exp_ps(v4sf x) {
  v4sf tmp, fx;

  v4sf one = vdupq_n_f32(1);
  x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
  x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

  /* perform a floorf */
  tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

  /* if greater, substract 1 */
  v4su mask = vcgtq_f32(tmp, fx);    
  mask = vandq_u32(mask, vreinterpretq_u32_f32(one));


  fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

  tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
  v4sf z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
  x = vsubq_f32(x, tmp);
  x = vsubq_f32(x, z);

  static const float cephes_exp_p[6] = { c_cephes_exp_p0, c_cephes_exp_p1, c_cephes_exp_p2, c_cephes_exp_p3, c_cephes_exp_p4, c_cephes_exp_p5 };
  v4sf y = vld1q_dup_f32(cephes_exp_p+0);
  v4sf c1 = vld1q_dup_f32(cephes_exp_p+1); 
  v4sf c2 = vld1q_dup_f32(cephes_exp_p+2); 
  v4sf c3 = vld1q_dup_f32(cephes_exp_p+3); 
  v4sf c4 = vld1q_dup_f32(cephes_exp_p+4); 
  v4sf c5 = vld1q_dup_f32(cephes_exp_p+5);

  y = vmulq_f32(y, x);
  z = vmulq_f32(x,x);
  y = vaddq_f32(y, c1);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c2);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c3);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c4);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c5);
  
  y = vmulq_f32(y, z);
  y = vaddq_f32(y, x);
  y = vaddq_f32(y, one);

  /* build 2^n */
  int32x4_t mm;
  mm = vcvtq_s32_f32(fx);
  mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
  mm = vshlq_n_s32(mm, 23);
  v4sf pow2n = vreinterpretq_f32_s32(mm);

  y = vmulq_f32(y, pow2n);
  return y;
}

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
template <typename F, typename T, T... indexes>
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

struct FP16Vec8 : public Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  float16x8_t reg;

  explicit FP16Vec8(__fp16 v) : reg(vmovq_n_f16(v)) {}

  explicit FP16Vec8(const void *ptr) : reg(vld1q_f16((const __fp16 *)ptr)) {}

  explicit FP16Vec8(float16x8_t data) : reg(data) {}

  explicit FP16Vec8(float32x4x2_t v) : reg(vcombine_f16(vcvt_f16_f32(v.val[0]), vcvt_f16_f32(v.val[1]))) {}

  FP16Vec8 operator*(const FP16Vec8 &b) const {
    return FP16Vec8(vmulq_f16(reg, b.reg));
  }

  FP16Vec8 operator+(const FP16Vec8 &b) const {
    return FP16Vec8(vaddq_f16(reg, b.reg));
  }

  FP16Vec8 operator-(const FP16Vec8 &b) const {
    return FP16Vec8(vsubq_f16(reg, b.reg));
  }

  FP16Vec8 operator/(const FP16Vec8 &b) const {
    return FP16Vec8(vdivq_f16(reg, b.reg));
  }

  void save(void *ptr) const { vst1q_f16((__fp16 *)ptr, reg); }
};

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  bfloat16x8_t reg;

  explicit BF16Vec8(const void *ptr)
      : reg(*reinterpret_cast<const bfloat16x8_t *>(ptr)) {}

  explicit BF16Vec8(bfloat16x8_t data) : reg(data) {}

  explicit BF16Vec8(float32x4x2_t v) : reg(vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.val[0]), v.val[1])) {}

  void save(void *ptr) const { *reinterpret_cast<bfloat16x8_t *>(ptr) = reg; }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  bfloat16x8x2_t reg;

  explicit BF16Vec16(const void *ptr)
      : reg(*reinterpret_cast<const bfloat16x8x2_t *>(ptr)) {}

  explicit BF16Vec16(bfloat16x8x2_t data) : reg(data) {}

  explicit BF16Vec16(float32x4x4_t v) : reg({
    vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.val[0]), v.val[1]),
    vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(v.val[2]), v.val[3])
  }){};

  void save(void *ptr) const { *reinterpret_cast<bfloat16x8x2_t *>(ptr) = reg; }
};

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;

  bfloat16x8x4_t reg;

  explicit BF16Vec32(const void *ptr)
      : reg(*reinterpret_cast<const bfloat16x8x4_t *>(ptr)) {}

  explicit BF16Vec32(bfloat16x8x4_t data) : reg(data) {}

  explicit BF16Vec32(const BF16Vec8 &vec8_data) : reg({
    vec8_data.reg,
    vec8_data.reg,
    vec8_data.reg,
    vec8_data.reg
  }) {}

  void save(void *ptr) const { *reinterpret_cast<bfloat16x8x4_t *>(ptr) = reg; }
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  union AliasReg {
    float32x4x2_t reg;
    float values[VEC_ELEM_NUM];
  };

  float32x4x2_t reg;

  explicit FP32Vec8(float v) : reg({vmovq_n_f32(v), vmovq_n_f32(v)}) {}

  explicit FP32Vec8() : reg({vmovq_n_f32(0.0), vmovq_n_f32(0.0)}) {}

  explicit FP32Vec8(const float *ptr) : reg({vld1q_f32(ptr), vld1q_f32(ptr + 4)}) {}

  explicit FP32Vec8(float32x4x2_t data) : reg(data) {}

  explicit FP32Vec8(float16x8_t v) : reg({vcvt_f32_f16(vget_low_f16(v)), vcvt_f32_f16(vget_high_f16(v))}) {}

  explicit FP32Vec8(bfloat16x8_t v) : reg({vcvtq_low_f32_bf16(v), vcvtq_high_f32_bf16(v)}) {}

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float ans = 0;
    unroll_loop<int, VEC_ELEM_NUM>([&ans, &ar](int i) { ans += ar.values[i]; });

    return ans;
  }

  FP32Vec8 exp() const {

    return FP32Vec8(float32x4x2_t({exp_ps(reg.val[0]), exp_ps(reg.val[1])}));
  }

  FP32Vec8 operator*(const FP32Vec8 &b) const {
    return FP32Vec8(float32x4x2_t({vmulq_f32(reg.val[0], b.reg.val[0]), vmulq_f32(reg.val[1], b.reg.val[1])}));
  }

  FP32Vec8 operator+(const FP32Vec8 &b) const {
    return FP32Vec8(float32x4x2_t({vaddq_f32(reg.val[0], b.reg.val[0]), vaddq_f32(reg.val[1], b.reg.val[1])}));
  }

  FP32Vec8 operator-(const FP32Vec8 &b) const {
    return FP32Vec8(float32x4x2_t({vsubq_f32(reg.val[0], b.reg.val[0]), vsubq_f32(reg.val[1], b.reg.val[1])}));
  }

  FP32Vec8 operator/(const FP32Vec8 &b) const {
    return FP32Vec8(float32x4x2_t({vdivq_f32(reg.val[0], b.reg.val[0]), vdivq_f32(reg.val[1], b.reg.val[1])}));
  }

  void save(float *ptr) const {
    vst1q_f32(ptr, reg.val[0]);
    vst1q_f32(ptr + 4, reg.val[1]);
  }
};

struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    float32x4x4_t reg;
    float values[VEC_ELEM_NUM];
  };

  float32x4x4_t reg;

  explicit FP32Vec16(float v) : reg({vmovq_n_f32(v), vmovq_n_f32(v), vmovq_n_f32(v), vmovq_n_f32(v)}) {}

  explicit FP32Vec16() : reg({vmovq_n_f32(0.0), vmovq_n_f32(0.0), vmovq_n_f32(0.0), vmovq_n_f32(0.0)}) {}

  explicit FP32Vec16(const float *ptr) : reg({vld1q_f32(ptr), vld1q_f32(ptr + 4), vld1q_f32(ptr + 8), vld1q_f32(ptr + 12)}) {}

  explicit FP32Vec16(float32x4x4_t data) : reg(data) {}

  explicit FP32Vec16(bfloat16x8x2_t v) : reg({
    vcvtq_low_f32_bf16(v.val[0]),
    vcvtq_high_f32_bf16(v.val[0]),
    vcvtq_low_f32_bf16(v.val[1]),
    vcvtq_high_f32_bf16(v.val[1])
  }) {}

  FP32Vec16 operator+(const FP32Vec16 &b) const {
    return FP32Vec16(float32x4x4_t({
        vaddq_f32(reg.val[0], b.reg.val[0]),
        vaddq_f32(reg.val[1], b.reg.val[1]),
        vaddq_f32(reg.val[2], b.reg.val[2]),
        vaddq_f32(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 operator*(const FP32Vec16 &b) const {
    return FP32Vec16(float32x4x4_t({
        vmulq_f32(reg.val[0], b.reg.val[0]),
        vmulq_f32(reg.val[1], b.reg.val[1]),
        vmulq_f32(reg.val[2], b.reg.val[2]),
        vmulq_f32(reg.val[3], b.reg.val[3])}));
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

  void save(float *ptr) const {
    vst1q_f32(ptr, reg.val[0]);
    vst1q_f32(ptr + 4, reg.val[1]);
    vst1q_f32(ptr + 8, reg.val[2]);
    vst1q_f32(ptr + 12, reg.val[3]);
  }
};

template <typename T> struct VecType { using vec_type = void; };

template <typename T> using vec_t = typename VecType<T>::vec_type;

template <> struct VecType<float> { using vec_type = FP32Vec8; };

template <> struct VecType<c10::Half> { using vec_type = FP16Vec8; };

template <> struct VecType<c10::BFloat16> { using vec_type = BF16Vec8; };

template <typename T> void storeFP32ToT(float v, T *ptr) { *ptr = v; }

template <> inline void storeFP32ToT<c10::Half>(float v, c10::Half *ptr) {
  *reinterpret_cast<__fp16 *>(ptr) = v;
}

inline FP32Vec16 fma(BF16Vec32 &a, BF16Vec32 &b, FP32Vec16 &c) {
  return FP32Vec16(float32x4x4_t({
    vbfmlaltq_f32(vbfmlalbq_f32(c.reg.val[0], a.reg.val[0], b.reg.val[0]), a.reg.val[0], b.reg.val[0]),
    vbfmlaltq_f32(vbfmlalbq_f32(c.reg.val[1], a.reg.val[1], b.reg.val[1]), a.reg.val[1], b.reg.val[1]),
    vbfmlaltq_f32(vbfmlalbq_f32(c.reg.val[2], a.reg.val[2], b.reg.val[2]), a.reg.val[2], b.reg.val[2]),
    vbfmlaltq_f32(vbfmlalbq_f32(c.reg.val[3], a.reg.val[3], b.reg.val[3]), a.reg.val[3], b.reg.val[3]),
  }));
}

template <>
inline void storeFP32ToT<c10::BFloat16>(float v, c10::BFloat16 *ptr) {
  *reinterpret_cast<__bf16 *>(ptr) = vcvth_bf16_f32(v);
}

}; // namespace vec_op


#else // __aarch64__
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

#endif // ndef __aarch64__
#endif
