
#ifndef CPU_TYPES_VXE_HPP
#define CPU_TYPES_VXE_HPP

#include <vecintrin.h>
#include <cmath>
#include <limits>
#include <torch/all.h>
namespace vec_op {

#define vec_neg(a) (-(a))
#define vec_add(a, b) ((a) + (b))
#define vec_sub(a, b) ((a) - (b))
#define vec_mul(a, b) ((a) * (b))
#define vec_div(a, b) ((a) / (b))
#define vec_sr(a, b) ((a) >> (b))  // Vector Shift Right Algebraic
#define vec_sl(a, b) ((a) << (b))  // Vector Shift Left

// NOTE: FP16 (Half) is supported on s390x via custom bit-manipulation
// conversion. PyTorch itself lacks native s390x FP16 support.
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

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F&& f) {
  (f(std::integral_constant<T, indexes>{}), ...);
}
};  // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F&& f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T>
struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; }
};

typedef struct ss16x8x2_t {
  __vector signed short val[2];
} ss16x8x2_t;

typedef struct ss16x8x4_t {
  __vector signed short val[4];
} ss16x8x4_t;

typedef struct f32x4x2_t {
  __vector float val[2];
} f32x4x2_t;

typedef struct f32x4x4_t {
  __vector float val[4];
} f32x4x4_t;

struct FP32Vec8;
struct FP32Vec16;

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  __vector signed short reg;

  explicit BF16Vec8(const void* ptr) : reg(*(__vector signed short*)ptr) {}
  explicit BF16Vec8(const FP32Vec8&);

  void save(void* ptr) const {
    *reinterpret_cast<__vector signed short*>(ptr) = reg;
  }
};

struct FP16Vec8 : public Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  __vector signed short reg;

  explicit FP16Vec8(const void* ptr) : reg(*(__vector signed short*)ptr) {}
  explicit FP16Vec8(const FP32Vec8&);

  void save(void* ptr) const {
    *reinterpret_cast<__vector signed short*>(ptr) = reg;
  }
};

struct FP16Vec16 : public Vec<FP16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  ss16x8x2_t reg;

  explicit FP16Vec16(const void* ptr) {
    // Load 256 bits (16 FP16 values) in two parts
    reg.val[0] = (__vector signed short)vec_xl(0, (signed short*)ptr);
    reg.val[1] = (__vector signed short)vec_xl(16, (signed short*)ptr);
  }

  explicit FP16Vec16(const FP32Vec16&);

  void save(void* ptr) const {
    // Save 256 bits in two parts
    vec_xst(reg.val[0], 0, (signed short*)ptr);
    vec_xst(reg.val[1], 16, (signed short*)ptr);
  }
};

struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  ss16x8x2_t reg;

  explicit BF16Vec16(const void* ptr) {
    // Load 256 bits in two parts
    reg.val[0] = (__vector signed short)vec_xl(0, (signed short*)ptr);
    reg.val[1] = (__vector signed short)vec_xl(16, (signed short*)ptr);
  }

  explicit BF16Vec16(const FP32Vec16&);

  void save(void* ptr) const {
    // Save 256 bits in two parts
    vec_xst(reg.val[0], 0, (signed short*)ptr);
    vec_xst(reg.val[1], 16, (signed short*)ptr);
  }
};

const static __vector signed short zero = vec_splats((signed short)0);

FORCE_INLINE __vector float fp16_to_fp32_bits(__vector unsigned int x) {
  const __vector unsigned int mask_sign = {0x8000, 0x8000, 0x8000, 0x8000};
  const __vector unsigned int mask_exp = {0x7C00, 0x7C00, 0x7C00, 0x7C00};
  const __vector unsigned int mask_mant = {0x03FF, 0x03FF, 0x03FF, 0x03FF};
  const __vector unsigned int bias_adj = {112, 112, 112, 112};
  const __vector unsigned int exp_max_fp16 = {0x1F, 0x1F, 0x1F,
                                              0x1F};  // FP16 NaN/Inf exponent
  const __vector unsigned int exp_max_fp32 = {0xFF, 0xFF, 0xFF,
                                              0xFF};  // FP32 NaN/Inf exponent

  __vector unsigned int s = (x & mask_sign) << 16;
  __vector unsigned int e = (x & mask_exp) >> 10;
  __vector unsigned int m = (x & mask_mant) << 13;

  // Check for NaN/Inf: exponent = 0x1F in FP16
  __vector __bool int is_nan_inf = vec_cmpeq(e, exp_max_fp16);

  // Normal: adjust bias; NaN/Inf: set to 0xFF
  __vector unsigned int e_normal = e + bias_adj;
  e = vec_sel(e_normal, exp_max_fp32, is_nan_inf);

  return (__vector float)(s | (e << 23) | m);
}

FORCE_INLINE __vector unsigned int fp32_to_fp16_bits(__vector float f_in) {
  __vector unsigned int in = (__vector unsigned int)f_in;

  const __vector unsigned int mask_sign_32 = {0x80000000, 0x80000000,
                                              0x80000000, 0x80000000};
  const __vector unsigned int mask_exp_32 = {0x7F800000, 0x7F800000, 0x7F800000,
                                             0x7F800000};
  const __vector unsigned int mask_mant_32 = {0x007FFFFF, 0x007FFFFF,
                                              0x007FFFFF, 0x007FFFFF};

  // Use SIGNED integers for exponent math to handle underflow check
  const __vector signed int bias_adj = {112, 112, 112, 112};
  const __vector signed int zero = {0, 0, 0, 0};
  const __vector signed int max_exp = {31, 31, 31, 31};  // Max FP16 exp
  const __vector unsigned int exp_max_fp32 = {0xFF, 0xFF, 0xFF, 0xFF};
  const __vector unsigned int exp_max_fp16 = {0x1F, 0x1F, 0x1F, 0x1F};

  __vector unsigned int s = (in & mask_sign_32) >> 16;
  __vector unsigned int e_u = (in & mask_exp_32) >> 23;

  // Check for NaN/Inf: exponent = 0xFF in FP32
  __vector __bool int is_nan_inf = vec_cmpeq(e_u, exp_max_fp32);

  __vector signed int e_s = (__vector signed int)e_u;
  e_s = vec_sub(e_s, bias_adj);
  e_s = vec_max(e_s, zero);
  e_s = vec_min(e_s, max_exp);
  __vector unsigned int e_normal = (__vector unsigned int)e_s;

  __vector unsigned int e_final = vec_sel(e_normal, exp_max_fp16, is_nan_inf);

  const __vector unsigned int one_v = {1, 1, 1, 1};
  const __vector unsigned int mask_sticky = {0xFFF, 0xFFF, 0xFFF, 0xFFF};

  __vector unsigned int round_bit = (in >> 12) & one_v;
  __vector unsigned int sticky = in & mask_sticky;
  __vector unsigned int m = (in & mask_mant_32) >> 13;
  __vector unsigned int lsb = m & one_v;  // LSB of mantissa for tie-breaking

  // Round up if: round_bit && (sticky || lsb)
  __vector __bool int sticky_nonzero =
      vec_cmpgt(sticky, (__vector unsigned int){0, 0, 0, 0});
  __vector __bool int lsb_set = vec_cmpeq(lsb, one_v);
  __vector __bool int round_up =
      vec_and(vec_cmpeq(round_bit, one_v), vec_or(sticky_nonzero, lsb_set));

  m = vec_sel(m, m + one_v, round_up);

  const __vector unsigned int mant_mask = {0x3FF, 0x3FF, 0x3FF, 0x3FF};
  const __vector unsigned int max_normal_exp = {0x1E, 0x1E, 0x1E, 0x1E};
  __vector __bool int mant_overflows = vec_cmpgt(m, mant_mask);
  __vector __bool int would_overflow_to_inf =
      vec_and(mant_overflows, vec_cmpeq(e_final, max_normal_exp));
  __vector unsigned int e_inc = vec_min(e_final + one_v, exp_max_fp16);
  e_final = vec_sel(e_final, e_inc, mant_overflows);
  m = vec_and(m, mant_mask);
  e_final = vec_sel(e_final, max_normal_exp, would_overflow_to_inf);
  m = vec_sel(m, mant_mask, would_overflow_to_inf);

  return s | (e_final << 10) | m;
}

struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;

  ss16x8x4_t reg;
  explicit BF16Vec32(const void* ptr)
      : reg(*reinterpret_cast<const ss16x8x4_t*>(ptr)) {}

  explicit BF16Vec32(ss16x8x4_t data) : reg(data) {}

  explicit BF16Vec32(const BF16Vec8& vec8_data)
      : reg({vec8_data.reg, vec8_data.reg, vec8_data.reg, vec8_data.reg}) {}

  void save(void* ptr) const { *reinterpret_cast<ss16x8x4_t*>(ptr) = reg; }
};

struct FP32Vec4 : public Vec<FP32Vec4> {
  constexpr static int VEC_ELEM_NUM = 4;
  union AliasReg {
    __vector float reg;
    float values[VEC_ELEM_NUM];
  };

  __vector float reg;

  explicit FP32Vec4(float v) : reg(vec_splats(v)) {}

  explicit FP32Vec4() : reg(vec_splats(0.0f)) {}

  explicit FP32Vec4(const float* ptr) : reg(vec_xl(0, ptr)) {}

  explicit FP32Vec4(__vector float data) : reg(data) {}

  explicit FP32Vec4(const FP32Vec4& data) : reg(data.reg) {}
};

struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;
  union AliasReg {
    f32x4x2_t reg;
    float values[VEC_ELEM_NUM];
  };

  f32x4x2_t reg;

  explicit FP32Vec8(float v) {
    reg.val[0] = vec_splats(v);
    reg.val[1] = vec_splats(v);
  }

  explicit FP32Vec8() {
    reg.val[0] = vec_splats(0.0f);
    reg.val[1] = vec_splats(0.0f);
  }

  explicit FP32Vec8(const float* ptr) {
    reg.val[0] = vec_xl(0, ptr);
    reg.val[1] = vec_xl(16, ptr);
  }

  explicit FP32Vec8(f32x4x2_t data) : reg(data) {}

  explicit FP32Vec8(const FP32Vec8& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[1];
  }

  explicit FP32Vec8(const BF16Vec8& v) {
    // On big-endian s390x, place BF16 first to get correct byte order
    reg.val[0] = (__vector float)vec_mergeh(v.reg, zero);
    reg.val[1] = (__vector float)vec_mergel(v.reg, zero);
  }

  explicit FP32Vec8(const FP16Vec8& v) {
    // Cast to UNSIGNED short vector to prevent sign-extension during unpack
    __vector unsigned short raw_u = (__vector unsigned short)v.reg;

    // Unpack 8x16-bit to two 4x32-bit vectors (Zero extended)
    __vector unsigned int raw_hi = (__vector unsigned int)vec_unpackh(raw_u);
    __vector unsigned int raw_lo = (__vector unsigned int)vec_unpackl(raw_u);

    reg.val[0] = fp16_to_fp32_bits(raw_hi);
    reg.val[1] = fp16_to_fp32_bits(raw_lo);
  }

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float result = 0;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&result, &ar](int i) { result += ar.values[i]; });

    return result;
  }

  FP32Vec8 exp() const {
    f32x4x2_t out;

    const __vector float log2e = vec_splats(1.44269504088896341f);
    const __vector float one = vec_splats(1.0f);
    const __vector float min_x = vec_splats(-87.3f);
    const __vector float max_x = vec_splats(88.7f);

    // 5th-degree minimax polynomial for 2^r (r in [0,1))
    const __vector float c1 = vec_splats(0.6931471805599453f);
    const __vector float c2 = vec_splats(0.240226506959101f);
    const __vector float c3 = vec_splats(0.05550410866482158f);
    const __vector float c4 = vec_splats(0.009618129107628477f);
    const __vector float c5 = vec_splats(0.0013333558146428443f);

    for (int i = 0; i < 2; i++) {
      __vector float x = reg.val[i];

      x = vec_max(x, min_x);
      x = vec_min(x, max_x);

      __vector float y = vec_mul(x, log2e);

      __vector float kf = vec_floor(y);
      __vector float r = vec_sub(y, kf);

      __vector signed int k = vec_signed(kf);
      const __vector signed int min_k = vec_splats((signed int)-126);
      const __vector signed int max_k = vec_splats((signed int)127);
      k = vec_min(vec_max(k, min_k), max_k);

      // Build 2^k from exponent bits
      __vector signed int exp_int = vec_add(k, vec_splats((signed int)127));
      __vector unsigned int bits = (__vector unsigned int)exp_int;
      bits = vec_sl(bits, vec_splats((unsigned int)23));
      __vector float pow2k = (__vector float)bits;

      // Improved minimax polynomial
      __vector float poly = vec_madd(c5, r, c4);
      poly = vec_madd(poly, r, c3);
      poly = vec_madd(poly, r, c2);
      poly = vec_madd(poly, r, c1);
      poly = vec_madd(poly, r, one);

      out.val[i] = vec_mul(pow2k, poly);
    }

    return FP32Vec8(out);
  }

  FP32Vec8 tanh() const {
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    const __vector float one = vec_splats(1.0f);
    const __vector float two = vec_splats(2.0f);
    const __vector float zero = vec_splats(0.0f);
    const __vector float sat =
        vec_splats(9.0f);  // beyond this, tanh(x) ~ sign(x)

    f32x4x2_t out;

    for (int i = 0; i < 2; i++) {
      __vector float x = reg.val[i];
      __vector float ax = vec_abs(x);

      // sign(x): +1 or -1
      __vector float sign = vec_sel(vec_splats(-1.0f), one, vec_cmpgt(x, zero));

      // saturation mask: |x| > sat
      __vector __bool int saturated = vec_cmpgt(ax, sat);

      // 2x
      __vector float two_x = vec_mul(x, two);

      // Build a temporary FP32Vec8 with both lanes = 2x, reuse exp()
      f32x4x2_t tmp;
      tmp.val[0] = two_x;
      tmp.val[1] = two_x;
      FP32Vec8 exp_2x_vec(tmp);

      FP32Vec8 e2x = exp_2x_vec.exp();
      __vector float e = e2x.reg.val[i];

      // tanh(x) = (e - 1) / (e + 1)
      __vector float num = vec_sub(e, one);
      __vector float den = vec_add(e, one);

      __vector float t = vec_div(num, den);

      // For large |x|, clamp to sign(x)
      out.val[i] = vec_sel(t, sign, saturated);
    }

    return FP32Vec8(out);
  }

  FP32Vec8 er() const {
    // A&S 7.1.26 approximation:
    // erf(x) = sign(x) * (1 - ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t *
    // exp(-x^2)) t = 1 / (1 + p*|x|),  p = 0.3275911

    const __vector float one = vec_splats(1.0f);
    const __vector float zero = vec_splats(0.0f);
    const __vector float p = vec_splats(0.3275911f);

    // Polynomial coeffs
    const __vector float a1 = vec_splats(0.254829592f);
    const __vector float a2 = vec_splats(-0.284496736f);
    const __vector float a3 = vec_splats(1.421413741f);
    const __vector float a4 = vec_splats(-1.453152027f);
    const __vector float a5 = vec_splats(1.061405429f);

    // Threshold where erf(x) ~ sign(x)
    const __vector float sat = vec_splats(6.0f);

    f32x4x2_t out;

    for (int lane = 0; lane < 2; lane++) {
      __vector float x = reg.val[lane];
      __vector float ax = vec_abs(x);

      // sign(x)
      __vector float sign = vec_sel(vec_splats(-1.0f), one, vec_cmpgt(x, zero));

      // |x| > 6 → erf(x) = ±1
      __vector __bool int saturated = vec_cmpgt(ax, sat);

      // t = 1 / (1 + p * |x|)
      __vector float t = vec_madd(p, ax, one);
      t = vec_div(one, t);

      // poly = a5
      __vector float poly = a5;
      poly = vec_madd(poly, t, a4);
      poly = vec_madd(poly, t, a3);
      poly = vec_madd(poly, t, a2);
      poly = vec_madd(poly, t, a1);

      // full polynomial: poly = poly * t
      poly = vec_mul(poly, t);

      // Compute exp(-x^2)
      __vector float x2 = vec_mul(x, x);
      __vector float neg_x2 = vec_neg(x2);

      f32x4x2_t tmp;
      tmp.val[0] = neg_x2;
      tmp.val[1] = neg_x2;
      FP32Vec8 exp_neg_x2(tmp);

      FP32Vec8 e = exp_neg_x2.exp();
      __vector float ex = e.reg.val[lane];

      // erf(x) = sign * (1 - poly * exp(-x^2))
      __vector float term = vec_mul(poly, ex);
      __vector float y = vec_sub(one, term);
      y = vec_mul(y, sign);

      // saturated → ±1
      __vector float sat_val = vec_mul(sign, one);
      out.val[lane] = vec_sel(y, sat_val, saturated);
    }

    return FP32Vec8(out);
  }
  // Elementwise sigmoid(x) = 1 / (1 + exp(-x))
  FP32Vec8 sigmoid() const {
    const __vector float one = vec_splats(1.0f);

    f32x4x2_t neg;
    for (int i = 0; i < 2; ++i) {
      neg.val[i] = vec_neg(reg.val[i]);
    }

    FP32Vec8 neg_x(neg);
    FP32Vec8 e = neg_x.exp();  // exp(-x)

    f32x4x2_t denom;
    for (int i = 0; i < 2; ++i) {
      denom.val[i] = vec_add(one, e.reg.val[i]);
    }

    FP32Vec8 denom_vec(denom);
    FP32Vec8 one_vec(1.0f);

    return one_vec / denom_vec;
  }

  // Tanh-based GELU:
  // gelu(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
  FP32Vec8 gelu_tanh() const {
    const __vector float k_s2pi = vec_splats(0.7978845608028654f);  // √(2/π)
    const __vector float k_0_0447 = vec_splats(0.044715f);

    f32x4x2_t x2, x3, inner;
    for (int i = 0; i < 2; ++i) {
      __vector float x = reg.val[i];
      x2.val[i] = vec_mul(x, x);                            // x^2
      x3.val[i] = vec_mul(x2.val[i], x);                    // x^3
      __vector float t = vec_madd(k_0_0447, x3.val[i], x);  // x + 0.044715*x^3
      inner.val[i] = vec_mul(k_s2pi, t);                    // √(2/π)*(...)
    }

    FP32Vec8 inner_vec(inner);
    FP32Vec8 t = inner_vec.tanh();  // tanh part

    FP32Vec8 one_vec(1.0f);
    FP32Vec8 half_vec(0.5f);

    FP32Vec8 x_vec(*this);
    return x_vec * half_vec * (one_vec + t);
  }

  // Erf-based GELU:
  // gelu(x) = 0.5 * x * (1 + erf(x / √2))
  FP32Vec8 gelu_erf() const {
    const __vector float inv_sqrt2 = vec_splats(0.7071067811865476f);  // 1/√2
    FP32Vec8 x_vec(*this);

    f32x4x2_t scaled;
    for (int i = 0; i < 2; ++i) {
      scaled.val[i] = vec_mul(reg.val[i], inv_sqrt2);
    }
    FP32Vec8 x_scaled(scaled);

    FP32Vec8 erf_x = x_scaled.er();

    FP32Vec8 one_vec(1.0f);
    FP32Vec8 half_vec(0.5f);

    return x_vec * half_vec * (one_vec + erf_x);
  }

  // Elementwise reciprocal: 1/x (scalar per lane, for correctness)
  FP32Vec8 rcp() const {
    AliasReg in, out;
    in.reg = reg;

    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      out.values[i] = 1.0f / in.values[i];
    }
    return FP32Vec8(out.reg);
  }

  // Elementwise rsqrt(x) = 1 / sqrt(x) (scalar per lane, for correctness)
  FP32Vec8 rsqrt() const {
    AliasReg in, out;
    in.reg = reg;

    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      out.values[i] = 1.0f / std::sqrt(in.values[i]);
    }
    return FP32Vec8(out.reg);
  }

  FP32Vec8 operator*(const FP32Vec8& b) const {
    return FP32Vec8(
        {vec_mul(reg.val[0], b.reg.val[0]), vec_mul(reg.val[1], b.reg.val[1])});
  }

  FP32Vec8 operator+(const FP32Vec8& b) const {
    return FP32Vec8(
        {vec_add(reg.val[0], b.reg.val[0]), vec_add(reg.val[1], b.reg.val[1])});
  }

  FP32Vec8 operator-(const FP32Vec8& b) const {
    return FP32Vec8(
        {vec_sub(reg.val[0], b.reg.val[0]), vec_sub(reg.val[1], b.reg.val[1])});
  }

  FP32Vec8 operator/(const FP32Vec8& b) const {
    return FP32Vec8(
        {vec_div(reg.val[0], b.reg.val[0]), vec_div(reg.val[1], b.reg.val[1])});
  }

  void save(float* ptr) const {
    vec_xst(reg.val[0], 0, ptr);
    vec_xst(reg.val[1], 16, ptr);
  }
};

struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    f32x4x4_t reg;
    float values[VEC_ELEM_NUM];
  };

  f32x4x4_t reg;

  explicit FP32Vec16(float v) {
    reg.val[0] = vec_splats(v);
    reg.val[1] = vec_splats(v);
    reg.val[2] = vec_splats(v);
    reg.val[3] = vec_splats(v);
  }

  explicit FP32Vec16() {
    reg.val[0] = vec_splats(0.0f);
    reg.val[1] = vec_splats(0.0f);
    reg.val[2] = vec_splats(0.0f);
    reg.val[3] = vec_splats(0.0f);
  }

  explicit FP32Vec16(const float* ptr) {
    reg.val[0] = vec_xl(0, ptr);
    reg.val[1] = vec_xl(16, ptr);
    reg.val[2] = vec_xl(32, ptr);
    reg.val[3] = vec_xl(48, ptr);
  }

  explicit FP32Vec16(f32x4x4_t data) : reg(data) {}

  explicit FP32Vec16(const FP32Vec16& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[1];
    reg.val[2] = data.reg.val[2];
    reg.val[3] = data.reg.val[3];
  }

  explicit FP32Vec16(const FP32Vec4& data) {
    reg.val[0] = data.reg;
    reg.val[1] = data.reg;
    reg.val[2] = data.reg;
    reg.val[3] = data.reg;
  }

  explicit FP32Vec16(const FP32Vec8& data) {
    reg.val[0] = data.reg.val[0];
    reg.val[1] = data.reg.val[1];
    reg.val[2] = data.reg.val[0];
    reg.val[3] = data.reg.val[1];
  }

  explicit FP32Vec16(const BF16Vec16& v) {
    // On big-endian s390x, place BF16 first to get correct byte order
    reg.val[0] = (__vector float)vec_mergeh(v.reg.val[0], zero);
    reg.val[1] = (__vector float)vec_mergel(v.reg.val[0], zero);
    reg.val[2] = (__vector float)vec_mergeh(v.reg.val[1], zero);
    reg.val[3] = (__vector float)vec_mergel(v.reg.val[1], zero);
  }

  explicit FP32Vec16(const FP16Vec16& v) {
    __vector unsigned int raw_hi_0 =
        (__vector unsigned int)vec_unpackh(v.reg.val[0]);
    __vector unsigned int raw_lo_0 =
        (__vector unsigned int)vec_unpackl(v.reg.val[0]);
    reg.val[0] = fp16_to_fp32_bits(raw_hi_0);
    reg.val[1] = fp16_to_fp32_bits(raw_lo_0);

    __vector unsigned int raw_hi_1 =
        (__vector unsigned int)vec_unpackh(v.reg.val[1]);
    __vector unsigned int raw_lo_1 =
        (__vector unsigned int)vec_unpackl(v.reg.val[1]);
    reg.val[2] = fp16_to_fp32_bits(raw_hi_1);
    reg.val[3] = fp16_to_fp32_bits(raw_lo_1);
  }

  explicit FP32Vec16(const BF16Vec8& v) : FP32Vec16(FP32Vec8(v)) {}

  FP32Vec16 operator*(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_mul(reg.val[0], b.reg.val[0]),
                                vec_mul(reg.val[1], b.reg.val[1]),
                                vec_mul(reg.val[2], b.reg.val[2]),
                                vec_mul(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 operator+(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_add(reg.val[0], b.reg.val[0]),
                                vec_add(reg.val[1], b.reg.val[1]),
                                vec_add(reg.val[2], b.reg.val[2]),
                                vec_add(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_sub(reg.val[0], b.reg.val[0]),
                                vec_sub(reg.val[1], b.reg.val[1]),
                                vec_sub(reg.val[2], b.reg.val[2]),
                                vec_sub(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_div(reg.val[0], b.reg.val[0]),
                                vec_div(reg.val[1], b.reg.val[1]),
                                vec_div(reg.val[2], b.reg.val[2]),
                                vec_div(reg.val[3], b.reg.val[3])}));
  }

  float reduce_sum() const {
    AliasReg ar;
    ar.reg = reg;
    float result = 0;
    unroll_loop<int, VEC_ELEM_NUM>(
        [&result, &ar](int i) { result += ar.values[i]; });

    return result;
  }

  template <int group_size>
  float reduce_sub_sum(int idx) {
    static_assert(VEC_ELEM_NUM % group_size == 0);

    AliasReg ar;
    ar.reg = reg;
    float result = 0;
    const int start = idx * group_size;
    unroll_loop<int, group_size>(
        [&result, &start, ar](int i) { result += ar.values[start + i]; });

    return result;
  }

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_max(reg.val[0], b.reg.val[0]),
                                vec_max(reg.val[1], b.reg.val[1]),
                                vec_max(reg.val[2], b.reg.val[2]),
                                vec_max(reg.val[3], b.reg.val[3])}));
  }

  float reduce_max() const {
    AliasReg ar;
    ar.reg = reg;
    float result = ar.values[0];
    unroll_loop<int, VEC_ELEM_NUM>([&result, &ar](int i) {
      if (ar.values[i] > result) result = ar.values[i];
    });
    return result;
  }

  void save(float* ptr) const {
    vec_xst(reg.val[0], 0, ptr);
    vec_xst(reg.val[1], 16, ptr);
    vec_xst(reg.val[2], 32, ptr);
    vec_xst(reg.val[3], 48, ptr);
  }
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
struct VecType<c10::BFloat16> {
  using vec_type = BF16Vec8;
};

template <>
struct VecType<c10::Half> {
  using vec_type = FP16Vec8;
};

template <typename T>
void storeFP32(float v, T* ptr) {
  *ptr = v;
}

namespace c10 {
struct BFloat16 {
  uint16_t value;  // Assume BFloat16 is defined as a struct containing a 16-bit
                   // value.
};
}  // namespace c10

template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  c10::BFloat16 __attribute__((__may_alias__))* v_ptr =
      reinterpret_cast<c10::BFloat16*>(&v);
  *ptr = *(v_ptr + 1);
}

template <>
inline void storeFP32<::c10::Half>(float v, ::c10::Half* ptr) {
  // Use bit-manipulation for IEEE FP32 to FP16 conversion since vector
  // intrinsics for FP32 to FP16 conversion does not use IEEE rounding and can
  // produce incorrect results for some inputs. Process each of the 4 vectors
  // separately.
  uint32_t in;
  std::memcpy(&in, &v, sizeof(in));

  uint32_t s = (in & 0x80000000) >> 16;  // Sign
  uint32_t e = (in & 0x7F800000) >> 23;  // Exponent
  uint32_t round_bit = (in >> 12) & 1;
  uint32_t sticky = (in & 0xFFF) != 0;  // Any bits in [11..0]
  uint32_t m = (in & 0x007FFFFF) >> 13;
  uint32_t lsb = m & 1;  // LSB of mantissa for tie-breaking

  // Check for NaN/Inf before rounding
  bool is_nan_inf = (e == 0xFF);

  if (round_bit && (sticky || lsb)) {
    m++;
    // Handle mantissa overflow: if m overflows 10 bits, increment exponent
    if (m > 0x3FF) {
      m = 0;
      e++;
    }
  }

  if (is_nan_inf) {
    // NaN/Inf: preserve it
    e = 0x1F;
  } else {
    // Normal: adjust bias (127 - 15), flush subnormals to zero
    e = (e >= 112) ? (e - 112) : 0;
    // If exponent overflows to Inf range, saturate to max normal FP16 value
    if (e > 0x1E) {
      e = 0x1E;   // Max normal exponent
      m = 0x3FF;  // Max mantissa
    }
  }

  uint16_t fp16 = (uint16_t)(s | (e << 10) | m);

  *reinterpret_cast<uint16_t*>(ptr) = fp16;
}

#ifndef __VEC_CLASS_FP_NAN
  #define __VEC_CLASS_FP_NAN (1 << 6)
#endif

// Optimized FMA (Fused Multiply-Add) implementations using IBM Z vector
// intrinsics

// FP32Vec4 FMA: acc = acc + (a * b) or equivalently acc = fma(a, b, acc)
FORCE_INLINE void fma(FP32Vec4& acc, const FP32Vec4& a, const FP32Vec4& b) {
  acc.reg = vec_madd(a.reg, b.reg, acc.reg);
}

// FP32Vec8 FMA: acc = acc + (a * b)
FORCE_INLINE void fma(FP32Vec8& acc, const FP32Vec8& a, const FP32Vec8& b) {
  acc.reg.val[0] = vec_madd(a.reg.val[0], b.reg.val[0], acc.reg.val[0]);
  acc.reg.val[1] = vec_madd(a.reg.val[1], b.reg.val[1], acc.reg.val[1]);
}

// FP32Vec16 FMA: acc = acc + (a * b)
FORCE_INLINE void fma(FP32Vec16& acc, const FP32Vec16& a, const FP32Vec16& b) {
  acc.reg.val[0] = vec_madd(a.reg.val[0], b.reg.val[0], acc.reg.val[0]);
  acc.reg.val[1] = vec_madd(a.reg.val[1], b.reg.val[1], acc.reg.val[1]);
  acc.reg.val[2] = vec_madd(a.reg.val[2], b.reg.val[2], acc.reg.val[2]);
  acc.reg.val[3] = vec_madd(a.reg.val[3], b.reg.val[3], acc.reg.val[3]);
}

// Multiply-Subtract: acc = acc - (a * b)
FORCE_INLINE void fms(FP32Vec4& acc, const FP32Vec4& a, const FP32Vec4& b) {
  acc.reg = vec_msub(a.reg, b.reg, acc.reg);
}

FORCE_INLINE void fms(FP32Vec8& acc, const FP32Vec8& a, const FP32Vec8& b) {
  acc.reg.val[0] = vec_msub(a.reg.val[0], b.reg.val[0], acc.reg.val[0]);
  acc.reg.val[1] = vec_msub(a.reg.val[1], b.reg.val[1], acc.reg.val[1]);
}

FORCE_INLINE void fms(FP32Vec16& acc, const FP32Vec16& a, const FP32Vec16& b) {
  acc.reg.val[0] = vec_msub(a.reg.val[0], b.reg.val[0], acc.reg.val[0]);
  acc.reg.val[1] = vec_msub(a.reg.val[1], b.reg.val[1], acc.reg.val[1]);
  acc.reg.val[2] = vec_msub(a.reg.val[2], b.reg.val[2], acc.reg.val[2]);
  acc.reg.val[3] = vec_msub(a.reg.val[3], b.reg.val[3], acc.reg.val[3]);
}

// Negative Multiply-Add: acc = -(a * b) + acc
FORCE_INLINE void nfma(FP32Vec4& acc, const FP32Vec4& a, const FP32Vec4& b) {
  acc.reg = vec_nmadd(a.reg, b.reg, acc.reg);
}

FORCE_INLINE void nfma(FP32Vec8& acc, const FP32Vec8& a, const FP32Vec8& b) {
  acc.reg.val[0] = vec_nmadd(a.reg.val[0], b.reg.val[0], acc.reg.val[0]);
  acc.reg.val[1] = vec_nmadd(a.reg.val[1], b.reg.val[1], acc.reg.val[1]);
}

FORCE_INLINE void nfma(FP32Vec16& acc, const FP32Vec16& a, const FP32Vec16& b) {
  acc.reg.val[0] = vec_nmadd(a.reg.val[0], b.reg.val[0], acc.reg.val[0]);
  acc.reg.val[1] = vec_nmadd(a.reg.val[1], b.reg.val[1], acc.reg.val[1]);
  acc.reg.val[2] = vec_nmadd(a.reg.val[2], b.reg.val[2], acc.reg.val[2]);
  acc.reg.val[3] = vec_nmadd(a.reg.val[3], b.reg.val[3], acc.reg.val[3]);
}

// Negative Multiply-Subtract: acc = -(a * b) - acc
FORCE_INLINE void nfms(FP32Vec4& acc, const FP32Vec4& a, const FP32Vec4& b) {
  acc.reg = vec_nmsub(a.reg, b.reg, acc.reg);
}

FORCE_INLINE void nfms(FP32Vec8& acc, const FP32Vec8& a, const FP32Vec8& b) {
  acc.reg.val[0] = vec_nmsub(a.reg.val[0], b.reg.val[0], acc.reg.val[0]);
  acc.reg.val[1] = vec_nmsub(a.reg.val[1], b.reg.val[1], acc.reg.val[1]);
}

FORCE_INLINE void nfms(FP32Vec16& acc, const FP32Vec16& a, const FP32Vec16& b) {
  acc.reg.val[0] = vec_nmsub(a.reg.val[0], b.reg.val[0], acc.reg.val[0]);
  acc.reg.val[1] = vec_nmsub(a.reg.val[1], b.reg.val[1], acc.reg.val[1]);
  acc.reg.val[2] = vec_nmsub(a.reg.val[2], b.reg.val[2], acc.reg.val[2]);
  acc.reg.val[3] = vec_nmsub(a.reg.val[3], b.reg.val[3], acc.reg.val[3]);
}

const static __vector unsigned char omask = {2,  3,  6,  7,  10, 11, 14, 15,
                                             18, 19, 22, 23, 26, 27, 30, 31};
const static __vector unsigned int bias = {0x00007fff, 0x00007fff, 0x00007fff,
                                           0x00007fff};
const static __vector unsigned int nan = {0x7fc00000, 0x7fc00000, 0x7fc00000,
                                          0x7fc00000};
const static __vector unsigned int sh16 = {16, 16, 16, 16};
const static __vector unsigned int one = {1, 1, 1, 1};

inline BF16Vec8::BF16Vec8(const FP32Vec8& v) {
  __vector unsigned int inp0 = (__vector unsigned int)(v.reg.val[0]);
  __vector unsigned int inp1 = (__vector unsigned int)(v.reg.val[1]);
  __vector unsigned int lsb0 = inp0 >> sh16;
  __vector unsigned int lsb1 = inp1 >> sh16;
  lsb0 = lsb0 & one;
  lsb1 = lsb1 & one;
  __vector unsigned int rnd0 = lsb0 + bias;
  __vector unsigned int rnd1 = lsb1 + bias;
  inp0 = inp0 + rnd0;
  inp1 = inp1 + rnd1;
  int cc;
  __vector __bool int sel0 =
      vec_fp_test_data_class(v.reg.val[0], __VEC_CLASS_FP_NAN, &cc);
  __vector __bool int sel1 =
      vec_fp_test_data_class(v.reg.val[1], __VEC_CLASS_FP_NAN, &cc);
  inp0 = vec_sel(inp0, nan, sel0);
  inp1 = vec_sel(inp1, nan, sel1);
  inp0 = inp0 >> sh16;
  inp1 = inp1 >> sh16;

  reg = (__vector signed short)vec_perm(inp0, inp1, omask);
}

inline BF16Vec16::BF16Vec16(const FP32Vec16& v) {
  __vector unsigned int inp0 = (__vector unsigned int)(v.reg.val[0]);
  __vector unsigned int inp1 = (__vector unsigned int)(v.reg.val[1]);
  __vector unsigned int inp2 = (__vector unsigned int)(v.reg.val[2]);
  __vector unsigned int inp3 = (__vector unsigned int)(v.reg.val[3]);
  __vector unsigned int lsb0 = inp0 >> sh16;
  __vector unsigned int lsb1 = inp1 >> sh16;
  __vector unsigned int lsb2 = inp2 >> sh16;
  __vector unsigned int lsb3 = inp3 >> sh16;
  lsb0 = lsb0 & one;
  lsb1 = lsb1 & one;
  lsb2 = lsb2 & one;
  lsb3 = lsb3 & one;
  __vector unsigned int rnd0 = lsb0 + bias;
  __vector unsigned int rnd1 = lsb1 + bias;
  __vector unsigned int rnd2 = lsb2 + bias;
  __vector unsigned int rnd3 = lsb3 + bias;
  inp0 = inp0 + rnd0;
  inp1 = inp1 + rnd1;
  inp2 = inp2 + rnd2;
  inp3 = inp3 + rnd3;
  int cc;
  __vector __bool int sel0 =
      vec_fp_test_data_class(v.reg.val[0], __VEC_CLASS_FP_NAN, &cc);
  __vector __bool int sel1 =
      vec_fp_test_data_class(v.reg.val[1], __VEC_CLASS_FP_NAN, &cc);
  __vector __bool int sel2 =
      vec_fp_test_data_class(v.reg.val[2], __VEC_CLASS_FP_NAN, &cc);
  __vector __bool int sel3 =
      vec_fp_test_data_class(v.reg.val[3], __VEC_CLASS_FP_NAN, &cc);
  inp0 = vec_sel(inp0, nan, sel0);
  inp1 = vec_sel(inp1, nan, sel1);
  inp2 = vec_sel(inp2, nan, sel2);
  inp3 = vec_sel(inp3, nan, sel3);
  inp0 = inp0 >> sh16;
  inp1 = inp1 >> sh16;
  inp2 = inp2 >> sh16;
  inp3 = inp3 >> sh16;

  reg.val[0] = (__vector signed short)vec_perm(inp0, inp1, omask);
  reg.val[1] = (__vector signed short)vec_perm(inp2, inp3, omask);
}

inline FP16Vec8::FP16Vec8(const FP32Vec8& v) {
  // Use bit-manipulation for IEEE FP32 to FP16 conversion since vector
  // intrinsics for FP32 to FP16 conversion does not use IEEE rounding and can
  // produce incorrect results for some inputs. Process each of the 4 vectors
  // separately.
  __vector unsigned int res_hi = fp32_to_fp16_bits(v.reg.val[0]);
  __vector unsigned int res_lo = fp32_to_fp16_bits(v.reg.val[1]);

  const __vector unsigned char perm_pack = {
      2,  3,  6,  7,  10, 11, 14, 15,  // Select lower 2 bytes from res_hi
      18, 19, 22, 23, 26, 27, 30, 31   // Select lower 2 bytes from res_lo
  };

  reg = vec_perm((__vector signed short)res_hi, (__vector signed short)res_lo,
                 perm_pack);
}

inline FP16Vec16::FP16Vec16(const FP32Vec16& v) {
  // Use bit-manipulation for IEEE FP32 to FP16 conversion since vector
  // intrinsics for FP32 to FP16 conversion does not use IEEE rounding and can
  // produce incorrect results for some inputs. Process each of the 4 vectors
  // separately.
  __vector unsigned int res_0 = fp32_to_fp16_bits(v.reg.val[0]);
  __vector unsigned int res_1 = fp32_to_fp16_bits(v.reg.val[1]);
  __vector unsigned int res_2 = fp32_to_fp16_bits(v.reg.val[2]);
  __vector unsigned int res_3 = fp32_to_fp16_bits(v.reg.val[3]);

  const __vector unsigned char perm_pack = {
      2,  3,  6,  7,  10, 11, 14, 15,  // Lower 2 bytes from first vector
      18, 19, 22, 23, 26, 27, 30, 31   // Lower 2 bytes from second vector
  };

  reg.val[0] = vec_perm((__vector signed short)res_0,
                        (__vector signed short)res_1, perm_pack);
  reg.val[1] = vec_perm((__vector signed short)res_2,
                        (__vector signed short)res_3, perm_pack);
}

// 1D softmax over `n` elements in `input`, writes result to `output`.
// Uses FP32Vec8 for main body, scalar tail handling.
// Requirement: n > 0
FORCE_INLINE void softmax_fp32vec8(float* output, const float* input, int n) {
  if (n <= 0) return;

  // ---------- Pass 1: find max ----------
  float max_val = -std::numeric_limits<float>::infinity();
  int i = 0;

  for (; i + FP32Vec8::VEC_ELEM_NUM <= n; i += FP32Vec8::VEC_ELEM_NUM) {
    FP32Vec8 v(input + i);
    FP32Vec8::AliasReg ar;
    ar.reg = v.reg;
    for (int j = 0; j < FP32Vec8::VEC_ELEM_NUM; ++j) {
      if (ar.values[j] > max_val) max_val = ar.values[j];
    }
  }
  for (; i < n; ++i) {
    if (input[i] > max_val) max_val = input[i];
  }

  // ---------- Pass 2: compute exp(x - max) and sum ----------
  float sum = 0.0f;
  i = 0;

  for (; i + FP32Vec8::VEC_ELEM_NUM <= n; i += FP32Vec8::VEC_ELEM_NUM) {
    float tmp[FP32Vec8::VEC_ELEM_NUM];
    for (int j = 0; j < FP32Vec8::VEC_ELEM_NUM; ++j) {
      tmp[j] = input[i + j] - max_val;
    }

    FP32Vec8 v(tmp);
    FP32Vec8 e = v.exp();

    FP32Vec8::AliasReg ar;
    ar.reg = e.reg;
    for (int j = 0; j < FP32Vec8::VEC_ELEM_NUM; ++j) {
      output[i + j] = ar.values[j];
      sum += ar.values[j];
    }
  }

  // Tail
  for (; i < n; ++i) {
    float x = input[i] - max_val;
    float ex = std::exp(x);  // scalar tail
    output[i] = ex;
    sum += ex;
  }

  // ---------- Pass 3: normalize ----------
  float inv_sum = 1.0f / sum;
  i = 0;

  for (; i + FP32Vec8::VEC_ELEM_NUM <= n; i += FP32Vec8::VEC_ELEM_NUM) {
    float tmp[FP32Vec8::VEC_ELEM_NUM];
    for (int j = 0; j < FP32Vec8::VEC_ELEM_NUM; ++j) {
      tmp[j] = output[i + j] * inv_sum;
    }
    FP32Vec8 v(tmp);
    v.save(output + i);
  }

  for (; i < n; ++i) {
    output[i] *= inv_sum;
  }
}

// 1D RMSNorm kernel:
//   input:  x[0..n-1]
//   weight: w[0..n-1] (gamma), may be nullptr
//   output: y[i] = x[i] * inv_rms * (weight[i] if weight != nullptr else 1)
//   eps: small epsilon for numerical stability
FORCE_INLINE void rmsnorm_fp32vec8(float* output, const float* input,
                                   const float* weight, int n, float eps) {
  if (n <= 0) return;

  // ---------- Pass 1: compute sum of squares ----------
  float sum_sq = 0.0f;
  int i = 0;

  for (; i + FP32Vec8::VEC_ELEM_NUM <= n; i += FP32Vec8::VEC_ELEM_NUM) {
    FP32Vec8 x_vec(input + i);

    FP32Vec8 sq = x_vec * x_vec;

    FP32Vec8::AliasReg ar;
    ar.reg = sq.reg;
    for (int j = 0; j < FP32Vec8::VEC_ELEM_NUM; ++j) {
      sum_sq += ar.values[j];
    }
  }

  // Tail
  for (; i < n; ++i) {
    float v = input[i];
    sum_sq += v * v;
  }

  float mean_sq = sum_sq / static_cast<float>(n);
  float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

  // ---------- Pass 2: scale (and apply weight if given) ----------
  const float inv_rms_f = inv_rms;
  i = 0;

  if (weight) {
    // with gamma
    for (; i + FP32Vec8::VEC_ELEM_NUM <= n; i += FP32Vec8::VEC_ELEM_NUM) {
      FP32Vec8 x_vec(input + i);

      float wtmp[FP32Vec8::VEC_ELEM_NUM];
      for (int j = 0; j < FP32Vec8::VEC_ELEM_NUM; ++j) {
        wtmp[j] = weight[i + j];
      }
      FP32Vec8 w_vec(wtmp);

      FP32Vec8 scale_vec(inv_rms_f);
      FP32Vec8 y = x_vec * scale_vec * w_vec;
      y.save(output + i);
    }

    for (; i < n; ++i) {
      output[i] = input[i] * inv_rms_f * weight[i];
    }
  } else {
    // without gamma
    for (; i + FP32Vec8::VEC_ELEM_NUM <= n; i += FP32Vec8::VEC_ELEM_NUM) {
      FP32Vec8 x_vec(input + i);
      FP32Vec8 scale_vec(inv_rms_f);
      FP32Vec8 y = x_vec * scale_vec;
      y.save(output + i);
    }

    for (; i < n; ++i) {
      output[i] = input[i] * inv_rms_f;
    }
  }
}

// Prefetch data to cache for better memory access performance
FORCE_INLINE void prefetch(const void* addr) {
  __builtin_prefetch(addr, 0, 3);  // 0=read, 3=high temporal locality
}

};  // namespace vec_op

#endif