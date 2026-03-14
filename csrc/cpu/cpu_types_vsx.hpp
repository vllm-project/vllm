
#ifndef CPU_TYPES_VSX_HPP
#define CPU_TYPES_VSX_HPP

#include <altivec.h>
#include <cmath>
#include <algorithm>
#include <torch/all.h>

namespace vec_op {

// FIXME: FP16 is not fully supported in Torch-CPU
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

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

typedef struct i32x4x4_t {
  __vector int32_t val[4];
} i32x4x4_t;

struct FP32Vec8;
struct FP32Vec16;

struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  __vector signed short reg;

  explicit BF16Vec8(const void* ptr)
      : reg((__vector signed short)vec_xl(0, (__vector signed short*)ptr)) {}

  explicit BF16Vec8(const FP32Vec8&);

  void save(void* ptr) const {
    *reinterpret_cast<__vector signed short*>(ptr) = reg;
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

  void save(void* ptr, const int elem_num) const {
    const int clamped_elem = std::max(0, std::min(elem_num, 16));

    // Calculate elements to store in each 128-bit part (8 elements each)
    const int elements_val0 = std::min(clamped_elem, 8);
    const int elements_val1 = std::max(clamped_elem - 8, 0);

    // Convert elements to bytes (2 bytes per element)
    const size_t bytes_val0 = elements_val0 * sizeof(signed short);
    const size_t bytes_val1 = elements_val1 * sizeof(signed short);

    signed short* dest = static_cast<signed short*>(ptr);
    // Store the first part using vec_xst_len
    if (bytes_val0 > 0) {
      vec_xst_len(reg.val[0], dest, bytes_val0);
    }
    // Store the second part if needed
    if (bytes_val1 > 0) {
      vec_xst_len(reg.val[1], dest + elements_val0, bytes_val1);
    }
  }
};

const static __vector signed short zero = vec_splats((signed short)0);

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
    reg.val[0] = (__vector float)vec_mergeh(zero, v.reg);
    reg.val[1] = (__vector float)vec_mergel(zero, v.reg);
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
    // TODO: Vectorize this
    AliasReg ar;
    ar.reg = reg;
    f32x4x4_t ret;
    ret.val[0][0] = std::exp(ar.values[0]);
    ret.val[0][1] = std::exp(ar.values[1]);
    ret.val[0][2] = std::exp(ar.values[2]);
    ret.val[0][3] = std::exp(ar.values[3]);
    ret.val[1][0] = std::exp(ar.values[4]);
    ret.val[1][1] = std::exp(ar.values[5]);
    ret.val[1][2] = std::exp(ar.values[6]);
    ret.val[1][3] = std::exp(ar.values[7]);
    return FP32Vec8(f32x4x2_t({ret.val[0], ret.val[1]}));
  }

  FP32Vec8 tanh() const {
    // TODO: Vectorize this
    AliasReg ar;
    ar.reg = reg;
    f32x4x4_t ret;
    ret.val[0][0] = std::tanh(ar.values[0]);
    ret.val[0][1] = std::tanh(ar.values[1]);
    ret.val[0][2] = std::tanh(ar.values[2]);
    ret.val[0][3] = std::tanh(ar.values[3]);
    ret.val[1][0] = std::tanh(ar.values[4]);
    ret.val[1][1] = std::tanh(ar.values[5]);
    ret.val[1][2] = std::tanh(ar.values[6]);
    ret.val[1][3] = std::tanh(ar.values[7]);
    return FP32Vec8(f32x4x2_t({ret.val[0], ret.val[1]}));
  }

  FP32Vec8 er() const {
    // TODO: Vectorize this
    AliasReg ar;
    ar.reg = reg;
    f32x4x4_t ret;
    ret.val[0][0] = std::erf(ar.values[0]);
    ret.val[0][1] = std::erf(ar.values[1]);
    ret.val[0][2] = std::erf(ar.values[2]);
    ret.val[0][3] = std::erf(ar.values[3]);
    ret.val[1][0] = std::erf(ar.values[4]);
    ret.val[1][1] = std::erf(ar.values[5]);
    ret.val[1][2] = std::erf(ar.values[6]);
    ret.val[1][3] = std::erf(ar.values[7]);
    return FP32Vec8(f32x4x2_t({ret.val[0], ret.val[1]}));
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

struct INT32Vec16 : public Vec<INT32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;
  union AliasReg {
    i32x4x4_t reg;
    int32_t values[VEC_ELEM_NUM];
  };

  i32x4x4_t reg;

  explicit INT32Vec16(const void* data_ptr) {
    reg.val[0] = vec_xl(0, reinterpret_cast<const __vector int32_t*>(data_ptr));
    reg.val[1] =
        vec_xl(16, reinterpret_cast<const __vector int32_t*>(data_ptr));
    reg.val[2] =
        vec_xl(32, reinterpret_cast<const __vector int32_t*>(data_ptr));
    reg.val[3] =
        vec_xl(48, reinterpret_cast<const __vector int32_t*>(data_ptr));
  }

  void save(int32_t* ptr) const {
    vec_xst(reg.val[0], 0, reinterpret_cast<__vector int32_t*>(ptr));
    vec_xst(reg.val[1], 16, reinterpret_cast<__vector int32_t*>(ptr));
    vec_xst(reg.val[2], 32, reinterpret_cast<__vector int32_t*>(ptr));
    vec_xst(reg.val[3], 48, reinterpret_cast<__vector int32_t*>(ptr));
  }

  void save(int32_t* ptr, const int elem_num) const {
    const int elements_in_chunk1 =
        (elem_num >= 0) ? ((elem_num >= 4) ? 4 : elem_num) : 0;
    const int elements_in_chunk2 =
        (elem_num > 4) ? ((elem_num >= 8) ? 4 : elem_num - 4) : 0;
    const int elements_in_chunk3 =
        (elem_num > 8) ? ((elem_num >= 12) ? 4 : elem_num - 8) : 0;
    const int elements_in_chunk4 =
        (elem_num > 12) ? ((elem_num >= 16) ? 4 : elem_num - 12) : 0;

    const size_t bytes_chunk1 =
        static_cast<size_t>(elements_in_chunk1 * sizeof(int32_t));
    const size_t bytes_chunk2 =
        static_cast<size_t>(elements_in_chunk2 * sizeof(int32_t));
    const size_t bytes_chunk3 =
        static_cast<size_t>(elements_in_chunk3 * sizeof(int32_t));
    const size_t bytes_chunk4 =
        static_cast<size_t>(elements_in_chunk4 * sizeof(int32_t));

    vec_xst_len(reg.val[0], reinterpret_cast<int32_t*>(ptr), bytes_chunk1);
    vec_xst_len(reg.val[1],
                reinterpret_cast<int32_t*>(reinterpret_cast<char*>(ptr) + 16),
                bytes_chunk2);
    vec_xst_len(reg.val[2],
                reinterpret_cast<int32_t*>(reinterpret_cast<char*>(ptr) + 32),
                bytes_chunk3);
    vec_xst_len(reg.val[3],
                reinterpret_cast<int32_t*>(reinterpret_cast<char*>(ptr) + 48),
                bytes_chunk4);
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
    reg.val[0] = (__vector float)vec_mergeh(zero, v.reg.val[0]);
    reg.val[1] = (__vector float)vec_mergel(zero, v.reg.val[0]);
    reg.val[2] = (__vector float)vec_mergeh(zero, v.reg.val[1]);
    reg.val[3] = (__vector float)vec_mergel(zero, v.reg.val[1]);
  }

  explicit FP32Vec16(const BF16Vec8& v) : FP32Vec16(FP32Vec8(v)) {}

  explicit FP32Vec16(const INT32Vec16& v) {
    reg.val[0] = vec_ctf(v.reg.val[0], 0);
    reg.val[1] = vec_ctf(v.reg.val[1], 0);
    reg.val[2] = vec_ctf(v.reg.val[2], 0);
    reg.val[3] = vec_ctf(v.reg.val[3], 0);
  }

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

  FP32Vec16 clamp(const FP32Vec16& min, const FP32Vec16& max) const {
    return FP32Vec16(f32x4x4_t(
        {vec_min(max.reg.val[0], vec_max(min.reg.val[0], reg.val[0])),
         vec_min(max.reg.val[1], vec_max(min.reg.val[1], reg.val[1])),
         vec_min(max.reg.val[2], vec_max(min.reg.val[2], reg.val[2])),
         vec_min(max.reg.val[3], vec_max(min.reg.val[3], reg.val[3]))}));
  }

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_max(reg.val[0], b.reg.val[0]),
                                vec_max(reg.val[1], b.reg.val[1]),
                                vec_max(reg.val[2], b.reg.val[2]),
                                vec_max(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 max(const FP32Vec16& b, int elem_num) const {
    FP32Vec16 result;

    // Create a vector of element indices for each chunk
    __vector unsigned int indices = {0, 1, 2, 3};
    __vector unsigned int elem_num_vec =
        vec_splats(static_cast<unsigned int>(elem_num));

    // Compute masks for each chunk
    __vector unsigned int chunk_offset0 = {0, 0, 0,
                                           0};  // Chunk 0: Elements 0-3
    __vector unsigned int chunk_offset1 = {4, 4, 4,
                                           4};  // Chunk 1: Elements 4-7
    __vector unsigned int chunk_offset2 = {8, 8, 8,
                                           8};  // Chunk 2: Elements 8-11
    __vector unsigned int chunk_offset3 = {12, 12, 12,
                                           12};  // Chunk 3: Elements 12-15

    // Compute masks for each chunk
    __vector bool int mask0 = vec_cmplt(indices + chunk_offset0, elem_num_vec);
    __vector bool int mask1 = vec_cmplt(indices + chunk_offset1, elem_num_vec);
    __vector bool int mask2 = vec_cmplt(indices + chunk_offset2, elem_num_vec);
    __vector bool int mask3 = vec_cmplt(indices + chunk_offset3, elem_num_vec);

    // Apply masks to compute the result for each chunk
    result.reg.val[0] = vec_sel(this->reg.val[0],
                                vec_max(this->reg.val[0], b.reg.val[0]), mask0);
    result.reg.val[1] = vec_sel(this->reg.val[1],
                                vec_max(this->reg.val[1], b.reg.val[1]), mask1);
    result.reg.val[2] = vec_sel(this->reg.val[2],
                                vec_max(this->reg.val[2], b.reg.val[2]), mask2);
    result.reg.val[3] = vec_sel(this->reg.val[3],
                                vec_max(this->reg.val[3], b.reg.val[3]), mask3);

    return FP32Vec16(result.reg);
  }

  FP32Vec16 min(const FP32Vec16& b) const {
    return FP32Vec16(f32x4x4_t({vec_min(reg.val[0], b.reg.val[0]),
                                vec_min(reg.val[1], b.reg.val[1]),
                                vec_min(reg.val[2], b.reg.val[2]),
                                vec_min(reg.val[3], b.reg.val[3])}));
  }

  FP32Vec16 min(const FP32Vec16& b, int elem_num) const {
    FP32Vec16 result;

    vector unsigned int indices = {0, 1, 2, 3};
    vector unsigned int elem_num_vec =
        vec_splats(static_cast<unsigned int>(elem_num));

    vector unsigned int chunk_offset0 = {0, 0, 0, 0};
    vector unsigned int chunk_offset1 = {4, 4, 4, 4};
    vector unsigned int chunk_offset2 = {8, 8, 8, 8};
    vector unsigned int chunk_offset3 = {12, 12, 12, 12};

    vector bool int mask0 = vec_cmplt(indices + chunk_offset0, elem_num_vec);
    vector bool int mask1 = vec_cmplt(indices + chunk_offset1, elem_num_vec);
    vector bool int mask2 = vec_cmplt(indices + chunk_offset2, elem_num_vec);
    vector bool int mask3 = vec_cmplt(indices + chunk_offset3, elem_num_vec);

    result.reg.val[0] = vec_sel(this->reg.val[0],
                                vec_min(this->reg.val[0], b.reg.val[0]), mask0);
    result.reg.val[1] = vec_sel(this->reg.val[1],
                                vec_min(this->reg.val[1], b.reg.val[1]), mask1);
    result.reg.val[2] = vec_sel(this->reg.val[2],
                                vec_min(this->reg.val[2], b.reg.val[2]), mask2);
    result.reg.val[3] = vec_sel(this->reg.val[3],
                                vec_min(this->reg.val[3], b.reg.val[3]), mask3);

    return FP32Vec16(result.reg);
  }

  FP32Vec16 abs() const {
    return FP32Vec16(f32x4x4_t({vec_abs(reg.val[0]), vec_abs(reg.val[1]),
                                vec_abs(reg.val[2]), vec_abs(reg.val[3])}));
  }

  float reduce_max() {
    __vector float max01 = vec_max(reg.val[0], reg.val[1]);
    __vector float max23 = vec_max(reg.val[2], reg.val[3]);
    __vector float max_all = vec_max(max01, max23);
    __vector float temp = vec_max(max_all, vec_sld(max_all, max_all, 8));
    temp = vec_max(temp, vec_sld(temp, temp, 4));
    return vec_extract(temp, 0);
  }

  float reduce_min() {
    __vector float min01 = vec_min(reg.val[0], reg.val[1]);
    __vector float min23 = vec_min(reg.val[2], reg.val[3]);
    __vector float min_all = vec_min(min01, min23);
    __vector float temp = vec_min(min_all, vec_sld(min_all, min_all, 8));
    temp = vec_min(temp, vec_sld(temp, temp, 4));
    return vec_extract(temp, 0);
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

  void save(float* ptr) const {
    vec_xst(reg.val[0], 0, ptr);
    vec_xst(reg.val[1], 16, ptr);
    vec_xst(reg.val[2], 32, ptr);
    vec_xst(reg.val[3], 48, ptr);
  }

  void save(float* ptr, const int elem_num) const {
    const int elements_in_chunk1 =
        (elem_num >= 0) ? ((elem_num >= 4) ? 4 : elem_num) : 0;
    const int elements_in_chunk2 =
        (elem_num > 4) ? ((elem_num >= 8) ? 4 : elem_num - 4) : 0;
    const int elements_in_chunk3 =
        (elem_num > 8) ? ((elem_num >= 12) ? 4 : elem_num - 8) : 0;
    const int elements_in_chunk4 =
        (elem_num > 12) ? ((elem_num >= 16) ? 4 : elem_num - 12) : 0;

    const size_t bytes_chunk1 =
        static_cast<size_t>(elements_in_chunk1 * sizeof(float));
    const size_t bytes_chunk2 =
        static_cast<size_t>(elements_in_chunk2 * sizeof(float));
    const size_t bytes_chunk3 =
        static_cast<size_t>(elements_in_chunk3 * sizeof(float));
    const size_t bytes_chunk4 =
        static_cast<size_t>(elements_in_chunk4 * sizeof(float));

    vec_xst_len(reg.val[0], ptr, bytes_chunk1);
    vec_xst_len(reg.val[1],
                reinterpret_cast<float*>(reinterpret_cast<char*>(ptr) + 16),
                bytes_chunk2);
    vec_xst_len(reg.val[2],
                reinterpret_cast<float*>(reinterpret_cast<char*>(ptr) + 32),
                bytes_chunk3);
    vec_xst_len(reg.val[3],
                reinterpret_cast<float*>(reinterpret_cast<char*>(ptr) + 48),
                bytes_chunk4);
  }
};

struct INT8Vec16 : public Vec<INT8Vec16> {
  constexpr static int VEC_NUM_ELEM = 16;  // 128 bits / 8 bits = 16

  union AliasReg {
    __vector signed char reg;
    int8_t values[VEC_NUM_ELEM];
  };

  __vector signed char reg;

  explicit INT8Vec16(const FP32Vec16& vec) {
    __vector signed int ret[4];
    ret[0] = vec_cts(vec.reg.val[0], 0);
    ret[1] = vec_cts(vec.reg.val[1], 0);
    ret[2] = vec_cts(vec.reg.val[2], 0);
    ret[3] = vec_cts(vec.reg.val[3], 0);

    __vector signed short packed1 = vec_packs(ret[0], ret[1]);
    __vector signed short packed2 = vec_packs(ret[2], ret[3]);

    reg = vec_packs(packed1, packed2);
  }

  void save(void* ptr) const {
    *reinterpret_cast<__vector signed char*>(ptr) = reg;
  }
  void save(signed char* ptr, const int elem_num) {
    vec_xst_len(reg, ptr, static_cast<size_t>(elem_num));
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

template <typename T>
void storeFP32(float v, T* ptr) {
  *ptr = v;
}

inline void fma(FP32Vec16& acc, FP32Vec16& a, FP32Vec16& b) {
  acc = acc + a * b;
}

template <>
inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  c10::BFloat16 __attribute__((__may_alias__))* v_ptr =
      reinterpret_cast<c10::BFloat16*>(&v);
  *ptr = *(v_ptr + 1);
}

#ifndef __VEC_CLASS_FP_NAN
  #define __VEC_CLASS_FP_NAN (1 << 6)
#endif

const static __vector unsigned char omask = {0,  1,  4,  5,  8,  9,  12, 13,
                                             16, 17, 20, 21, 24, 25, 28, 29};
#ifndef _ARCH_PWR10
const static __vector unsigned int bias = {0x00007fff, 0x00007fff, 0x00007fff,
                                           0x00007fff};
const static __vector unsigned int nan = {0x7fc00000, 0x7fc00000, 0x7fc00000,
                                          0x7fc00000};
const static __vector unsigned int sh16 = {16, 16, 16, 16};
const static __vector unsigned int one = {1, 1, 1, 1};
#endif

inline BF16Vec8::BF16Vec8(const FP32Vec8& v) {
#ifdef _ARCH_PWR10
  __vector signed short ret[2];
  ret[0] = (__vector signed short)__builtin_vsx_xvcvspbf16(
      (__vector unsigned char)v.reg.val[0]);
  ret[1] = (__vector signed short)__builtin_vsx_xvcvspbf16(
      (__vector unsigned char)v.reg.val[1]);
  reg = vec_perm(ret[0], ret[1], omask);
#elif defined(_ARCH_PWR9)
  __vector unsigned int inp0 = (__vector unsigned int)(v.reg.val[0]);
  __vector unsigned int inp1 = (__vector unsigned int)(v.reg.val[1]);
  __vector unsigned int lsb0 = vec_sr(inp0, sh16);
  __vector unsigned int lsb1 = vec_sr(inp1, sh16);
  lsb0 = vec_and(lsb0, one);
  lsb1 = vec_and(lsb1, one);
  __vector unsigned int rnd0 = vec_add(lsb0, bias);
  __vector unsigned int rnd1 = vec_add(lsb1, bias);
  inp0 = vec_add(inp0, rnd0);
  inp1 = vec_add(inp1, rnd1);
  __vector __bool int sel0 =
      vec_test_data_class(v.reg.val[0], __VEC_CLASS_FP_NAN);
  __vector __bool int sel1 =
      vec_test_data_class(v.reg.val[1], __VEC_CLASS_FP_NAN);
  inp0 = vec_sel(inp0, nan, sel0);
  inp1 = vec_sel(inp1, nan, sel1);
  inp0 = vec_sr(inp0, sh16);
  inp1 = vec_sr(inp1, sh16);
  reg = (__vector signed short)vec_perm(inp0, inp1, omask);
#endif
}

inline BF16Vec16::BF16Vec16(const FP32Vec16& v) {
#ifdef _ARCH_PWR10
  __vector signed short ret[4];
  ret[0] = (__vector signed short)__builtin_vsx_xvcvspbf16(
      (__vector unsigned char)v.reg.val[0]);
  ret[1] = (__vector signed short)__builtin_vsx_xvcvspbf16(
      (__vector unsigned char)v.reg.val[1]);
  ret[2] = (__vector signed short)__builtin_vsx_xvcvspbf16(
      (__vector unsigned char)v.reg.val[2]);
  ret[3] = (__vector signed short)__builtin_vsx_xvcvspbf16(
      (__vector unsigned char)v.reg.val[3]);
  reg.val[0] = vec_perm(ret[0], ret[1], omask);
  reg.val[1] = vec_perm(ret[2], ret[3], omask);
#elif defined(_ARCH_PWR9)
  __vector unsigned int inp0 = (__vector unsigned int)(v.reg.val[0]);
  __vector unsigned int inp1 = (__vector unsigned int)(v.reg.val[1]);
  __vector unsigned int inp2 = (__vector unsigned int)(v.reg.val[2]);
  __vector unsigned int inp3 = (__vector unsigned int)(v.reg.val[3]);
  __vector unsigned int lsb0 = vec_sr(inp0, sh16);
  __vector unsigned int lsb1 = vec_sr(inp1, sh16);
  __vector unsigned int lsb2 = vec_sr(inp2, sh16);
  __vector unsigned int lsb3 = vec_sr(inp3, sh16);
  lsb0 = vec_and(lsb0, one);
  lsb1 = vec_and(lsb1, one);
  lsb2 = vec_and(lsb2, one);
  lsb3 = vec_and(lsb3, one);
  __vector unsigned int rnd0 = vec_add(lsb0, bias);
  __vector unsigned int rnd1 = vec_add(lsb1, bias);
  __vector unsigned int rnd2 = vec_add(lsb2, bias);
  __vector unsigned int rnd3 = vec_add(lsb3, bias);
  inp0 = vec_add(inp0, rnd0);
  inp1 = vec_add(inp1, rnd1);
  inp2 = vec_add(inp2, rnd2);
  inp3 = vec_add(inp3, rnd3);
  __vector __bool int sel0 =
      vec_test_data_class(v.reg.val[0], __VEC_CLASS_FP_NAN);
  __vector __bool int sel1 =
      vec_test_data_class(v.reg.val[1], __VEC_CLASS_FP_NAN);
  __vector __bool int sel2 =
      vec_test_data_class(v.reg.val[2], __VEC_CLASS_FP_NAN);
  __vector __bool int sel3 =
      vec_test_data_class(v.reg.val[3], __VEC_CLASS_FP_NAN);
  inp0 = vec_sel(inp0, nan, sel0);
  inp1 = vec_sel(inp1, nan, sel1);
  inp2 = vec_sel(inp2, nan, sel2);
  inp3 = vec_sel(inp3, nan, sel3);
  inp0 = vec_sr(inp0, sh16);
  inp1 = vec_sr(inp1, sh16);
  inp2 = vec_sr(inp2, sh16);
  inp3 = vec_sr(inp3, sh16);
  reg.val[0] = (__vector signed short)vec_perm(inp0, inp1, omask);
  reg.val[1] = (__vector signed short)vec_perm(inp2, inp3, omask);
#endif
}

inline void prefetch(const void* addr) {
  __asm__ __volatile__("dcbt 0, %0" : : "r"(addr) : "memory");
}

};  // namespace vec_op

#endif
