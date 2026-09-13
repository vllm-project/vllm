// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cmath>
#include <type_traits>

#include <torch/all.h>

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

#if defined(__APPLE__)
  #include "omp.h"
#endif

using namespace at::vec;

namespace vec_op {

struct fp8_e4m3_tag {};
struct fp8_e5m2_tag {};

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
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
// Number of elements in single ASIMD/SVE vector of given Datatype
#define NUM_ELEMENTS_REG(vec) (sizeof(vec) / sizeof(vec[0]))

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F&& f) {
  (f(std::integral_constant<T, indexes>{}), ...);
};
};  // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
inline constexpr void unroll_loop(F&& f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

template <typename T, typename... Ts>
struct is_one_of : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

template <typename T, typename... Ts>
inline constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;

struct uninit_t {
  explicit constexpr uninit_t() = default;
};
inline constexpr uninit_t uninit{};

template <typename NxVectorizedTVecReg, typename T, int VEC_ELEM_NUM>
union AliasReg {
  NxVectorizedTVecReg reg;
  T values[VEC_ELEM_NUM];
};

// Template over at::vec::Vectorized<T> to support
// multiple vectorised registers into 1 of length VEC_REG_NUM val
template <int N, typename T>
struct NxVectorizedTVecReg {
  using value_t = T;
  using VectorizedT = Vectorized<T>;

  VectorizedT val[N];

  NxVectorizedTVecReg() = default;
  NxVectorizedTVecReg(const NxVectorizedTVecReg&) = default;
  NxVectorizedTVecReg(NxVectorizedTVecReg&&) = default;
  NxVectorizedTVecReg& operator=(const NxVectorizedTVecReg&) = default;
  NxVectorizedTVecReg& operator=(NxVectorizedTVecReg&&) = default;

  explicit NxVectorizedTVecReg(uninit_t) noexcept {};

  FORCE_INLINE explicit NxVectorizedTVecReg(const VectorizedT& vec_t) {
    unroll_loop<int, N>([&](int i) { val[i] = vec_t; });
  };

  FORCE_INLINE explicit NxVectorizedTVecReg(T v) noexcept {
    VectorizedT vv(v);
    unroll_loop<int, N>([&](int i) { val[i] = vv; });
  }

  FORCE_INLINE explicit NxVectorizedTVecReg(const void* ptr) { load(ptr); }
  explicit NxVectorizedTVecReg(const void* ptr, const int elem_num) {
    load(ptr, elem_num);
  }

  static constexpr int size() noexcept { return N * VectorizedT::size(); }

  FORCE_INLINE void save(void* ptr) const {
    value_t* base = reinterpret_cast<value_t*>(ptr);
    unroll_loop<int, N>(
        [&](int i) { val[i].store(base + i * VectorizedT::size()); });
  }
  FORCE_INLINE void load(const void* ptr) {
    const value_t* base = reinterpret_cast<const value_t*>(ptr);
    unroll_loop<int, N>([&](int i) {
      val[i] = VectorizedT::loadu(base + i * VectorizedT::size());
    });
  }

  FORCE_INLINE void save(void* ptr, const int elem_num) const {
    value_t* base = reinterpret_cast<value_t*>(ptr);
    save_partial(base, elem_num);
  }

  FORCE_INLINE void load(const void* ptr, const int elem_num) {
    const value_t* base = reinterpret_cast<const value_t*>(ptr);
    load_partial(base, elem_num);
  }

  FORCE_INLINE void save_partial(value_t* base, int elem_num) const {
    const int w = VectorizedT::size();
    int full = elem_num / w;
    int rem = elem_num % w;
    for (int i = 0; i < full; i++) val[i].store(base + i * w);
    if (rem) val[full].store(base + full * w, rem);
  }

  FORCE_INLINE void load_partial(const value_t* base, int elem_num) {
    const int w = VectorizedT::size();
    int full = elem_num / w;
    int rem = elem_num % w;
    for (int i = 0; i < full; i++) val[i] = VectorizedT::loadu(base + i * w);
    if (rem) val[full] = VectorizedT::loadu(base + full * w, rem);
  }

  template <VectorizedT (VectorizedT::*torch_vec_func)() const,
            value_t (*std_func)(value_t)>
  FORCE_INLINE NxVectorizedTVecReg opt_vec_func_impl() const {
    NxVectorizedTVecReg result;

    if constexpr (torch_vec_func != nullptr) {
      unroll_loop<int, N>(
          [&](int i) { result.val[i] = (val[i].*torch_vec_func)(); });
    } else {
      for (int i = 0; i < N; i++) {
        alignas(64) value_t buf[VectorizedT::size()];
        val[i].store(buf);
        for (int j = 0; j < VectorizedT::size(); ++j) {
          buf[j] = std_func(buf[j]);
        }
        result.val[i] = VectorizedT::loadu(buf);
      }
    }
    return result;
  }
};

template <typename DerivedClassT, int N, typename T>
struct VectorizedRegWrapper {
  using ScalarT = T;
  using VectorizedT = Vectorized<T>;
  using NxVectorizedTArray = NxVectorizedTVecReg<N, T>;

  constexpr static int VEC_REG_NUM = N;
  constexpr static int VEC_ELEM_NUM = VEC_REG_NUM * VectorizedT::size();
  constexpr static int get_elem_num() { return VEC_ELEM_NUM; };

  NxVectorizedTArray reg;

  VectorizedRegWrapper() noexcept = default;
  explicit VectorizedRegWrapper(uninit_t) noexcept : reg{uninit} {};
  FORCE_INLINE explicit VectorizedRegWrapper(T v) : reg(v){};
  explicit VectorizedRegWrapper(const void* ptr) : reg(ptr) {};
  explicit VectorizedRegWrapper(const void* ptr, const int elem_num)
      : reg(ptr, elem_num) {};
  explicit VectorizedRegWrapper(const VectorizedT& r) : reg(r) {};
  explicit VectorizedRegWrapper(const NxVectorizedTArray& r) : reg(r) {};

  VectorizedRegWrapper(const VectorizedRegWrapper&) = default;
  VectorizedRegWrapper(VectorizedRegWrapper&&) = default;
  VectorizedRegWrapper& operator=(VectorizedRegWrapper&&) = default;
  VectorizedRegWrapper& operator=(const VectorizedRegWrapper&) = default;

  FORCE_INLINE void save(void* ptr) const { reg.save(ptr); }
  void save(void* ptr, const int elem_num) const { reg.save(ptr, elem_num); }

// Define optimized functions using at::vec::Vectorized<T> where possible
// Fallback to std:: functions when not available
#define OPT_TORCH_IMPL(FUNC_NAME, STD_FUNC_NAME, TORCH_FUNC_NAME, ...)         \
  FORCE_INLINE DerivedClassT FUNC_NAME() const {                               \
    if constexpr (is_one_of_v<T, __VA_ARGS__>) {                               \
      return DerivedClassT{                                                    \
          reg.template opt_vec_func_impl<&VectorizedT::TORCH_FUNC_NAME,        \
                                         std::STD_FUNC_NAME>()};               \
    } else {                                                                   \
      return DerivedClassT{reg.template opt_vec_func_impl<                     \
          nullptr, static_cast<ScalarT (*)(ScalarT)>(&std::STD_FUNC_NAME)>()}; \
    }                                                                          \
  }

  // Define optimized functions for datatypes passed in __VA_ARGS__
  OPT_TORCH_IMPL(abs, abs, abs, c10::Half, float)
  OPT_TORCH_IMPL(er, erf, erf, float)
  OPT_TORCH_IMPL(exp, exp, fexp_u20, float)
  OPT_TORCH_IMPL(exp_u20, exp, exp_u20, float)
  OPT_TORCH_IMPL(sin, sin, sin, float)
  OPT_TORCH_IMPL(sinh, sinh, sinh, float)
  OPT_TORCH_IMPL(cos, cos, cos, float)
  OPT_TORCH_IMPL(cosh, cosh, cosh, float)
  OPT_TORCH_IMPL(log, log, log, float)
  OPT_TORCH_IMPL(log10, log10, log10, float)
  OPT_TORCH_IMPL(sqrt, sqrt, sqrt, c10::Half, float)
  OPT_TORCH_IMPL(tan, tan, tan, float)
  OPT_TORCH_IMPL(tanh, tanh, tanh, float)

#undef OPT_TORCH_IMPL
};

// Wrapper around the vectorized types that don't use the full vector-width.
// eg BF16Vec8, FP16Vec8 and FP32Vec4 on sve-256
template <typename DerivedClassT, typename T, int LogicalSize>
struct PartialVectorizedRegWrapper {
  using ScalarT = T;
  using VectorizedT = Vectorized<T>;
  using Reg = NxVectorizedTVecReg<1, T>;

  static constexpr int VEC_REG_NUM = 1;
  static constexpr int VEC_ELEM_NUM = LogicalSize;
  static_assert(VEC_ELEM_NUM <= VectorizedT::size());

  Reg reg;

  PartialVectorizedRegWrapper() noexcept = default;
  explicit PartialVectorizedRegWrapper(uninit_t) noexcept : reg{uninit} {}
  explicit PartialVectorizedRegWrapper(T value) : reg(value) {}
  explicit PartialVectorizedRegWrapper(const void* ptr)
      : reg(VectorizedT::loadu(ptr, VEC_ELEM_NUM)) {}
  explicit PartialVectorizedRegWrapper(const VectorizedT& value) : reg(value) {}
  explicit PartialVectorizedRegWrapper(const Reg& value) : reg(value) {}

  static constexpr int get_elem_num() { return VEC_ELEM_NUM; }

  FORCE_INLINE void save(void* ptr) const {
    reg.val[0].store(ptr, VEC_ELEM_NUM);
  }

  FORCE_INLINE void save(void* ptr, int elem_num) const {
    TORCH_CHECK(elem_num >= 0 && elem_num <= VEC_ELEM_NUM);
    reg.val[0].store(ptr, elem_num);
  }

 private:
  template <ScalarT (*std_func)(ScalarT)>
  FORCE_INLINE DerivedClassT std_func_impl() const {
    DerivedClassT result(uninit);
    alignas(64) ScalarT values[VEC_ELEM_NUM];
    reg.val[0].store(values, VEC_ELEM_NUM);
    for (int i = 0; i < VEC_ELEM_NUM; ++i) {
      values[i] = std_func(values[i]);
    }
    result.reg.val[0] = VectorizedT::loadu(values, VEC_ELEM_NUM);
    return result;
  }

 public:
#define OPT_PARTIAL_TORCH_IMPL(FUNC_NAME, STD_FUNC_NAME, TORCH_FUNC_NAME, ...) \
  FORCE_INLINE DerivedClassT FUNC_NAME() const {                               \
    if constexpr (is_one_of_v<T, __VA_ARGS__>) {                               \
      DerivedClassT result(uninit);                                            \
      result.reg.val[0] = reg.val[0].TORCH_FUNC_NAME();                        \
      return result;                                                           \
    } else {                                                                   \
      return std_func_impl<static_cast<ScalarT (*)(ScalarT)>(                  \
          &std::STD_FUNC_NAME)>();                                             \
    }                                                                          \
  }

  // Define optimized functions for datatypes passed in __VA_ARGS__
  OPT_PARTIAL_TORCH_IMPL(abs, abs, abs, c10::Half, float)
  OPT_PARTIAL_TORCH_IMPL(er, erf, erf, float)
  OPT_PARTIAL_TORCH_IMPL(exp, exp, fexp_u20, float)
  OPT_PARTIAL_TORCH_IMPL(exp_u20, exp, exp_u20, float)
  OPT_PARTIAL_TORCH_IMPL(sin, sin, sin, float)
  OPT_PARTIAL_TORCH_IMPL(sinh, sinh, sinh, float)
  OPT_PARTIAL_TORCH_IMPL(cos, cos, cos, float)
  OPT_PARTIAL_TORCH_IMPL(cosh, cosh, cosh, float)
  OPT_PARTIAL_TORCH_IMPL(log, log, log, float)
  OPT_PARTIAL_TORCH_IMPL(log10, log10, log10, float)
  OPT_PARTIAL_TORCH_IMPL(sqrt, sqrt, sqrt, c10::Half, float)
  OPT_PARTIAL_TORCH_IMPL(tan, tan, tan, float)
  OPT_PARTIAL_TORCH_IMPL(tanh, tanh, tanh, float)

#undef OPT_PARTIAL_TORCH_IMPL
};

// forward declare vectorised dtypes
struct FP32Vec8;
struct FP32Vec16;
struct FP16Vec8;
struct FP16Vec16;
struct BF16Vec8;
struct BF16Vec16;
struct INT8Vec16;
struct INT32Vec16;

// Only used for int types for now could be replaced when
// int8/32 vectorised ops are added in ATen
template <typename T>
struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; };
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

template <>
inline void storeFP32<c10::Half>(float v, c10::Half* ptr) {
  *reinterpret_cast<__fp16*>(ptr) = v;
}

inline void prefetch(const void* addr) { __builtin_prefetch(addr, 0, 1); };

};  // namespace vec_op
