#ifndef CPU_TYPES_BASE_HPP
#define CPU_TYPES_BASE_HPP

#include <type_traits>
#include <torch/torch.h>

namespace vec_op {

/****************************************/
/*               FP16Vec8               */
/****************************************/
struct FP16Vec8Base {
  virtual void save(void* ptr) const = 0;

  virtual ~FP16Vec8Base() = default;
};

/****************************************/
/*               FP16Vec16              */
/****************************************/
struct FP16Vec16Base {
  virtual void save(void* ptr) const = 0;

  virtual void save(void* ptr, const int elem_num) const = 0;

  virtual ~FP16Vec16Base() = default;
};

/****************************************/
/*               BF16Vec8               */
/****************************************/
struct BF16Vec8Base {
  virtual void save(void* ptr) const = 0;

  virtual ~BF16Vec8Base() = default;
};

/****************************************/
/*               BF16Vec16              */
/****************************************/
struct BF16Vec16Base {
  virtual void save(void* ptr) const = 0;

  virtual void save(void* ptr, const int elem_num) const = 0;

  virtual ~BF16Vec16Base() = default;
};

/****************************************/
/*               BF16Vec32              */
/****************************************/
struct BF16Vec32Base {
  virtual void save(void* ptr) const = 0;

  virtual ~BF16Vec32Base() = default;
};

/****************************************/
/*               FP32Vec4               */
/****************************************/
struct FP32Vec4Base {
  virtual void save(float* ptr) const = 0;

  virtual ~FP32Vec4Base() = default;
};

/****************************************/
/*               FP32Vec8               */
/****************************************/
template <typename VecImpl>
struct FP32Vec8Base {
  virtual float reduce_sum() const = 0;

  virtual VecImpl exp() const = 0;

  virtual VecImpl tanh() const = 0;

  virtual VecImpl er() const = 0;

  virtual VecImpl operator*(const VecImpl& b) const = 0;

  virtual VecImpl operator+(const VecImpl& b) const = 0;

  virtual VecImpl operator-(const VecImpl& b) const = 0;

  virtual VecImpl operator/(const VecImpl& b) const = 0;

  virtual void save(float* ptr) const = 0;

  virtual ~FP32Vec8Base() = default;
};

/****************************************/
/*               FP32Vec16              */
/****************************************/
template <typename VecImpl>
struct FP32Vec16Base {
  virtual VecImpl operator*(const VecImpl& b) const = 0;

  virtual VecImpl operator+(const VecImpl& b) const = 0;

  virtual VecImpl operator-(const VecImpl& b) const = 0;

  virtual VecImpl operator/(const VecImpl& b) const = 0;

  virtual VecImpl clamp(const VecImpl& min, const VecImpl& max) const = 0;

  virtual VecImpl max(const VecImpl& b) const = 0;

  virtual VecImpl max(const VecImpl& b, const int elem_num) const = 0;

  virtual VecImpl min(const VecImpl& b) const = 0;

  virtual VecImpl min(const VecImpl& b, const int elem_num) const = 0;

  virtual VecImpl abs() const = 0;

  virtual float reduce_max() const = 0;

  virtual float reduce_min() const = 0;

  virtual float reduce_sum() const = 0;

  virtual float reduce_sub_sum(int idx, int group_size) const = 0;

  virtual void save(float* ptr) const = 0;

  virtual void save(float* ptr, const int elem_num) const = 0;

  virtual ~FP32Vec16Base() = default;
};

/****************************************/
/*               INT32Vec16             */
/****************************************/
struct INT32Vec16Base {
  virtual void save(int32_t* ptr) const = 0;

  virtual void save(int32_t* ptr, const int elem_num) const = 0;

  virtual ~INT32Vec16Base() = default;
};

/****************************************/
/*               INT8Vec16              */
/****************************************/
struct INT8Vec16Base {
  virtual void save(int8_t* ptr) const = 0;

  virtual void save(int8_t* ptr, const int elem_num) const = 0;

  virtual ~INT8Vec16Base() = default;
};

/****************************************/
/*                 Helpers              */
/****************************************/
#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)                                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                         \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)                      \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#ifndef CPU_OP_GUARD
#define CPU_KERNEL_GUARD_IN(NAME)
#define CPU_KERNEL_GUARD_OUT(NAME)
#else
#define CPU_KERNEL_GUARD_IN(NAME)                                              \
  RECORD_FUNCTION(#NAME, c10::ArrayRef<c10::IValue>({}));
#define CPU_KERNEL_GUARD_OUT(NAME)
#endif

#define FORCE_INLINE __attribute__((always_inline)) inline

namespace {
template <typename T, T... indexes, typename F>
constexpr void unroll_loop_item(std::integer_sequence<T, indexes...>, F&& f) {
  (f(std::integral_constant<T, indexes>{}), ...);
}
}; // namespace

template <typename T, T count, typename F,
          typename = std::enable_if_t<std::is_invocable_v<F, T>>>
constexpr void unroll_loop(F&& f) {
  unroll_loop_item(std::make_integer_sequence<T, count>{}, std::forward<F>(f));
}

}; // namespace vec_op

#endif
