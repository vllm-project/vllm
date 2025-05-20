#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/record_function.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

// dispatch bool
#define AT_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)                                 \
  [&] {                                                                          \
    if (BOOL_V) {                                                                \
      constexpr bool BOOL_NAME = true;                                           \
      return __VA_ARGS__();                                                      \
    } else {                                                                     \
      constexpr bool BOOL_NAME = false;                                          \
      return __VA_ARGS__();                                                      \
    }                                                                            \
  }()

// dispatch: bfloat16, float16, int8_t, fp8_e4m3
#define CPU_DISPATCH_PACKED_TYPES(TYPE, ...)                                    \
  [&] {                                                                         \
    switch (TYPE) {                                                             \
      case at::ScalarType::BFloat16 : {                                         \
        using packed_t = at::BFloat16;                                          \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      case at::ScalarType::Half: {                                              \
        using packed_t = at::Half;                                              \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      case at::ScalarType::Char : {                                             \
        using packed_t = int8_t;                                                \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      case at::ScalarType::Float8_e4m3fn : {                                    \
        using packed_t = at::Float8_e4m3fn;                                     \
        return __VA_ARGS__();                                                   \
      }                                                                         \
      default:                                                                  \
        TORCH_CHECK(false, "Unsupported floating data type.\n");                \
    }                                                                           \
  }()

#define UNUSED(x) (void)(x)

#define CHECK_CPU(x) TORCH_CHECK(x.device().type() == at::kCPU, #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, #x "must be contiguous at last dimention")

#define CHECK_INPUT(x) \
  CHECK_CPU(x);        \
  CHECK_CONTIGUOUS(x)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CPU(x);                            \
  CHECK_LAST_DIM_CONTIGUOUS(x)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

// parallel routines
constexpr int GRAIN_SIZE = 1024;

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) { return (x + y - 1) / y; }

template <typename T>
inline void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#if 0
    // onednn partition pattern
    T& n_my = n_end;
    if (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#else
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif
}

template <typename func_t>
inline void parallel_for(int n, const func_t& f) {
#if defined(_OPENMP)
#pragma omp parallel
{
    int nth = omp_get_num_threads();
    int ith = omp_get_thread_num();
    int tbegin, tend;
    balance211(n, nth, ith, tbegin, tend);
    f(tbegin, tend);
}
#else
    f(0, n);
#endif
}

// data indexing for dimension collapse
template <typename T>
inline T data_index_init(T offset) {
  return offset;
}

template <typename T, typename... Args>
inline T data_index_init(T offset, T& x, const T& X, Args&&... args) {
  offset = data_index_init(offset, std::forward<Args>(args)...);
  x = offset % X;
  return offset / X;
}

inline bool data_index_step() {
  return true;
}

template <typename T, typename... Args>
inline bool data_index_step(T& x, const T& X, Args&&... args) {
  if (data_index_step(std::forward<Args>(args)...)) {
    x = ((x + 1) == X) ? 0 : (x + 1);
    return x == 0;
  }
  return false;
}

// forced unroll for perf critical path

#if __has_attribute(always_inline)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

template <int n>
struct Unroll {
  template <typename Func, typename... Args>
  ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    Unroll<n - 1>{}(f, args...);
    f(std::integral_constant<int, n - 1>{}, args...);
  }
};

template <>
struct Unroll<1> {
  template <typename Func, typename... Args>
  ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
    f(std::integral_constant<int, 0>{}, args...);
  }
};

} // anonymous namespace
