// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/record_function.h>

// clang-format off

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

// for 1d parallel, use `actual_nth`
// for 2d parallel, use even nths, e.g. 43->42
int inline adjust_num_threads(int m) {
  int actual_nth = at::get_num_threads();
  if (m == 1) {
    return actual_nth;
  }
  return std::max(1, (actual_nth >> 1) * 2);
}

template <typename func_t>
inline void parallel_2d(int m, int n, const func_t& f) {

  // make sure we have even num_threads
  int nth = adjust_num_threads(m);

  // [NOTE] thread blocking:
  //
  //   1) prefer square block per thread
  //   2) use even number of CPU cores
  //   3) use all `num_threads` cores
  //
  //   we have:
  //     TM * TN = T
  //     BM / TM = BN / TN
  //   then:
  //     TM = ((BM / BN) * T) ^ 0.5
  //
  float r = float(m) / n;
  int nth_m = std::ceil(std::sqrt(r * nth));
  int nth_n = 1;
  for (; nth_m > 0; --nth_m) {
    nth_n = nth / nth_m;
    if (nth_m * nth_n == nth) {
      break;
    }
  }

#if defined(_OPENMP)
#pragma omp parallel num_threads(nth)
{
  int ith = omp_get_thread_num();
  int ith_m = ith / nth_n;
  int ith_n = ith % nth_n;

  int thread_block_m = div_up(m, nth_m);
  int thread_block_n = div_up(n, nth_n);

  int begin_m = ith_m * thread_block_m;
  int end_m = std::min(m, begin_m + thread_block_m);
  int begin_n = ith_n * thread_block_n;
  int end_n = std::min(n, begin_n + thread_block_n);

  f(begin_m, end_m, begin_n, end_n);
}
#else
  f(0, m, 0, n);
#endif
}

template <typename T>
int get_cache_blocks(int BLOCK_SIZE, int K) {
  // L2 2MB and ratio of 50%
  const int L2_size = 2048 * 1024 >> 1;
  return std::max(1, int(L2_size / (BLOCK_SIZE * K * sizeof(T))));
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
