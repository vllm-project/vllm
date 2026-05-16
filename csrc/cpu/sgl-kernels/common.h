// Adapted from
// https://github.com/sgl-project/sglang/tree/main/sgl-kernel/csrc/cpu

// clang-format off

#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

// dispatch bool
#define AT_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...) \
  [&] {                                          \
    if (BOOL_V) {                                \
      constexpr bool BOOL_NAME = true;           \
      return __VA_ARGS__();                      \
    } else {                                     \
      constexpr bool BOOL_NAME = false;          \
      return __VA_ARGS__();                      \
    }                                            \
  }()

#define AT_DISPATCH_BOOL2(BOOL_V1, BOOL_NAME1, BOOL_V2, BOOL_NAME2, ...) \
  [&] {                                                                  \
    if (BOOL_V1) {                                                       \
      constexpr bool BOOL_NAME1 = true;                                  \
      if (BOOL_V2) {                                                     \
        constexpr bool BOOL_NAME2 = true;                                \
        return __VA_ARGS__();                                            \
      } else {                                                           \
        constexpr bool BOOL_NAME2 = false;                               \
        return __VA_ARGS__();                                            \
      }                                                                  \
    } else {                                                             \
      constexpr bool BOOL_NAME1 = false;                                 \
      if (BOOL_V2) {                                                     \
        constexpr bool BOOL_NAME2 = true;                                \
        return __VA_ARGS__();                                            \
      } else {                                                           \
        constexpr bool BOOL_NAME2 = false;                               \
        return __VA_ARGS__();                                            \
      }                                                                  \
    }                                                                    \
  }()

// dispatch: bfloat16, float16, int8_t, fp8_e4m3, uint8_t(mxfp4/int4)
#define CPU_DISPATCH_PACKED_TYPES(TYPE, ...)                     \
  [&] {                                                          \
    switch (TYPE) {                                              \
      case at::ScalarType::BFloat16: {                           \
        using packed_t = at::BFloat16;                           \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Half: {                               \
        using packed_t = at::Half;                               \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Char: {                               \
        using packed_t = int8_t;                                 \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Float8_e4m3fn: {                      \
        using packed_t = at::Float8_e4m3fn;                      \
        return __VA_ARGS__();                                    \
      }                                                          \
      case at::ScalarType::Byte: {                               \
        using packed_t = uint8_t;                                \
        return __VA_ARGS__();                                    \
      }                                                          \
      default:                                                   \
        TORCH_CHECK(false, "Unsupported floating data type.\n"); \
    }                                                            \
  }()

// Helper MICRO for CPU_DISPATCH_FLOATING_TYPES_EXT:
//   TYPE1: the primary dtype (input, output, weight);
//   TYPE2: defined as PARAM_T input
#define CPU_DISPATCH_TYPE1_WITH_PARAM(TYPE1, PARAM_T, ...)   \
  switch (TYPE1) {                                           \
    case at::ScalarType::BFloat16: {                         \
      using scalar_t = at::BFloat16;                         \
      using param_t = PARAM_T;                               \
      return __VA_ARGS__();                                  \
    }                                                        \
    case at::ScalarType::Half: {                             \
      using scalar_t = at::Half;                             \
      using param_t = PARAM_T;                               \
      return __VA_ARGS__();                                  \
    }                                                        \
    case at::ScalarType::Float: {                            \
      using scalar_t = float;                                \
      using param_t = PARAM_T;                               \
      return __VA_ARGS__();                                  \
    }                                                        \
    default:                                                 \
      TORCH_CHECK(false, "Unsupported floating data type."); \
  }

// Helper MICRO for CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT:
//   TYPE1: the primary dtype (input, output, weight);
//   TYPE2: defined as PARAM_T input
#define CPU_DISPATCH_TYPE1_WITH_PARAM_REDUCED(TYPE1, PARAM_T, ...) \
  switch (TYPE1) {                                                 \
    case at::ScalarType::BFloat16: {                               \
      using scalar_t = at::BFloat16;                               \
      using param_t = PARAM_T;                                     \
      return __VA_ARGS__();                                        \
    }                                                              \
    case at::ScalarType::Half: {                                   \
      using scalar_t = at::Half;                                   \
      using param_t = PARAM_T;                                     \
      return __VA_ARGS__();                                        \
    }                                                              \
    default:                                                       \
      TORCH_CHECK(false, "Unsupported floating data type.");       \
  }

// Helper MICRO for CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT:
//   TYPE1: the dtype both for scalar_t and param_t
#define CPU_DISPATCH_TYPE1_WITH_SAME_PARAM_REDUCED(TYPE1, ...)       \
  switch (TYPE1) {                                                   \
    case at::ScalarType::BFloat16: {                                 \
      using scalar_t = at::BFloat16;                                 \
      using param_t = at::BFloat16;                                  \
      return __VA_ARGS__();                                          \
    }                                                                \
    case at::ScalarType::Half: {                                     \
      using scalar_t = at::Half;                                     \
      using param_t = at::Half;                                      \
      return __VA_ARGS__();                                          \
    }                                                                \
    default:                                                         \
      TORCH_CHECK(false, "Unsupported reduced floating data type."); \
  }

// dispatch with mixed dtypes (TYPE1, TYPE2):
//   TYPE1: the primary dtype (input, output, weight);
//   TYPE2: the secondary dtype (bias, etc.).
#define CPU_DISPATCH_FLOATING_TYPES_EXT(TYPE1, TYPE2, ...)            \
  [&] {                                                               \
    if (TYPE2 == at::kFloat) {                                        \
      CPU_DISPATCH_TYPE1_WITH_PARAM(TYPE1, float, __VA_ARGS__)        \
    } else if (TYPE2 == at::ScalarType::BFloat16) {                   \
      CPU_DISPATCH_TYPE1_WITH_PARAM(TYPE1, at::BFloat16, __VA_ARGS__) \
    } else if (TYPE2 == at::ScalarType::Half) {                       \
      CPU_DISPATCH_TYPE1_WITH_PARAM(TYPE1, at::Half, __VA_ARGS__)     \
    } else {                                                          \
      TORCH_CHECK(false, "Unsupported floating data type.");          \
    }                                                                 \
  }()

// dispatch with mixed dtypes (reduced one, no float for TYPE1) (TYPE1, TYPE2):
//   TYPE1: the primary dtype (input, output, weight);
//   TYPE2: the secondary dtype (bias, etc.).
#define CPU_DISPATCH_REDUCED_FLOATING_TYPES_EXT(TYPE1, TYPE2, ...)     \
  [&] {                                                                \
    if (TYPE2 == at::kFloat) {                                         \
      CPU_DISPATCH_TYPE1_WITH_PARAM_REDUCED(TYPE1, float, __VA_ARGS__) \
    } else {                                                           \
      TORCH_CHECK(TYPE1 == TYPE2);                                     \
      CPU_DISPATCH_TYPE1_WITH_SAME_PARAM_REDUCED(TYPE1, __VA_ARGS__)   \
    }                                                                  \
  }()

#define UNUSED(x) (void)(x)

#define CHECK_CPU(x) TORCH_CHECK(x.device().type() == at::kCPU, #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, #x "must be contiguous at last dimension")

#define CHECK_INPUT(x) \
  CHECK_CPU(x);        \
  CHECK_CONTIGUOUS(x)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CPU(x);                            \
  CHECK_LAST_DIM_CONTIGUOUS(x)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)
#define CHECK_GT(a, b) TORCH_CHECK((a) > (b), "CHECK_GT(" #a ", " #b ") failed. ", a, " vs ", b)
#define CHECK_GE(a, b) TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

template <bool is_only_lastdim_contiguous>
static inline void CHECK_INPUT_SHAPE_DTYPE(const at::Tensor& tensor, const at::IntArrayRef sizes, at::ScalarType st) {
  TORCH_CHECK(tensor.sizes() == sizes, "Input tensor shape mismatch: expected ", sizes, ", got ", tensor.sizes());
  TORCH_CHECK(tensor.scalar_type() == st, "Input tensor dtype mismatch");
  if constexpr (is_only_lastdim_contiguous) {
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(tensor);
  } else {
    CHECK_INPUT(tensor);
  }
}

// [NB] Parallel Routines
//
//  * at::parallel_for - applies for most of generic use cases, this will be compiled
//                       against openmp in default torch release.
//
//  * parallel_for     - same function as above, can choose payload partition scheme in
//                       balance211.
//
//  * parallel_2d      - parallel for 2 dimensions, used in GEMM, etc.
//                       this one will do payload balance across 2 dimensions.
//

// grain size for each thread
constexpr int GRAIN_SIZE = 1024;

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) {
  return (x + y - 1) / y;
}

// you can only use at::get_thread_num() with at::parallel_for()
// as it is lazy initialized, otherwise it will always return 0.
inline int get_thread_num() {
#if defined(_OPENMP)
  return omp_get_thread_num();
#else
  return 0;
#endif
}

// balance payload across each thread
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

// limit max cache blocks
// when we need to do pre-unpack for weights, e.g. fp8
#define MAX_CACHE_BLOCK_SIZE 4

template <typename T>
inline int get_cache_blocks(int chunk_size) {
  // L2 2MB and ratio of 50%
  const int L2_size = 2048 * 1024 >> 1;
  return std::max(1, int(L2_size / (chunk_size * sizeof(T))));
}

template <>
inline int get_cache_blocks<at::Float8_e4m3fn>(int chunk_size) {
  // fp8 uses bf16 as accumulate type
  int cache_block_size = get_cache_blocks<at::BFloat16>(chunk_size);
  return std::min(MAX_CACHE_BLOCK_SIZE, cache_block_size);
}

// 2d sequential loop in range : [mb0, mb1), [nb0, nb1)
template <typename T, typename func_t>
inline void loop_2d(int64_t mb0, int64_t mb1, int64_t nb0, int64_t nb1, int64_t chunk_size, const func_t& f) {
  // get number of blocks for L2 in most inner loop
  int64_t cache_blocks_nb = get_cache_blocks<T>(chunk_size);

  // loop order: [NB / cache_blocks_nb, MB, cache_blocks_nb]
  // TODO: implement reverse order of [MB / cache_blocks_mb, NB, cache_blocks_mb]
  for (int64_t nbb = nb0; nbb < nb1; nbb += cache_blocks_nb) {
    for (int64_t mb = mb0; mb < mb1; ++mb) {
      for (int64_t nb = nbb; nb < std::min(nbb + cache_blocks_nb, nb1); ++nb) {
        f(mb, nb, nb - nbb);
      }
    }
  }
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

// conditional data ptr for optional tensor
template <typename T>
inline T* conditional_data_ptr(const std::optional<at::Tensor>& opt) {
  return opt.has_value() ? opt.value().data_ptr<T>() : nullptr;
}

}  // anonymous namespace
