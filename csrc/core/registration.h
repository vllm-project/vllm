#pragma once

#include <Python.h>

#include <tuple>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// REGISTER_EXTENSION allows the shared library to be loaded and initialized
// via python's import statement.
#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }


//
// Get the Nth element from a parameter pack.
//
template <size_t N, typename T, typename... Rest>
inline constexpr auto const& get_nth(T const& head,
                                     Rest const&... rest) noexcept {
  if constexpr (N == 0) {
    return head;
  } else {
    static_assert(N - 1 < sizeof...(Rest),
                  "parameter pack get: index out of range");
    return get_nth<N - 1>(rest...);
  }
}

template <size_t N, typename T, typename... Rest>
inline constexpr auto get_nth(T&& head, Rest&&... rest) noexcept {
  if constexpr (N == 0) {
    return std::forward<T>(head);
  } else {
    static_assert(N - 1 < sizeof...(Rest),
                  "parameter pack get: index out of range");
    return get_nth<N - 1>(std::forward<Rest>(rest)...);
  }
}

///////////////////////////////////////////////////////////////////

template <size_t ArgIdx, typename T>
struct meta_fn_helper;

template <size_t ArgIdx1, size_t ArgIdx2, typename T>
struct meta_fn_helper2;

template <size_t ArgIdx1, size_t ArgIdx2, size_t ArgIdx3, typename T>
struct meta_fn_helper3;

template <size_t ArgIdx, typename R, typename... Args>
struct meta_fn_helper<ArgIdx, R(Args...)> {
  static auto fn(Args... args) {
    // TODO: assert R == nth_type<ArgIdx, Args>()?
    return get_nth<ArgIdx>(std::forward<Args>(args)...);
  }
};

template <size_t ArgIdx, typename R, typename... Args>
struct meta_fn_helper<ArgIdx, R (*)(Args...)> {
  static auto fn(Args... args) {
    // TODO: assert R == nth_type<ArgIdx, Args>()?
    return get_nth<ArgIdx>(std::forward<Args>(args)...);
  }
};

template <size_t ArgIdx1, size_t ArgIdx2, typename R, typename... Args>
struct meta_fn_helper2<ArgIdx1, ArgIdx2, R(Args...)> {
  static auto fn(Args... args) {
    return std::make_tuple(get_nth<ArgIdx1>(std::forward<Args>(args)...),
                           get_nth<ArgIdx2>(std::forward<Args>(args)...));
  }
};

template <size_t ArgIdx1, size_t ArgIdx2, typename R, typename... Args>
struct meta_fn_helper2<ArgIdx1, ArgIdx2, R (*)(Args...)> {
  static auto fn(Args... args) {
    return std::make_tuple(get_nth<ArgIdx1>(std::forward<Args>(args)...),
                           get_nth<ArgIdx2>(std::forward<Args>(args)...));
  }
};

template <size_t ArgIdx1, size_t ArgIdx2, size_t ArgIdx3, typename R,
          typename... Args>
struct meta_fn_helper3<ArgIdx1, ArgIdx2, ArgIdx3, R(Args...)> {
  static auto fn(Args... args) {
    return std::make_tuple(get_nth<ArgIdx1>(std::forward<Args>(args)...),
                           get_nth<ArgIdx2>(std::forward<Args>(args)...),
                           get_nth<ArgIdx3>(std::forward<Args>(args)...));
  }
};

template <size_t ArgIdx1, size_t ArgIdx2, size_t ArgIdx3, typename R,
          typename... Args>
struct meta_fn_helper3<ArgIdx1, ArgIdx2, ArgIdx3, R (*)(Args...)> {
  static auto fn(Args... args) {
    return std::make_tuple(get_nth<ArgIdx1>(std::forward<Args>(args)...),
                           get_nth<ArgIdx2>(std::forward<Args>(args)...),
                           get_nth<ArgIdx3>(std::forward<Args>(args)...));
  }
};

template <size_t ArgIdx, typename FnType>
inline constexpr auto meta_fn = &meta_fn_helper<ArgIdx, FnType>::fn;

template <size_t ArgIdx1, size_t ArgIdx2, typename FnType>
inline constexpr auto meta_fn2 = &meta_fn_helper2<ArgIdx1, ArgIdx2, FnType>::fn;

template <size_t ArgIdx1, size_t ArgIdx2, size_t ArgIdx3, typename FnType>
inline constexpr auto meta_fn3 =
    &meta_fn_helper3<ArgIdx1, ArgIdx2, ArgIdx3, FnType>::fn;

/////////////////////////////

template <typename T, T fn, size_t... ArgIdx>
struct inplace_fn_helper;

/*
template <typename R, typename... Args, R fn(Args...), size_t... ArgIdx>
struct inplace_fn_helper<R(Args...), fn, ArgIdx...> {
  static auto fn(Args... args) {
    static_assert(std::is_void_v<R>);
    fn(std::forward<Args>(args)...);
    if constexpr (sizeof...(ArgIdx) == 1) {
      // TODO: assert R == nth_type<ArgIdx, Args>()?
      return (get_nth<ArgIdx>(std::forward<Args>(args)...), ...);
    } else {
      return std::make_tuple(get_nth<ArgIdx>(std::forward<Args>(args)...)...);
    }
  }
};
*/

template <typename R, typename... Args, R (*fptr)(Args...), size_t... ArgIdx>
struct inplace_fn_helper<R (*)(Args...), fptr, ArgIdx...> {
  static auto fn(Args... args) {
    static_assert(std::is_void_v<R>);
    fptr(std::forward<Args>(args)...);
    if constexpr (sizeof...(ArgIdx) == 1) {
      // TODO: assert R == nth_type<ArgIdx, Args>()?
      return (..., get_nth<ArgIdx>(std::forward<Args>(args)...));
    } else {
      return std::make_tuple(get_nth<ArgIdx>(std::forward<Args>(args)...)...);
    }
  }
};

template <size_t ArgIdx, typename FnType, FnType fn>
inline constexpr auto inplace_fn = &inplace_fn_helper<FnType, fn, ArgIdx>::fn;

template <size_t ArgIdx1, size_t ArgIdx2, typename FnType, FnType fn>
inline constexpr auto inplace_fn2 =
    &inplace_fn_helper<FnType, fn, ArgIdx1, ArgIdx2>::fn;

template <size_t ArgIdx1, size_t ArgIdx2, size_t ArgIdx3, typename FnType,
          FnType fn>
inline constexpr auto inplace_fn3 =
    &inplace_fn_helper<FnType, fn, ArgIdx1, ArgIdx2, ArgIdx3>::fn;

#define inplace_func(idx, fn) inplace_fn<idx, decltype(fn), fn>
#define inplace_func2(idx1, idx2, fn) inplace_fn2<idx1, idx2, decltype(fn), fn>
#define inplace_func3(idx1, idx2, idx3, fn) \
  inplace_fn3<idx1, idx2, idx3, decltype(fn), fn>
