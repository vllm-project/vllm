#pragma once

#include <torch/library.h>

/**
 * PyBind and PyTorch Library apis generally require different type signatures.
 * This file provides a shim to (mostly, there may be missing conversions) to
 * convert from function designed to be used with PyBind to one that can be used
 * with PyTorch Library. This is done using `make_pytorch_shim` which creates a
 * lambda that exponses the API using PyTorch compatible types to the types.
 * This is useful when trying to ingergate PyBind based external libraries into
 * vLLM.
 *
 * Example:
 *
 * PYBIND11_MODULE(NAME, m) {
 *   m.def("foo", &foo);
 * }
 *
 * could be replaced with (using the shim):
 * TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
 *   m.def("foo", make_pytorch_shim(&foo));
 *   m.impl("foo", torch::kCUDA, make_pytorch_shim(&foo));
 * }
 *
 * The `pytorch_library_compatible_type` struct is used to map from the
 * flash_attn ops types to a PyTorch library compatible one. The main issues is
 * that the following types are not support by PyTorch library bindings:
 *  - `int`
 *  - `float`
 *  - `c10::optional<T> &`
 *  - `c10::optional<const at::Tensor> &`
 * So we convert them to (respectively):
 *  - `int64_t`
 *  - `double`
 *  - `const c10::optional<T>&`
 *  - `const c10::optional<at::Tensor>&`
 */

template <typename T>
struct pytorch_library_compatible_type {
  using type = T;
  static T convert_from_type(T arg) { return arg; }
};

template <typename T>
using pytorch_library_compatible_type_t =
    typename pytorch_library_compatible_type<T>::type;

template <typename T>
T convert_from_pytorch_compatible_type(
    pytorch_library_compatible_type_t<T> arg) {
  return pytorch_library_compatible_type<T>::convert_from_type(arg);
}

// Map `c10::optional<T> &` -> `const c10::optional<T>&`
//  (NOTE: this is bit unsafe but non of the ops in flash_attn mutate
//   the optional container)
template <typename T>
struct pytorch_library_compatible_type<c10::optional<T>&> {
  using type = const c10::optional<T>&;
  static c10::optional<T>& convert_from_type(const c10::optional<T>& arg) {
    return const_cast<c10::optional<T>&>(arg);
  }
};

// Map `c10::optional<T>` ->
//          `c10::optional<pytorch_library_compatible_type_t<T>>`
//  (NOTE: tested for `c10::optional<int>` -> `c10::optional<int64_t>`)
template <typename T>
struct pytorch_library_compatible_type<c10::optional<T>> {
  using type = c10::optional<pytorch_library_compatible_type_t<T>>;
  static c10::optional<pytorch_library_compatible_type_t<T>> convert_from_type(
      c10::optional<T> arg) {
    return arg;
  }
};

// Map `c10::optional<const at::Tensor>&` -> `const c10::optional<at::Tensor>&`
template <>
struct pytorch_library_compatible_type<c10::optional<const at::Tensor>&> {
  using type = const c10::optional<at::Tensor>&;
  static c10::optional<const at::Tensor>& convert_from_type(
      const c10::optional<at::Tensor>& arg) {
    return const_cast<c10::optional<const at::Tensor>&>(
        reinterpret_cast<const c10::optional<const at::Tensor>&>(arg));
  }
};

// Map `int` -> `int64_t`
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(),
                "int64_t value is too large to be converted to int");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(),
                "int64_t value is too small to be converted to int");
    return arg;
  }
};

// Map `float` -> `double`
template <>
struct pytorch_library_compatible_type<float> {
  using type = double;
  static float convert_from_type(double arg) {
    TORCH_CHECK(std::abs(arg) <= std::numeric_limits<float>::max(),
                "double value is too large to be converted to float");
    return arg;
  }
};

//
//  Shim Utils
//

template <typename Ret, typename... Args>
auto make_pytorch_shim(Ret (*fun)(Args... args)) {
  return [fun](pytorch_library_compatible_type_t<Args>... args) {
    return fun(convert_from_pytorch_compatible_type<Args>(args)...);
  };
}
