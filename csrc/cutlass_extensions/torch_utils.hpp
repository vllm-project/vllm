#pragma once

#include <torch/all.h>

#include "cutlass/layout/matrix.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

using ColumnMajor = typename cutlass::layout::ColumnMajor;
using RowMajor = typename cutlass::layout::RowMajor;

static inline bool is_row_major(torch::Tensor const tensor) {
  TORCH_CHECK(tensor.dim() == 2);
  return tensor.is_contiguous();
}

static inline bool is_column_major(torch::Tensor const tensor) {
  TORCH_CHECK(tensor.dim() == 2);
  return tensor.stride(0) == 1 && tensor.stride(1) == tensor.size(0);
}

template <typename T, typename Layout = RowMajor>
T* maybe_data_ptr(c10::optional<torch::Tensor const> maybe_tensor,
                  char const* name) {
  if constexpr (std::is_same_v<Layout, RowMajor>) {
    TORCH_CHECK(!maybe_tensor || is_row_major(*maybe_tensor), "Expected ", name,
                " to be RowMajor");
  } else if constexpr (std::is_same_v<Layout, ColumnMajor>) {
    TORCH_CHECK(!maybe_tensor || is_column_major(*maybe_tensor), "Expected ",
                name, " to be ColumnMajor");
  } else {
    TORCH_CHECK(false, "Unknown Layout");
  }

  return (maybe_tensor == at::nullopt)
             ? nullptr
             : reinterpret_cast<T*>(maybe_tensor->data_ptr());
}

template <typename T, typename Layout = RowMajor>
T* data_ptr(torch::Tensor const tensor, char const* name) {
  if constexpr (std::is_same_v<Layout, RowMajor>) {
    TORCH_CHECK(is_row_major(tensor), "Expected ", name, " to be RowMajor");
  } else if constexpr (std::is_same_v<Layout, ColumnMajor>) {
    TORCH_CHECK(is_column_major(tensor), "Expected ", name,
                " to be ColumnMajor");
  } else {
    TORCH_CHECK(false, "Unknown Layout");
  }

  return reinterpret_cast<T*>(tensor.data_ptr());
}

//
//  Torch Type to Cutlass Type (equivalent_cutlass_type)
//

template <typename T>
struct equivalent_cutlass_type {
  using type = T;
};

template <typename T>
using equivalent_cutlass_type_t = typename equivalent_cutlass_type<T>::type;

template <>
struct equivalent_cutlass_type<c10::Half> {
  using type = cutlass::half_t;
};

template <>
struct equivalent_cutlass_type<c10::BFloat16> {
  using type = cutlass::bfloat16_t;
};

//
// equivalent_scalar_t (basically inverse of equivalent_cutlass_type)
//

// Return a `c10::CppTypeToScalarType<T>` compatible type, i.e. get the C++ from
// c10 that is equivalent to T, e.g.: `cutlass::half_t -> c10::Half`
template <typename T>
struct equivalent_scalar_type {
  using type = T;
};

template <typename T>
using equivalent_scalar_type_t = typename equivalent_scalar_type<T>::type;

template <>
struct equivalent_scalar_type<cutlass::half_t> {
  using type = c10::Half;
};

template <>
struct equivalent_scalar_type<cutlass::bfloat16_t> {
  using type = c10::BFloat16;
};

// get equivalent c10::ScalarType tag from compile time type
template <typename T>
static inline constexpr c10::ScalarType equivalent_scalar_type_v =
    c10::CppTypeToScalarType<equivalent_scalar_type_t<T>>::value;