#pragma once

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/shim_utils.h>
#include <torch/headeronly/core/ScalarType.h>

#include "cute/layout.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

using ColumnMajor = typename cutlass::layout::ColumnMajor;
using RowMajor = typename cutlass::layout::RowMajor;

namespace cute {

namespace detail {

template <class T, class F, class G, int... I>
CUTE_HOST_DEVICE constexpr auto tapply_with_idx(T&& t, F&& f, G&& g,
                                                seq<I...>) {
  return g(f(cute::get<I>(static_cast<T&&>(t)), I)...);
}

template <class F, int... I>
CUTE_HOST_DEVICE constexpr auto make_shape_from_idx(F&& f, seq<I...>) {
  return make_shape(f(I)...);
}

};  // namespace detail

template <class T, class F>
CUTE_HOST_DEVICE constexpr auto transform_with_idx(T const& t, F&& f) {
  if constexpr (cute::is_tuple<T>::value) {
    return detail::tapply_with_idx(
        t, f, [](auto const&... a) { return cute::make_tuple(a...); },
        tuple_seq<T>{});
  } else {
    return f(t);
  }

  CUTE_GCC_UNREACHABLE;
}

// calls: make_shape(f(0), f(1), ..., f(N-1))
template <int N, class F>
CUTE_HOST_DEVICE constexpr auto make_shape_from_idx(F&& f) {
  return detail::make_shape_from_idx(f, make_seq<N>{});
}

};  // namespace cute

// Make a layout from a stable tensor with `rank(Stride{})`, where the shape is
// the shape of the passed in tensor and the strides are of type `Stride` and
// contain the strides of the passed in tensor, checking that any static strides
// in `Stride{}` match the strides of the passed in tensor.
// If `tensor.dim() < rank(Stride{})`, the shape is padded with 1s and the extra
// strides are set to be 0 or 1.
template <typename Stride>
static inline auto make_cute_layout(torch::stable::Tensor const& tensor,
                                    std::string_view name = "tensor") {
  STD_TORCH_CHECK(tensor.dim() <= cute::rank(Stride{}));
  auto stride = cute::transform_with_idx(Stride{}, [&](auto const& stride_ele,
                                                       auto const& idx) {
    using StrideEle = std::decay_t<decltype(stride_ele)>;

    if (idx < tensor.dim()) {
      if constexpr (cute::is_static_v<StrideEle>) {
        STD_TORCH_CHECK(StrideEle::value == tensor.stride(idx), "Expected ",
                        name, ".stride(", idx, ") to be ", StrideEle::value);
        return StrideEle{};
      } else {
        if (tensor.size(idx) == 1) {
          // use 0 stride for dim with size 1, this is easier for
          // cute/cutlass to optimize (helps the TMA code flatten dims)
          return StrideEle{0};
        } else {
          return tensor.stride(idx);
        }
      }
    } else {
      // Extra strides are assumed to be 0 or 1
      if constexpr (cute::is_static_v<StrideEle>) {
        static_assert(StrideEle::value == 0 || StrideEle::value == 1);
      }
      return StrideEle{};
    }
  });

  auto shape =
      cute::make_shape_from_idx<cute::rank(Stride{})>([&](auto const& idx) {
        if (idx < tensor.dim())
          return tensor.size(idx);
        else
          return int64_t(1);
      });

  return cute::make_layout(shape, stride);
}

template <typename Stride>
static inline auto maybe_make_cute_layout(
    std::optional<torch::stable::Tensor> const& tensor,
    std::string_view name = "tensor") {
  using Layout = decltype(make_cute_layout<Stride>(*tensor));

  if (tensor) {
    return std::optional<Layout>{make_cute_layout<Stride>(*tensor, name)};
  } else {
    return std::optional<Layout>{};
  }
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

// Note: In stable API, we don't have access to c10::Half/BFloat16 types
// The cutlass types are used directly
template <>
struct equivalent_cutlass_type<cutlass::half_t> {
  using type = cutlass::half_t;
};

template <>
struct equivalent_cutlass_type<cutlass::bfloat16_t> {
  using type = cutlass::bfloat16_t;
};

//
// equivalent_scalar_type (basically inverse of equivalent_cutlass_type)
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
static inline constexpr torch::headeronly::ScalarType equivalent_scalar_type_v =
    torch::headeronly::CppTypeToScalarType<equivalent_scalar_type_t<T>>::value;
