#pragma once

#include <cute/tensor.hpp>

////////////////////////////////////////////////////////////////////
// make_cute_stride
////////////////////////////////////////////////////////////////////

//
// Row Major Batched
//
template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<IntT, cute::Int<1>> make_cute_stride(
    cute::Stride<IntT, cute::Int<1>> s, cute::Shape<int, int, int> shape_MNL) {
  static_assert(std::is_integral_v<IntT>,
                "Stride must have an integral type so it can be set "
                "dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<0>(s_copy) = static_cast<IntT>(cute::get<1>(shape_MNL));
  return s_copy;
}

template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<IntT, cute::Int<1>> make_cute_stride(
    cute::Stride<IntT, cute::Int<1>> s, int M, int N, int L) {
  return make_cute_stride(s, cute::make_shape(M, N, L));
}

template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<cute::Int<1>, IntT> make_cute_stride(
    cute::Stride<cute::Int<1>, IntT> s, cute::Shape<int, int, int> shape_MNL) {
  static_assert(std::is_integral_v<IntT>,
                "Stride must have an integral type so it can be set "
                "dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<1>(s_copy) = static_cast<IntT>(cute::get<0>(shape_MNL));
  return s_copy;
}

template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<cute::Int<1>, IntT> make_cute_stride(
    cute::Stride<cute::Int<1>, IntT> s, int M, int N, int L) {
  return make_cute_stride(s, cute::make_shape(M, N, L));
}

//
// Row Major Batched
//
template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<IntT, cute::Int<1>, int64_t> make_cute_stride(
    cute::Stride<IntT, cute::Int<1>, int64_t> s,
    cute::Shape<int, int, int> shape_MNL) {
  static_assert(std::is_integral_v<IntT>,
                "Stride must have an integral type so it can be set "
                "dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<0>(s_copy) = static_cast<IntT>(cute::get<1>(shape_MNL));
  int batch_count = cute::get<2>(shape_MNL);
  if (batch_count > 1) {
    cute::get<2>(s_copy) =
        static_cast<IntT>(cute::get<0>(shape_MNL) * cute::get<1>(shape_MNL));
  } else {
    cute::get<2>(s_copy) = static_cast<IntT>(0);
  }
  return s_copy;
}

template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<IntT, cute::Int<1>, int64_t> make_cute_stride(
    cute::Stride<IntT, cute::Int<1>, int64_t> s, int M, int N, int L) {
  return make_cute_stride(s, cute::make_shape(M, N, L));
}

//
// Col Major Batched
//
template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<cute::Int<1>, IntT, int64_t> make_cute_stride(
    cute::Stride<cute::Int<1>, IntT, int64_t> s,
    cute::Shape<int, int, int> shape_MNL) {
  static_assert(std::is_integral_v<IntT>,
                "Stride must have an integral type so it can be set "
                "dynamically. Static strides not supported.");
  auto s_copy = s;
  cute::get<1>(s_copy) = static_cast<IntT>(cute::get<0>(shape_MNL));
  int batch_count = cute::get<2>(shape_MNL);
  if (batch_count > 1) {
    cute::get<2>(s_copy) =
        static_cast<IntT>(cute::get<0>(shape_MNL) * cute::get<1>(shape_MNL));
  } else {
    cute::get<2>(s_copy) = static_cast<IntT>(0);
  }
  return s_copy;
}

template <class IntT>
CUTLASS_HOST_DEVICE cute::Stride<cute::Int<1>, IntT, int64_t> make_cute_stride(
    cute::Stride<cute::Int<1>, IntT, int64_t> s, int M, int N, int L) {
  return make_cute_stride(s, cute::make_shape(M, N, L));
}

////////////////////////////////////////////////////////////////////
// Pointer utils
////////////////////////////////////////////////////////////////////

template <class PointerType>
static constexpr auto get_logical_ptr(PointerType* ptr) {
  if constexpr (cute::sizeof_bits_v<PointerType> < 8) {
    return cute::subbyte_iterator<PointerType>(ptr);
  } else {
    return ptr;
  }
}