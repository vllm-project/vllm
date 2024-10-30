/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

namespace example {

using namespace cute;

// Empty type used to disable gather/scatter for a GEMM argument
struct NoGather
{
  template<class... Ts>
  NoGather(Ts...) {};
};

/// Function object that applies an index to its argument
template <class Index>
struct IndexedGather
{
  CUTE_HOST_DEVICE constexpr
  IndexedGather(Index const *indices = {}): indices_(indices) {}

  template <typename I>
  CUTE_HOST_DEVICE constexpr
  Index
  operator()(I i) const { return indices_[i]; }

  CUTE_HOST_DEVICE friend
  void 
  print(IndexedGather const &s) {
    cute::print("Indexed");
  }

  Index const *indices_;
};

/// Function object that applies a stride to its argument
/// Example: StridedFunc<int,_2> gathers every other row/column
template <class Stride>
struct StridedGather
{
  CUTE_HOST_DEVICE constexpr
  StridedGather(Stride stride = {}): stride_(stride) {}

  template <class I>
  CUTE_HOST_DEVICE constexpr
  auto
  operator()(I i) const { return i * stride_; }

  CUTE_HOST_DEVICE friend
  void 
  print(StridedGather const &s) {
    cute::print("Strided{");
    print(s.stride_);
    cute::print("}");
  }

  Stride stride_;
};

/// Custom stride object that applies a function followed by a stride
template <class Func, class Stride>
struct CustomStride
{
  CUTE_HOST_DEVICE constexpr
  CustomStride(Func const &func, Stride const &stride): func_(func), stride_(stride) {}

  template <class I>
  CUTE_HOST_DEVICE constexpr friend
  auto
  operator*(I i, CustomStride const &s) { return s.func_(i) * s.stride_; }

  template <class I>
  CUTE_HOST_DEVICE constexpr friend
  auto
  operator*(CustomStride const &s, I i) { return s.func_(i) * s.stride_; }

  CUTE_HOST_DEVICE friend
  void
  print(CustomStride const & s) {
    cute::print("Custom{");
    print(s.func_);
    cute::print(",");
    print(s.stride_);
    cute::print("}");
  }

  template<class Div>
  CUTE_HOST_DEVICE constexpr friend
  auto
  safe_div(CustomStride const &s, Div const &div)
  {
    return CustomStride<Func, decltype(safe_div(s.stride_, div))>(s.func_, safe_div(s.stride_, div));
  }

  // Circumvent the requirement on make_layout that shape and stride are integral
  template <class Shape>
  CUTE_HOST_DEVICE constexpr friend
  auto
  make_layout(Shape const &shape, CustomStride const &stride)
  {
    return Layout<Shape, CustomStride>(shape, stride);
  }

  Func func_;
  Stride stride_;
};

template<class Stride, class Func>
CUTLASS_HOST_DEVICE
auto
make_custom_stride_layout(Stride const &stride, Func&& func)
{
  // Use a dummy shape and replace the first non-unit stride with a custom gather stride
  auto idx = find_if(stride, [](auto x){ return not is_constant<1, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;
  return make_layout(repeat_like(stride, _1{}),
                     replace<I>(stride, CustomStride{static_cast<Func&&>(func), get<I>(stride)}));
}

/// Helper function to optionally create a gather tensor
template<class Iterator, class Shape, class Stride, class Func>
CUTLASS_HOST_DEVICE
auto 
make_gather_tensor(Iterator iter, Shape const &shape, Stride const &stride, Func &&func)
{
  if constexpr (not cutlass::platform::is_same<remove_cvref_t<Func>, NoGather>::value) {
    Layout matrix_layout = make_identity_layout(shape);
    auto offset = as_arithmetic_tuple(repeat_like(shape, _0{}));
    Layout gather_layout = make_custom_stride_layout(stride, static_cast<Func&&>(func));
    return make_tensor(iter, ComposedLayout{gather_layout, offset, matrix_layout});
  } else {
    return make_tensor(iter, shape, stride);
  }
}

} // namespace example

namespace cute
{

template<int N, int I, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
upcast(Shape const& shape, Stride const& stride)
{
  if constexpr (is_tuple<Shape>::value) {
    return transform_layout(shape, stride, [](auto const& s, auto const& d) { return upcast<N,I>(s,d); });
  } else if constexpr (is_scaled_basis<Stride>::value) {
    if constexpr (Stride::mode() == I) {
      return make_layout(shape_div(shape, Int<N>{}), shape_div(stride, Int<N>{}));
    } else {
      return make_layout(shape, stride);
    }
  } else {
    return upcast<N>(shape, stride);
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N, class OuterShape, class OuterStride, class Offset, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto
upcast(ComposedLayout<Layout<OuterShape,OuterStride>,Offset,Layout<Shape,Stride>> const& layout)
{
  // Find index of the stride-1 mode - that is the only one that requires updating inner shape and offset
  auto idx = find_if(layout.layout_a().stride(), [](auto x){ return is_constant<1, decltype(x)>{}; });
  constexpr int I = decltype(idx)::value;

  // Upcast the outer layout (works as expected)
  auto outer = upcast<N>(layout.layout_a());

  // Upcast the accumulated offset along stride-1 mode
  auto offset = as_arithmetic_tuple(replace<I>(layout.offset(), upcast<N>(get<I>(layout.offset()))));

  // Upcast the inner layout's shape along stride-1 mode
  auto inner = upcast<N,I>(layout.layout_b().shape(), layout.layout_b().stride());

  return composition(outer, offset, inner);
}

} // namespace example
