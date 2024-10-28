/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/* \file
  \brief Defines host-side elementwise operations on TensorView.
*/

#pragma once

// Standard Library includes
#include <utility>

// Cutlass includes
#include "cutlass/cutlass.h"
#include "tensor_foreach.h"

namespace cutlass {
namespace reference {
namespace host {

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Helper to convert between types
template <
  typename DstElement,
  typename SrcElement
>
struct TrivialConvert {

  TrivialConvert() { }

  DstElement operator()(SrcElement src) const {
    return DstElement(src);
  }
};

/// Helper to conditionally copy between tensor views.
template <
  typename DstElement,
  typename DstLayout,
  typename SrcElement,
  typename SrcLayout,
  typename F
>
struct TensorCopyIf {

  using DstTensorView = TensorView<DstElement, DstLayout>;
  using SrcTensorView = TensorView<SrcElement, SrcLayout>;

  //
  // Data members
  //

  DstTensorView dst;
  SrcTensorView src;
  F convert;

  //
  // Methods
  //

  TensorCopyIf() { }

  TensorCopyIf(
    DstTensorView const &dst_, 
    SrcTensorView const &src_,
    F const &convert_): dst(dst_), src(src_), convert(convert_) {}

  /// Copies based on destination and source bounds
  void operator()(Coord<DstLayout::kRank> const &coord) {
    if (dst.contains(coord) && src.contains(coord)) {
      dst.at(coord) = convert(src.at(coord));
    }
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Copies elements from one tensor view into another, satisfying bounds of each tensor.
template <
  typename DstElement,          /// Destination tensor's element type
  typename DstLayout,           /// Destination tensor's layout
  typename SrcElement,          /// Source tensor's element type
  typename SrcLayout,           /// Source tensor's layout
  typename F                    /// Transformation functor
>
void TensorCopy(
  TensorView<DstElement, DstLayout> dst,
  TensorView<SrcElement, SrcLayout> src,
  F const &transform) {

  using CopyIf = detail::TensorCopyIf<
    DstElement,
    DstLayout,
    SrcElement,
    SrcLayout,
    F>;

  CopyIf copy_if(dst, src, transform);

  TensorForEach(dst.extent(), copy_if);
}


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Copies elements from a TensorRef into a TensorView. Assumes source tensor has sufficient extent
/// to avoid out of bounds accesses.
template <
  typename DstElement,          /// Destination tensor's element type
  typename DstLayout,           /// Destination tensor's layout
  typename SrcElement,          /// Source tensor's element type
  typename SrcLayout,           /// Source tensor's layout
  typename F                    /// Transformation functor
>
void TensorCopy(
  TensorView<DstElement, DstLayout> dst,
  TensorRef<SrcElement, SrcLayout> src,
  F const &transform) {

  using CopyIf = detail::TensorCopyIf<
    DstElement,
    DstLayout,
    SrcElement,
    SrcLayout,
    F>;

  TensorView<SrcElement, SrcLayout> src_view(src, dst.extent());

  CopyIf copy_if(dst, src_view, transform);

  TensorForEach(dst.extent(), copy_if);
}

/// Copies elements from a TensorRef into a TensorView. Assumes source tensor has sufficient extent
/// to avoid out of bounds accesses.
template <
  typename DstElement,          /// Destination tensor's element type
  typename DstLayout,           /// Destination tensor's layout
  typename SrcElement,          /// Source tensor's element type
  typename SrcLayout,           /// Source tensor's layout
  typename F                    /// Transformation functor
>
void TensorCopy(
  TensorRef<DstElement, DstLayout> dst,
  TensorView<SrcElement, SrcLayout> src,
  F const &transform) {

  using CopyIf = detail::TensorCopyIf<
    DstElement,
    DstLayout,
    SrcElement,
    SrcLayout,
    F>;

  TensorView<DstElement, DstLayout> dst_view(dst, src.extent());

  CopyIf copy_if(dst_view, src, transform);

  TensorForEach(src.extent(), copy_if);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Copies elements from one tensor view into another, satisfying bounds of each tensor. Succeeds
/// if SrcElement can be converted to DstElement.
template <
  typename DstElement,          /// Destination tensor's element type
  typename DstLayout,           /// Destination tensor's layout
  typename SrcElement,          /// Source tensor's element type
  typename SrcLayout            /// Source tensor's layout
>
void TensorCopy(
  TensorView<DstElement, DstLayout> dst,
  TensorView<SrcElement, SrcLayout> src) {

  detail::TrivialConvert<DstElement, SrcElement> convert;

  TensorCopy(dst, src, convert);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Copies elements from one tensor view into another, satisfying bounds of each tensor. Succeeds
/// if SrcElement can be converted to DstElement.
template <
  typename DstElement,          /// Destination tensor's element type
  typename DstLayout,           /// Destination tensor's layout
  typename SrcElement,          /// Source tensor's element type
  typename SrcLayout,           /// Source tensor's layout
  typename F                    /// Transformation functor
>
void TensorCopy(
  TensorView<DstElement, DstLayout> dst,
  TensorRef<SrcElement, SrcLayout> src) {

  detail::TrivialConvert<DstElement, SrcElement> convert;

  TensorCopy(dst, src, convert);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Copies elements from one tensor view into another, satisfying bounds of each tensor. Succeeds
/// if SrcElement can be converted to DstElement.
template <
  typename DstElement,          /// Destination tensor's element type
  typename DstLayout,           /// Destination tensor's layout
  typename SrcElement,          /// Source tensor's element type
  typename SrcLayout            /// Source tensor's layout
>
void TensorCopy(
  TensorRef<DstElement, DstLayout> dst,
  TensorView<SrcElement, SrcLayout> src) {

  detail::TrivialConvert<DstElement, SrcElement> convert;

  TensorCopy(dst, src, convert);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
