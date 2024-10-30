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
#pragma once

#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/complex.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/util/reference/detail/linear_to_coordinate.h"
#include "cutlass/core_io.h"

namespace cutlass  {
namespace reference {
namespace host {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Transform-reduce operation over the elements of a tensor. This helper allocates the device-side
/// workspace
template <
  typename Element,
  typename Layout,
  typename ComputeType,
  typename ReduceOp,
  typename TransformOp
>
ComputeType TensorTransformReduce(
  TensorView<Element, Layout> view,
  ComputeType identity,
  ReduceOp reduce,
  TransformOp transform
) {

  for (int64_t idx = 0; idx < int64_t(view.size()); ++idx) {
    typename Layout::TensorCoord coord;
    cutlass::reference::detail::LinearToCoordinate<Layout::kRank>()(coord, idx, view.extent());

    if (view.contains(coord)) {
      Element x = view.at(coord);
      identity = reduce(identity, transform(x));
    }
  }

  return identity;
}

/// Transform-reduce operation over the elements of a tensor. This helper allocates the device-side
/// workspace
template <
  typename Element,
  typename Layout,
  typename ComputeType,
  typename ReduceOp,
  typename TransformOp
>
ComputeType TensorTransformReduce(
  TensorView<Element, Layout> view_A,
  TensorView<Element, Layout> view_B,
  ComputeType identity,
  ReduceOp reduce,
  TransformOp transform) {
  
  if (view_A.extent() != view_B.extent()) {
    throw std::runtime_error("Tensor extents must match.");
  }

  for (int64_t idx = 0; idx < int64_t(view_A.size()); ++idx) {

    typename Layout::TensorCoord coord;
    cutlass::reference::detail::LinearToCoordinate<Layout::kRank>()(coord, idx, view_A.extent());

    if (view_A.contains(coord)) {
      Element a = view_A.at(coord);
      Element b = view_B.at(coord);
      identity = reduce(identity, transform(a, b));
    }
  }

  return identity;
}

/// Helper to compute the sum of the elements of a tensor
template <
  typename Element,
  typename Layout,
  typename ComputeType = Element
>
ComputeType TensorSum(
  TensorView<Element, Layout> view,
  ComputeType identity = ComputeType()
) {

  plus<ComputeType> reduce;
  NumericConverter<ComputeType, Element> transform;

  return TensorTransformReduce(
    view, identity, reduce, transform);
}

/// Helper to compute the sum of the squares of the elements of a tensor
template <
  typename Element,
  typename Layout,
  typename ComputeType = Element
>
ComputeType TensorSumSq(
  TensorView<Element, Layout> view,
  ComputeType identity = ComputeType()
) {

  plus<ComputeType> reduce;
  magnitude_squared<Element, ComputeType> transform;

  return TensorTransformReduce(
    view, identity, reduce, transform);
}

/// Helper to compute the norm of the elements of a tensor.
template <
  typename Element,
  typename Layout,
  typename ComputeType = double
>
ComputeType TensorNorm(
  TensorView<Element, Layout> view,
  ComputeType identity = ComputeType()
) {

  return std::sqrt(TensorSumSq(view, identity));
}

/// Helper to compute the sum of the squares of the differences of two tensors
template <
  typename Element,
  typename Layout,
  typename ComputeType = double
>
ComputeType TensorSumSqDiff(
  TensorView<Element, Layout> view_A,
  TensorView<Element, Layout> view_B,
  ComputeType identity = ComputeType()
) {

  plus<ComputeType> reduce;
  magnitude_squared_difference<Element, ComputeType> transform;

  return TensorTransformReduce(
    view_A, view_B, identity, reduce, transform);
}


/// Helper to compute the norm of the tensor computed as the difference of two tensors in memory
template <
  typename Element,
  typename Layout,
  typename ComputeType = double
>
ComputeType TensorNormDiff(
  TensorView<Element, Layout> view_A,
  TensorView<Element, Layout> view_B,
  ComputeType identity = ComputeType()
) {

  return std::sqrt(TensorSumSqDiff(view_A, view_B, identity));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
