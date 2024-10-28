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
  \brief Provides several functions for filling tensors with data.
*/

#pragma once

// Standard Library includes
#include <utility>
#include <cstdlib>
#include <cmath>

// Cute includes
#include "cute/tensor.hpp"

// Cutlass includes
#include "cutlass/cutlass.h"
#include "cutlass/complex.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/quaternion.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reference {
namespace host {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Tensor reductions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Transform-reduce operation over the elements of a tensor. This helper allocates the device-side
/// workspace
template <
  typename Tensor,
  typename ComputeType,
  typename ReduceOp,
  typename TransformOp
>
ComputeType TensorTransformReduce(
  Tensor view,
  ComputeType identity,
  ReduceOp reduce,
  TransformOp transform
) {

  for (int64_t idx = 0; idx < cute::size(view); ++idx) {
    identity = reduce(identity, transform(view(idx)));
  }

  return identity;
}

/// Transform-reduce operation over the elements of a tensor. This helper allocates the device-side
/// workspace
template <
  typename TensorA,
  typename TensorB,
  typename ComputeType,
  typename ReduceOp,
  typename TransformOp
>
ComputeType TensorTransformReduce(
  TensorA view_A,
  TensorB view_B,
  ComputeType identity,
  ReduceOp reduce,
  TransformOp transform) {
  
  if (cute::size(view_A) != cute::size(view_B)) {
    throw std::runtime_error("Tensor sizes must match.");
  }

  for (int64_t idx = 0; idx < cute::size(view_A); ++idx) {
    identity = reduce(identity, transform(view_A(idx), view_B(idx)));
  }

  return identity;
}

/// Helper to compute the sum of the elements of a tensor
template <
  typename Tensor,
  typename ComputeType = typename Tensor::value_type
>
ComputeType TensorSum(
  Tensor view,
  ComputeType identity = ComputeType()
) {

  plus<ComputeType> reduce;
  NumericConverter<ComputeType, typename Tensor::value_type> transform;

  return TensorTransformReduce(
    view, identity, reduce, transform);
}

/// Helper to compute the sum of the squares of the elements of a tensor
template <
  typename Tensor,
  typename ComputeType = typename Tensor::value_type
>
ComputeType TensorSumSq(
  Tensor view,
  ComputeType identity = ComputeType()
) {

  plus<ComputeType> reduce;
  magnitude_squared<typename Tensor::value_type, ComputeType> transform;

  return TensorTransformReduce(
    view, identity, reduce, transform);
}

/// Helper to compute the norm of the elements of a tensor.
template <
  typename Tensor,
  typename ComputeType = double
>
ComputeType TensorNorm(
  Tensor view,
  ComputeType identity = ComputeType()
) {

  return std::sqrt(TensorSumSq(view, identity));
}

/// Helper to compute the sum of the squares of the differences of two tensors
template <
  typename TensorA,
  typename TensorB,
  typename ComputeType = double
>
ComputeType TensorSumSqDiff(
  TensorA view_A,
  TensorB view_B,
  ComputeType identity = ComputeType()
) {

  plus<ComputeType> reduce;
  magnitude_squared_difference<typename TensorA::value_type, ComputeType> transform;

  return TensorTransformReduce(
    view_A, view_B, identity, reduce, transform);
}


/// Helper to compute the norm of the tensor computed as the difference of two tensors in memory
template <
  typename TensorA,
  typename TensorB,
  typename ComputeType = double
>
ComputeType TensorNormDiff(
  TensorA view_A,
  TensorB view_B,
  ComputeType identity = ComputeType()
) {

  return std::sqrt(TensorSumSqDiff(view_A, view_B, identity));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
