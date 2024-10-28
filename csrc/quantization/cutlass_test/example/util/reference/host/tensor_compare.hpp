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
#include "cutlass/quaternion.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reference {
namespace host {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if two tensor views are equal.
template <
  typename TensorL,
  typename TensorR
>
bool TensorEquals(
  TensorL lhs,
  TensorR rhs) {

  // Extents must be identical
  if (cute::size(lhs) != cute::size(rhs)) {
    return false;
  }

  for (int64_t idx = 0; idx < cute::size(lhs); ++idx) {
    if (lhs(idx) != rhs(idx)) {
      return false;
    }
  }

  return true;
}

/// Returns true if two tensor views are NOT equal.
template <
  typename TensorL,
  typename TensorR
>
bool TensorNotEquals(
  TensorL lhs,
  TensorR rhs) {

  return TensorEquals(lhs, rhs);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
