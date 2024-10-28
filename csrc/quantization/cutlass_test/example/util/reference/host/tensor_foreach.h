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

#include <stdexcept>
#include "cutlass/cutlass.h"

namespace cutlass  {
namespace reference {
namespace host {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines several helpers
namespace detail {

/// Helper to perform for-each operation
template <typename Func, int Rank, int RankRemaining>
struct TensorForEachHelper {

  /// Index of the active rank
  static int const kActiveRank = Rank - RankRemaining - 1;

  /// Constructor for general rank
  TensorForEachHelper(
    Func &func,
    Coord<Rank> const &extent,
    Coord<Rank> &coord) {

    for (int i = 0; i < extent.at(kActiveRank); ++i) {
      coord[kActiveRank] = i;
      TensorForEachHelper<Func, Rank, RankRemaining - 1>(func, extent, coord);
    }
  }
};

/// Helper to perform for-each operation
template <typename Func, int Rank>
struct TensorForEachHelper<Func, Rank, 0> {

  /// Index of the active rank
  static int const kActiveRank = Rank - 1;

  /// Constructor for fastest changing rank
  TensorForEachHelper(
    Func &func,
    Coord<Rank> const &extent,
    Coord<Rank> &coord) {

    for (int i = 0; i < extent.at(kActiveRank); ++i) {
      coord[kActiveRank] = i;
      func(coord);
    }
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterates over the index space of a tensor
template <
  typename Func,          ///< function applied to each point in a tensor's index space
  int Rank>               ///< rank of index space
void TensorForEach(Coord<Rank> extent, Func & func) {
  Coord<Rank> coord;
  detail::TensorForEachHelper<Func, Rank, Rank - 1>(func, extent, coord);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Iterates over the index space of a tensor and calls a C++ lambda
template <
  typename Func,          ///< function applied to each point in a tensor's index space
  int Rank>               ///< rank of index space
void TensorForEachLambda(Coord<Rank> extent, Func func) {
  Coord<Rank> coord;
  detail::TensorForEachHelper<Func, Rank, Rank - 1>(func, extent, coord);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, typename Func>
struct BlockForEach {

  /// Constructor performs the operation.
  BlockForEach(
    Element *ptr, 
    size_t capacity,
    typename Func::Params params = typename Func::Params()) {
  
    Func func(params);

    for (size_t index = 0; index < capacity; ++index) {
      ptr[index] = func();
    }    
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
