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
/*! \file
    \brief Reference implementation for GEMM in host-side code.
*/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reference {
namespace detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Rank, int Index>
struct LinearToCoordinateHelper {

  CUTLASS_HOST_DEVICE
  void operator()(Coord<Rank> &coord, int64_t idx, Coord<Rank> const &extent) const {

    int64_t prod = 1;

    CUTLASS_PRAGMA_UNROLL
    for (int i = Rank - Index; i < Rank; ++i) {
      prod *= int64_t(extent[i]);
    }

    coord[Rank - Index - 1] = int(idx / prod);

    int64_t residual = idx % prod;
    LinearToCoordinateHelper<Rank, Index - 1>()(coord, residual, extent);
  }
};

template <int Rank>
struct LinearToCoordinateHelper<Rank, 0> {

  CUTLASS_HOST_DEVICE
  void operator()(Coord<Rank> &coord, int64_t idx, Coord<Rank> const &) const {
    coord[Rank - 1] = int(idx);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Rank>
struct LinearToCoordinate {

  CUTLASS_HOST_DEVICE
  void operator()(Coord<Rank> &coord, int64_t idx, Coord<Rank> const &extent) const {
    LinearToCoordinateHelper<Rank, Rank - 1>()(coord, idx, extent);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail
} // namespace reference
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

