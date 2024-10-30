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
  \brief Defines device-side elementwise operations on TensorView. Note, the operations defined
    in this header are not specialized for any particular data layout and are therefore not
    intended to offer the best possible performance. Rather, they are intended to be generic
    reference implementations to support the CUTLASS unit tests.
*/

#pragma once

// Cutlass includes
#include "cutlass/cutlass.h"
#include "cutlass/tensor_view.h"

#include "cutlass/util/reference/device/tensor_foreach.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reference {
namespace device {

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorReLuFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Coordinate in tensor's index space
  using TensorCoord = typename TensorView::TensorCoord;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    Element threshold;


    //
    // Methods
    //

    Params(
      TensorView view_ = TensorView(),
      Element threshold_ = Element(0)
    ):
      view(view_), threshold(threshold_) {

    }
  };

  //
  // Data members
  //

  Params params;

  //
  // Methods
  //

  CUTLASS_DEVICE
  TensorReLuFunc(Params const &params): params(params) {

  }

  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {

    Element const & value = params.view.at(coord);
    params.view.at(coord) = (value < params.threshold) ? params.threshold : value;
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Apply ReLu on a tensor
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorReLu(
  TensorView<Element, Layout> view,       ///< destination tensor
  Element threshold = Element(0)) {         ///< ReLu threshold
  
  using Func = detail::TensorReLuFunc<Element, Layout>;
  using Params = typename Func::Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, threshold)
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace reference
} // namespace cutlass
