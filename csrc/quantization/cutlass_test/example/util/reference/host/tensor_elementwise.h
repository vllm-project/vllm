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

// Cutlass includes
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"

#include "tensor_foreach.h"

namespace cutlass {
namespace reference {
namespace host {

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to apply a binary operator in place
template <
  typename ElementA, 
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementD,
  typename LayoutD,
  typename BinaryFunc>
struct TensorFuncBinaryOp {

  //
  // Data members
  //

  /// View of left-hand-side tensor
  TensorView<ElementD, LayoutD> view_d;
  TensorRef<ElementA, LayoutA> view_a;
  TensorRef<ElementB, LayoutB> view_b;
  BinaryFunc func;

  //
  // Methods
  //

  /// Constructor
  TensorFuncBinaryOp() { }

  /// Constructor
  TensorFuncBinaryOp(
    TensorView<ElementD, LayoutD> const & view_d_,
    TensorRef<ElementA, LayoutA> const & view_a_,
    TensorRef<ElementB, LayoutB> const & view_b_,
    BinaryFunc func = BinaryFunc()
  ):
    view_d(view_d_), view_a(view_a_), view_b(view_b_), func(func) { }

  /// Equality check
  void operator()(Coord<LayoutD::kRank> const &coord) const {
    view_d.at(coord) = func(
      ElementD(view_a.at(coord)),
      ElementD(view_b.at(coord))
    );
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Adds two tensors and stores in the destination tensor: d = a + b
template <
  typename ElementD,
  typename LayoutD,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB
>
void TensorAdd(
  TensorView<ElementD, LayoutD> d,      ///< destination tensor view
  TensorRef<ElementA, LayoutA> a,       ///< A tensor reference
  TensorRef<ElementB, LayoutB> b        ///< B tensor reference
) {

  detail::TensorFuncBinaryOp<
    ElementD, 
    LayoutD,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    cutlass::plus<ElementD>
  > func(d, a, b);

  TensorForEach(
    d.extent(),
    func); 
}

/// Adds a tensor in place: d = d .+ a
template <
  typename ElementD,
  typename LayoutD,
  typename ElementA,
  typename LayoutA
>
void TensorAdd(
  TensorView<ElementD, LayoutD> d,      ///< destination tensor view
  TensorRef<ElementA, LayoutA> a        ///< A tensor reference
) {
  TensorAdd(d, d, a);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Subtracts two tensors and stores in the destination tensor: d = a - b
template <
  typename ElementD,
  typename LayoutD,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB
>
void TensorSub(
  TensorView<ElementD, LayoutD> d,      ///< destination tensor view
  TensorRef<ElementA, LayoutA> a,       ///< A tensor reference
  TensorRef<ElementB, LayoutB> b        ///< B tensor reference
  ) {

  detail::TensorFuncBinaryOp<
    ElementD, 
    LayoutD,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    cutlass::minus<ElementD>
  > func(d, a, b);

  TensorForEach(
    d.extent(),
    func);
}

/// Subtracts two tensors in place: d = d .- a
template <
  typename ElementD,
  typename LayoutD,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB
>
void TensorSub(
  TensorView<ElementD, LayoutD> d,      ///< destination tensor view
  TensorRef<ElementA, LayoutA> a        ///< A tensor reference
  ) {
  
  TensorSub(d, d, a);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Multiplies two tensors and stores in the destination tensor: d = a .* b
template <
  typename ElementD,
  typename LayoutD,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB
>
void TensorMul(
  TensorView<ElementD, LayoutD> d,      ///< destination tensor view
  TensorRef<ElementA, LayoutA> a,       ///< A tensor reference
  TensorRef<ElementB, LayoutB> b        ///< B tensor reference
) {
  
  detail::TensorFuncBinaryOp<
    ElementD, 
    LayoutD,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    cutlass::multiplies<ElementD>
  > func(d, a, b);

  TensorForEach(
    d.extent(),
    func);
}

/// Multiplies tensors in place: d = d .* a
template <
  typename ElementD,
  typename LayoutD,
  typename ElementA,
  typename LayoutA
>
void TensorMul(
  TensorView<ElementD, LayoutD> d,      ///< destination tensor view
  TensorRef<ElementA, LayoutA> a        ///< A tensor reference
) {
  TensorMul(d, d, a);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Divides two tensors and stores in the destination tensor: d = a ./ b
template <
  typename ElementD,
  typename LayoutD,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB
>
void TensorDiv(
  TensorView<ElementD, LayoutD> d,      ///< destination tensor view
  TensorRef<ElementA, LayoutA> a,       ///< A tensor reference
  TensorRef<ElementB, LayoutB> b        ///< B tensor reference
) {
  
  detail::TensorFuncBinaryOp<
    ElementD, 
    LayoutD,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    cutlass::divides<ElementD>
  > func(d, a, b);

  TensorForEach(
    d.extent(),
    func);
}

/// Divides tensors in place: d = d ./ a
template <
  typename ElementD,
  typename LayoutD,
  typename ElementA,
  typename LayoutA
>
void TensorDiv(
  TensorView<ElementD, LayoutD> d,      ///< destination tensor view
  TensorRef<ElementA, LayoutA> a        ///< A tensor reference
) {
  TensorDiv(d, d, a);
}


///////////////////////////////////////////////////////////////////////////////////////////////////

/// Divides two tensors and stores in the destination tensor: d = a ./ b
template <
  typename ElementD,
  typename LayoutD,
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB
>
void TensorModulus(
  TensorView<ElementD, LayoutD> d,      ///< destination tensor view
  TensorRef<ElementA, LayoutA> a,       ///< A tensor reference
  TensorRef<ElementB, LayoutB> b        ///< B tensor reference
) {
  
  detail::TensorFuncBinaryOp<
    ElementD, 
    LayoutD,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    cutlass::divides<ElementD>
  > func(d, a, b);

  TensorForEach(
    d.extent(),
    func);
}

/// Divides tensors in place: d = d ./ a
template <
  typename ElementD,
  typename LayoutD,
  typename ElementA,
  typename LayoutA
>
void TensorModulus(
  TensorView<ElementD, LayoutD> d,      ///< destination tensor view
  TensorRef<ElementA, LayoutA> a        ///< A tensor reference
) {
  TensorDiv(d, d, a);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
