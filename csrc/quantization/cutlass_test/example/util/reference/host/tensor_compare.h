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
#include "cutlass/relatively_equal.h"
#include "cutlass/tensor_view.h"
#include "cutlass/tensor_view_planar_complex.h"

#include "cutlass/util/distribution.h"
#include "tensor_foreach.h"

namespace cutlass {
namespace reference {
namespace host {

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorEqualsFunc {

  //
  // Data members
  //

  TensorView<Element, Layout> lhs;
  TensorView<Element, Layout> rhs;
  bool result;

  /// Ctor
  TensorEqualsFunc(): result(true) { }

  /// Ctor
  TensorEqualsFunc(
    TensorView<Element, Layout> const &lhs_,
    TensorView<Element, Layout> const &rhs_
  ) :
    lhs(lhs_), rhs(rhs_), result(true) { }

  /// Visits a coordinate
  void operator()(Coord<Layout::kRank> const &coord) {

    Element lhs_ = lhs.at(coord);
    Element rhs_ = rhs.at(coord);

    if (lhs_ != rhs_) {
      result = false;
    }
  }

  /// Returns true if equal
  operator bool() const {
    return result;
  }
};

template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorRelativelyEqualsFunc {

  //
  // Data members
  //

  TensorView<Element, Layout> lhs;
  TensorView<Element, Layout> rhs;
  Element epsilon;
  Element nonzero_floor;
  bool result;

  /// Ctor
  TensorRelativelyEqualsFunc(
    TensorView<Element, Layout> const &lhs_,
    TensorView<Element, Layout> const &rhs_,
    Element epsilon_,
    Element nonzero_floor_
  ) :
    lhs(lhs_),
    rhs(rhs_),
    epsilon(epsilon_),
    nonzero_floor(nonzero_floor_),
    result(true) { }

  /// Visits a coordinate
  void operator()(Coord<Layout::kRank> const &coord) {

    Element lhs_ = lhs.at(coord);
    Element rhs_ = rhs.at(coord);

    if (!relatively_equal(lhs_, rhs_, epsilon, nonzero_floor)) {
      result = false;
    }
  }

  /// Returns true if equal
  operator bool() const {
    return result;
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if two tensor views are equal.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
bool TensorEquals(
  TensorView<Element, Layout> const &lhs,
  TensorView<Element, Layout> const &rhs) {

  // Extents must be identical
  if (lhs.extent() != rhs.extent()) {
    return false;
  }

  detail::TensorEqualsFunc<Element, Layout> func(lhs, rhs);
  TensorForEach(
    lhs.extent(),
    func
  );

  return bool(func);
}

/// Returns true if two tensor views are equal.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
bool TensorEquals(
  TensorViewPlanarComplex<Element, Layout> const &lhs,
  TensorViewPlanarComplex<Element, Layout> const &rhs) {

  // Extents must be identical
  if (lhs.extent() != rhs.extent()) {
    return false;
  }

  detail::TensorEqualsFunc<Element, Layout> real_func(
    {lhs.data(), lhs.layout(), lhs.extent()},
    {rhs.data(), rhs.layout(), rhs.extent()}
  );

  TensorForEach(
    lhs.extent(),
    real_func
  );

  if (!bool(real_func)) {
    return false;
  }

  detail::TensorEqualsFunc<Element, Layout> imag_func(
    {lhs.data() + lhs.imaginary_stride(), lhs.layout(), lhs.extent()}, 
    {rhs.data() + rhs.imaginary_stride(), rhs.layout(), rhs.extent()}
    );

  TensorForEach(
    lhs.extent(),
    imag_func
  );

  return bool(imag_func);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if two tensor views are relatively equal.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
bool TensorRelativelyEquals(
  TensorView<Element, Layout> const &lhs,
  TensorView<Element, Layout> const &rhs,
  Element epsilon,
  Element nonzero_floor) {

  // Extents must be identical
  if (lhs.extent() != rhs.extent()) {
    return false;
  }

  detail::TensorRelativelyEqualsFunc<Element, Layout> func(lhs, rhs, epsilon, nonzero_floor);
  TensorForEach(
    lhs.extent(),
    func
  );

  return bool(func);
}

/// Returns true if two tensor views are relatively equal.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
bool TensorRelativelyEquals(
  TensorViewPlanarComplex<Element, Layout> const &lhs,
  TensorViewPlanarComplex<Element, Layout> const &rhs,
  Element epsilon,
  Element nonzero_floor) {

  // Extents must be identical
  if (lhs.extent() != rhs.extent()) {
    return false;
  }

  detail::TensorRelativelyEqualsFunc<Element, Layout> real_func(
    {lhs.data(), lhs.layout(), lhs.extent()},
    {rhs.data(), rhs.layout(), rhs.extent()},
    epsilon,
    nonzero_floor
  );

  TensorForEach(
    lhs.extent(),
    real_func
  );

  if (!bool(real_func)) {
    return false;
  }

  detail::TensorEqualsFunc<Element, Layout> imag_func(
    {lhs.data() + lhs.imaginary_stride(), lhs.layout(), lhs.extent()},
    {rhs.data() + rhs.imaginary_stride(), rhs.layout(), rhs.extent()},
    epsilon,
    nonzero_floor
  );

  TensorForEach(
    lhs.extent(),
    imag_func
  );

  return bool(imag_func);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if two tensor views are NOT equal.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
bool TensorNotEquals(
  TensorView<Element, Layout> const &lhs,
  TensorView<Element, Layout> const &rhs) {

  // Extents must be identical
  if (lhs.extent() != rhs.extent()) {
    return true;
  }

  detail::TensorEqualsFunc<Element, Layout> func(lhs, rhs);
  TensorForEach(
    lhs.extent(),
    func
  );

  return !bool(func);
}

/// Returns true if two tensor views are equal.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
bool TensorNotEquals(
  TensorViewPlanarComplex<Element, Layout> const &lhs,
  TensorViewPlanarComplex<Element, Layout> const &rhs) {

  return !TensorEquals(lhs, rhs);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorContainsFunc {

  //
  // Data members
  //

  TensorView<Element, Layout> view;
  Element value;
  bool contains;
  Coord<Layout::kRank> location;

  //
  // Methods
  //

  /// Ctor
  TensorContainsFunc(): contains(false) { }

  /// Ctor
  TensorContainsFunc(
    TensorView<Element, Layout> const &view_,
    Element value_
  ) :
    view(view_), value(value_), contains(false) { }

  /// Visits a coordinate
  void operator()(Coord<Layout::kRank> const &coord) {

    if (view.at(coord) == value) {
      if (!contains) {
        location = coord;
      }
      contains = true;
    }
  }

  /// Returns true if equal
  operator bool() const {
    return contains;
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns true if a value is present in a tensor
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
bool TensorContains(
  TensorView<Element, Layout> const & view,
  Element value) {

  detail::TensorContainsFunc<Element, Layout> func(
    view,
    value
  );

  TensorForEach(
    view.extent(),
    func
  );

  return bool(func);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns a pair containing a boolean of whether a value exists in a tensor and the location of
/// of the first occurrence. If the value is not contained in the tensor, the second element of the
/// pair is undefined.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
std::pair<bool, Coord<Layout::kRank> > TensorFind(
  TensorView<Element, Layout> const & view,
  Element value) {

  detail::TensorContainsFunc<Element, Layout> func(
    view,
    value
  );

  TensorForEach(
    view.extent(),
    func
  );

  return std::make_pair(bool(func), func.location);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
