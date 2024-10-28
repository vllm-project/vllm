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

#include "cutlass/core_io.h"
#include "cutlass/tensor_view.h"
#include "cutlass/tensor_view_planar_complex.h"
#include "cutlass/complex.h"

namespace cutlass {

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Helper to write the least significant rank of a TensorView
template <
  typename Element,
  typename Layout
>
inline std::ostream & TensorView_WriteLeastSignificantRank(
  std::ostream& out, 
  TensorView<Element, Layout> const& view,
  Coord<Layout::kRank> const &start_coord,
  int rank,
  std::streamsize width) {

  for (int idx = 0; idx < view.extent(rank); ++idx) {

    Coord<Layout::kRank> coord(start_coord);
    coord[rank] = idx;

    if (idx) {
      out.width(0);
      out << ", ";
    }
    if (idx || coord) {
      out.width(width);
    }
    out << ScalarIO<Element>(view.at(coord));
  }

  return out;
}

/// Helper to write a rank of a TensorView
template <
  typename Element,
  typename Layout
>
inline std::ostream & TensorView_WriteRank(
  std::ostream& out, 
  TensorView<Element, Layout> const& view,
  Coord<Layout::kRank> const &start_coord,
  int rank,
  std::streamsize width) {

  // If called on the least significant rank, write the result as a row
  if (rank + 1 == Layout::kRank) {
    return TensorView_WriteLeastSignificantRank(out, view, start_coord, rank, width);
  }

  // Otherwise, write a sequence of rows and newlines
  for (int idx = 0; idx < view.extent(rank); ++idx) {

    Coord<Layout::kRank> coord(start_coord);
    coord[rank] = idx;

    if (rank + 2 == Layout::kRank) {
      // Write least significant ranks asa matrix with rows delimited by "\n"
      if (idx) {
        out << ",\n";
      }
      TensorView_WriteLeastSignificantRank(out, view, coord, rank + 1, width);
    }
    else {
      // Higher ranks are separated by newlines
      if (idx) {
        out << ",\n\n";
      }
      TensorView_WriteRank(out, view, coord, rank + 1, width);
    }
  }

  return out;
}

/// Helper to write the least significant rank of a TensorView
template <
  typename Element,
  typename Layout
>
inline std::ostream & TensorViewPlanarComplex_WriteLeastSignificantRank(
  std::ostream& out, 
  TensorViewPlanarComplex<Element, Layout> const& view,
  Coord<Layout::kRank> const &start_coord,
  int rank,
  std::streamsize width) {

  for (int idx = 0; idx < view.extent(rank); ++idx) {

    Coord<Layout::kRank> coord(start_coord);
    coord[rank] = idx;

    if (idx) {
      out.width(0);
      out << ", ";
    }
    if (idx || coord) {
      out.width(width);
    }

    complex<Element> x = view.at(coord);
    out << x;
  }

  return out;
}

/// Helper to write a rank of a TensorView
template <
  typename Element,
  typename Layout
>
inline std::ostream & TensorViewPlanarComplex_WriteRank(
  std::ostream& out, 
  TensorViewPlanarComplex<Element, Layout> const& view,
  Coord<Layout::kRank> const &start_coord,
  int rank,
  std::streamsize width) {

  // If called on the least significant rank, write the result as a row
  if (rank + 1 == Layout::kRank) {
    return TensorViewPlanarComplex_WriteLeastSignificantRank(out, view, start_coord, rank, width);
  }

  // Otherwise, write a sequence of rows and newlines
  for (int idx = 0; idx < view.extent(rank); ++idx) {

    Coord<Layout::kRank> coord(start_coord);
    coord[rank] = idx;

    if (rank + 2 == Layout::kRank) {
      // Write least significant ranks asa matrix with rows delimited by ";\n"
      if (idx) {
        out << ";\n";
      }
      TensorViewPlanarComplex_WriteLeastSignificantRank(out, view, coord, rank + 1, width);
    }
    else {
      // Higher ranks are separated by newlines
      if (idx) {
        out << "\n";
      }
      TensorViewPlanarComplex_WriteRank(out, view, coord, rank + 1, width);
    }
  }

  return out;
}

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Prints human-readable representation of a TensorView to an ostream
template <
  typename Element,
  typename Layout
>
inline std::ostream& TensorViewWrite(
  std::ostream& out, 
  TensorView<Element, Layout> const& view) {

  // Prints a TensorView according to the following conventions:
  //   - least significant rank is printed as rows separated by ";\n"
  //   - all greater ranks are delimited with newlines
  //
  // The result is effectively a whitespace-delimited series of 2D matrices.

  return detail::TensorView_WriteRank(out, view, Coord<Layout::kRank>(), 0, out.width());
}

/// Prints human-readable representation of a TensorView to an ostream
template <
  typename Element,
  typename Layout
>
inline std::ostream& operator<<(
  std::ostream& out, 
  TensorView<Element, Layout> const& view) {

  // Prints a TensorView according to the following conventions:
  //   - least significant rank is printed as rows separated by ";\n"
  //   - all greater ranks are delimited with newlines
  //
  // The result is effectively a whitespace-delimited series of 2D matrices.

  return TensorViewWrite(out, view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Prints human-readable representation of a TensorView to an ostream
template <
  typename Element,
  typename Layout
>
inline std::ostream& TensorViewWrite(
  std::ostream& out, 
  TensorViewPlanarComplex<Element, Layout> const& view) {

  // Prints a TensorView according to the following conventions:
  //   - least significant rank is printed as rows separated by ";\n"
  //   - all greater ranks are delimited with newlines
  //
  // The result is effectively a whitespace-delimited series of 2D matrices.

  return detail::TensorViewPlanarComplex_WriteRank(out, view, Coord<Layout::kRank>(), 0, out.width());
}

/// Prints human-readable representation of a TensorView to an ostream
template <
  typename Element,
  typename Layout
>
inline std::ostream& operator<<(
  std::ostream& out, 
  TensorViewPlanarComplex<Element, Layout> const& view) {

  // Prints a TensorView according to the following conventions:
  //   - least significant rank is printed as rows separated by ";\n"
  //   - all greater ranks are delimited with newlines
  //
  // The result is effectively a whitespace-delimited series of 2D matrices.

  return TensorViewWrite(out, view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass
