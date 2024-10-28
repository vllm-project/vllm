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

/*! \file
    \brief This header contains a class to parametrize a statistical distribution function.
*/

#include <ostream>

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Distribution type
struct Distribution {
  /// Variant types
  enum Kind { Invalid, Uniform, Gaussian, Identity, Sequential, AllZeros, AllOnes };

  /// Distribution state
  union {
    /// Uniform distribution
    struct {
      double min;
      double max;
      // Percent elements set to NaN
      double pnan;
    } uniform;

    /// Gaussian distribution
    struct {
      double mean;
      double stddev;
      double pnz;
      double pnzA;
      double pnzB;
      double pnzC;
    } gaussian;

    /// Elements are linear combination of row and column index
    struct {
      double start;
      double delta;
    } sequential;
  };

  /// Active variant kind
  Kind kind;

  /// Random values are cast to integer after scaling by this power of two
  int int_scale;

  //
  // Methods
  //

  Distribution() : kind(Invalid), int_scale(0) {}

/// Configures distribution as uniform random
  Distribution &set_uniform(double _min, double _max, int _int_scale = 0, double _pnan = 0) {
    kind = Uniform;
    uniform.min = _min;
    uniform.max = _max;
    int_scale = _int_scale;
    uniform.pnan = _pnan;
    return *this;
  }

  /// Configures distribution as Gaussian distribution
  Distribution &set_gaussian(double _mean, double _stddev, int _int_scale = 0, double _pnz = 1.0) {
    kind = Gaussian;
    gaussian.mean = _mean;
    gaussian.stddev = _stddev;
    gaussian.pnz = _pnz;
    int_scale = _int_scale;
    return *this;
  }

  /// Sets identity
  Distribution &set_identity() {
    kind = Identity;
    return *this;
  }

  /// Sets sequential
  Distribution &set_sequential(double start, double delta, int _int_scale = 0) {
    kind = Sequential;
    sequential.start = start;
    sequential.delta = delta;
    int_scale = _int_scale;
    return *this;
  }
};

}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Prints a Distribution to ostream
inline std::ostream &operator<<(std::ostream &out, cutlass::Distribution const &dist) {
  switch (dist.kind) {
    case cutlass::Distribution::Uniform:
      out << "uniform, min: " << dist.uniform.min << ", max: " << dist.uniform.max
          << ", pnan: " << dist.uniform.pnan;
      break;
    case cutlass::Distribution::Gaussian:
      out << "gaussian, mean: " << dist.gaussian.mean << ", stddev: " << dist.gaussian.stddev
          << ", pnzA: " << dist.gaussian.pnzA << ", pnzB: "
          << dist.gaussian.pnzB << ", pnzC: " << dist.gaussian.pnzC;
      break;
    case cutlass::Distribution::Identity:
      out << "identity";
      break;
    case cutlass::Distribution::Sequential:
      out << "sequential";
      break;
    default:
      out << "unknown";
  }

  out << ", int_scale: " << dist.int_scale;

  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
