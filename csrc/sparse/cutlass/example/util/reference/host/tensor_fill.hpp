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
//
// Uniform and procedural tensor fills
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with a scalar element
template <typename Tensor>
void TensorFill(Tensor dst, typename Tensor::value_type element) {

  for (int64_t idx = 0; idx < cute::size(dst); ++idx) {
    dst(idx) = element;
  }
}

/// Fills a tensor with the contents of its layout
template <typename Tensor>
void TensorFillSequential(Tensor dst) {

  auto layout = dst.layout();

  for (int64_t idx = 0; idx < cute::size(dst); ++idx) {
    dst(idx) = layout(idx);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Random uniform values
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element>
struct RandomUniformFunc {

  using Real = typename RealType<Element>::Type;
  
  uint64_t seed;
  double range;
  double min;
  int int_scale;

  //
  // Methods
  //

  RandomUniformFunc(
    uint64_t seed_ = 0, 
    double max = 1,
    double min_ = 0,
    int int_scale_ = -1
  ):
    seed(seed_), range(max - min_), min(min_), int_scale(int_scale_) {
      std::srand((unsigned)seed);
    }


  /// Compute random value and update RNG state
  Element operator()() const {

    double rnd = double(std::rand()) / double(RAND_MAX);

    rnd = min + range * rnd;

    // Random values are cast to integer after scaling by a power of two to facilitate error
    // testing
    Element result;
    
    if (int_scale >= 0) {
      rnd = double(int64_t(rnd * double(1 << int_scale))) / double(1 << int_scale);
      result = static_cast<Element>(Real(rnd));
    }
    else {
      result = static_cast<Element>(Real(rnd));
    }

    return result;
  }
};

/// Partial specialization for initializing a complex value.
template <typename Element>
struct RandomUniformFunc<complex<Element> > {

  using Real = typename RealType<Element>::Type;
  
  uint64_t seed;
  double range;
  double min;
  int int_scale;

  //
  // Methods
  //

  RandomUniformFunc(
    uint64_t seed_ = 0, 
    double max = 1,
    double min_ = 0,
    int int_scale_ = -1
  ):
    seed(seed_), range(max - min_), min(min_), int_scale(int_scale_) {
      std::srand((unsigned)seed);
    }


  /// Compute random value and update RNG state
  complex<Element> operator()() const {

    Element reals[2];

    for (int i = 0; i < 2; ++i) {
      double rnd = double(std::rand()) / double(RAND_MAX);

      rnd = min + range * rnd;

      // Random values are cast to integer after scaling by a power of two to facilitate error
      // testing
      
      if (int_scale >= 0) {
        rnd = double(int(rnd * double(1 << int_scale)));
        reals[i] = from_real<Element>(Real(rnd / double(1 << int_scale)));
      }
      else {
        reals[i] = from_real<Element>(Real(rnd));
      }
    }

    return complex<Element>(reals[0], reals[1]);
  }
};

/// Partial specialization for initializing a Quaternion value.
template <typename Element>
struct RandomUniformFunc<Quaternion<Element> > {

  using Real = typename RealType<Element>::Type;

  uint64_t seed;
  double range;
  double min;
  int int_scale;

  //
  // Methods
  //

  RandomUniformFunc(
    uint64_t seed_ = 0,
    double max = 1,
    double min_ = 0,
    int int_scale_ = -1
  ):
    seed(seed_), range(max - min_), min(min_), int_scale(int_scale_) {
      std::srand((unsigned)seed);
    }


  /// Compute random value and update RNG state
  Quaternion<Element> operator()() const {

    Element reals[4];

    for (int i = 0; i < 4; ++i) {
      double rnd = double(std::rand()) / double(RAND_MAX);

      rnd = min + range * rnd;

      // Random values are cast to integer after scaling by a power of two to facilitate error
      // testing

      if (int_scale >= 0) {
        rnd = double(int(rnd * double(1 << int_scale)));
        reals[i] = from_real<Element>(Real(rnd / double(1 << int_scale)));
      }
      else {
        reals[i] = from_real<Element>(Real(rnd));
      }
    }

    return make_Quaternion(reals[0], reals[1], reals[2], reals[3]);
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a uniform random distribution.
template <typename Tensor>                ///< Tensor object
void TensorFillRandomUniform(
  Tensor dst,                             ///< destination tensor
  uint64_t seed,                          ///< seed for RNG
  double max = 1,                         ///< upper bound of distribution
  double min = 0,                         ///< lower bound for distribution
  int bits = -1) {                        ///< If non-negative, specifies number of fractional bits that 
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.   

  detail::RandomUniformFunc<typename Tensor::value_type> random_func(seed, max, min, bits);

  for (int64_t idx = 0; idx < cute::size(dst); ++idx) {
    dst(idx) = random_func();
  }
}

/// Fills a block with random values with a uniform random distribution.
template <
  typename Element                        ///< Element type
>
void BlockFillRandomUniform(
  Element *ptr,
  size_t capacity,
  uint64_t seed,                          ///< seed for RNG
  double max = 1,                         ///< upper bound of distribution
  double min = 0,                         ///< lower bound for distribution
  int bits = -1) {                        ///< If non-negative, specifies number of fractional bits that 
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.                 
  detail::RandomUniformFunc<Element> random_func(seed, max, min, bits);

  for (size_t i = 0; i < capacity; ++i) {
    ptr[i] = random_func();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Random Gaussian
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element>
struct RandomGaussianFunc {

  uint64_t seed;
  double mean;
  double stddev;
  int int_scale;
  double pi;

  //
  // Methods
  //
  RandomGaussianFunc(
    uint64_t seed_ = 0, 
    double mean_ = 0, 
    double stddev_ = 1,
    int int_scale_ = -1
  ):
    seed(seed_), mean(mean_), stddev(stddev_), int_scale(int_scale_), pi(std::acos(-1)) {
      std::srand((unsigned)seed);
  }

  /// Compute random value and update RNG state
  Element operator()() const {

    // Box-Muller transform to generate random numbers with Normal distribution
    double u1 = double(std::rand()) / double(RAND_MAX);
    double u2 = double(std::rand()) / double(RAND_MAX);

    // Compute Gaussian random value
    double rnd = std::sqrt(-2 * std::log(u1)) * std::cos(2 * pi * u2);
    rnd = mean + stddev * rnd;

    // Scale and convert final result
    Element result;

    if (int_scale >= 0) {
      rnd = double(int64_t(rnd * double(1 << int_scale))) / double(1 << int_scale);
      result = static_cast<Element>(rnd);
    }
    else {
      result = static_cast<Element>(rnd);
    }

    return result;
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a Gaussian distribution.
template <
  typename Tensor
>
void TensorFillRandomGaussian(
  Tensor  dst,                            ///< destination tensor
  uint64_t seed,                          ///< seed for RNG
  double mean = 0,                        ///< Gaussian distribution's mean
  double stddev = 1,                      ///< Gaussian distribution's standard deviation
  int bits = -1) {                        ///< If non-negative, specifies number of fractional bits that 
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.
  
  detail::RandomGaussianFunc<typename Tensor::value_type> random_func(seed, mean, stddev, bits);

  for (int64_t idx = 0; idx < cute::size(dst); ++idx) {
    dst(idx) = random_func();
  }
}

/// Fills a block with random values with a Gaussian distribution.
template <
  typename Element                        ///< Element type
>
void BlockFillRandomGaussian(
  Element *ptr,                           ///< destination buffer
  size_t capacity,                        ///< number of elements
  uint64_t seed,                          ///< seed for RNG
  double mean = 0,                        ///< Gaussian distribution's mean
  double stddev = 1,                      ///< Gaussian distribution's standard deviation
  int bits = -1) {                        ///< If non-negative, specifies number of fractional bits that 
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.
  
  detail::RandomGaussianFunc<Element> random_func(seed, mean, stddev, bits);

  for (size_t i = 0; i < capacity; ++i) {
    ptr[i] = random_func();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a block of data with sequential elements
template <
  typename Element
>
void BlockFillSequential(
  Element *ptr,
  int64_t capacity,
  Element v = Element(1),
  Element s = Element(0)) {
  int i = 0;

  while (i < capacity) {

    ptr[i] = Element(s + v);
    ++i;
  }
}

/// Fills a block of data with sequential elements
template <
  typename Element
>
void BlockFillSequentialModN(
  Element *ptr,
  int64_t capacity,
  int64_t mod,
  int64_t v = int64_t(1),
  int64_t s = int64_t(0)) {
  int i = 0;

  while (i < capacity) {

    ptr[i] = static_cast<Element>(int32_t(int64_t(s + v) % mod));
    ++i;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass

///////////////////////////////////////////////////////////////////////////////////////////////////
