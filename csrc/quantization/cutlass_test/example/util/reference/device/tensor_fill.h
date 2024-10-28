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

#if !defined(__CUDACC_RTC__)

// Standard Library includes
#include <utility>
#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <cstdint>

#endif

// CUDA includes
#include <curand_kernel.h>

// Cutlass includes
#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/complex.h"
#include "cutlass/tensor_view.h"
#include "cutlass/blas3.h"
#include "cutlass/numeric_types.h"

#include "cutlass/layout/vector.h"

#include "cutlass/util/reference/device/tensor_foreach.h"
#include "cutlass/util/distribution.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace reference {
namespace device {

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename FloatType>
CUTLASS_DEVICE
FloatType random_normal_float(curandState_t *state) {
  return curand_normal(state);
}

template <>
CUTLASS_DEVICE
double random_normal_float<double>(curandState_t *state) {
  return curand_normal_double(state);
}

template <typename FloatType>
CUTLASS_DEVICE
FloatType random_uniform_float(curandState_t *state) {
  return curand_uniform(state);
}

template <>
CUTLASS_DEVICE
double random_uniform_float<double>(curandState_t *state) {
  return curand_uniform_double(state);
}

template <typename Element>
struct RandomGaussianFunc {

  using FloatType = typename std::conditional<(sizeof(Element) > 4), double, float>::type;
  using IntType = typename std::conditional<(sizeof(Element) > 4), int64_t, int>::type;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    uint64_t seed;
    FloatType mean;
    FloatType stddev;
    int int_scale;
    FloatType float_scale_up;
    FloatType float_scale_down;
    int exclude_zero;           ///< If non-negative, excludes zeros

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      uint64_t seed_ = 0,
      Element mean_ = 0, 
      Element stddev_ = 1,
      int int_scale_ = -1,
      int exclude_zero_ = -1
    ):
      seed(seed_), 
      mean(static_cast<FloatType>(mean_)), 
      stddev(static_cast<FloatType>(stddev_)), 
      int_scale(int_scale_),
      exclude_zero(exclude_zero_) {

      float_scale_up = FloatType(IntType(2) << int_scale); // scale up to clamp low order bits
      float_scale_down = FloatType(1) / FloatType(IntType(2) << int_scale);
    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// RNG state object
  curandState_t rng_state;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  RandomGaussianFunc(Params const &params): params(params) {

    uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(params.seed, gtid, 0, &rng_state);
  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  Element operator()() {

    FloatType rnd = random_normal_float<FloatType>(&rng_state);
    rnd = params.mean + params.stddev * rnd;

    Element result;
    if (params.int_scale >= 0) {
      rnd = FloatType(IntType(std::llround(rnd * params.float_scale_up)));
      result = Element(IntType(rnd * params.float_scale_down));
    }
    else {
      result = Element(rnd);
    }

    if (params.exclude_zero >=0 && result == Element(0.0)) {
      if (rnd > FloatType(0)) {
        rnd += FloatType(1);
      } else {
        rnd -= FloatType(1);
      }
      result = Element(rnd);
    }

    return result;
  }
};


template <typename Real>
struct RandomGaussianFunc<complex<Real>> {

  using Element = complex<Real>;
  using FloatType = typename std::conditional<(sizeof(Real) > 4), double, float>::type;
  using IntType = typename std::conditional<(sizeof(Real) > 4), int64_t, int>::type;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    uint64_t seed;
    FloatType mean;
    FloatType stddev;
    int int_scale;
    FloatType float_scale_up;
    FloatType float_scale_down;
    int exclude_zero;           ///< If non-negative, excludes zeros

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      uint64_t seed_ = 0,
      Real mean_ = 0, 
      Real stddev_ = 1,
      int int_scale_ = -1,
      int exclude_zero_ = -1
    ):
      seed(seed_), 
      mean(static_cast<FloatType>(mean_)), 
      stddev(static_cast<FloatType>(stddev_)), 
      int_scale(int_scale_),
      exclude_zero(exclude_zero_) {

      float_scale_up = FloatType(IntType(1) << int_scale);
      float_scale_up += FloatType(0.5) * float_scale_up;
      float_scale_down = FloatType(1) / FloatType(IntType(1) << int_scale);
    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// RNG state object
  curandState_t rng_state;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  RandomGaussianFunc(Params const &params): params(params) {

    uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(params.seed, gtid, 0, &rng_state);
  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  Element operator()() {

    FloatType rnd_r = random_normal_float<FloatType>(&rng_state);
    FloatType rnd_i = random_normal_float<FloatType>(&rng_state);
    rnd_r = params.mean + params.stddev * rnd_r;
    rnd_i = params.mean + params.stddev * rnd_i;

    Element result;
    if (params.int_scale >= 0) {
      rnd_r = FloatType(IntType(rnd_r * params.float_scale_up));
      rnd_i = FloatType(IntType(rnd_i * params.float_scale_down));

      result = {
        Real(rnd_r * params.float_scale_down),
        Real(rnd_i * params.float_scale_down)
      };
    }
    else {
      result = Element(Real(rnd_r), Real(rnd_i));
    }

    if (params.exclude_zero >= 0 && 
        result.real() == Real(0.0) &&
        result.imag() == Real(0.0)) {

      if (rnd_r > FloatType(0)) {
        rnd_r += FloatType(1);
      } else {
        rnd_r -= FloatType(1);
      }
      result = Element(Real(rnd_r), Real(rnd_i));
    }

    return result;
  }
};

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillRandomGaussianFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  using RandomFunc = RandomGaussianFunc<Element>;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    typename RandomFunc::Params random;

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      TensorView view_ = TensorView(),
      typename RandomFunc::Params random_ = typename RandomFunc::Params()
    ):
      view(view_), random(random_) {

    }
  };

  //
  // Data members
  //

  Params params;
  RandomFunc random;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  TensorFillRandomGaussianFunc(Params const &params): params(params), random(params.random) {

  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {

    params.view.at(coord) = random();
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a Gaussian distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillRandomGaussian(
  TensorView<Element, Layout> view,       ///< destination tensor
  uint64_t seed,                          ///< seed for RNG
  typename RealType<Element>::Type mean = Element(0),   ///< Gaussian distribution's mean
  typename RealType<Element>::Type stddev = Element(1), ///< Gaussian distribution's standard deviation
  int bits = -1,                          ///< If non-negative, specifies number of fractional bits that
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.
  int exclude_zero = -1,                  ///< If non-negative, excludes zeros from tensor init
  cudaStream_t stream = nullptr) {

  using RandomFunc = detail::RandomGaussianFunc<Element>;
  using Func = detail::TensorFillRandomGaussianFunc<Element, Layout>;
  using Params = typename Func::Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, typename RandomFunc::Params(seed, mean, stddev, bits, exclude_zero)),
    /*grid_size*/0, /*block_size*/0,
    stream
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a Gaussian distribution.
template <typename Element>               ///< Element type
void BlockFillRandomGaussian(
  Element *ptr,
  size_t capacity,
  uint64_t seed,                              ///< seed for RNG
  typename RealType<Element>::Type mean,      ///< Gaussian distribution's mean
  typename RealType<Element>::Type stddev,    ///< Gaussian distribution's standard deviation
  int bits = -1,                              ///< If non-negative, specifies number of fractional bits that
                                              ///  are not truncated to zero. Permits reducing precision of
                                              ///  data.
  cudaStream_t stream = nullptr) {

  using RandomFunc = detail::RandomGaussianFunc<Element>;

  typename RandomFunc::Params params(seed, mean, stddev, bits);

  BlockForEach<Element, RandomFunc>(ptr, capacity, params, /*grid_size*/0, /*block_size*/0, stream);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Computes a random uniform distribution
template <typename Element>                ///< Element type 
struct RandomUniformFunc {

  using FloatType = typename std::conditional<
    (sizeof(Element) > 4),
    double,
    float>::type;

  using IntType = typename std::conditional<
    (sizeof(Element) > 4),
    int64_t,
    int>::type;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    uint64_t seed;
    FloatType range;
    FloatType max;
    int int_scale;
    double pnan;
    FloatType float_scale_up;
    FloatType float_scale_down;
    int exclude_zero;           ///< If non-negative, excludes zeros

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      uint64_t seed_ = 0, 
      Element max_ = 1,
      Element min = 0,
      int int_scale_ = -1,
      double pnan_ = 0,
      int exclude_zero_ = -1
    ):
      seed(seed_), 
      range(static_cast<FloatType>(max_) - static_cast<FloatType>(min)), 
      max(static_cast<FloatType>(max_)),
      int_scale(int_scale_),
      pnan(pnan_),
      exclude_zero(exclude_zero_) {
      
      float_scale_up = FloatType(IntType(2) << int_scale); // scale up to clamp low order bits
      float_scale_down = FloatType(1) / FloatType(IntType(2) << int_scale);

      // Handle cases where min = 0 or max = 0 for excluding zeros
      if (exclude_zero >= 0) {
        range = (min == Element(0)) ? range - FloatType(1): range;
        max = (max_ == Element(0)) ? max - FloatType(1): max; 
      }
    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// RNG state object
  curandState_t rng_state;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  RandomUniformFunc(Params const &params): params(params) {

    uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(params.seed, gtid, 0, &rng_state);
  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  Element operator()() {

    // Draw random float in [0.0, 1.0] to determine if element should be NaN.
    if constexpr (std::numeric_limits<Element>::has_quiet_NaN) {
      if (params.pnan > 0 && (curand_uniform(&rng_state) < (params.pnan))) {
        return Element(NAN);
      }
    }

    FloatType rnd = random_uniform_float<FloatType>(&rng_state);
    rnd = params.max - params.range * rnd;

    // Random values are cast to integer after scaling by a power of two to facilitate error
    // testing
    Element result;

    if (params.int_scale >= 0) {
      rnd = FloatType(IntType(std::llround(rnd * params.float_scale_up)));
      result = Element(IntType(rnd * params.float_scale_down));
    }
    else {
      result = Element(rnd);
    }

    if (params.exclude_zero >=0 && result == Element(0.0)) {
      if (rnd > FloatType(0)) {
        rnd = std::min(params.max, rnd + FloatType(1));
      } else {
        rnd = std::max((params.max - params.range), rnd - FloatType(1));
      }
      result = Element(rnd);
    }

    return result;
  }
};

/// Computes a random Gaussian distribution
template <typename Real>
struct RandomUniformFunc<complex<Real>> {

  using Element = complex<Real>;

  using FloatType = typename std::conditional<
    (sizeof(Real) > 4),
    double,
    float>::type;

  using IntType = typename std::conditional<
    (sizeof(Real) > 4),
    int64_t,
    int>::type;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    uint64_t seed;
    FloatType range;
    FloatType min;
    int int_scale;
    double pnan;
    FloatType float_scale_up;
    FloatType float_scale_down;
    int exclude_zero;           ///< If non-negative, excludes zeros

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      uint64_t seed_ = 0, 
      FloatType max = 1,
      FloatType min_ = 0,
      int int_scale_ = -1,
      double pnan_ = 0,
      int exclude_zero_ = -1
    ):
      seed(seed_), 
      range(static_cast<FloatType>(max - min_)), 
      min(static_cast<FloatType>(min_)), 
      int_scale(int_scale_),
      pnan(pnan_),
      exclude_zero(exclude_zero_) {

      float_scale_up = FloatType(IntType(1) << int_scale);
      float_scale_up += FloatType(0.5) * float_scale_up;
      float_scale_down = FloatType(1) / FloatType(IntType(1) << int_scale);

      // Handle cases where min = 0 or max = 0 for excluding zeros
      if (exclude_zero >= 0) {
        min = (min == FloatType(0)) ? min + FloatType(1): min;
        range = (max == FloatType(0)) ? range - FloatType(1): range; 
      }
    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// RNG state object
  curandState_t rng_state;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  RandomUniformFunc(Params const &params): params(params) {

    uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(params.seed, gtid, 0, &rng_state);
  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  Element operator()() {

    // Draw random float in [0.0, 1.0] to determine if element should be NaN.
    if constexpr (std::numeric_limits<Element>::has_quiet_NaN) {
      if (params.pnan > 0 && (curand_uniform(&rng_state) < (params.pnan))) {
        return Element(Real(NAN), Real(NAN));
      }
    }

    FloatType rnd_r = random_uniform_float<FloatType>(&rng_state);
    FloatType rnd_i = random_uniform_float<FloatType>(&rng_state);

    rnd_r = params.min + params.range * rnd_r;
    rnd_i = params.min + params.range * rnd_i;

    // Random values are cast to integer after scaling by a power of two to facilitate error
    // testing
    Element result;

    if (params.int_scale >= 0) {
      rnd_r = FloatType(IntType(rnd_r * params.float_scale_up));
      rnd_i = FloatType(IntType(rnd_i * params.float_scale_up));

      result = {
        Real(rnd_r * params.float_scale_down),
        Real(rnd_i * params.float_scale_down)
      };
    }
    else {
      result = Element(Real(rnd_r), Real(rnd_i));
    }

    if (params.exclude_zero >= 0 && 
        result.real() == Real(0.0) &&
        result.imag() == Real(0.0)) {

      if (rnd_r > FloatType(0)) {
        rnd_r = std::min(params.min + params.range, rnd_r + FloatType(1));
      } else {
        rnd_r = std::max((params.min), rnd_r - FloatType(1));
      }
      result = Element(Real(rnd_r), Real(rnd_i));
    }

    return result;
  }
};

/// Computes a random uniform distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillRandomUniformFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  using RandomFunc = RandomUniformFunc<Element>;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    typename RandomFunc::Params random;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      TensorView view_ = TensorView(),
      typename RandomFunc::Params random_ = RandomFunc::Params()
    ):
      view(view_), random(random_) {

    }
  };

  //
  // Data members
  //

  Params params;
  RandomFunc random;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  TensorFillRandomUniformFunc(Params const &params): params(params), random(params.random) {
  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {

    params.view.at(coord) = random();
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a uniform random distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillRandomUniform(
  TensorView<Element, Layout> view,       ///< destination tensor
  uint64_t seed,                          ///< seed for RNG
  typename RealType<Element>::Type max = Element(1), ///< upper bound of distribution
  typename RealType<Element>::Type min = Element(0), ///< lower bound for distribution
  int bits = -1,                          ///< If non-negative, specifies number of fractional bits that
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.
  double pnan = 0,                        ///< Percentage of NaN elements.
  int exclude_zero = -1,               ///< If non-negative, excludes zeros from tensor init
  cudaStream_t stream = nullptr) {

  using RandomFunc = detail::RandomUniformFunc<Element>;
  using Func = detail::TensorFillRandomUniformFunc<Element, Layout>;
  using Params = typename Func::Params;

  typename RandomFunc::Params random(seed, max, min, bits, pnan, exclude_zero);

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, random),
    /*grid_size*/0, /*block_size*/0,
    stream
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a uniform random distribution.
template <typename Element>
void BlockFillRandomUniform(
  Element *ptr,
  size_t capacity,
  uint64_t seed,                          ///< seed for RNG
  typename RealType<Element>::Type max,   ///< upper bound of distribution
  typename RealType<Element>::Type min,   ///< lower bound for distribution
  int bits = -1,                          ///< If non-negative, specifies number of fractional bits that
                                          ///  are not truncated to zero. Permits reducing precision of
                                          ///  data.
  double pnan = 0,                        ///< Percentage of NaN elements.
  cudaStream_t stream = nullptr) {

  using RandomFunc = detail::RandomUniformFunc<Element>;

  typename RandomFunc::Params params(seed, max, min, bits, pnan);

  BlockForEach<Element, RandomFunc>(ptr, capacity, params, /*grid_size*/0, /*block_size*/0, stream);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Computes a random sparse meta 
template <typename Element>               ///< Element type
struct RandomSparseMetaFunc {

  using FloatType = float;

  using IntType = int32_t;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    uint64_t seed;
    FloatType range;
    int MetaSizeInBits;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      uint64_t seed_ = 0, 
      int MetaSizeInBits_ = 2 
    ):
      seed(seed_), 
      MetaSizeInBits(MetaSizeInBits_) {
      if (MetaSizeInBits_ == 2) {
        range = 6;
      }
      else if (MetaSizeInBits_ == 4) {
        range = 2;
      }
      else {
        throw std::invalid_argument("Invalid MetaSizeInBits");
      }
    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// RNG state object
  curandState_t rng_state;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  RandomSparseMetaFunc(Params const &params): params(params) {

    uint64_t gtid = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(params.seed, gtid, 0, &rng_state);
  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  Element operator()() {
    Element FourToTwoMeta[6] = {0x4, 0x8, 0x9, 0xc, 0xd, 0xe};
    Element TwoToOneMeta[2] = {0x4, 0xe};

    Element *MetaArray =
        (params.MetaSizeInBits == 2) ? FourToTwoMeta : TwoToOneMeta;

    Element result = 0x0;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < cutlass::sizeof_bits<Element>::value / 4; ++i) {
      FloatType rnd = random_uniform_float<FloatType>(&rng_state);
      rnd = params.range * rnd;
      Element meta = MetaArray[(int)rnd];

      result = (Element)(result | ((Element)(meta << (i * 4))));
    }

    return result;
  }
};

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillRandomSparseMetaFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  using RandomFunc = RandomSparseMetaFunc<Element>;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    typename RandomFunc::Params random;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      TensorView view_ = TensorView(),
      typename RandomFunc::Params random_ = RandomFunc::Params()
    ):
      view(view_), random(random_) {

    }
  };

  //
  // Data members
  //

  Params params;
  RandomFunc random;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  TensorFillRandomSparseMetaFunc(Params const &params): params(params), random(params.random) {
  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {

    params.view.at(coord) = random();
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a uniform random distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillRandomSparseMeta(
  TensorView<Element, Layout> view,       ///< destination tensor
  uint64_t seed,                          ///< seed for RNG
  int MetaSizeInBits = 2,                 ///< meta data size
  cudaStream_t stream = nullptr) {

  using RandomFunc = detail::RandomSparseMetaFunc<Element>;
  using Func = detail::TensorFillRandomUniformFunc<Element, Layout>;
  using Params = typename Func::Params;

  typename RandomFunc::Params random(seed, MetaSizeInBits);

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, random),
    /*grid_size*/0, /*block_size*/0,
    stream
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values with a uniform random distribution.
template <typename Element>
void BlockFillRandomSparseMeta(
  Element *ptr,
  size_t capacity,
  uint64_t seed,                          ///< seed for RNG
  int MetaSizeInBits = 2,                 ///< meta data size
  cudaStream_t stream = nullptr) {

  using RandomFunc = detail::RandomSparseMetaFunc<Element>;

  typename RandomFunc::Params params(seed, MetaSizeInBits);

  BlockForEach<Element, RandomFunc>(ptr, capacity, params, /*grid_size*/0, /*block_size*/0, stream);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Functor to fill a tensor with zeros off the diagonal and a uniform value on the diagonal.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillDiagonalFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    Element diag;
    Element other;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    //
    // Methods
    //

    Params(
      TensorView view_ = TensorView(),
      Element diag_ = Element(1),
      Element other_ = Element(0)
    ):
      view(view_), diag(diag_), other(other_) {

    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  TensorFillDiagonalFunc(Params const &params): params(params) {

  }

  /// Updates the tensor
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {

    bool is_diag = true;
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < Layout::kRank; ++i) {
      if (coord[i] != coord[i - 1]) {
        is_diag = false;
        break;
      }
    }

    params.view.at(coord) = (is_diag ? params.diag : params.other);
  }
};

// Overwrites the elements of a tensor with a uniform value depending on fill mode
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillPartialFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    Element element;
    FillMode fill_mode;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params(): fill_mode(FillMode::kNone) { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      TensorView view_,
      Element element_,
      FillMode fill_mode_
    ):
      view(view_), element(element_), fill_mode(fill_mode_) {

    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  CUTLASS_DEVICE
  TensorFillPartialFunc(Params const &params): params(params) {

  }

  /// Overwrites the element if it is within the covered region.
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {

    bool predicate = true;
      
    switch (params.fill_mode) {
    case FillMode::kFull:
      predicate = true;
      break;

    case FillMode::kLower:
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < Layout::kRank; ++i) {
        if (coord[i - 1] < coord[i]) {
          predicate = false;
          break;
        }
      }
      break;

    case FillMode::kUpper:
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < Layout::kRank; ++i) {
        if (coord[i - 1] > coord[i]) {
          predicate = false;
          break;
        }
      }
      break;

    case FillMode::kDiagonal:
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < Layout::kRank; ++i) {
        if (coord[i - 1] != coord[i]) {
          predicate = false;
          break;
        }
      }
      break;

    case FillMode::kNone: // fall-through
    
    default:
      predicate = false;
      break;
    }
    
    if (predicate) {
      params.view.at(coord) = params.element;
    }
  }
};


template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorClearPartialFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  /// 
  static_assert((Layout::kRank == 2), "TensorClearPartial is only supported for matrices");

  /// Parameters structure
  struct Params {
    TensorView view{};
    Element element{};
    FillMode fill_mode{FillMode::kNone};
    int alignment{0};
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  CUTLASS_DEVICE
  TensorClearPartialFunc(Params const &params): params(params) {

  }

  /// Overwrites the element if it is within the covered region.
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {

    bool predicate = true;
      
    switch (params.fill_mode) {

    case FillMode::kLower:
      if ((coord[0] >= coord[1]) || 
          ((coord[1] - coord[0]) >= params.alignment))  {
          predicate = false;
        break;
      }
      break;

    case FillMode::kUpper:
      if ((coord[0] <= coord[1]) ||
          ((coord[0] - coord[1]) >= params.alignment))  {
          predicate = false;
        break;
      }
      break;

    case FillMode::kNone: // fall-through
    
    default:
      predicate = false;
      break;
    }
    
    if (predicate) {
      params.view.at(coord) = params.element;
    }
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor everywhere with a unique value for its diagonal.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillDiagonal(
  TensorView<Element, Layout> view,       ///< destination tensor
  Element diag = Element(1),              ///< value to write in the diagonal
  Element other = Element(0),             ///< value to write off the diagonal
  cudaStream_t stream = nullptr) {

  typedef detail::TensorFillDiagonalFunc<Element, Layout> Func;
  typedef typename Func::Params Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, diag, other),
    /*grid_size*/0, /*block_size*/0,
    stream
  );
}

/// Fills a tensor partially depending on fill mode. Elements not covered by the fillmode are
/// not written.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillPartial(
  TensorView<Element, Layout> view,       ///< destination tensor
  Element element,
  FillMode fill_mode,
  cudaStream_t stream = nullptr) {

  typedef detail::TensorFillPartialFunc<Element, Layout> Func;
  typedef typename Func::Params Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, element, fill_mode),
    stream
  );
}

/// Clears a tensor partially depending on fill mode and alignment. Elements on the wrong-side
/// of fillmode (upto the alignment) are overwritten with the user supplied element (typically zeros)
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorClearPartial(
  TensorView<Element, Layout> view,       ///< destination tensor
  Element element,
  FillMode fill_mode,
  int alignment,
  cudaStream_t stream = nullptr) {

  typedef detail::TensorClearPartialFunc<Element, Layout> Func;
  typedef typename Func::Params Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params{view, element, fill_mode, alignment},
    /*grid_size*/0, /*block_size*/0,
    stream
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with a uniform value
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFill(
  TensorView<Element, Layout> view,         ///< destination tensor
  Element val = Element(0),                 ///< value to uniformly fill it with
  cudaStream_t stream = nullptr) {

  TensorFillDiagonal(view, val, val, stream);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor's diagonal with 1 and 0 everywhere else.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillIdentity(
  TensorView<Element, Layout> view,                 ///< destination tensor
  cudaStream_t stream = nullptr) {

  TensorFillDiagonal(view, Element(1), Element(0), stream);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorUpdateDiagonalFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    Element diag;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      TensorView view_ = TensorView(),
      Element diag_ = Element(1)
    ):
      view(view_), diag(diag_) {

    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  TensorUpdateDiagonalFunc(Params const &params): params(params) {

  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {

    bool is_diag = true;
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < Layout::kRank; ++i) {
      if (coord[i] != coord[i - 1]) {
        is_diag = false;
        break;
      }
    }

    if (is_diag) {
      params.view.at(coord) = params.diag;  
    }
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Writes a uniform value to the diagonal of a tensor without modifying off-diagonal elements.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorUpdateDiagonal(
  TensorView<Element, Layout> view,                 ///< destination tensor
  Element diag = Element(1),
  cudaStream_t stream = nullptr) {

  typedef detail::TensorUpdateDiagonalFunc<Element, Layout> Func;
  typedef typename Func::Params Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, diag),
    /*grid_size*/0, /*block_size*/0,
    stream
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorUpdateOffDiagonalFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    Element other;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      TensorView view_ = TensorView(),
      Element other_ = Element(0)
    ):
      view(view_), other(other_) {

    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  TensorUpdateOffDiagonalFunc(Params const &params): params(params) {

  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {

    bool is_diag = true;
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < Layout::kRank; ++i) {
      if (coord[i] != coord[i - 1]) {
        is_diag = false;
        break;
      }
    }

    if (!is_diag) {
      params.view.at(coord) = params.other;  
    }
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Writes a uniform value to all elements in the tensor without modifying diagonal elements.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorUpdateOffDiagonal(
  TensorView<Element, Layout> view,      ///< destination tensor
  Element other = Element(1),
  cudaStream_t stream = nullptr) {

  typedef detail::TensorUpdateOffDiagonalFunc<Element, Layout> Func;
  typedef typename Func::Params Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, other),
    /*grid_size*/0, /*block_size*/0,
    stream
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorFillLinearFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    Array<Element, Layout::kRank> v;
    Element s;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      TensorView view_,      ///< destination tensor
      Array<Element, Layout::kRank> const & v_,
      Element s_ = Element(0)
    ):
      view(view_), v(v_), s(s_) { 

    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  TensorFillLinearFunc(Params const &params): params(params) {

  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {

    Element sum = params.s;
    
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Layout::kRank; ++i) {
      if constexpr (is_complex<Element>::value) {
        if constexpr (sizeof_bits<Element>::value <= 32) {
          sum = Element(static_cast<complex<float>>(sum) + 
                  static_cast<complex<float>>(params.v[i]) * static_cast<complex<float>>(coord[i]));
        }
      }
      else if constexpr (sizeof_bits<Element>::value <= 32) {
        if constexpr (std::numeric_limits<Element>::is_integer) {
          sum = Element(static_cast<int32_t>(sum) + 
                  static_cast<int32_t>(params.v[i]) * static_cast<int32_t>(coord[i]));
        }
        else {
          sum = Element(static_cast<float>(sum) + 
                  static_cast<float>(params.v[i]) * static_cast<float>(coord[i]));
        }
      }
      else {
        sum += params.v[i] * coord[i];
      }
    }

    params.view.at(coord) = sum;
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills tensor with a linear combination of its coordinate and another vector
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillLinear(
  TensorView<Element, Layout> view,      ///< destination tensor
  Array<Element, Layout::kRank> const & v,
  Element s = Element(0),
  cudaStream_t stream = nullptr) {

  using Func = detail::TensorFillLinearFunc<Element, Layout>;
  using Params = typename Func::Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, v, s),
    /*grid_size*/0, /*block_size*/0,
    stream
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a tensor with random values from a distribution.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorFillRandom(
  TensorView<Element, Layout> view,       ///< destination tensor
  uint64_t seed,
  Distribution dist,
  cudaStream_t stream = nullptr,
  int exclude_zero = -1                   ///< If non-negative, excludes 0.
                                          ///  Note that setting this flag will result in more 1's,
                                          ///  as we use a simple mechanism to replace 0's by adding/subtracting 1's.
  ) {

  using Real = typename RealType<Element>::Type;

  if (dist.kind == Distribution::Gaussian) {
    TensorFillRandomGaussian<Element, Layout>(
      view,
      seed,
      static_cast<Real>(dist.gaussian.mean),
      static_cast<Real>(dist.gaussian.stddev),
      dist.int_scale,
      exclude_zero,
      stream);
  } else if (dist.kind == Distribution::Uniform) {
    TensorFillRandomUniform<Element, Layout>(
      view,
      seed,
      static_cast<Real>(dist.uniform.max),
      static_cast<Real>(dist.uniform.min),
      dist.int_scale,
      dist.uniform.pnan,
      exclude_zero,
      stream);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
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

  using Layout = layout::PackedVectorLayout;
  Layout::TensorCoord size(static_cast<Layout::Index>(capacity)); // -Wconversion
  Layout layout = Layout::packed(size);
  TensorView<Element, Layout> view(ptr, layout, size);

  Array<Element, Layout::kRank> c{};
  c[0] = v;

  TensorFillLinear(view, c, s);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Fills a block of data with sequential elements
template <
  typename Element
>
void BlockFillRandom(
  Element *ptr,
  size_t capacity,
  uint64_t seed,
  Distribution dist,
  cudaStream_t stream = nullptr) {

  using Real = typename RealType<Element>::Type;

  if (dist.kind == Distribution::Gaussian) {
    BlockFillRandomGaussian<Element>(
      ptr,
      capacity,
      seed,
      static_cast<Real>(dist.gaussian.mean),
      static_cast<Real>(dist.gaussian.stddev),
      dist.int_scale,
      stream);
  }
  else if (dist.kind == Distribution::Uniform) {
    BlockFillRandomUniform<Element>(
      ptr,
      capacity,
      seed,
      static_cast<Real>(dist.uniform.max),
      static_cast<Real>(dist.uniform.min),
      dist.int_scale,
      dist.uniform.pnan,
      stream);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorCopyDiagonalInFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    Element const *ptr;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      TensorView view_,      ///< destination tensor
      Element const *ptr_
    ):
      view(view_), ptr(ptr_) { 

    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  TensorCopyDiagonalInFunc(Params const &params): params(params) {

  }

  /// Only update the diagonal element
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {
    bool is_diagonal = true;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < Layout::kRank; ++i) {
      if (coord[i] != coord[0]) {
        is_diagonal = false;
      }
    }
    if (is_diagonal) {
      params.view.at(coord) = params.ptr[coord[0]];
    }
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Copies a diagonal in from host memory without modifying off-diagonal elements.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorCopyDiagonalIn(
  TensorView<Element, Layout> view,   ///< destination tensor
  Element const *ptr,                        ///< dense buffer of elements
  cudaStream_t stream = nullptr) {

  using Func = detail::TensorCopyDiagonalInFunc<Element, Layout>;
  using Params = typename Func::Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, ptr),
    /*grid_size*/0, /*block_size*/0,
    stream
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////


namespace detail {

/// Computes a random Gaussian distribution
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
struct TensorCopyDiagonalOutFunc {

  /// View type
  using TensorView = TensorView<Element, Layout>;

  /// Scalar type
  typedef typename TensorView::Element T;

  /// Coordinate in tensor's index space
  typedef typename TensorView::TensorCoord TensorCoord;

  /// Parameters structure
  struct Params {

    //
    // Data members
    //

    TensorView view;
    Element *ptr;

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() { }

    //
    // Methods
    //

    /// Construction of Gaussian RNG functor.
    Params(
      TensorView view_,      ///< destination tensor
      Element *ptr_
    ):
      view(view_), ptr(ptr_) { 

    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  //
  // Methods
  //

  /// Device-side initialization of RNG
  CUTLASS_DEVICE
  TensorCopyDiagonalOutFunc(Params const &params): params(params) {

  }

  /// Compute random value and update RNG state
  CUTLASS_DEVICE
  void operator()(TensorCoord const &coord) {
    bool is_diagonal = true;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 1; i < Layout::kRank; ++i) {
      if (coord[i] != coord[0]) {
        is_diagonal = false;
      }
    }
    if (is_diagonal) {
      params.ptr[coord[0]] = params.view.at(coord);  
    }
  }
};

} // namespace detail

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Copies the diagonal of a tensor into a dense buffer in host memory.
template <
  typename Element,               ///< Element type
  typename Layout>                ///< Layout function
void TensorCopyDiagonalOut(
  Element *ptr,                               ///< dense buffer of elements
  TensorView<Element, Layout> view,      ///< source tensor
  cudaStream_t stream = nullptr) {

  using Func = detail::TensorCopyDiagonalOutFunc<Element, Layout>;
  using Params = typename Func::Params;

  TensorForEach<Func, Layout::kRank, Params>(
    view.extent(),
    Params(view, ptr),
    /*grid_size*/0, /*block_size*/0,
    stream
  );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace reference
} // namespace cutlass
