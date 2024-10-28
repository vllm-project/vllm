/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Reference implementation for CONV in host-side code.
*/
#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/complex.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/activation.h"

#include "cute/tensor.hpp"

#include <cuda_runtime.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::reference::host {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<class EngineAct, class LayoutAct>
bool
is_activation_in_bounds(
    cute::Tensor<EngineAct, LayoutAct> const& activation,
    int32_t n_, int32_t d_, int32_t h_, int32_t w_, int32_t c_) {
  return ((n_ >= 0 && n_ < size<4>(activation)) &&
          (d_ >= 0 && d_ < size<3>(activation)) &&
          (h_ >= 0 && h_ < size<2>(activation)) &&
          (w_ >= 0 && w_ < size<1>(activation)) &&
          (c_ >= 0 && c_ < size<0>(activation)));
}

template<class EngineAct, class LayoutAct>
bool
is_activation_in_bounds(
    cute::Tensor<EngineAct, LayoutAct> const& activation,
    int32_t n_, int32_t h_, int32_t w_, int32_t c_) {
  return ((n_ >= 0 && n_ < size<3>(activation)) &&
          (h_ >= 0 && h_ < size<2>(activation)) &&
          (w_ >= 0 && w_ < size<1>(activation)) &&
          (c_ >= 0 && c_ < size<0>(activation)));
}

template<class EngineAct, class LayoutAct>
bool
is_activation_in_bounds(
    cute::Tensor<EngineAct, LayoutAct> const& activation,
    int32_t n_, int32_t w_, int32_t c_) {
  return ((n_ >= 0 && n_ < size<2>(activation)) &&
          (w_ >= 0 && w_ < size<1>(activation)) &&
          (c_ >= 0 && c_ < size<0>(activation)));
}

} // namespace detail

template<
  class ElementAcc_,
  class ElementScalar_,
  class ElementCompute_,
  class ElementC_,
  class ElementOut_,
  class TensorAlpha_,
  class TensorBeta_,
  class TensorBias_,
  class ActivationFunctor_ = cutlass::epilogue::thread::Identity<ElementCompute_>
>
struct ConvEpilogueFusionParams {
  using ElementAcc = ElementAcc_;
  using ElementScalar = ElementScalar_;
  using ElementCompute = ElementCompute_;
  using ElementC = ElementC_;
  using ElementOut = ElementOut_;
  using TensorAlpha = TensorAlpha_;
  using TensorBeta = TensorBeta_;
  using TensorBias = TensorBias_;
  using ActivationFunctor = ActivationFunctor_;
  ElementScalar alpha = ElementScalar(1);
  ElementScalar beta = ElementScalar(0);

  TensorAlpha tensor_alpha{};
  TensorBeta tensor_beta{};
  TensorBias tensor_bias{};
};

template<
  cutlass::conv::Operator ConvOp,
  int NumSpatialDims,
  class TensorA,
  class TensorB,
  class TensorC,
  class TensorD,
  class ShapePadding,
  class StrideTraversal,
  class ShapeDilation,
  class EpilogueFusionParams
>
struct ConvReferenceImpl {
  // Hard code accumlulator type to float to avoid data lost in accumulating add.
  using ElementAcc = cutlass::platform::conditional_t<cutlass::platform::is_same_v<typename EpilogueFusionParams::ElementAcc, double>, double, float>;
  using ElementC = typename EpilogueFusionParams::ElementC;
  using ElementOut = typename EpilogueFusionParams::ElementOut;
  using ElementScalar = typename EpilogueFusionParams::ElementScalar;
  using ElementCompute = typename EpilogueFusionParams::ElementCompute;
  using ElementBias = typename EpilogueFusionParams::TensorBias::value_type;
  using ActivationFunctor = typename EpilogueFusionParams::ActivationFunctor;

  // Input related converter
  NumericConverter<ElementCompute, ElementAcc> acc_converter;
  NumericConverter<ElementCompute, ElementC> residual_converter;
  NumericConverter<ElementCompute, ElementBias> bias_converter;
  // Scale related converter
  NumericConverter<ElementCompute, ElementScalar> scale_converter;
  // Output related converter
  NumericConverter<ElementOut, ElementCompute> output_converter;

  EpilogueFusionParams& epi_fusion_params_;
  TensorA const& tensor_a_;
  TensorB const& tensor_b_;
  TensorC const& tensor_c_;
  TensorD& tensor_d_;

  ShapePadding const& padding_;
  StrideTraversal const& tstride_;
  ShapeDilation const& dilation_;

  // Epilogue activation operation
  ActivationFunctor epi_activation;

  ConvReferenceImpl(
    TensorA const& tensor_a,
    TensorB const& tensor_b,
    TensorC const& tensor_c,
    TensorD& tensor_d,
    ShapePadding const& padding,
    StrideTraversal const& tstride,
    ShapeDilation const& dilation,
    EpilogueFusionParams& epi_fusion_params)
  : tensor_a_(tensor_a),
    tensor_b_(tensor_b),
    tensor_c_(tensor_c),
    tensor_d_(tensor_d),
    padding_(padding),
    tstride_(tstride),
    dilation_(dilation),
    epi_fusion_params_(epi_fusion_params)
  {
    static_assert(rank(ShapePadding{}) == rank(ShapeDilation{}));
    static_assert(rank(ShapePadding{}) == rank(StrideTraversal{}));
  }

  void compute_reference() {
    if constexpr (ConvOp == cutlass::conv::Operator::kFprop) {
      fprop_reference(cute::Int<NumSpatialDims>{});
    }
    else if constexpr (ConvOp == cutlass::conv::Operator::kDgrad) {
      dgrad_reference(cute::Int<NumSpatialDims>{});
    }
    else {
      wgrad_reference(cute::Int<NumSpatialDims>{});
    }
  }

private:
  // Specialization for 1D fprop kernel
  void fprop_reference(cute::Int<1> spatial_dims) {
    int32_t N = size<2>(tensor_d_);
    int32_t Q = size<1>(tensor_d_);
    int32_t K = size<0>(tensor_d_);
    int32_t S = size<1>(tensor_b_);
    int32_t C = size<0>(tensor_b_);

#if defined(_OPENMP)
  #pragma omp parallel for collapse(2)
#endif
    for (int32_t n = 0; n < N; ++n) {
      for (int32_t q = 0; q < Q; ++q) {
        for (int32_t k = 0; k < K; ++k) {
          auto accumulator = ElementAcc(0);
          for (int32_t s = 0; s < S; ++s) {
            for (int32_t c = 0; c < C; ++c) {
              int32_t w =  q * cute::get<0>(tstride_) - cute::get<0>(padding_) + s * cute::get<0>(dilation_);
              if (detail::is_activation_in_bounds(tensor_a_, n, w, c)) {
                auto a = tensor_a_(c, w, n);
                auto b = tensor_b_(c, s, k);
                accumulator += ElementAcc(a * b);
              }
            }
          }
          ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data()) ?
            epi_fusion_params_.tensor_alpha[k] : epi_fusion_params_.alpha;
          ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data()) ?
            epi_fusion_params_.tensor_beta[k] : epi_fusion_params_.beta;
          ElementCompute output = scale_converter(alpha) * acc_converter(accumulator) +
                                  scale_converter(beta) * residual_converter(tensor_c_(k, q, n));
          if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
            output += bias_converter(epi_fusion_params_.tensor_bias[k]);
          }
          output = epi_activation(output);
          tensor_d_(k, q, n) = output_converter(output);
        }
      }
    }

  }

  // Specialization for 2D fprop kernel
  void fprop_reference(cute::Int<2> spatial_dims) {
    int32_t N = size<3>(tensor_d_);
    int32_t P = size<2>(tensor_d_);
    int32_t Q = size<1>(tensor_d_);
    int32_t K = size<0>(tensor_d_);
    int32_t R = size<2>(tensor_b_);
    int32_t S = size<1>(tensor_b_);
    int32_t C = size<0>(tensor_b_);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int32_t n = 0; n < N; ++n) {
      for (int32_t p = 0; p < P; ++p) {
        for (int32_t q = 0; q < Q; ++q) {
          for (int32_t k = 0; k < K; ++k) {
            auto accumulator = ElementAcc(0);
            for (int32_t r = 0; r < R; ++r) {
              for (int32_t s = 0; s < S; ++s) {
                for (int32_t c = 0; c < C; ++c) {
                  int32_t w =  q * cute::get<0>(tstride_) - cute::get<0>(padding_) + s * cute::get<0>(dilation_);
                  int32_t h =  p * cute::get<1>(tstride_) - cute::get<1>(padding_) + r * cute::get<1>(dilation_);
                  if (detail::is_activation_in_bounds(tensor_a_, n, h, w, c)) {
                    auto a = tensor_a_(c, w, h, n);
                    auto b = tensor_b_(c, s, r, k);
                    accumulator += ElementAcc(a * b);
                  }
                }
              }
            }
            ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data()) ?
              epi_fusion_params_.tensor_alpha[k] : epi_fusion_params_.alpha;
            ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data()) ?
              epi_fusion_params_.tensor_beta[k] : epi_fusion_params_.beta;
            ElementCompute output = scale_converter(alpha) * acc_converter(accumulator) +
                                    scale_converter(beta) * residual_converter(tensor_c_(k, q, p, n));
            if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
              output += bias_converter(epi_fusion_params_.tensor_bias[k]);
            }
            output = epi_activation(output);
            tensor_d_(k, q, p, n) = output_converter(output);
          }
        }
      }
    }

  }

  // Specialization for 3D fprop kernel
  void fprop_reference(cute::Int<3> spatial_dims) {
    int32_t N = size<4>(tensor_d_);
    int32_t Z = size<3>(tensor_d_);
    int32_t P = size<2>(tensor_d_);
    int32_t Q = size<1>(tensor_d_);
    int32_t K = size<0>(tensor_d_);
    int32_t T = size<3>(tensor_b_);
    int32_t R = size<2>(tensor_b_);
    int32_t S = size<1>(tensor_b_);
    int32_t C = size<0>(tensor_b_);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int32_t n = 0; n < N; ++n) {
      for (int32_t z = 0; z < Z; ++z) {
        for (int32_t p = 0; p < P; ++p) {
          for (int32_t q = 0; q < Q; ++q) {
            for (int32_t k = 0; k < K; ++k) {
              auto accumulator = ElementAcc(0);
              for (int32_t t = 0; t < T; ++t) {
                for (int32_t r = 0; r < R; ++r) {
                  for (int32_t s = 0; s < S; ++s) {
                    for (int32_t c = 0; c < C; ++c) {
                      int32_t w =  q * cute::get<0>(tstride_) - cute::get<0>(padding_) + s * cute::get<0>(dilation_);
                      int32_t h =  p * cute::get<1>(tstride_) - cute::get<1>(padding_) + r * cute::get<1>(dilation_);
                      int32_t d =  z * cute::get<2>(tstride_) - cute::get<2>(padding_) + t * cute::get<2>(dilation_);
                      if (detail::is_activation_in_bounds(tensor_a_, n, d, h, w, c)) {
                        auto a = tensor_a_(c, w, h, d, n);
                        auto b = tensor_b_(c, s, r, t, k);
                        accumulator += ElementAcc(a * b);
                      }
                    }
                  }
                }
              }
              ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data()) ?
                epi_fusion_params_.tensor_alpha[k] : epi_fusion_params_.alpha;
              ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data()) ?
                epi_fusion_params_.tensor_beta[k] : epi_fusion_params_.beta;
              ElementCompute output = scale_converter(alpha) * acc_converter(accumulator) +
                                      scale_converter(beta) * residual_converter(tensor_c_(k, q, p, z, n));
              if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
                output += bias_converter(epi_fusion_params_.tensor_bias[k]);
              }
              output = epi_activation(output);
              tensor_d_(k, q, p, z, n) = output_converter(output);
            }
          }
        }
      }
    }

  }

  // Specialization for 1D dgrad kernel
  void dgrad_reference(cute::Int<1> spatial_dims) {
    int32_t N = size<2>(tensor_d_);
    int32_t W = size<1>(tensor_d_);
    int32_t C = size<0>(tensor_d_);
    int32_t K = size<2>(tensor_b_);
    int32_t S = size<1>(tensor_b_);

#if defined(_OPENMP)
   #pragma omp parallel for collapse(2)
#endif
    for (int32_t n = 0; n < N; ++n) {
      for (int32_t w = 0; w < W; ++w) {
        for (int32_t c = 0; c < C; ++c) {
          auto accumulator = ElementAcc(0);
          for (int32_t k = 0; k < K; ++k) {
            for (int32_t s = 0; s < S; ++s) {
              int32_t q = w + cute::get<0>(padding_) - s * cute::get<0>(dilation_);

              if (q % cute::get<0>(tstride_) == 0) {
                q /= cute::get<0>(tstride_);
              } else {
                continue;
              }

              if (detail::is_activation_in_bounds(tensor_a_, n, q, k)) {
                accumulator += ElementAcc(tensor_a_(k, q, n) * tensor_b_(c, s, k));
              }
            }
          }
          ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data())
            ? epi_fusion_params_.tensor_alpha[c] : epi_fusion_params_.alpha;
          ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data())
            ? epi_fusion_params_.tensor_beta[c] : epi_fusion_params_.beta;
          ElementCompute output = scale_converter(alpha) * acc_converter(accumulator) +
                                  scale_converter(beta) * residual_converter(tensor_c_(c, w, n));
          if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
            output += bias_converter(epi_fusion_params_.tensor_bias[c]);
          }
          output = epi_activation(output);
          tensor_d_(c, w, n) = output_converter(output);
        }
      }
    }

  }

  // Specialization for 2D dgrad kernel
  void dgrad_reference(cute::Int<2> spatial_dims) {
    int32_t N = size<3>(tensor_d_);
    int32_t H = size<2>(tensor_d_);
    int32_t W = size<1>(tensor_d_);
    int32_t C = size<0>(tensor_d_);
    int32_t K = size<3>(tensor_b_);
    int32_t R = size<2>(tensor_b_);
    int32_t S = size<1>(tensor_b_);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int32_t n = 0; n < N; ++n) {
      for (int32_t h = 0; h < H; ++h) {
        for (int32_t w = 0; w < W; ++w) {
          for (int32_t c = 0; c < C; ++c) {
            auto accumulator = ElementAcc(0);
            for (int32_t k = 0; k < K; ++k) {
              for (int32_t r = 0; r < R; ++r) {
                for (int32_t s = 0; s < S; ++s) {
                  int32_t q = w + cute::get<0>(padding_) - s * cute::get<0>(dilation_);
                  int32_t p = h + cute::get<1>(padding_) - r * cute::get<1>(dilation_);

                  if (q % cute::get<0>(tstride_) == 0) {
                    q /= cute::get<0>(tstride_);
                  } else {
                    continue;
                  }

                  if (p % cute::get<1>(tstride_) == 0) {
                    p /= cute::get<1>(tstride_);
                  } else {
                    continue;
                  }

                  if (detail::is_activation_in_bounds(tensor_a_, n, p, q, k)) {
                    accumulator += ElementAcc(tensor_a_(k, q, p, n) * tensor_b_(c, s, r, k));
                  }
                }
              }
            }
            ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data())
              ? epi_fusion_params_.tensor_alpha[c] : epi_fusion_params_.alpha;
            ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data())
              ? epi_fusion_params_.tensor_beta[c] : epi_fusion_params_.beta;
            ElementCompute output = scale_converter(alpha) * acc_converter(accumulator) +
                                    scale_converter(beta) * residual_converter(tensor_c_(c, w, h, n));
            if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
              output += bias_converter(epi_fusion_params_.tensor_bias[c]);
            }
            output = epi_activation(output);

            tensor_d_(c, w, h, n) = output_converter(output);
          }
        }
      }
    }

  }

  // Specialization for 3D dgrad kernel
  void dgrad_reference(cute::Int<3> spatial_dims) {
    int32_t N = size<4>(tensor_d_);
    int32_t D = size<3>(tensor_d_);
    int32_t H = size<2>(tensor_d_);
    int32_t W = size<1>(tensor_d_);
    int32_t C = size<0>(tensor_d_);
    int32_t K = size<4>(tensor_b_);
    int32_t T = size<3>(tensor_b_);
    int32_t R = size<2>(tensor_b_);
    int32_t S = size<1>(tensor_b_);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int32_t n = 0; n < N; ++n) {
      for (int32_t d = 0; d < D; ++d) {
        for (int32_t h = 0; h < H; ++h) {
          for (int32_t w = 0; w < W; ++w) {
            for (int32_t c = 0; c < C; ++c) {
              auto accumulator = ElementAcc(0);
              for (int32_t k = 0; k < K; ++k) {
                for (int32_t t = 0; t < T; ++t) {
                  for (int32_t r = 0; r < R; ++r) {
                    for (int32_t s = 0; s < S; ++s) {
                      int32_t q = w + cute::get<0>(padding_) - s * cute::get<0>(dilation_);
                      int32_t p = h + cute::get<1>(padding_) - r * cute::get<1>(dilation_);
                      int32_t z = d + cute::get<2>(padding_) - t * cute::get<2>(dilation_);

                      if (q % cute::get<0>(tstride_) == 0) {
                        q /= cute::get<0>(tstride_);
                      } else {
                        continue;
                      }

                      if (p % cute::get<1>(tstride_) == 0) {
                        p /= cute::get<1>(tstride_);
                      } else {
                        continue;
                      }

                      if (z % cute::get<2>(tstride_) == 0) {
                        z /= cute::get<2>(tstride_);
                      } else {
                        continue;
                      }

                      if (detail::is_activation_in_bounds(tensor_a_, n, z, p, q, k)) {
                        accumulator += ElementAcc(tensor_a_(k, q, p, z, n) * tensor_b_(c, s, r, t, k));
                      }
                    }
                  }
                }
              }
              ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data())
                ? epi_fusion_params_.tensor_alpha[c] : epi_fusion_params_.alpha;
              ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data())
                ? epi_fusion_params_.tensor_beta[c] : epi_fusion_params_.beta;
              ElementCompute output = scale_converter(alpha) * acc_converter(accumulator) +
                                      scale_converter(beta) * residual_converter(tensor_c_(c, w, h, d, n));
              if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
                output += bias_converter(epi_fusion_params_.tensor_bias[c]);
              }
              output = epi_activation(output);
              tensor_d_(c, w, h, d, n) = output_converter(output);
            }
          }
        }
      }
    }

  }

  // Specialization for 1D wgrad kernel
  void wgrad_reference(cute::Int<1> spatial_dims) {
    int32_t N =
        size<2>(tensor_a_);
    int32_t Q =
        size<1>(tensor_a_);
    int32_t K =
        size<0>(tensor_a_);
    int32_t S = size<1>(tensor_d_);
    int32_t C = size<0>(tensor_d_);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(2)
#endif
    for (int32_t k = 0; k < K; ++k) {
      ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data()) ?
        epi_fusion_params_.tensor_alpha[k] : epi_fusion_params_.alpha;
      ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data()) ?
        epi_fusion_params_.tensor_beta[k] : epi_fusion_params_.beta;
      for (int32_t s = 0; s < S; ++s) {
        for (int32_t c = 0; c < C; ++c) {
          auto accumulator = ElementAcc(0);
          for (int32_t n = 0; n < N; ++n) {
            for (int32_t q = 0; q < Q; ++q) {
              int32_t w =  q * cute::get<0>(tstride_) - cute::get<0>(padding_) + s * cute::get<0>(dilation_);
              bool is_in_bounds =
                  detail::is_activation_in_bounds(tensor_b_, n, w, c);
              if (is_in_bounds) {
                auto act =
                    tensor_b_(c, w, n);
                auto xformed_act =
                    tensor_a_(k, q, n);
                accumulator += ElementAcc(act * xformed_act);
              }
            }
          }
          ElementCompute output = scale_converter(alpha) * acc_converter(accumulator) +
                                  scale_converter(beta) * residual_converter(tensor_c_(c, s, k));
          if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
            output += bias_converter(epi_fusion_params_.tensor_bias[k]);
          }
          output = epi_activation(output);
          tensor_d_(c, s, k) = output_converter(output);
        }
      }
    }
  }

  // Specialization for 2D wgrad kernel
  void wgrad_reference(cute::Int<2> spatial_dims) {
    int32_t N =
        size<3>(tensor_a_);
    int32_t P =
        size<2>(tensor_a_);
    int32_t Q =
        size<1>(tensor_a_);
    int32_t K =
        size<0>(tensor_a_);
    int32_t R = size<2>(tensor_d_);
    int32_t S = size<1>(tensor_d_);
    int32_t C = size<0>(tensor_d_);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int32_t k = 0; k < K; ++k) {
      ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data()) ?
        epi_fusion_params_.tensor_alpha[k] : epi_fusion_params_.alpha;
      ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data()) ?
        epi_fusion_params_.tensor_beta[k] : epi_fusion_params_.beta;
      for (int32_t r = 0; r < R; ++r) {
        for (int32_t s = 0; s < S; ++s) {
          for (int32_t c = 0; c < C; ++c) {
            auto accumulator = ElementAcc(0);
            for (int32_t n = 0; n < N; ++n) {
              for (int32_t p = 0; p < P; ++p) {
                for (int32_t q = 0; q < Q; ++q) {
                  int32_t w =  q * cute::get<0>(tstride_) - cute::get<0>(padding_) + s * cute::get<0>(dilation_);
                  int32_t h =  p * cute::get<1>(tstride_) - cute::get<1>(padding_) + r * cute::get<1>(dilation_);
                  bool is_in_bounds =
                      detail::is_activation_in_bounds(tensor_b_, n, h, w, c);
                  if (is_in_bounds) {
                    auto act =
                        tensor_b_(c, w, h, n);
                    auto xformed_act =
                        tensor_a_(k, q, p, n);
                    accumulator += ElementAcc(act * xformed_act);
                  }
                }
              }
            }
            ElementCompute output = scale_converter(alpha) * acc_converter(accumulator) +
                                    scale_converter(beta) * residual_converter(tensor_c_(c, s, r, k));
            if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
              output += bias_converter(epi_fusion_params_.tensor_bias[k]);
            }
            output = epi_activation(output);
            tensor_d_(c, s, r, k) = output_converter(output);
          }
        }
      }
    }
  }

  // Specialization for 3D wgrad kernel
  void wgrad_reference(cute::Int<3> spatial_dims) {
    int32_t N =
        size<4>(tensor_a_);
    int32_t Z =
        size<3>(tensor_a_);
    int32_t P =
        size<2>(tensor_a_);
    int32_t Q =
        size<1>(tensor_a_);
    int32_t K =
        size<0>(tensor_a_);
    int32_t T = size<3>(tensor_d_);
    int32_t R = size<2>(tensor_d_);
    int32_t S = size<1>(tensor_d_);
    int32_t C = size<0>(tensor_d_);

#if defined(_OPENMP)
    #pragma omp parallel for collapse(3)
#endif
    for (int32_t k = 0; k < K; ++k) {
      ElementScalar alpha = raw_pointer_cast(epi_fusion_params_.tensor_alpha.data()) ?
        epi_fusion_params_.tensor_alpha[k] : epi_fusion_params_.alpha;
      ElementScalar beta = raw_pointer_cast(epi_fusion_params_.tensor_beta.data()) ?
        epi_fusion_params_.tensor_beta[k] : epi_fusion_params_.beta;
      for (int32_t t = 0; t < T; ++t) {
        for (int32_t r = 0; r < R; ++r) {
          for (int32_t s = 0; s < S; ++s) {
            for (int32_t c = 0; c < C; ++c) {
              auto accumulator = ElementAcc(0);
              for (int32_t n = 0; n < N; ++n) {
                for (int32_t z = 0; z < Z; ++z) {
                  for (int32_t p = 0; p < P; ++p) {
                    for (int32_t q = 0; q < Q; ++q) {
                      int32_t w =  q * cute::get<0>(tstride_) - cute::get<0>(padding_) + s * cute::get<0>(dilation_);
                      int32_t h =  p * cute::get<1>(tstride_) - cute::get<1>(padding_) + r * cute::get<1>(dilation_);
                      int32_t d =  z * cute::get<2>(tstride_) - cute::get<2>(padding_) + t * cute::get<2>(dilation_);
                      bool is_in_bounds =
                          detail::is_activation_in_bounds(tensor_b_, n, d, h, w, c);
                      if (is_in_bounds) {
                        auto act =
                            tensor_b_(c, w, h, d, n);
                        auto xformed_act =
                            tensor_a_(k, q, p, z, n);
                        accumulator += ElementAcc(act * xformed_act);
                      }
                    }
                  }
                }
              }
              ElementCompute output = scale_converter(alpha) * acc_converter(accumulator) +
                                      scale_converter(beta) * residual_converter(tensor_c_(c, s, r, t, k));
              if (raw_pointer_cast(epi_fusion_params_.tensor_bias.data())) {
                output += bias_converter(epi_fusion_params_.tensor_bias[k]);
              }
              output = epi_activation(output);
              tensor_d_(c, s, r, t, k) = output_converter(output);
            }
          }
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // cutlass::reference::host

/////////////////////////////////////////////////////////////////////////////////////////////////
