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
    \brief Reference implementation for GETT in host-side code.
*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////
#include "cutlass/gemm/gemm.h"
#include "cutlass/complex.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/relatively_equal.h"

#include "cute/tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::reference::host {

template<class T, class = void>
struct ElementTraits {
  using type = T;
};

template<class T>
struct ElementTraits<T, std::enable_if_t<!std::is_same_v<decltype(std::declval<T>().get()), void> > >  {
  using type = decltype(std::declval<T>().get());
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  class ElementAccumulator_,
  class TensorA_,                                                                                         // (M, K, L)
  class TensorB_                                                                                          // (N, K, L)
>
struct GettMainloopParams {
  using ElementAccumulator = ElementAccumulator_;
  using TensorA = TensorA_;
  using TensorB = TensorB_;
  using EngineA = typename TensorA::engine_type;
  using LayoutA = typename TensorA::layout_type;
  using EngineB = typename TensorB::engine_type;
  using LayoutB = typename TensorB::layout_type;

  TensorA A{};
  TensorB B{};

  ComplexTransform transform_A = ComplexTransform::kNone;
  ComplexTransform transform_B = ComplexTransform::kNone;
  
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template<
  class ElementScalar_,
  class ElementScalingFactor_,
  class ElementAccumulator_,
  class ElementCompute_,
  class TensorC_,                                                                                          // (M, N, L)
  class TensorD_,                                                                                          // (M, N, L)
  class VectorBias_ = TensorD_,                                                                            //    (M, 1)
  class TensorAux_ = TensorD_,                                                                             // (M, N, L)
  class VectorAlpha_ = TensorD_,                                                                           //    (M, 1)
  class VectorBeta_ = VectorAlpha_,                                                                        //    (M, 1)
  class ActivationFunctor_ = cutlass::epilogue::thread::Identity<ElementCompute_>,
  class BiasBinaryOp_ = cutlass::plus<ElementCompute_>,
  bool PerColumnBias_ = false
>
struct GettEpilogueParams {
  using ElementScalar = ElementScalar_;
  using ElementScalingFactor = ElementScalingFactor_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using TensorC = TensorC_;
  using TensorD = TensorD_;
  using TensorAux = TensorAux_;
  using VectorBias = VectorBias_;
  using VectorAlpha = VectorAlpha_;
  using VectorBeta = VectorBeta_;
  using ActivationFunctor = ActivationFunctor_;
  using BiasBinaryOp = BiasBinaryOp_;

  using EngineC = typename TensorC::engine_type;
  using LayoutC = typename TensorC::layout_type;
  using EngineD =  typename TensorD::engine_type;
  using LayoutD = typename TensorD::layout_type;
  static constexpr bool PerColumnBias = PerColumnBias_;
  ElementScalar alpha = ElementScalar(1);
  ElementScalar beta = ElementScalar(0);

  TensorC C{};
  TensorD D{};
  VectorBias Bias{};
  TensorAux Aux{};
  VectorAlpha Valpha{};
  VectorBeta Vbeta{};
  ElementCompute st = ElementCompute(1);

  ElementAccumulator* abs_max_D = nullptr;
  ElementAccumulator* abs_max_Aux = nullptr;

  ElementScalingFactor scale_a = ElementScalingFactor(1);
  ElementScalingFactor scale_b = ElementScalingFactor(1);
  ElementScalingFactor scale_c = ElementScalingFactor(1);
  ElementScalingFactor scale_d = ElementScalingFactor(1);
  ElementScalingFactor scale_aux = ElementScalingFactor(1);

  bool beta_per_channel_scaling = false;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GETT - General Tensor-Tensor contraction reference kernel
template <
  class MainloopParams,
  class EpilogueParams
>
void Gett(
    MainloopParams const& mainloop_params,
    EpilogueParams const& epilogue_params)
{

  static int constexpr kBlockM = 64;
  static int constexpr kBlockN = 64;

#if defined(_OPENMP)
  #pragma omp parallel for collapse(3)
#endif
  for (int64_t l = 0; l < cute::size<2>(mainloop_params.A.layout()); ++l) {
    for (int64_t m = 0; m < cute::size<0>(mainloop_params.A.layout()); m += kBlockM) {
      for (int64_t n = 0; n < cute::size<0>(mainloop_params.B.layout()); n += kBlockN) {
        typename MainloopParams::ElementAccumulator acc[kBlockM][kBlockN];
        gett_mainloop(mainloop_params, m, n, l, acc);
        gett_epilogue(epilogue_params, m, n, l, acc);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GETT - Mainloop
template <class MainloopParams, class ElementAccumulator, int kBlockM, int kBlockN>
void gett_mainloop(
    MainloopParams const& mainloop_params,
    int64_t m,
    int64_t n,
    int64_t l,
    ElementAccumulator (&acc)[kBlockM][kBlockN])
{

  static_assert(cute::rank(typename MainloopParams::LayoutA{}) == 3, "M, K, B");
  static_assert(cute::rank(typename MainloopParams::LayoutB{}) == 3, "N, K, B");
  
  using cute::raw_pointer_cast;

  using ElementA = typename ElementTraits<typename MainloopParams::EngineA::value_type>::type;
  using ElementB = typename ElementTraits<typename MainloopParams::EngineB::value_type>::type;

  using RingOp = multiply_add<ElementAccumulator, ElementAccumulator, ElementAccumulator>;
  RingOp fma_op;

  // Zero out accumulators
  for (int m_b = 0; m_b < kBlockM; ++m_b) {
    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      acc[m_b][n_b] = ElementAccumulator(0); // RingOp::AdditionIdentity
    }
  }

  // Compute on this k-block
  for (int64_t k = 0; k < cute::size<1>(mainloop_params.A.layout()); ++k) {
    // Load A
    ElementAccumulator a_frag[kBlockM];
    for (int m_b = 0; m_b < kBlockM; ++m_b) {
      if (m + m_b < cute::size<0>(mainloop_params.A.layout())) {
        // Perform reference GEMM calculations at the accumulator's precision. Cast A value to accumulator type.
        a_frag[m_b] = static_cast<ElementAccumulator>(ElementA(mainloop_params.A(m + m_b, k, l)));
        
        if (mainloop_params.transform_A == ComplexTransform::kConjugate) {
          a_frag[m_b] = conj(a_frag[m_b]);
        }
      } else {
        a_frag[m_b] = ElementAccumulator(0); // RingOp::AdditionIdentity
      }
    }

    // Load B
    ElementAccumulator b_frag[kBlockN];
    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      if (n + n_b < cute::size<0>(mainloop_params.B.layout())) {
        // Perform reference GEMM calculations at the accumulator's precision. Cast A value to accumulator type.
        b_frag[n_b] = static_cast<ElementAccumulator>(ElementB(mainloop_params.B(n + n_b, k, l)));

        if (mainloop_params.transform_B == ComplexTransform::kConjugate) {
          b_frag[n_b] = conj(b_frag[n_b]);
        }
      } else {
        b_frag[n_b] = ElementAccumulator(0); // RingOp::AdditionIdentity
      }
    }

    // do compute
    for (int m_b = 0; m_b < kBlockM; ++m_b) {
      for (int n_b = 0; n_b < kBlockN; ++n_b) {
        acc[m_b][n_b] = fma_op(a_frag[m_b], b_frag[n_b], acc[m_b][n_b]);
      }
    }

  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// GETT - Epilogue
template <class EpilogueParams, class ElementAccumulator, int kBlockM, int kBlockN>
void gett_epilogue(
    EpilogueParams const& epilogue_params,
    int64_t m,
    int64_t n,
    int64_t l,
    ElementAccumulator (&acc)[kBlockM][kBlockN])
{
  static_assert(cute::rank(typename EpilogueParams::LayoutC{}) == 3, "M, K, B");
  static_assert(cute::rank(typename EpilogueParams::LayoutD{}) == 3, "N, K, B");

  using cute::raw_pointer_cast;

  using ElementCompute = typename EpilogueParams::ElementCompute;
  using ElementC = typename EpilogueParams::TensorC::value_type;
  using ElementD = typename EpilogueParams::TensorD::value_type;
  using ElementAux = typename EpilogueParams::TensorAux::value_type;
  using ElementBias = typename EpilogueParams::VectorBias::value_type;
  using ElementScalar = typename EpilogueParams::ElementScalar;
  using ElementScalingFactor = typename EpilogueParams::ElementScalingFactor;
  using ActivationFunctor = typename EpilogueParams::ActivationFunctor;
  using BiasBinaryOp = typename EpilogueParams::BiasBinaryOp;

  constexpr bool PerColBias = EpilogueParams::PerColumnBias;
  constexpr bool IsScalingAndAmaxOutputNeeded = 
      cute::is_same_v<ElementD, cutlass::float_e4m3_t> or
      cute::is_same_v<ElementD, cutlass::float_e5m2_t>;

  constexpr bool IsScalingAndAmaxAuxOutputNeeded =
      cute::is_same_v<ElementAux, cutlass::float_e4m3_t> or
      cute::is_same_v<ElementAux, cutlass::float_e5m2_t>;

  constexpr bool IsReLUAuxNeeded =
      (cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::ReLu<ElementCompute>> or
       cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::Clamp<ElementCompute>>) and 
      cute::is_same_v<ElementAux, cutlass::uint1b_t>;
  constexpr bool IsClamp =
      cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::Clamp<ElementCompute>>;

  constexpr bool IsBackpropFusion =
      cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::dGELU<ElementCompute>> or
      cute::is_same_v<ActivationFunctor, cutlass::epilogue::thread::dReLU<ElementCompute>>;

  // Input related converter
  NumericConverter<ElementCompute, ElementAccumulator> accumulator_converter;
  NumericConverter<ElementCompute, ElementC> source_converter;
  NumericConverter<ElementCompute, ElementBias> bias_converter;
  [[maybe_unused]] NumericConverter<ElementCompute, ElementAux> aux_source_converter;

  // Scale related converter
  NumericConverter<ElementCompute, ElementScalar> scale_converter;
  NumericConverter<ElementCompute, ElementScalingFactor> scaling_factor_converter;

  // Abs max converter
  [[maybe_unused]] NumericConverter<ElementAccumulator, ElementCompute> abs_max_output_converter;

  // Output related converter
  NumericConverter<ElementD, ElementCompute> destination_converter;
  [[maybe_unused]] NumericConverter<ElementAux, ElementCompute> aux_destination_converter;
  NumericConverter<ElementBias, ElementCompute> dBias_converter;

  // Epilogue operations
  multiply_add<ElementCompute, ElementCompute, ElementCompute> epilogue_fma;
  multiplies<ElementCompute> mul;
  plus<ElementCompute> add;

  // Activation operation
  ActivationFunctor activation;

  // Bias binary operation
  BiasBinaryOp bias_op;

  // Do conversion
  ElementCompute converted_alpha = scale_converter(epilogue_params.alpha);
  ElementCompute converted_beta = scale_converter(epilogue_params.beta);
  ElementCompute converted_scale_a = scaling_factor_converter(epilogue_params.scale_a);
  ElementCompute converted_scale_b = scaling_factor_converter(epilogue_params.scale_b);
  ElementCompute converted_scale_c = scaling_factor_converter(epilogue_params.scale_c);
  ElementCompute converted_scale_d = scaling_factor_converter(epilogue_params.scale_d);
  ElementCompute converted_scale_aux = scaling_factor_converter(epilogue_params.scale_aux);

  // Init local var
  [[maybe_unused]] ElementCompute local_abs_max_output = ElementCompute(0);
  [[maybe_unused]] ElementCompute local_abs_max_aux_output = ElementCompute(0);

  converted_alpha = mul(converted_alpha, mul(converted_scale_a, converted_scale_b));
  converted_beta = mul(converted_beta, converted_scale_c);

  ElementCompute inter_accum[kBlockM][kBlockN];

  for (int m_b = 0; m_b < kBlockM; ++m_b) {
    ElementCompute local_dBias = ElementCompute(0);

    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      if (m + m_b < cute::size<0>(epilogue_params.D.layout()) && n + n_b < cute::size<1>(epilogue_params.D.layout())) {
        // Convert every type to ElementCompute first, do compute, convert to output type, write it out
        ElementCompute converted_acc = accumulator_converter(acc[m_b][n_b]);
        // per-row alpha
        if (raw_pointer_cast(epilogue_params.Valpha.data())) {
          converted_alpha = scale_converter(epilogue_params.Valpha(m + m_b, n + n_b, l));
          converted_alpha = mul(converted_alpha, mul(converted_scale_a, converted_scale_b));
        }
        ElementCompute output = mul(converted_alpha, converted_acc);

        if (raw_pointer_cast(epilogue_params.Bias.data()) && not IsBackpropFusion) {
          ElementCompute converted_bias = bias_converter(epilogue_params.Bias(PerColBias ? n + n_b : m + m_b));
          output = bias_op(output, converted_bias);
        }

        if (raw_pointer_cast(epilogue_params.C.data())) {
          ElementCompute converted_src = source_converter(epilogue_params.C(m + m_b, n + n_b, l));
          // per-row beta
          if (epilogue_params.Vbeta.data()) {
            converted_beta = scale_converter(epilogue_params.Vbeta(m + m_b, n + n_b, l));
            converted_beta = mul(converted_beta, converted_scale_c);
          }
          output = epilogue_fma(converted_beta, converted_src, output);
        }

        if constexpr (IsBackpropFusion) {
          ElementAux aux_input = ElementAux(0);
          if (raw_pointer_cast(epilogue_params.Aux.data())) {
            aux_input = epilogue_params.Aux(m + m_b, n + n_b, l);
          }

          output = activation(output, aux_source_converter(aux_input));
          local_dBias = add(local_dBias, output);
        }
        else {
          if (raw_pointer_cast(epilogue_params.Aux.data())) {
            auto aux_output = output;
            if constexpr (IsScalingAndAmaxAuxOutputNeeded) {
              maximum_absolute_value_reduction<ElementCompute, true> amax_op;
              local_abs_max_aux_output = amax_op(local_abs_max_aux_output, aux_output);
              aux_output = epilogue_fma(converted_scale_aux, aux_output, ElementCompute(0));
            }

            if constexpr (IsReLUAuxNeeded) {
              epilogue_params.Aux(m + m_b, n + n_b, l) = not (aux_output < 0) ? uint1b_t(1) : uint1b_t(0);
            } else {
              epilogue_params.Aux(m + m_b, n + n_b, l) = aux_destination_converter(aux_output);
            }
          }

          if constexpr (IsClamp) { // Treat Clamp as ReLU
            output = activation(output, {0, std::numeric_limits<ElementCompute>::max()});
          }
          else {
            output = activation(output);
          }
        }

        if constexpr (IsScalingAndAmaxOutputNeeded) {
          maximum_absolute_value_reduction<ElementCompute, true> amax_op;
          local_abs_max_output = amax_op(local_abs_max_output, output);
          output = epilogue_fma(converted_scale_d, output, ElementCompute(0));
        }

        inter_accum[m_b][n_b] = ElementCompute(output);
      }
    } // n_b

    if (m + m_b < cute::size<0>(epilogue_params.D.layout()) && n < cute::size<1>(epilogue_params.D.layout())) {
      if (raw_pointer_cast(epilogue_params.Bias.data()) && IsBackpropFusion) {
        ElementCompute converted_dBias = bias_converter(epilogue_params.Bias(m + m_b));
        local_dBias = add(local_dBias, converted_dBias);
        epilogue_params.Bias(m + m_b) = dBias_converter(local_dBias);
      }
    }
  } // m_b
  for (int m_b = 0; m_b < kBlockM; ++m_b) {
    for (int n_b = 0; n_b < kBlockN; ++n_b) {
      if (m + m_b < cute::size<0>(epilogue_params.D.layout()) && n + n_b < cute::size<1>(epilogue_params.D.layout())) {
        epilogue_params.D(m + m_b, n + n_b, l) = destination_converter(inter_accum[m_b][n_b]);
      }
    }
  }

#if defined(_OPENMP)
  #pragma omp critical(Abs_Max_Data_Update)
#endif
  {
    if constexpr (IsScalingAndAmaxOutputNeeded) {
      if (epilogue_params.abs_max_D) {
        *epilogue_params.abs_max_D = maximum_with_nan_propogation<ElementAccumulator>{}(
          *epilogue_params.abs_max_D, abs_max_output_converter(local_abs_max_output));
      }
    }

    if constexpr (IsScalingAndAmaxAuxOutputNeeded) {
      if (epilogue_params.abs_max_Aux) {
        *epilogue_params.abs_max_Aux = maximum_with_nan_propogation<ElementAccumulator>{}(
            *epilogue_params.abs_max_Aux, abs_max_output_converter(local_abs_max_aux_output));
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class TensorType>
auto make_layout_rank3(const TensorType& tensor) {
  // append a batch mode of size 1 if we do not have tensors that are rank 3
  return make_layout(
      make_shape(cute::get<0>(tensor.shape()), cute::get<1>(tensor.shape()), cute::Int<1>{}),
      make_stride(cute::get<0>(tensor.stride()), cute::get<1>(tensor.stride()), int64_t(cosize(tensor.layout()))));
}

/// GEMM - General Matrix-Matrix contraction without conjugation options
template <
  class MainloopParams,
  class EpilogueParams
>
void Gemm3x(
    MainloopParams const& mainloop_params,
    EpilogueParams const& epilogue_params)
{
  using namespace cute;

  static_assert(cute::rank(typename MainloopParams::LayoutA{}) == cute::rank(typename MainloopParams::LayoutB{}));
  static_assert(cute::rank(typename EpilogueParams::LayoutC{}) == cute::rank(typename EpilogueParams::LayoutD{}));
  static_assert(cute::rank(typename MainloopParams::LayoutA{}) == cute::rank(typename EpilogueParams::LayoutC{}));

  if constexpr (cute::rank(typename MainloopParams::LayoutA{}) == 2) {
    cute::Layout layout_A = make_layout_rank3(mainloop_params.A);
    cute::Layout layout_B = make_layout_rank3(mainloop_params.B);
    cute::Layout layout_C = make_layout_rank3(epilogue_params.C);
    cute::Layout layout_D = make_layout_rank3(epilogue_params.D);
    cute::Layout layout_Aux = make_layout_rank3(epilogue_params.Aux);
    cute::Layout layout_Bias = make_layout_rank3(epilogue_params.Bias);
    cute::Layout layout_Valpha = make_layout_rank3(epilogue_params.Valpha);
    cute::Layout layout_Vbeta = make_layout_rank3(epilogue_params.Vbeta);
    
    auto TensorA = make_tensor(mainloop_params.A.data(), layout_A);
    auto TensorB = make_tensor(mainloop_params.B.data(), layout_B);
    auto TensorC = make_tensor(epilogue_params.C.data(), layout_C);
    auto TensorD = make_tensor(epilogue_params.D.data(), layout_D);
    auto TensorAux = make_tensor(epilogue_params.Aux.data(), layout_Aux);
    auto VectorBias = make_tensor(epilogue_params.Bias.data(), layout_Bias);
    auto VectorAlpha = make_tensor(epilogue_params.Valpha.data(), layout_Valpha);
    auto VectorBeta = make_tensor(epilogue_params.Vbeta.data(), layout_Vbeta);

    // Reconstruct mainloop params
    GettMainloopParams<typename MainloopParams::ElementAccumulator,
                       decltype(TensorA),
                       decltype(TensorB)>
        mainloop_params_converted{TensorA,
                                  TensorB,
                                  mainloop_params.transform_A,
                                  mainloop_params.transform_B};

    // Reconstruct epilogue params
    GettEpilogueParams<typename EpilogueParams::ElementScalar,
                       typename EpilogueParams::ElementScalingFactor,
                       typename EpilogueParams::ElementAccumulator,
                       typename EpilogueParams::ElementCompute,
                       decltype(TensorC),
                       decltype(TensorD),
                       decltype(VectorBias),
                       decltype(TensorAux),
                       decltype(VectorAlpha),
                       decltype(VectorBeta)
                      >
        epilogue_params_converted{epilogue_params.alpha,
                                  epilogue_params.beta,
                                  TensorC,
                                  TensorD,
                                  VectorBias,
                                  TensorAux,
                                  VectorAlpha,
                                  VectorBeta,
                                  epilogue_params.abs_amax_D,
                                  epilogue_params.abs_amax_Aux,
                                  epilogue_params.scale_a,
                                  epilogue_params.scale_b,
                                  epilogue_params.scale_c,
                                  epilogue_params.scale_d,
                                  epilogue_params.scale_aux
                                  };

    Gett(mainloop_params_converted, epilogue_params_converted);
  }
  else {
    // if we already have a batch mode, just pass it through
    Gett(mainloop_params, epilogue_params);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // cutlass::reference::host

/////////////////////////////////////////////////////////////////////////////////////////////////
