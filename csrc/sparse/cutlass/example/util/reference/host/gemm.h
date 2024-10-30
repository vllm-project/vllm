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

#include "cutlass/coord.h"
#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/tensor_view.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/mma.h"
#include "cutlass/util/host_tensor.h"

namespace cutlass {
namespace reference {
namespace host {

template<typename Out, typename In>
struct CastIfScalar {
  static Out cast(In in) {
    return Out(in);
  }
};

template<typename OutScalar, typename In>
struct CastIfScalar<cutlass::complex<OutScalar>, In> {
  typedef cutlass::complex<OutScalar> Out;
  static Out cast(In in) {
    return Out(static_cast<OutScalar>(in));
  }
};

template<typename OutScalar, typename InScalar>
struct CastIfScalar<cutlass::complex<OutScalar>, cutlass::complex<InScalar>> {
  typedef cutlass::complex<OutScalar> Out;
  typedef cutlass::complex<InScalar> In;
  static Out cast(In in) {
    return Out(in);
  }
};

template<typename Out, typename In>
Out cast_if_scalar(In in) {
  return CastIfScalar<Out, In>::cast(in);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product among matrices (tensors of rank=2) pointed to by TensorRef
/// objects.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ComputeType,
  typename InnerProductOp = multiply_add<ComputeType>,
  typename ConvertOp = NumericConverter<ElementC, ScalarType>
>
void compute_gemm(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRef<ElementA, LayoutA> tensor_a,
  TensorRef<ElementB, LayoutB> tensor_b,
  ScalarType beta,
  TensorRef<ElementC, LayoutC> tensor_c,
  TensorRef<ElementC, LayoutC> tensor_d,
  ComputeType initial_accum) {

  static_assert(
    LayoutA::kRank == 2 &&
    LayoutB::kRank == 2 &&
    LayoutC::kRank == 2, "Tensors must be of rank 2");


  // Note: batch is ignored.
  int const M = problem_size.m();
  int const N = problem_size.n();
  int const K = problem_size.k();

  // Blocking necessary to speedup reference implementation
  int const Mblock = 16;
  int const Nblock = 16;

  ConvertOp convert_op;
  InnerProductOp inner_product_op;

  for (int row_block = 0; row_block < M; row_block += Mblock) {
    for (int col_block = 0; col_block < N; col_block += Nblock) {

      ComputeType accum[Mblock][Nblock];

      for (int j = 0; j < Nblock; j++) {
        for (int i = 0; i < Mblock; i++) {
          accum[i][j] = initial_accum;
        }
      }

      for (int k_block = 0; k_block < K; ++k_block) {
        for (int j = 0; j < Nblock; j++) {
          for (int i = 0; i < Mblock; i++) {
            int row = row_block + i;
            int col = col_block + j;

            if (row < M && col < N) {
              ElementA a = tensor_a.at(MatrixCoord(row, k_block));
              ElementB b = tensor_b.at(MatrixCoord(k_block, col));

              ComputeType compute_a(cast_if_scalar<ComputeType>(a));
              ComputeType compute_b(cast_if_scalar<ComputeType>(b));

              accum[i][j] = inner_product_op(compute_a, compute_b, accum[i][j]);
            }
          }
        }
      }

      for (int j = 0; j < Nblock; j++) {
        for (int i = 0; i < Mblock; i++) {
          int row = row_block + i;
          int col = col_block + j;

          MatrixCoord coord = MatrixCoord(row, col);

          if (row < M && col < N) {
            tensor_d.at(coord) = convert_op(
              alpha * ScalarType(accum[i][j]) +
              beta * ScalarType(tensor_c.at(coord)));
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product among matrices (tensors of rank=2) pointed to by TensorRef
/// objects.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ComputeType,
  typename InnerProductOp = multiply_add<ComputeType>,
  typename ConvertOp = NumericConverter<ElementC, ScalarType>
>
void compute_gemm(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRef<ElementA, LayoutA> tensor_a,
  TensorRef<ElementB, LayoutB> tensor_b,
  ScalarType beta,
  TensorRef<ElementC, LayoutC> tensor_c,
  ComputeType initial_accum) {
  compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
               ScalarType, ComputeType, InnerProductOp, ConvertOp>(
      problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_c,
      initial_accum);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ComputeType,
  typename InnerProductOp = cutlass::arch::OpMultiplyAdd
>
struct Gemm;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for multiply-add
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ScalarType, typename ComputeType>
struct Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ScalarType,
            ComputeType, arch::OpMultiplyAdd> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, multiply_add<ComputeType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, initial_accum);
  }
  
  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  TensorRef<ElementC, LayoutC> tensor_d,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, multiply_add<ComputeType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_d, initial_accum);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for multiply-add
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ScalarType, typename ComputeType>
struct Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ScalarType,
            ComputeType, arch::OpMultiplyAddFastBF16> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, multiply_add<ComputeType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, initial_accum);
  }
  
  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  TensorRef<ElementC, LayoutC> tensor_d,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, multiply_add<ComputeType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_d, initial_accum);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for multiply-add-saturate
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ScalarType, typename ComputeType>
struct Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ScalarType,
            ComputeType, arch::OpMultiplyAddSaturate> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, multiply_add<ComputeType>,
                 NumericConverterClamp<ElementC, ScalarType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, initial_accum);
  }

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  TensorRef<ElementC, LayoutC> tensor_d,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, multiply_add<ComputeType>,
                 NumericConverterClamp<ElementC, ScalarType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_d, initial_accum);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for XOR-popc
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ScalarType, typename ComputeType>
struct Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ScalarType,
            ComputeType, arch::OpXorPopc> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, xor_add<ComputeType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, initial_accum);
  }

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  TensorRef<ElementC, LayoutC> tensor_d,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, xor_add<ComputeType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_d, initial_accum);
  }
};

/// Partial specialization for AND-popc
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ScalarType, typename ComputeType>
struct Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ScalarType,
            ComputeType, arch::OpAndPopc> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, and_add<ComputeType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, initial_accum);
  }

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  TensorRef<ElementC, LayoutC> tensor_d,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, and_add<ComputeType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_d, initial_accum);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for multiply-add
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ScalarType, typename ComputeType>
struct Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ScalarType,
            ComputeType, arch::OpMultiplyAddFastF32> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, multiply_add<ComputeType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, initial_accum);
  }
  
  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  TensorRef<ElementC, LayoutC> tensor_d,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, multiply_add<ComputeType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_d, initial_accum);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Batched GEMM
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a batch of GEMMs over a set of matrices of common dimension.
//
// TensorRefCollection* is a type satisfying the TensorRefCollection concept.
//
template <
  typename TensorRefCollectionA,
  typename TensorRefCollectionB,
  typename TensorRefCollectionC,
  typename ScalarType,
  typename AccumulatorType
>
void BatchedGemm(
  gemm::GemmCoord problem_size,
  int batch_count,
  ScalarType alpha,
  TensorRefCollectionA const& tensor_a,
  TensorRefCollectionB const& tensor_b,
  ScalarType beta,
  TensorRefCollectionC &tensor_c,
  AccumulatorType initial_accum) {

  typename TensorRefCollectionA::ConstIterator tensor_a_it = tensor_a.begin();
  typename TensorRefCollectionB::ConstIterator tensor_b_it = tensor_b.begin();
  typename TensorRefCollectionC::ConstIterator tensor_c_it = tensor_c.begin();

  for (int batch = 0;
    batch < batch_count;
    ++batch, ++tensor_a_it, ++tensor_b_it, ++tensor_c_it) {
    
    Gemm<typename TensorRefCollectionA::Element,
         typename TensorRefCollectionA::Layout,
         typename TensorRefCollectionB::Element,
         typename TensorRefCollectionB::Layout,
         typename TensorRefCollectionC::Element,
         typename TensorRefCollectionC::Layout,
         typename TensorRefCollectionC::Element,
         typename TensorRefCollectionC::Element>
        gemm;

    gemm(problem_size, alpha, *tensor_a_it, *tensor_b_it, beta, *tensor_c_it,
         initial_accum);
  }
}

/// Computes a general matrix product among matrices (tensors of rank=2) pointed to by TensorRef
/// objects.
//
// TensorRefCollection* is a type satisfying the TensorRefCollection concept.
//
template <
  typename TensorRefCollectionA,
  typename TensorRefCollectionB,
  typename TensorRefCollectionC,
  typename ScalarType,
  typename AccumulatorType
>
void BatchedGemm(
  gemm::GemmCoord problem_size,
  int batch_count,
  ScalarType alpha,
  TensorRefCollectionA const& tensor_a,
  TensorRefCollectionB const& tensor_b,
  ScalarType beta,
  TensorRefCollectionC &tensor_c) {

  BatchedGemm(problem_size, batch_count, alpha, tensor_a, tensor_b, beta, tensor_c, ScalarType(0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
