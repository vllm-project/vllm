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
    \brief Reference implementation for complex-valued TRMM in host-side code.

  
*/

#pragma once

#include "cutlass/blas3.h"
#include "cutlass/complex.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_view.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/util/reference/host/gemm.h"

namespace cutlass {
namespace reference {
namespace host {

/// Computes a Triangular Matrix Multiplication (tensors of rank=2) pointed to by TensorRef
/// objects.
template <
  typename ElementA,
  typename LayoutA,
  ComplexTransform TransformA,
  SideMode SideModeA,
  FillMode FillModeA,
  DiagType DiagTypeA,
  typename ElementB,
  typename LayoutB,
  ComplexTransform TransformB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ComputeType,
  typename InnerProductOp = multiply_add<ComputeType>,
  typename ConvertOp = NumericConverter<ElementC, ScalarType>
>
void compute_trmm_complex(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRef<ElementA, LayoutA> tensor_a,
  TensorRef<ElementB, LayoutB> tensor_b,
  TensorRef<ElementC, LayoutC> tensor_d,
  ComputeType initial_accum) {

  static_assert(
    LayoutA::kRank == 2 &&
    LayoutC::kRank == 2, "Tensors must be of rank 2");

  static_assert(SideModeA != SideMode::kInvalid
                , "Side Mode can either be Left or Right.");

  static_assert(FillModeA == FillMode::kLower || FillModeA == FillMode::kUpper
                , "Fill Mode can either be Lower or Upper.");

  using CompareOp = typename TrMatrixCompareOp<FillModeA, DiagTypeA>::Type;
  
  // Note: batch is ignored.
  int const M = problem_size.m();
  int const N = problem_size.n();
  // Assuming correct k-dimension value is passed
  int const K = problem_size.k();
 
  // Blocking necessary to speedup reference implementation
  int const Mblock = 16;
  int const Nblock = 16;

  ConvertOp convert_op;
  InnerProductOp inner_product_op;
  CompareOp compare_op;
  
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
              ElementA a = ElementA();
              ElementB b = ElementB();
              
              if (SideModeA == SideMode::kLeft) {
                a = (compare_op(row, k_block)) ? 
                              (tensor_a.at(MatrixCoord(row, k_block))) : ElementA(0);
                if (row == k_block && DiagTypeA == DiagType::kUnit) {
                  a = ElementA(1);
                }
                b = tensor_b.at(MatrixCoord(k_block, col));
              } else if (SideModeA == SideMode::kRight) {
                a = tensor_b.at(MatrixCoord(row, k_block));
                b = (compare_op(k_block, col)) ? 
                      tensor_a.at(MatrixCoord(k_block, col)) : ElementA(0);
                if (k_block == col && DiagTypeA == DiagType::kUnit) {
                  b = ElementA(1);
                }
              }

              ComputeType a_ik = ComputeType(a);
              ComputeType b_kj = ComputeType(b);
              
              // Conjugate, and hence hermitian, is only allowed for the triangular matrix
              if (SideModeA == SideMode::kLeft && TransformA == ComplexTransform::kConjugate) {
                a_ik = conj(a_ik);
              } else if (SideModeA == SideMode::kRight && TransformA == ComplexTransform::kConjugate) {
                b_kj = conj(b_kj);
              }

              accum[i][j] = inner_product_op(a_ik, b_kj,  accum[i][j]);
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
              alpha * ScalarType(accum[i][j]));
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA,
  typename LayoutA,
  ComplexTransform TransformA,
  SideMode SideModeA,
  FillMode FillModeA,
  DiagType DiagTypeA,
  typename ElementB,
  typename LayoutB,
  ComplexTransform TransformB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ComputeType,
  typename InnerProductOp = cutlass::arch::OpMultiplyAddComplex
>
struct TrmmComplex;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for multiply-add
template <typename ElementA, typename LayoutA, ComplexTransform TransformA,
          SideMode SideModeA, FillMode FillModeA, DiagType DiagTypeA, 
          typename ElementB, typename LayoutB, ComplexTransform TransformB,
          typename ElementC, typename LayoutC,
          typename ScalarType, typename ComputeType>
struct TrmmComplex<ElementA, LayoutA, TransformA, 
                   SideModeA, FillModeA, DiagTypeA,
                   ElementB, LayoutB, TransformB,
                   ElementC, LayoutC, ScalarType,
                   ComputeType, arch::OpMultiplyAddComplex> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b,
                  TensorRef<ElementC, LayoutC> tensor_d,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_trmm_complex<ElementA, LayoutA, TransformA,
                 SideModeA, FillModeA, DiagTypeA,
                 ElementB, LayoutB, TransformB,
                 ElementC, LayoutC, 
                 ScalarType, ComputeType, multiply_add<ComputeType>>(
                 problem_size, alpha, tensor_a, tensor_b, tensor_d, initial_accum);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for gaussian multiply-add 
template <typename ElementA, typename LayoutA, ComplexTransform TransformA,
          SideMode SideModeA, FillMode FillModeA, DiagType DiagTypeA, 
          typename ElementB, typename LayoutB, ComplexTransform TransformB,
          typename ElementC, typename LayoutC,
          typename ScalarType, typename ComputeType>
struct TrmmComplex<ElementA, LayoutA, TransformA, 
                   SideModeA, FillModeA, DiagTypeA,
                   ElementB, LayoutB, TransformB,
                   ElementC, LayoutC, ScalarType,
                   ComputeType, arch::OpMultiplyAddGaussianComplex> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b,
                  TensorRef<ElementC, LayoutC> tensor_d,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_trmm_complex<ElementA, LayoutA, TransformA,
                 SideModeA, FillModeA, DiagTypeA,
                 ElementB, LayoutB, TransformB,
                 ElementC, LayoutC, 
                 ScalarType, ComputeType, multiply_add<ComputeType>>(
                 problem_size, alpha, tensor_a, tensor_b, tensor_d, initial_accum);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
