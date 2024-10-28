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
    \brief Reference implementation for SYMM update in host-side code.
    
    

*/

#pragma once

#include "cutlass/blas3.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/tensor_view.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/mma.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"

namespace cutlass {
namespace reference {
namespace host {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product among matrices (tensors of rank=2) pointed to by TensorRef
/// objects.
template <
  typename ElementA,
  typename LayoutA,
  SideMode SideModeA,
  FillMode FillModeA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ComputeType,
  typename InnerProductOp = multiply_add<ComputeType>,
  typename ConvertOp = NumericConverter<ElementC, ScalarType>
>
void compute_symm(
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
    LayoutC::kRank == 2, 
    "Tensors must be of rank 2");

  static_assert(SideModeA != SideMode::kInvalid
                , "Side Mode can either be Left or Right.");

  static_assert(
    FillModeA == FillMode::kLower || 
    FillModeA == FillMode::kUpper, 
    "Fill Mode can either be Lower or Upper.");

  using CompareOp_w_diag =  typename TrMatrixCompareOp<FillModeA, DiagType::kNonUnit>::Type;
  using CompareOp_wo_diag = typename TrMatrixCompareOp<FillModeA, DiagType::kZero>::Type;

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
  CompareOp_w_diag compare_op_1;
  CompareOp_wo_diag compare_op_2;

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
              ElementA a_1 = ElementA();
              ElementB b_1 = ElementB();
              ElementA a_2 = ElementA();
              ElementB b_2 = ElementB();

              // A x B or B x A (with diagonal)
              if (SideModeA == SideMode::kLeft) {
                a_1 = (compare_op_1(row, k_block)) ? 
                      (tensor_a.at(MatrixCoord(row, k_block))) : ElementA();
                b_1 = tensor_b.at(MatrixCoord(k_block, col));
              } else if (SideModeA == SideMode::kRight) {
                a_1 = tensor_b.at(MatrixCoord(row, k_block));
                b_1 = (compare_op_1(k_block, col)) ? 
                      tensor_a.at(MatrixCoord(k_block, col)) : ElementA();
              }

              ComputeType compute_a_1(cast_if_scalar<ComputeType>(a_1));
              ComputeType compute_b_1(cast_if_scalar<ComputeType>(b_1));

              accum[i][j] = inner_product_op(compute_a_1, compute_b_1, accum[i][j]);

              // A^T x B or B x A^T (without diagonal)
              if (SideModeA == SideMode::kLeft) {
                a_2 = (compare_op_2(k_block, row)) ? 
                      (tensor_a.at(MatrixCoord(k_block, row))) : ElementA();
                b_2 = tensor_b.at(MatrixCoord(k_block, col));
              } else if (SideModeA == SideMode::kRight) {
                a_2 = tensor_b.at(MatrixCoord(row, k_block));
                b_2 = (compare_op_2(col, k_block)) ? 
                      tensor_a.at(MatrixCoord(col, k_block)) : ElementA();
              }

              ComputeType compute_a_2(cast_if_scalar<ComputeType>(a_2));
              ComputeType compute_b_2(cast_if_scalar<ComputeType>(b_2));

              accum[i][j] = inner_product_op(compute_a_2, compute_b_2, accum[i][j]);
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

/// Computes a general Symm update (tensors of rank=2) pointed to by TensorRef
/// objects.
template <
  typename ElementA,
  typename LayoutA,
  SideMode SideModeA,
  FillMode FillModeA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ComputeType,
  typename InnerProductOp = multiply_add<ComputeType>,
  typename ConvertOp = NumericConverter<ElementC, ScalarType>
>
void compute_symm(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRef<ElementA, LayoutA> tensor_a,
  TensorRef<ElementB, LayoutB> tensor_b,
  ScalarType beta,
  TensorRef<ElementC, LayoutC> tensor_c,
  ComputeType initial_accum) {
  compute_symm<ElementA, LayoutA, SideModeA, FillModeA, ElementB, LayoutB, ElementC, LayoutC,
               ScalarType, ComputeType, InnerProductOp, ConvertOp>(
      problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_c,
      initial_accum);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA,
  typename LayoutA,
  SideMode SideModeA,
  FillMode FillModeA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ComputeType,
  typename InnerProductOp = cutlass::arch::OpMultiplyAdd
>
struct Symm;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for multiply-add
template <typename ElementA, typename LayoutA, 
          SideMode SideModeA, FillMode FillModeA,
          typename ElementB, typename LayoutB, 
          typename ElementC, typename LayoutC,
          typename ScalarType, typename ComputeType>
struct Symm<ElementA, LayoutA, SideModeA, FillModeA, ElementB, LayoutB, ElementC, LayoutC, ScalarType,
            ComputeType, arch::OpMultiplyAdd> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_symm<ElementA, LayoutA, SideModeA, FillModeA, ElementB, LayoutB, ElementC, LayoutC,
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

    compute_symm<ElementA, LayoutA, SideModeA, FillModeA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, ComputeType, multiply_add<ComputeType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_d, initial_accum);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
