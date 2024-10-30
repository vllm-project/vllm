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
    \brief Reference implementation for Rank 2k update in host-side code.
    
    

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
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  FillMode FillModeC,
  typename ScalarType,
  typename ComputeType,
  typename InnerProductOp = multiply_add<ComputeType>,
  typename ConvertOp = NumericConverter<ElementC, ScalarType>
>
void compute_rank2k(
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

  static_assert(
    FillModeC == FillMode::kLower || 
    FillModeC == FillMode::kUpper, 
    "Fill Mode can either be Lower or Upper.");

  using CompareOp = typename platform::conditional<(FillModeC == FillMode::kLower), 
                                                    std::greater_equal<int>, 
                                                    std::less_equal<int>>::type;

  // Note: batch is ignored.
  // Note: M is same as N for Rank 2k update
  int const N = problem_size.n();
  int const K = problem_size.k();

  // Blocking necessary to speedup reference implementation
  int const Nblock = 16;

  ConvertOp convert_op;
  InnerProductOp inner_product_op;
  CompareOp compare_op;

  for (int row_block = 0; row_block < N; row_block += Nblock) {
    for (int col_block = 0; col_block < N; col_block += Nblock) {

      ComputeType accum[Nblock][Nblock];

      for (int j = 0; j < Nblock; j++) {
        for (int i = 0; i < Nblock; i++) {
          accum[i][j] = initial_accum;
        }
      }

      for (int k_block = 0; k_block < K; ++k_block) {
        for (int j = 0; j < Nblock; j++) {
          for (int i = 0; i < Nblock; i++) {
            int row = row_block + i;
            int col = col_block + j;

            if (row < N && col < N && compare_op(row, col)) 
            {

              // A x B^T
              ElementA a = tensor_a.at(MatrixCoord(row, k_block));
              ElementB b_t = tensor_b.at(MatrixCoord(col, k_block));

              ComputeType compute_a(cast_if_scalar<ComputeType>(a));
              ComputeType compute_b_t(cast_if_scalar<ComputeType>(b_t));

              accum[i][j] = inner_product_op(compute_a, compute_b_t, accum[i][j]);

              // B x A^T
              ElementB b = tensor_b.at(MatrixCoord(row, k_block));
              ElementA a_t = tensor_a.at(MatrixCoord(col, k_block));

              ComputeType compute_b(cast_if_scalar<ComputeType>(b));
              ComputeType compute_a_t(cast_if_scalar<ComputeType>(a_t));

              accum[i][j] = inner_product_op(compute_b, compute_a_t, accum[i][j]);
            }
          }
        }
      }

      for (int j = 0; j < Nblock; j++) {
        for (int i = 0; i < Nblock; i++) {
          int row = row_block + i;
          int col = col_block + j;

          MatrixCoord coord = MatrixCoord(row, col);

          if (row < N && col < N && 
              ( (FillModeC == FillMode::kLower && row >= col) || 
                (FillModeC == FillMode::kUpper && row <= col) )
          ) {
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

/// Computes a general Rank 2k update (tensors of rank=2) pointed to by TensorRef
/// objects.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  FillMode FillModeC,
  typename ScalarType,
  typename ComputeType,
  typename InnerProductOp = multiply_add<ComputeType>,
  typename ConvertOp = NumericConverter<ElementC, ScalarType>
>
void compute_rank2k(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRef<ElementA, LayoutA> tensor_a,
  TensorRef<ElementB, LayoutB> tensor_b,
  ScalarType beta,
  TensorRef<ElementC, LayoutC> tensor_c,
  ComputeType initial_accum) {
  compute_rank2k<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, FillModeC,
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
  FillMode FillModeC,
  typename ScalarType,
  typename ComputeType,
  typename InnerProductOp = cutlass::arch::OpMultiplyAdd
>
struct Rank2K;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for multiply-add
template <typename ElementA, typename LayoutA, 
          typename ElementB, typename LayoutB, 
          typename ElementC, typename LayoutC, FillMode FillModeC,
          typename ScalarType, typename ComputeType>
struct Rank2K<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, FillModeC, ScalarType,
            ComputeType, arch::OpMultiplyAdd> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  ComputeType initial_accum = ComputeType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_rank2k<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, FillModeC,
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

    compute_rank2k<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, FillModeC,
                 ScalarType, ComputeType, multiply_add<ComputeType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_d, initial_accum);
  }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
