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
    \brief Reference implementation for complex-valued Rank 2K update in host-side code.

    
*/

#pragma once

#include "cutlass/blas3.h"
#include "cutlass/complex.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_view.h"
#include "cutlass/gemm/gemm.h"
#include <assert.h>

namespace cutlass {
namespace reference {
namespace host {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product among matrices (tensors of rank=2) pointed to by TensorRef
/// objects.
///
/// Explicitly naming types needed by this template can be cumbersome, particularly for the
/// accumulator type, so a function argument 'initial_accum' is exposed. Passing
/// AccumulatorType(0) as the last function argument can be easier than naming all template
/// arguments explicitly.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ComputeType,
  typename ConvertOp = NumericConverter<ElementC, ScalarType>,
  typename InnerProductOp = multiply_add<ComputeType>
>
void Rank2KComplex(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRef<ElementA, LayoutA> tensor_a,
  ComplexTransform transform_a,
  ScalarType beta,
  TensorRef<ElementC, LayoutC> tensor_c,
  TensorRef<ElementC, LayoutC> tensor_d,
  ComputeType initial_accum,
  FillMode fill_mode_c,
  BlasMode blas_mode,
  int batch_count = 1,
  int64_t batch_stride_A = 0,
  int64_t batch_stride_C = 0,
  int64_t batch_stride_D = 0) {

  static_assert(
    LayoutA::kRank == 2 &&
    LayoutC::kRank == 2, "Tensors must be of rank 2");

  // Note: batch is ignored.
  int const M = problem_size.m();
  int const N = problem_size.n();
  int const K = problem_size.k();

  // Rank2K update operates on A=NxK, B=NxK, and C=NxN
  assert(M==N);

  // Blocking necessary to speedup reference implementation
  int const Mblock = 16;
  int const Nblock = 16;

  ConvertOp convert_op;
  InnerProductOp inner_product_op;

  for (int batch_idx = 0; batch_idx < batch_count; ++batch_idx) {

    // Compute matrix product using blocks
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

              if (row < M && col < N &&
                 ( (fill_mode_c == FillMode::kLower && row >= col) || 
                  (fill_mode_c == FillMode::kUpper && row <= col) )               
                ) {
                
                // A x A^T (Symmetric) or A x A^H (Hermitian)
                // complex conjugation on operandB (a_t) (function of blas3 computation)
                ElementA a = tensor_a.at(MatrixCoord(row, k_block));
                ElementA a_t = (blas_mode == BlasMode::kHermitian) ? 
                              conj(tensor_a.at(MatrixCoord(col, k_block))) : 
                              tensor_a.at(MatrixCoord(col, k_block));

                ComputeType a_ik = ComputeType(a);
                ComputeType b_jk = ComputeType(a_t);

                // complex conjugation (function of input layouts)
                if (transform_a == ComplexTransform::kConjugate) {
                  a_ik = conj(a_ik);
                }
                // complex conjugation (function of input layouts)
                if (transform_a == ComplexTransform::kConjugate) {
                  b_jk = conj(b_jk);
                }

                accum[i][j] = inner_product_op(a_ik, b_jk,  accum[i][j]);

              }
            }
          }
        }

        for (int j = 0; j < Nblock; j++) {
          for (int i = 0; i < Mblock; i++) {
            int row = row_block + i;
            int col = col_block + j;

            MatrixCoord coord = MatrixCoord(row, col);

            if (row < M && col < N && 
                ((fill_mode_c == FillMode::kLower && row >= col) || 
                 (fill_mode_c == FillMode::kUpper && row <= col))
              ) {

              ScalarType c = tensor_c.at(coord);
              // The imaginary parts of the diagonal elements of 
              // a complex data type are assumed and set to zero
              if (blas_mode == BlasMode::kHermitian) {
                c = (row == col) ? real(c) : c;
              }

              ScalarType tmp_d = convert_op(
                alpha * ScalarType(accum[i][j]) + 
                beta * c);

              if (blas_mode == BlasMode::kHermitian && row == col ) {
                tensor_d.at(coord) = real(tmp_d);
              } else {
                tensor_d.at(coord) = tmp_d;
              }
            }
          }
        }

      } // for (col_block)
    } // for (row_block)

    tensor_a.add_pointer_offset(batch_stride_A);
    tensor_c.add_pointer_offset(batch_stride_C);
    tensor_d.add_pointer_offset(batch_stride_D);

  } // for (batch_idx)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes a general matrix product among matrices (tensors of rank=2) pointed to by TensorRef
/// objects.
///
/// This assumes the accumulator type is the same type as the scalars.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementC,
  typename LayoutC,
  typename ScalarType
>
void RankKComplex(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRef<ElementA, LayoutA> tensor_a,
  ComplexTransform transform_a,
  ScalarType beta,
  TensorRef<ElementC, LayoutC> tensor_c,
  TensorRef<ElementC, LayoutC> tensor_d,
  FillMode fill_mode_c,
  BlasMode blas_mode) {

  Rank2KComplex(
    problem_size, alpha, 
    tensor_a, transform_a, 
    beta, tensor_c, tensor_d, 
    ScalarType(0),
    fill_mode_c,
    blas_mode);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
