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
    \brief Reference implementation for complex-valued GEMM in host-side code.
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/complex.h"
#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/matrix_coord.h"

#include "cutlass/tensor_view.h"

#include "cutlass/gemm/gemm.h"

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
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ComputeType,
  typename ElementD = ElementC,
  typename ConvertOp = NumericConverter<ElementD, ScalarType>,
  typename InnerProductOp = multiply_add<ComputeType>
>
void GemmComplex(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRef<ElementA, LayoutA> tensor_a,
  ComplexTransform transform_a,
  TensorRef<ElementB, LayoutB> tensor_b,
  ComplexTransform transform_b,
  ScalarType beta,
  TensorRef<ElementC, LayoutC> tensor_c,
  TensorRef<ElementD, LayoutC> tensor_d,
  ComputeType initial_accum,
  int batch_count = 1,
  int64_t batch_stride_A = 0,
  int64_t batch_stride_B = 0,
  int64_t batch_stride_C = 0,
  int64_t batch_stride_D = 0) {

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

              if (row < M && col < N) {
                ElementA a = tensor_a.at(MatrixCoord(row, k_block));
                ElementB b = tensor_b.at(MatrixCoord(k_block, col));

                ComputeType a_ik = ComputeType(a);
                ComputeType b_kj = ComputeType(b);

                if (transform_a == ComplexTransform::kConjugate) {
                  a_ik = conj(a_ik);
                }

                if (transform_b == ComplexTransform::kConjugate) {
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
                alpha * ScalarType(accum[i][j]) + 
                beta * ScalarType(tensor_c.at(coord)));
            }
          }
        }

      } // for (col_block)
    } // for (row_block)

    tensor_a.add_pointer_offset(batch_stride_A);
    tensor_b.add_pointer_offset(batch_stride_B);
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
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename ElementD = ElementC
>
void GemmComplex(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRef<ElementA, LayoutA> tensor_a,
  ComplexTransform transform_a,
  TensorRef<ElementB, LayoutB> tensor_b,
  ComplexTransform transform_b,
  ScalarType beta,
  TensorRef<ElementC, LayoutC> tensor_c,
  TensorRef<ElementD, LayoutC> tensor_d) {

  GemmComplex(problem_size, alpha, tensor_a, transform_a, tensor_b, transform_b, beta, tensor_c, tensor_d, ScalarType(0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass
