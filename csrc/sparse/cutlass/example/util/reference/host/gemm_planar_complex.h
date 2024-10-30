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
#include "cutlass/tensor_ref_planar_complex.h"

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
  typename ConvertOp = NumericConverter<ElementC, ScalarType>,
  typename InnerProductOp = multiply_add<complex<ComputeType>>
>
void GemmPlanarComplex(
  gemm::GemmCoord problem_size,
  complex<ScalarType> alpha,
  TensorRefPlanarComplex<ElementA, LayoutA> tensor_a,
  ComplexTransform transform_a,
  TensorRefPlanarComplex<ElementB, LayoutB> tensor_b,
  ComplexTransform transform_b,
  complex<ScalarType> beta,
  TensorRefPlanarComplex<ElementC, LayoutC> tensor_c,
  TensorRefPlanarComplex<ElementC, LayoutC> tensor_d,
  complex<ComputeType> initial_accum) {

  static_assert(
    LayoutA::kRank == 2 &&
    LayoutB::kRank == 2 &&
    LayoutC::kRank == 2, "Tensors must be of rank 2");

  using ComplexA = typename TensorRefPlanarComplex<ElementA, LayoutA>::ComplexElement;
  using ComplexB = typename TensorRefPlanarComplex<ElementB, LayoutB>::ComplexElement;
  using ComplexC = typename TensorRefPlanarComplex<ElementC, LayoutC>::ComplexElement;

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

      complex<ComputeType> accum[Mblock][Nblock];

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

              ComplexA a_ik = tensor_a.at(MatrixCoord(row, k_block));
              ComplexB b_kj = tensor_b.at(MatrixCoord(k_block, col));

              complex<ComputeType> a = complex<ComputeType>{
                ComputeType(a_ik.real()),
                ComputeType(a_ik.imag())
              };

              complex<ComputeType> b = complex<ComputeType>{
                ComputeType(b_kj.real()),
                ComputeType(b_kj.imag())
              };

              if (transform_a == ComplexTransform::kConjugate) {
                a = conj(a);
              }

              if (transform_b == ComplexTransform::kConjugate) {
                b = conj(b);
              }

              accum[i][j] = inner_product_op(a, b,  accum[i][j]);
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

            complex<ScalarType> acc{
              ScalarType(accum[i][j].real()),
              ScalarType(accum[i][j].imag())
            };

            ComplexC d_ij = tensor_c.at(coord);

            complex<ScalarType> src{
              ScalarType(d_ij.real()),
              ScalarType(d_ij.imag())
            };

            complex<ScalarType> result = alpha * acc + beta * src;

            d_ij.real() = convert_op(result.real());
            d_ij.imag() = convert_op(result.imag());

            tensor_d.at(coord) = d_ij;
          }
        }
      }
    }
  }
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
  typename ScalarType
>
void GemmPlanarComplex(
  gemm::GemmCoord problem_size,
  complex<ScalarType> alpha,
  TensorRefPlanarComplex<ElementA, LayoutA> tensor_a,
  ComplexTransform transform_a,
  TensorRefPlanarComplex<ElementB, LayoutB> tensor_b,
  ComplexTransform transform_b,
  complex<ScalarType> beta,
  TensorRefPlanarComplex<ElementC, LayoutC> tensor_c,
  TensorRefPlanarComplex<ElementC, LayoutC> tensor_d) {

  GemmPlanarComplex(
    problem_size, 
    alpha, 
    tensor_a, transform_a, 
    tensor_b, transform_b, 
    beta, 
    tensor_c,
    tensor_d,
    complex<ScalarType>());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace host
} // namespace reference
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
