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
    \brief Reference implementation for complex-valued GEMM in device code.
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/complex.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_ref_planar_complex.h"

#include "cutlass/tensor_view.h"
#include "cutlass/gemm/gemm.h"

namespace cutlass {
namespace reference {
namespace device {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

////////////////////////////////////////////////////////////////////////////////////////////////////

static int const kGemmPlanarComplexBlockSize = 4;

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
__global__ void GemmPlanarComplex(
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

  int const kMblock = kGemmPlanarComplexBlockSize;
  int const kNblock = kGemmPlanarComplexBlockSize;

  using ComplexA = typename TensorRefPlanarComplex<ElementA, LayoutA>::ComplexElement;
  using ComplexB = typename TensorRefPlanarComplex<ElementB, LayoutB>::ComplexElement;
  using ComplexC = typename TensorRefPlanarComplex<ElementC, LayoutC>::ComplexElement;

  // Note: batch is ignored.
  int const M = problem_size.m();
  int const N = problem_size.n();
  int const K = problem_size.k();

  ConvertOp convert_op;
  InnerProductOp inner_product_op;

  complex<ComputeType> accum[kMblock][kNblock];
  
  int row_block = (blockIdx.x * blockDim.x + threadIdx.x) * kMblock;
  int col_block = (blockIdx.y * blockDim.y + threadIdx.y) * kNblock; 

  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kNblock; j++) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kMblock; i++) {
      accum[i][j] = initial_accum;
    }
  }

  CUTLASS_PRAGMA_NO_UNROLL
  for (int k_block = 0; k_block < K; ++k_block) {

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kNblock; j++) {

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kMblock; i++) {

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

  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < kNblock; j++) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kMblock; i++) {

      int row = row_block + i;
      int col = col_block + j;

      MatrixCoord coord = MatrixCoord(row, col);

      if (row < M && col < N) {

        complex<ScalarType> acc{
          ScalarType(accum[i][j].real()),
          ScalarType(accum[i][j].imag())
        };

        ComplexC c_ij = ComplexC();

        if (beta.real() != ScalarType() || beta.imag() != ScalarType()) {
          c_ij = tensor_c.at(coord);
        }

        complex<ScalarType> src{
          ScalarType(c_ij.real()),
          ScalarType(c_ij.imag())
        };

        complex<ScalarType> result = alpha * acc + beta * src;

        ComplexC d_ij;

        d_ij.real() = convert_op(result.real());
        d_ij.imag() = convert_op(result.imag());

        tensor_d.at(coord) = d_ij;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel

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

  int const kMblock = kernel::kGemmPlanarComplexBlockSize;
  int const kNblock = kernel::kGemmPlanarComplexBlockSize;

  dim3 block(16, 8);

  dim3 grid(
    (problem_size.m() + block.x * kMblock - 1) / (block.x * kMblock),
    (problem_size.n() + block.y * kNblock - 1) / (block.y * kNblock),
    1);

  kernel::GemmPlanarComplex<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ScalarType,
    ComputeType,
    ConvertOp,
    InnerProductOp
  ><<< grid, block >>>(
    problem_size,
    alpha,
    tensor_a,
    transform_a,
    tensor_b,
    transform_b,
    beta,    
    tensor_c,
    tensor_d,
    initial_accum
  );
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

} // namespace device
} // namespace reference
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
