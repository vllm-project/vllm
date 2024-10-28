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
    \brief Reference implementation for complex-valued GEMM in device-side code.
*/

#pragma once

#include "cutlass/coord.h"
#include "cutlass/complex.h"
#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/tensor_view.h"
#include "cutlass/gemm/gemm.h"

namespace cutlass {
namespace reference {
namespace device {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace kernel {

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
  typename InnerProductOp = multiply_add<ComputeType>,
  int kMblock = 4,
  int kNblock = 4
>
__global__ void GemmComplex(
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

  int const M = problem_size.m();
  int const N = problem_size.n();
  int const K = problem_size.k();

  ConvertOp convert_op;
  InnerProductOp inner_product_op;
  
  int row_block = (blockIdx.x * blockDim.x + threadIdx.x) * kMblock;
  int col_block = (blockIdx.y * blockDim.y + threadIdx.y) * kNblock; 
  int batch_idx = blockIdx.z;

  tensor_a.add_pointer_offset(batch_idx * batch_stride_A);
  tensor_b.add_pointer_offset(batch_idx * batch_stride_B);
  tensor_c.add_pointer_offset(batch_idx * batch_stride_C);
  tensor_d.add_pointer_offset(batch_idx * batch_stride_D);

  for (; batch_idx < batch_count; batch_idx += gridDim.z) {

    // Compute matrix product using blocks
    ComputeType accum[kMblock][kNblock];

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kNblock; j++) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kMblock; i++) {
        accum[i][j] = initial_accum;
      }
    }

    for (int k_block = 0; k_block < K; ++k_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kNblock; j++) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kMblock; i++) {
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

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kNblock; j++) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kMblock; i++) {
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

    tensor_a.add_pointer_offset(batch_stride_A * gridDim.z);
    tensor_b.add_pointer_offset(batch_stride_B * gridDim.z);
    tensor_c.add_pointer_offset(batch_stride_C * gridDim.z);
    tensor_d.add_pointer_offset(batch_stride_D * gridDim.z);

  } // for (batch_idx)
}

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
 
  int const kMblock = 4;
  int const kNblock = 4;

  dim3 block(16, 8);
  dim3 grid(
    (problem_size.m() + block.x * kMblock - 1) / (block.x * kMblock),
    (problem_size.n() + block.y * kNblock - 1) / (block.y * kNblock),
    batch_count % std::numeric_limits<uint16_t>::max()
  );

  if (grid.y <= std::numeric_limits<uint16_t>::max()) {
    kernel::GemmComplex<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ScalarType,
      ComputeType,
      ElementD,
      ConvertOp,
      InnerProductOp,
      kMblock,
      kNblock
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
      initial_accum,
      batch_count,
      batch_stride_A,
      batch_stride_B,
      batch_stride_C,
      batch_stride_D
    );
  } else {
    // Using bigger thread tile size
    int const kBigMblock = 4;
    int const kBigNblock = 16;

    dim3 Bigblock(16, 8);
    dim3 Biggrid(
      (problem_size.m() + block.x * kBigMblock - 1) / (block.x * kBigMblock),
      (problem_size.n() + block.y * kBigNblock - 1) / (block.y * kBigNblock),
      batch_count % std::numeric_limits<uint16_t>::max()
    );

    kernel::GemmComplex<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ScalarType,
      ComputeType,
      ElementD,
      ConvertOp,
      InnerProductOp,
      kBigMblock,
      kBigNblock
    ><<< Biggrid, Bigblock >>>(
      problem_size,
      alpha,
      tensor_a,
      transform_a,
      tensor_b,
      transform_b,
      beta,
      tensor_c,
      tensor_d,
      initial_accum,
      batch_count,
      batch_stride_A,
      batch_stride_B,
      batch_stride_C,
      batch_stride_D
    );
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

} // namespace device
} // namespace reference
} // namespace cutlass
