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
    \brief Reference implementation for GEMM in device-side code.
*/

#pragma once

#include "cutlass/coord.h"

#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/tensor_view.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/util/reference/device/kernel/gemm.h"

namespace cutlass {
namespace reference {
namespace device {

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
  typename AccumulatorType,
  typename InnerProductOp = multiply_add<AccumulatorType>,
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
  AccumulatorType initial_accum) {

  static_assert(
    LayoutA::kRank == 2 &&
    LayoutB::kRank == 2 &&
    LayoutC::kRank == 2, "Tensors must be of rank 2");

  // Blocking structure potentially improves performance of reference implementation
  // with a minor increase in complexity.
  //
  // Note, this reference implementation is NOT expected to approach peak performance.
  using OutputTile = MatrixShape<4, 4>;

  dim3 block(16, 8);

  dim3 grid(
    (problem_size.m() + block.x * OutputTile::kRow - 1) / (block.x * OutputTile::kRow),
    (problem_size.n() + block.y * OutputTile::kColumn - 1) / (block.y * OutputTile::kColumn)
  );

  // Launch a GEMM kernel
  kernel::Gemm<
    TensorRef<ElementA, LayoutA>,
    TensorRef<ElementB, LayoutB>,
    TensorRef<ElementC, LayoutC>,
    ScalarType,
    AccumulatorType,
    OutputTile,
    InnerProductOp,
    ConvertOp
  ><<< grid, block >>>(
    problem_size,
    alpha,
    tensor_a,
    tensor_b,
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
  typename ScalarType,
  typename AccumulatorType,
  typename InnerProductOp = multiply_add<AccumulatorType>,
  typename ConvertOp = NumericConverter<ElementC, ScalarType>
>
void compute_gemm(
  gemm::GemmCoord problem_size,
  ScalarType alpha,
  TensorRef<ElementA, LayoutA> tensor_a,
  TensorRef<ElementB, LayoutB> tensor_b,
  ScalarType beta,
  TensorRef<ElementC, LayoutC> tensor_c,
  AccumulatorType initial_accum) {

  compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                ScalarType, AccumulatorType, InnerProductOp, ConvertOp>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_c,
        initial_accum);
}

template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ScalarType,
  typename AccumulatorType,
  typename InnerProductOp = cutlass::arch::OpMultiplyAdd
>
struct Gemm;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for multiply-add
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ScalarType, typename AccumulatorType>
struct Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
            ScalarType, AccumulatorType, arch::OpMultiplyAdd> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  AccumulatorType initial_accum = AccumulatorType(0)) {

    static_assert(
      LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
      "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                  ScalarType, AccumulatorType, multiply_add<AccumulatorType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, initial_accum);
  }

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  TensorRef<ElementC, LayoutC> tensor_d,
                  AccumulatorType initial_accum = AccumulatorType(0)) {
    static_assert(
      LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
      "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                ScalarType, AccumulatorType, multiply_add<AccumulatorType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_d, initial_accum);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for multiply-add-saturate
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ScalarType, typename AccumulatorType>
struct Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ScalarType,
            AccumulatorType, arch::OpMultiplyAddSaturate> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  AccumulatorType initial_accum = AccumulatorType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, AccumulatorType, multiply_add<AccumulatorType>,
                 NumericConverterClamp<ElementC, ScalarType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, initial_accum);
  }

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  TensorRef<ElementC, LayoutC> tensor_d,
                  AccumulatorType initial_accum = AccumulatorType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, AccumulatorType, multiply_add<AccumulatorType>,
                 NumericConverterClamp<ElementC, ScalarType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_d, initial_accum);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for XOR-popc
template <typename ElementA, typename LayoutA, typename ElementB,
          typename LayoutB, typename ElementC, typename LayoutC,
          typename ScalarType, typename AccumulatorType>
struct Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ScalarType,
            AccumulatorType, arch::OpXorPopc> {

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  AccumulatorType initial_accum = AccumulatorType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, AccumulatorType, xor_add<AccumulatorType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, initial_accum);
  }

  void operator()(gemm::GemmCoord problem_size, ScalarType alpha,
                  TensorRef<ElementA, LayoutA> tensor_a,
                  TensorRef<ElementB, LayoutB> tensor_b, ScalarType beta,
                  TensorRef<ElementC, LayoutC> tensor_c,
                  TensorRef<ElementC, LayoutC> tensor_d,
                  AccumulatorType initial_accum = AccumulatorType(0)) {
    static_assert(
        LayoutA::kRank == 2 && LayoutB::kRank == 2 && LayoutC::kRank == 2,
        "Tensors must be of rank 2");

    compute_gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                 ScalarType, AccumulatorType, xor_add<AccumulatorType>>(
        problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, tensor_d, initial_accum);
  }
};


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
  typename AccumulatorType,
  typename InnerProductOp,
  typename ConvertOp
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

  static_assert(
    TensorRefCollectionA::kRank == 2 &&
    TensorRefCollectionB::kRank == 2 &&
    TensorRefCollectionC::kRank == 2, "Tensors must be of rank 2");

  // Blocking structure potentially improves performance of reference implementation
  // with a minor increase in complexity.
  //
  // Note, this reference implementation is NOT expected to approach peak performance.
  using OutputTile = MatrixShape<4, 4>;

  dim3 block(16, 8);
  dim3 grid(
    (problem_size.m() + block.x * OutputTile::kRow - 1) / (block.x * OutputTile::kRow),
    (problem_size.n() + block.y * OutputTile::kColumn - 1) / (block.y * OutputTile::kColumn),
    batch_count
  );

  // Launch a GEMM kernel
  kernel::BatchedGemm<
    TensorRefCollectionA,
    TensorRefCollectionB,
    TensorRefCollectionC,
    ScalarType,
    AccumulatorType,
    OutputTile,
    InnerProductOp,
    ConvertOp
  ><<< grid, block >>>(
    problem_size,
    alpha,
    tensor_a,
    tensor_b,
    beta,
    tensor_c,
    initial_accum
  );
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

  BatchedGemm(problem_size, alpha, tensor_a, tensor_b, beta, tensor_c, ScalarType(0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace reference
} // namespace cutlass
