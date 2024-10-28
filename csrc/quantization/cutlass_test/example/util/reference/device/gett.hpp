/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief GETT device reference code
*/
#pragma once

#include <cute/tensor.hpp>

namespace cutlass::reference::device {

template <
  class ATensor,
  class BTensor,
  class CTensor,
  class DTensor,
  class ElementAccumulator,
  class ElementEpilogue>
__global__ static
void
gett_kernel(
  DTensor       D,
  ATensor const A,
  BTensor const B,
  CTensor const C,
  ElementEpilogue alpha, ElementEpilogue beta,
  ElementAccumulator acc_init)
{
  using namespace cute;

  static_assert(DTensor::rank == 3, "(M,N,L)");
  static_assert(ATensor::rank == 3, "(M,K,L)");
  static_assert(BTensor::rank == 3, "(N,K,L)");
  static_assert(CTensor::rank == 3, "(M,N,L)");

  assert(size<0>(A) == size<0>(D));  // M
  assert(size<0>(C) == size<0>(D));  // M
  assert(size<0>(B) == size<1>(D));  // N
  assert(size<1>(C) == size<1>(D));  // N
  assert(size<1>(A) == size<1>(B));  // K
  assert(size<2>(A) == size<2>(D));  // L
  assert(size<2>(B) == size<2>(D));  // L
  assert(size<2>(C) == size<2>(D));  // L

  NumericConverter<ElementAccumulator, typename ATensor::value_type> a_converter;
  NumericConverter<ElementAccumulator, typename BTensor::value_type> b_converter;
  NumericConverter<ElementEpilogue, ElementAccumulator> acc_converter;
  NumericConverter<ElementEpilogue, typename CTensor::value_type> source_converter;
  NumericConverter<typename DTensor::value_type, ElementEpilogue> output_converter;

  // Thread id to each element of D
  for (int tid = threadIdx.x + blockDim.x * blockIdx.x;
       tid < size(D);
       tid += blockDim.x * gridDim.x) {
    // (m,n,l) coordinate
    auto mnl_coord = idx2crd(tid, product_each(shape(D)));
    auto m = get<0>(mnl_coord);
    auto n = get<1>(mnl_coord);
    auto l = get<2>(mnl_coord);

    auto A_ml = A(m,_,l);
    auto B_nl = B(n,_,l);

    ElementAccumulator accum = ElementAccumulator(0);
    for (int k = 0; k < size<1>(A); ++k) {
      ElementAccumulator a = a_converter(A_ml(k));
      ElementAccumulator b = b_converter(B_nl(k));
      accum += a * b;
    }

    ElementEpilogue scaled_output = (alpha * acc_converter(accum)) + (beta * source_converter(C(m,n,l)));
    D(m,n,l) = output_converter(scaled_output);
  }
}

// Most general version
template <
  class ProblemShapeMNKL,
  class ElementA,
  class StrideA,
  class ElementB,
  class StrideB,
  class ElementAccumulator,
  class ElementC,
  class StrideC,
  class ElementD,
  class StrideD,
  class ElementEpilogue>
void
gett(
    ProblemShapeMNKL problem_shape_mnkl,
    ElementA const* ptr_A, StrideA stride_a_mkl,
    ElementB const* ptr_B, StrideB stride_b_nkl,
    ElementAccumulator _,
    ElementC const* ptr_C, StrideC stride_c_mnl,
    ElementD      * ptr_D, StrideD stride_d_mnl,
    ElementEpilogue alpha, ElementEpilogue beta,
    cudaStream_t stream = 0) {
  using namespace cute;

  static_assert(cute::rank(ProblemShapeMNKL{}) == 4);
  auto M = get<0>(problem_shape_mnkl);
  auto N = get<1>(problem_shape_mnkl);
  auto K = get<2>(problem_shape_mnkl);
  auto L = get<3>(problem_shape_mnkl);

  // Represent the full tensors
  auto A = make_tensor(make_gmem_ptr(ptr_A), make_shape(M,K,L), stride_a_mkl); // (M,K,L)
  auto B = make_tensor(make_gmem_ptr(ptr_B), make_shape(N,K,L), stride_b_nkl); // (N,K,L)
  auto C = make_tensor(make_gmem_ptr(ptr_C), make_shape(M,N,L), stride_c_mnl); // (M,N,L)
  auto D = make_tensor(make_gmem_ptr(ptr_D), make_shape(M,N,L), stride_d_mnl); // (M,N,L)

  dim3 dimBlock(256);
  dim3 dimGrid(240);
  gett_kernel<<< dimGrid, dimBlock, 0, stream >>>(D, A, B, C, alpha, beta, ElementAccumulator(0));
}

} // namespace cutlass::reference::device
