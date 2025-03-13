/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group
using ElementInput = cutlass::float_e2m1_t;                                // Element type for Input matrix operands
using ElementSF    = cutlass::float_ue4m3_t;                               // Element type for SF matrix operands
// using ElementC     = cutlass::half_t;                                      // Element type for Output matrix operands









#else

void runGroupedGemm(at::Tensor& gC, at::Tensor& at::Tensor& gA, at::Tensor& gB,
 at::Tensor& ) {
  TORCH_CHECK(false,
              "Unsupported Configuration for CUTLASS based FP4 Grouped Gemm."
              "Please use Blackwell and above to compile NVFP4 Kernels.");
}


#endif 


torch::



torch::Tensor fp4_moe_gemm(
  const torch::Tensor& out, 
  const torch::Tensor& a_fp4_inputs, const torch::Tensor& b_fp4_weights,
  const torch::Tensor& a_blockscales, const torch::Tensor& b_blockscales,
  const torch::Tensor& alphas, const torch::Tensor& sorted_ids, 
  torch::Tensor& topk_weights, torch::Tensor& topk_ids){





}

