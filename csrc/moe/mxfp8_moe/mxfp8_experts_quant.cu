// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cu

#include <torch/all.h>

#include "mxfp8_experts_quant.cuh"

void mxfp8_experts_quant(const torch::Tensor& input,
                         const torch::Tensor& problem_sizes,
                         const torch::Tensor& expert_offsets,
                         const torch::Tensor& blockscale_offsets,
                         torch::Tensor& quant_output,
                         torch::Tensor& scale_factor) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  TORCH_CHECK(input.dim() == 2, "input must be 2D tensor");
  TORCH_CHECK(input.size(1) % 128 == 0, "k must align to 128");
  TORCH_CHECK(input.strides()[1] == 1, "input must be row major");
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32,
              "problem_sizes must be int32");
  TORCH_CHECK(expert_offsets.dtype() == torch::kInt32,
              "expert_offsets must be int32");
  TORCH_CHECK(blockscale_offsets.dtype() == torch::kInt32,
              "blockscale_offsets must be int32");

  auto groups = problem_sizes.size(0);
  TORCH_CHECK(
      expert_offsets.dim() == 1 && expert_offsets.size(0) == groups,
      "expert_offsets must be 1D and have size equal to the number of groups");
  TORCH_CHECK(
      blockscale_offsets.dim() == 1 && blockscale_offsets.size(0) == groups,
      "blockscale_offsets must be 1D and have size equal to the number of "
      "groups");

  auto stream = at::cuda::getCurrentCUDAStream();
  if (input.dtype() == torch::kBFloat16) {
    expert_specialization::launch_mxfp8_experts_quant<__nv_bfloat16>(
        input, problem_sizes, expert_offsets, blockscale_offsets, quant_output,
        scale_factor);
  } else if (input.dtype() == torch::kFloat16) {
    expert_specialization::launch_mxfp8_experts_quant<__half>(
        input, problem_sizes, expert_offsets, blockscale_offsets, quant_output,
        scale_factor);
  } else {
    TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
#else
  TORCH_CHECK(false,
              "No implemented mxfp8_experts_quant for "
              "current device");
#endif
}

#include "core/registration.h"

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("mxfp8_experts_quant", mxfp8_experts_quant);
}