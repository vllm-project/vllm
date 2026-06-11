// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cu

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/torch_utils.h"

#include "mxfp8_experts_quant.cuh"

void mxfp8_experts_quant(const torch::stable::Tensor& input,
                         const torch::stable::Tensor& problem_sizes,
                         const torch::stable::Tensor& expert_offsets,
                         const torch::stable::Tensor& blockscale_offsets,
                         torch::stable::Tensor& quant_output,
                         torch::stable::Tensor& scale_factor) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  STD_TORCH_CHECK(input.dim() == 2, "input must be 2D tensor");
  STD_TORCH_CHECK(input.size(1) % 128 == 0, "k must align to 128");
  STD_TORCH_CHECK(input.stride(1) == 1, "input must be row major");
  STD_TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  STD_TORCH_CHECK(
      problem_sizes.scalar_type() == torch::headeronly::ScalarType::Int,
      "problem_sizes must be int32");
  STD_TORCH_CHECK(
      expert_offsets.scalar_type() == torch::headeronly::ScalarType::Int,
      "expert_offsets must be int32");
  STD_TORCH_CHECK(
      blockscale_offsets.scalar_type() == torch::headeronly::ScalarType::Int,
      "blockscale_offsets must be int32");

  auto groups = problem_sizes.size(0);
  STD_TORCH_CHECK(
      expert_offsets.dim() == 1 && expert_offsets.size(0) == groups,
      "expert_offsets must be 1D and have size equal to the number of groups");
  STD_TORCH_CHECK(
      blockscale_offsets.dim() == 1 && blockscale_offsets.size(0) == groups,
      "blockscale_offsets must be 1D and have size equal to the number of "
      "groups");

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  if (input.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    expert_specialization::launch_mxfp8_experts_quant<__nv_bfloat16>(
        input, problem_sizes, expert_offsets, blockscale_offsets, quant_output,
        scale_factor);
  } else if (input.scalar_type() == torch::headeronly::ScalarType::Half) {
    expert_specialization::launch_mxfp8_experts_quant<__half>(
        input, problem_sizes, expert_offsets, blockscale_offsets, quant_output,
        scale_factor);
  } else {
    STD_TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
#else
  STD_TORCH_CHECK(false,
                  "No implemented mxfp8_experts_quant for "
                  "current device");
#endif
}

// Registered here (not torch_bindings.cpp) because ENABLE_ES_MXFP8_GROUPED_MM
// is applied only under COMPILE_LANGUAGE:CUDA.
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("mxfp8_experts_quant", TORCH_BOX(&mxfp8_experts_quant));
}
