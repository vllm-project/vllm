// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled.cu

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/torch_utils.h"

#include "cutlass_mxfp8_grouped_mm_launcher.cuh"

void cutlass_mxfp8_grouped_mm(const torch::stable::Tensor& a,
                              const torch::stable::Tensor& b,
                              const torch::stable::Tensor& sfa,
                              const torch::stable::Tensor& sfb,
                              torch::stable::Tensor& d,
                              const torch::stable::Tensor& problem_sizes,
                              const torch::stable::Tensor& expert_offsets,
                              const torch::stable::Tensor& blockscale_offsets) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  STD_TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  STD_TORCH_CHECK(problem_sizes.size(1) == 3,
                  "problem_sizes must have shape (num_experts, 3)");
  STD_TORCH_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0),
      "Number of experts in problem_sizes must match expert_offsets");
  STD_TORCH_CHECK(
      problem_sizes.scalar_type() == torch::headeronly::ScalarType::Int,
      "problem_sizes must be int32");
  STD_TORCH_CHECK(
      expert_offsets.scalar_type() == torch::headeronly::ScalarType::Int,
      "expert_offsets must be int32");
  STD_TORCH_CHECK(
      blockscale_offsets.scalar_type() == torch::headeronly::ScalarType::Int,
      "blockscale_offsets must be int32");
  STD_TORCH_CHECK(a.dim() == 2,
                  "a must be a 2D tensor of shape (num_tokens, k)");
  STD_TORCH_CHECK(b.dim() == 3,
                  "b must be a 3D tensor of shape (num_experts, k, n)");
  STD_TORCH_CHECK(a.size(1) == b.size(1) && a.size(1) % 128 == 0,
                  "k should align 128");
  STD_TORCH_CHECK(b.size(2) % 128 == 0, "n should align 128");
  STD_TORCH_CHECK(a.stride(1) == 1, "a must be row major");
  STD_TORCH_CHECK(b.stride(1) == 1, "b must be column major");

  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());
  auto stream = get_current_cuda_stream(a.get_device_index());
  if (d.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    expert_specialization::cutlass_mxfp8_grouped_mm_dispatch_out_dtype<
        cutlass::bfloat16_t>(a, b, sfa, sfb, d, problem_sizes, expert_offsets,
                             blockscale_offsets, stream);
  } else if (d.scalar_type() == torch::headeronly::ScalarType::Half) {
    expert_specialization::cutlass_mxfp8_grouped_mm_dispatch_out_dtype<
        cutlass::half_t>(a, b, sfa, sfb, d, problem_sizes, expert_offsets,
                         blockscale_offsets, stream);
  } else {
    STD_TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
#else
  STD_TORCH_CHECK(false,
                  "No implemented cutlass_mxfp8_grouped_mm for "
                  "current device");
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("cutlass_mxfp8_grouped_mm", TORCH_BOX(&cutlass_mxfp8_grouped_mm));
}
