// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled.cu

#include <torch/all.h>

#include "cutlass_mxfp8_grouped_mm_launcher.cuh"

void cutlass_mxfp8_grouped_mm(const torch::Tensor& a, const torch::Tensor& b,
                              const torch::Tensor& sfa,
                              const torch::Tensor& sfb, torch::Tensor& d,
                              const torch::Tensor& problem_sizes,
                              const torch::Tensor& expert_offsets,
                              const torch::Tensor& blockscale_offsets) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3,
              "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(problem_sizes.size(0) == expert_offsets.size(0),
              "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32,
              "problem_sizes must be int32");
  TORCH_CHECK(expert_offsets.dtype() == torch::kInt32,
              "expert_offsets must be int32");
  TORCH_CHECK(blockscale_offsets.dtype() == torch::kInt32,
              "blockscale_offsets must be int32");
  TORCH_CHECK(a.dim() == 2, "a must be a 2D tensor of shape (num_tokens, k)");
  TORCH_CHECK(b.dim() == 3,
              "b must be a 3D tensor of shape (num_experts, k, n)");
  TORCH_CHECK(a.size(1) == b.size(1) && a.size(1) % 128 == 0,
              "k should align 128");
  TORCH_CHECK(b.size(2) % 128 == 0, "n should align 128");
  TORCH_CHECK(a.strides()[1] == 1, "a must be row major");
  TORCH_CHECK(b.strides()[1] == 1, "b must be column major");

  auto stream = at::cuda::getCurrentCUDAStream();
  if (d.dtype() == torch::kBFloat16) {
    expert_specialization::cutlass_mxfp8_grouped_mm_dispatch_out_dtype<
        cutlass::bfloat16_t>(a, b, sfa, sfb, d, problem_sizes, expert_offsets,
                             blockscale_offsets, stream);
  } else if (d.dtype() == torch::kFloat16) {
    expert_specialization::cutlass_mxfp8_grouped_mm_dispatch_out_dtype<
        cutlass::half_t>(a, b, sfa, sfb, d, problem_sizes, expert_offsets,
                         blockscale_offsets, stream);
  } else {
    TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
#else
  TORCH_CHECK(false,
              "No implemented cutlass_mxfp8_grouped_mm for "
              "current device");
#endif
}

#include "core/registration.h"

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("cutlass_mxfp8_grouped_mm", cutlass_mxfp8_grouped_mm);
}