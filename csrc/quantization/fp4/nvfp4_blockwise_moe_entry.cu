#include <torch/all.h>

#if defined ENABLE_NVFP4_SM100 && ENABLE_NVFP4_SM100
void cutlass_fp4_group_mm_sm100(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_blockscale, const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets);
#endif

void cutlass_fp4_group_mm(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_blockscale, const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets) {
#if defined ENABLE_NVFP4_SM100 && ENABLE_NVFP4_SM100
  cutlass_fp4_group_mm_sm100(output, a, b, a_blockscale, b_blockscales, alphas,
                             problem_sizes, expert_offsets, sf_offsets);
#else
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_fp4_group_mm kernel, vLLM must "
      "be compiled with ENABLE_NVFP4_SM100 for SM100+ and CUDA "
      "12.8 or above.");
#endif
}