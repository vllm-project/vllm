#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass_extensions/common.hpp"

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90

void cutlass_moe_mm_sm90_8_bit(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides);

void cutlass_moe_mm_sm90_16_bit(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides);

void get_cutlass_moe_mm_data_caller(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k);

#endif

void cutlass_moe_mm(torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
                    torch::Tensor const& b_tensors,
                    std::optional<torch::Tensor> const& a_scales,
                    std::optional<torch::Tensor> const& b_scales,
                    torch::Tensor const& expert_offsets,
                    torch::Tensor const& problem_sizes,
                    torch::Tensor const& a_strides,
                    torch::Tensor const& b_strides,
                    torch::Tensor const& c_strides) {
  int32_t version_num = get_sm_version_num();
#if defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90
  if (a_tensors.dtype() == torch::kBFloat16 ||
      a_tensors.dtype() == torch::kFloat16) {
    TORCH_CHECK(!a_scales.has_value());
    TORCH_CHECK(!b_scales.has_value());
    cutlass_moe_mm_sm90_16_bit(out_tensors, a_tensors, b_tensors,
                               expert_offsets, problem_sizes, a_strides,
                               b_strides, c_strides);
  } else {
    TORCH_CHECK(a_scales.has_value());
    TORCH_CHECK(b_scales.has_value());
    cutlass_moe_mm_sm90_8_bit(
        out_tensors, a_tensors, b_tensors, a_scales.value(), b_scales.value(),
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides);
  }
  return;
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm for CUDA device capability: ", version_num,
      ". Required capability: 90");
}

void get_cutlass_moe_mm_data(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k) {
  // This function currently gets compiled only if we have a valid cutlass moe
  // mm to run it for.
  int32_t version_num = get_sm_version_num();
#if defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90
  get_cutlass_moe_mm_data_caller(topk_ids, expert_offsets, problem_sizes1,
                                 problem_sizes2, input_permutation,
                                 output_permutation, num_experts, n, k);
  return;
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled get_cutlass_moe_mm_data: no cutlass_scaled_mm kernel for "
      "CUDA device capability: ",
      version_num, ". Required capability: 90");
}
