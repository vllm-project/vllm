#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "grouped_mm_c3x_with_blockwise_scaling.cuh"

using namespace cute;

namespace vllm::cutlass_moe::blockwise_scaling {

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>

// TODO: figure out the best configs
struct sm90_fp8_config_default {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::
      KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_blockwise_group_gemm<InType, OutType, Epilogue, TileShape,
                                      ClusterShape, KernelSchedule,
                                      EpilogueSchedule>;
};

template <typename InType, typename OutType>
void run_cutlass_moe_blockwise_mm_sm90(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_block) {
  TORCH_CHECK(a_tensors.size(0) > 0, "No input A tensors provided.");
  TORCH_CHECK(b_tensors.size(0) > 0, "No input B tensors provided.");
  TORCH_CHECK(out_tensors.size(0) > 0, "No output tensors provided.");

  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn,
              "A tensors must be of type float8_e4m3fn.");
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn,
              "B tensors must be of type float8_e4m3fn.");

  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn);

  using Cutlass3xGemmDefault = typename sm90_fp8_config_default<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

  uint32_t const m = a_tensors.size(0);
  uint32_t const n = out_tensors.size(1);
  uint32_t const k = a_tensors.size(1);

  cutlass_blockwise_group_gemm_caller<Cutlass3xGemmDefault>(
      out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
      problem_sizes, a_strides, b_strides, c_strides, per_act_block);
}

void dispatch_moe_blockwise_mm_sm90(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_block) {
  if (out_tensors.dtype() == torch::kBFloat16) {
    run_cutlass_moe_blockwise_mm_sm90<cutlass::float_e4m3_t,
                                      cutlass::bfloat16_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_block);
  } else {
    run_cutlass_moe_blockwise_mm_sm90<cutlass::float_e4m3_t, cutlass::half_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_block);
  }
}

}  // namespace vllm::cutlass_moe::blockwise_scaling

void cutlass_moe_blockwise_mm_sm90(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_block) {
  vllm::cutlass_moe::blockwise_scaling::dispatch_moe_blockwise_mm_sm90(
      out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
      problem_sizes, a_strides, b_strides, c_strides, per_act_block);
}