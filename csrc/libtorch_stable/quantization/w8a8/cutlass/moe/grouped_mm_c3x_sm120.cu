#include <cudaTypedefs.h>

#include "libtorch_stable/torch_utils.h"
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "cutlass/cutlass.h"
#include "grouped_mm_c3x.cuh"

using namespace cute;

namespace {

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm120_fp8_config_default {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  // SM120 / SM121 (consumer Blackwell / DGX Spark) FP8 grouped GEMM with
  // tensor/token-level FP8 scaling (scaling done in epilogue, not mainloop).
  // Uses the SM120 dense ptr-array schedule (NOT the Blockwise variant —
  // Blockwise expects per-block scale factors in the mainloop layout,
  // which our model does not provide).
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeSm120<2>;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ArchTag = cutlass::arch::Sm120;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType>
void run_cutlass_moe_mm_sm120(torch::stable::Tensor& out_tensors,
                              torch::stable::Tensor const& a_tensors,
                              torch::stable::Tensor const& b_tensors,
                              torch::stable::Tensor const& a_scales,
                              torch::stable::Tensor const& b_scales,
                              torch::stable::Tensor const& expert_offsets,
                              torch::stable::Tensor const& problem_sizes,
                              torch::stable::Tensor const& a_strides,
                              torch::stable::Tensor const& b_strides,
                              torch::stable::Tensor const& c_strides,
                              bool per_act_token, bool per_out_ch) {
  STD_TORCH_CHECK(a_tensors.size(0) > 0, "No input A tensors provided.");
  STD_TORCH_CHECK(b_tensors.size(0) > 0, "No input B tensors provided.");
  STD_TORCH_CHECK(out_tensors.size(0) > 0, "No output tensors provided.");

  STD_TORCH_CHECK(
      a_tensors.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
      "A tensors must be of type float8_e4m3fn.");
  STD_TORCH_CHECK(
      b_tensors.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
      "B tensors must be of type float8_e4m3fn.");

  // Single-config dispatch for v1: prove the SM120 CUTLASS path runs
  // correctly before adding M-bucket or N-bucket specializations.
  using Cutlass3xGemmDefault = typename sm120_fp8_config_default<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

  cutlass_group_gemm_caller<Cutlass3xGemmDefault>(
      out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
      problem_sizes, a_strides, b_strides, c_strides, per_act_token,
      per_out_ch);
}
}  // namespace

void dispatch_moe_mm_sm120(torch::stable::Tensor& out_tensors,
                           torch::stable::Tensor const& a_tensors,
                           torch::stable::Tensor const& b_tensors,
                           torch::stable::Tensor const& a_scales,
                           torch::stable::Tensor const& b_scales,
                           torch::stable::Tensor const& expert_offsets,
                           torch::stable::Tensor const& problem_sizes,
                           torch::stable::Tensor const& a_strides,
                           torch::stable::Tensor const& b_strides,
                           torch::stable::Tensor const& c_strides,
                           bool per_act_token, bool per_out_ch) {
  if (out_tensors.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    run_cutlass_moe_mm_sm120<cutlass::float_e4m3_t, cutlass::bfloat16_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else {
    run_cutlass_moe_mm_sm120<cutlass::float_e4m3_t, cutlass::half_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  }
}

void cutlass_moe_mm_sm120(torch::stable::Tensor& out_tensors,
                          torch::stable::Tensor const& a_tensors,
                          torch::stable::Tensor const& b_tensors,
                          torch::stable::Tensor const& a_scales,
                          torch::stable::Tensor const& b_scales,
                          torch::stable::Tensor const& expert_offsets,
                          torch::stable::Tensor const& problem_sizes,
                          torch::stable::Tensor const& a_strides,
                          torch::stable::Tensor const& b_strides,
                          torch::stable::Tensor const& c_strides,
                          bool per_act_token, bool per_out_ch) {
  dispatch_moe_mm_sm120(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                        expert_offsets, problem_sizes, a_strides, b_strides,
                        c_strides, per_act_token, per_out_ch);
}
