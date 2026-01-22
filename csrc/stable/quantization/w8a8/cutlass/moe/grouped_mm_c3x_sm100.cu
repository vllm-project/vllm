#include <cudaTypedefs.h>

#include "stable/torch_utils.h"

#include "cutlass/cutlass.h"
#include "grouped_mm_c3x.cuh"

using namespace cute;

namespace {

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_config_default {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ArchTag = cutlass::arch::Sm100;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_config_M64 {
  // M in [1,64]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using TileShape = cute::Shape<cute::_128, cute::_16, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ArchTag = cutlass::arch::Sm100;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule,
                            true>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_config_N8192 {
  // N in [8192, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_128>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using ArchTag = cutlass::arch::Sm100;

  using Cutlass3xGemm =
      cutlass_3x_group_gemm<InType, OutType, ArchTag, Epilogue, TileShape,
                            ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType>
void run_cutlass_moe_mm_sm100(torch::stable::Tensor& out_tensors,
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

  using Cutlass3xGemmDefault = typename sm100_fp8_config_default<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmN8192 = typename sm100_fp8_config_N8192<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemmM64 = typename sm100_fp8_config_M64<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

  uint32_t const m = a_tensors.size(0);
  uint32_t const n = out_tensors.size(1);

  if (m <= 64) {
    cutlass_group_gemm_caller<Cutlass3xGemmM64>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else if (n >= 8192) {
    cutlass_group_gemm_caller<Cutlass3xGemmN8192>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else {
    cutlass_group_gemm_caller<Cutlass3xGemmDefault>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  }
}
}  // namespace

void dispatch_moe_mm_sm100(torch::stable::Tensor& out_tensors,
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
    run_cutlass_moe_mm_sm100<cutlass::float_e4m3_t, cutlass::bfloat16_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  } else {
    run_cutlass_moe_mm_sm100<cutlass::float_e4m3_t, cutlass::half_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides, per_act_token,
        per_out_ch);
  }
}

void cutlass_moe_mm_sm100(torch::stable::Tensor& out_tensors,
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
  dispatch_moe_mm_sm100(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                        expert_offsets, problem_sizes, a_strides, b_strides,
                        c_strides, per_act_token, per_out_ch);
}
