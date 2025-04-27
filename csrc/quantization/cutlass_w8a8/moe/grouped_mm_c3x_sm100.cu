#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "grouped_mm_c3x_sm100.cuh"

using namespace cute;

namespace {

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_1sm_config {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = cute::Shape<_128, _256, _128>;
  using ClusterShape = cute::Shape<_2, _2, _1>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;  // Kernel to
                                                                // launch
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;  // Epilogue to launch
  using Cutlass3xGemm =
      cutlass_3x_group_gemm_sm100<InType, OutType, Epilogue, TileShape,
                                  ClusterShape, KernelSchedule,
                                  EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_2sm_config {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = Shape<_256, _256, _128>;
  using ClusterShape = Shape<_4, _2, _1>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;  // Kernel to
                                                                // launch
  using EpilogueSchedule =
      cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;  // Epilogue to launch
  using Cutlass3xGemm =
      cutlass_3x_group_gemm_sm100<InType, OutType, Epilogue, TileShape,
                                  ClusterShape, KernelSchedule,
                                  EpilogueSchedule>;
};

template <typename InType, typename OutType>
void run_cutlass_moe_mm_sm100(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  TORCH_CHECK(a_tensors.size(0) > 0, "No input A tensors provided.");
  TORCH_CHECK(b_tensors.size(0) > 0, "No input B tensors provided.");
  TORCH_CHECK(out_tensors.size(0) > 0, "No output tensors provided.");

  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn,
              "A tensors must be of type float8_e4m3fn.");
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn,
              "B tensors must be of type float8_e4m3fn.");

  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn);

  using Cutlass3xGemm1sm = typename sm100_fp8_1sm_config<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;
  using Cutlass3xGemm2sm = typename sm100_fp8_2sm_config<
      InType, OutType, vllm::c3x::ScaledEpilogueArray>::Cutlass3xGemm;

  uint32_t const m = a_tensors.size(0);
  uint32_t const n = out_tensors.size(1);
  uint32_t const k = a_tensors.size(1);

  // TODO: add 2 sm config (@kushanam)
  cutlass_group_gemm_caller<Cutlass3xGemm1sm>(
      out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
      problem_sizes, a_strides, b_strides, c_strides);
}

void dispatch_moe_mm_sm100(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  if (out_tensors.dtype() == torch::kBFloat16) {
    run_cutlass_moe_mm_sm100<cutlass::float_e4m3_t, cutlass::bfloat16_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides);
  } else {
    run_cutlass_moe_mm_sm100<cutlass::float_e4m3_t, cutlass::half_t>(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
        problem_sizes, a_strides, b_strides, c_strides);
  }
}

}  // namespace

void cutlass_moe_mm_sm100(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  dispatch_moe_mm_sm100(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                        expert_offsets, problem_sizes, a_strides, b_strides,
                        c_strides);
}
