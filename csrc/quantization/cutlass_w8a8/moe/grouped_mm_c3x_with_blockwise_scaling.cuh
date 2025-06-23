#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "cutlass_extensions/common.hpp"
#include "get_group_starts.cuh"

using namespace cute;

namespace vllm::cutlass_moe::blockwise_scaling {

using ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

template <typename ElementAB_, typename ElementC_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_3x_blockwise_group_gemm {
  using ElementAB = ElementAB_;
  using ElementC = void;
  using ElementD = ElementC_;
  using ElementAccumulator = float;
  using ElementScale = float;

  using Epilogue = Epilogue_<ElementAccumulator, ElementD, TileShape>;

  using StrideC =
      cute::remove_pointer_t<cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>>;

  // TODO should not be fixed
  static constexpr int ScaleGranularityM = 1;
  static constexpr int ScaleGranularityN = 128;
  static constexpr int ScaleGranularityK = 128;
  static constexpr int ScaleMsPerTile =
      size<0>(TileShape{}) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile =
      size<1>(TileShape{}) / ScaleGranularityN;

  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<
      ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;
  using LayoutSFA =
      decltype(ScaleConfig::deduce_layoutSFA());  // Layout type for SFA matrix
                                                  // operand
  using LayoutSFB =
      decltype(ScaleConfig::deduce_layoutSFB());  // Layout type for SFB matrix
                                                  // operand

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;

  using DefaultOperation =
      cutlass::epilogue::fusion::LinearCombination<ElementD,
                                                   ElementAccumulator>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutC*, AlignmentC, ElementD,
          LayoutC*, AlignmentC, EpilogueSchedule,
          DefaultOperation>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementAB, cute::tuple<LayoutA*, LayoutSFA*>,
          AlignmentAB, ElementAB, cute::tuple<LayoutB*, LayoutSFB*>, AlignmentAB,
          ElementAccumulator, TileShape, ClusterShape, Stages,
          KernelSchedule>::CollectiveOp;

  using KernelType = enable_sm90_only<cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};

template <typename Gemm>
void cutlass_blockwise_group_gemm_caller(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_block) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using ElementScale = typename Gemm::ElementScale;
  using ScaleConfig = typename Gemm::ScaleConfig;
  using LayoutSFA = typename Gemm::LayoutSFA;
  using LayoutSFB = typename Gemm::LayoutSFB;

  int num_experts = static_cast<int>(expert_offsets.size(0));
  int k_size = a_tensors.size(1);
  int n_size = out_tensors.size(1);
  int n_scale_size = b_scales.size(2);
  int k_scale_size = b_scales.size(1);

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  auto options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(a_tensors.device());
  auto options_int32 =
      torch::TensorOptions().dtype(torch::kInt32).device(a_tensors.device());

  torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);
  torch::Tensor layout_SFA = torch::empty({num_experts, 5}, options_int32);
  torch::Tensor layout_SFB = torch::empty({num_experts, 5}, options_int32);

  run_get_group_gemm_starts_blockscale_fp8<LayoutSFA, LayoutSFB, ScaleConfig>(
      expert_offsets, problem_sizes, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs,
      b_scales_ptrs, a_tensors, b_tensors, out_tensors, a_scales, b_scales,
      layout_SFA, layout_SFB, n_scale_size, k_scale_size, per_act_block);

  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = Stride<int64_t, Int<1>, Int<0>>;
  using StrideB = Stride<int64_t, Int<1>, Int<0>>;
  using StrideC = typename GemmKernel::InternalStrideC;

  ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<ProblemShape::UnderlyingProblemShape*>(
          problem_sizes.data_ptr());
  ProblemShape prob_shape{num_experts, problem_sizes_as_shapes, nullptr};

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementAB**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(a_strides.data_ptr()),
      static_cast<const ElementAB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(b_strides.data_ptr()),
      static_cast<const ElementScale**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFA*>(layout_SFA.data_ptr()),
      static_cast<const ElementScale**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_SFB.data_ptr())};

  // Currently, we are only able to do broadcast on either all or none
  // a_scales and on either all or none b_scales
  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      static_cast<StrideC*>(c_strides.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(c_strides.data_ptr())};

  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped, prob_shape, mainloop_args,
      epilogue_args};

  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a_tensors.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

}  // namespace
