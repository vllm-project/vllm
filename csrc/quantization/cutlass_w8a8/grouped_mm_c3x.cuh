#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "cutlass_extensions/common.hpp"

using namespace cute;

#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
  #define ENABLE_SM90_KERNEL_LEVEL 1
#endif

__global__ void get_group_gemm_starts(
    int32_t* expert_offsets, int64_t* a_offsets, int64_t* b_offsets,
    int64_t* out_offsets, int64_t* a_scales_offsets, int64_t* b_scales_offsets,
    const int64_t a_base_as_int, const int64_t b_base_as_int,
    const int64_t out_base_as_int, const int64_t a_scales_base_as_int,
    const int64_t b_scales_base_as_int, int64_t n, int64_t k,
    bool per_act_token, bool per_out_ch, int64_t ab_size, int64_t c_size,
    int64_t acc_size) {
  int expert_id = threadIdx.x;

  int64_t expert_offset = expert_offsets[expert_id];

  a_offsets[expert_id] = a_base_as_int + expert_offset * k * ab_size;
  b_offsets[expert_id] = b_base_as_int + expert_id * k * n * ab_size;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n * c_size;
  a_scales_offsets[expert_id] =
      a_scales_base_as_int + (per_act_token ? expert_offset : 0) * acc_size;
  b_scales_offsets[expert_id] =
      b_scales_base_as_int +
      (per_out_ch ? n * expert_id : expert_id) * acc_size;
}

namespace {

using ProblemShape =
    cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using ElementAB_Type = cutlass::float_e4m3_t;
using ElementC_Type = cutlass::half_t;

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
struct cutlass_3x_group_gemm {
  using ElementAB = ElementAB_;
  using ElementC = void;
  using ElementD = ElementC_;
  using ElementAccumulator = float;

  using Epilogue = Epilogue_<ElementAccumulator, ElementD, TileShape>;

  using StrideC =
      cute::remove_pointer_t<cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>>;

  const int AlignmentAB = 128 / cutlass::sizeof_bits<ElementAB>::value;
  const int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;

  using EVTCompute = typename Epilogue::EVTCompute;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutC*, 4, ElementD, LayoutC*, 4,
          EpilogueSchedule, EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementAB, LayoutA*, 16, ElementAB, LayoutB*,
          16, ElementAccumulator, TileShape, ClusterShape, Stages,
          KernelSchedule>::CollectiveOp;

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};

template <typename Gemm>
void cutlass_group_gemm_caller(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementC = typename Gemm::ElementC;
  using ElementD = typename Gemm::ElementD;

  int groups = (int)expert_offsets.size(0);
  int k_size = a_tensors.size(1);
  int n_size = out_tensors.size(1);

  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != groups;

  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  auto options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(a_tensors.device());

  torch::Tensor a_ptrs = torch::empty(groups, options_int);
  torch::Tensor b_ptrs = torch::empty(groups, options_int);
  torch::Tensor out_ptrs = torch::empty(groups, options_int);
  torch::Tensor a_scales_ptrs = torch::empty(groups, options_int);
  torch::Tensor b_scales_ptrs = torch::empty(groups, options_int);

  get_group_gemm_starts<<<1, groups, 0, stream>>>(
      reinterpret_cast<int32_t*>(expert_offsets.data_ptr()),
      reinterpret_cast<int64_t*>(a_ptrs.data_ptr()),
      reinterpret_cast<int64_t*>(b_ptrs.data_ptr()),
      reinterpret_cast<int64_t*>(out_ptrs.data_ptr()),
      reinterpret_cast<int64_t*>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<int64_t*>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<int64_t>(a_tensors.data_ptr()),
      reinterpret_cast<int64_t>(b_tensors.data_ptr()),
      reinterpret_cast<int64_t>(out_tensors.data_ptr()),
      reinterpret_cast<int64_t>(a_scales.data_ptr()),
      reinterpret_cast<int64_t>(b_scales.data_ptr()), out_tensors.size(1),
      a_tensors.size(1), per_act_token, per_out_ch, sizeof(ElementAB_Type),
      sizeof(ElementC_Type), sizeof(ElementAccumulator));

  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = Stride<int64_t, Int<1>, Int<0>>;
  using StrideB = Stride<int64_t, Int<1>, Int<0>>;
  using StrideC = typename GemmKernel::InternalStrideC;

  ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          problem_sizes.data_ptr());
  ProblemShape prob_shape{groups, problem_sizes_as_shapes, nullptr};

  typename GemmKernel::MainloopArguments mainloop_args{
      reinterpret_cast<const ElementAB_Type**>(a_ptrs.data_ptr()),
      reinterpret_cast<StrideA*>(a_strides.data_ptr()),
      reinterpret_cast<const ElementAB_Type**>(b_ptrs.data_ptr()),
      reinterpret_cast<StrideB*>(b_strides.data_ptr())};

  // Currently, we are only able to do broadcast on either all or none a_scales
  // and on either all or none b_scales
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(reinterpret_cast<const ElementAccumulator**>(
                                       a_scales_ptrs.data_ptr()),
                                   reinterpret_cast<const ElementAccumulator**>(
                                       b_scales_ptrs.data_ptr()),
                                   per_act_token, per_out_ch),
      nullptr, reinterpret_cast<StrideC*>(c_strides.data_ptr()),
      reinterpret_cast<ElementC_Type**>(out_ptrs.data_ptr()),
      reinterpret_cast<StrideC*>(c_strides.data_ptr())};

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
