#include "core/registration.h"

#include <torch/all.h>
#include <cutlass/arch/arch.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "get_group_starts.cuh"

#include <cassert>

using namespace cute;

template <typename OutType, typename ScheduleConfig, typename LayoutD>
void run_blockwise_scaled_group_mm_sm100(
    torch::Tensor& out_ptrs, const torch::Tensor& a_ptrs,
    const torch::Tensor& b_ptrs, const torch::Tensor& a_scales_ptrs,
    const torch::Tensor& b_scales_ptrs, const torch::Tensor& stride_a,
    const torch::Tensor& stride_b, const torch::Tensor& stride_c,
    const torch::Tensor& layout_sfa, const torch::Tensor& layout_sfb,
    const torch::Tensor& problem_sizes, const torch::Tensor& expert_offsets) {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

  // Types
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementC = OutType;
  using ElementD = ElementC;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = LayoutD;

  // Alignments
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, typename ScheduleConfig::MmaTileShape,
          typename ScheduleConfig::ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, void, LayoutC*, AlignmentC, ElementD, LayoutC*,
          AlignmentC, typename ScheduleConfig::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA,
          cute::tuple<LayoutA*, typename ScheduleConfig::LayoutSFA*>,
          AlignmentA, ElementB,
          cute::tuple<LayoutB*, typename ScheduleConfig::LayoutSFB*>,
          AlignmentB, ElementAccumulator, typename ScheduleConfig::MmaTileShape,
          typename ScheduleConfig::ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename ScheduleConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  using UnderlyingProblemShape = ProblemShape::UnderlyingProblemShape;
  int num_experts = (int)expert_offsets.size(0);

  Gemm gemm_op;

  // Mainloop Arguments
  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA**>(a_ptrs.data_ptr()),
      static_cast<StrideA*>(stride_a.data_ptr()),
      static_cast<const ElementB**>(b_ptrs.data_ptr()),
      static_cast<StrideB*>(stride_b.data_ptr()),
      static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFA*>(
          layout_sfa.data_ptr()),
      static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()),
      reinterpret_cast<typename ScheduleConfig::LayoutSFB*>(
          layout_sfb.data_ptr())};

  int device_id = a_ptrs.device().index();
  static const cutlass::KernelHardwareInfo hw_info{
      device_id, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
                     device_id)};

  // Epilogue Arguments
  typename GemmKernel::EpilogueArguments epilogue_args{
      {},  // epilogue.thread
      nullptr,
      static_cast<StrideC*>(stride_c.data_ptr()),
      static_cast<ElementD**>(out_ptrs.data_ptr()),
      static_cast<StrideC*>(stride_c.data_ptr())};

  UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<UnderlyingProblemShape*>(problem_sizes.data_ptr());

  // Gemm Arguments
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  at::cuda::CUDAGuard device_guard{(char)a_ptrs.device().index()};
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(a_ptrs.get_device());

  auto can_implement_status = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess,
              "Failed to implement GEMM");

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a_ptrs.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <typename OutType>
void blockwise_scaled_group_mm_sm100_dispatch_shape(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& scales_a, const torch::Tensor& scales_b,
    const torch::Tensor& expert_offsets, const torch::Tensor& problem_sizes,
    const torch::Tensor& a_strides, const torch::Tensor& b_strides,
    const torch::Tensor& c_strides, bool per_act_block) {
  struct MmaConfig {
    using ElementA = cutlass::float_e4m3_t;
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
    using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
        1, 128, 128, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
    using LayoutC = cutlass::layout::RowMajor;
    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
  };

  int num_experts = (int)expert_offsets.size(0);

  auto a_ptrs = torch::empty(
      {num_experts},
      torch::TensorOptions().dtype(torch::kInt64).device(a.device()));
  auto b_ptrs = torch::empty(
      {num_experts},
      torch::TensorOptions().dtype(torch::kInt64).device(a.device()));
  auto out_ptrs = torch::empty(
      {num_experts},
      torch::TensorOptions().dtype(torch::kInt64).device(a.device()));
  auto a_scales_ptrs = torch::empty(
      {num_experts},
      torch::TensorOptions().dtype(torch::kInt64).device(a.device()));
  auto b_scales_ptrs = torch::empty(
      {num_experts},
      torch::TensorOptions().dtype(torch::kInt64).device(a.device()));

  auto layout_sfa = torch::empty(
      {num_experts, 5},
      torch::TensorOptions().dtype(torch::kInt32).device(a.device()));
  auto layout_sfb = torch::empty(
      {num_experts, 5},
      torch::TensorOptions().dtype(torch::kInt32).device(a.device()));

  int n_scale_size = scales_b.size(1);
  int k_scale_size = scales_b.size(2);
  run_get_group_gemm_starts_blockscale_fp8<typename MmaConfig::LayoutSFA,
                                           typename MmaConfig::LayoutSFB,
                                           typename MmaConfig::ScaleConfig>(
      expert_offsets, problem_sizes, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs,
      b_scales_ptrs, a, b, output, scales_a, scales_b, layout_sfa, layout_sfb,
      n_scale_size, k_scale_size, per_act_block);

  run_blockwise_scaled_group_mm_sm100<OutType, MmaConfig,
                                      typename MmaConfig::LayoutC>(
      out_ptrs, a_ptrs, b_ptrs, a_scales_ptrs, b_scales_ptrs, a_strides,
      b_strides, c_strides, layout_sfa, layout_sfb, problem_sizes,
      expert_offsets);
}

void cutlass_blockwise_scaled_grouped_mm_sm100(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& scales_a, const torch::Tensor& scales_b,
    const torch::Tensor& expert_offsets, const torch::Tensor& problem_sizes,
    const torch::Tensor& a_strides, const torch::Tensor& b_strides,
    const torch::Tensor& c_strides, bool per_act_block) {
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3,
              "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(problem_sizes.size(0) == expert_offsets.size(0),
              "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32,
              "problem_sizes must be int32");
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn,
              "a must be kFloat8_e4m3fn");
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn,
              "b must be kFloat8_e4m3fn");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16 ||
                  output.scalar_type() == torch::kHalf,
              "output must be bfloat16 or half");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32,
              "scales_a must be float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32,
              "scales_b must be float32");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32,
              "expert_offsets must be int32");

  TORCH_CHECK(a_strides.dim() == 1, "a_strides must be 1D tensor");
  TORCH_CHECK(b_strides.dim() == 1, "b_strides must be 1D tensor");
  TORCH_CHECK(c_strides.dim() == 1, "c_strides must be 1D tensor");

  TORCH_CHECK(output.dim() == 2, "output must be 2D tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D tensor");
  TORCH_CHECK(b.dim() == 3, "b must be 3D tensor");
  TORCH_CHECK(scales_a.dim() == 2, "scales_a must be 2D tensor");
  TORCH_CHECK(scales_b.dim() == 3, "scales_b must be 3D tensor");
  TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  TORCH_CHECK(problem_sizes.size(1) == 3,
              "problem_sizes must have shape (num_experts, 3)");
  TORCH_CHECK(problem_sizes.size(0) == expert_offsets.size(0),
              "Number of experts in problem_sizes must match expert_offsets");
  TORCH_CHECK(problem_sizes.dtype() == torch::kInt32,
              "problem_sizes must be int32");
  TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be 1D tensor");
  TORCH_CHECK(a_strides.size(0) == b.size(0),
              "a_strides must have shape (num_experts)");
  TORCH_CHECK(b_strides.size(0) == b.size(0),
              "b_strides must have shape (num_experts)");
  TORCH_CHECK(c_strides.size(0) == b.size(0),
              "c_strides must have shape (num_experts)");

#if defined(ENABLE_CUTLASS_MOE_SM100) && ENABLE_CUTLASS_MOE_SM100
  if (output.scalar_type() == torch::kBFloat16) {
    blockwise_scaled_group_mm_sm100_dispatch_shape<cutlass::bfloat16_t>(
        output, a, b, scales_a, scales_b, expert_offsets, problem_sizes,
        a_strides, b_strides, c_strides, per_act_block);
  } else if (output.scalar_type() == torch::kFloat16) {
    blockwise_scaled_group_mm_sm100_dispatch_shape<cutlass::half_t>(
        output, a, b, scales_a, scales_b, expert_offsets, problem_sizes,
        a_strides, b_strides, c_strides, per_act_block);
  } else {
    TORCH_CHECK(false, "Unsupported output tensor type");
  }
#endif
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("cutlass_blockwise_scaled_grouped_mm_sm100",
         &cutlass_blockwise_scaled_grouped_mm_sm100);
}
