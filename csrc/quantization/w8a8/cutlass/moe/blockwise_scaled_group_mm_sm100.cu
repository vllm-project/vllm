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
#include <cassert>

using namespace cute;

template <typename ElementAB, typename ElementC, typename ElementAccumulator,
          typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
__global__ void get_ggemm_starts(
    int32_t* expert_offsets, ElementAB** a_offsets, ElementAB** b_offsets,
    ElementC** out_offsets, ElementAccumulator** a_scale_offsets,
    ElementAccumulator** b_scale_offsets, ElementAB* a_base_as_int,
    ElementAB* b_base_as_int, ElementC* out_base_as_int,
    ElementAccumulator* a_scale_base_as_int,
    ElementAccumulator* b_scale_base_as_int, LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int, int* problem_sizes) {
  int expert_id = threadIdx.x;

  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }

  int m = problem_sizes[expert_id * 3];
  int n = problem_sizes[expert_id * 3 + 1];
  int k = problem_sizes[expert_id * 3 + 2];

  int32_t expert_offset = expert_offsets[expert_id];
  int a_stride = expert_offset * k;
  int b_stride = expert_id * k * n;
  int a_scale_stride = expert_offset * k / 128;
  int b_scale_stride = expert_id * k * n / 128 / 128;

  a_offsets[expert_id] = a_base_as_int + a_stride;
  b_offsets[expert_id] = b_base_as_int + b_stride;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scale_offsets[expert_id] = a_scale_base_as_int + a_scale_stride;
  b_scale_offsets[expert_id] = b_scale_base_as_int + b_scale_stride;

  LayoutSFA* layout_sfa_ptr = layout_sfa_base_as_int + expert_id;
  LayoutSFB* layout_sfb_ptr = layout_sfb_base_as_int + expert_id;

  *layout_sfa_ptr =
      ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  *layout_sfb_ptr =
      ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));
}

#define __CALL_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE, LayoutSFA, LayoutSFB, \
                                 ScaleConfig)                                 \
  else if (out_tensors.dtype() == TENSOR_C_TYPE) {                            \
    get_ggemm_starts<cutlass::float_e4m3_t, C_TYPE, float, LayoutSFA,         \
                     LayoutSFB, ScaleConfig><<<1, num_experts, 0, stream>>>(  \
        static_cast<int32_t*>(expert_offsets.data_ptr()),                     \
        static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),              \
        static_cast<cutlass::float_e4m3_t**>(b_ptrs.data_ptr()),              \
        static_cast<C_TYPE**>(out_ptrs.data_ptr()),                           \
        static_cast<float**>(a_scales_ptrs.data_ptr()),                       \
        static_cast<float**>(b_scales_ptrs.data_ptr()),                       \
        static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr()),            \
        static_cast<cutlass::float_e4m3_t*>(b_tensors.data_ptr()),            \
        static_cast<C_TYPE*>(out_tensors.data_ptr()),                         \
        static_cast<float*>(a_scales.data_ptr()),                             \
        static_cast<float*>(b_scales.data_ptr()),                             \
        reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),                  \
        reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()),                  \
        static_cast<int*>(problem_sizes.data_ptr()));                         \
  }

template <typename LayoutSFA, typename LayoutSFB, typename ScaleConfig>
void run_get_ggemm_starts(
    torch::Tensor const& expert_offsets, torch::Tensor& a_ptrs,
    torch::Tensor& b_ptrs, torch::Tensor& out_ptrs,
    torch::Tensor& a_scales_ptrs, torch::Tensor& b_scales_ptrs,
    torch::Tensor const& a_tensors, torch::Tensor const& b_tensors,
    torch::Tensor out_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& layout_sfa,
    torch::Tensor const& layout_sfb, torch::Tensor const& problem_sizes) {
  TORCH_CHECK(a_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b_tensors.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(out_tensors.size(1) % 128 == 0 or out_tensors.size(0) % 128 == 0);
  TORCH_CHECK(a_tensors.size(1) % 128 == 0 or a_tensors.size(0) % 128 == 0);

  int num_experts = (int)expert_offsets.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());

  if (false) {
  }
  __CALL_GET_STARTS_KERNEL(torch::kBFloat16, cutlass::bfloat16_t, LayoutSFA,
                           LayoutSFB, ScaleConfig)
  __CALL_GET_STARTS_KERNEL(torch::kFloat16, cutlass::half_t, LayoutSFA,
                           LayoutSFB, ScaleConfig)
  else {
    TORCH_CHECK(false, "Unsupported output tensor type");
  }
}

template <typename OutType, typename ScheduleConfig, typename LayoutD>
void run_blockwise_scaled_group_mm(
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
void blockwise_scaled_group_mm_dispatch_shape(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& scales_a, const torch::Tensor& scales_b,
    const torch::Tensor& problem_sizes, const torch::Tensor& expert_offsets) {
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

  auto stride_a = torch::full(
      {num_experts}, a.size(1),
      torch::TensorOptions().dtype(torch::kInt64).device(a.device()));
  auto stride_b = torch::full(
      {num_experts}, a.size(1),
      torch::TensorOptions().dtype(torch::kInt64).device(a.device()));
  auto stride_c = torch::full(
      {num_experts}, output.size(1),
      torch::TensorOptions().dtype(torch::kInt64).device(a.device()));

  torch::TensorOptions options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(a.device());

  run_get_ggemm_starts<typename MmaConfig::LayoutSFA,
                       typename MmaConfig::LayoutSFB,
                       typename MmaConfig::ScaleConfig>(
      expert_offsets, a_ptrs, b_ptrs, out_ptrs, a_scales_ptrs, b_scales_ptrs, a,
      b, output, scales_a, scales_b, layout_sfa, layout_sfb, problem_sizes);

  run_blockwise_scaled_group_mm<OutType, MmaConfig,
                                typename MmaConfig::LayoutC>(
      out_ptrs, a_ptrs, b_ptrs, a_scales_ptrs, b_scales_ptrs, stride_a,
      stride_b, stride_c, layout_sfa, layout_sfb, problem_sizes,
      expert_offsets);
}

void cutlass_blockwise_scaled_grouped_mm(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& scales_a, const torch::Tensor& scales_b,
    const torch::Tensor& problem_sizes, const torch::Tensor& expert_offsets) {
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

#if defined(ENABLE_CUTLASS_MOE_SM100) && ENABLE_CUTLASS_MOE_SM100
  if (output.scalar_type() == torch::kBFloat16) {
    blockwise_scaled_group_mm_dispatch_shape<cutlass::bfloat16_t>(
        output, a, b, scales_a, scales_b, problem_sizes, expert_offsets);
  } else if (output.scalar_type() == torch::kFloat16) {
    blockwise_scaled_group_mm_dispatch_shape<cutlass::half_t>(
        output, a, b, scales_a, scales_b, problem_sizes, expert_offsets);
  } else {
    TORCH_CHECK(false, "Unsupported output tensor type");
  }
#endif
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("cutlass_blockwise_scaled_grouped_mm",
         &cutlass_blockwise_scaled_grouped_mm);
}
