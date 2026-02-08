#include <vector>
#include <tuple>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"

// vllm includes
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "cutlass_extensions/torch_utils.hpp"
#include "cutlass_extensions/common.hpp"

#include "core/registration.h"
#include "get_group_starts.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "w4a8_utils.cuh"

namespace vllm::cutlass_w4a16_moe {

using namespace cute;

// -------------------------------------------------------------------------------------
// Static configuration shared across all instantiations
// -------------------------------------------------------------------------------------
using ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per
                                                             // group
using MmaType = cutlass::bfloat16_t;
using QuantType = cutlass::int4b_t;

constexpr int TileShapeK = 64;  // TODO(siyuan): make this tunable
// TODO(siyuan): relax the SmemLayoutAtomScale in cutlass and make TileShapeK
// tunnable
static int constexpr PackFactor = 8;  // 8 int4 packed into int32

// A matrix configuration
using ElementA = MmaType;
using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
constexpr int AlignmentA =
    128 /
    cutlass::sizeof_bits<ElementA>::value;  // Alignment of A matrix in units of
                                            // elements (up to 16 bytes)

// B matrix configuration
using ElementB = QuantType;  // Element type for B matrix operand
using LayoutB =
    cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
constexpr int AlignmentB =
    128 / cutlass::sizeof_bits<
              ElementB>::value;  // Memory access granularity/alignment of B
                                 // matrix in units of elements (up to 16 bytes)

// TODO(siyuan): Only transpose when batch size is small
using LayoutA_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutB>::type;

// Need to pass a pointer type to make the 3rd dimension of Stride be _0
using StrideA =
    cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
using StrideB =
    cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

// Define the CuTe layout for reoredered quantized tensor B
// LayoutAtomQuant places values that will be read by the same thread in
// contiguous locations in global memory. It specifies the reordering within a
// single warp's fragment
using ValueShuffle =
    Layout<Shape<_2, _4>, Stride<_4, _1>>;  // order [0,2,4,6,1,3,5,7]
int constexpr NumShuffleAtoms = 1;
using MmaAtomShape = Layout<Shape<_1, Int<NumShuffleAtoms>>>;
using LayoutAtomQuant =
    decltype(cutlass::compute_memory_reordering_atom<MmaType, MmaAtomShape,
                                                     ValueShuffle>());
using LayoutB_Reordered = decltype(cute::tile_to_shape(
    LayoutAtomQuant{}, Layout<Shape<int, int, Int<1>>, StrideB>{}));

using ElementScale = cutlass::bfloat16_t;
using LayoutScale = cutlass::layout::RowMajor;

// C/D matrix configuration
using ElementC =
    cutlass::bfloat16_t;  // Element type for C and D matrix operands
using LayoutC =
    cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<
              ElementC>::value;  // Memory access granularity/alignment of C
                                 // matrix in units of elements (up to 16 bytes)

// D matrix configuration
using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel configurations
using ElementAccumulator = float;     // Element type for internal accumulation
using ArchTag = cutlass::arch::Sm90;  // Tag indicating the minimum SM that
                                      // supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
using StageCountType =
    cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based
                                                // on the tile size

// per-channel and per-token scales for epilogue
using ElementSChannel = float;

template <class TileShape_MN, class ClusterShape_MNK, class KernelSchedule,
          class EpilogueSchedule>
struct W4A16GroupedGemmKernel {
  using TileShape =
      decltype(cute::append(TileShape_MN{}, cute::Int<TileShapeK>{}));
  using ClusterShape = ClusterShape_MNK;

  // BF16 activation doesn't need scales. use default epilogue
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator, ElementC,
          typename cutlass::layout::LayoutTranspose<LayoutC>::type*, AlignmentC,
          ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type*,
          AlignmentD, EpilogueSchedule>::CollectiveOp;

  // =========================================================== MIXED INPUT
  // WITH SCALES
  // ===========================================================================
  // The Scale information must get paired with the operand that will be scaled.
  // In this example, B is scaled so we make a tuple of B's information and the
  // scale information.
  using CollectiveMainloopShuffled =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, cute::tuple<ElementB, ElementScale>,
          LayoutB_Reordered*, AlignmentB, ElementA, LayoutA_Transpose*,
          AlignmentA, ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernelShuffled = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloopShuffled, CollectiveEpilogue>;

  using GemmShuffled =
      cutlass::gemm::device::GemmUniversalAdapter<GemmKernelShuffled>;

  using StrideC = typename GemmKernelShuffled::InternalStrideC;
  using StrideD = typename GemmKernelShuffled::InternalStrideD;

  using StrideS = typename CollectiveMainloopShuffled::StrideScale;

  // static asserts for passing in strides/layouts
  // pack to 2x int64
  static_assert(sizeof(StrideS) == 2 * sizeof(int64_t));
  // pack to 3xint32,
  static_assert(sizeof(LayoutB_Reordered) % sizeof(int32_t) == 0,
                "LayoutB_Reordered size must be divisible by 4 bytes");

  static void grouped_mm(
      torch::Tensor& out_tensors, const torch::Tensor& a_tensors,
      const torch::Tensor& b_tensors, const torch::Tensor& b_group_scales,
      const int64_t b_group_size, const torch::Tensor& expert_offsets,
      const torch::Tensor& problem_sizes_torch, const torch::Tensor& a_strides,
      const torch::Tensor& b_strides, const torch::Tensor& c_strides,
      const torch::Tensor& group_scale_strides) {
    auto device = a_tensors.device();
    auto device_id = device.index();
    const at::cuda::OptionalCUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);

    int num_experts = static_cast<int>(expert_offsets.size(0));
    int n = static_cast<int>(b_tensors.size(1));
    int k = static_cast<int>(b_tensors.size(2)) * PackFactor;

    auto options_int =
        torch::TensorOptions().dtype(torch::kInt64).device(device);
    torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor b_group_scales_ptrs = torch::empty(num_experts, options_int);

    // get the correct offsets to pass to gemm
    run_get_group_gemm_starts(expert_offsets, a_ptrs, b_ptrs, out_ptrs,
                              std::nullopt, std::nullopt, b_group_scales_ptrs,
                              a_tensors, b_tensors, out_tensors, std::nullopt,
                              std::nullopt, b_group_scales, b_group_size);

    // construct args
    using Args = typename GemmShuffled::Arguments;
    using MainloopArguments = typename GemmKernelShuffled::MainloopArguments;
    using EpilogueArguments = typename GemmKernelShuffled::EpilogueArguments;
    Args arguments;

    ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
        static_cast<ProblemShape::UnderlyingProblemShape*>(
            problem_sizes_torch.data_ptr());
    ProblemShape prob_shape{num_experts, problem_sizes_as_shapes, nullptr};

    // SwapAB so B operands come first
    MainloopArguments mainloop_arguments{
        static_cast<const QuantType**>(b_ptrs.data_ptr()),
        static_cast<LayoutB_Reordered*>(b_strides.data_ptr()),
        static_cast<const MmaType**>(a_ptrs.data_ptr()),
        static_cast<StrideA*>(a_strides.data_ptr()),
        static_cast<const ElementScale**>(b_group_scales_ptrs.data_ptr()),
        static_cast<StrideS*>(group_scale_strides.data_ptr()),
        static_cast<int>(b_group_size)};

    decltype(arguments.epilogue.thread) fusion_args{};
    // trivial epilogue. TODO(siyuan): expose alpha and beta as arguments
    fusion_args.alpha = 1.0;
    fusion_args.beta = 0.0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
    EpilogueArguments epilogue_arguments{
        fusion_args,
        nullptr,                                       // C
        static_cast<StrideC*>(c_strides.data_ptr()),   // C
        static_cast<ElementD**>(out_ptrs.data_ptr()),  // D
        static_cast<StrideC*>(c_strides.data_ptr())    // D
    };

    static const cutlass::KernelHardwareInfo hw_info{
        device_id,
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            device_id)};

    arguments = Args{cutlass::gemm::GemmUniversalMode::kGrouped, prob_shape,
                     mainloop_arguments, epilogue_arguments, hw_info};

    // Allocate workspace
    size_t workspace_size = GemmShuffled::get_workspace_size(arguments);
    torch::Tensor workspace =
        torch::empty(workspace_size,
                     torch::TensorOptions().dtype(torch::kU8).device(device));

    // Run GEMM
    GemmShuffled gemm;
    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
    CUTLASS_CHECK(gemm.run(stream));
  }
};

// ----------------------------------------------------------------------------
// Kernel instantiations and dispatch logic
// ----------------------------------------------------------------------------
using Coop = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using CoopEpi = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

// Kernel_TileShape_ClusterShape_Schedule
using Kernel_128x16_1x1x1_Coop =
    W4A16GroupedGemmKernel<Shape<_128, _16>, Shape<_1, _1, _1>, Coop, CoopEpi>;
using Kernel_128x16_2x1x1_Coop =
    W4A16GroupedGemmKernel<Shape<_128, _16>, Shape<_2, _1, _1>, Coop, CoopEpi>;

using Kernel_256x16_1x1x1_Coop =
    W4A16GroupedGemmKernel<Shape<_256, _16>, Shape<_1, _1, _1>, Coop, CoopEpi>;
using Kernel_256x16_2x1x1_Coop =
    W4A16GroupedGemmKernel<Shape<_256, _16>, Shape<_2, _1, _1>, Coop, CoopEpi>;

using Kernel_256x32_1x1x1_Coop =
    W4A16GroupedGemmKernel<Shape<_256, _32>, Shape<_1, _1, _1>, Coop, CoopEpi>;
using Kernel_256x32_2x1x1_Coop =
    W4A16GroupedGemmKernel<Shape<_256, _32>, Shape<_2, _1, _1>, Coop, CoopEpi>;

using Kernel_256x64_1x1x1_Coop =
    W4A16GroupedGemmKernel<Shape<_256, _64>, Shape<_1, _1, _1>, Coop, CoopEpi>;
using Kernel_256x64_2x1x1_Coop =
    W4A16GroupedGemmKernel<Shape<_256, _64>, Shape<_2, _1, _1>, Coop, CoopEpi>;

using Kernel_256x128_1x1x1_Coop =
    W4A16GroupedGemmKernel<Shape<_256, _128>, Shape<_1, _1, _1>, Coop, CoopEpi>;
using Kernel_256x128_2x1x1_Coop =
    W4A16GroupedGemmKernel<Shape<_256, _128>, Shape<_2, _1, _1>, Coop, CoopEpi>;

using Kernel_128x256_2x1x1_Coop =
    W4A16GroupedGemmKernel<Shape<_128, _256>, Shape<_2, _1, _1>, Coop, CoopEpi>;

void mm_dispatch(
    torch::Tensor& out_tensors, const torch::Tensor& a_tensors,
    const torch::Tensor& b_tensors, const torch::Tensor& b_group_scales,
    const int64_t b_group_size, const torch::Tensor& expert_offsets,
    const torch::Tensor& problem_sizes, const torch::Tensor& a_strides,
    const torch::Tensor& b_strides, const torch::Tensor& c_strides,
    const torch::Tensor& group_scale_strides, const std::string& schedule) {
  if (schedule == "Kernel_128x16_1x1x1_Coop") {
    Kernel_128x16_1x1x1_Coop::grouped_mm(
        out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        group_scale_strides);
  } else if (schedule == "Kernel_128x16_2x1x1_Coop") {
    Kernel_128x16_2x1x1_Coop::grouped_mm(
        out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        group_scale_strides);
  } else if (schedule == "Kernel_256x16_1x1x1_Coop") {
    Kernel_256x16_1x1x1_Coop::grouped_mm(
        out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        group_scale_strides);
  } else if (schedule == "Kernel_256x16_2x1x1_Coop") {
    Kernel_256x16_2x1x1_Coop::grouped_mm(
        out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        group_scale_strides);
  } else if (schedule == "Kernel_256x32_1x1x1_Coop") {
    Kernel_256x32_1x1x1_Coop::grouped_mm(
        out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        group_scale_strides);
  } else if (schedule == "Kernel_256x32_2x1x1_Coop") {
    Kernel_256x32_2x1x1_Coop::grouped_mm(
        out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        group_scale_strides);
  } else if (schedule == "Kernel_256x64_1x1x1_Coop") {
    Kernel_256x64_1x1x1_Coop::grouped_mm(
        out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        group_scale_strides);
  } else if (schedule == "Kernel_256x64_2x1x1_Coop") {
    Kernel_256x64_2x1x1_Coop::grouped_mm(
        out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        group_scale_strides);
  } else if (schedule == "Kernel_256x128_1x1x1_Coop") {
    Kernel_256x128_1x1x1_Coop::grouped_mm(
        out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        group_scale_strides);
  } else if (schedule == "Kernel_256x128_2x1x1_Coop") {
    Kernel_256x128_2x1x1_Coop::grouped_mm(
        out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        group_scale_strides);
  } else if (schedule == "Kernel_128x256_2x1x1_Coop") {
    Kernel_128x256_2x1x1_Coop::grouped_mm(
        out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
        expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
        group_scale_strides);
  } else {
    TORCH_CHECK(false,
                "cutlass_w4a16_moe_mm: unknown schedule string: ", schedule);
  }
}

void mm(torch::Tensor& out_tensors, const torch::Tensor& a_tensors,
        const torch::Tensor& b_tensors, const torch::Tensor& b_group_scales,
        const int64_t b_group_size, const torch::Tensor& expert_offsets,
        const torch::Tensor& problem_sizes, const torch::Tensor& a_strides,
        const torch::Tensor& b_strides, const torch::Tensor& c_strides,
        const torch::Tensor& group_scale_strides,
        std::optional<std::string> maybe_schedule) {
  // TODO(siyuan): add sanity checks for input tensors
  // user has specified a schedule
  if (maybe_schedule) {
    mm_dispatch(out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
                expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
                group_scale_strides, *maybe_schedule);
    return;
  }

  // use heuristic
  int m_full = a_tensors.size(0);
  int n = b_tensors.size(1);
  int k = b_tensors.size(2) * PackFactor;  // logical k
  int num_experts = b_tensors.size(0);
  // per-expert batch size assuming uniform distribution
  int m_expert = m_full / num_experts;

  std::string schedule;
  if (m_expert <= 16) {
    schedule = "Kernel_128x16_2x1x1_Coop";
  } else if (m_expert <= 32) {
    schedule = "Kernel_256x32_1x1x1_Coop";
  } else if (m_expert <= 64) {
    schedule = "Kernel_256x64_1x1x1_Coop";
  } else if (m_expert <= 128) {
    schedule = "Kernel_256x128_2x1x1_Coop";
  } else {  // m_expert > 128
    schedule = "Kernel_128x256_2x1x1_Coop";
  }

  mm_dispatch(out_tensors, a_tensors, b_tensors, b_group_scales, b_group_size,
              expert_offsets, problem_sizes, a_strides, b_strides, c_strides,
              group_scale_strides, schedule);
}

// 16bit LUT is not supported yet. reorder without re-encoding.
std::tuple<torch::Tensor, torch::Tensor> reorder_int4b(
    torch::Tensor& b_tensors) {
  TORCH_CHECK(b_tensors.dtype() == torch::kInt32);
  TORCH_CHECK(b_tensors.dim() == 3);  // (experts, n, k)
  TORCH_CHECK(b_tensors.is_contiguous());
  TORCH_CHECK(b_tensors.is_cuda());

  int n = static_cast<int>(b_tensors.size(1));
  int k = static_cast<int>(b_tensors.size(2)) * PackFactor;  // logical k

  // CUTLASS reorder_tensor requires k % 256 == 0 and n % 16 == 0.
  // These misalignments cause silent OOB unless run under Compute Sanitizer.
  TORCH_CHECK(k % 256 == 0, "logical k must be divisible by 256");
  TORCH_CHECK(n % 16 == 0, "n must be divisible by 16");

  // we will store the layout to an int32 tensor;
  // this is the number of elements we need per layout
  constexpr size_t layout_width = sizeof(LayoutB_Reordered) / sizeof(int32_t);

  int num_experts = static_cast<int>(b_tensors.size(0));

  auto b_ptr = static_cast<QuantType*>(b_tensors.data_ptr());

  // construct the layout once; assumes each expert has the same layout
  using LayoutType = LayoutB_Reordered;
  std::vector<LayoutType> layout_B_reordered_host(num_experts);
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, Int<1>{}});
  auto shape_B = cute::make_shape(n, k, Int<1>{});
  auto layout_B = make_layout(shape_B, stride_B);
  LayoutType layout_B_reordered = tile_to_shape(LayoutAtomQuant{}, shape_B);

  // reorder weights for each expert
  for (int i = 0; i < num_experts; i++) {
    // since the storage type of int4b is 1 byte but one element is 4 bits
    // we need to adjust the offset
    int64_t offset =
        1ull * i * n * k * cutlass::sizeof_bits<QuantType>::value / 8;
    cutlass::reorder_tensor(b_ptr + offset, layout_B, layout_B_reordered);
  }

  // save the packed layout to torch tensor so we can re-use it
  auto cpu_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  torch::Tensor layout_cpu =
      torch::empty({num_experts, layout_width}, cpu_opts);

  int32_t* layout_data = layout_cpu.data_ptr<int32_t>();
  for (int i = 0; i < num_experts; ++i) {
    std::memcpy(layout_data + i * layout_width,  // dst (int32*)
                &layout_B_reordered,             // src (LayoutType*)
                sizeof(LayoutType));             // number of bytes
  }

  torch::Tensor packed_layout =
      layout_cpu.to(b_tensors.device(), /*non_blocking=*/false);

  return {b_tensors, packed_layout};
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("cutlass_w4a16_moe_mm", &mm);
  m.impl("cutlass_reorder_int4b_grouped", &reorder_int4b);
}

}  // namespace vllm::cutlass_w4a16_moe
/////////////////////////////////////////////////////////////////////////////////////////////////