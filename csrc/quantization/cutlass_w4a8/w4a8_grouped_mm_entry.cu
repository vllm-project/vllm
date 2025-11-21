#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <typeinfo>
#include <float.h>

#include "cutlass/cutlass.h"

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
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"

#include "helper.h"
#include "grouped_mixed_dtype_utils.hpp"

// vllm includes
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "cutlass_extensions/torch_utils.hpp"

#include "core/registration.h"
#include "get_group_starts.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"


namespace vllm::cutlass_w4a8_moe {
using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group
using MmaType = cutlass::float_e4m3_t;
using QuantType = cutlass::int4b_t;
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = MmaType;
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = QuantType;                                      // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// This example manually swaps and transposes, so keep transpose of input layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

// Need to pass a pointer type to make the 3rd dimension of Stride be _0
using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

// Define the CuTe layout for reoredered quantized tensor B
// LayoutAtomQuant places values that will be read by the same thread in contiguous locations in global memory.
// It specifies the reordering within a single warp's fragment
using LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<MmaType>());
using LayoutB_Reordered = decltype(cute::tile_to_shape(LayoutAtomQuant{}, Layout<Shape<int,int,Int<1>>, StrideB>{}));

using ElementZero = cutlass::float_e4m3_t;
using ElementScale = cutlass::float_e4m3_t;
using LayoutScale = cutlass::layout::RowMajor;

// C/D matrix configuration
using         ElementC    = cutlass::bfloat16_t;                                // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = ElementC;
using         LayoutD     = LayoutC;
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_128,_16,cute::Int<TileShapeK>>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster
using StageCountType = cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative; // Epilogue to launch

// per-chan per-tok epilogue
using ElementSChannel = float;
using ChTokScalesEpilogue =
    typename vllm::c3x::ScaledEpilogueArray<ElementAccumulator, ElementD,
                                        TileShape>;
using EVTCompute = typename ChTokScalesEpilogue::EVTCompute;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementSChannel,
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type *, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type *, AlignmentD,
    EpilogueSchedule, EVTCompute
  >::CollectiveOp;

// =========================================================== MIXED INPUT WITH SCALES ===========================================================================
// The Scale information must get paired with the operand that will be scaled. In this example, B is scaled so we make a tuple of B's information and the scale information.
using CollectiveMainloopShuffled = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementB, cutlass::Array<ElementScale, 8>>, LayoutB_Reordered *, AlignmentB,
    ElementA, LayoutA_Transpose *, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernelShuffled = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, 
    CollectiveMainloopShuffled,
    CollectiveEpilogue
>;

using GemmShuffled  = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelShuffled>;

using StrideC = typename GemmKernelShuffled::InternalStrideC;
using StrideD = typename GemmKernelShuffled::InternalStrideD;

using StrideC_ref = cutlass::detail::TagToStrideC_t<LayoutC>;
using StrideD_ref = cutlass::detail::TagToStrideC_t<LayoutD>;
using StrideS = typename CollectiveMainloopShuffled::StrideScale;
using StrideS_ref = cutlass::detail::TagToStrideB_t<LayoutScale>;

uint64_t seed = 2020;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

// In the mainloop, PRMT selects 1 byte from only 8 bytes so the sign bit is handled in an extra PRMT.
// Here the encodings of positive values and negative values are unified (except for the sign bit). 
// For instance, 1 becomes 0b0111, which is the same encoding as -1 (0b1111).
  static void grouped_mm(
    torch::Tensor& out_tensors,
    const torch::Tensor& a_tensors,
    const torch::Tensor& b_tensors,
    const torch::Tensor& a_scales,
    const torch::Tensor& b_scales,
    const torch::Tensor& b_group_scales,
    const int64_t b_group_size,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& problem_sizes_torch,
    const torch::Tensor& a_strides,
    const torch::Tensor& b_strides,
    const torch::Tensor& c_strides,
    const torch::Tensor& group_scale_strides
  ) {
    // validation?
    // TODO: cuda stream/guard
    auto device = a_tensors.device();
    auto device_id = device.index();
    const at::cuda::OptionalCUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);

    int num_experts = static_cast<int>(expert_offsets.size(0));
    int n = static_cast<int>(b_tensors.size(1));
    int k = static_cast<int>(b_tensors.size(2)) * 8; // int4 -> int32 pack factor

    // reconstruct b and S stride
    // TODO: need some way to serialize this info to torch so we don't have to rebuild each time
    // perhaps through the pack methods
    cutlass::DeviceAllocation<LayoutB_Reordered> layout_B_reordered_local;
    cutlass::DeviceAllocation<StrideS> stride_S_local;
    std::vector<LayoutB_Reordered> layout_B_reordered_host(num_experts);
    std::vector<StrideS> stride_S_host_local;
    int const scale_k = cutlass::ceil_div(k, b_group_size);
    // for building this we only use n and k
    for (int32_t i = 0; i < num_experts; ++i) {
      // this happens after initialize (problem shape transposed) so we need to swap it, gets logical N, K
      auto shape_B = cute::make_shape(n, k, Int<1>{});
      // Repeat the reorder layout atom to tile the whole tensor shape 
      layout_B_reordered_host[i] = tile_to_shape(LayoutAtomQuant{}, shape_B);
      // logical N, scale_k
      stride_S_host_local.push_back(cutlass::make_cute_packed_stride(StrideS{}, {n, scale_k, 1}));
    }
    // copy to device
    layout_B_reordered_local.reset(num_experts);
    layout_B_reordered_local.copy_from_host(layout_B_reordered_host.data());
    stride_S_local.reset(num_experts);
    stride_S_local.copy_from_host(stride_S_host_local.data());

    auto options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
    torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor b_group_scales_ptrs = torch::empty(num_experts, options_int);

    // get the correct offsets to pass to gemm
    run_get_group_gemm_starts(expert_offsets, a_ptrs, b_ptrs, out_ptrs,
                            a_scales_ptrs, b_scales_ptrs, b_group_scales_ptrs, a_tensors, b_tensors,
                            out_tensors, a_scales, b_scales, b_group_scales, b_group_size);
    
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
        static_cast<const QuantType**>(b_ptrs.data_ptr()), layout_B_reordered_local.get(),
        static_cast<const MmaType**>(a_ptrs.data_ptr()), static_cast<StrideA*>(a_strides.data_ptr()),
        static_cast<const cutlass::Array<ElementScale, 8> **>(b_group_scales_ptrs.data_ptr()),
        stride_S_local.get(),
        static_cast<int>(b_group_size)
    };

    EpilogueArguments epilogue_arguments {
      // since we are doing SwapAB the channel scales comes first, then token scales
      ChTokScalesEpilogue::prepare_args( // see ScaledEpilogueArray
          static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()), // per-channel
          static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()), // per-token
          true,
          true
      ),
      nullptr, // C
      static_cast<StrideC*>(c_strides.data_ptr()), // C
      static_cast<ElementD**>(out_ptrs.data_ptr()), // D
      static_cast<StrideC*>(c_strides.data_ptr()) // D
    };
    
    static const cutlass::KernelHardwareInfo hw_info{
    device_id, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
                  device_id)};

    arguments = Args {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      prob_shape,
      mainloop_arguments,
      epilogue_arguments,
      hw_info
    };

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

void mm(
    torch::Tensor& out_tensors,
    const torch::Tensor& a_tensors,
    const torch::Tensor& b_tensors, // expected to be correctly packed/reordered/encoded
    const torch::Tensor& a_scales,
    const torch::Tensor& b_scales,
    const torch::Tensor& b_group_scales, // expected to be packed fp8
    const int64_t b_group_size,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& a_strides,
    const torch::Tensor& b_strides,
    const torch::Tensor& c_strides,
    const torch::Tensor& group_scale_strides,
    c10::optional<std::string> maybe_schedule
) {
    // no dispatch logic for now, just call one kernel
    // TODO: inputs validation
    return grouped_mm(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales,
        b_group_scales, b_group_size, expert_offsets, problem_sizes,
        a_strides, b_strides, c_strides, group_scale_strides
    );
}

torch::Tensor encode_and_reorder_int4b(torch::Tensor const& b_tensors){
  TORCH_CHECK(b_tensors.dtype() == torch::kInt32);
  TORCH_CHECK(b_tensors.dim() == 3); // (experts, n, k) TODO: this shape is unclear how it should be passed in but seems correct so far
  TORCH_CHECK(b_tensors.is_contiguous());
  TORCH_CHECK(b_tensors.is_cuda());

  torch::Tensor b_tensors_packed = torch::empty_like(b_tensors);
  int num_experts = static_cast<int>(b_tensors.size(0));
  int n = static_cast<int>(b_tensors.size(1));
  int k = static_cast<int>(b_tensors.size(2)) * 8; // packed factor to get logical shapes

  auto b_ptr = static_cast<QuantType const*>(b_tensors.const_data_ptr());
  auto b_packed_ptr = static_cast<QuantType*>(b_tensors_packed.data_ptr());
  
  // encode first
  cutlass::unified_encode_int4b(b_ptr, b_packed_ptr, num_experts * n * k);

  // offsets and loop through experts
  for (int i = 0; i < num_experts; i++){
    auto shape_B = cute::make_shape(n, k, Int<1>{});
    auto stride_B_local = cutlass::make_cute_packed_stride(StrideB{}, {n, k, Int<1>{}});
    auto layout_B = make_layout(shape_B, stride_B_local);
    LayoutB_Reordered layout_B_reordered_local = tile_to_shape(LayoutAtomQuant{}, shape_B);
    auto offset = i * n * k * cutlass::sizeof_bits<QuantType>::value / 8; // bytes/storage type
    cutlass::reorder_tensor(b_packed_ptr + offset, layout_B, layout_B_reordered_local);
  }

  return b_tensors_packed;

}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("cutlass_w4a8_moe_mm", &mm);
  m.impl("cutlass_encode_and_reorder_int4b_grouped", &encode_and_reorder_int4b);
}

} // namespace vllm::cutlass_w4a8_moe
/////////////////////////////////////////////////////////////////////////////////////////////////