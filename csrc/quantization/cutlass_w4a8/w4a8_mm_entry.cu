#include <iostream>
#include "core/scalar_type.hpp"
#include <torch/all.h>
#include <Python.h>
#include "cutlass_extensions/torch_utils.hpp"

#include "core/registration.h"

// cutlass imports
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/mixed_dtype_utils.hpp"

#include "cutlass_extensions/common.hpp"
#include "mixed_dtype_utils.hpp"

namespace vllm::cutlass_w4a8 {

using namespace cute;
// end cutlass imports

// start w4a8 example
/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
using MmaType = cutlass::float_e4m3_t;
using QuantType = cutlass::int4b_t;
constexpr int TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;

// A matrix configuration
using ElementA = MmaType;                   // Element type for A matrix operand
using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
constexpr int AlignmentA =
    128 / cutlass::sizeof_bits<
              ElementA>::value;  // Memory access granularity/alignment of A
                                 // matrix in units of elements (up to 16 bytes)

// B matrix configuration
using ElementB = QuantType;  // Element type for B matrix operand
using LayoutB =
    cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
constexpr int AlignmentB =
    128 / cutlass::sizeof_bits<
              ElementB>::value;  // Memory access granularity/alignment of B
                                 // matrix in units of elements (up to 16 bytes)

// This example manually swaps and transposes, so keep transpose of input
// layouts
using LayoutA_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutB>::type;

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

// Define the CuTe layout for reoredered quantized tensor B
// LayoutAtomQuant places values that will be read by the same thread in
// contiguous locations in global memory. It specifies the reordering within a
// single warp's fragment
using LayoutAtomQuant =
    decltype(cutlass::compute_memory_reordering_atom<MmaType>());
using LayoutB_Reordered = decltype(cute::tile_to_shape(
    LayoutAtomQuant{}, Layout<Shape<int, int, int>, StrideB>{}));

using ElementScale = MmaType;
using ElementZero = ElementScale;  // only for verify
using LayoutScale = cutlass::layout::RowMajor;

// C/D matrix configuration
using ElementC = cutlass::half_t;  // Element type for C and D matrix operands
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
using ElementCompute = float;         // Element type for epilogue computation
using ArchTag = cutlass::arch::Sm90;  // Tag indicating the minimum SM that
                                      // supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
// czhu: sweep
using TileShape =
    Shape<_128, _128, cute::Int<TileShapeK>>;  // Threadblock-level tile size
using ClusterShape =
    Shape<_1, _1, _1>;  // Shape of the threadblocks in a cluster
// end sweep
using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedCooperative;  // Kernel to launch
                                                         // based on the default
                                                         // setting in the
                                                         // Collective Builder
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
        ClusterShape, EpilogueTileType, ElementAccumulator, ElementAccumulator,
        // Transpose layout of D here since we use explicit swap + transpose
        // the void type for C tells the builder to allocate 0 smem for the C
        // matrix. We can enable this if beta == 0 by changing ElementC to void
        // below.
        ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type,
        AlignmentC, ElementD,
        typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
        EpilogueSchedule  // This is the only epi supporting the required swap +
                          // transpose.
        >::CollectiveOp;

// =========================================================== MIXED INPUT WITH
// SCALES
// ===========================================================================
// The Scale information must get paired with the operand that will be scaled.
// In this example, B is scaled so we make a tuple of B's information and the
// scale information.
using CollectiveMainloopShuffled =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        cute::tuple<ElementB, cutlass::Array<ElementScale, 8>>,
        LayoutB_Reordered, AlignmentB, ElementA, LayoutA_Transpose, AlignmentA,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernelShuffled = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,  // Indicates ProblemShape
    CollectiveMainloopShuffled, CollectiveEpilogue>;

using GemmShuffled =
    cutlass::gemm::device::GemmUniversalAdapter<GemmKernelShuffled>;

using StrideC = typename GemmKernelShuffled::StrideC;
using StrideD = typename GemmKernelShuffled::StrideD;
using StrideS = typename CollectiveMainloopShuffled::StrideScale;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideC stride_C;
StrideD stride_D;
StrideS stride_S;
uint64_t seed;

LayoutB_Reordered layout_B_reordered;

cutlass::DeviceAllocation<ElementC> block_C;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options : MixedDtypeOptions {
  bool shuffle = true;

  // Parses the command line
  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("shuffle", shuffle);

    this->MixedDtypeOptions::parse(argc, args);

    mode = 1;  // override the mode value to always be scale only mode
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////
torch::Tensor mm(torch::Tensor const& A,
                 torch::Tensor const& B,  // already packed
                 int64_t b_type_id,
                 std::optional<at::ScalarType> const& maybe_out_type,
                 torch::Tensor const& group_scales,  // already packed
                 std::optional<int64_t> maybe_group_size,
                 std::optional<torch::Tensor> const& maybe_channel_scales,
                 std::optional<torch::Tensor> const& maybe_token_scales) {
  Options options;
  // try mnk = 5120x4096x6144
  options.m = 5120;
  options.k = 6144;
  options.n = 4096;

  // Allocate output
  using ElementOutput = typename GemmShuffled::EpilogueOutputOp::ElementOutput;
  auto device = A.device();
  torch::Tensor D =
      torch::zeros({options.m, options.n},
                   torch::TensorOptions()
                       .dtype(equivalent_scalar_type_v<ElementOutput>)
                       .device(device));

  // run logic, pass in S_ptr so we can pass in scales
  auto B_ptr = static_cast<QuantType const*>(B.const_data_ptr());

  // Instantiate CUTLASS kernel depending on templates
  GemmShuffled gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an
  // instance of Gemm auto arguments = args_from_options<GemmShuffled>(options);
  /// Populates a Gemm::Arguments structure from the given commandline options
  /// Swap the A and B tensors, as well as problem shapes here.
  using Args = typename GemmShuffled::Arguments;
  auto D_ptr = static_cast<ElementOutput*>(D.data_ptr());
  auto A_ptr = static_cast<MmaType const*>(A.const_data_ptr());
  auto S_ptr = static_cast<cutlass::Array<ElementScale, 8> const*>(
      group_scales.const_data_ptr());
  // currently uses all the input (A, B, scales)
  // init strides here
  int const scale_k = cutlass::ceil_div(options.k, options.g);
  stride_A = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(options.m, options.k, options.l));
  // Reverse stride here due to swap and transpose
  stride_C = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(options.n, options.m, options.l));
  // Reverse stride here due to swap and transpose
  stride_D = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(options.n, options.m, options.l));
  stride_S = cutlass::make_cute_packed_stride(
      StrideS{}, cute::make_shape(options.n, scale_k, options.l));
  auto arguments = Args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.n, options.m, options.k, options.l},
      {B_ptr, layout_B_reordered, A_ptr, stride_A, S_ptr, stride_S, options.g},
      {{options.alpha, options.beta},
       D_ptr,  // dont accumulate anything anyways since beta=0
       stride_C,
       D_ptr,
       stride_D}};

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = GemmShuffled::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  return D;
}

torch::Tensor pack_scale_fp8(torch::Tensor const& scales) {
  // TODO: input validation + row/col major ordering of scales? type?
  // template on the mma/scale type?
  auto packed_scales =
      torch::empty({scales.numel() * 8},
                   torch::TensorOptions()
                       .dtype(scales.dtype())  // torch.float8_e4m3fn
                       .device(scales.device()));
  auto scales_ptr = static_cast<MmaType const*>(scales.const_data_ptr());
  auto packed_scales_ptr =
      static_cast<cutlass::Array<ElementScale, 8>*>(packed_scales.data_ptr());
  cutlass::pack_scale_fp8(scales_ptr, packed_scales_ptr, scales.numel());
  // what to return as?
  return packed_scales;
}

torch::Tensor encode_and_reorder_int4b(torch::Tensor const& B) {
  // todo: input validation (type, shape, device)
  auto sizes = B.sizes();
  int k_packed = sizes[0];
  int n = sizes[1];
  int k = k_packed * 8;
  torch::Tensor B_packed = torch::empty_like(B);
  auto B_ptr = static_cast<QuantType const*>(B.const_data_ptr());
  auto B_packed_ptr = static_cast<QuantType*>(B_packed.data_ptr());
  // encode
  cutlass::unified_encode_int4b(B_ptr, B_packed_ptr, n * k);  // try this
  // reorder
  auto shape_B = cute::make_shape(n, k, 1);
  layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
  // layoutright/row major
  auto layout_B = make_layout(shape_B, LayoutRight{});
  cutlass::reorder_tensor(B_packed_ptr, layout_B, layout_B_reordered);
  return B_packed;
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("cutlass_w4a8_mm", &mm);
  m.impl("cutlass_pack_scale_fp8", &pack_scale_fp8);
  m.impl("cutlass_encode_and_reorder_int4b", &encode_and_reorder_int4b);
}

}  // namespace vllm::cutlass_w4a8