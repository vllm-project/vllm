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
using CollectiveMainloopScaleOnly =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        cute::tuple<ElementB, cutlass::Array<ElementScale, 8>>,
        LayoutB_Transpose, AlignmentB, ElementA, LayoutA_Transpose, AlignmentA,
        ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernelScaleOnly = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,  // Indicates ProblemShape
    CollectiveMainloopScaleOnly, CollectiveEpilogue>;

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

using GemmScaleOnly =
    cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;
using GemmShuffled =
    cutlass::gemm::device::GemmUniversalAdapter<GemmKernelShuffled>;

using StrideC = typename GemmKernelScaleOnly::StrideC;
using StrideD = typename GemmKernelScaleOnly::StrideD;

using StrideC_ref = cutlass::detail::TagToStrideC_t<LayoutC>;
using StrideD_ref = cutlass::detail::TagToStrideC_t<LayoutD>;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideC_ref stride_C_ref;
StrideD stride_D;
StrideD_ref stride_D_ref;
uint64_t seed;

LayoutB_Reordered layout_B_reordered;

using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;
using StrideS_ref = cutlass::detail::TagToStrideB_t<LayoutScale>;
StrideS stride_S;
StrideS_ref stride_S_ref;

cutlass::DeviceAllocation<ElementA> block_A;
cutlass::DeviceAllocation<ElementB> block_B;
cutlass::DeviceAllocation<ElementB> block_B_modified;
cutlass::DeviceAllocation<ElementA> block_B_dq;
cutlass::DeviceAllocation<ElementScale> block_scale;
cutlass::DeviceAllocation<cutlass::Array<ElementScale, 8>> block_scale_packed;
cutlass::DeviceAllocation<ElementZero> block_zero;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<
    typename GemmScaleOnly::EpilogueOutputOp::ElementOutput>
    block_D;
cutlass::DeviceAllocation<
    typename GemmScaleOnly::EpilogueOutputOp::ElementOutput>
    block_ref_D;

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

  /// Prints the usage statement.
  std::ostream& print_usage(std::ostream& out) const {
    out << "55_hopper_int4_fp8_gemm\n\n"
        << "  Hopper Mixed Data Type GEMM using a Warp Specialized kernel.\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage "
           "statement\n\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --l=<int>                   The number of independent gemm "
           "problems with mnk shape\n"
        << "  --g=<int>                   The size of each group for the "
           "scales. To broadcast a vector of scales or zeros, set the group "
           "size to K.\n"
        << "  --alpha=<f32>               Epilogue scalar alpha\n"
        << "  --beta=<f32>                Epilogue scalar beta\n\n"
        << "  --iterations=<int>          Number of profiling iterations to "
           "perform.\n\n"
        << "  --warmup=<int>              Number of warmup iterations to "
           "perform.\n\n"
        << "  --shuffle=<boolean>         Enable the offline layout "
           "swizzling.\n\n";

    out << "\n\nExamples:\n\n"
        << "$ " << "55_hopper_int4_fp8_gemm"
        << " --m=1024 --n=512 --k=1024 -g=1024 --l=10 --alpha=2 --beta=0.707 "
           "\n\n";

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(Options const& options, const ElementScale* S_ptr,
                const ElementB* B_ptr) {
  auto shape_B = cute::make_shape(options.n, options.k, options.l);
  int const scale_k = cutlass::ceil_div(options.k, options.g);
  stride_A = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
  // Reverse stride here due to swap and transpose
  stride_C = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(options.n, options.m, options.l));
  stride_C_ref = cutlass::make_cute_packed_stride(
      StrideC_ref{}, cute::make_shape(options.m, options.n, options.l));
  // Reverse stride here due to swap and transpose
  stride_D = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(options.n, options.m, options.l));
  stride_D_ref = cutlass::make_cute_packed_stride(
      StrideD_ref{}, cute::make_shape(options.m, options.n, options.l));

  auto layout_B = make_layout(shape_B, stride_B);

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto c_coord = cutlass::make_Coord(options.m * options.l, options.n);

  block_B.reset(b_coord.product());
  block_B_modified.reset(b_coord.product());
  block_B_dq.reset(b_coord.product());
  block_C.reset(c_coord.product());
  block_D.reset(c_coord.product());
  block_ref_D.reset(c_coord.product());

  block_scale.reset(scale_k * options.l * options.n);
  block_scale_packed.reset(scale_k * options.l * options.n);
  block_zero.reset(scale_k * options.l * options.n);

  initialize_tensor(block_B, seed + 2021);
  // czhu change to B_ptr which we passed in
  cutlass::unified_encode_int4b(
      B_ptr, block_B_modified.get(),
      block_B.size());  // this drop perf by ~30 tflops? wtf
  // cutlass::unified_encode_int4b(block_B.get(), block_B_modified.get(),
  // block_B.size());
  initialize_tensor(block_C, seed + 2020);
  // czhu: support pass in our own block_scale (unpacked)
  initialize_scale(block_scale, options);
  cutlass::pack_scale_fp8(S_ptr, block_scale_packed.get(), block_scale.size());
  // cutlass::pack_scale_fp8(block_scale.get(), block_scale_packed.get(),
  // block_scale.size());
  initialize_zero(block_zero, options);

  auto shape_scale_zero = cute::make_shape(options.n, scale_k, options.l);
  stride_S = cutlass::make_cute_packed_stride(
      StrideS{}, cute::make_shape(options.n, scale_k, options.l));
  stride_S_ref = cutlass::make_cute_packed_stride(
      StrideS_ref{}, cute::make_shape(options.n, scale_k, options.l));
  auto layout_scale_zero = make_layout(shape_scale_zero, stride_S_ref);

  // czhu: this moved to verify
  // cudaStream_t stream = cudaStreamDefault;
  // cutlass::dequantize(block_B_dq.get(), block_B.get(), layout_B,
  // block_scale.get(), block_zero.get(), layout_scale_zero, options.g, stream);

  if (options.shuffle) {
    // Repeat the reorder layout atom to tile the whole tensor shape
    layout_B_reordered = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);
    cutlass::reorder_tensor(block_B_modified.get(), layout_B,
                            layout_B_reordered);
    print("in initialize\n");
    print("layout_B: ");
    print(layout_B);
    print("\n");
    print("Quantized tensor layout: ");
    print(layout_B_reordered);
    print("\n");
  }
}

bool verify(Options const& options, ElementD* D_ptr, const MmaType* A_ptr,
            const ElementScale* S_ptr, const QuantType* B_ptr) {
  //
  // Compute reference output
  //

  // In this example, we use the GPU default kernels as a reference (unfused
  // scale). This avoids numerical differences due to different accumulation
  // order.

  // Again, due to numerical differences, we must use fast acc here when the mma
  // type is FP8 as the fused implementation only supports fast acc at the
  // moment.
  constexpr bool IsFP8Input = cute::is_same_v<MmaType, cutlass::float_e4m3_t> ||
                              cute::is_same_v<MmaType, cutlass::float_e5m2_t>;
  using FP8Sched = cute::conditional_t<
      size<0>(TileShape{}) == 64,
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>;
  using ScheduleRef =
      cute::conditional_t<IsFP8Input, FP8Sched,
                          cutlass::gemm::collective::KernelScheduleAuto>;

  using CollectiveMainloopRef =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaType, LayoutA, AlignmentA, MmaType,
          LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAuto, ScheduleRef>::CollectiveOp;

  using CollectiveEpilogueRef =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD,
          cutlass::epilogue::NoSmemWarpSpecialized>::CollectiveOp;

  using GemmKernelRef = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloopRef, CollectiveEpilogueRef>;

  using GemmRef = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelRef>;

  // do the dequantize step here so we can pass in the original scales and
  // weights block_B_dq
  auto shape_B = cute::make_shape(options.n, options.k, options.l);
  int const scale_k = cutlass::ceil_div(options.k, options.g);
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, shape_B);
  auto layout_B = make_layout(shape_B, stride_B);

  auto shape_scale_zero = cute::make_shape(options.n, scale_k, options.l);
  stride_S_ref = cutlass::make_cute_packed_stride(
      StrideS_ref{}, cute::make_shape(options.n, scale_k, options.l));
  auto layout_scale_zero = make_layout(shape_scale_zero, stride_S_ref);

  cudaStream_t stream = cudaStreamDefault;
  // block_zero is initialized to 0s so we can ignore it basically
  // S_ptr is unpacked scales fp8, B_ptr is the original weights B before
  // encoding and reorder
  cutlass::dequantize(block_B_dq.get(), B_ptr, layout_B, S_ptr,
                      block_zero.get(), layout_scale_zero, options.g, stream);
  // uses activations we passed in and dequantized B, no scales
  // dequantize b is from unpacked B_ptr and unpacked S_ptr (it is fp8 type)
  // GemmRef is fp8 fastaccum gemm, alpha=1.0, beta=0.0
  typename GemmRef::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {options.m, options.n, options.k, options.l},
      {A_ptr, stride_A, block_B_dq.get(), stride_B},
      {{options.alpha, options.beta},
       block_C.get(),
       stride_C_ref,
       block_ref_D.get(),
       stride_D_ref}};

  // Run the gemm where the scaling is performed outside of the kernel.
  GemmRef gemm_ref;
  size_t workspace_size = GemmRef::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm_ref.can_implement(arguments));
  CUTLASS_CHECK(gemm_ref.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(gemm_ref.run());

  // compare_reference
  ElementD const epsilon(1e-2f);
  ElementD const non_zero_floor(1e-4f);
  bool passed = cutlass::reference::device::BlockCompareRelativelyEqual(
      block_ref_D.get(), D_ptr, block_D.size(), epsilon, non_zero_floor);

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// end w4a8 example
torch::Tensor mm(
    torch::Tensor const& A, torch::Tensor const& B,
    torch::Tensor const& B_packed,  // just for testing convenience
    int64_t b_type_id, std::optional<at::ScalarType> const& maybe_out_type,
    torch::Tensor const& group_scales,
    torch::Tensor const&
        group_scales_unpacked,  // czhu: this is just for testing convenience,
                                // remove from final thing
    std::optional<int64_t> maybe_group_size,
    std::optional<torch::Tensor> const& maybe_channel_scales,
    std::optional<torch::Tensor> const& maybe_token_scales) {
  // run the stuff
  Options options;
  // try mnk = 5120x4096x6144
  options.m = 5120;
  options.k = 6144;
  options.n = 4096;
  constexpr bool shuffled = true;
  // Allocate output
  using ElementOutput = typename GemmScaleOnly::EpilogueOutputOp::ElementOutput;
  auto device = A.device();
  torch::Tensor D =
      torch::zeros({options.m, options.n},
                   torch::TensorOptions()
                       .dtype(equivalent_scalar_type_v<ElementOutput>)
                       .device(device));

  std::cout << "Running in per-column scale mode." << std::endl;
  if (shuffled) {
    std::cout << "Offline shuffle enabled." << std::endl;
  } else {
    std::cout << "Offline shuffle disabled." << std::endl;
  }

  // run logic, pass in S_ptr so we can pass in scales
  auto S_ptr =
      static_cast<ElementScale const*>(group_scales_unpacked.const_data_ptr());
  auto B_ptr = static_cast<QuantType const*>(B.const_data_ptr());
  auto B_packed_ptr = static_cast<QuantType const*>(B_packed.const_data_ptr());
  initialize(options, S_ptr, B_ptr);

  // Instantiate CUTLASS kernel depending on templates
  using GemmType = std::conditional_t<shuffled, GemmShuffled, GemmScaleOnly>;
  GemmType gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an
  // instance of Gemm auto arguments = args_from_options<GemmShuffled>(options);
  /// Populates a Gemm::Arguments structure from the given commandline options
  /// Swap the A and B tensors, as well as problem shapes here.
  using Args = typename GemmType::Arguments;
  auto D_ptr = static_cast<ElementOutput*>(D.data_ptr());
  auto A_ptr = static_cast<MmaType const*>(A.const_data_ptr());
  auto S_packed_ptr =
      static_cast<cutlass::Array<ElementScale, 8>*>(group_scales.data_ptr());
  // currently uses all the input (A, B, scales)
  // B_packed_ptr vs block_B_modified.get()
  auto arguments = Args{cutlass::gemm::GemmUniversalMode::kGemm,
                        {options.n, options.m, options.k, options.l},
                        {B_packed_ptr, layout_B_reordered, A_ptr, stride_A,
                         S_packed_ptr, stride_S, options.g},
                        {{options.alpha, options.beta},
                         block_C.get(),
                         stride_C,
                         D_ptr,
                         stride_D}};
  // end args logic

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = GemmType::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  // MixedDtypeResult result;
  // S_ptr is the scales before packing, that is used to check the reference
  // result.passed = verify(options, D_ptr, A_ptr, S_ptr, B_ptr);
  // mixed_dtype_profiling(gemm, options, result);
  // std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") <<
  // std::endl; if (!result.passed) {
  //   exit(-1);
  // }

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
  // print("in encode_and_reorder_int4b\n");
  // print("layout_B: ");
  // print(layout_B);
  // print("\n");
  // print("Quantized tensor layout: ");
  // print(layout_B_reordered);
  // print("\n");
  return B_packed;
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("cutlass_w4a8_mm", &mm);
  m.impl("cutlass_pack_scale_fp8", &pack_scale_fp8);
  m.impl("cutlass_encode_and_reorder_int4b", &encode_and_reorder_int4b);
}

}  // namespace vllm::cutlass_w4a8