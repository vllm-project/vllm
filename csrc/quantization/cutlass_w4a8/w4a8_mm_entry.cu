//
// Based off of:
//   https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu
//

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "cutlass_extensions/torch_utils.hpp"
#include "w4a8_utils.cuh"

#include "core/registration.h"

#include "cutlass/cutlass.h"
#include <limits>

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/mixed_dtype_utils.hpp"

#include "cutlass_extensions/common.hpp"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

#include <cuda_runtime.h>

namespace vllm::cutlass_w4a8 {

using namespace cute;

// -------------------------------------------------------------------------------------
// Static configuration shared across all instantiations
// -------------------------------------------------------------------------------------
using MmaType = cutlass::float_e4m3_t;  // A/scale element type
using QuantType = cutlass::int4b_t;     // B element type (packed int4)

static int constexpr TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;
static int constexpr ScalePackSize = 8;  // pack 8 scale elements together
static int constexpr PackFactor = 8;     // 8 4-bit packed into int32

// A matrix configuration
using ElementA = MmaType;                   // Element type for A matrix operand
using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
using LayoutA_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutA>::type;
constexpr int AlignmentA =
    128 / cutlass::sizeof_bits<
              ElementA>::value;  // Memory access granularity/alignment of A
                                 // matrix in units of elements (up to 16 bytes)
using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;

// B matrix configuration
using ElementB = QuantType;  // Element type for B matrix operand
using LayoutB =
    cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
using LayoutB_Transpose =
    typename cutlass::layout::LayoutTranspose<LayoutB>::type;
constexpr int AlignmentB =
    128 / cutlass::sizeof_bits<
              ElementB>::value;  // Memory access granularity/alignment of B
                                 // matrix in units of elements (up to 16 bytes)
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

// Define the CuTe layout for reordered quantized tensor B
// LayoutAtomQuant places values that will be read by the same thread in
// contiguous locations in global memory. It specifies the reordering within a
// single warp's fragment
using LayoutAtomQuant =
    decltype(cutlass::compute_memory_reordering_atom<MmaType>());
using LayoutB_Reordered = decltype(cute::tile_to_shape(
    LayoutAtomQuant{}, Layout<Shape<int, int, int>, StrideB>{}));

// Group-wise scales
using ElementScale = MmaType;
using LayoutScale = cutlass::layout::RowMajor;

// Per-tok, per-chan scales
using ElementSChannel = float;

// C/D matrix configuration
using ElementC =
    cutlass::bfloat16_t;  // Element type for C and D matrix operands
using LayoutC =
    cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
constexpr int AlignmentC =
    128 / cutlass::sizeof_bits<
              ElementC>::value;  // Memory access granularity/alignment of C
                                 // matrix in units of elements (up to 16 bytes)

using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Core kernel configurations
using ElementAccumulator = float;     // Element type for internal accumulation
using ElementCompute = float;         // Element type for epilogue computation
using ArchTag = cutlass::arch::Sm90;  // Tag indicating the minimum SM that
                                      // supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedCooperative;  // Kernel to launch
                                                         // based on the default
                                                         // setting in the
                                                         // Collective Builder
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

// ----------------------------------------------------------------------------
// Kernel template — Tile/Cluster shapes
// ----------------------------------------------------------------------------
template <class TileShape_MN, class ClusterShape_MNK>
struct W4A8GemmKernel {
  using TileShape =
      decltype(cute::append(TileShape_MN{}, cute::Int<TileShapeK>{}));
  using ClusterShape = ClusterShape_MNK;

  // Epilogue per-tok, per-chan scales
  using ChTokScalesEpilogue =
      typename vllm::c3x::ScaledEpilogue<ElementAccumulator, ElementD,
                                         TileShape>;
  using EVTCompute = typename ChTokScalesEpilogue::EVTCompute;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
          ElementAccumulator, ElementSChannel,
          // Transpose layout of D here since we use explicit swap + transpose
          // the void type for C tells the builder to allocate 0 smem for the C
          // matrix. We can enable this if beta == 0 by changing ElementC to
          // void below.
          ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type,
          AlignmentC, ElementD,
          typename cutlass::layout::LayoutTranspose<LayoutD>::type, AlignmentD,
          EpilogueSchedule,  // This is the only epi supporting the required
                             // swap + transpose.
          EVTCompute>::CollectiveOp;

  // The Scale information must get paired with the operand that will be scaled.
  // In this example, B is scaled so we make a tuple of B's information and the
  // scale information.
  using CollectiveMainloopShuffled =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          cute::tuple<ElementB, cutlass::Array<ElementScale, ScalePackSize>>,
          LayoutB_Reordered, AlignmentB, ElementA, LayoutA_Transpose,
          AlignmentA, ElementAccumulator, TileShape, ClusterShape,
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

  static torch::Tensor mm(torch::Tensor const& A,
                          torch::Tensor const& B,             // already packed
                          torch::Tensor const& group_scales,  // already packed
                          int64_t group_size,
                          torch::Tensor const& channel_scales,
                          torch::Tensor const& token_scales,
                          std::optional<at::ScalarType> const& maybe_out_type) {
    // TODO: param validation
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);

    // safely cast group_size to int
    TORCH_CHECK(group_size > 0 && group_size <= std::numeric_limits<int>::max(),
                "group_size out of supported range for int: ", group_size);
    int const group_size_int = static_cast<int>(group_size);

    // Allocate output
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    auto device = A.device();
    auto stream = at::cuda::getCurrentCUDAStream(device.index());
    torch::Tensor D =
        torch::empty({m, n}, torch::TensorOptions()
                                 .dtype(equivalent_scalar_type_v<ElementD>)
                                 .device(device));
    // prepare arg pointers
    auto A_ptr = static_cast<MmaType const*>(A.const_data_ptr());
    auto B_ptr = static_cast<QuantType const*>(B.const_data_ptr());
    auto D_ptr = static_cast<ElementD*>(D.data_ptr());
    // can we avoid hardcode the 8 here
    auto S_ptr =
        static_cast<cutlass::Array<ElementScale, ScalePackSize> const*>(
            group_scales.const_data_ptr());

    // runtime layout for B
    auto shape_B = cute::make_shape(n, k, 1);
    LayoutB_Reordered layout_B_reordered =
        cute::tile_to_shape(LayoutAtomQuant{}, shape_B);

    // strides
    int const scale_k = cutlass::ceil_div(k, group_size_int);
    StrideA stride_A =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    // Reverse stride here due to swap and transpose
    StrideD stride_D =
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));
    StrideS stride_S = cutlass::make_cute_packed_stride(
        StrideS{}, cute::make_shape(n, scale_k, 1));

    // Create a structure of gemm kernel arguments suitable for invoking an
    // instance of Gemm auto arguments =
    // args_from_options<GemmShuffled>(options);
    /// Populates a Gemm::Arguments structure from the given arguments
    /// Swap the A and B tensors, as well as problem shapes here.
    using Args = typename GemmShuffled::Arguments;
    using MainloopArguments = typename GemmKernelShuffled::MainloopArguments;
    using EpilogueArguments = typename GemmKernelShuffled::EpilogueArguments;

    MainloopArguments mainloop_arguments{
        B_ptr, layout_B_reordered, A_ptr,         stride_A,
        S_ptr, stride_S,           group_size_int};

    EpilogueArguments epilogue_arguments{
        ChTokScalesEpilogue::prepare_args(channel_scales, token_scales),
        nullptr,
        {},  // no C
        D_ptr,
        stride_D};

    Args arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                   {n, m, k, 1},  // shape
                   mainloop_arguments,
                   epilogue_arguments};

    // Workspace
    size_t workspace_size = GemmShuffled::get_workspace_size(arguments);
    torch::Tensor workspace =
        torch::empty(workspace_size,
                     torch::TensorOptions().dtype(torch::kU8).device(device));

    // Run GEMM
    GemmShuffled gemm;
    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
    CUTLASS_CHECK(gemm.run(stream));

    return D;
  }
};

// ----------------------------------------------------------------------------
// Kernel instantiations and dispatch logic
// ----------------------------------------------------------------------------
using Kernel_256x128_1x1x1 =
    W4A8GemmKernel<Shape<_256, _128>, Shape<_1, _1, _1>>;
using Kernel_256x64_1x1x1 = W4A8GemmKernel<Shape<_256, _64>, Shape<_1, _1, _1>>;
using Kernel_256x32_1x1x1 = W4A8GemmKernel<Shape<_256, _32>, Shape<_1, _1, _1>>;
using Kernel_256x16_1x1x1 = W4A8GemmKernel<Shape<_256, _16>, Shape<_1, _1, _1>>;
using Kernel_128x256_2x1x1 =
    W4A8GemmKernel<Shape<_128, _256>, Shape<_2, _1, _1>>;
using Kernel_128x256_1x1x1 =
    W4A8GemmKernel<Shape<_128, _256>, Shape<_1, _1, _1>>;
using Kernel_128x128_1x1x1 =
    W4A8GemmKernel<Shape<_128, _128>, Shape<_1, _1, _1>>;
using Kernel_128x64_1x1x1 = W4A8GemmKernel<Shape<_128, _64>, Shape<_1, _1, _1>>;
using Kernel_128x32_1x1x1 = W4A8GemmKernel<Shape<_128, _32>, Shape<_1, _1, _1>>;
using Kernel_128x16_1x1x1 = W4A8GemmKernel<Shape<_128, _16>, Shape<_1, _1, _1>>;

torch::Tensor mm_dispatch(torch::Tensor const& A,
                          torch::Tensor const& B,             // already packed
                          torch::Tensor const& group_scales,  // already packed
                          int64_t group_size,
                          torch::Tensor const& channel_scales,
                          torch::Tensor const& token_scales,
                          std::optional<at::ScalarType> const& maybe_out_type,
                          const std::string& schedule) {
  if (schedule == "256x128_1x1x1") {
    return Kernel_256x128_1x1x1::mm(A, B, group_scales, group_size,
                                    channel_scales, token_scales,
                                    maybe_out_type);
  } else if (schedule == "256x64_1x1x1") {
    return Kernel_256x64_1x1x1::mm(A, B, group_scales, group_size,
                                   channel_scales, token_scales,
                                   maybe_out_type);
  } else if (schedule == "256x32_1x1x1") {
    return Kernel_256x32_1x1x1::mm(A, B, group_scales, group_size,
                                   channel_scales, token_scales,
                                   maybe_out_type);
  } else if (schedule == "256x16_1x1x1") {
    return Kernel_256x16_1x1x1::mm(A, B, group_scales, group_size,
                                   channel_scales, token_scales,
                                   maybe_out_type);
  } else if (schedule == "128x256_2x1x1") {
    return Kernel_128x256_2x1x1::mm(A, B, group_scales, group_size,
                                    channel_scales, token_scales,
                                    maybe_out_type);
  } else if (schedule == "128x256_1x1x1") {
    return Kernel_128x256_1x1x1::mm(A, B, group_scales, group_size,
                                    channel_scales, token_scales,
                                    maybe_out_type);
  } else if (schedule == "128x128_1x1x1") {
    return Kernel_128x128_1x1x1::mm(A, B, group_scales, group_size,
                                    channel_scales, token_scales,
                                    maybe_out_type);
  } else if (schedule == "128x64_1x1x1") {
    return Kernel_128x64_1x1x1::mm(A, B, group_scales, group_size,
                                   channel_scales, token_scales,
                                   maybe_out_type);
  } else if (schedule == "128x32_1x1x1") {
    return Kernel_128x32_1x1x1::mm(A, B, group_scales, group_size,
                                   channel_scales, token_scales,
                                   maybe_out_type);
  } else if (schedule == "128x16_1x1x1") {
    return Kernel_128x16_1x1x1::mm(A, B, group_scales, group_size,
                                   channel_scales, token_scales,
                                   maybe_out_type);
  }
  TORCH_CHECK(false, "Unknown W4A8 schedule: ", schedule);
  return {};
}

torch::Tensor mm(torch::Tensor const& A,
                 torch::Tensor const& B,             // already packed
                 torch::Tensor const& group_scales,  // already packed
                 int64_t group_size, torch::Tensor const& channel_scales,
                 torch::Tensor const& token_scales,
                 std::optional<at::ScalarType> const& maybe_out_type,
                 std::optional<std::string> maybe_schedule) {
  // requested a specific schedule
  if (maybe_schedule) {
    return mm_dispatch(A, B, group_scales, group_size, channel_scales,
                       token_scales, maybe_out_type, *maybe_schedule);
  }
  std::string schedule;
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  // heuristic
  if (M <= 16) {
    schedule = (K == 16384 && N == 18432) ? "256x16_1x1x1" : "128x16_1x1x1";
  } else if (M <= 32) {
    schedule = (K == 16384 && N == 18432) ? "256x32_1x1x1" : "128x32_1x1x1";
  } else if (M <= 64) {
    if (K == 16384 && N == 18432)
      schedule = "256x64_1x1x1";
    else if (N <= 8192 && K <= 8192)
      schedule = "128x32_1x1x1";
    else
      schedule = "128x64_1x1x1";
  } else if (M <= 128) {
    if (K == 16384 && N == 18432)
      schedule = "256x128_1x1x1";
    else if (N <= 8192)
      schedule = "128x64_1x1x1";
    else
      schedule = "128x128_1x1x1";
  } else if (M <= 256) {
    if (N <= 4096)
      schedule = "128x64_1x1x1";
    else if (N <= 8192)
      schedule = "128x128_1x1x1";
    else
      schedule = "128x256_1x1x1";
  } else if (M <= 512 && N <= 4096) {
    schedule = "128x128_1x1x1";
  } else if (M <= 1024) {
    schedule = "128x256_1x1x1";
  } else {
    schedule = "128x256_2x1x1";
  }
  return mm_dispatch(A, B, group_scales, group_size, channel_scales,
                     token_scales, maybe_out_type, schedule);
}

// ----------------------------------------------------------------------------
// Pre-processing utils
// ----------------------------------------------------------------------------
torch::Tensor pack_scale_fp8(torch::Tensor const& scales) {
  TORCH_CHECK(scales.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(scales.is_contiguous());
  TORCH_CHECK(scales.is_cuda());

  auto packed_scales = torch::empty(
      {scales.numel() * ScalePackSize},
      torch::TensorOptions().dtype(scales.dtype()).device(scales.device()));
  auto scales_ptr = static_cast<MmaType const*>(scales.const_data_ptr());
  auto packed_scales_ptr =
      static_cast<cutlass::Array<ElementScale, ScalePackSize>*>(
          packed_scales.data_ptr());

  cutlass::pack_scale_fp8(scales_ptr, packed_scales_ptr, scales.numel());

  return packed_scales;
}

torch::Tensor encode_and_reorder_int4b(torch::Tensor const& B) {
  TORCH_CHECK(B.dtype() == torch::kInt32);
  TORCH_CHECK(B.dim() == 2);

  torch::Tensor B_packed = torch::empty_like(B);

  int k = B.size(0) * PackFactor;  // logical k
  int n = B.size(1);
  TORCH_CHECK((n * k) % 32 == 0, "need multiples of 32 int4s for 16B chunks");

  auto B_ptr = static_cast<QuantType const*>(B.const_data_ptr());
  auto B_packed_ptr = static_cast<QuantType*>(B_packed.data_ptr());
  auto shape_B = cute::make_shape(n, k, 1);
  auto layout_B = make_layout(shape_B, LayoutRight{});  // row major
  LayoutB_Reordered layout_B_reordered =
      cute::tile_to_shape(LayoutAtomQuant{}, shape_B);

  bool ok = vllm::cutlass_w4a8_utils::unified_encode_int4b(B_ptr, B_packed_ptr,
                                                           n * k);
  TORCH_CHECK(ok, "unified_encode_int4b failed");
  cutlass::reorder_tensor(B_packed_ptr, layout_B, layout_B_reordered);

  return B_packed;
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("cutlass_w4a8_mm", &mm);
  m.impl("cutlass_pack_scale_fp8", &pack_scale_fp8);
  m.impl("cutlass_encode_and_reorder_int4b", &encode_and_reorder_int4b);
}

}  // namespace vllm::cutlass_w4a8