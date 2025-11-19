//
// Based off of:
//   https://github.com/NVIDIA/cutlass/blob/main/examples/69_hopper_mixed_dtype_grouped_gemm/69_hopper_int4_fp8_grouped_gemm.cu
//

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include "cutlass_extensions/torch_utils.hpp"

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
#include "get_group_starts.cuh"

namespace vllm::cutlass_w4a8_moe {

using namespace cute;

// -------------------------------------------------------------------------------------
// Static configuration shared across all instantiations
// -------------------------------------------------------------------------------------
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M, N, K> per group
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
// Need to pass a pointer type to make the 3rd dimension of Stride be _0
// using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
// try this instead
  using StrideA = Stride<int64_t, Int<1>, Int<0>>;
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
using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

// Define the CuTe layout for reordered quantized tensor B
// LayoutAtomQuant places values that will be read by the same thread in
// contiguous locations in global memory. It specifies the reordering within a
// single warp's fragment
using LayoutAtomQuant =
    decltype(cutlass::compute_memory_reordering_atom<MmaType>());
using LayoutB_Reordered = decltype(cute::tile_to_shape(LayoutAtomQuant{}, Layout<Shape<int,int,Int<1>>, StrideB>{}));

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
// tileshapes/clustershapes are template parameters to the kernel
using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
// i forgot to change this and got rekt
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative; // Epilogue to launch
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;


// ----------------------------------------------------------------------------
// Kernel template — Tile/Cluster shapes
// ----------------------------------------------------------------------------
template <class TileShape_MN, class ClusterShape_MNK>
struct W4A8GroupedGemmKernel {
  using TileShape =
      decltype(cute::append(TileShape_MN{}, cute::Int<TileShapeK>{}));
  using ClusterShape = ClusterShape_MNK;

  // Epilogue per-tok, per-chan scales
  // the api same as ScaledEpilogue? see scaled_mm_epilogues_c3x.hpp
  using ChTokScalesEpilogue =
      typename vllm::c3x::ScaledEpilogueArray<ElementAccumulator, ElementD,
                                         TileShape>;
// TODO: check if this is the correct way to specify the evt compute
// see grouped_mm_c3x.cuh epilogue part
  using EVTCompute = typename ChTokScalesEpilogue::EVTCompute;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
          ElementAccumulator, ElementSChannel,
          // Transpose layout of D here since we use explicit swap + transpose
          // the void type for C tells the builder to allocate 0 smem for the C
          // matrix. We can enable this if beta == 0 by changing ElementC to
          // void below.
          ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type *, AlignmentC,
          ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type *, AlignmentD,
          EpilogueSchedule,  // This is the only epi supporting the required
                             // swap + transpose.
          EVTCompute>::CollectiveOp;

  // The Scale information must get paired with the operand that will be scaled.
  // In this example, B is scaled so we make a tuple of B's information and the
  // scale information.
  using CollectiveMainloopShuffled =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          cute::tuple<ElementB, cutlass::Array<ElementScale, 8>>,
          LayoutB_Reordered *, AlignmentB, ElementA, LayoutA_Transpose *,
          AlignmentA, ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernelShuffled = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloopShuffled,
      CollectiveEpilogue
  >;

  using GemmShuffled =
      cutlass::gemm::device::GemmUniversalAdapter<GemmKernelShuffled>;
  
  // works once we use correct schedule (oops)
  using StrideC = typename GemmKernelShuffled::InternalStrideC;
  using StrideD = typename GemmKernelShuffled::InternalStrideD;
  using StrideS = typename CollectiveMainloopShuffled::StrideScale;

  static void grouped_mm(
    torch::Tensor& out_tensors,
    const torch::Tensor& a_tensors,
    const torch::Tensor& b_tensors,
    const torch::Tensor& a_scales,
    const torch::Tensor& b_scales,
    const torch::Tensor& b_group_scales,
    const int64_t b_group_size,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& a_strides,
    const torch::Tensor& b_strides,
    const torch::Tensor& c_strides,
    const torch::Tensor& group_scale_strides
  ) {
    // assume we already did param validation elsewhere
    auto device = a_tensors.device();
    auto device_id = device.index();
    const at::cuda::OptionalCUDAGuard device_guard(device);
    auto stream = at::cuda::getCurrentCUDAStream(device_id);

    // allocate ptrs 
    int num_experts = static_cast<int>(expert_offsets.size(0));
    int n = out_tensors.size(1);
    int k = a_tensors.size(1);

    auto options_int =
      torch::TensorOptions().dtype(torch::kInt64).device(device);
    torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor b_group_scales_ptrs = torch::empty(num_experts, options_int);

    // get the correct offsets
    // TODO: we can modify `run_get_group_gemm_starts` to do the correct thing when
    // there are group scales also 
    // TODO: move to gpu. just keep the calculation on cpu now to check correctness
    // if we skip it the gemm should still technically compile; let's test that first.
    // run will fail cause the ptrs will be invalid
    run_get_group_gemm_starts(expert_offsets, a_ptrs, b_ptrs, out_ptrs,
                            a_scales_ptrs, b_scales_ptrs, b_group_scales_ptrs, a_tensors, b_tensors,
                            out_tensors, a_scales, b_scales, b_group_scales, b_group_size);
    // for now we can have a slow version which does this on CPU
    // iterate through problem_sizes.data_ptr()
    // each entry is (M, N, K) so we might have to transpose it

    // now we can build the grouped gemm args since all pointers are populated
    // Swap the A and B tensors, as well as problem shapes here.
    using Args = typename GemmShuffled::Arguments;
    using MainloopArguments = typename GemmKernelShuffled::MainloopArguments;
    using EpilogueArguments = typename GemmKernelShuffled::EpilogueArguments;

    // dont need concern with per-tok/per-chan scales here, but need group scales
    // SwapAB so B (weights) comes first
    // TODO: this a workaround to just construct the layouts
    // assume we already did encoding + reordering
    cutlass::DeviceAllocation<LayoutB_Reordered> layout_B_reordered;
    std::vector<LayoutB_Reordered> layout_B_reordered_host(num_experts);
    // the stride might need reverse?
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    for (int i = 0; i < num_experts; i++){
      // before transpose
      auto shape_B = cute::make_shape(n, k, Int<1>{});
      auto layout_B = make_layout(shape_B, stride_B);
      layout_B_reordered_host[i] = cute::tile_to_shape(LayoutAtomQuant{}, shape_B);

      if (i == 0) {
        print("Quantized tensor layout: ");
        print(layout_B_reordered_host[0]);
        print("\n");
        print("original layout: ");
        print(layout_B);
        print("\n");
      }
    }
    // copy to device
    layout_B_reordered.reset(num_experts);
    layout_B_reordered.copy_from_host(layout_B_reordered_host.data());

    MainloopArguments mainloop_arguments{
        static_cast<const QuantType**>(b_ptrs.data_ptr()),
        static_cast<LayoutB_Reordered*>(b_strides.data_ptr()),
        static_cast<const MmaType**>(a_ptrs.data_ptr()),
        static_cast<StrideA*>(a_strides.data_ptr()),
        static_cast<const cutlass::Array<ElementScale, 8> **>(b_group_scales_ptrs.data_ptr()),
        static_cast<StrideS*>(group_scale_strides.data_ptr()),
        b_group_size
    };

    EpilogueArguments epilogue_arguments{
        // since we are doing SwapAB the channel scales comes first, then token scales
        ChTokScalesEpilogue::prepare_args( // see ScaledEpilogueArray
            static_cast<const ElementAccumulator**>(b_scales_ptrs.data_ptr()), // per-channel
            static_cast<const ElementAccumulator**>(a_scales_ptrs.data_ptr()), // per-token
            true,
            true
        ),
        nullptr,
        static_cast<StrideC*>(c_strides.data_ptr()),  // this might not be needed? this has some issue
        static_cast<ElementD**>(out_ptrs.data_ptr()),
        static_cast<StrideC*>(c_strides.data_ptr()) // this also has issue
    };

    // get the input problem shape
    // TODO: since we are always swapping AB we need to swap it here
    // OR swap it in the caller
    // CAUTION: the fp8 code path assumes swapAB is enabled depending on M
    // this should not be followed here or we will get rekt
    ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<ProblemShape::UnderlyingProblemShape*>(
          problem_sizes.data_ptr());
    ProblemShape prob_shape{num_experts, problem_sizes_as_shapes, nullptr};
    static const cutlass::KernelHardwareInfo hw_info{
        device_id,
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id)
    };

    Args arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                prob_shape, // TODO: we need to either transpose this or take care of it correctly in the caller
                mainloop_arguments,
                epilogue_arguments,
                hw_info};

    // // Workspace
    size_t workspace_size = GemmShuffled::get_workspace_size(arguments);
    torch::Tensor workspace =
        torch::empty(workspace_size,
                     torch::TensorOptions().dtype(torch::kU8).device(device));

    // // Run grouped GEMM
    GemmShuffled gemm;
    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
    CUTLASS_CHECK(gemm.run(stream));
  }
};

// ----------------------------------------------------------------------------
// Kernel instantiations and dispatch logic
// TODO: sweep the tile/cluster shapes etc. to get good perf for variety of problem shapes
// ----------------------------------------------------------------------------
// default one given in example
using Kernel_128x16_1x1x1 = W4A8GroupedGemmKernel<Shape<_128, _16>, Shape<_1, _1, _1>>;

// void mm_dispatch(torch::Tensor const& A,
//                           torch::Tensor const& B,             // already packed
//                           torch::Tensor const& group_scales,  // already packed
//                           int64_t group_size,
//                           torch::Tensor const& channel_scales,
//                           torch::Tensor const& token_scales,
//                           std::optional<at::ScalarType> const& maybe_out_type,
//                           const std::string& schedule) {
//   if (schedule == "128x16_1x1x1") {
//     Kernel_128x16_1x1x1::mm(A, B, group_scales, group_size,
//                                    channel_scales, token_scales,
//                                    maybe_out_type);
//   }
//   TORCH_CHECK(false, "Unknown W4A8 Grouped GEMM schedule: ", schedule);
//   return;
// }

// just check we can call this function correctly first
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
    return Kernel_128x16_1x1x1::grouped_mm(
        out_tensors, a_tensors, b_tensors, a_scales, b_scales,
        b_group_scales, b_group_size, expert_offsets, problem_sizes,
        a_strides, b_strides, c_strides, group_scale_strides
    );
}

// void _mm(torch::Tensor const& A,
//                  torch::Tensor const& B,             // already packed
//                  torch::Tensor const& group_scales,  // already packed
//                  int64_t group_size, torch::Tensor const& channel_scales,
//                  torch::Tensor const& token_scales,
//                  std::optional<at::ScalarType> const& maybe_out_type,
//                  std::optional<std::string> maybe_schedule) {
//   // requested a specific schedule
//   if (maybe_schedule) {
//     mm_dispatch(A, B, group_scales, group_size, channel_scales,
//                        token_scales, maybe_out_type, *maybe_schedule);
//   }
//   std::string schedule;
//   int M = A.size(0);
//   int K = A.size(1);
//   int N = B.size(1);
//   // TODO: heuristic stuff later
//   schedule = "128x16_1x1x1"; // default
//   mm_dispatch(A, B, group_scales, group_size, channel_scales,
//                      token_scales, maybe_out_type, schedule);
// }

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

/*
  GPU-accelerated implementation of cutlass::unified_encode_int4b.
  Constructs a lookup table in constant memory to map 8 bits
  (two 4-bit values) at a time. Assumes memory is contiguous
  and pointers are 16-byte aligned.
*/
__constant__ uint8_t kNibbleLUT[256];

__global__ void unified_encode_int4b_device(const uint8_t* in, uint8_t* out,
                                            size_t nbytes) {
  constexpr size_t V = sizeof(uint4);  // 16 bytes
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t nthreads = size_t(gridDim.x) * blockDim.x;
  const size_t nvec = nbytes / V;

  // 1-D grid-stride loop over 16-byte chunks
  for (size_t vec = tid; vec < nvec; vec += nthreads) {
    uint4 v = reinterpret_cast<const uint4*>(in)[vec];
    uint8_t* b = reinterpret_cast<uint8_t*>(&v);
#pragma unroll
    for (int i = 0; i < int(V); ++i) b[i] = kNibbleLUT[b[i]];
    reinterpret_cast<uint4*>(out)[vec] = v;
  }
}

static bool upload_lut() {
  std::array<uint8_t, 256> lut{};
  auto map_nib = [](uint8_t v) -> uint8_t {
    // 1..7 -> (8 - v); keep 0 and 8..15
    return (v == 0 || (v & 0x8)) ? v : uint8_t(8 - v);
  };
  for (int b = 0; b < 256; ++b) {
    uint8_t lo = b & 0xF;
    uint8_t hi = (b >> 4) & 0xF;
    lut[b] = uint8_t((map_nib(hi) << 4) | map_nib(lo));
  }
  cudaError_t e = cudaMemcpyToSymbol(kNibbleLUT, lut.data(), lut.size(),
                                     /*offset=*/0, cudaMemcpyHostToDevice);

  return (e == cudaSuccess);
}

static bool unified_encode_int4b(cutlass::int4b_t const* in,
                                 cutlass::int4b_t* out, size_t num_int4_elems) {
  // Build/upload LUT
  if (!upload_lut()) return false;

  static_assert(sizeof(typename cutlass::int4b_t::Storage) == 1,
                "int4 storage must be 1 byte");
  const size_t nbytes = num_int4_elems >> 1;

  auto* in_bytes = reinterpret_cast<uint8_t const*>(in);
  auto* out_bytes = reinterpret_cast<uint8_t*>(out);

  // kernel launch params
  constexpr int block = 256;
  const size_t nvec = nbytes / sizeof(uint4);  // # of 16B vectors
  int grid = int((nvec + block - 1) / block);
  if (grid == 0) grid = 1;  // ensure we still cover the tail in the kernel

  unified_encode_int4b_device<<<grid, block>>>(in_bytes, out_bytes, nbytes);
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess);
}

// TODO: this needs some other information
// layout_B_reordered
// options.problem_sizes_host[i]
// stride_B_host
// block_B_modified (packed ptr)
// offset_B
// what shape is B actually?
// torch::Tensor encode_and_reorder_int4b(torch::Tensor const& b_tensors) {
//   TORCH_CHECK(b_tensors.dtype() == torch::kInt32);
//   TORCH_CHECK(b_tensors.dim() == 3); // (experts, n, k)

//   torch::Tensor B_packed = torch::empty_like(B);

//   // TODO: need the total number of (logical) elements here
//   int k = B.size(0) * PackFactor;  // logical k
//   int n = B.size(1);
//   TORCH_CHECK((n * k) % 32 == 0, "need multiples of 32 int4s for 16B chunks");

//   auto B_ptr = static_cast<QuantType const*>(B.const_data_ptr());
//   auto B_packed_ptr = static_cast<QuantType*>(B_packed.data_ptr());
//   auto shape_B = cute::make_shape(n, k, 1);
//   auto layout_B = make_layout(shape_B, LayoutRight{});  // row major

//   // TODO: pull out to separte utils file
//   bool ok =
//       vllm::cutlass_w4a8_moe::unified_encode_int4b(B_ptr, B_packed_ptr, n * k);
//   TORCH_CHECK(ok, "unified_encode_int4b failed");

//   // grouped gemm needs to do this reordering for each expert
//   int num_experts = 8; // TODO: need to pipe this in
//   std::vector<LayoutB_Reordered> layout_B_reordered_host(num_experts);
//   for (int32_t i = 0; i < num_experts; ++i){
//     auto shape_B = cute::make_shape(cute::get<1>(options.problem_sizes_host[i]), cute::get<2>(options.problem_sizes_host[i]), Int<1>{});
//     auto layout_B = make_layout(shape_B, stride_B_host.at(i));
//     // Repeat the reorder layout atom to tile the whole tensor shape 
//     layout_B_reordered_host[i] = tile_to_shape(LayoutAtomQuant{}, shape_B);
//     // reorder_tensor(B_packed_ptr, layout_B, layout_B_reordered)
//     // offset_B.at(i) is the number of elements? ok just treat it like that
//     // TODO: need to veirfy this
//     // apparently it works for torch data_ptr too...so offset_b had better be in elements
//     // these are weights though right so i think they should be the same for each expert, only the M changes
//     // another assumption is B is contiguous
//     // sizeof<int4b_t>::value is 4 since it's a 4-bit type
//     cutlass::reorder_tensor(block_B_modified.get() + offset_B.at(i), layout_B, layout_B_reordered_host[i]);
//     if (i == 0){
//         print("Quantized tensor layout: ");
//         print(layout_B_reordered_host[0]);
//         print("\n");
//     }
//   }
// }

//   // we might not need this because we can pass in b_strides directly
//   // in dense gemm this is only used as mainloop args, here we can apss in strides directly
//   layout_B_reordered.reset(num_experts);
//   layout_B_reordered.copy_from_host(layout_B_reordered_host.data());

//   return B_packed;
// }

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("cutlass_w4a8_moe_mm", &mm);
  // TODO: can pull these to common utils file? used in w4a8 and w4a8_moe. but they might need some layout info
  // this needs MmaType, ScalePackSize, ElementScale
//   m.impl("cutlass_pack_scale_fp8", &pack_scale_fp8);
// this needs QuantType, LayoutRight, LayoutB_Reordered, LayoutAtomQuant
// the reorder also works slightly differently for moe
// have it separate for now but should merge it later
  // m.impl("cutlass_encode_and_reorder_int4b_grouped", &encode_and_reorder_int4b);
}

}  // namespace vllm::cutlass_w4a8_moe