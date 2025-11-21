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

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type *, AlignmentC,
    ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type *, AlignmentD,
    EpilogueSchedule
  >::CollectiveOp;

// =========================================================== MIXED INPUT WITH SCALES ===========================================================================
// The Scale information must get paired with the operand that will be scaled. In this example, B is scaled so we make a tuple of B's information and the scale information.
using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementB, cutlass::Array<ElementScale, 8>>, LayoutB_Transpose *, AlignmentB,
    ElementA, LayoutA_Transpose *, AlignmentA,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernelScaleOnly = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, 
    CollectiveMainloopScaleOnly,
    CollectiveEpilogue
>;

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

using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;
using GemmShuffled  = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelShuffled>;

using StrideC = typename GemmKernelScaleOnly::InternalStrideC;
using StrideD = typename GemmKernelScaleOnly::InternalStrideD;

using StrideC_ref = cutlass::detail::TagToStrideC_t<LayoutC>;
using StrideD_ref = cutlass::detail::TagToStrideC_t<LayoutD>;
using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;
using StrideS_ref = cutlass::detail::TagToStrideB_t<LayoutScale>;

// Host-side allocations
std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_B_dq;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;
std::vector<int64_t> offset_scale;
std::vector<int64_t> offset_zero;

std::vector<StrideA> stride_A_host;
std::vector<StrideB> stride_B_host;
std::vector<StrideC> stride_C_host;
std::vector<StrideD> stride_D_host;
std::vector<StrideC_ref> stride_C_host_ref;
std::vector<StrideD_ref> stride_D_host_ref;
std::vector<StrideS> stride_S_host;
std::vector<StrideS_ref> stride_S_host_ref;

std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;

uint64_t seed = 2020;

// Device-side allocations
cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

cutlass::DeviceAllocation<MmaType> block_A;
cutlass::DeviceAllocation<QuantType> block_B;
cutlass::DeviceAllocation<ElementB> block_B_modified;
cutlass::DeviceAllocation<MmaType> block_B_dq;
cutlass::DeviceAllocation<ElementScale> block_scale;
cutlass::DeviceAllocation<cutlass::Array<ElementScale, 8>> block_scale_packed;
cutlass::DeviceAllocation<ElementZero> block_zero;
cutlass::DeviceAllocation<ElementC> block_C;
cutlass::DeviceAllocation<typename GemmScaleOnly::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename GemmScaleOnly::EpilogueOutputOp::ElementOutput> block_ref_D;

cutlass::DeviceAllocation<const MmaType *> ptr_A;
cutlass::DeviceAllocation<const QuantType *> ptr_B;
cutlass::DeviceAllocation<const MmaType *> ptr_B_dq;
cutlass::DeviceAllocation<const cutlass::Array<ElementScale, 8> *> ptr_scale_packed;
cutlass::DeviceAllocation<const ElementZero *> ptr_zero;
cutlass::DeviceAllocation<const ElementC *> ptr_C;
cutlass::DeviceAllocation<typename GemmScaleOnly::EpilogueOutputOp::ElementOutput *> ptr_D;

cutlass::DeviceAllocation<StrideA> stride_A;
cutlass::DeviceAllocation<StrideB> stride_B;
cutlass::DeviceAllocation<LayoutB_Reordered> layout_B_reordered;
cutlass::DeviceAllocation<StrideC> stride_C;
cutlass::DeviceAllocation<StrideD> stride_D;
cutlass::DeviceAllocation<StrideC_ref> stride_C_ref;
cutlass::DeviceAllocation<StrideD_ref> stride_D_ref;
cutlass::DeviceAllocation<StrideS_ref> stride_S_ref;
cutlass::DeviceAllocation<StrideS> stride_S;

// Note, this is an array of pointers to alpha and beta scaling values per group
cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
cutlass::DeviceAllocation<ElementAccumulator> block_beta;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// debug utils
template <typename ElementC>
void print_ptr_tensor(const torch::Tensor& ptr_tensor) {
  // Expect a 1D tensor of int64 device pointers on CUDA
  TORCH_CHECK(ptr_tensor.dim() == 1, "ptr_tensor must be 1D");
  TORCH_CHECK(ptr_tensor.device().is_cuda(), "ptr_tensor must be on CUDA");
  TORCH_CHECK(ptr_tensor.scalar_type() == torch::kLong,
              "ptr_tensor must have dtype int64 (kLong), storing raw pointers");

  // Move to CPU so we can safely read the values
  torch::Tensor ptr_host = ptr_tensor.to(torch::kCPU).contiguous();
  auto* data = ptr_host.data_ptr<int64_t>();
  const int64_t num = ptr_host.size(0);

  printf("ptr_tensor (size = %lld):\n", static_cast<long long>(num));

  if (num == 0) {
    return;
  }

  // Baseline pointer for delta computation
  ElementC* base_ptr = reinterpret_cast<ElementC*>(data[0]);

  for (int64_t i = 0; i < num; ++i) {
    ElementC* p = reinterpret_cast<ElementC*>(data[i]);
    printf("  [%2lld]  %p", static_cast<long long>(i), (void*)p);

    if (i == 0) {
      printf("  (baseline)\n");
    } else {
      ptrdiff_t delta_bytes = reinterpret_cast<char*>(p) -
                              reinterpret_cast<char*>(base_ptr);
      ptrdiff_t delta_elems = delta_bytes / sizeof(ElementC);
      printf("  (delta: %td bytes, %td elems)\n", delta_bytes, delta_elems);
    }
  }
}

// Command line options parsing
struct Options : GroupedMixedDtypeOptions<QuantType> {
  using Base = GroupedMixedDtypeOptions<QuantType>;

  bool shuffle = true;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    cmd.get_cmd_line_argument("shuffle", shuffle);

    this->Base::parse(argc, args);

    mode = 1; // override the mode value to always be scale only mode
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "69_hopper_int4_fp8_grouped_gemm\n\n"
      << "  Hopper Mixed Dtype Grouped GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM for all groups\n"
      << "  --n=<int>                   Sets the N extent of the GEMM for all groups\n"
      << "  --k=<int>                   Sets the K extent of the GEMM for all groups\n"
      << "  --groups=<int>              Sets the number of individual GEMM problems for Grouped GEMM\n"
      << "  --c=<int>                   The size of each chunk for the scales and zeros. To broadcast a vector of scales or zeros, set the group size to K.\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform\n\n"
      << "  --warmup=<int>              Number of warmup iterations to perform\n\n"
      << "  --shuffle=<boolean>         Enable the offline layout swizzling.\n\n"
      << "  --benchmark=<str>           Executes a benchmark problem size.\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "69_hopper_int4_fp8_grouped_gemm" << " --m=1024 --n=512 --k=1024 --groups=10 --alpha=1 --beta=0 \n\n";

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

// In the mainloop, PRMT selects 1 byte from only 8 bytes so the sign bit is handled in an extra PRMT.
// Here the encodings of positive values and negative values are unified (except for the sign bit). 
// For instance, 1 becomes 0b0111, which is the same encoding as -1 (0b1111).

/// Allocates device-side data
void allocate(Options const& options) {
  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_B_dq = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;
  int64_t total_elements_scale = 0;
  int64_t total_elements_zero = 0;

  for (int32_t i = 0; i < options.groups; ++i) {

    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    int const scale_k = cutlass::ceil_div(options.k, options.c);

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B * cutlass::sizeof_bits<QuantType>::value / 8);
    offset_B_dq.push_back(total_elements_B_dq);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);
    offset_scale.push_back(total_elements_scale);
    offset_zero.push_back(total_elements_zero);

    int64_t elements_A = M * K;
    int64_t elements_B = K * N ;
    int64_t elements_B_dq = K * N;
    int64_t elements_C = M * N;
    int64_t elements_D = M * N;
    int64_t elements_scale = scale_k * N;
    int64_t elements_zero = scale_k * N;

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_B_dq += elements_B_dq;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
    total_elements_scale += elements_scale;
    total_elements_zero += elements_zero;

    stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {N, M, 1}));
    stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {N, M, 1}));
    stride_C_host_ref.push_back(cutlass::make_cute_packed_stride(StrideC_ref{}, {M, N, 1}));
    stride_D_host_ref.push_back(cutlass::make_cute_packed_stride(StrideD_ref{}, {M, N, 1}));
    stride_S_host_ref.push_back(cutlass::make_cute_packed_stride(StrideS_ref{}, {N, scale_k, 1}));
    stride_S_host.push_back(cutlass::make_cute_packed_stride(StrideS{}, {N, scale_k, 1}));
  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_B_modified.reset(total_elements_B);
  block_B_dq.reset(total_elements_B_dq);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_ref_D.reset(total_elements_D);
  block_scale.reset(total_elements_scale);
  block_scale_packed.reset(total_elements_scale);
  block_zero.reset(total_elements_zero);

  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(Options& options) {

  uint64_t seed = 2020;

  problem_sizes.reset(options.groups);
  problem_sizes.copy_from_host(options.problem_sizes_host.data());

  //
  // Assign pointers
  //

  std::vector<MmaType *> ptr_A_host(options.groups);
  std::vector<QuantType *> ptr_B_host(options.groups);
  std::vector<MmaType *> ptr_B_dq_host(options.groups);
  std::vector<ElementC *> ptr_C_host(options.groups);
  std::vector<ElementC *> ptr_D_host(options.groups);
  std::vector<cutlass::Array<ElementScale, 8> *> ptr_scale_packed_host(options.groups);
  std::vector<ElementZero *> ptr_zero_host(options.groups);
  std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

  for (int32_t i = 0; i < options.groups; ++i) {
    ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i) = block_B_modified.get() + offset_B.at(i);
    ptr_B_dq_host.at(i) = block_B_dq.get() + offset_B_dq.at(i);
    ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    ptr_scale_packed_host.at(i) = block_scale_packed.get() + offset_scale.at(i);
    ptr_zero_host.at(i) = block_zero.get() + offset_zero.at(i);
    alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
    beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
    ptr_alpha_host.at(i) = block_alpha.get() + i;
    ptr_beta_host.at(i) = block_beta.get() + i;
  }

  ptr_A.reset(options.groups);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.groups);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_B_dq.reset(options.groups);
  ptr_B_dq.copy_from_host(ptr_B_dq_host.data());

  ptr_C.reset(options.groups);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.groups);
  ptr_D.copy_from_host(ptr_D_host.data());

  ptr_scale_packed.reset(options.groups);
  ptr_scale_packed.copy_from_host(ptr_scale_packed_host.data());

  ptr_zero.reset(options.groups);
  ptr_zero.copy_from_host(ptr_zero_host.data());

  stride_A.reset(options.groups);
  stride_A.copy_from_host(stride_A_host.data());

  stride_B.reset(options.groups);
  stride_B.copy_from_host(stride_B_host.data());

  stride_C.reset(options.groups);
  stride_C.copy_from_host(stride_C_host.data());

  stride_D.reset(options.groups);
  stride_D.copy_from_host(stride_D_host.data());

  stride_C_ref.reset(options.groups);
  stride_C_ref.copy_from_host(stride_C_host_ref.data());

  stride_D_ref.reset(options.groups);
  stride_D_ref.copy_from_host(stride_D_host_ref.data());

  stride_S_ref.reset(options.groups);
  stride_S_ref.copy_from_host(stride_S_host_ref.data());

  stride_S.reset(options.groups);
  stride_S.copy_from_host(stride_S_host.data());

  alpha_device.reset(options.groups);
  alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);
  beta_device.copy_from_host(ptr_beta_host.data());

  initialize_tensor(block_A, seed + 2023);
  initialize_tensor(block_B, seed + 2022);
  cutlass::unified_encode_int4b(block_B.get(), block_B_modified.get(), block_B.size());
  initialize_tensor(block_C, seed + 2021);
  initialize_scale(block_scale, options);
  cutlass::pack_scale_fp8(block_scale.get(), block_scale_packed.get(), block_scale.size());
  initialize_zero(block_zero, options);
  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());

  problem_sizes.reset(options.groups);
  // since we reconstructing strides for B at runtime, we just need the reorder_tensor part here.
  if (options.shuffle) {
    std::vector<LayoutB_Reordered> layout_B_reordered_host(options.groups);
    for (int32_t i = 0; i < options.groups; ++i) {
      // logical n, k (before transpose)
      auto shape_B = cute::make_shape(cute::get<1>(options.problem_sizes_host[i]), cute::get<2>(options.problem_sizes_host[i]), Int<1>{});
      auto layout_B = make_layout(shape_B, stride_B_host.at(i));
      // Repeat the reorder layout atom to tile the whole tensor shape 
      layout_B_reordered_host[i] = tile_to_shape(LayoutAtomQuant{}, shape_B);
      cutlass::reorder_tensor(block_B_modified.get() + offset_B.at(i), layout_B, layout_B_reordered_host[i]);
      if (i == 0) {
        print("Quantized tensor layout: ");
        print(layout_B_reordered_host[0]);
        print("\n");
      }
    }
    layout_B_reordered.reset(options.groups);
    layout_B_reordered.copy_from_host(layout_B_reordered_host.data());

      printf("layout_B_reordered_host (size = %zu):\n", layout_B_reordered_host.size());

    for (size_t i = 0; i < layout_B_reordered_host.size(); ++i) {
        printf("  [%2zu] = ", i);
        cute::print(layout_B_reordered_host[i]);   // prints the tuple
        printf("\n");
    }
  }

  // Reverse MN -> NM for SwapAB
  // try what happens if we dont do this - nothing? dafuq
  for (int32_t i = 0; i < options.groups; ++i) {
    auto [M, N, K] = options.problem_sizes_host[i];
    options.problem_sizes_host[i] = make_tuple(N, M, K);
  }
  problem_sizes.copy_from_host(options.problem_sizes_host.data());

  // print stuff since it is on the host already
  printf("ptr_D_host (size = %d):\n", options.groups);
  for (int i = 0; i < options.groups; ++i) {
      ElementC* p = ptr_D_host[i];
      printf("  [%2d]  %p", i, (void*)p);

      // Optional: also print deltas relative to first expert
      if (i == 0) {
          printf("  (baseline)\n");
      } else {
          ptrdiff_t delta_bytes = (char*)p - (char*)ptr_D_host[0];
          ptrdiff_t delta_elems = delta_bytes / sizeof(ElementC);
          printf("  (delta: %td bytes, %td elems)\n", delta_bytes, delta_elems);
      }
  }
  printf("ptr_A_host (size = %d):\n", options.groups);
  for (int i = 0; i < options.groups; ++i) {
      ElementA* p = ptr_A_host[i];
      printf("  [%2d]  %p", i, (void*)p);

      // Optional: also print deltas relative to first expert
      if (i == 0) {
          printf("  (baseline)\n");
      } else {
          ptrdiff_t delta_bytes = (char*)p - (char*)ptr_A_host[0];
          ptrdiff_t delta_elems = delta_bytes / sizeof(ElementA);
          printf("  (delta: %td bytes, %td elems)\n", delta_bytes, delta_elems);
      }
  }

  printf("ptr_B_host (size = %d):\n", options.groups);
  for (int i = 0; i < options.groups; ++i) {
      ElementB* p = ptr_B_host[i];
      printf("  [%2d]  %p", i, (void*)p);

      // Optional: also print deltas relative to first expert
      if (i == 0) {
          printf("  (baseline)\n");
      } else {
          ptrdiff_t delta_bytes = (char*)p - (char*)ptr_B_host[0];
          ptrdiff_t delta_elems = delta_bytes / sizeof(ElementB);
          printf("  (delta: %td bytes, %td elems)\n", delta_bytes, delta_elems);
      }
  }
  
  printf("ptr_scale_packed_host (size = %d):\n", options.groups);
  for (int i = 0; i < options.groups; ++i) {
      cutlass::Array<ElementScale, 8>* p = ptr_scale_packed_host[i];
      printf("  [%2d]  %p", i, (void*)p);

      // Optional: also print deltas relative to first expert
      if (i == 0) {
          printf("  (baseline)\n");
      } else {
          ptrdiff_t delta_bytes = (char*)p - (char*)ptr_scale_packed_host[0];
          ptrdiff_t delta_elems = delta_bytes / sizeof(cutlass::Array<ElementScale, 8>);
          printf("  (delta: %td bytes, %td elems)\n", delta_bytes, delta_elems);
      }
  }
  // strides info
  printf("stride_A_host (size = %zu):\n", stride_A_host.size());
  for (size_t i = 0; i < stride_A_host.size(); ++i) {
      printf("  [%2zu] = ", i);
      cute::print(stride_A_host[i]);   // prints the tuple
      printf("\n");
  }
  printf("stride_B_host (size = %zu):\n", stride_B_host.size());
  for (size_t i = 0; i < stride_B_host.size(); ++i) {
      printf("  [%2zu] = ", i);
      cute::print(stride_B_host[i]);   // prints the tuple
      printf("\n");
  }
  printf("stride_C_host (size = %zu):\n", stride_C_host.size());
  for (size_t i = 0; i < stride_C_host.size(); ++i) {
      printf("  [%2zu] = ", i);
      cute::print(stride_C_host[i]);   // prints the tuple
      printf("\n");
  }
  printf("stride_D_host (size = %zu):\n", stride_D_host.size());
  for (size_t i = 0; i < stride_D_host.size(); ++i) {
      printf("  [%2zu] = ", i);
      cute::print(stride_D_host[i]);   // prints the tuple
      printf("\n");
  }

  printf("stride_S_host (size = %zu):\n", stride_S_host.size());

  for (size_t i = 0; i < stride_S_host.size(); ++i) {
      printf("  [%2zu] = ", i);
      cute::print(stride_S_host[i]);   // prints the tuple
      printf("\n");
  }

  // try different print
  for (int i = 0; i < ptr_A_host.size(); i++) {
    printf("ptr_A[%d] = %p\n", i, (void*)ptr_A_host[i]);
  }
}

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
    Options options;
    options.parse(0, nullptr);
    printf("Experts=%d group size=%d\n", options.groups, options.c);
    for(int i = 0; i < options.problem_sizes_host.size(); i++){
      auto problem = options.problem_sizes_host.at(i);
      auto M = cute::get<0>(problem);
      auto N = cute::get<1>(problem);
      auto K = cute::get<2>(problem);
      printf("Problem %d: M=%d N=%d K=%d\n", i, M, N, K);
    }
    printf("alpha=%f beta=%f\n", options.alpha, options.beta);
    allocate(options);
    initialize(options);

    printf("after initialization/allocate\n");
    printf("Experts=%d group size=%d\n", options.groups, options.c);
    for(int i = 0; i < options.problem_sizes_host.size(); i++){
      auto problem = options.problem_sizes_host.at(i);
      auto M = cute::get<0>(problem);
      auto N = cute::get<1>(problem);
      auto K = cute::get<2>(problem);
      printf("Problem %d: M=%d N=%d K=%d\n", i, M, N, K);
    }

    // get args ---------------------------------
    using Args = typename GemmShuffled::Arguments;
    cutlass::KernelHardwareInfo hw_info;
    // Change device_id to another value if you are running on a machine with multiple GPUs and wish
    // to use a GPU other than that with device ID 0.
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    Args arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
    fusion_args.alpha = options.alpha;
    fusion_args.beta = options.beta;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // Single alpha and beta for all groups
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

    // reconstruct b and S stride
    // TODO: need some way to serialize this info to torch so we don't have to rebuild each time
    // perhaps through the pack methods
    cutlass::DeviceAllocation<LayoutB_Reordered> layout_B_reordered_local;
    cutlass::DeviceAllocation<StrideS> stride_S_local;
    std::vector<LayoutB_Reordered> layout_B_reordered_host(options.groups);
    std::vector<StrideS> stride_S_host_local;
    int const scale_k = cutlass::ceil_div(options.k, options.c);
    for (int32_t i = 0; i < options.groups; ++i) {
      // this happens after initialize (problem shape transposed) so we need to swap it, gets logical N, K
      auto shape_B = cute::make_shape(cute::get<0>(options.problem_sizes_host[i]), cute::get<2>(options.problem_sizes_host[i]), Int<1>{});
      // Repeat the reorder layout atom to tile the whole tensor shape 
      layout_B_reordered_host[i] = tile_to_shape(LayoutAtomQuant{}, shape_B);
      // logical N, scale_k
      stride_S_host_local.push_back(cutlass::make_cute_packed_stride(StrideS{}, {cute::get<0>(options.problem_sizes_host[i]), scale_k, 1}));
    }
    // copy to device
    layout_B_reordered_local.reset(options.groups);
    layout_B_reordered_local.copy_from_host(layout_B_reordered_host.data());
    stride_S_local.reset(options.groups);
    stride_S_local.copy_from_host(stride_S_host_local.data());

    // try to replace ptr_D
    auto device = a_tensors.device();
    int num_experts = static_cast<int>(expert_offsets.size(0));
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
    
    // check values - this looks fine
    // print_ptr_tensor<ElementC>(out_ptrs);
    // print_ptr_tensor<ElementA>(a_ptrs);
    // print_ptr_tensor<ElementB>(b_ptrs);
    // print_ptr_tensor<cutlass::Array<ElementScale, 8>>(b_group_scales_ptrs);
    
    // compare pointer values starting with A
    torch::Tensor a_ptrs_cpu = a_ptrs.cpu();
    auto p = a_ptrs_cpu.data_ptr<int64_t>();
    for (int i = 0; i < a_ptrs_cpu.size(0); i++) {
        printf("a_ptrs[%d] = %p\n", i, (void*)p[i]);
    }
    // torch problem sizes
    ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<ProblemShape::UnderlyingProblemShape*>(
          problem_sizes_torch.data_ptr());
    arguments = Args {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      {
        static_cast<const QuantType**>(b_ptrs.data_ptr()), layout_B_reordered_local.get(),
        static_cast<const MmaType**>(a_ptrs.data_ptr()), static_cast<StrideA*>(a_strides.data_ptr()),
        static_cast<const cutlass::Array<ElementScale, 8> **>(b_group_scales_ptrs.data_ptr()),
        stride_S_local.get(),
        // static_cast<StrideS*>(group_scale_strides.data_ptr()), // this leads to illegal memory access
        static_cast<int>(b_group_size)},
      {fusion_args, nullptr, static_cast<StrideC*>(c_strides.data_ptr()), // epilogue should be good
        static_cast<ElementD**>(out_ptrs.data_ptr()), static_cast<StrideC*>(c_strides.data_ptr())},
      hw_info
    };

    GemmShuffled gemm;
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = GemmShuffled::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

    // Correctness / Warmup iteration
    CUTLASS_CHECK(gemm.run());
    print("grouped gemm done!\n");

    // print/debug output
    // First compute total output size (sum of all experts)
    size_t total_output_elems = 9240576; // 4512 x 2048

    // Copy device → host
    std::vector<ElementD> host_D(total_output_elems);
    cudaError_t err = cudaMemcpy(host_D.data(),
              block_D.get(),
              total_output_elems * sizeof(ElementD),
              cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    }
    // -------------------------------------------------------
    // Print a few random values from each expert output
    // -------------------------------------------------------
    // print last few values
    print("Sampled output values:\n");
    print("D[0..9]:\n");
    for (int i = 0; i < 10; ++i) {
        print(host_D[i]); print(" ");
    }
    print("D[-10..-1]:\n");
    for (int i = total_output_elems-10; i < total_output_elems; ++i) {
        print(host_D[i]); print(" ");
    }
    print("\n");
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
    auto offset = i * n * k * cutlass::sizeof_bits<QuantType>::value / 8;
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