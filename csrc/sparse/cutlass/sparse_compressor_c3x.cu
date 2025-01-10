// clang-format will break include orders
// clang-format off
#include <cudaTypedefs.h>

#if defined CUDA_VERSION && CUDA_VERSION >= 12020
#include "sparse_scaled_mm_c3x.cuh"

#include "cutlass/numeric_conversion.h"
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

using namespace cute;
using namespace vllm;

/// Make A structured sparse by replacing elements with 0 and compress it
template <typename ElementA_, typename ElementAcc_>
bool cutlass_sparse_compress(torch::Tensor& a_nzs, torch::Tensor& a_meta,
                             torch::Tensor const& a) {
  // Checks for conformality
  TORCH_CHECK(a.dtype() == torch::kInt8 || a.dtype() == torch::kFloat8_e4m3fn ||
              a.dtype() == torch::kFloat16 || a.dtype() == torch::kBFloat16);
  TORCH_CHECK(a.dim() == 2)
  // Check for strides and alignment
  TORCH_CHECK(a.stride(0) % 4 == 0)  // Required for semi-structured sparsity
  TORCH_CHECK(a.stride(1) == 1)

  int m = a.size(0);
  int k = a.size(1);

  // Sparse kernel setup; this kernel is not used for matmul,
  // but just for setting up the compressor utility
  // A matrix configuration
  using ElementA = ElementA_;
  using LayoutTagA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  // B matrix configuration
  using ElementB = ElementA;
  using LayoutTagB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  // C/D matrix configuration
  using ElementC = float;
  using LayoutTagC = cutlass::layout::ColumnMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  // Core kernel configurations
  using ElementAccumulator = ElementAcc_;
  using TileShape = Shape<_128, _128, _128>;
  using TileShapeRef = Shape<_128, _128, _64>;
  using ClusterShape = Shape<_1, _2, _1>;
  using KernelSchedule = typename std::conditional<
      std::is_same_v<ElementA, cutlass::float_e4m3_t>,
      cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum,
      cutlass::gemm::KernelTmaWarpSpecialized>::type;

  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using ProblemShape = Shape<int, int, int, int>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator, ElementC, LayoutTagC,
          AlignmentC, ElementC, LayoutTagC, AlignmentC,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassSparseTensorOp, ElementA,
          LayoutTagA, AlignmentA, ElementB, LayoutTagB, AlignmentB,
          ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
  using StrideE = StrideA;

  using StrideA = Stride<int64_t, Int<1>, int64_t>;

  // The n (=1) dimension does not matter for the compressor
  typename GemmKernel::ProblemShape prob_shape{m, 1, k, 1};

  using LayoutA = typename GemmKernel::CollectiveMainloop::LayoutA;
  using LayoutE = typename GemmKernel::CollectiveMainloop::LayoutE;

  using ElementE = typename GemmKernel::CollectiveMainloop::ElementE;
  using SparseConfig = typename GemmKernel::CollectiveMainloop::SparseConfig;

  // Offline compressor kernel
  using CompressorUtility =
      cutlass::transform::kernel::StructuredSparseCompressorUtility<
          ProblemShape, ElementA, LayoutTagA, SparseConfig>;

  using CompressorKernel =
      cutlass::transform::kernel::StructuredSparseCompressor<
          ProblemShape, ElementA, LayoutTagA, SparseConfig,
          cutlass::arch::Sm90>;

  using Compressor =
      cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

  auto [M, N, K, L] = prob_shape;

  StrideA stride_A;
  stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));

  CompressorUtility compressor_utility(prob_shape, stride_A);

  int ME = compressor_utility.get_metadata_m_physical();
  int KE = compressor_utility.get_metadata_k_physical();
  int KC = compressor_utility.get_tensorA_k_physical();

  auto a_ptr = static_cast<ElementA*>(a.data_ptr());

  auto a_nzs_ptr = static_cast<ElementA*>(a_nzs.data_ptr());
  auto a_meta_ptr = static_cast<typename Gemm::CollectiveMainloop::ElementE*>(
      a_meta.data_ptr());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);
  typename Compressor::Arguments arguments{
      prob_shape, {a_ptr, stride_A, a_nzs_ptr, a_meta_ptr}, {hw_info}};

  Compressor compressor_op;
  size_t workspace_size = Compressor::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(compressor_op.can_implement(arguments));
  CUTLASS_CHECK(compressor_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(compressor_op.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  return true;
}

bool cutlass_sparse_compress_sm90(torch::Tensor& a_nzs, torch::Tensor& a_meta,
                                  torch::Tensor const& a) {
  if (a.dtype() == torch::kBFloat16) {
    return cutlass_sparse_compress<cutlass::bfloat16_t, float>(a_nzs, a_meta,
                                                               a);
  } else if (a.dtype() == torch::kFloat16) {
    return cutlass_sparse_compress<cutlass::half_t, float>(a_nzs, a_meta, a);
  } else if (a.dtype() == torch::kFloat8_e4m3fn) {
    return cutlass_sparse_compress<cutlass::float_e4m3_t, float>(a_nzs, a_meta,
                                                                 a);
  } else if (a.dtype() == torch::kInt8) {
    return cutlass_sparse_compress<int8_t, int32_t>(a_nzs, a_meta, a);
  }
  return false;
}
#endif
