#pragma once

// clang-format will break include orders
// clang-format off
#include <cudaTypedefs.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>

#include "cuda_utils.h"
#include "../../torch_utils.h"

#include "cutlass/cutlass.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"

#include "core/math.hpp"
#include "stable/cutlass_extensions/cute_utils.cuh"
#include "stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "stable/cutlass_extensions/common.hpp"
#include "stable/cutlass_extensions/torch_utils.hpp"
// clang-format on

using namespace cute;

/*
   This file defines 2:4 sparse GEMM operations using the CUTLASS 3.x API,
   for NVIDIA GPUs with sm90a (Hopper) or later.
*/

namespace {

// A wrapper for the GEMM kernel that is used to guard against compilation on
// architectures that will never use the kernel. The purpose of this is to
// reduce the size of the compiled binary.
// __CUDA_ARCH__ is not defined in host code, so this lets us smuggle the ifdef
// into code that will be executed on the device where it is defined.
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

using GemmUniversalMode = cutlass::gemm::GemmUniversalMode;

/*
 * cutlass_sparse_3x_gemm defines a 2:4 sparse GEMM kernel via CUTLASS
 * for SM90 Hopper systems.
 */
template <typename ElementAB_, typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_sparse_3x_gemm {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;
  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutC_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutC>::type;

  using EVTCompute = typename Epilogue::EVTCompute;

  // These are the minimum alignments needed for the kernels to compile
  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc, float, ElementC, LayoutC_Transpose, AlignmentCD, ElementD,
          LayoutC_Transpose, AlignmentCD, EpilogueSchedule,
          EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  // clang-format off
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassSparseTensorOp,
          ElementAB, cutlass::layout::RowMajor, AlignmentAB,
          ElementAB, cutlass::layout::ColumnMajor, AlignmentAB,
          ElementAcc, TileShape, ClusterShape,
          Stages,
          KernelSchedule>::CollectiveOp;
  // clang-format on

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>>;

  struct GemmKernel : public KernelType {};

  // Sparse compressor definitions
  using SparseConfig = typename GemmKernel::CollectiveMainloop::SparseConfig;
  using LayoutTagA = cutlass::layout::RowMajor;
  using CompressorUtility =
      cutlass::transform::kernel::StructuredSparseCompressorUtility<
          typename GemmKernel::ProblemShape, ElementAB, LayoutTagA,
          SparseConfig>;
  using CompressorKernel =
      cutlass::transform::kernel::StructuredSparseCompressor<
          typename GemmKernel::ProblemShape, ElementAB, LayoutTagA,
          SparseConfig, cutlass::arch::Sm90>;
  using Compressor =
      cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;
};

template <typename Gemm, typename... EpilogueArgs>
void cutlass_sparse_gemm_caller(torch::stable::Tensor& out,
                                torch::stable::Tensor const& a,
                                torch::stable::Tensor const& bt_nzs,
                                torch::stable::Tensor const& bt_meta,
                                EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  // Interface stride expected from the argument a (will get transposed)
  // We compute C^T = B^T * A^T, but we assume B is transposed before
  // compression and hence the bt_* naming
  using LayoutB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
  using LayoutE = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;

  // M, N, K after transposition
  int32_t m = out.size(1);
  int32_t n = out.size(0);
  int32_t k = a.size(1);

  int64_t lda = a.stride(0);
  int64_t ldc = out.stride(0);

  using StrideA = Stride<int64_t, Int<1>, int64_t>;
  using StrideC = Stride<Int<1>, int64_t, int64_t>;

  StrideA a_stride{lda, Int<1>{}, Int<0>{}};
  StrideC c_stride{Int<1>{}, ldc, Int<0>{}};

  using GemmKernel = typename Gemm::GemmKernel;
  typename GemmKernel::ProblemShape prob_shape{m, n, k, 1};

  using ElementE = typename GemmKernel::CollectiveMainloop::ElementE;
  using SparseConfig = typename GemmKernel::CollectiveMainloop::SparseConfig;

  LayoutB b_layout = SparseConfig::fill_layoutA(prob_shape);
  LayoutE e_layout = SparseConfig::fill_layoutE(prob_shape);

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(bt_nzs.data_ptr());
  auto e_ptr = static_cast<ElementE*>(bt_meta.data_ptr());
  typename GemmKernel::MainloopArguments mainloop_args{
      b_ptr, b_layout, a_ptr, a_stride, e_ptr, e_layout};

  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptr, c_stride, c_ptr, c_stride};

  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape, mainloop_args, epilogue_args};

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace =
      torch::stable::new_empty(a, {static_cast<int64_t>(workspace_size)},
                               torch::headeronly::ScalarType::Byte);

  auto stream = get_current_cuda_stream(a.get_device_index());

  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

//////////////////////////////////////////////////
// Gemm Configs are defined below
//////////////////////////////////////////////////

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_config_default {};

template <typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_config_default<half_t, OutType, Epilogue> {
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<half_t, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_config_default<cutlass::bfloat16_t, OutType, Epilogue> {
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<cutlass::bfloat16_t, OutType, Epilogue, TileShape,
                             ClusterShape, KernelSchedule, EpilogueSchedule>;
};

//////////////////////// Cherry-Picking Kernels ////////////////////////
template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_1 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_8, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_2 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
  using EpilogueSchedule =
      typename cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileShape = Shape<_128, _64, _256>;
  using ClusterShape = Shape<_8, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_3 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_1, _2, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_4 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule =
      typename cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileShape = Shape<_64, _128, _256>;
  using ClusterShape = Shape<_8, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_5 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<_8, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_6 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _256>;
  using ClusterShape = Shape<_1, _2, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_7 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
  using EpilogueSchedule =
      typename cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_8 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
  using EpilogueSchedule =
      typename cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileShape = Shape<_128, _256, _128>;
  using ClusterShape = Shape<_8, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};
////////////////////////////////////////////////////////////////////////

template <typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_config_default<cutlass::float_e4m3_t, OutType, Epilogue> {
  // M in (128, inf)
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _2, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<cutlass::float_e4m3_t, OutType, Epilogue,
                             TileShape, ClusterShape, KernelSchedule,
                             EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M64 {
  // M in [1, 64]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule =
      typename cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_1, _1, _1>;

  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M128 {
  // M in (64, 128]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _256>;
  using ClusterShape = Shape<_1, _1, _1>;

  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M256 {
  // M in (128, 256]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
  using EpilogueSchedule =
      typename cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<_1, _1, _1>;

  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M512 {
  // M in (256, ]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
  using EpilogueSchedule =
      typename cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<_1, _1, _1>;

  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_config_default<int8_t, OutType, Epilogue> {
  // For M > 128 and any N
  using KernelSchedule =
      typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<int8_t, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_M128 {
  // For M in (64, 128] and any N
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule =
      typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_M64 {
  // For M in (32, 64] and any N
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_M32_NBig {
  // For M in [1, 32] and N >= 8192
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _256>;
  using ClusterShape = Shape<_1, _4, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_M32_NSmall {
  // For M in [1, 32] and N < 8192
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_1, _8, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                             KernelSchedule, EpilogueSchedule>;
};

}  // namespace
