#include <cudaTypedefs.h>

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass_extensions/epilogue/broadcast_load_epilogue_c3x.hpp"
#include "cutlass_extensions/common.hpp"

using namespace cute;

/*
   This file defines sparse quantized GEMM operations using the CUTLASS 3.x API, for
   NVIDIA GPUs with sm90a (Hopper) or later.

   Epilogue functions can be defined to post-process the output before it is
   written to GPU memory.
   Epilogues must contain a public type named EVTCompute of type Sm90EVT,
   as well as a static prepare_args function that constructs an
   EVTCompute::Arguments struct.
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

template <typename ElementAB_, typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule, typename AccType,
          typename TileSchedule = cutlass::gemm::PersistentScheduler,
          GemmUniversalMode Mode_ = GemmUniversalMode::kGemm>
struct cutlass_sparse_3x_gemm {
  static const GemmUniversalMode Mode = Mode_;
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;
  using ElementAcc = AccType;

  using EpilogueDescriptor =
      cutlass::epilogue::collective::detail::EpilogueDescriptor<
          TileShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementD,
          ElementD, EpilogueSchedule>;

  using Epilogue = Epilogue_<ElementAcc, ElementD, EpilogueDescriptor>;

  using StrideD = Stride<Int<1>, int64_t, Int<0>>;
  using ElementC = void;
  using StrideC = StrideD;

  using EVTCompute = typename Epilogue::EVTCompute;

  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc, ElementAcc, ElementC, StrideC, AlignmentCD, ElementD,
          StrideD, AlignmentCD, EpilogueSchedule, EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  // clang-format off
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassSparseTensorOp, 
          ElementAB, cutlass::layout::RowMajor, AlignmentA, 
          ElementAB, cutlass::layout::ColumnMajor, AlignmentB, 
          ElementAcc, TileShape, ClusterShape,
          Stages,
          KernelSchedule>::CollectiveOp;
  // clang-format on

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      TileSchedule>>;

  struct GemmKernel : public KernelType {};
};

template <typename Gemm, typename... EpilogueArgs>
void cutlass_sparse_gemm_caller(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& e, torch::Tensor const& b,
                                EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = b.size(0);

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(1);

  using LayoutA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
  using LayoutE = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideB b_stride{ldb, Int<1>{}, 0};
  StrideC c_stride{Int<1>{}, ldc, Int<0>{}};

  using GemmKernel = typename Gemm::GemmKernel;
  typename GemmKernel::ProblemShape prob_shape{m, n, k, 1};

  using ElementE = typename GemmKernel::CollectiveMainloop::ElementE;
  using SparseConfig = typename GemmKernel::CollectiveMainloop::SparseConfig;

  LayoutA a_layout = SparseConfig::fill_layoutA(prob_shape);
  LayoutE e_layout = SparseConfig::fill_layoutE(prob_shape);

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  auto e_ptr = static_cast<ElementE*>(e.data_ptr());
  typename GemmKernel::MainloopArguments mainloop_args{
      a_ptr, a_layout, b_ptr, b_stride, e_ptr, e_layout};

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
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp16_config_default {
  // M in (128, inf)
  static_assert(std::is_same<InType, cutlass::half_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule, float>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_bf16_config_default {
  // M in (128, inf)
  static_assert(std::is_same<InType, cutlass::bfloat16_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule, float>;
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
                      KernelSchedule, EpilogueSchedule, float>;
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
                      KernelSchedule, EpilogueSchedule, float>;
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
                      KernelSchedule, EpilogueSchedule, float>;
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
                      KernelSchedule, EpilogueSchedule, float>;
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
                      KernelSchedule, EpilogueSchedule, float>;
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
                      KernelSchedule, EpilogueSchedule, float>;
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
                      KernelSchedule, EpilogueSchedule, float>;
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
                      KernelSchedule, EpilogueSchedule, float>;
};
////////////////////////////////////////////////////////////////////////

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_default {
  // M in (128, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _2, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule, float>;
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

  using TileSchedule = cutlass::gemm::PersistentScheduler;

  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule, float, TileSchedule>;
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

  using TileSchedule = cutlass::gemm::PersistentScheduler;

  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule, float, TileSchedule>;
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

  using TileSchedule = cutlass::gemm::PersistentScheduler;

  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule, float, TileSchedule>;
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

  using TileSchedule = cutlass::gemm::PersistentScheduler;

  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule, float, TileSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_default {
  // For M > 128 and any N
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule =
      typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_sparse_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule, int32_t>;
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
                      KernelSchedule, EpilogueSchedule, int32_t>;
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
                      KernelSchedule, EpilogueSchedule, int32_t>;
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
                      KernelSchedule, EpilogueSchedule, int32_t>;
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
                      KernelSchedule, EpilogueSchedule, int32_t>;
};

}  // namespace