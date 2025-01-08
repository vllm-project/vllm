#pragma once

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "cutlass_extensions/common.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass_extensions/gemm/dispatch_policy.hpp"
#include "cutlass_extensions/gemm/collective/collective_builder.hpp"

using namespace cute;

template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

template <typename OutType, int GroupSizeM_, int GroupSizeN_, int GroupSizeK_,
          int TileSizeM_ = 128>
struct cutlass_3x_gemm_fp8_blockwise {
  using GroupSizeM = Int<GroupSizeM_>;
  using GroupSizeN = Int<GroupSizeN_>;
  using GroupSizeK = Int<GroupSizeK_>;
  using TileSizeM = Int<TileSizeM_>;

  static_assert(TileSizeM_ % GroupSizeM_ == 0,
                "TileSizeM must be a multiple of GroupSizeM");

  using ElementAB = cutlass::float_e4m3_t;

  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementAB;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using StrideD = Stride<int64_t, Int<1>, Int<0>>;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void;
  using StrideC = StrideD;
  static constexpr int AlignmentC = 4;

  using ElementAccumulator = float;
  using ElementBlockScale = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = Shape<TileSizeM, GroupSizeN, GroupSizeK>;
  using ClusterShape = Shape<_1, _2, _1>;

  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum<
          GroupSizeM_>;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using StoreEpilogueCompute = typename cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
          ElementAccumulator, ElementCompute, ElementC, StrideC, AlignmentC,
          ElementD, StrideD, AlignmentD, EpilogueSchedule,
          StoreEpilogueCompute>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB,
          LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>>;

  struct GemmKernel : public KernelType {};

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  //   using StrideC = typename GemmKernel::StrideC;
  //   using StrideD = StrideC;
};

Shape<int, int, int, int> get_problem_shape(torch::Tensor const& a,
                                            torch::Tensor const& b) {
  int32_t m = a.size(0), n = b.size(1), k = a.size(1);
  return {m, n, k, 1};
}

template <typename GemmKernel>
void cutlass_gemm_caller(torch::Device device,
                         Shape<int, int, int, int> prob_shape,
                         typename GemmKernel::MainloopArguments mainloop_args,
                         typename GemmKernel::EpilogueArguments epilogue_args) {
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape, mainloop_args, epilogue_args};

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(device);
  auto workspace = torch::empty(workspace_size, workspace_options);

  auto stream = at::cuda::getCurrentCUDAStream(device.index());

  cutlass::Status status = gemm_op.run(args, workspace.data_ptr(), stream);
  CUTLASS_CHECK(status);
}

template <typename Gemm>
void cutlass_gemm_caller_blockwise(torch::Tensor& out, torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales) {
  using GemmKernel = typename Gemm::GemmKernel;

  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  auto prob_shape = get_problem_shape(a, b);

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

  using StrideA = Stride<int64_t, Int<1>, int64_t>;
  using StrideB = Stride<int64_t, Int<1>, int64_t>;
  using StrideC = typename Gemm::StrideC;

  StrideA a_stride{lda, Int<1>{}, 0};
  StrideB b_stride{ldb, Int<1>{}, 0};
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  auto a_scales_ptr = static_cast<float*>(a_scales.data_ptr());
  auto b_scales_ptr = static_cast<float*>(b_scales.data_ptr());

  uint32_t mma_promotion_interval = 4;

  typename GemmKernel::MainloopArguments mainloop_args{
      a_ptr,        a_stride,    b_ptr, b_stride, mma_promotion_interval,
      a_scales_ptr, b_scales_ptr};

  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, c_ptr, c_stride, c_ptr, c_stride};

  cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                  epilogue_args);
}

template <typename OutType>
void cutlass_gemm_blockwise_sm90_fp8_dispatch(torch::Tensor& out,
                                              torch::Tensor const& a,
                                              torch::Tensor const& b,
                                              torch::Tensor const& a_scales,
                                              torch::Tensor const& b_scales) {
  cutlass_gemm_caller_blockwise<
      cutlass_3x_gemm_fp8_blockwise<OutType, 1, 128, 128>>(out, a, b, a_scales,
                                                           b_scales);
}