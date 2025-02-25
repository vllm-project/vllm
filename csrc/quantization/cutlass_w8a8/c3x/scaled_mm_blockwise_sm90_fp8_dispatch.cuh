#pragma once

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

#include "cutlass_gemm_caller.cuh"

namespace vllm {

using namespace cute;

template <typename SchedulerType, typename OutType, int GroupSizeM_,
          int GroupSizeN_, int GroupSizeK_, int TileSizeM_ = 128,
          class ClusterShape = Shape<_1, _2, _1>>
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
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementBlockScale = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = Shape<TileSizeM, GroupSizeN, GroupSizeK>;

  using KernelSchedule = cutlass::gemm::
      KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum<
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
      SchedulerType>>;

  struct GemmKernel : public KernelType {};

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
};

template <typename Gemm>
void cutlass_gemm_caller_blockwise(torch::Tensor& out, torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales) {
  using GemmKernel = typename Gemm::GemmKernel;

  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  auto prob_shape = c3x::get_problem_shape(a, b);
  int32_t m = get<0>(prob_shape), n = get<1>(prob_shape),
          k = get<2>(prob_shape);

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

  // Check is the t is contiguous and is 1D or 2D with one of the dimensions
  // being 1 (i.e. a row or column vector)
  auto is_contiguous_vector = [](const torch::Tensor& t) {
    auto t_sizes = t.sizes();
    return t.is_contiguous() &&
           (t.dim() == 1 ||
            (t.dim() == 2 &&
             *std::min_element(t_sizes.begin(), t_sizes.end()) == 1));
  };

  // TODO(lucas): lets clean-up the kernel so that we pass in Strides so
  //  we don't have to deal with enforcing implicit layouts
  TORCH_CHECK(a_scales.size(0) == m / Gemm::GroupSizeM::value);
  TORCH_CHECK(a_scales.size(1) == k / Gemm::GroupSizeK::value);
  TORCH_CHECK(a_scales.stride(0) == 1 || is_contiguous_vector(a_scales),
              "a_scales must be M major");
  TORCH_CHECK(b_scales.size(0) == k / Gemm::GroupSizeK::value);
  TORCH_CHECK(b_scales.size(1) == n / Gemm::GroupSizeN::value);
  TORCH_CHECK(b_scales.stride(0) == 1 || is_contiguous_vector(b_scales),
              "b_scales must be K major");
  typename GemmKernel::MainloopArguments mainloop_args{
      a_ptr, a_stride, b_ptr, b_stride, a_scales_ptr, b_scales_ptr};

  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, c_ptr, c_stride, c_ptr, c_stride};

  typename GemmKernel::TileSchedulerArguments scheduler;

  static constexpr bool UsesStreamKScheduler =
      cute::is_same_v<typename GemmKernel::TileSchedulerTag,
                      cutlass::gemm::StreamKScheduler>;

  if constexpr (UsesStreamKScheduler) {
    using DecompositionMode = typename cutlass::gemm::kernel::detail::
        PersistentTileSchedulerSm90StreamKParams::DecompositionMode;
    using ReductionMode = typename cutlass::gemm::kernel::detail::
        PersistentTileSchedulerSm90StreamKParams::ReductionMode;

    scheduler.decomposition_mode = DecompositionMode::StreamK;
    scheduler.reduction_mode = ReductionMode::Nondeterministic;
  }

  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                       epilogue_args, scheduler);
}

template <typename OutType>
void cutlass_gemm_blockwise_sm90_fp8_dispatch(torch::Tensor& out,
                                              torch::Tensor const& a,
                                              torch::Tensor const& b,
                                              torch::Tensor const& a_scales,
                                              torch::Tensor const& b_scales) {
  auto k = a.size(1);
  auto n = b.size(1);

  if (k > 3 * n) {
    cutlass_gemm_caller_blockwise<cutlass_3x_gemm_fp8_blockwise<
        cutlass::gemm::StreamKScheduler, OutType, 1, 128, 128>>(
        out, a, b, a_scales, b_scales);
  } else {
    cutlass_gemm_caller_blockwise<cutlass_3x_gemm_fp8_blockwise<
        cutlass::gemm::PersistentScheduler, OutType, 1, 128, 128>>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace vllm