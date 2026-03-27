#pragma once

#include <torch/headeronly/util/shim_utils.h>

#include "cuda_utils.h"
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

#include "cutlass_gemm_caller.cuh"

namespace vllm {

using namespace cute;

// clang-format off
template <class OutType, int ScaleGranularityM,
          int ScaleGranularityN, int ScaleGranularityK,
          class MmaTileShape, class ClusterShape,
          class EpilogueScheduler, class MainloopScheduler,
          bool swap_ab_ = false>
struct cutlass_3x_gemm_fp8_blockwise {
  static constexpr bool swap_ab = swap_ab_;
  using ElementAB = cutlass::float_e4m3_t;

  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementAB;
  // ColumnMajor is used for B to match the CUTLASS convention.
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void; // TODO: support bias
  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float; 

  using ScaleConfig = conditional_t<swap_ab,
      cutlass::detail::Sm120BlockwiseScaleConfig<
        ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
        cute::UMMA::Major::K, cute::UMMA::Major::MN>,
      cutlass::detail::Sm120BlockwiseScaleConfig<
        ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
        cute::UMMA::Major::MN, cute::UMMA::Major::K>>;

  // layout_SFA and layout_SFB cannot be swapped since they are deduced.
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using ElementScalar = float;
  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      MmaTileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      conditional_t<swap_ab, LayoutC_Transpose, LayoutC>,
      AlignmentC,
      ElementD,
      conditional_t<swap_ab, LayoutD_Transpose, LayoutD>,
      AlignmentD,
      EpilogueScheduler,
      DefaultOperation
  >::CollectiveOp;
 
  using StageCountType = cutlass::gemm::collective::StageCountAuto; 
  using CollectiveMainloop = conditional_t<swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementB,
          cute::tuple<LayoutB_Transpose, LayoutSFA>,
          AlignmentB,
          ElementA,
          cute::tuple<LayoutA_Transpose, LayoutSFB>,
          AlignmentA,
          ElementAccumulator,
          MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduler
      >::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementA,
          cute::tuple<LayoutA, LayoutSFA>,
          AlignmentA,
          ElementB,
          cute::tuple<LayoutB, LayoutSFB>,
          AlignmentB,
          ElementAccumulator,
          MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduler
      >::CollectiveOp>;

  // SM12x family to support both SM120 (RTX 5090) and SM121 (DGX Spark)
  using KernelType = enable_sm120_family<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};

// Tile configurations for different M ranges
template <typename OutType>
struct sm120_blockwise_fp8_config_default {
  // use 128x128x128 tile with Cooperative (Auto) schedule
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  // ScaleGranularity must match the actual quantization block size (1, 128, 128)
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule>;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_pingpong {
  // use 64x128x128 tile with Pingpong schedule
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  // ScaleGranularity stays (1, 128, 128) to match actual quantization data
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 1, 128, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule>;
};

template <typename OutType>
struct sm120_blockwise_fp8_config_swapab {
  // use 128x32x128 tile with Cooperative schedule
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Gemm = cutlass_3x_gemm_fp8_blockwise<
      OutType, 128, 1, 128, TileShape, ClusterShape,
      EpilogueSchedule, KernelSchedule, true>;
};

template <typename Gemm>
void cutlass_gemm_caller_blockwise(torch::stable::Tensor& out, torch::stable::Tensor const& a,
                                   torch::stable::Tensor const& b,
                                   torch::stable::Tensor const& a_scales,
                                   torch::stable::Tensor const& b_scales) {
  static constexpr bool swap_ab = Gemm::swap_ab;
  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutSFA = typename Gemm::LayoutSFA;
  using LayoutSFB = typename Gemm::LayoutSFB;
  using ScaleConfig = typename Gemm::ScaleConfig;

  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using ElementBlockScale = typename Gemm::ElementBlockScale;

  int32_t m = a.size(0), n = b.size(1), k = a.size(1);

  StrideA a_stride;
  StrideB b_stride;
  StrideC c_stride;
  a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  c_stride =
      cutlass::make_cute_packed_stride(StrideC{}, swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  LayoutSFA layout_SFA = swap_ab ?
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, m, k, 1)) :
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = swap_ab ?
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, m, k, 1)) :
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB const*>(b.data_ptr());
  auto a_scales_ptr = static_cast<ElementBlockScale const*>(a_scales.data_ptr());
  auto b_scales_ptr = static_cast<ElementBlockScale const*>(b_scales.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args{};
  mainloop_args.layout_SFA = layout_SFA;
  mainloop_args.layout_SFB = layout_SFB;
  if (swap_ab) {
    mainloop_args.ptr_A = b_ptr;
    mainloop_args.dA = b_stride;
    mainloop_args.ptr_B = a_ptr;
    mainloop_args.dB = a_stride;
    mainloop_args.ptr_SFA = b_scales_ptr;
    mainloop_args.ptr_SFB = a_scales_ptr;
  } else {
    mainloop_args.ptr_A = a_ptr;
    mainloop_args.dA = a_stride;
    mainloop_args.ptr_B = b_ptr;
    mainloop_args.dB = b_stride;
    mainloop_args.ptr_SFA = a_scales_ptr;
    mainloop_args.ptr_SFB = b_scales_ptr;
  }
  auto prob_shape = swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);

  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, c_ptr, c_stride, c_ptr, c_stride};
  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                       epilogue_args);
}

template <typename OutType>
void cutlass_gemm_blockwise_sm120_fp8_dispatch(torch::stable::Tensor& out,
                                               torch::stable::Tensor const& a,
                                               torch::stable::Tensor const& b,
                                               torch::stable::Tensor const& a_scales,
                                               torch::stable::Tensor const& b_scales) {
  int M = a.size(0);
  // more heuristic tuning can be done here by checking N/K dimensions as well
  bool swap_ab = (M <= 64) || (M % 4 != 0);

  if (!swap_ab) {
    // Use if constexpr to avoid instantiating the 64x64 tile config when
    // scale factor layouts in the blockwise MMA collective).
    if (M <= 256) {
      using Gemm = typename sm120_blockwise_fp8_config_pingpong<OutType>::Gemm;
      return cutlass_gemm_caller_blockwise<Gemm>(
          out, a, b, a_scales, b_scales);
    }
    // M > 256: use default 128x128x128 config with Cooperative (Auto) schedule
    using Gemm = typename sm120_blockwise_fp8_config_default<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        out, a, b, a_scales, b_scales);
  } else {
    // Swap A/B for small M to improve performance
    // Use TILE_N=32 as the minimum compatible tile size.
    using Gemm = typename sm120_blockwise_fp8_config_swapab<OutType>::Gemm;
    return cutlass_gemm_caller_blockwise<Gemm>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace vllm
