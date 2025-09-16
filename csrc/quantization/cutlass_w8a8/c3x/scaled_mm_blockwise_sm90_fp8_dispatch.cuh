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

#include "cutlass_gemm_caller.cuh"

namespace vllm {

using namespace cute;

// clang-format off
template <class OutType, int ScaleGranularityM,
          int ScaleGranularityN, int ScaleGranularityK,
          class MmaTileShape, class ClusterShape,
          class EpilogueScheduler, class MainloopScheduler>
struct cutlass_3x_gemm_fp8_blockwise {
  using ElementAB = cutlass::float_e4m3_t;

  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementAB;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void; // TODO: support bias
  using LayoutC = LayoutD;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float;

  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<
        ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag = cutlass::arch::Sm90;
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
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutD,
      AlignmentD,
      EpilogueScheduler,
      DefaultOperation
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
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
  >::CollectiveOp;

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};

template <typename Gemm>
void cutlass_gemm_caller_blockwise(torch::Tensor& out, torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales) {
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

  TORCH_CHECK(m % 4 == 0, "m must be divisible by 4");

  StrideA a_stride;
  StrideB b_stride;
  StrideC c_stride;
  a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  c_stride =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));

  LayoutSFA layout_SFA = 
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB = 
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB const*>(b.data_ptr());
  auto a_scales_ptr = static_cast<ElementBlockScale const*>(a_scales.data_ptr());
  auto b_scales_ptr = static_cast<ElementBlockScale const*>(b_scales.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args{};
  mainloop_args.ptr_A = a_ptr;
  mainloop_args.dA = a_stride;
  mainloop_args.ptr_B = b_ptr;
  mainloop_args.dB = b_stride;
  mainloop_args.ptr_SFA = a_scales_ptr;
  mainloop_args.layout_SFA = layout_SFA;
  mainloop_args.ptr_SFB = b_scales_ptr;
  mainloop_args.layout_SFB = layout_SFB;
  auto prob_shape = cute::make_shape(m, n, k, 1);

  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, c_ptr, c_stride, c_ptr, c_stride};
  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                       epilogue_args);
}

template <typename OutType>
void cutlass_gemm_blockwise_sm90_fp8_dispatch(torch::Tensor& out,
                                              torch::Tensor const& a,
                                              torch::Tensor const& b,
                                              torch::Tensor const& a_scales,
                                              torch::Tensor const& b_scales) {
  // TODO: better heuristics
  cutlass_gemm_caller_blockwise<cutlass_3x_gemm_fp8_blockwise<
      OutType, 1, 128, 128, Shape<_128, _128, _128>,
      Shape<_1, _2, _1>, cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum>>(
      out, a, b, a_scales, b_scales);
}

}  // namespace vllm