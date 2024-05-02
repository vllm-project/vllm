#include <torch/extension.h>

#include <iostream>
#include <sstream>
#include <vector>

// clang-format will break include orders
// clang-format off
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "common.hpp"
// clang-format on 

/////////////////////////////////////////
// Begin automatically generated section
// clang-format off

using namespace cute;

namespace int8_kernel
{

using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
  cute::Shape<_128, _128, _128>, cutlass::epilogue::collective::EpilogueTileAuto,
  cutlass::bfloat16_t, cutlass::bfloat16_t,
  cutlass::epilogue::TmaWarpSpecialized
>;

using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
    0 /*Stages*/, typename EpilogueDescriptor::TileShape, float,
    cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>
>;

using ScaleBDescriptor = cutlass::epilogue::collective::detail::RowBroadcastDescriptor<EpilogueDescriptor, float>;

using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
    ScaleBDescriptor::Stages, typename EpilogueDescriptor::TileShape,
    typename ScaleBDescriptor::Element, cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>
>;

using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<
    Compute0,
    ScaleB,
    Accum>;

using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies, cutlass::bfloat16_t, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<
    Compute1,
    ScaleA,
    EVTCompute0>;

using ElementD = cutlass::bfloat16_t;
using StrideD = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using ElementC = void;
using StrideC = StrideD;



using CollectiveEpilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_2,cute::_1,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    int32_t, float,
    ElementC, StrideC, 4,
    ElementD, StrideD, 4,
    cutlass::epilogue::TmaWarpSpecialized,
    EVTCompute1
  >::CollectiveOp;

using CollectiveMainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    int8_t, cutlass::layout::RowMajor, 16,
    int8_t, cutlass::layout::ColumnMajor, 16,
    int32_t,
    cute::Shape<cute::_128, cute::_128, cute::_128>,
    cute::Shape<cute::_2,cute::_1,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_i64x128x32gemm_s8_s8_s32_bf16_bf16_128x128x128_2x1x1_0_tnt_align16_warpspecialized_pingpong_epi_tma
using cutlass3x_sm90_tensorop_i64x128x32gemm_s8_s8_s32_bf16_bf16_128x128x128_2x1x1_0_tnt_align16_warpspecialized_pingpong_epi_tma_base = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler
>;

// Define named type
struct GemmKernel :
  public cutlass3x_sm90_tensorop_i64x128x32gemm_s8_s8_s32_bf16_bf16_128x128x128_2x1x1_0_tnt_align16_warpspecialized_pingpong_epi_tma_base { };

} // namespace int8_kernel

namespace fp8_kernel
{

using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<
  cute::Shape<_256, _128, _128>, cutlass::epilogue::collective::EpilogueTileAuto,
  cutlass::bfloat16_t, cutlass::bfloat16_t,
  cutlass::epilogue::TmaWarpSpecializedCooperative
>;

using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
    0 /*Stages*/, typename EpilogueDescriptor::TileShape, float,
    cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>
>;

using ScaleBDescriptor = cutlass::epilogue::collective::detail::RowBroadcastDescriptor<EpilogueDescriptor, float>;

using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
    ScaleBDescriptor::Stages, typename EpilogueDescriptor::TileShape,
    typename ScaleBDescriptor::Element, cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>
>;

using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<
    Compute0,
    ScaleB,
    Accum>;

using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
    cutlass::multiplies, cutlass::bfloat16_t, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<
    Compute1,
    ScaleA,
    EVTCompute0>;

using ElementD = cutlass::bfloat16_t;
using StrideD = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using ElementC = void;
using StrideC = StrideD;



using CollectiveEpilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_256, cute::_128, cute::_128>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float,
    ElementC, StrideC, 1,
    ElementD, StrideD, 1,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    EVTCompute1
  >::CollectiveOp;

using CollectiveMainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16,
    float,
    cute::Shape<cute::_256, cute::_128, cute::_128>,
    cute::Shape<cute::_1,cute::_2,cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_256x128x128_1x2x1_0_tnt_align16_warpspecialized_cooperative_epi_tma
using cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_256x128x128_1x2x1_0_tnt_align16_warpspecialized_cooperative_epi_tma_base = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler
>;

// Define named type
struct GemmKernel :
  public cutlass3x_sm90_tensorop_s64x128x32gemm_e4m3_e4m3_f32_bf16_bf16_256x128x128_1x2x1_0_tnt_align16_warpspecialized_cooperative_epi_tma_base { };

} // namespace fp8_kernel

// clang-format on
// End automatically generated section
/////////////////////////////////////////

using StrideA = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
using StrideB = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;

template <typename GemmKernel, typename ScaleA, typename ScaleB,
          typename StrideC, typename ElementIn, typename ElementOut>
void cutlass_scaled_mm_dq_dispatcher(torch::Tensor &out, torch::Tensor const &a,
                                     torch::Tensor const &b,
                                     torch::Tensor const &a_scales,
                                     torch::Tensor const &b_scales) {

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);

  StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC c_stride = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});

  typename GemmKernel::ProblemShape prob_shape{m, n, k, 1};

  auto a_ptr = static_cast<ElementIn *>(a.data_ptr());
  auto b_ptr = static_cast<ElementIn *>(b.data_ptr());
  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr,
                                                       b_stride};

  auto c_ptr = static_cast<ElementOut *>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, c_ptr, c_stride, c_ptr, c_stride};

  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape, mainloop_args, epilogue_args};

  typename ScaleA::Arguments a_args =
      a_scales.numel() == 1
          ? typename ScaleA::Arguments{nullptr, a_scales.item<float>(), {}}
          : typename ScaleA::Arguments{a_scales.data_ptr<float>(), {}, {}};

  typename ScaleB::Arguments b_args =
      b_scales.numel() == 1
          ? typename ScaleB::Arguments{nullptr, b_scales.item<float>(), {}}
          : typename ScaleB::Arguments{b_scales.data_ptr<float>(), {}, {}};

  args.epilogue.thread = {a_args, {b_args}};

  // Launch the CUTLASS GEMM kernel.
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  Gemm gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));
  cutlass::Status status = gemm_op.run(args);
  CUTLASS_CHECK(status);
}

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
void cutlass_scaled_mm_dq_sm90(torch::Tensor &out, torch::Tensor const &a,
                               torch::Tensor const &b,
                               torch::Tensor const &a_scales,
                               torch::Tensor const &b_scales) {
  if (a.dtype() == torch::kInt8) {

    return cutlass_scaled_mm_dq_dispatcher<
        int8_kernel::GemmKernel, int8_kernel::ScaleA, int8_kernel::ScaleB,
        int8_kernel::StrideC, int8_t, cutlass::bfloat16_t>(out, a, b, a_scales,
                                                           b_scales);
  } else {

    return cutlass_scaled_mm_dq_dispatcher<
        fp8_kernel::GemmKernel, fp8_kernel::ScaleA, fp8_kernel::ScaleB,
        fp8_kernel::StrideC, cutlass::float_e4m3_t, cutlass::bfloat16_t>(
        out, a, b, a_scales, b_scales);
  }
}
#endif

