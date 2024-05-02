#include <torch/extension.h>

#include <assert.h>
#include <stddef.h>

// clang-format will break include orders
// clang-format off
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"

#include "cutlass_visitor_2x_broadcast_epilogue.hpp"
#include "common.hpp"
// clang-format on 

/////////////////////////////////////////

template<typename Arch, typename ElementIn, typename ElementOut, typename ElementAcc>
struct sm8x_gemm
{

using Operator = typename std::conditional<std::is_same_v<ElementIn, int8_t>,
         cutlass::arch::OpMultiplyAddSaturate, cutlass::arch::OpMultiplyAdd>::type;

using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    float,
    4,
    1 /* epilogue stages */
>;


using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

using ScaleA = cutlass::epilogue::threadblock::VisitorColOrScalarBroadcast<
    OutputTileThreadMap, float,
    cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>
>;

using ScaleB = cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
    OutputTileThreadMap, float,
    cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>
>;

using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<
    Compute0,
    ScaleB,
    Accum>;

using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, ElementOut, float,
    cutlass::FloatRoundStyle::round_to_nearest
>;

using EVTCompute1 = cutlass::epilogue::threadblock::Sm80EVT<
    Compute1,
    ScaleA,
    EVTCompute0>;

using D = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, cutlass::bfloat16_t, cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>
>;

using EVTD = cutlass::epilogue::threadblock::Sm80EVT<
    D,
    EVTCompute1>;


// Gemm operator cutlass_tensorop_f32_i16832gemm_s8_256x128_64x3_tn_align16
using cutlass_tensorop_f32_i16832gemm_s8_256x128_64x3_tn_align16_base =
    typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementIn, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
    ElementIn, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
    float, cutlass::layout::RowMajor, 4,
    ElementAcc,
    float,
    cutlass::arch::OpClassTensorOp,
    Arch,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    EVTD,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
    3,
    Operator,
    1 /* epilogue stages */
>::GemmKernel;

using Op = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass_tensorop_f32_i16832gemm_s8_256x128_64x3_tn_align16_base>;
};

/////////////////////////////////////////

template <typename Gemm, typename ElementIn, typename ElementOut>
void cutlass_scaled_mm_dq_dispatcher(torch::Tensor &out, torch::Tensor const &a,
                                     torch::Tensor const &b,
                                     torch::Tensor const &a_scales,
                                     torch::Tensor const &b_scales) {

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);
  cutlass::gemm::GemmCoord problem_size{m, n, k};

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

  using StrideC = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
  StrideC c_stride = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});

  auto a_ptr = static_cast<ElementIn const *>(a.data_ptr());
  auto b_ptr = static_cast<ElementIn const *>(b.data_ptr());
  auto c_ptr = static_cast<ElementOut *>(out.data_ptr());

  auto a_scales_ptr = a_scales.data_ptr<float>();
  auto b_scales_ptr = b_scales.data_ptr<float>();

  using ScaleAArgs = typename Gemm::ScaleA::Arguments;
  ScaleAArgs a_args = a_scales.numel() == 1
                          ? ScaleAArgs{nullptr, a_scales.item<float>(), {}}
                          : ScaleAArgs{a_scales.data_ptr<float>(), {}, {}};

  using ScaleBArgs = typename Gemm::ScaleB::Arguments;
  ScaleBArgs b_args = b_scales.numel() == 1
                          ? ScaleBArgs{nullptr, b_scales.item<float>(), {}}
                          : ScaleBArgs{b_scales.data_ptr<float>(), {}, {}};

  typename Gemm::EVTCompute0::Arguments evt0_compute_args{b_args};

  typename Gemm::EVTCompute1::Arguments evt1_compute_args{a_args,
                                                          evt0_compute_args};
  typename Gemm::D::Arguments d_args{c_ptr, c_stride};

  typename Gemm::EVTD::Arguments epilogue_args{
      evt1_compute_args,
      d_args,
  };

  typename Gemm::Op::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, // universal mode
      problem_size,                            // problem size
      1,                                       // batch count
      epilogue_args,
      a_ptr,
      b_ptr,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      lda,
      ldb,
      ldc,
      ldc};

  // Launch the CUTLASS GEMM kernel.
  typename Gemm::Op gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(gemm_op.can_implement(args));
  cutlass::Status status = gemm_op(args, workspace.get());
  CUTLASS_CHECK(status);
}

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
void cutlass_scaled_mm_dq_sm80(torch::Tensor &out, torch::Tensor const &a,
                          torch::Tensor const &b, torch::Tensor const &a_scales,
                          torch::Tensor const &b_scales) {
  assert(a.dtype() == torch::kInt8);
  assert(b.dtype() == torch::kInt8);
  assert(a_scales.dtype() == torch::kFloat32);
  assert(b_scales.dtype() == torch::kFloat32);
  assert(out.dtype() == torch::kBFloat16);

  return cutlass_scaled_mm_dq_dispatcher<sm8x_gemm<cutlass::arch::Sm80, int8_t, cutlass::bfloat16_t, int32_t>, int8_t, cutlass::bfloat16_t>(out, a, b, a_scales, b_scales);
}
#endif

#if defined(CUTLASS_ARCH_MMA_SM89_SUPPORTED)
void cutlass_scaled_mm_dq_sm89(torch::Tensor &out, torch::Tensor const &a,
                          torch::Tensor const &b, torch::Tensor const &a_scales,
                          torch::Tensor const &b_scales) {
  if (a.dtype() == torch::kInt8) {
    assert(b.dtype() == torch::kInt8);
    assert(a_scales.dtype() == torch::kFloat32);
    assert(b_scales.dtype() == torch::kFloat32);
    assert(out.dtype() == torch::kBFloat16);

    return cutlass_scaled_mm_dq_dispatcher<
        sm8x_gemm<cutlass::arch::Sm89, int8_t, cutlass::bfloat16_t, int32_t>, int8_t,
        cutlass::bfloat16_t>(out, a, b, a_scales, b_scales);
  } else {
    assert(a.dtype() == torch::kFloat8_e4m3fn);
    assert(b.dtype() == torch::kFloat8_e4m3fn);
    assert(a_scales.dtype() == torch::kFloat32);
    assert(b_scales.dtype() == torch::kFloat32);
    assert(out.dtype() == torch::kBFloat16);

    return cutlass_scaled_mm_dq_dispatcher<
        sm8x_gemm<cutlass::arch::Sm89, cutlass::float_e4m3_t, cutlass::bfloat16_t, float>,
        cutlass::float_e4m3_t, cutlass::bfloat16_t>(out, a, b, a_scales,
                                                    b_scales);
  }
}
#endif
