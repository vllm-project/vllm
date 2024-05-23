#include <stddef.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

// clang-format will break include orders
// clang-format off
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/util/device_memory.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"

#include "broadcast_load_epilogue_c2x.hpp"
#include "common.hpp"
// clang-format on

using namespace cute;

namespace {

/*
   This epilogue is used for defining a quantized GEMM operation with
   dequantized output, similar to torch._scaled_mm. It uses the CUTLASS 2.x API,
   and is used for NVIDIA GPUs with SM versions prior to sm90 (Hopper).

   A and B may be both either int8 or fp8_e4m3. A can be quantized per-tensor or
   per-row. B can be quantized per-tensor or per-column.
   Any combination of per-tensor and per-row or column is supported.
   A and B must have symmetric quantization (zero point == 0).

   So the GEMM operation is D = (a_scales * A) (b_scales * B), where the
   scales are applied elementwise with numpy-style broadcasting.

   ScaleA and ScaleB define the epilogue functions that apply the scales for
   the A and B operands respectively. These scales may be either per-tensor or
   per row or column.
*/

template <typename ElementD, typename OutputTileThreadMap>
struct ScaledDequantEpilogue {
private:
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  using ScaleA = cutlass::epilogue::threadblock::VisitorColOrScalarBroadcast<
      OutputTileThreadMap, float, Stride<Int<1>, Int<0>, Int<0>>>;

  using ScaleB = cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
      OutputTileThreadMap, float, Stride<Int<0>, Int<1>, Int<0>>>;

  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;


public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA, EVTCompute0>;
  using ArgumentType = typename EVTCompute::Arguments;

  template <typename... Args>
  static ArgumentType prepare_args(Args... args) {
    auto tuple = std::make_tuple(args...);

    torch::Tensor const& a_scales = std::get<0>(tuple);
    torch::Tensor const& b_scales = std::get<1>(tuple);

    auto a_scales_ptr = a_scales.data_ptr<float>();
    auto b_scales_ptr = b_scales.data_ptr<float>();

    using ScaleAArgs = typename ScaleA::Arguments;
    using ScaleBArgs = typename ScaleB::Arguments;

    ScaleBArgs b_args{b_scales.data_ptr<float>(), b_scales.numel() != 1, {}};
    ScaleAArgs a_args{a_scales.data_ptr<float>(), a_scales.numel() != 1, {}};

    typename EVTCompute0::Arguments evt0_compute_args{b_args};

    typename EVTCompute::Arguments evt_compute_args{a_args, evt0_compute_args};
    return evt_compute_args;
  }
};

/*
   This is the same as ScaledDequantEpilogue, but the output is quantized,
   so there is an additional scale and down-conversion to an 8-bit type.
*/
template <typename ElementD, typename OutputTileThreadMap>
struct ScaledQuantEpilogue {
private:
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  // Define epilogue to multiply the output by scale_b

  using ScaleB = cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
      OutputTileThreadMap, float, Stride<Int<0>, Int<1>, Int<0>>>;

  using ComputeB = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeB =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeB, ScaleB, Accum>;

  // Define epilogue to multiply the output by scale_c

  using ScaleC = cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
      OutputTileThreadMap, float, Stride<Int<0>, Int<1>, Int<0>>>;

  using ComputeC = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeC =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeC, ScaleC, EVTComputeB>;

  // Define epilogue to multiply the output by scale_a

  using ScaleA = cutlass::epilogue::threadblock::VisitorColOrScalarBroadcast<
      OutputTileThreadMap, float, Stride<Int<1>, Int<0>, Int<0>>>;

  using ComputeA = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

public:
  using EVTCompute =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeA, ScaleA, EVTComputeC>;

  using ArgumentType = typename EVTCompute::Arguments;

  template <typename... Args>
  static ArgumentType prepare_args(Args... args) {
    auto tuple = std::make_tuple(args...);

    torch::Tensor const& a_scales = std::get<0>(tuple);
    torch::Tensor const& b_scales = std::get<1>(tuple);
    torch::Tensor const& c_scales = std::get<2>(tuple);

    auto a_scales_ptr = a_scales.data_ptr<float>();
    auto b_scales_ptr = b_scales.data_ptr<float>();
    auto c_scales_ptr = b_scales.data_ptr<float>();

    using ScaleAArgs = typename ScaleA::Arguments;
    using ScaleBArgs = typename ScaleB::Arguments;
    using ScaleCArgs = typename ScaleC::Arguments;

    ScaleAArgs a_args{a_scales.data_ptr<float>(), a_scales.numel() != 1, {}};
    ScaleBArgs b_args{b_scales.data_ptr<float>(), b_scales.numel() != 1, {}};
    ScaleCArgs c_args{c_scales.data_ptr<float>(), c_scales.numel() != 1, {}};

    typename EVTComputeB::Arguments evtb_compute_args{b_args};
    typename EVTComputeC::Arguments evtc_compute_args{c_args,
                                                      evtb_compute_args};
    typename EVTCompute::Arguments evta_compute_args{a_args, evtc_compute_args};
    return evta_compute_args;
  }
};

template <typename Arch, typename ElementAB_, typename ElementD_,
          template <typename, typename> typename Epilogue_, typename TileShape,
          typename WarpShape, typename InstructionShape, int32_t MainLoopStages>
struct cutlass_2x_gemm {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;

  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using Operator =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>,
                                cutlass::arch::OpMultiplyAddSaturate,
                                cutlass::arch::OpMultiplyAdd>::type;

  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          TileShape, WarpShape, float, 4, 1 /* epilogue stages */
          >;

  using Epilogue = Epilogue_<ElementD, OutputTileThreadMap>;
  using EVTCompute = typename Epilogue::EVTCompute;

  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, Int<1>, Int<0>>>;

  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

  // clang-format off
  using RowMajor = typename cutlass::layout::RowMajor;
  using ColumnMajor = typename cutlass::layout::ColumnMajor;
  using KernelType = 
    typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementAB, RowMajor, cutlass::ComplexTransform::kNone, 16, 
      ElementAB, ColumnMajor, cutlass::ComplexTransform::kNone, 16, 
      float, cutlass::layout::RowMajor, 4,
      ElementAcc, float, cutlass::arch::OpClassTensorOp, 
      Arch, 
      TileShape, WarpShape, InstructionShape,
      EVTD,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
      MainLoopStages, Operator,
      1 /* epilogue stages */
      >::GemmKernel;
  // clang-format on

  using Op = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};

template <typename Gemm, typename... EpilogueArgs>
void cutlass_scaled_mm_dispatcher(torch::Tensor& c, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);
  cutlass::gemm::GemmCoord problem_size{m, n, k};

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = c.stride(0);

  using StrideC = Stride<int64_t, Int<1>, Int<0>>;
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB const*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(c.data_ptr());

  typename Gemm::D::Arguments d_args{c_ptr, c_stride};

  using ASDF = typename Gemm::Epilogue;
  auto evt_args = ASDF::prepare_args(epilogue_params...);

  typename Gemm::EVTD::Arguments epilogue_args{
      evt_args,
      d_args,
  };

  typename Gemm::Op::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,  // universal mode
      problem_size,                                           // problem size
      1,                                                      // batch count
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
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  cutlass::Status status = gemm_op(args, workspace.get(), stream);
  CUTLASS_CHECK(status);
}

}  // namespace

void cutlass_scaled_mm_dq_sm75(torch::Tensor& c, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<8, 8, 16>;

  if (c.dtype() == torch::kBFloat16) {
    return cutlass_scaled_mm_dispatcher<cutlass_2x_gemm<
        cutlass::arch::Sm75, int8_t, cutlass::bfloat16_t, ScaledDequantEpilogue,
        TileShape, WarpShape, InstructionShape, 2>>(c, a, b, a_scales,
                                                    b_scales);
  } else {
    TORCH_CHECK(c.dtype() == torch::kFloat16);
    return cutlass_scaled_mm_dispatcher<cutlass_2x_gemm<
        cutlass::arch::Sm75, int8_t, cutlass::half_t, ScaledDequantEpilogue,
        TileShape, WarpShape, InstructionShape, 2>>(c, a, b, a_scales,
                                                    b_scales);
  }
}

void cutlass_scaled_mm_dq_sm80(torch::Tensor& c, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  if (c.dtype() == torch::kBFloat16) {
    return cutlass_scaled_mm_dispatcher<cutlass_2x_gemm<
        cutlass::arch::Sm80, int8_t, cutlass::bfloat16_t, ScaledDequantEpilogue,
        TileShape, WarpShape, InstructionShape, 5>>(c, a, b, a_scales,
                                                    b_scales);
  } else {
    TORCH_CHECK(c.dtype() == torch::kFloat16);
    return cutlass_scaled_mm_dispatcher<cutlass_2x_gemm<
        cutlass::arch::Sm80, int8_t, cutlass::half_t, ScaledDequantEpilogue,
        TileShape, WarpShape, InstructionShape, 5>>(c, a, b, a_scales,
                                                    b_scales);
  }
}

void cutlass_scaled_mm_dq_sm89(torch::Tensor& c, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales) {
  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (a.dtype() == torch::kInt8) {
    TORCH_CHECK(b.dtype() == torch::kInt8);

    if (c.dtype() == torch::kBFloat16) {
      return cutlass_scaled_mm_dispatcher<cutlass_2x_gemm<
          cutlass::arch::Sm89, int8_t, cutlass::bfloat16_t,
          ScaledDequantEpilogue, TileShape, WarpShape, InstructionShape, 5>>(
          c, a, b, a_scales, b_scales);
    } else {
      assert(c.dtype() == torch::kFloat16);
      return cutlass_scaled_mm_dispatcher<cutlass_2x_gemm<
          cutlass::arch::Sm89, int8_t, cutlass::half_t, ScaledDequantEpilogue,
          TileShape, WarpShape, InstructionShape, 5>>(c, a, b, a_scales,
                                                      b_scales);
    }
  } else {
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

    if (c.dtype() == torch::kBFloat16) {
      return cutlass_scaled_mm_dispatcher<cutlass_2x_gemm<
          cutlass::arch::Sm89, cutlass::float_e4m3_t, cutlass::bfloat16_t,
          ScaledDequantEpilogue, TileShape, WarpShape, InstructionShape, 5>>(
          c, a, b, a_scales, b_scales);
    } else {
      TORCH_CHECK(c.dtype() == torch::kFloat16);
      return cutlass_scaled_mm_dispatcher<cutlass_2x_gemm<
          cutlass::arch::Sm89, cutlass::float_e4m3_t, cutlass::half_t,
          ScaledDequantEpilogue, TileShape, WarpShape, InstructionShape, 5>>(
          c, a, b, a_scales, b_scales);
    }
  }
}

// Kernels with quantized output

void cutlass_scaled_mm_qout_sm75(torch::Tensor& c, torch::Tensor const& a,
                                 torch::Tensor const& b,
                                 torch::Tensor const& a_scales,
                                 torch::Tensor const& b_scales,
                                 torch::Tensor const& c_scales) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);
  TORCH_CHECK(c.dtype() == torch::kInt8);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(c_scales.dtype() == torch::kFloat32);

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<8, 8, 16>;

  return cutlass_scaled_mm_dispatcher<
      cutlass_2x_gemm<cutlass::arch::Sm75, int8_t, int8_t, ScaledQuantEpilogue,
                      TileShape, WarpShape, InstructionShape, 2>>(
      c, a, b, a_scales, b_scales, c_scales);
}

void cutlass_scaled_mm_qout_sm80(torch::Tensor& c, torch::Tensor const& a,
                                 torch::Tensor const& b,
                                 torch::Tensor const& a_scales,
                                 torch::Tensor const& b_scales,
                                 torch::Tensor const& c_scales) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);
  TORCH_CHECK(c.dtype() == torch::kInt8);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(c_scales.dtype() == torch::kFloat32);

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  return cutlass_scaled_mm_dispatcher<
      cutlass_2x_gemm<cutlass::arch::Sm80, int8_t, int8_t, ScaledQuantEpilogue,
                      TileShape, WarpShape, InstructionShape, 5>>(
      c, a, b, a_scales, b_scales, c_scales);
}

void cutlass_scaled_mm_qout_sm89(torch::Tensor& c, torch::Tensor const& a,
                                 torch::Tensor const& b,
                                 torch::Tensor const& a_scales,
                                 torch::Tensor const& b_scales,
                                 torch::Tensor const& c_scales) {
  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(c_scales.dtype() == torch::kFloat32);

  if (a.dtype() == torch::kInt8) {
    TORCH_CHECK(b.dtype() == torch::kInt8);
    TORCH_CHECK(c.dtype() == torch::kInt8);

    return cutlass_scaled_mm_dispatcher<cutlass_2x_gemm<
        cutlass::arch::Sm89, int8_t, int8_t, ScaledDequantEpilogue, TileShape,
        WarpShape, InstructionShape, 5>>(c, a, b, a_scales, b_scales, c_scales);
  } else {
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(c.dtype() == torch::kFloat8_e4m3fn);

    return cutlass_scaled_mm_dispatcher<cutlass_2x_gemm<
        cutlass::arch::Sm89, cutlass::float_e4m3_t, cutlass::float_e4m3_t,
        ScaledDequantEpilogue, TileShape, WarpShape, InstructionShape, 5>>(
        c, a, b, a_scales, b_scales, c_scales);
  }
}

