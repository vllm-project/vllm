#include <stddef.h>
#include <torch/all.h>

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

/*
   This file defines quantized GEMM operations using the CUTLASS 2.x API, for
   NVIDIA GPUs with SM versions prior to sm90 (Hopper).

   Epilogue functions can be defined to post-process the output before it is
   written to GPU memory.
   Epilogues must contain a public type named EVTCompute of type Sm80EVT,
   as well as a static prepare_args function that constructs an
   EVTCompute::Arguments struct.
*/

namespace {

// Wrappers for the GEMM kernel that is used to guard against compilation on
// architectures that will never use the kernel. The purpose of this is to
// reduce the size of the compiled binary.
// __CUDA_ARCH__ is not defined in host code, so this lets us smuggle the ifdef
// into code that will be executed on the device where it is defined.
template <typename Kernel>
struct enable_sm75_to_sm80 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
    Kernel::invoke(std::forward<Args>(args)...);
#endif
  }
};

template <typename Kernel>
struct enable_sm80_to_sm89 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 890
    Kernel::invoke(std::forward<Args>(args)...);
#endif
  }
};

template <typename Kernel>
struct enable_sm89_to_sm90 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE static void invoke(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 890 && __CUDA_ARCH__ < 900
    Kernel::invoke(std::forward<Args>(args)...);
#endif
  }
};

/*
 * This class provides the common ScaleA and ScaleB descriptors for the
 * ScaledEpilogue and ScaledEpilogueBias classes.
 */
template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBase {
 protected:
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

  using ScaleA = cutlass::epilogue::threadblock::VisitorColOrScalarBroadcast<
      OutputTileThreadMap, float, Stride<Int<1>, Int<0>, Int<0>>>;

  using ScaleB = cutlass::epilogue::threadblock::VisitorRowOrScalarBroadcast<
      OutputTileThreadMap, float, Stride<Int<0>, Int<1>, Int<0>>>;
};

/*
 This epilogue function defines a quantized GEMM operation similar to
 torch._scaled_mm.

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
struct ScaledEpilogue
    : private ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 private:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::ScaleA;
  using ScaleB = typename SUPER::ScaleB;

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

  static ArgumentType prepare_args(torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales) {
    using ScaleAArgs = typename ScaleA::Arguments;
    using ScaleBArgs = typename ScaleB::Arguments;

    ScaleBArgs b_args{b_scales.data_ptr<float>(), b_scales.numel() != 1, {}};
    ScaleAArgs a_args{a_scales.data_ptr<float>(), a_scales.numel() != 1, {}};

    typename EVTCompute0::Arguments evt0_compute_args{b_args};

    typename EVTCompute::Arguments evt_compute_args{a_args, evt0_compute_args};
    return evt_compute_args;
  }
};

template <typename ElementD, typename OutputTileThreadMap>
struct ScaledEpilogueBias
    : private ScaledEpilogueBase<ElementD, OutputTileThreadMap> {
 private:
  using SUPER = ScaledEpilogueBase<ElementD, OutputTileThreadMap>;
  using Accum = typename SUPER::Accum;
  using ScaleA = typename SUPER::ScaleA;
  using ScaleB = typename SUPER::ScaleB;

  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiply_add, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, ElementD, Stride<Int<0>, Int<1>, Int<0>>>;

 public:
  using EVTCompute = cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA,
                                                             EVTCompute0, Bias>;
  using ArgumentType = typename EVTCompute::Arguments;

  static ArgumentType prepare_args(torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales,
                                   torch::Tensor const& bias) {
    using ScaleAArgs = typename ScaleA::Arguments;
    using ScaleBArgs = typename ScaleB::Arguments;
    using BiasArgs = typename Bias::Arguments;

    ScaleBArgs b_args{b_scales.data_ptr<float>(), b_scales.numel() != 1, {}};
    ScaleAArgs a_args{a_scales.data_ptr<float>(), a_scales.numel() != 1, {}};
    BiasArgs bias_args{static_cast<ElementD*>(bias.data_ptr()), {}};

    typename EVTCompute0::Arguments evt0_compute_args{b_args};

    typename EVTCompute::Arguments evt_compute_args{a_args, evt0_compute_args,
                                                    bias_args};
    return evt_compute_args;
  }
};

template <typename Arch, template <typename> typename ArchGuard,
          typename ElementAB_, typename ElementD_,
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
    ArchGuard<typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
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
      >::GemmKernel>;
  // clang-format on

  using Op = cutlass::gemm::device::GemmUniversalAdapter<KernelType>;
};

template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller(torch::Tensor& out, torch::Tensor const& a,
                         torch::Tensor const& b,
                         EpilogueArgs&&... epilogue_params) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);
  cutlass::gemm::GemmCoord problem_size{m, n, k};

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

  using StrideC = Stride<int64_t, Int<1>, Int<0>>;
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB const*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  typename Gemm::D::Arguments d_args{c_ptr, c_stride};

  using Epilogue = typename Gemm::Epilogue;
  auto evt_args =
      Epilogue::prepare_args(std::forward<EpilogueArgs>(epilogue_params)...);

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
  size_t workspace_size = gemm_op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  CUTLASS_CHECK(gemm_op.can_implement(args));
  cutlass::Status status = gemm_op(args, workspace.get(), stream);
  CUTLASS_CHECK(status);
}

template <typename Gemm, typename FallbackGemm, typename... EpilogueArgs>
void fallback_cutlass_gemm_caller(torch::Tensor& out, torch::Tensor const& a,
                                  torch::Tensor const& b,
                                  EpilogueArgs&&... args) {
  // In some cases, the GPU isn't able to accommodate the
  // shared memory requirements of the Gemm. In such cases, use
  // the FallbackGemm instead.
  static const int max_shared_mem_per_block_opt_in =
      get_cuda_max_shared_memory_per_block_opt_in(0);

  size_t const gemm_shared_mem_size =
      sizeof(typename Gemm::KernelType::SharedStorage);
  size_t const fallback_gemm_shared_mem_size =
      sizeof(typename FallbackGemm::KernelType::SharedStorage);

  if (gemm_shared_mem_size <= max_shared_mem_per_block_opt_in) {
    return cutlass_gemm_caller<Gemm>(out, a, b,
                                     std::forward<EpilogueArgs>(args)...);
  } else {
    TORCH_CHECK(fallback_gemm_shared_mem_size <=
                max_shared_mem_per_block_opt_in);
    return cutlass_gemm_caller<FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
}

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm80_config_default {
  // This config is used in 2 cases,
  //  - M in (128, inf)
  //  - M in (64, 128] and N >= 8192
  // Shared Memory required by this Gemm - 81920 bytes
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 5>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm80_config_M64 {
  // This config is used in 2 cases,
  // - M in (32, 64]
  // - M in (64, 128] and N < 8192
  // Shared Memory required by this Gemm - 122880 bytes
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = typename cutlass::gemm::GemmShape<64, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 5>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm80_config_M32 {
  // M in (16, 32]
  // Shared Memory required by this Gemm - 61440 bytes
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = typename cutlass::gemm::GemmShape<32, 64, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 5>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm80_config_M16 {
  // M in [1, 16]
  // Shared Memory required by this Gemm - 51200 bytes
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = typename cutlass::gemm::GemmShape<16, 64, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 5>;
};

}  // namespace

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_gemm_sm80_dispatch(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, int8_t>());
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  using Cutlass2xGemmDefault =
      typename sm80_config_default<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM128BigN =
      typename sm80_config_default<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM128SmallN =
      typename sm80_config_M64<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM64 =
      typename sm80_config_M64<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM32 =
      typename sm80_config_M32<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM16 =
      typename sm80_config_M16<InType, OutType, Epilogue>::Cutlass2xGemm;

  // Due to shared memory requirements, some Gemms may fail to run on some
  // GPUs. As the name indicates, the Fallback Gemm is used as an alternative
  // in such cases.
  // sm80_config_M16 has the least shared-memory requirement. However,
  // based on some profiling, we select sm80_config_M32 as a better alternative
  // performance wise.
  using FallbackGemm =
      typename sm80_config_M32<InType, OutType, Epilogue>::Cutlass2xGemm;

  uint32_t const m = a.size(0);
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(16), next_pow_2(m));  // next power of 2
  if (mp2 <= 16) {
    // M in [1, 16]
    return fallback_cutlass_gemm_caller<Cutlass2xGemmM16, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 32) {
    // M in (16, 32]
    return fallback_cutlass_gemm_caller<Cutlass2xGemmM32, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 64) {
    // M in (32, 64]
    return fallback_cutlass_gemm_caller<Cutlass2xGemmM64, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // M in (64, 128]
    uint32_t const n = out.size(1);
    bool const small_n = n < 8192;
    if (small_n) {
      return fallback_cutlass_gemm_caller<Cutlass2xGemmM128SmallN,
                                          FallbackGemm>(
          out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      return fallback_cutlass_gemm_caller<Cutlass2xGemmM128BigN, FallbackGemm>(
          out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  } else {
    // M in (128, inf)
    return fallback_cutlass_gemm_caller<Cutlass2xGemmDefault, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
}

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm75_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<8, 8, 16>;

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_caller<cutlass_2x_gemm<
        cutlass::arch::Sm75, enable_sm75_to_sm80, int8_t, cutlass::bfloat16_t,
        Epilogue, TileShape, WarpShape, InstructionShape, 2>>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_caller<cutlass_2x_gemm<
        cutlass::arch::Sm75, enable_sm75_to_sm80, int8_t, cutlass::half_t,
        Epilogue, TileShape, WarpShape, InstructionShape, 2>>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

void cutlass_scaled_mm_sm75(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm75_epilogue<ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm75_epilogue<ScaledEpilogue>(out, a, b, a_scales,
                                                           b_scales);
  }
}

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm80_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_sm80_dispatch<int8_t, cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_sm80_dispatch<int8_t, cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

void cutlass_scaled_mm_sm80(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm80_epilogue<ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm80_epilogue<ScaledEpilogue>(out, a, b, a_scales,
                                                           b_scales);
  }
}

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm89_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  if (a.dtype() == torch::kInt8) {
    TORCH_CHECK(b.dtype() == torch::kInt8);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_caller<cutlass_2x_gemm<
          cutlass::arch::Sm89, enable_sm89_to_sm90, int8_t, cutlass::bfloat16_t,
          Epilogue, TileShape, WarpShape, InstructionShape, 5>>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      assert(out.dtype() == torch::kFloat16);
      return cutlass_gemm_caller<cutlass_2x_gemm<
          cutlass::arch::Sm89, enable_sm89_to_sm90, int8_t, cutlass::half_t,
          Epilogue, TileShape, WarpShape, InstructionShape, 5>>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  } else {
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                          cutlass::float_e4m3_t, cutlass::bfloat16_t, Epilogue,
                          TileShape, WarpShape, InstructionShape, 5>>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);
      return cutlass_gemm_caller<
          cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                          cutlass::float_e4m3_t, cutlass::half_t, Epilogue,
                          TileShape, WarpShape, InstructionShape, 5>>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  }
}

void cutlass_scaled_mm_sm89(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm89_epilogue<ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm89_epilogue<ScaledEpilogue>(out, a, b, a_scales,
                                                           b_scales);
  }
}
