// clang-format will break include orders
// clang-format off
#include <cudaTypedefs.h>

#if defined CUDA_VERSION && CUDA_VERSION >= 12000

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "broadcast_load_epilogue_c3x.hpp"
#include "common.hpp"
// clang-format on

using namespace cute;

/*
   This defines a quantized GEMM operation with dequantized output, similar to
   torch._scaled_mm. It is defined using the CUTLASS 3.x API, and is used for
   NVIDIA GPUs with sm90a (Hopper) or later.

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

namespace {

uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

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

template <typename ElementAB_, typename ElementD_, typename TileShape,
          typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_3x_gemm {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;
  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using EpilogueDescriptor =
      cutlass::epilogue::collective::detail::EpilogueDescriptor<
          TileShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementD,
          ElementD, EpilogueSchedule>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using ScaleA = cutlass::epilogue::fusion::Sm90ColOrScalarBroadcast<
      0 /*Stages*/, typename EpilogueDescriptor::TileShape, float,
      Stride<Int<1>, Int<0>, Int<0>>>;

  using ScaleBDescriptor =
      cutlass::epilogue::collective::detail::RowBroadcastDescriptor<
          EpilogueDescriptor, float>;

  using ScaleB = cutlass::epilogue::fusion::Sm90RowOrScalarBroadcast<
      ScaleBDescriptor::Stages, typename EpilogueDescriptor::TileShape,
      typename ScaleBDescriptor::Element, Stride<Int<0>, Int<1>, Int<0>>>;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 =
      cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleB, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementD, float,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 =
      cutlass::epilogue::fusion::Sm90EVT<Compute1, ScaleA, EVTCompute0>;

  using StrideD = Stride<int64_t, Int<1>, Int<0>>;
  using ElementC = void;
  using StrideC = StrideD;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc, float, ElementC, StrideC, 4, ElementD, StrideD, 4,
          EpilogueSchedule, EVTCompute1>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  // clang-format off
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, 
          ElementAB, cutlass::layout::RowMajor, 16, 
          ElementAB, cutlass::layout::ColumnMajor, 16, 
          ElementAcc, TileShape, ClusterShape,
          Stages,
          KernelSchedule>::CollectiveOp;
  // clang-format on

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>>;

  struct GemmKernel : public KernelType {};
};

template <typename Gemm>
void cutlass_scaled_mm_dq_dispatcher(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     torch::Tensor const& a_scales,
                                     torch::Tensor const& b_scales) {
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

  using StrideA = Stride<int64_t, Int<1>, Int<0>>;
  using StrideB = Stride<int64_t, Int<1>, Int<0>>;
  using StrideC = typename Gemm::StrideC;

  StrideA a_stride{lda, Int<1>{}, Int<0>{}};
  StrideB b_stride{ldb, Int<1>{}, Int<0>{}};
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  using GemmKernel = typename Gemm::GemmKernel;
  typename GemmKernel::ProblemShape prob_shape{m, n, k, 1};

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr,
                                                       b_stride};

  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, c_ptr, c_stride, c_ptr, c_stride};

  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                      prob_shape, mainloop_args, epilogue_args};

  using ScaleA_Args = typename Gemm::ScaleA::Arguments;
  using ScaleB_Args = typename Gemm::ScaleB::Arguments;

  ScaleA_Args a_args{a_scales.data_ptr<float>(), a_scales.numel() != 1, {}};
  ScaleB_Args b_args{b_scales.data_ptr<float>(), b_scales.numel() != 1, {}};

  args.epilogue.thread = {a_args, {b_args}};

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  cutlass::Status status = gemm_op.run(args, workspace.get(), stream);
  CUTLASS_CHECK(status);
}

template <typename InType, typename OutType, int32_t M>
struct sm90_fp8_config {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;

  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, TileShape, ClusterShape, KernelSchedule,
                      EpilogueSchedule>;
};

template <typename InType, typename OutType>
struct sm90_fp8_config<InType, OutType, 128> {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;

  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, TileShape, ClusterShape, KernelSchedule,
                      EpilogueSchedule>;
};

template <typename InType, typename OutType>
struct sm90_fp8_config<InType, OutType, 64> {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _128>;
  using ClusterShape = Shape<_1, _8, _1>;

  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, TileShape, ClusterShape, KernelSchedule,
                      EpilogueSchedule>;
};

}  // namespace

template <typename InType, typename OutType>
void cutlass_scaled_mm_dq_sm90_fp8_dispatch(torch::Tensor& out,
                                            torch::Tensor const& a,
                                            torch::Tensor const& b,
                                            torch::Tensor const& a_scales,
                                            torch::Tensor const& b_scales) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  using Cutlass3xGemmDefault =
      typename sm90_fp8_config<InType, OutType, 0>::Cutlass3xGemm;
  using Cutlass3xGemmM64 =
      typename sm90_fp8_config<InType, OutType, 64>::Cutlass3xGemm;
  using Cutlass3xGemmM128 =
      typename sm90_fp8_config<InType, OutType, 128>::Cutlass3xGemm;

  uint32_t const m = a.size(0);
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(64), next_pow_2(m));  // next power of 2

  if (mp2 <= 64) {
    // m in [1, 64]
    return cutlass_scaled_mm_dq_dispatcher<Cutlass3xGemmM64>(
        out, a, b, a_scales, b_scales);
  } else if (mp2 <= 128) {
    // m in (64, 128]
    return cutlass_scaled_mm_dq_dispatcher<Cutlass3xGemmM128>(
        out, a, b, a_scales, b_scales);
  } else {
    // m in (128, inf)
    return cutlass_scaled_mm_dq_dispatcher<Cutlass3xGemmDefault>(
        out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_dq_sm90(torch::Tensor& out, torch::Tensor const& a,
                               torch::Tensor const& b,
                               torch::Tensor const& a_scales,
                               torch::Tensor const& b_scales) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (a.dtype() == torch::kInt8) {
    TORCH_CHECK(b.dtype() == torch::kInt8);

    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _2, _1>;
    using KernelSchedule =
        typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
    using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_scaled_mm_dq_dispatcher<
          cutlass_3x_gemm<int8_t, cutlass::bfloat16_t, TileShape, ClusterShape,
                          KernelSchedule, EpilogueSchedule>>(
          out, a, b, a_scales, b_scales);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);

      return cutlass_scaled_mm_dq_dispatcher<
          cutlass_3x_gemm<int8_t, cutlass::half_t, TileShape, ClusterShape,
                          KernelSchedule, EpilogueSchedule>>(
          out, a, b, a_scales, b_scales);
    }
  } else {
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_scaled_mm_dq_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                                    cutlass::bfloat16_t>(
          out, a, b, a_scales, b_scales);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);
      return cutlass_scaled_mm_dq_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                                    cutlass::half_t>(
          out, a, b, a_scales, b_scales);
    }
  }
}

#endif
