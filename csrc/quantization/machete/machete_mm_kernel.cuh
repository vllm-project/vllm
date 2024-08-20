#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

// clang-format off
// The cutlass include order matters (annoyingly)
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
// clang-format on

#include "cutlass_extensions/cute_utils.cuh"
#include "cutlass_extensions/vllm_numeric_conversion.cuh"
#include "machete_collective_builder.cuh"
#include "machete_prepacked_layout.cuh"
#include "machete_interleaving_utils.cuh"

namespace machete {

using namespace cute;

// NOTE This kernel computes D = alpha * A * B + beta * C by computing
//   D^t = alpha * B^t * A^t + beta * C^t, this is because the wgmma
//   instructions only support sourcing from registers for the left-hand
//   operand, we want to upconvert/decompress the quantized operand in
//   register. Since the primary use case we want to support is Y = XW^t where
//   W is quantized, in this situation or right-hand operand is quantized so
//   we compute the transpose to move it to the left-hand side.
template <typename ElementA_, typename ElementB_, typename ElementD_,
          typename AccumulatorT, typename ScaleT, typename ZeroT,
          class KernelSchedule, typename ScheduleConfig, bool with_C,
          bool with_scales, bool with_zeropoints>
struct MacheteKernelTemplate {
  using MmaType = ElementA_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementD = ElementD_;
  using ElementC = cute::conditional_t<with_C, ElementD, void>;
  using ElementZ = ZeroT;
  using ElementS = ScaleT;

  using ElementAccumulator =
      AccumulatorT;  // Element type for internal accumulation
  using ElementCompute = AccumulatorT;  // For Epilogue

  using BTypeTuple = cute::conditional_t<
      with_scales,
      cute::conditional_t<with_zeropoints,
                          cute::tuple<ElementB, ElementS, ElementZ>,
                          cute::tuple<ElementB, ElementS>>,
      ElementB>;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;
  using LayoutScale = cutlass::layout::RowMajor;
  // not actually used since B has the prepacked layout, but required by cutlass
  using _LayoutB = cutlass::layout::ColumnMajor;

  // Interface strides expected by create_arguments (will get transposed)
  using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using StrideC = cutlass::detail::TagToStrideA_t<LayoutC>;
  using StrideD = cutlass::detail::TagToStrideA_t<LayoutD>;
  using StrideS = cutlass::detail::TagToStrideA_t<LayoutScale>;
  using StrideZ = StrideS;

  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutC_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutC>::type;
  using LayoutD_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutD>::type;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using PrepackedLayoutB =
      PrepackedLayoutBTemplate<ElementA_, ElementB_, ElementD_, AccumulatorT,
                               LayoutA_Transpose, KernelSchedule>;

  static int constexpr TileShapeK =
      128 * 8 / cutlass::sizeof_bits<MmaType>::value;
  static int constexpr AlignmentA = 128 / cutlass::sizeof_bits_v<ElementA>;
  static int constexpr AlignmentB = 128 / cutlass::sizeof_bits_v<ElementB>;
  static int constexpr AlignmentC =
      (with_C) ? 128 / cutlass::sizeof_bits_v<ElementC> : 0;
  static int constexpr AlignmentD = 128 / cutlass::sizeof_bits_v<ElementD>;

  using TileShape = decltype(append(typename ScheduleConfig::TileShapeNM{},
                                    cute::Int<TileShapeK>{}));
  using ClusterShape = typename ScheduleConfig::ClusterShape;
  using EpilogueSchedule = typename ScheduleConfig::EpilogueSchedule;
  using EpilogueTileType = typename ScheduleConfig::EpilogueTileType;
  using TileScheduler = typename ScheduleConfig::TileScheduler;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
          ElementAccumulator, ElementAccumulator, ElementC, LayoutC_Transpose,
          AlignmentC, ElementD, LayoutD_Transpose, AlignmentD,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::VLLMCollectiveBuilder<
          cutlass::gemm::collective::MacheteKernelTag, ArchTag, OperatorClass,
          BTypeTuple, PrepackedLayoutB, AlignmentB, ElementA, LayoutA_Transpose,
          AlignmentA, ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // stride_B is unused (since B is prepacked), but still required by cutlass
  using _StrideB = cutlass::detail::TagToStrideB_t<_LayoutB>;

  using Arguments = typename Gemm::Arguments;
  using MainloopArguments = typename GemmKernel::MainloopArguments;
  using EpilogueArguments = typename GemmKernel::EpilogueArguments;

  template <typename ShapeA, typename ShapeC, typename ShapeD, typename ShapeS,
            typename ShapeZ>
  static Arguments create_arguments(
      cudaStream_t stream,
      ElementA const* A_ptr,  // A is an MxK matrix
      Layout<ShapeA, StrideA> const& layout_A,
      ElementB const* B_ptr,  // B is an KxN prepacked matrix
      ElementD* D_ptr,        // D is an MxN matrix
      Layout<ShapeD, StrideD> const& layout_D,
      ElementC const* C_ptr,  // C is an MxN matrix
      std::optional<Layout<ShapeC, StrideC>> const& layout_C,
      ElementS const* S_ptr,  // S is an scale_KxN matrix
      std::optional<Layout<ShapeS, StrideS>> const& layout_S,
      ElementZ const* Z_ptr,  // Z is an scale_KxN matrix
      std::optional<Layout<ShapeZ, StrideZ>> const& layout_Z,
      ElementCompute alpha, ElementCompute beta,
      std::optional<int> maybe_group_size) {
    static_assert(!with_zeropoints || with_scales);

    int M = size<0>(layout_A), N = size<1>(layout_D), K = size<1>(layout_A);

    int const group_size = maybe_group_size.value_or(K);
    int const scale_k = (K + group_size - 1) / group_size;

    TORCH_CHECK(size<0>(layout_A) == M && size<1>(layout_A) == K);
    TORCH_CHECK(size<0>(layout_D) == M && size<1>(layout_D) == N);

    if constexpr (with_C) {
      TORCH_CHECK(C_ptr && layout_C);
    } else {
      TORCH_CHECK(!C_ptr, "C not supported");
    }

    if constexpr (with_scales) {
      TORCH_CHECK(S_ptr && layout_S);
      TORCH_CHECK((size<0>(*layout_S) == scale_k && size<1>(*layout_S) == N));
    } else {
      TORCH_CHECK(!S_ptr, "Scales not supported");
    }

    if constexpr (with_zeropoints) {
      TORCH_CHECK(Z_ptr && layout_Z);
      TORCH_CHECK((size<0>(*layout_Z) == scale_k && size<1>(*layout_Z) == N));
      TORCH_CHECK(layout_S && *layout_Z == *layout_S,
                  "Scales and zeros must have the same layout");
    } else {
      TORCH_CHECK(!Z_ptr, "Zeropoints not supported");
    }

    // Transpose A and D
    // A doesn't need to be transposed since cutlass expects a NxK matrix
    //  for B (which is At)
    auto stride_At = layout_A.stride();
    auto stride_Dt = permute_layout<1, 0, 2>(layout_D).stride();
    auto stride_Ct = stride_Dt;
    if (layout_C) {
      stride_Ct = permute_layout<1, 0, 2>(*layout_C).stride();
    }

    MainloopArguments mainloop_arguments{};
    EpilogueArguments epilogue_arguments{
        {alpha, beta}, C_ptr, stride_Ct, D_ptr, stride_Dt};

    if constexpr (with_scales && with_zeropoints) {
      auto stride_S = permute_layout<1, 0, 2>(*layout_S).stride();
      mainloop_arguments =
          MainloopArguments{B_ptr, _StrideB{}, A_ptr,      stride_At,
                            S_ptr, stride_S,   group_size, Z_ptr};
    } else if constexpr (with_scales) {
      auto stride_S = permute_layout<1, 0, 2>(*layout_S).stride();
      mainloop_arguments = MainloopArguments{
          B_ptr, _StrideB{}, A_ptr, stride_At, S_ptr, stride_S, group_size};
    } else {
      mainloop_arguments =
          MainloopArguments{B_ptr, _StrideB{}, A_ptr, stride_At};
    }

    return Arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                     {N, M, K, 1},
                     mainloop_arguments,
                     epilogue_arguments};
  };

  static size_t get_workspace_size(Arguments const& args) {
    return Gemm::get_workspace_size(args);
  }

  static bool can_implement(Arguments const& args) {
    return Gemm::can_implement(args) == cutlass::Status::kSuccess;
  }

  static void run(Arguments const& args, void* workspace, cudaStream_t stream) {
    Gemm gemm_op;

    cutlass::Status status = gemm_op.initialize(args, workspace, stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "Machete kernel failed to initialize workspace");

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "Machete kernel failed");
  }
};

};  // namespace machete
