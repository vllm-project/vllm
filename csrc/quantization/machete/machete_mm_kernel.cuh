#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

// clang-format off
// The cutlass inlcude order matters (annoyingly)
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

#include "cutlass/util/device_memory.h"

#include "cutlass_extensions/gemm/kernel/vllm_tile_schedulers.cuh"
#include "cutlass_extensions/cute_utils.cuh"
#include "cutlass_extensions/vllm_numeric_conversion.cuh"
#include "machete_collective_builder.cuh"
#include "machete_prepacked_layout.cuh"

namespace machete {

using namespace cute;

// NOTE This kernel computes D = alpha * A * B + beta * C by computing
//   D^t = alpha * B^t * A^t + beta * C^t, this is because the wgmma
//   instructions only support sourcing from registers for the left-hand
//   operand, we want to upconvert/decompress the quantized oprand in
//   register. Since the primary use case we want to support is Y = XW^t where
//   W is quantized, in this situation or right-hand operand is quantized so
//   we compute the transpose to move it to the left-hand side.
template <typename ElementA_, typename ElementB_, typename ElementD_,
          typename AccumulatorT, typename ScaleT, typename ZeroT,
          class KernelSchedule>
struct KernelTemplate {
  using MmaType = ElementA_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementD = ElementD_;
  using ElementAccumulator = AccumulatorT;

  using LayoutA_ = cutlass::layout::RowMajor;
  using LayoutScale_ = cutlass::layout::RowMajor;

  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA_>::type;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using PrepackedLayoutBB =
      PrepackedLayoutBBTemplate<ElementA_, ElementB_, ElementD_, AccumulatorT,
                                LayoutA_Transpose, KernelSchedule>;

  // clang-format off
  template <typename ScheduleConfig,
            bool with_C, 
            bool with_scales, 
            bool with_zeropoints>
  // clang-format on
  struct Speacialization {
    using MmaType = ElementA_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementD = ElementD_;
    using ElementC = cute::conditional_t<with_C, ElementD, void>;
    using ElementZero = ZeroT;
    using ElementScale = ScaleT;
    using ElementAccumulator =
        AccumulatorT;  // Element type for internal accumulation
    using ElementCompute = AccumulatorT;  // For Epilogue

    using BTypeTuple = cute::conditional_t<
        with_scales,
        cute::conditional_t<with_zeropoints,
                            cute::tuple<ElementB, ElementScale, ElementZero>,
                            cute::tuple<ElementB, ElementScale>>,
        ElementB>;

    using LayoutA = LayoutA_;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = LayoutC;
    using LayoutScale = cutlass::layout::RowMajor;

    using LayoutB_Transpose =
        typename cutlass::layout::LayoutTranspose<LayoutB>::type;
    using LayoutC_Transpose =
        typename cutlass::layout::LayoutTranspose<LayoutC>::type;
    using LayoutD_Transpose =
        typename cutlass::layout::LayoutTranspose<LayoutD>::type;

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
            BTypeTuple, PrepackedLayoutBB, AlignmentB, ElementA,
            LayoutA_Transpose, AlignmentA, ElementAccumulator, TileShape,
            ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                sizeof(typename CollectiveEpilogue::SharedStorage))>,
            KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,  // Indicates ProblemShape
        CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
    using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;
    using StrideS = typename CollectiveMainloop::StrideScale;

    using Arguments = typename Gemm::Arguments;
    using MainloopArguments = typename GemmKernel::MainloopArguments;
    using EpilogueArguments = typename GemmKernel::EpilogueArguments;

    static Arguments create_arguments(cudaStream_t stream, int M, int N, int K,
                                      ElementA const* A, ElementB const* B,
                                      ElementC const* C, ElementD* D,
                                      ElementScale const* scales,
                                      ElementZero const* zeros,
                                      ElementCompute alpha, ElementCompute beta,
                                      std::optional<int> maybe_group_size) {
      // if we have zeropoints we need scales
      static_assert(!with_zeropoints || with_scales);
      // if beta != 0 then we need C
      TORCH_CHECK(with_C || (!with_C && beta == 0));
      // if with_scales, we need a scales pointer
      TORCH_CHECK(with_scales || !scales);
      // if with_zeropoints, we need a zeros pointer
      TORCH_CHECK(with_zeropoints || !zeros);

      static int constexpr L = 1;
      int const group_size = maybe_group_size.value_or(K);
      int const scale_k = (K + group_size - 1) / group_size;

      // not stride_B is unused
      auto stride_A = make_cute_stride(StrideA{}, N, K, L);
      auto stride_B = make_cute_stride(StrideB{}, M, K, L);
      auto stride_C = make_cute_stride(StrideC{}, N, M, L);
      auto stride_D = make_cute_stride(StrideD{}, N, M, L);
      auto stride_S = make_cute_stride(StrideS{}, N, scale_k, L);

      MainloopArguments mainloop_arguments{};
      EpilogueArguments epilogue_arguments{
          {alpha, beta}, C, stride_C, D, stride_D};

      if constexpr (with_scales && with_zeropoints) {
        mainloop_arguments = MainloopArguments{
            B, stride_B, A, stride_A, scales, stride_S, group_size, zeros};
      } else if constexpr (with_scales) {
        mainloop_arguments = MainloopArguments{
            B, stride_B, A, stride_A, scales, stride_S, group_size};
      } else {
        mainloop_arguments = MainloopArguments{B, stride_B, A, stride_A};
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

    static void run(Arguments const& args, void* workspace,
                    cudaStream_t stream) {
      Gemm gemm_op;

      cutlass::Status status = gemm_op.initialize(args, workspace, stream);
      TORCH_CHECK(status == cutlass::Status::kSuccess,
                  "Machete kernel failed to initialize workspace");

      status = gemm_op.run(stream);
      TORCH_CHECK(status == cutlass::Status::kSuccess, "Machete kernel failed");
    }
  };
};

};  // namespace machete
