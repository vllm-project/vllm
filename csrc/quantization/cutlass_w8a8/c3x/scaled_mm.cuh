#pragma once

// clang-format will break include orders
// clang-format off

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "core/math.hpp"
#include "cutlass_extensions/common.hpp"
// clang-format on

/*
  Epilogues defined in,
  csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp,
  must contain a public type named EVTCompute of type Sm90EVT, as well as a
  static prepare_args function that constructs an EVTCompute::Arguments struct.
*/

using namespace cute;

namespace vllm {

template <typename ElementAB_, typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_3x_gemm {
  using ElementAB = ElementAB_;
  using ElementD = ElementD_;
  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;

  using StrideD = Stride<int64_t, Int<1>, Int<0>>;
  using ElementC = void;
  using StrideC = StrideD;

  using EVTCompute = typename Epilogue::EVTCompute;

  // These are the minimum alignments needed for the kernels to compile
  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc, float, ElementC, StrideC, AlignmentCD, ElementD, StrideD,
          AlignmentCD, EpilogueSchedule, EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  // clang-format off
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, 
          ElementAB, cutlass::layout::RowMajor, AlignmentAB, 
          ElementAB, cutlass::layout::ColumnMajor, AlignmentAB, 
          ElementAcc, TileShape, ClusterShape,
          Stages,
          KernelSchedule>::CollectiveOp;
  // clang-format on

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>>;

  struct GemmKernel : public KernelType {};
};

template <typename ElementAB_, typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_3x_gemm_sm100 {
  using ElementAB = ElementAB_;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementAB>::value;

  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementAB>::value;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementD_>::value;

  using ElementD = ElementD_;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = AlignmentC;

  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;
  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;

  // MMA type
  using ElementAccumulator = float;

  // Epilogue types
  using ElementBias = cutlass::half_t;
  using ElementCompute = float;
  using ElementAux = ElementD;
  using LayoutAux = LayoutD;
  using ElementAmax = float;

  using EVTCompute = typename Epilogue::EVTCompute;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD, EpilogueSchedule,
          EVTCompute>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementAB,
          LayoutA, AlignmentA, ElementAB, LayoutB, AlignmentB,
          ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
};

template <typename ElementAB_, typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_3x_gemm_sm120 {
  using ElementAB = ElementAB_;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementAB>::value;

  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementAB>::value;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementD_>::value;

  using ElementD = ElementD_;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = AlignmentC;

  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;
  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;

  // MMA type
  using ElementAccumulator = float;

  // Epilogue types
  using ElementBias = cutlass::half_t;
  using ElementCompute = float;
  using ElementAux = ElementD;
  using LayoutAux = LayoutD;
  using ElementAmax = float;

  using EVTCompute = typename Epilogue::EVTCompute;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD, EpilogueSchedule,
          EVTCompute>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementAB,
          LayoutA, AlignmentA, ElementAB, LayoutB, AlignmentB,
          ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
};

}  // namespace vllm
