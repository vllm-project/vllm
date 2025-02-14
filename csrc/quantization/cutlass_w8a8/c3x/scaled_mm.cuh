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

  using EpilogueDescriptor =
      cutlass::epilogue::collective::detail::EpilogueDescriptor<
          TileShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementD,
          ElementD, EpilogueSchedule>;

  using Epilogue = Epilogue_<ElementAcc, ElementD, EpilogueDescriptor>;

  using StrideD = Stride<int64_t, Int<1>, Int<0>>;
  using ElementC = void;
  using StrideC = StrideD;

  using EVTCompute = typename Epilogue::EVTCompute;

  // These are the minimum alignments needed for the kernels to compile
  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD = 4;

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

}  // namespace vllm
