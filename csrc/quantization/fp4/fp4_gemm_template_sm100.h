#pragma once

#ifndef _WIN32
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // #ifndef _WIN32

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/gemm.h"

#ifndef _WIN32
  #pragma GCC diagnostic pop
#endif  // #ifndef _WIN32

using namespace cute;

struct DeviceGemmFp4GemmSm100_Half {
  using OutElementType = cutlass::half_t;
  using CTAShape = cute::Shape<cute::_128, cute::_128, cute::_256>;

  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ElementType = cutlass::float_e2m1_t;
  using Arch = cutlass::arch::Sm100;
  // Input A
  using ElementA = ElementType;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementType>::value;
  // Input B
  using ElementB = ElementType;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementType>::value;
  // Input C
  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 4;

  using SFType = cutlass::float_ue4m3_t;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          Arch, OperatorClass, CTAShape, ClusterShape, EpilogueTileType,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          OutElementType, LayoutC, AlignmentC, EpilogueSchedule,
          cutlass::epilogue::fusion::LinearCombination<
              OutElementType, float, void, float>>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          Arch, cutlass::arch::OpClassBlockScaledTensorOp,
          cute::tuple<ElementA, SFType>, LayoutA, AlignmentA,
          cute::tuple<ElementB, SFType>, LayoutB, AlignmentB,
          ElementAccumulator, CTAShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledOmmaVs16Sm100>::
          CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct DeviceGemmFp4GemmSm100_BFloat16 {
  using OutElementType = cutlass::bfloat16_t;
  using CTAShape = cute::Shape<cute::_128, cute::_128, cute::_256>;

  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ElementType = cutlass::float_e2m1_t;
  using Arch = cutlass::arch::Sm100;
  // Input A
  using ElementA = ElementType;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementType>::value;
  // Input B
  using ElementB = ElementType;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementType>::value;
  // Input C
  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 4;

  using SFType = cutlass::float_ue4m3_t;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          Arch, OperatorClass, CTAShape, ClusterShape, EpilogueTileType,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          OutElementType, LayoutC, AlignmentC, EpilogueSchedule,
          cutlass::epilogue::fusion::LinearCombination<
              OutElementType, float, void, float>>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          Arch, cutlass::arch::OpClassBlockScaledTensorOp,
          cute::tuple<ElementA, SFType>, LayoutA, AlignmentA,
          cute::tuple<ElementB, SFType>, LayoutB, AlignmentB,
          ElementAccumulator, CTAShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledOmmaVs16Sm100>::
          CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct DeviceGemmFp4GemmSm100_Float {
  using OutElementType = float;
  using CTAShape = cute::Shape<cute::_128, cute::_128, cute::_256>;

  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using ElementType = cutlass::float_e2m1_t;
  using Arch = cutlass::arch::Sm100;
  // Input A
  using ElementA = ElementType;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementType>::value;
  // Input B
  using ElementB = ElementType;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementType>::value;
  // Input C
  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 4;

  using SFType = cutlass::float_ue4m3_t;
  using ElementCompute = float;
  using ElementAccumulator = float;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          Arch, OperatorClass, CTAShape, ClusterShape, EpilogueTileType,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          OutElementType, LayoutC, AlignmentC, EpilogueSchedule,
          cutlass::epilogue::fusion::LinearCombination<
              OutElementType, float, void, float>>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          Arch, cutlass::arch::OpClassBlockScaledTensorOp,
          cute::tuple<ElementA, SFType>, LayoutA, AlignmentA,
          cute::tuple<ElementB, SFType>, LayoutB, AlignmentB,
          ElementAccumulator, CTAShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledOmmaVs16Sm100>::
          CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;

  using Gemm = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};
