// Modified from: cutlass/gemm/collective/builders/sm90_gmma_builder.inl
// clang-format off
#pragma once

#include "cutlass/gemm/collective/builders/sm90_gmma_builder.inl"

#include "cutlass_extensions/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_SS (BlockScaled Builders)
template <
  class ElementA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  int ScaleGranularityM
>
struct CollectiveBuilder<
    arch::Sm90,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutATag,
    AlignmentA,
    ElementB,
    GmemLayoutBTag,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum<ScaleGranularityM>,
    cute::enable_if_t<
      not detail::is_use_rmem_A<ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()>
> {
  using KernelScheduleType = KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum<ScaleGranularityM>;

  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
  static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
  static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
                "Should meet TMA alignment requirement\n");

  static constexpr bool IsArrayOfPointersGemm = (cute::is_any_of_v<KernelScheduleType,
                                                                   KernelPtrArrayTmaWarpSpecializedCooperative,
                                                                   KernelPtrArrayTmaWarpSpecializedPingpong>);
  static constexpr bool IsFP8Input = detail::is_input_fp8<ElementA, ElementB>();
  static_assert((!IsFP8Input || !IsArrayOfPointersGemm),
                "KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum is only compatible with FP8 Blocked Scaled version right now.");

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

  static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementAMma, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementBMma, GmemLayoutBTag>();

  static constexpr bool IsCooperative = cute::is_any_of_v<KernelScheduleType,
                                                          KernelTmaWarpSpecializedCooperative,
                                                          KernelPtrArrayTmaWarpSpecializedCooperative,
                                                          KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum<ScaleGranularityM>>;
  using AtomLayoutMNK = cute::conditional_t<IsCooperative,
      Layout<Shape<_2,_1,_1>>, Layout<Shape<_1,_1,_1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<
      ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(), AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA = decltype(detail::ss_smem_selector<
      GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB = decltype(detail::ss_smem_selector<
      GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  static constexpr size_t TensorMapStorage = IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * 2 /* for A and B */ : 0;
  static constexpr int KernelSmemCarveout = static_cast<int>(TensorMapStorage);

  static constexpr int PipelineStages = detail::compute_stage_count_or_override<detail::sm90_smem_capacity_bytes - KernelSmemCarveout,
      ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedBlockScalingSubGroupMFP8<PipelineStages, ClusterShape_MNK, KernelScheduleType, ScaleGranularityM>;

  using SmemCopyAtomA = void;
  using SmemCopyAtomB = void;

  using CollectiveOp = CollectiveMma<
      DispatchPolicy,
      TileShape_MNK,
      ElementA,
      TagToStrideA_t<GmemLayoutATag>,
      ElementB,
      TagToStrideB_t<GmemLayoutBTag>,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomB,
      cute::identity
    >;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
