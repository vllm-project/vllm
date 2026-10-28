#pragma once

#include "cutlass_extensions/vllm_collective_builder.cuh"
#include "machete_mainloop.cuh"

namespace cutlass::gemm::collective {
using namespace cute;

struct MacheteKernelTag {};

template <class ElementPairA_, class GmemLayoutA_, int AlignmentA,
          class ElementPairB_, class GmemLayoutB_, int AlignmentB,
          class ElementAccumulator, class TileShape_MNK, class ClusterShape_MNK,
          class StageCountType, class KernelScheduleType>
struct VLLMCollectiveBuilder<
    MacheteKernelTag, arch::Sm90, arch::OpClassTensorOp, ElementPairA_,
    GmemLayoutA_, AlignmentA, ElementPairB_, GmemLayoutB_, AlignmentB,
    ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType,
    KernelScheduleType,
    cute::enable_if_t<(
        cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecialized> ||
        cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpong> ||
        cute::is_same_v<KernelScheduleType,
                        KernelTmaWarpSpecializedCooperative>)>> {
  using CollectiveOp = machete::MacheteCollectiveMma<
      ElementPairA_, GmemLayoutA_, AlignmentA, ElementPairB_, GmemLayoutB_,
      AlignmentB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK,
      StageCountType, KernelScheduleType>;
};

};  // namespace cutlass::gemm::collective
