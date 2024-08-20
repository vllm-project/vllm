/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

// SM90 Collective Builders should be used only starting CUDA 12.0
#if (__CUDACC_VER_MAJOR__ >= 12)
#define CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective
{

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail
{

// Returns the maximum number of smem tiles that can be used with a given smem capacity, or overrides with manual count.
template <int CapacityBytes, class ElementA, class ElementB, class TileShapeMNK, bool SwapAB, int carveout_bytes>
constexpr int compute_stage_count_or_override_gated(StageCountAutoCarveout<carveout_bytes> stage_count)
{
    // 32 bytes to account for barriers etc.
    constexpr int stage_barrier_bytes = 32;
    constexpr int a_bits = static_cast<int>(sizeof_bits<ElementA>::value);
    constexpr int b_bits = static_cast<int>(sizeof_bits<ElementB>::value);
    constexpr int stage_bytes = [&]() -> int
    {
        if constexpr (SwapAB)
        {
            return (a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{}) * 2) / 8
                + (b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) / 8 + stage_barrier_bytes;
        }
        else
        {
            return (a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) / 8
                + (b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{}) * 2) / 8 + stage_barrier_bytes;
        }
    }();

    return (CapacityBytes - carveout_bytes) / stage_bytes;
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_SS
template <class ElementA, class GmemLayoutA, int AlignmentA, class ElementB, class GmemLayoutB, int AlignmentB,
    class ElementAccumulator, class TileShape_MNK, class ClusterShape_MNK, class StageCountType,
    class KernelScheduleType, template <class /* ElementCompute */> class Activation, bool SwapAB>
struct CollectiveBuilderGated<arch::Sm90, arch::OpClassTensorOp, ElementA, GmemLayoutA, AlignmentA, ElementB,
    GmemLayoutB, AlignmentB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType, KernelScheduleType,
    Activation, SwapAB,
    cute::enable_if_t<(cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecialized>
        || cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpong>
        || cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative>
        || cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedCooperative>) &&not detail::
            is_use_rmem_A<ElementA, GmemLayoutA, ElementB, GmemLayoutB>()>>
{
    static_assert(is_static<TileShape_MNK>::value);
    static_assert(is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
    static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
    static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
        "Should meet TMA alignment requirement\n");

    static constexpr bool IsArrayOfPointersGemm
        = (cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedCooperative>);
    static constexpr bool IsFP8Input = detail::is_input_fp8<ElementA, ElementB>();
    static_assert(!IsFP8Input || (IsFP8Input && !IsArrayOfPointersGemm),
        "Kernel[Array/Group]TmaWarpSpecializedCooperative is only compatible with FP8 FastAccum version right now\n");

    // For fp32 types, map to tf32 MMA value type
    using MmaElementA = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;
    using MmaElementB = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;

    static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<MmaElementA, GmemLayoutA>();
    static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<MmaElementB, GmemLayoutB>();

    using AtomLayoutMNK = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperative>
            || IsArrayOfPointersGemm,
        Layout<Shape<_2, _1, _1>>, Layout<Shape<_1, _1, _1>>>;

    using TiledMma = decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<MmaElementA, MmaElementB,
                                                       ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(),
        AtomLayoutMNK{}));

    using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
    using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

    using SmemLayoutAtomA = decltype(detail::ss_smem_selector<GmmaMajorA, MmaElementA,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomB = decltype(detail::ss_smem_selector<GmmaMajorB, MmaElementB,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

    static constexpr int PipelineStages
        = detail::compute_stage_count_or_override_gated<detail::sm90_smem_capacity_bytes, MmaElementA, MmaElementB,
            TileShape_MNK, SwapAB>(StageCountType{});
    using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
        MainloopSm90ArrayTmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
        /* For FP8 use a separate mainloop compared to other datatypes */
        cute::conditional_t<IsFP8Input,
            MainloopSm90TmaGmmaWarpSpecializedFP8<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
            MainloopSm90TmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>>>;

    using SmemCopyAtomA = void;
    using SmemCopyAtomB = void;

    using CollectiveOp = CollectiveMmaGated<DispatchPolicy, TileShape_MNK, ElementA, TagToStrideA_t<GmemLayoutA>,
        ElementB, TagToStrideB_t<GmemLayoutB>, TiledMma, GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
        GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity, Activation, SwapAB>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// GMMA_TMA_WS_FP8_FAST_ACCUM_SS
template <class ElementA, class GmemLayoutA, int AlignmentA, class ElementB, class GmemLayoutB, int AlignmentB,
    class ElementAccumulator, class TileShape_MNK, class ClusterShape_MNK, class StageCountType,
    class KernelScheduleType, template <class /* ElementCompute */> class Activation, bool SwapAB>
struct CollectiveBuilderGated<arch::Sm90, arch::OpClassTensorOp, ElementA, GmemLayoutA, AlignmentA, ElementB,
    GmemLayoutB, AlignmentB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType, KernelScheduleType,
    Activation, SwapAB,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedFP8FastAccum>
        || cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedPingpongFP8FastAccum>
        || cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperativeFP8FastAccum>
        || cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum>>>
{
    static_assert(is_static<TileShape_MNK>::value);
    static_assert(is_static<ClusterShape_MNK>::value);
    static_assert(detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB, detail::tma_alignment_bytes>(),
        "Not meet TMA alignment requirement yet\n");
    static_assert(
        detail::is_input_fp8<ElementA, ElementB>(), "Only FP8 datatypes are compatible with these kernel schedules\n");
    // Dispatch TN fp8 kernels only to TMA warp specialized FP8 builder
    static_assert(!detail::is_use_rmem_A<ElementA, GmemLayoutA, ElementB, GmemLayoutB>(),
        "Not supported for fp8 non-TN warp specialized kernels yet\n");
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
    static_assert(cutlass::detail::dependent_false<ElementA>, "Unsupported Toolkit for SM90 Collective Builder\n");
#endif

    static constexpr cute::GMMA::Major GmmaMajorA = detail::gmma_ss_tag_to_major_A<ElementA, GmemLayoutA>();
    static constexpr cute::GMMA::Major GmmaMajorB = detail::gmma_ss_tag_to_major_B<ElementB, GmemLayoutB>();

    static constexpr bool IsArrayOfPointersGemm
        = (cute::is_same_v<KernelScheduleType, KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum>);
    using AtomLayoutMNK
        = cute::conditional_t<cute::is_same_v<KernelScheduleType, KernelTmaWarpSpecializedCooperativeFP8FastAccum>
                || IsArrayOfPointersGemm,
            Layout<Shape<_2, _1, _1>>, Layout<Shape<_1, _1, _1>>>;

    using TiledMma = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<ElementA, ElementB, ElementAccumulator, TileShape_MNK, GmmaMajorA, GmmaMajorB>(),
        AtomLayoutMNK{}));

    using GmemTiledCopyA = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape_MNK{})));
    using GmemTiledCopyB = decltype(detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape_MNK{})));

    using SmemLayoutAtomA = decltype(detail::ss_smem_selector<GmmaMajorA, ElementA,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomB = decltype(detail::ss_smem_selector<GmmaMajorB, ElementB,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

    static constexpr int PipelineStages
        = detail::compute_stage_count_or_override_gated<detail::sm90_smem_capacity_bytes, ElementA, ElementB,
            TileShape_MNK, SwapAB>(StageCountType{});
    using DispatchPolicy = cute::conditional_t<IsArrayOfPointersGemm,
        MainloopSm90ArrayTmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>,
        MainloopSm90TmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>>;

    using SmemCopyAtomA = void;
    using SmemCopyAtomB = void;

    using CollectiveOp = CollectiveMmaGated<DispatchPolicy, TileShape_MNK, ElementA, TagToStrideA_t<GmemLayoutA>,
        ElementB, TagToStrideB_t<GmemLayoutB>, TiledMma, GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
        GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity, Activation, SwapAB>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
