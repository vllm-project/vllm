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

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/tensor_predicate.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/fp8_accumulation.hpp"
#include "cutlass/trace.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective
{
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <int Stages, class ClusterShape, class KernelSchedule, class TileShape_, class ElementA_, class StrideA_,
    class ElementB_, class StrideB_, class TiledMma_, class GmemTiledCopyA_, class SmemLayoutAtomA_,
    class SmemCopyAtomA_, class TransformA_, class GmemTiledCopyB_, class SmemLayoutAtomB_, class SmemCopyAtomB_,
    class TransformB_, template <class /* ElementCompute */> class Activation_, bool SwapAB_>
struct CollectiveMmaGated<MainloopSm90TmaGmmaWarpSpecializedFP8<Stages, ClusterShape, KernelSchedule>, TileShape_,
    ElementA_, StrideA_, ElementB_, StrideB_, TiledMma_, GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_,
    GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_, Activation_, SwapAB_>
{
    static constexpr bool isGated = true;
    static constexpr bool SwapAB = SwapAB_;

    //
    // Type Aliases
    //
    using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedFP8<Stages, ClusterShape, KernelSchedule>;
    using TileShape = TileShape_;
    using ElementA = ElementA_;
    using StrideA = StrideA_;
    using ElementB = ElementB_;
    using StrideB = StrideB_;
    using TiledMma = TiledMma_;
    using ElementAccumulator = typename TiledMma::ValTypeC;
    using GmemTiledCopyA = GmemTiledCopyA_;
    using GmemTiledCopyB = GmemTiledCopyB_;
    using SmemLayoutAtomA = SmemLayoutAtomA_;
    using SmemLayoutAtomB = SmemLayoutAtomB_;
    using SmemCopyAtomA = SmemCopyAtomA_;
    using SmemCopyAtomB = SmemCopyAtomB_;
    using TransformA = TransformA_;
    using TransformB = TransformB_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using Activation = Activation_<ElementAccumulator>;

    using ElementAux = cute::conditional_t<SwapAB, ElementA_, ElementB_>;
    using ValTypeAux = cute::conditional_t<SwapAB, typename TiledMma::ValTypeA, typename TiledMma::ValTypeB>;

    using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
    using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

    using PipelineParams = typename MainloopPipeline::Params;

    static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert(
        (size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
    static_assert(
        (size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

    static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert(
        (size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
    static_assert(
        (size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

    // Tile along modes in a way that maximizes the TMA box size.
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
        conditional_t<::cutlass::gemm::detail::is_major<0, StrideA>(), Step<_2, _1, _3>, Step<_1, _2, _3>>{}));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{},
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
        conditional_t<::cutlass::gemm::detail::is_major<0, StrideB>(), Step<_2, _1, _3>, Step<_1, _2, _3>>{}));
    using SmemLayoutAux = cute::conditional_t<SwapAB, SmemLayoutA, SmemLayoutB>;

    static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
    static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value
            && cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
        "MMA atom must source both A and B operand from smem_desc for this mainloop.");
    static_assert(
        cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
        "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
    static_assert(
        cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
        "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

    struct SharedStorage
    {
        struct TensorStorage : cute::aligned_struct<128>
        {
            cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
            cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
            cute::array_aligned<ValTypeAux, cute::cosize_v<SmemLayoutAux>> smem_Aux;
        } tensors;

        using PipelineStorage = typename MainloopPipeline::SharedStorage;
        PipelineStorage pipeline;
    };

    using TensorStorage = typename SharedStorage::TensorStorage;
    using PipelineStorage = typename SharedStorage::PipelineStorage;

    // Host side kernel arguments
    struct Arguments
    {
        ElementA const* ptr_A;
        StrideA dA;
        ElementB const* ptr_B;
        StrideB dB;
        float scale_d0 = 1.0f;
        float scale_d1 = 1.0f;
        uint32_t mma_promotion_interval = 4;
    };

    // Device side kernel params
    struct Params
    {
        // Assumption: StrideA is congruent with Problem_MK
        using TMA_A = decltype(make_tma_copy(GmemTiledCopyA{},
            make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
            SmemLayoutA{}(_, _, 0), make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
            size<1>(ClusterShape{}))); // mcast along N mode for this M load, if any
        // Assumption: StrideB is congruent with Problem_NK
        using TMA_B = decltype(make_tma_copy(GmemTiledCopyB{},
            make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
            SmemLayoutB{}(_, _, 0), make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
            size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any
        using TMA_Aux = cute::conditional_t<SwapAB, TMA_A, TMA_B>;
        TMA_A tma_load_a;
        TMA_B tma_load_b;
        TMA_Aux tma_load_aux;
        float scale_d0 = 1.0f;
        float scale_d1 = 1.0f;
        uint32_t mma_promotion_interval = 4;
    };

    //
    // Methods
    //

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(
        ProblemShape const& problem_shape, Arguments const& args, void* workspace)
    {
        (void) workspace;

        // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
        auto problem_shape_MNKL = append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_MNKL;

        auto ptr_A = reinterpret_cast<ElementA const*>(args.ptr_A);
        auto ptr_B = reinterpret_cast<ElementB const*>(args.ptr_B);

        Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M, K, L), args.dA));
        Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N, K, L), args.dB));
        typename Params::TMA_A tma_load_a = make_tma_copy(GmemTiledCopyA{}, tensor_a,
            SmemLayoutA{}(_, _, cute::Int<0>{}), make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
            size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
        typename Params::TMA_B tma_load_b = make_tma_copy(GmemTiledCopyB{}, tensor_b,
            SmemLayoutB{}(_, _, cute::Int<0>{}), make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
            size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
        if constexpr (SwapAB)
        {
            auto ptr_Aux = reinterpret_cast<ElementA const*>(args.ptr_A + size(make_shape(M, K, L)));
            Tensor tensor_aux = make_tensor(ptr_Aux, make_layout(make_shape(M, K, L), args.dA));
            typename Params::TMA_Aux tma_load_aux = make_tma_copy(GmemTiledCopyA{}, tensor_aux,
                SmemLayoutA{}(_, _, cute::Int<0>{}), make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
                size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
            return {tma_load_a, tma_load_b, tma_load_aux, args.scale_d0, args.scale_d1, args.mma_promotion_interval};
        }
        else
        {
            auto ptr_Aux = reinterpret_cast<ElementB const*>(args.ptr_B + size(make_shape(N, K, L)));
            Tensor tensor_aux = make_tensor(ptr_Aux, make_layout(make_shape(N, K, L), args.dB));
            typename Params::TMA_Aux tma_load_aux = make_tma_copy(GmemTiledCopyB{}, tensor_aux,
                SmemLayoutB{}(_, _, cute::Int<0>{}), make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
                size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
            return {tma_load_a, tma_load_b, tma_load_aux, args.scale_d0, args.scale_d1, args.mma_promotion_interval};
        }
    }

    template <class ProblemShape>
    CUTLASS_HOST_DEVICE static bool can_implement(
        ProblemShape const& problem_shape, [[maybe_unused]] Arguments const& args)
    {
        constexpr int tma_alignment_bits = 128;
        auto problem_shape_MNKL = append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_MNKL;

        bool implementable = true;
        constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
        implementable = implementable
            && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M, K, L), StrideA{});
        constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
        implementable = implementable
            && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N, K, L), StrideB{});
        /* MMA promotion interval should be a multiple of 4, since each mainloop iteration would issue 4 MMA
         * instructions. */
        implementable = implementable && (args.mma_promotion_interval % 4 == 0);

        if (!implementable)
        {
            CUTLASS_TRACE_HOST(
                "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
        }
        return implementable;
    }

    static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
    static constexpr int K_PIPE_MMAS = 1;
    static constexpr uint32_t TmaTransactionBytes
        = (size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value)) / 8
        + (size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value)) / 8
        + (size<0>(SmemLayoutAux{}) * size<1>(SmemLayoutAux{}) * static_cast<uint32_t>(sizeof_bits<ElementAux>::value))
            / 8;

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params)
    {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_aux.get_tma_descriptor());
    }

    /// Set up the data needed by this collective for load and mma.
    /// Returns a tuple of tensors. The collective and the kernel layer have the contract
    /// Returned tuple must contain at least two elements, with the first two elements being:
    /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
    /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
    /// gAux_xkl - The tma tensor, A/B after a local tile so it has shape  (BLK_N,BLK_K,m/n,k,l)
    template <class ProblemShape_MNKL>
    CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const
    {
        using X = Underscore;
        // Separate out problem shape for convenience
        auto [M, N, K, L] = problem_shape_MNKL;

        // TMA requires special handling of strides to deal with coord codomain mapping
        // Represent the full tensors -- get these from TMA
        Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M, K, L)); // (m,k,l)
        Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N, K, L)); // (n,k,l)

        // Make tiled views, defer the slice
        Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{}); // (BLK_M,BLK_K,m,k,l)
        Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{}); // (BLK_N,BLK_K,n,k,l)

        if constexpr (SwapAB)
        {
            Tensor mAux_xkl = mainloop_params.tma_load_aux.get_tma_tensor(make_shape(M, K, L)); // (m,k,l)
            Tensor gAux_xkl
                = local_tile(mAux_xkl, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});    // (BLK_M,BLK_K,m,k,l)
            return cute::make_tuple(gA_mkl, gB_nkl, gAux_xkl);
        }
        else
        {
            Tensor mAux_xkl = mainloop_params.tma_load_aux.get_tma_tensor(make_shape(N, K, L)); // (n,k,l)
            Tensor gAux_xkl
                = local_tile(mAux_xkl, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});    // (BLK_N,BLK_K,n,k,l)
            return cute::make_tuple(gA_mkl, gB_nkl, gAux_xkl);
        }
    }

    /// Perform a collective-scoped matrix multiply-accumulate
    /// Producer Perspective
    template <class TensorA, class TensorB, class TensorAux, class KTileIterator, class BlockCoord>
    CUTLASS_DEVICE void load(Params const& mainloop_params, MainloopPipeline pipeline, PipelineState smem_pipe_write,
        cute::tuple<TensorA, TensorB, TensorAux> const& load_inputs, BlockCoord const& blk_coord,
        KTileIterator k_tile_iter, int k_tile_count, int thread_idx, uint32_t block_rank_in_cluster,
        TensorStorage& shared_tensors)
    {
        int lane_predicate = cute::elect_one_sync();

        if (lane_predicate)
        {
            Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
            Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
            Tensor sAux = make_tensor(make_smem_ptr(shared_tensors.smem_Aux.data()), SmemLayoutAux{});

            //
            // Prepare the TMA loads for A and B
            //

            constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
            uint2 cluster_local_block_id
                = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

            Tensor gA_mkl = get<0>(load_inputs);
            Tensor gB_nkl = get<1>(load_inputs);
            Tensor gAux_xkl = get<2>(load_inputs);

            auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
            auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);
            auto block_tma_aux = SwapAB ? mainloop_params.tma_load_aux.get_slice(cluster_local_block_id.y)
                                        : mainloop_params.tma_load_aux.get_slice(cluster_local_block_id.x);

            // Partition the inputs based on the current block coordinates.
            auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
            Tensor gA = gA_mkl(_, _, m_coord, _, l_coord); // (BLK_M,BLK_K,k)
            Tensor gB = gB_nkl(_, _, n_coord, _, l_coord); // (BLK_N,BLK_K,k)
            Tensor gAux = SwapAB ? gAux_xkl(_, _, m_coord, _, l_coord) : gAux_xkl(_, _, n_coord, _, l_coord);

            // Applies the mapping from block_tma_a
            Tensor tAgA = block_tma_a.partition_S(gA); // (TMA,TMA_M,TMA_K,k)
            Tensor tAsA = block_tma_a.partition_D(sA); // (TMA,TMA_M,TMA_K,PIPE)

            Tensor tBgB = block_tma_b.partition_S(gB); // (TMA,TMA_N,TMA_K,k)
            Tensor tBsB = block_tma_b.partition_D(sB); // (TMA,TMA_N,TMA_K,PIPE)

            Tensor tAuxgAux = block_tma_aux.partition_S(gAux);
            Tensor tAuxsAux = block_tma_aux.partition_D(sAux);

            uint16_t mcast_mask_a = 0;
            uint16_t mcast_mask_b = 0;
            uint16_t mcast_mask_aux = 0;

            // Issue TmaLoads
            // Maps the tile -> block, value
            if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>)
            {
                auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
                for (int n = 0; n < size<1>(block_layout); ++n)
                {
                    mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, Int<0>{}));
                }
            }

            if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>)
            {
                auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
                for (int m = 0; m < size<0>(block_layout); ++m)
                {
                    mcast_mask_b |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, Int<0>{}));
                }
            }

            if constexpr (SwapAB)
            {
                mcast_mask_aux = mcast_mask_a;
            }
            else
            {
                mcast_mask_aux = mcast_mask_b;
            }

            // Mainloop
            CUTLASS_PRAGMA_NO_UNROLL
            for (; k_tile_count > 0; --k_tile_count)
            {
                // LOCK smem_pipe_write for _writing_
                pipeline.producer_acquire(smem_pipe_write);

                //
                // Copy gmem to smem for *k_tile_iter
                //

                using BarrierType = typename MainloopPipeline::ProducerBarrierType;
                BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

                int write_stage = smem_pipe_write.index();
                copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_, _, _, *k_tile_iter),
                    tAsA(_, _, _, write_stage));
                copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_, _, _, *k_tile_iter),
                    tBsB(_, _, _, write_stage));
                copy(mainloop_params.tma_load_aux.with(*tma_barrier, mcast_mask_aux), tAuxgAux(_, _, _, *k_tile_iter),
                    tAuxsAux(_, _, _, write_stage));
                ++k_tile_iter;

                // Advance smem_pipe_write
                ++smem_pipe_write;
            }
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write)
    {
        int lane_predicate = cute::elect_one_sync();

        // Issue the epilogue waits
        if (lane_predicate)
        {
            /* This helps avoid early exit of blocks in Cluster
             * Waits for all stages to either be released (all
             * Consumer UNLOCKs), or if the stage was never used
             * then would just be acquired since the phase was
             * still inverted from make_producer_start_state
             */
            pipeline.producer_tail(smem_pipe_write);
        }
    }

    /// Perform a collective-scoped matrix multiply-accumulate
    /// Consumer Perspective
    template <class FrgTensorC>
    CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read, FrgTensorC& accum0,
        FrgTensorC& accum1, int k_tile_count, int thread_idx, TensorStorage& shared_tensors,
        Params const& mainloop_params)
    {

        static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
        static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
        static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
        static_assert(cute::is_void_v<SmemCopyAtomA>,
            "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
        static_assert(cute::is_void_v<SmemCopyAtomB>,
            "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

        Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
        Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
        Tensor sAux = make_tensor(make_smem_ptr(shared_tensors.smem_Aux.data()), SmemLayoutAux{});

        //
        // Define C accumulators and A/B partitioning
        //

        TiledMma tiled_mma;
        auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

        Tensor tCsA = thread_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
        Tensor tCsB = thread_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

        // Allocate "fragments/descriptors"
        Tensor tCrA = thread_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
        Tensor tCrB = thread_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)

        auto tCsAux = [&]() -> auto
        {
            if constexpr (SwapAB)
            {
                return thread_mma.partition_A(sAux);
            }
            else
            {
                return thread_mma.partition_B(sAux);
            }
        }();
        auto tCrAux = [&]() -> auto
        {
            if constexpr (SwapAB)
            {
                return thread_mma.make_fragment_A(tCsAux);
            }
            else
            {
                return thread_mma.make_fragment_B(tCsAux);
            }
        }();

        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum0)); // M
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum0)); // N
        CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));   // K
        CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));   // PIPE
        if constexpr (SwapAB)
        {
            CUTE_STATIC_ASSERT_V(size<1>(tCsAux) == size<1>(accum1)); // M
            CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum1));   // N
            CUTE_STATIC_ASSERT_V(size<2>(tCsB) == size<2>(tCsAux));   // K
            CUTE_STATIC_ASSERT_V(size<3>(tCsB) == size<3>(tCsAux));   // PIPE
        }
        else
        {
            CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum1));           // M
            CUTE_STATIC_ASSERT_V(size<1>(tCsAux) == size<2>(accum1));         // N
            CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsAux));           // K
            CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsAux));           // PIPE
        }
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));   // PIPE
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));   // PIPE
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sAux)); // PIPE

        //
        // PIPELINED MAIN LOOP
        //
        static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS < K_PIPE_MAX), "ERROR : Incorrect number of MMAs in flight");

        // We release buffers to producer warps(dma load) with some mmas in flight
        PipelineState smem_pipe_release = smem_pipe_read;

        // Prologue GMMAs
        int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);

        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

        GmmaFP8Accumulation accumulation0(accum0, mainloop_params.mma_promotion_interval, size<2>(tCrA));
        GmmaFP8Accumulation accumulation1(accum1, mainloop_params.mma_promotion_interval, size<2>(tCrA));
        warpgroup_fence_operand(accumulation0());
        warpgroup_fence_operand(accumulation1());
        CUTLASS_PRAGMA_UNROLL
        for (int k_tile_prologue = prologue_mma_count; k_tile_prologue > 0; --k_tile_prologue)
        {
            // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);

            if (accumulation0.prepare_if_needed())
            {
                tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
            }

            int read_stage = smem_pipe_read.index();
            warpgroup_arrive();
            // Unroll the K mode manually to set scale D to 1
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tCrA); ++k_block)
            {
                // (V,M,K) x (V,N,K) => (V,M,N)
                cute::gemm(
                    tiled_mma, tCrA(_, _, k_block, read_stage), tCrB(_, _, k_block, read_stage), accumulation0());
                if constexpr (SwapAB)
                {
                    cute::gemm(
                        tiled_mma, tCrAux(_, _, k_block, read_stage), tCrB(_, _, k_block, read_stage), accumulation1());
                }
                else
                {
                    cute::gemm(
                        tiled_mma, tCrA(_, _, k_block, read_stage), tCrAux(_, _, k_block, read_stage), accumulation1());
                }
                tiled_mma.accumulate_ = GMMA::ScaleOut::One;
            }
            warpgroup_commit_batch();

            accumulation0.promote_if_needed();
            accumulation1.promote_if_needed();

            ++smem_pipe_read;
        }

        warpgroup_fence_operand(accumulation0());
        warpgroup_fence_operand(accumulation1());
        // Mainloop GMMAs
        k_tile_count -= prologue_mma_count;

        CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > 0; --k_tile_count)
        {
            // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);

            //
            // Compute on k_tile
            //

            int read_stage = smem_pipe_read.index();

            if (accumulation0.prepare_if_needed())
            {
                tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
            }

            warpgroup_fence_operand(accumulation0());
            warpgroup_fence_operand(accumulation1());
            warpgroup_arrive();
            // Unroll the K mode manually to set scale D to 1
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tCrA); ++k_block)
            {
                // (V,M,K) x (V,N,K) => (V,M,N)
                cute::gemm(
                    tiled_mma, tCrA(_, _, k_block, read_stage), tCrB(_, _, k_block, read_stage), accumulation0());
                if constexpr (SwapAB)
                {
                    cute::gemm(
                        tiled_mma, tCrAux(_, _, k_block, read_stage), tCrB(_, _, k_block, read_stage), accumulation1());
                }
                else
                {
                    cute::gemm(
                        tiled_mma, tCrA(_, _, k_block, read_stage), tCrAux(_, _, k_block, read_stage), accumulation1());
                }
                tiled_mma.accumulate_ = GMMA::ScaleOut::One;
            }
            warpgroup_commit_batch();

            /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
            warpgroup_wait<K_PIPE_MMAS>();
            warpgroup_fence_operand(accumulation0());
            warpgroup_fence_operand(accumulation1());

            accumulation0.promote_if_needed();
            accumulation1.promote_if_needed();

            pipeline.consumer_release(smem_pipe_release); // UNLOCK smem_pipe_release, done _computing_ on it

            // Advance smem_pipe_read and smem_pipe_release
            ++smem_pipe_read;
            ++smem_pipe_release;
        }

        accumulation0.promote_residue_if_needed();
        accumulation1.promote_residue_if_needed();

        warpgroup_fence_operand(accumulation0());
        warpgroup_fence_operand(accumulation1());
    }

    /// Perform a Consumer Epilogue to release all buffers
    CUTLASS_DEVICE void mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count)
    {
        // Prologue GMMAs
        int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
        k_tile_count -= prologue_mma_count;

        smem_pipe_release.advance(k_tile_count);

        // Wait on all GMMAs to complete
        warpgroup_wait<0>();

        for (int count = 0; count < prologue_mma_count; ++count)
        {
            pipeline.consumer_release(smem_pipe_release); // UNLOCK smem_pipe_release, done _computing_ on it
            ++smem_pipe_release;
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
