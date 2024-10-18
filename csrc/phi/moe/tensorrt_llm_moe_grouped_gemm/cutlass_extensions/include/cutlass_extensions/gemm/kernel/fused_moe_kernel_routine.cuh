/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cutlass_extensions/gemm/kernel/fused_moe_kernel_traits.cuh>

namespace fused_moe
{

template <typename ElementInput_, typename ElementWeight_, typename ElementOutput_, int TileM_, int TileN_, int TileK_,
    int Stages_, Activation_Type activation_type_, typename Enable = void>
struct Fused_Moe_Kernel_routine_sm80;

template <typename ElementInput_, typename ElementWeight_, typename ElementOutput_, int TileM_, int TileN_, int TileK_,
    int Stages_, Activation_Type activation_type_>
struct Fused_Moe_Kernel_routine_sm80<ElementInput_, ElementWeight_, ElementOutput_, TileM_, TileN_, TileK_, Stages_,
    activation_type_, std::enable_if_t<isGateActivation(activation_type_)>>
{
    using KT = Fused_Moe_Kernel_traits_sm80<ElementInput_, ElementWeight_, ElementOutput_, TileM_, TileN_, TileK_,
        Stages_, activation_type_>;
    using Params = Routine_Params<ElementInput_, ElementWeight_, ElementOutput_>;

    CUTE_DEVICE auto gmem_tensor_init(int const problem_index, int const gemm_m, Params const& params)
    {
        using X = cute::Underscore;

        int const M = gemm_m;
        int const N1 = params.gemm_n;
        int const K1 = params.gemm_k;

        int const row_jump = ((problem_index == 0) ? 0 : params.total_rows_before_expert[problem_index - 1]);
        typename KT::ElementInput const* ptr_input_ = params.ptr_input + row_jump * K1;
        typename KT::ElementWeight const* ptr_fc1_gate_
            = params.ptr_fc1 + (2 * problem_index + 1) * N1 * K1; // TODO: we only focus on gated activation..
        typename KT::ElementWeight const* ptr_fc1_
            = params.ptr_fc1 + 2 * problem_index * N1 * K1;       // TODO: we only focus on gated activation..
        typename KT::ElementInput const* ptr_bias_
            = (params.ptr_bias == nullptr) ? nullptr : params.ptr_bias + 2 * problem_index * N1;
        typename KT::ElementInput const* ptr_bias_gate_
            = (params.ptr_bias == nullptr) ? nullptr : params.ptr_bias + (2 * problem_index + 1) * N1;
        typename KT::ElementOutput* ptr_output_ = params.ptr_output + row_jump * N1;

        cute::Tensor mInput_mk
            = cute::make_tensor(cute::make_gmem_ptr(static_cast<typename KT::ElementInput const*>(ptr_input_)),
                cute::make_shape(M, K1), cute::make_stride(K1, cute::_1{}));

        cute::Tensor mfc1_gate_nk
            = cute::make_tensor(cute::make_gmem_ptr(static_cast<typename KT::ElementWeight const*>(ptr_fc1_gate_)),
                cute::make_shape(N1, K1), cute::make_stride(K1, cute::_1{}));

        cute::Tensor mfc1_nk
            = cute::make_tensor(cute::make_gmem_ptr(static_cast<typename KT::ElementWeight const*>(ptr_fc1_)),
                cute::make_shape(N1, K1), cute::make_stride(K1, cute::_1{}));

        cute::Tensor mBias_mn = cute::make_tensor(
            cute::make_gmem_ptr(static_cast<typename KT::ElementInput const*>(ptr_bias_)), cute::make_shape(M, N1),
            cute::make_stride(cute::Int<0>{}, cute::_1{})); // trick: bias shape is [1, N], but we use [M, N].

        cute::Tensor mBias_gate_mn = cute::make_tensor(
            cute::make_gmem_ptr(static_cast<typename KT::ElementInput const*>(ptr_bias_gate_)), cute::make_shape(M, N1),
            cute::make_stride(cute::Int<0>{}, cute::_1{})); // trick: bias shape is [1, N], but we use [M, N].

        cute::Tensor mOutput_mn
            = cute::make_tensor(cute::make_gmem_ptr(static_cast<typename KT::ElementInput*>(ptr_output_)),
                cute::make_shape(M, N1), cute::make_stride(N1, cute::_1{}));

        cute::Tensor gInput_mk = cute::local_tile(mInput_mk, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<cute::_1, X, cute::_1>{}); // (BLK_M, BLK_K, m, k)
        cute::Tensor gfc1_gate_nk = cute::local_tile(mfc1_gate_nk, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<X, cute::_1, cute::_1>{}); // (BLK_N, BLK_K, n, k)
        cute::Tensor gfc1_nk = cute::local_tile(mfc1_nk, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<X, cute::_1, cute::_1>{}); // (BLK_N, BLK_K, n, k)

        cute::Tensor gBias_mn = cute::local_tile(mBias_mn, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<cute::_1, cute::_1, X>{}); // (BLK_M, BLK_N, m, n)

        cute::Tensor gBias_gate_mn = cute::local_tile(mBias_gate_mn, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<cute::_1, cute::_1, X>{}); // (BLK_M, BLK_N, m, n)

        cute::Tensor gOutput_mn = cute::local_tile(mOutput_mn, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<cute::_1, cute::_1, X>{}); // (BLK_M, BLK_N, m, n)

        return cute::make_tuple(gInput_mk, gfc1_gate_nk, gfc1_nk, gBias_mn, gBias_gate_mn, gOutput_mn);
    }

    // be careful, m_idx will change when use another tile shape..
    CUTE_DEVICE void run_routine(
        Params const& params, int const problem_index, int const block_m_idx, int const block_n_idx, int const gemm_m)
    {
        extern __shared__ char smem_[];
        typename KT::SharedStorage& shared_storage = *reinterpret_cast<typename KT::SharedStorage*>(smem_);
        int const thread_idx = threadIdx.x;
        // gmem tensor partition ..
        auto [gInput_mk, gfc1_gate_nk, gfc1_nk, gBias_mn, gBias_gate_mn, gOutput_mn]
            = gmem_tensor_init(problem_index, gemm_m, params);
        int const residue_m = gemm_m - block_m_idx * cute::size<0>(gInput_mk);
        auto const n_tile_count = cute::size<2>(gfc1_gate_nk);

        // smem tensor ..
        cute::Tensor sInput = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_input.data()), typename KT::SmemLayoutA{}); // (BLK_M, BLK_K, Stage)
        cute::Tensor sfc1_weight = cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_fc1_weight.data()),
            typename KT::SmemLayoutB{});                                                        // (BLK_N, BLK_K, Stage)
        cute::Tensor sfc1_gate_weight
            = cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_fc1_gate_weight.data()),
                typename KT::SmemLayoutB{});                                                // (BLK_N, BLK_K, Stage)
        cute::Tensor sO = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_o.data()), typename KT::SmemLayoutO{}); // (BLK_M, BLK_N)

        // (1) first step, get the fc1_res and fc1_gate

        // (1.1) get partition for gmem -> smem
        cute::Tensor gInput = gInput_mk(cute::_, cute::_, block_m_idx, cute::_);   // (BLK_M, BLK_K, k)
        cute::Tensor gfc1 = gfc1_nk(cute::_, cute::_, block_n_idx, cute::_);       // (BLK_N, BLK_K, k)
        cute::Tensor gfc1g = gfc1_gate_nk(cute::_, cute::_, block_n_idx, cute::_); // (BLK_N, BLK_K, k)

        typename KT::GmemTiledCopyA gmem_tiled_copy_A;
        typename KT::GmemTiledCopyB gmem_tiled_copy_B;
        auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
        auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

        cute::Tensor tInputgInput = gmem_thr_copy_A.partition_S(gInput);         // (ACPY,ACPY_M,ACPY_K,k)
        cute::Tensor tInputsInput = gmem_thr_copy_A.partition_D(sInput);         // (ACPY,ACPY_M,ACPY_K,Stage)
        cute::Tensor tfc1gfc1 = gmem_thr_copy_B.partition_S(gfc1);               // (BCPY,BCPY_N,BCPY_K,k)
        cute::Tensor tfc1sfc1 = gmem_thr_copy_B.partition_D(sfc1_weight);        // (BCPY,BCPY_N,BCPY_K,Stage)
        cute::Tensor tfc1ggfc1g = gmem_thr_copy_B.partition_S(gfc1g);            // (BCPY,BCPY_N,BCPY_K,k)
        cute::Tensor tfc1gsfc1g = gmem_thr_copy_B.partition_D(sfc1_gate_weight); // (BCPY,BCPY_N,BCPY_K,Stage)

        // Allocate predicate tensors for input and fc weight (actually we only need input predicate tensor)
        cute::Tensor tInputpInput
            = cute::make_tensor<bool>(cute::make_shape(cute::size<1>(tInputsInput), cute::size<2>(tInputsInput)),
                cute::Stride<cute::_1, cute::_0>{});
        // Construct identity layout for sInput
        cute::Tensor cInput = make_identity_tensor(
            make_shape(cute::size<0>(sInput), cute::size<1>(sInput))); // (BLK_M,BLK_K) -> (blk_m,blk_k)

        // Repeat the partitioning with identity layouts
        cute::Tensor tInputcInput = gmem_thr_copy_A.partition_S(cInput); // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

        // Set predicates for m bounds
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < cute::size<0>(tInputpInput); ++m)
        {
            tInputpInput(m, 0) = cute::get<0>(tInputcInput(0, m, 0)) < residue_m; // blk_m coord < residue_m
        }

        // (1.2) prefetch gmem -> smem
        cute::clear(tInputsInput);                                           // we don't need to clear tfc1sfc1..
        auto k_tile_iter = cute::make_coord_iterator(cute::size<2>(gInput)); // emm, iter start from 0
        int k_tile_count = cute::size<2>(gInput);
        CUTLASS_PRAGMA_UNROLL
        for (int k_pipe = 0; k_pipe < KT::Stages - 1; ++k_pipe)
        {
            if (k_tile_count <= 0)
            {
                cute::clear(tInputpInput);
            }
            // cute::copy(gmem_tiled_copy_A, tInputgInput(cute::_, cute::_, cute::_, *k_tile_iter),
            //    tInputsInput(cute::_, cute::_, cute::_, k_pipe));
            // use copy_if
            cute::copy_if(gmem_tiled_copy_A, tInputpInput, tInputgInput(cute::_, cute::_, cute::_, *k_tile_iter),
                tInputsInput(cute::_, cute::_, cute::_, k_pipe));
            cute::copy(gmem_tiled_copy_B, tfc1gfc1(cute::_, cute::_, cute::_, *k_tile_iter),
                tfc1sfc1(cute::_, cute::_, cute::_, k_pipe));
            cute::copy(gmem_tiled_copy_B, tfc1ggfc1g(cute::_, cute::_, cute::_, *k_tile_iter),
                tfc1gsfc1g(cute::_, cute::_, cute::_, k_pipe));
            cute::cp_async_fence();
            k_tile_count--;
            if (k_tile_count > 0)
            {
                ++k_tile_iter;
            }
        }

        // (1.3) get partition for rf
        typename KT::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        cute::Tensor tOrInput = thr_mma.partition_fragment_A(sInput(cute::_, cute::_, 0));          // (MMA,MMA_M,MMA_K)
        cute::Tensor tOrfc1 = thr_mma.partition_fragment_B(sfc1_weight(cute::_, cute::_, 0));       // (MMA,MMA_N,MMA_K)
        cute::Tensor tOrfc1g = thr_mma.partition_fragment_B(sfc1_gate_weight(cute::_, cute::_, 0)); // (MMA,MMA_N,MMA_K)

        cute::Tensor accum
            = cute::partition_fragment_C(tiled_mma, cute::take<0, 2>(typename KT::TileShape{})); // (MMA,MMA_M,MMA_N)
        cute::Tensor accum_gate
            = cute::partition_fragment_C(tiled_mma, cute::take<0, 2>(typename KT::TileShape{})); // (MMA,MMA_M,MMA_N)
        cute::clear(accum);
        cute::clear(accum_gate);
        // checkout the shape
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOrInput) == cute::size<1>(accum));      // MMA_M
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOrInput) == cute::size<1>(accum_gate)); // MMA_M
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOrfc1) == cute::size<2>(accum));        // MMA_N
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOrfc1) == cute::size<2>(accum_gate));   // MMA_N
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOrfc1g) == cute::size<2>(accum));       // MMA_N
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOrfc1g) == cute::size<2>(accum_gate));  // MMA_N
        CUTE_STATIC_ASSERT_V(cute::size<2>(tOrInput) == cute::size<2>(tOrfc1));     // MMA_K
        CUTE_STATIC_ASSERT_V(cute::size<2>(tOrInput) == cute::size<2>(tOrfc1g));    // MMA_K
        CUTE_STATIC_ASSERT_V(cute::size(gmem_tiled_copy_A) == cute::size(tiled_mma));
        CUTE_STATIC_ASSERT_V(cute::size(gmem_tiled_copy_B) == cute::size(tiled_mma));

        // (1.4)retiling the smem and rf for copy..
        auto smem_tiled_copy_A = cute::make_tiled_copy_A(typename KT::SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
        cute::Tensor tOsInput = smem_thr_copy_A.partition_S(sInput);                        // (CPY,CPY_M,CPY_K,Stage)
        cute::Tensor tOrInput_copy_view = smem_thr_copy_A.retile_D(tOrInput);               // (CPY,CPY_M,CPY_K)
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOsInput) == cute::size<1>(tOrInput_copy_view)); // CPY_M
        CUTE_STATIC_ASSERT_V(cute::size<2>(tOsInput) == cute::size<2>(tOrInput_copy_view)); // CPY_K

        auto smem_tiled_copy_B = cute::make_tiled_copy_B(typename KT::SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
        cute::Tensor tOsfc1 = smem_thr_copy_B.partition_S(sfc1_weight);                   // (CPY,CPY_N,CPY_K,Stage)
        cute::Tensor tOrfc1_copy_view = smem_thr_copy_B.retile_D(tOrfc1);                 // (CPY,CPY_N,CPY_K)
        cute::Tensor tOsfc1g = smem_thr_copy_B.partition_S(sfc1_gate_weight);             // (CPY,CPY_N,CPY_K,Stage)
        cute::Tensor tOrfc1g_copy_view = smem_thr_copy_B.retile_D(tOrfc1g);               // (CPY,CPY_N,CPY_K)
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOsfc1) == cute::size<1>(tOrfc1_copy_view));   // CPY_N
        CUTE_STATIC_ASSERT_V(cute::size<2>(tOsfc1) == cute::size<2>(tOrfc1_copy_view));   // CPY_K
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOsfc1g) == cute::size<1>(tOrfc1g_copy_view)); // CPY_N
        CUTE_STATIC_ASSERT_V(cute::size<2>(tOsfc1g) == cute::size<2>(tOrfc1g_copy_view)); // CPY_K

        // (1.5) mainloop
        // Current pipe index in smem to read from
        int smem_pipe_read = 0;
        // Current pipe index in smem to write to
        int smem_pipe_write = KT::Stages - 1;

        cute::Tensor tOsInput_p = tOsInput(cute::_, cute::_, cute::_, smem_pipe_read);
        cute::Tensor tOsfc1_p = tOsfc1(cute::_, cute::_, cute::_, smem_pipe_read);
        cute::Tensor tOsfc1g_p = tOsfc1g(cute::_, cute::_, cute::_, smem_pipe_read);

        constexpr int K_BLOCK_MAX = cute::size<2>(tOrInput);
        // prefetch register pipeline
        if constexpr (K_BLOCK_MAX > 1)
        {
            cute::cp_async_wait<KT::Stages - 2>();
            __syncthreads();

            // Prefetch the first rmem from the first k-tile
            cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, cute::Int<0>{}),
                tOrInput_copy_view(cute::_, cute::_, cute::Int<0>{}));
            cute::copy(smem_tiled_copy_B, tOsfc1_p(cute::_, cute::_, cute::Int<0>{}),
                tOrfc1_copy_view(cute::_, cute::_, cute::Int<0>{}));
            cute::copy(smem_tiled_copy_B, tOsfc1g_p(cute::_, cute::_, cute::Int<0>{}),
                tOrfc1g_copy_view(cute::_, cute::_, cute::Int<0>{}));
        }
        // k loop for mainloop (k - (stage - 1) -> -(stage - 1), if k_tile_count > 0, it means we still need to
        // fetch gmem to smem)
        CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > -(KT::Stages - 1); --k_tile_count)
        {
            cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{},
                [&](auto k_block)
                {
                    if (k_block == K_BLOCK_MAX - 1)
                    {
                        tOsInput_p = tOsInput(cute::_, cute::_, cute::_, smem_pipe_read);
                        tOsfc1_p = tOsfc1(cute::_, cute::_, cute::_, smem_pipe_read);
                        tOsfc1g_p = tOsfc1g(cute::_, cute::_, cute::_, smem_pipe_read);
                        cute::cp_async_wait<KT::Stages - 2>();
                        __syncthreads();
                    }
                    // Load A, B shmem->regs for k_block+1
                    auto k_block_next = (k_block + cute::_1{}) % K_BLOCK_MAX;
                    cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_block_next),
                        tOrInput_copy_view(cute::_, cute::_, k_block_next));
                    cute::copy(smem_tiled_copy_B, tOsfc1_p(cute::_, cute::_, k_block_next),
                        tOrfc1_copy_view(cute::_, cute::_, k_block_next));
                    cute::copy(smem_tiled_copy_B, tOsfc1g_p(cute::_, cute::_, k_block_next),
                        tOrfc1g_copy_view(cute::_, cute::_, k_block_next));
                    // Copy gmem to smem before computing gemm on each k-pipe
                    if (k_block == 0)
                    {
                        if (k_tile_count <= 0)
                        {
                            cute::clear(tInputpInput);
                        }
                        // cute::copy(gmem_tiled_copy_A, tInputgInput(cute::_, cute::_, cute::_, *k_tile_iter),
                        //    tInputsInput(cute::_, cute::_, cute::_, smem_pipe_write));
                        cute::copy_if(gmem_tiled_copy_A, tInputpInput,
                            tInputgInput(cute::_, cute::_, cute::_, *k_tile_iter),
                            tInputsInput(cute::_, cute::_, cute::_, smem_pipe_write));
                        cute::copy(gmem_tiled_copy_B, tfc1gfc1(cute::_, cute::_, cute::_, *k_tile_iter),
                            tfc1sfc1(cute::_, cute::_, cute::_, smem_pipe_write));
                        cute::copy(gmem_tiled_copy_B, tfc1ggfc1g(cute::_, cute::_, cute::_, *k_tile_iter),
                            tfc1gsfc1g(cute::_, cute::_, cute::_, smem_pipe_write));
                        cute::cp_async_fence();
                        if (k_tile_count - 1 > 0)
                        {
                            ++k_tile_iter;
                        }

                        // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
                        smem_pipe_write = smem_pipe_read;
                        ++smem_pipe_read;
                        smem_pipe_read = (smem_pipe_read == KT::Stages) ? 0 : smem_pipe_read;
                    }
                    // Thread-level register gemm for k_block
                    cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_block), tOrfc1(cute::_, cute::_, k_block),
                        accum);
                    cute::gemm(tiled_mma, accum_gate, tOrInput(cute::_, cute::_, k_block),
                        tOrfc1g(cute::_, cute::_, k_block), accum_gate);
                });
        }
        // if (cute::thread0()) {
        //     cute::print(accum_gate(0, 0, 0));
        //     printf("\n");
        // }
        // (2) add bias if it has..
        if (params.ptr_bias != nullptr)
        {
            cute::Tensor gBias = gBias_mn(cute::_, cute::_, 0, block_n_idx); // bias only have one row..
            cute::Tensor gBias_gate = gBias_gate_mn(cute::_, cute::_, 0, block_n_idx);
            cute::Tensor tOgBias = thr_mma.partition_C(gBias);
            cute::Tensor tOgBiasg = thr_mma.partition_C(gBias_gate);
            for (int i = 0; i < cute::size(accum); i++)
            {
                accum(i) += tOgBias(i);
                accum_gate(i) += tOgBiasg(i);
            }
        }

        // (3) calculate swiglu
        using ActivationFn = typename KT::ActivationFn;
        ActivationFn fn{};
        CUTLASS_PRAGMA_UNROLL
        for (int temp_iter = 0; temp_iter < cute::size(accum); temp_iter++)
        {
            accum(temp_iter) = fn(accum_gate(temp_iter)) * accum(temp_iter);
        }

        // (4) push all the result to smem
        // (4.1) convert result from ElementAccum to ElementInput
        cute::Tensor temp_accum = util_convert_type<KT::ElementOutput>(accum);
        // if (cute::thread0()) {
        //     cute::print(temp_accum(0, 0, 0));
        //     printf("\n");
        // }
        // (4.2) retile rf and smem for copy back..
        auto smem_tiled_copy_O = cute::make_tiled_copy_C(typename KT::SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
        // cute::clear(sO);
        cute::Tensor taccumrO = smem_thr_copy_O.retile_S(temp_accum);
        cute::Tensor taccumsO = smem_thr_copy_O.partition_D(sO);

        // (4.3) copy rf result to smem (TODO: maybe use forloop for better performance..)
        cute::copy(smem_tiled_copy_O, taccumrO, taccumsO);
        __syncthreads();

        // (4.4) sO -> rO -> gO

        typename KT::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        // auto gmem_thr_copy_Bias = gmem_tiled_copy_O.get_thread_slice(thread_idx % KT::kGmemTrheadsPerRow); //
        // remember, for all the threads in the same col, they have the same idx for bias..
        cute::Tensor gO = gOutput_mn(cute::_, cute::_, block_m_idx, block_n_idx);
        // cute::Tensor gBias = gBias_mn(cute::_, cute::_, 0, block_n_idx); // bias only have one row..
        auto tOsO = gmem_thr_copy_O.partition_S(sO);
        auto tOgO = gmem_thr_copy_O.partition_D(gO);
        // auto tOgBias = gmem_thr_copy_O.partition_D(gBias);
        cute::Tensor cOutput = cute::make_identity_tensor(
            cute::make_shape(cute::size<0>(typename KT::TileShape{}), cute::size<1>(typename KT::TileShape{})));
        cute::Tensor tOcO = gmem_thr_copy_O.partition_D(cOutput);
        cute::Tensor tOrO = cute::make_tensor<KT::ElementOutput>(cute::shape(tOgO));
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < cute::size<1>(tOgO); ++m)
        {
            if (cute::get<0>(tOcO(0, m, 0)) < residue_m)
            {
                cute::copy(gmem_tiled_copy_O, tOrO(cute::_, m, cute::_), tOgO(cute::_, m, cute::_));
            }
        }
    }
};

template <typename ElementInput_, typename ElementWeight_, typename ElementOutput_, int TileM_, int TileN_, int TileK_,
    int Stages_, Activation_Type activation_type_>
struct Fused_Moe_Kernel_routine_sm80<ElementInput_, ElementWeight_, ElementOutput_, TileM_, TileN_, TileK_, Stages_,
    activation_type_, std::enable_if_t<!isGateActivation(activation_type_)>>
{

    using KT = Fused_Moe_Kernel_traits_sm80<ElementInput_, ElementWeight_, ElementOutput_, TileM_, TileN_, TileK_,
        Stages_, activation_type_>;
    using Params = Routine_Params<ElementInput_, ElementWeight_, ElementOutput_>;

    CUTE_DEVICE auto gmem_tensor_init(int const problem_index, int const gemm_m, Params const& params)
    {
        using X = cute::Underscore;

        int const M = gemm_m;
        int const N1 = params.gemm_n;
        int const K1 = params.gemm_k;

        int const row_jump = ((problem_index == 0) ? 0 : params.total_rows_before_expert[problem_index - 1]);
        typename KT::ElementInput const* ptr_input_ = params.ptr_input + row_jump * K1;
        typename KT::ElementWeight const* ptr_fc1_ = params.ptr_fc1 + problem_index * N1 * K1;
        typename KT::ElementInput const* ptr_bias_
            = (params.ptr_bias == nullptr) ? nullptr : params.ptr_bias + problem_index * N1;
        typename KT::ElementOutput* ptr_output_ = params.ptr_output + row_jump * N1;

        cute::Tensor mInput_mk
            = cute::make_tensor(cute::make_gmem_ptr(static_cast<typename KT::ElementInput const*>(ptr_input_)),
                cute::make_shape(M, K1), cute::make_stride(K1, cute::_1{}));

        cute::Tensor mfc1_nk
            = cute::make_tensor(cute::make_gmem_ptr(static_cast<typename KT::ElementWeight const*>(ptr_fc1_)),
                cute::make_shape(N1, K1), cute::make_stride(K1, cute::_1{}));

        cute::Tensor mBias_mn = cute::make_tensor(
            cute::make_gmem_ptr(static_cast<typename KT::ElementInput const*>(ptr_bias_)), cute::make_shape(M, N1),
            cute::make_stride(cute::Int<0>{}, cute::_1{})); // trick: bias shape is [1, N], but we use [M, N].

        cute::Tensor mOutput_mn
            = cute::make_tensor(cute::make_gmem_ptr(static_cast<typename KT::ElementInput*>(ptr_output_)),
                cute::make_shape(M, N1), cute::make_stride(N1, cute::_1{}));

        cute::Tensor gInput_mk = cute::local_tile(mInput_mk, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<cute::_1, X, cute::_1>{}); // (BLK_M, BLK_K, m, k)
        cute::Tensor gfc1_nk = cute::local_tile(mfc1_nk, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<X, cute::_1, cute::_1>{}); // (BLK_N, BLK_K, n, k)

        cute::Tensor gBias_mn = cute::local_tile(mBias_mn, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<cute::_1, cute::_1, X>{}); // (BLK_M, BLK_N, m, n)

        cute::Tensor gOutput_mn = cute::local_tile(mOutput_mn, typename KT::TileShape{},
            cute::make_coord(cute::_, cute::_, cute::_), cute::Step<cute::_1, cute::_1, X>{}); // (BLK_M, BLK_N, m, n)

        return cute::make_tuple(gInput_mk, gfc1_nk, gBias_mn, gOutput_mn);
    }

    // be careful, m_idx will change when use another tile shape..
    CUTE_DEVICE void run_routine(
        Params const& params, int const problem_index, int const block_m_idx, int const block_n_idx, int const gemm_m)
    {
        extern __shared__ char smem_[];
        typename KT::SharedStorage& shared_storage = *reinterpret_cast<typename KT::SharedStorage*>(smem_);
        int const thread_idx = threadIdx.x;
        // gmem tensor partition ..
        auto [gInput_mk, gfc1_nk, gBias_mn, gOutput_mn] = gmem_tensor_init(problem_index, gemm_m, params);
        int const residue_m = gemm_m - block_m_idx * cute::size<0>(gInput_mk);
        auto const n_tile_count = cute::size<2>(gfc1_nk);

        // smem tensor ..
        cute::Tensor sInput = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_input.data()), typename KT::SmemLayoutA{}); // (BLK_M, BLK_K, Stage)
        cute::Tensor sfc1_weight = cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_fc1_weight.data()),
            typename KT::SmemLayoutB{});                                                        // (BLK_N, BLK_K, Stage)
        cute::Tensor sO = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_o.data()), typename KT::SmemLayoutO{});     // (BLK_M, BLK_N)

        // (1) first step, get the fc1_res and fc1_gate

        // (1.1) get partition for gmem -> smem
        cute::Tensor gInput = gInput_mk(cute::_, cute::_, block_m_idx, cute::_); // (BLK_M, BLK_K, k)
        cute::Tensor gfc1 = gfc1_nk(cute::_, cute::_, block_n_idx, cute::_);     // (BLK_N, BLK_K, k)

        typename KT::GmemTiledCopyA gmem_tiled_copy_A;
        typename KT::GmemTiledCopyB gmem_tiled_copy_B;
        auto gmem_thr_copy_A = gmem_tiled_copy_A.get_slice(thread_idx);
        auto gmem_thr_copy_B = gmem_tiled_copy_B.get_slice(thread_idx);

        cute::Tensor tInputgInput = gmem_thr_copy_A.partition_S(gInput);  // (ACPY,ACPY_M,ACPY_K,k)
        cute::Tensor tInputsInput = gmem_thr_copy_A.partition_S(sInput);  // (ACPY,ACPY_M,ACPY_K,Stage)
        cute::Tensor tfc1gfc1 = gmem_thr_copy_B.partition_S(gfc1);        // (BCPY,BCPY_N,BCPY_K,k)
        cute::Tensor tfc1sfc1 = gmem_thr_copy_B.partition_D(sfc1_weight); // (BCPY,BCPY_N,BCPY_K,Stage)

        // Allocate predicate tensors for input and fc weight (actually we only need input predicate tensor)
        cute::Tensor tInputpInput
            = cute::make_tensor<bool>(cute::make_shape(cute::size<1>(tInputsInput), cute::size<2>(tInputsInput)),
                cute::Stride<cute::_1, cute::_0>{});
        // Construct identity layout for sInput
        cute::Tensor cInput = make_identity_tensor(
            make_shape(cute::size<0>(sInput), cute::size<1>(sInput))); // (BLK_M,BLK_K) -> (blk_m,blk_k)

        // Repeat the partitioning with identity layouts
        cute::Tensor tInputcInput = gmem_thr_copy_A.partition_S(cInput); // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

        // Set predicates for m bounds
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < cute::size<0>(tInputpInput); ++m)
        {
            tInputpInput(m, 0) = cute::get<0>(tInputcInput(0, m, 0)) < residue_m; // blk_m coord < residue_m
        }

        // (1.2) prefetch gmem -> smem
        cute::clear(tInputsInput);                                           // we don't need to clear tfc1sfc1..
        auto k_tile_iter = cute::make_coord_iterator(cute::size<2>(gInput)); // emm, iter start from 0
        int k_tile_count = cute::size<2>(gInput);
        CUTLASS_PRAGMA_UNROLL
        for (int k_pipe = 0; k_pipe < KT::Stages - 1; ++k_pipe)
        {
            if (k_tile_count <= 0)
            {
                cute::clear(tInputpInput);
            }
            // cute::copy(gmem_tiled_copy_A, tInputgInput(cute::_, cute::_, cute::_, *k_tile_iter),
            //    tInputsInput(cute::_, cute::_, cute::_, k_pipe));
            // use copy_if
            cute::copy_if(gmem_tiled_copy_A, tInputpInput, tInputgInput(cute::_, cute::_, cute::_, *k_tile_iter),
                tInputsInput(cute::_, cute::_, cute::_, k_pipe));
            cute::copy(gmem_tiled_copy_B, tfc1gfc1(cute::_, cute::_, cute::_, *k_tile_iter),
                tfc1sfc1(cute::_, cute::_, cute::_, k_pipe));
            cute::cp_async_fence();
            k_tile_count--;
            if (k_tile_count > 0)
            {
                ++k_tile_iter;
            }
        }

        // (1.3) get partition for rf
        typename KT::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
        cute::Tensor tOrInput = thr_mma.partition_fragment_A(sInput(cute::_, cute::_, 0));    // (MMA,MMA_M,MMA_K)
        cute::Tensor tOrfc1 = thr_mma.partition_fragment_B(sfc1_weight(cute::_, cute::_, 0)); // (MMA,MMA_N,MMA_K)

        cute::Tensor accum
            = cute::partition_fragment_C(tiled_mma, cute::take<0, 2>(typename KT::TileShape{})); // (MMA,MMA_M,MMA_N)
        cute::clear(accum);
        // checkout the shape
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOrInput) == cute::size<1>(accum));  // MMA_M
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOrfc1) == cute::size<2>(accum));    // MMA_N
        CUTE_STATIC_ASSERT_V(cute::size<2>(tOrInput) == cute::size<2>(tOrfc1)); // MMA_K
        CUTE_STATIC_ASSERT_V(cute::size(gmem_tiled_copy_A) == cute::size(tiled_mma));
        CUTE_STATIC_ASSERT_V(cute::size(gmem_tiled_copy_B) == cute::size(tiled_mma));

        // (1.4)retiling the smem and rf for copy..
        auto smem_tiled_copy_A = cute::make_tiled_copy_A(typename KT::SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
        cute::Tensor tOsInput = smem_thr_copy_A.partition_S(sInput);                        // (CPY,CPY_M,CPY_K,Stage)
        cute::Tensor tOrInput_copy_view = smem_thr_copy_A.retile_D(tOrInput);               // (CPY,CPY_M,CPY_K)
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOsInput) == cute::size<1>(tOrInput_copy_view)); // CPY_M
        CUTE_STATIC_ASSERT_V(cute::size<2>(tOsInput) == cute::size<2>(tOrInput_copy_view)); // CPY_K

        auto smem_tiled_copy_B = cute::make_tiled_copy_B(typename KT::SmemCopyAtomB{}, tiled_mma);
        auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
        cute::Tensor tOsfc1 = smem_thr_copy_B.partition_S(sfc1_weight);                 // (CPY,CPY_N,CPY_K,Stage)
        cute::Tensor tOrfc1_copy_view = smem_thr_copy_B.retile_D(tOrfc1);               // (CPY,CPY_N,CPY_K)
        CUTE_STATIC_ASSERT_V(cute::size<1>(tOsfc1) == cute::size<1>(tOrfc1_copy_view)); // CPY_N
        CUTE_STATIC_ASSERT_V(cute::size<2>(tOsfc1) == cute::size<2>(tOrfc1_copy_view)); // CPY_K

        // (1.5) mainloop
        // Current pipe index in smem to read from
        int smem_pipe_read = 0;
        // Current pipe index in smem to write to
        int smem_pipe_write = KT::Stages - 1;

        cute::Tensor tOsInput_p = tOsInput(cute::_, cute::_, cute::_, smem_pipe_read);
        cute::Tensor tOsfc1_p = tOsfc1(cute::_, cute::_, cute::_, smem_pipe_read);

        constexpr int K_BLOCK_MAX = cute::size<2>(tOrInput);
        // prefetch register pipeline
        if constexpr (K_BLOCK_MAX > 1)
        {
            cute::cp_async_wait<KT::Stages - 2>();
            __syncthreads();

            // Prefetch the first rmem from the first k-tile
            cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, cute::Int<0>{}),
                tOrInput_copy_view(cute::_, cute::_, cute::Int<0>{}));
            cute::copy(smem_tiled_copy_B, tOsfc1_p(cute::_, cute::_, cute::Int<0>{}),
                tOrfc1_copy_view(cute::_, cute::_, cute::Int<0>{}));
        }
        // k loop for mainloop (k - (stage - 1) -> -(stage - 1), if k_tile_count > 0, it means we still need to
        // fetch gmem to smem)
        CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > -(KT::Stages - 1); --k_tile_count)
        {
            cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{},
                [&](auto k_block)
                {
                    if (k_block == K_BLOCK_MAX - 1)
                    {
                        tOsInput_p = tOsInput(cute::_, cute::_, cute::_, smem_pipe_read);
                        tOsfc1_p = tOsfc1(cute::_, cute::_, cute::_, smem_pipe_read);
                        cute::cp_async_wait<KT::Stages - 2>();
                        __syncthreads();
                    }
                    // Load A, B shmem->regs for k_block+1
                    auto k_block_next = (k_block + cute::_1{}) % K_BLOCK_MAX;
                    cute::copy(smem_tiled_copy_A, tOsInput_p(cute::_, cute::_, k_block_next),
                        tOrInput_copy_view(cute::_, cute::_, k_block_next));
                    cute::copy(smem_tiled_copy_B, tOsfc1_p(cute::_, cute::_, k_block_next),
                        tOrfc1_copy_view(cute::_, cute::_, k_block_next));
                    // Copy gmem to smem before computing gemm on each k-pipe
                    if (k_block == 0)
                    {
                        if (k_tile_count <= 0)
                        {
                            cute::clear(tInputpInput);
                        }
                        // cute::copy(gmem_tiled_copy_A, tInputgInput(cute::_, cute::_, cute::_, *k_tile_iter),
                        //    tInputsInput(cute::_, cute::_, cute::_, smem_pipe_write));
                        cute::copy_if(gmem_tiled_copy_A, tInputpInput,
                            tInputgInput(cute::_, cute::_, cute::_, *k_tile_iter),
                            tInputsInput(cute::_, cute::_, cute::_, smem_pipe_write));
                        cute::copy(gmem_tiled_copy_B, tfc1gfc1(cute::_, cute::_, cute::_, *k_tile_iter),
                            tfc1sfc1(cute::_, cute::_, cute::_, smem_pipe_write));
                        cute::cp_async_fence();
                        if (k_tile_count - 1 > 0)
                        {
                            ++k_tile_iter;
                        }

                        // Advance the pipe -- Doing it here accounts for K_BLOCK_MAX = 1 (no rmem pipe)
                        smem_pipe_write = smem_pipe_read;
                        ++smem_pipe_read;
                        smem_pipe_read = (smem_pipe_read == KT::Stages) ? 0 : smem_pipe_read;
                    }
                    // Thread-level register gemm for k_block
                    cute::gemm(tiled_mma, accum, tOrInput(cute::_, cute::_, k_block), tOrfc1(cute::_, cute::_, k_block),
                        accum);
                });
        }
        // if (cute::thread0()) {
        //     cute::print(accum_gate(0, 0, 0));
        //     printf("\n");
        // }
        // (2) add bias if it has..
        if (params.ptr_bias != nullptr)
        {
            cute::Tensor gBias = gBias_mn(cute::_, cute::_, 0, block_n_idx); // bias only have one row..
            cute::Tensor tOgBias = thr_mma.partition_C(gBias);
            for (int i = 0; i < cute::size(accum); i++)
            {
                accum(i) += tOgBias(i);
            }
        }
        // (3) calculate swiglu
        using ActivationFn = typename KT::ActivationFn;
        ActivationFn fn{};
        CUTLASS_PRAGMA_UNROLL
        for (int temp_iter = 0; temp_iter < cute::size(accum); temp_iter++)
        {
            accum(temp_iter) = fn(accum(temp_iter));
        }

        // (4) push all the result to smem
        // (4.1) convert result from ElementAccum to ElementInput
        cute::Tensor temp_accum = util_convert_type<KT::ElementOutput>(accum);
        // if (cute::thread0()) {
        //     cute::print(temp_accum(0, 0, 0));
        //     printf("\n");
        // }
        // (4.2) retile rf and smem for copy back..
        auto smem_tiled_copy_O = cute::make_tiled_copy_C(typename KT::SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);
        // cute::clear(sO);
        cute::Tensor taccumrO = smem_thr_copy_O.retile_S(temp_accum);
        cute::Tensor taccumsO = smem_thr_copy_O.partition_D(sO);

        // (4.3) copy rf result to smem (TODO: maybe use forloop for better performance..)
        cute::copy(smem_tiled_copy_O, taccumrO, taccumsO);
        __syncthreads();

        // (4.4) sO -> rO -> gO

        typename KT::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
        // auto gmem_thr_copy_Bias = gmem_tiled_copy_O.get_thread_slice(thread_idx % KT::kGmemTrheadsPerRow); //
        cute::Tensor gO = gOutput_mn(cute::_, cute::_, block_m_idx, block_n_idx);
        auto tOsO = gmem_thr_copy_O.partition_S(sO);
        auto tOgO = gmem_thr_copy_O.partition_D(gO);
        cute::Tensor cOutput = cute::make_identity_tensor(
            cute::make_shape(cute::size<0>(typename KT::TileShape{}), cute::size<1>(typename KT::TileShape{})));
        cute::Tensor tOcO = gmem_thr_copy_O.partition_D(cOutput);
        cute::Tensor tOrO = cute::make_tensor<KT::ElementOutput>(cute::shape(tOgO));
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < cute::size<1>(tOgO); ++m)
        {
            if (cute::get<0>(tOcO(0, m, 0)) < residue_m)
            {
                cute::copy(gmem_tiled_copy_O, tOrO(cute::_, m, cute::_), tOgO(cute::_, m, cute::_));
            }
        }
    }
};

} // namespace fused_moe
