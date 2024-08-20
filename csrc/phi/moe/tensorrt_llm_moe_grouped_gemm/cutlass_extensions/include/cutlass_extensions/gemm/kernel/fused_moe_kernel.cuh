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
#include <cutlass/gemm/kernel/gemm_grouped_problem_visitor.h>
#include <cutlass/trace.h>
#include <cutlass_extensions/gemm/kernel/fused_moe_kernel_routine.cuh>
#include <cutlass_extensions/gemm/kernel/fused_moe_kernel_traits.cuh>
#include <cutlass_extensions/gemm/kernel/moe_problem_visitor.h>

namespace fused_moe
{
template <typename ElementInput_, typename ElementWeight_, typename ElementOutput_, int MaxTileM_, int TileN_,
    int TileK_, int Stages_, Activation_Type activation_type_>
struct Fused_Moe_Kernel_sm80
{
    static constexpr int kMaxTileM = MaxTileM_;
    static constexpr int kTileN = isGateActivation(activation_type_) ? TileN_ / 2 : TileN_;
    static constexpr int kTileK = TileK_;
    static constexpr int kStages = Stages_;
    static constexpr Activation_Type activation_type = activation_type_;

    using ElementInput = ElementInput_;
    using ElementWeight = ElementWeight_;
    using ElementOutput = ElementOutput_;
    using BaseKernelTraits = Fused_Moe_Kernel_traits_sm80<ElementInput, ElementWeight, ElementOutput, kMaxTileM, kTileN,
        kTileK, kStages, activation_type>;
    using Routine_Arguments = Routine_Arguments<ElementInput, ElementWeight, ElementOutput>;
    using Routine_Params = Routine_Params<ElementInput, ElementWeight, ElementOutput>;
    using ProblemVisitor
        = cutlass::gemm::kernel::MoeProblemVisitor<cutlass::gemm::kernel::detail::GemmGroupedProblemSizeHelper<
                                                       cutlass::gemm::GemmShape<kMaxTileM, kTileN, kTileK>, false>,
            cutlass::gemm::GemmShape<kMaxTileM, kTileN, kTileK>, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
            BaseKernelTraits::kThreadCount, BaseKernelTraits::kThreadCount>;

    struct Arguments
    {
        Routine_Arguments routine_args;
        int problem_count{};
        int threadblock_count{};
    };

    struct Params
    {
        Routine_Params routine_params;
        int threadblock_count{};
        typename ProblemVisitor::Params problem_visitor_param;
    };

    using BaseKernelTraits_m16 = Fused_Moe_Kernel_traits_sm80<ElementInput, ElementWeight, ElementOutput, 16, kTileN,
        kTileK, kStages, activation_type>;
    static constexpr bool use_m16 = TileK_ >= 64; // use tileshape m = 16 when original tileshape k >= 64

    static constexpr int kSmemSize = use_m16
        ? (BaseKernelTraits::kSmemSize > BaseKernelTraits_m16::kSmemSize ? BaseKernelTraits::kSmemSize
                                                                         : BaseKernelTraits_m16::kSmemSize)
        : BaseKernelTraits::kSmemSize;
    static constexpr int kThreadCount = BaseKernelTraits::kThreadCount;

    static constexpr bool can_implement(int const avaliable_smem_size)
    {
        return BaseKernelTraits::can_implement(avaliable_smem_size);
    }

    static Params to_underlying_arguments(Arguments const& args)
    {
        return {{args.routine_args.ptr_input, args.routine_args.ptr_fc1, args.routine_args.ptr_bias,
                    args.routine_args.ptr_output, args.routine_args.total_rows_before_expert, args.routine_args.gemm_n,
                    args.routine_args.gemm_k, args.routine_args.num_expert},
            args.threadblock_count,
            {args.routine_args.total_rows_before_expert, args.routine_args.gemm_n, args.routine_args.gemm_k,
                args.problem_count, nullptr, 0}};
    }

    CUTE_DEVICE
    void run_device(Params const& params)
    {
#define ROUTINE_PATH(kTileM_size)                                                                                      \
    {                                                                                                                  \
        constexpr int kTileM = use_m16 ? (kTileM_size) : ((kTileM_size) == 16 ? 32 : (kTileM_size));                   \
        using RoutineTraits = Fused_Moe_Kernel_routine_sm80<ElementInput, ElementWeight, ElementOutput, kTileM,        \
            kTileN, kTileK, kStages, activation_type>;                                                                 \
        RoutineTraits routine{};                                                                                       \
        const int block_m_idx = (block_m_idx_temp) *kMaxTileM / kTileM;                                                \
        routine.run_routine(params.routine_params, problem_index, block_m_idx, block_n_idx, gemm_m);                   \
    }
        typename ProblemVisitor::SharedStorage dummy_storage{};
        ProblemVisitor problem_visitor(params.problem_visitor_param, dummy_storage, blockIdx.x);
        while (problem_visitor.next_tile())
        {
            auto problem_size = problem_visitor.problem_size();
            auto grid_size = problem_visitor.grid_shape(problem_size);
            auto problem_index = problem_visitor.problem_index();
            int32_t cta_idx = int32_t(problem_visitor.threadblock_idx());
            int const gemm_m = problem_size.m();
            const int32_t block_m_idx_temp = cta_idx / grid_size.n();
            const int32_t block_n_idx = cta_idx % grid_size.n();

            int const residue_m = gemm_m - kMaxTileM * block_m_idx_temp;
            if (residue_m > kMaxTileM / 2)
            {
                using RoutineTraits = Fused_Moe_Kernel_routine_sm80<ElementInput, ElementWeight, ElementOutput,
                    kMaxTileM, kTileN, kTileK, kStages, activation_type>;
                RoutineTraits routine{};
                routine.run_routine(params.routine_params, problem_index, block_m_idx_temp, block_n_idx, gemm_m);
            }
            else
            {

                if constexpr (kMaxTileM >= 128)
                {
                    if (residue_m > 32)
                    {
                        ROUTINE_PATH(64);
                    }
                    else if (residue_m > 16)
                    {
                        ROUTINE_PATH(32);
                    }
                    else
                    {
                        // TODO: use cuda core gemm here
                        ROUTINE_PATH(16);
                    }
                }
                else if (kMaxTileM == 64)
                {
                    if (residue_m > 16)
                    {
                        ROUTINE_PATH(32);
                    }
                    else
                    {
                        // TODO: use cuda core gemm here
                        ROUTINE_PATH(16);
                    }
                }
                else if (kMaxTileM == 32)
                {
                    // TODO: use cuda core gemm here
                    ROUTINE_PATH(16);
                }
                else
                {
                    // TODO: use cuda core gemm here
                    ROUTINE_PATH(16);
                }
            }
            problem_visitor.advance(gridDim.x);
        }
#undef ROUTINE_PATH
    }
};

template <typename GemmType>
__global__ void run_global(__grid_constant__ typename GemmType::Params const params)
{
    GemmType gemm;
    gemm.run_device(params);
}

/// Computes the maximum number of active blocks per multiprocessor
template <typename GemmType>
static int fused_gemm_maximum_active_blocks(int smem_capacity = -1)
{

    CUTLASS_TRACE_HOST("BaseGrouped::maximum_active_blocks()");

    constexpr int smem_size = GemmType::kSmemSize;

    CUTLASS_TRACE_HOST("  smem_size: " << smem_size << " bytes");

    cudaError_t result;
    if (smem_size > (48 << 10))
    {
        result = cudaFuncSetAttribute(run_global<GemmType>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        if (result != cudaSuccess)
        {
            // Call cudaGetLastError() to clear the error bit
            result = cudaGetLastError();
            CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error " << cudaGetErrorString(result));
            return -1;
        }
    }

    int max_active_blocks = -1;
    result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, run_global<GemmType>, GemmType::kThreadCount, smem_size);

    if (result != cudaSuccess)
    {
        // Call cudaGetLastError() to clear the error bit
        result = cudaGetLastError();
        CUTLASS_TRACE_HOST(
            "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error " << cudaGetErrorString(result));
        return -1;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
}
} // namespace fused_moe
