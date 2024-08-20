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

#include <cutlass/epilogue/thread/activation.h>
#include <cutlass_extensions/epilogue_helpers.h>
#include <cutlass_extensions/gemm/kernel/moe_cute_util.cuh>
#include <cutlass_extensions/gemm/kernel/moe_problem_visitor.h>

namespace fused_moe
{
template <typename ElementInput, typename ElementWeight, typename ElementOutput>
struct Routine_Arguments
{
    ElementInput* ptr_input{};
    ElementWeight* ptr_fc1{};
    ElementInput* ptr_bias{};
    ElementOutput* ptr_output{};
    int64_t* total_rows_before_expert{};
    int gemm_n{};
    int gemm_k{};
    int num_expert{};
};

template <typename ElementInput, typename ElementWeight, typename ElementOutput>
struct Routine_Params
{
    ElementInput* ptr_input{};
    ElementWeight* ptr_fc1{};
    ElementInput* ptr_bias{};
    ElementOutput* ptr_output{};
    int64_t* total_rows_before_expert{};
    int gemm_n{};
    int gemm_k{};
    int num_expert{};
};

enum class Activation_Type
{
    Gelu = 0,
    Relu,
    Silu,
    Swiglu,
    Geglu,
    Identity,
    InvalidType
};

constexpr bool isGateActivation(Activation_Type const& activation_type)
{
    return activation_type == Activation_Type::Swiglu || activation_type == Activation_Type::Geglu;
}

template <typename CutlassExtensionEpilogueTag>
constexpr Activation_Type EpilogueRouting(bool /*is_gate*/)
{
    return Activation_Type::InvalidType;
}

template <>
constexpr Activation_Type EpilogueRouting<tensorrt_llm::cutlass_extensions::EpilogueOpDefault>(bool /*is_gate*/)
{
    return Activation_Type::Identity;
}

template <>
constexpr Activation_Type EpilogueRouting<tensorrt_llm::cutlass_extensions::EpilogueOpDefaultReLU>(bool /*is_gate*/)
{
    return Activation_Type::Relu;
}

template <>
constexpr Activation_Type EpilogueRouting<tensorrt_llm::cutlass_extensions::EpilogueOpDefaultSilu>(bool is_gate)
{
    return is_gate ? Activation_Type::Swiglu : Activation_Type::Silu;
}

template <>
constexpr Activation_Type EpilogueRouting<tensorrt_llm::cutlass_extensions::EpilogueOpDefaultFtGelu>(bool is_gate)
{
    return is_gate ? Activation_Type::Geglu : Activation_Type::Gelu;
}

/* fusing all three kernels has many limitations. This is the simpler version. Just fuse first two kernels..*/
template <typename ElementInput_, typename ElementWeight_, typename ElementOutput_, int TileM_, int TileN_, int TileK_,
    int Stages_, Activation_Type activation_type>
struct Fused_Moe_Kernel_traits_sm80
{
    using ElementInput = ElementInput_;
    using ElementWeight = ElementWeight_;
    using ElementAccum = float;
    using ElementOutput = ElementOutput_;

    using index_t = uint32_t;
    static_assert(TileM_ % 16 == 0);
    static_assert(TileN_ % 32 == 0);
    static_assert(TileK_ % 32 == 0);
    static constexpr int Stages = Stages_;
    static constexpr int kTileM = TileM_;
    static constexpr int kTileN = TileN_;
    static constexpr int kTileK = (kTileM > 16) ? (TileK_) : (TileK_ >= 64 ? TileK_ : 64);

    // tile shape
    using TileShape = cute::Shape<cute::Int<kTileM>, cute::Int<kTileN>, cute::Int<kTileK>>;
    static constexpr int kWarpsCount = 4;
    static constexpr int kThreadCount = kWarpsCount * 32;

    // MMA atom arch and layout
    using MMA_Atom_Arch = std::conditional_t<std::is_same_v<ElementInput, cutlass::half_t>,
        cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>, cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>>;
    // using ValLayoutMNK = cute::Layout<cute::Shape<cute::_1, cute::_2, cute::_1>>;
    using ThreadLayoutMNK
        = std::conditional_t<kTileM == 16, cute::Layout<cute::Shape<cute::_1, cute::Int<kWarpsCount / 1>, cute::_1>>,
            cute::Layout<cute::Shape<cute::_2, cute::Int<kWarpsCount / 2>, cute::_1>>>;
    using ValLayoutMNK = std::conditional_t<kTileM == 16, cute::Tile<cute::_16, cute::_64, cute::_16>,
        cute::Tile<cute::_32, cute::_32, cute::_16>>;
    using TiledMma = cute::TiledMMA<MMA_Atom_Arch, ThreadLayoutMNK,
        ValLayoutMNK>; // 32x32x16 or 16x64x16 MMA for LDSM if kWarp = 4
    static constexpr int kAlignment = 8;
    static constexpr int kBlcokKSmem = (kTileM == 16) ? 64 : 32;
    // A memory copy operand
    using DefaultOperandA
        = DefaultGemm_TensorOpSm80_OperandA<ElementInput, cutlass::layout::RowMajor, kAlignment, kBlcokKSmem>;
    using SmemLayoutAtomA = typename DefaultOperandA::SmemLayoutAtom;
    using SmemCopyAtomA = typename DefaultOperandA::SmemCopyAtom;
    using GmemTiledCopyA = typename DefaultOperandA::GmemTiledCopy;

    // B memory copy operand
    using DefaultOperandB
        = DefaultGemm_TensorOpSm80_OperandB<ElementWeight, cutlass::layout::ColumnMajor, kAlignment, kBlcokKSmem>;
    using SmemLayoutAtomB = typename DefaultOperandB::SmemLayoutAtom;
    using SmemCopyAtomB = typename DefaultOperandB::SmemCopyAtom;
    using GmemTiledCopyB = typename DefaultOperandB::GmemTiledCopy;

    // Output memory copy operand
    using SmemLayoutAtomO = SmemLayoutAtomA;
    using SmemCopyAtomO = cute::Copy_Atom<cute::DefaultCopy, ElementOutput>;
    static constexpr int kGmemElementPerLoad = sizeof(cute::uint128_t) / sizeof(ElementOutput);
    static constexpr int kGmemTrheadsPerRow = kBlcokKSmem / kGmemElementPerLoad;
    using GmemLayoutAtomO
        = cute::Layout<cute::Shape<cute::Int<kThreadCount / kGmemTrheadsPerRow>, cute::Int<kGmemTrheadsPerRow>>,
            cute::Stride<cute::Int<kGmemTrheadsPerRow>, cute::_1>>;
    using GmemTiledCopyO = decltype(cute::make_tiled_copy(cute::Copy_Atom<cute::DefaultCopy, ElementOutput>{},
        GmemLayoutAtomO{}, cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));

    static_assert(cute::rank(SmemLayoutAtomA{}) == 2);
    static_assert(cute::size<0>(TileShape{}) % cute::size<0>(SmemLayoutAtomA{}) == 0); // M
    static_assert(cute::size<2>(TileShape{}) % cute::size<1>(SmemLayoutAtomA{}) == 0); // K
    static_assert(cute::rank(SmemLayoutAtomB{}) == 2);
    static_assert(cute::size<1>(TileShape{}) % cute::size<0>(SmemLayoutAtomB{}) == 0); // N
    static_assert(cute::size<2>(TileShape{}) % cute::size<1>(SmemLayoutAtomB{}) == 0); // K

    using SmemLayoutA = decltype(cute::tile_to_shape(SmemLayoutAtomA{},
        cute::make_shape(
            cute::shape<0>(TileShape{}), cute::shape<2>(TileShape{}), cute::Int<Stages>{}))); // BLK_M, BLK_K, Stages
    using SmemLayoutB = decltype(cute::tile_to_shape(SmemLayoutAtomB{},
        cute::make_shape(
            cute::shape<1>(TileShape{}), cute::shape<2>(TileShape{}), cute::Int<Stages>{}))); // BLK_N, BLK_K, Stages
    using SmemLayoutO = decltype(cute::tile_to_shape(
        SmemLayoutAtomO{}, cute::make_shape(cute::shape<0>(TileShape{}), cute::shape<1>(TileShape{})))); // BLK_M, BLK_N

    // we need at least 2 stages..
    static_assert(Stages >= 2);

    struct SharedStorageNormal : cute::aligned_struct<128>
    {
        cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>> smem_input;
        cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_fc1_weight;
        cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutO>> smem_o;
    };

    struct SharedStorageGate : cute::aligned_struct<128>
    {
        cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutA>> smem_input;
        cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_fc1_gate_weight;
        cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutB>> smem_fc1_weight;
        cute::array_aligned<ElementInput, cute::cosize_v<SmemLayoutO>> smem_o;
    };

    using SharedStorage = std::conditional_t<isGateActivation(activation_type), SharedStorageGate, SharedStorageNormal>;

    using ActivationFn = std::conditional_t<activation_type == Activation_Type::Gelu
            || activation_type == Activation_Type::Geglu,
        cutlass::epilogue::thread::GELU<float>,
        std::conditional_t<activation_type == Activation_Type::Relu, cutlass::epilogue::thread::ReLU<float>,
            std::conditional_t<activation_type == Activation_Type::Silu || activation_type == Activation_Type::Swiglu,
                cutlass::epilogue::thread::SiLu<float>, cutlass::epilogue::thread::Identity<float>>>>;

    static constexpr int kSmemSize = static_cast<int>(sizeof(SharedStorage));

    static constexpr bool can_implement(int const avaliable_smem_size)
    {
        return avaliable_smem_size > kSmemSize;
    }

    // #endif
};
} // namespace fused_moe
