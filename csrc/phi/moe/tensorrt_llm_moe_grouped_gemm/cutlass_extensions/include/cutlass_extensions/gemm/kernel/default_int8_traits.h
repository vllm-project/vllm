/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

namespace cutlass
{
namespace gemm
{
namespace kernel
{

template <typename arch>
struct Int8GemmArchTraits
{
    using OperatorClass = cutlass::arch::OpClassSimt;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
};

// ======================= Turing Traits ==============================
template <>
struct Int8GemmArchTraits<cutlass::arch::Sm75>
{
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
};

// ======================= Ampere Traits ==============================
template <>
struct Int8GemmArchTraits<cutlass::arch::Sm80>
{
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
};

} // namespace kernel
} // namespace gemm
} // namespace cutlass
