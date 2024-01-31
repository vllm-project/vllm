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

/*! \file
    \brief Scheduler for grouped GEMM
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped_problem_visitor.h"
#include "cutlass/matrix_coord.h"

#include "cutlass_extensions/gemm/kernel/gemm_moe_problem_visitor.h"
#include "cutlass_extensions/gemm/kernel/moe_problem_visitor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace gemm
{
namespace kernel
{

/// Visitor class to abstract away the algorithm for iterating over tiles
template <typename ThreadblockShape, GroupScheduleMode GroupScheduleMode_, int PrefetchTileCount, int ThreadCount,
    bool Transposed = false>
struct GemmMoeProblemVisitor
    : public MoeProblemVisitor<detail::GemmGroupedProblemSizeHelper<ThreadblockShape, Transposed>, ThreadblockShape,
          GroupScheduleMode_, PrefetchTileCount, ThreadCount>
{

    static bool const kTransposed = Transposed;

    using ProblemSizeHelper = detail::GemmGroupedProblemSizeHelper<ThreadblockShape, Transposed>;
    using Base
        = MoeProblemVisitor<ProblemSizeHelper, ThreadblockShape, GroupScheduleMode_, PrefetchTileCount, ThreadCount>;
    using Params = typename Base::Params;
    using SharedStorage = typename Base::SharedStorage;

    //
    // Methods
    //
    CUTLASS_DEVICE
    GemmMoeProblemVisitor(Params const& params_, SharedStorage& shared_storage_, int32_t block_idx)
        : Base(params_, shared_storage_, block_idx)
    {
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
