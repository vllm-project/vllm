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

#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel
{

////////////////////////////////////////////////////////////////////////////////

/*
 * Stateless universal device GEMM kernel type that treats GEMM as
 * a composition of a collective mainloop and a collective epilogue.
 *
 * Supports both the 2.x and 3.x APIs based on whether the first type is
 * a cute::tuple<> or not.
 * 2.x API implementation: cutlass/gemm/kernel/gemm_universal.h
 * 3.x API implementation: cutlass/gemm/kernel/gemm_*.hpp
 *
 * In the following declaration, the name preceding the 'Or' refers to
 * 3.x API type argument order, and the name succeeding the 'Or' refers to
 * 2.x API type argument order. Template arguments without two names
 * belong to the 3.x API only.
 **/
template <class ProblemShapeOrThreadblockMma_, // (m, n, k) or (m, n, k, l)
    class CollectiveMainloopOrEpilogue_, class CollectiveEpilogueOrThreadblockSwizzle_, class TileScheduler_ = void,
    class Enable = void>
class GemmUniversalGated;

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel

////////////////////////////////////////////////////////////////////////////////

#include "cutlass_extensions/gemm/kernel/sm90_gemm_gated_tma_warpspecialized_cooperative.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_gemm_gated_tma_warpspecialized_pingpong.hpp"
////////////////////////////////////////////////////////////////////////////////
