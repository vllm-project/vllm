/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*
 * Taken from SGLANG PR https://github.com/sgl-project/sglang/pull/6929
 * by Alcanderian JieXin Liang
 */

// clang-format off
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cute/tensor.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;
template<
    class ElementOut,
    class ElementAcc,
    class ElementScale,
    size_t kNumHeads,
    size_t kHeadDimLatent,
    int kMaxSplits
>
struct Sm100FmhaMlaReductionKernel {

  static const int SharedStorageSize = 0;
  static const int MaxThreadsPerBlock = 128;
  static const int MinBlocksPerMultiprocessor = 1;

  using ArchTag = cutlass::arch::Sm100;

  static_assert(kHeadDimLatent % MaxThreadsPerBlock == 0);
  struct Arguments {
    ElementAcc* ptr_oaccum = nullptr;
    ElementOut* ptr_o = nullptr;
    ElementAcc* ptr_lseaccum = nullptr;
    ElementAcc* ptr_lse = nullptr;
    ElementScale scale = 1.f;
    int num_batches = 0;
    int split_kv = -1;
    int dim_k = -1;
    int* ptr_seq = nullptr;
    int* ptr_split_kv = nullptr;
    int tile_shape_s = 128;
  };
  using Params = Arguments;

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {args.ptr_oaccum, args.ptr_o, args.ptr_lseaccum, args.ptr_lse,
	    args.scale, args.num_batches, args.split_kv, args.dim_k, args.ptr_seq,
	    args.ptr_split_kv, args.tile_shape_s};
  }

  static size_t get_workspace_size(Arguments const& /*args*/) {
    return 0;
  }

  static Status initialize_workspace(
      Arguments const& /*args*/, void* /*ws*/, cudaStream_t /*stream*/) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return dim3(kNumHeads, 1, params.num_batches);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  static bool can_implement(Arguments const& args) {
    if (args.num_batches <= 0) return false;
    if (args.split_kv <= 0) return false;
    return true;
  }

  CUTLASS_DEVICE void operator() (Params const& params, char* smem_raw) {
    if (params.split_kv <= 1) return;
    auto blk_coord = make_coord(blockIdx.x, _0{}, blockIdx.z);

    __shared__ ElementAcc sLseScale[kMaxSplits];
    const size_t offset_lseaccum = get<0>(blk_coord) + kNumHeads * params.split_kv * get<2>(blk_coord);
    const size_t offset_lse = get<0>(blk_coord) + kNumHeads * get<2>(blk_coord);

    Tensor gLSEaccum = make_tensor(make_gmem_ptr(params.ptr_lseaccum + offset_lseaccum),
                                   make_shape(params.split_kv), Stride<Int<kNumHeads>>{});

    Tensor gLSE = make_tensor(make_gmem_ptr(params.ptr_lse + offset_lse),
                              Shape<_1>{}, Stride<_1>{});

    auto dim_k = params.ptr_seq == nullptr ?  params.dim_k : params.ptr_seq[get<2>(blk_coord)];
    auto local_split_kv = params.ptr_split_kv == nullptr ? params.split_kv : params.ptr_split_kv[get<2>(blk_coord)];
    auto k_tile_total = ceil_div(dim_k, params.tile_shape_s);
    auto k_tile_per_cta = ceil_div(k_tile_total, local_split_kv);
    local_split_kv = ceil_div(k_tile_total, k_tile_per_cta);

    int warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == 0) {
      constexpr int kNLsePerThread = cute::ceil_div(kMaxSplits, 32);

      ElementAcc local_lse[kNLsePerThread];

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNLsePerThread; ++i) {
        const int split = i * 32 + threadIdx.x;
        local_lse[i] = split < local_split_kv ? gLSEaccum(split) : -std::numeric_limits<ElementAcc>::infinity();
      }

      ElementAcc lse_max = -std::numeric_limits<ElementAcc>::infinity();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNLsePerThread; ++i) {
        lse_max = max(lse_max, local_lse[i]);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int offset = 16; offset >= 1; offset /= 2) {
        lse_max = max(lse_max, __shfl_xor_sync(0xffffffff, lse_max, offset));
      }
      lse_max = lse_max == -std::numeric_limits<ElementAcc>::infinity() ? 0.0f : lse_max;  // In case all local LSEs are -inf
      lse_max = __shfl_sync(0xffffffff, lse_max, 0);

      ElementAcc sum_lse = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNLsePerThread; ++i) {
        sum_lse = sum_lse + expf(local_lse[i] - lse_max);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int offset = 16; offset >= 1; offset /= 2) {
        sum_lse = sum_lse + __shfl_xor_sync(0xffffffff, sum_lse, offset);
      }

      sum_lse = __shfl_sync(0xffffffff, sum_lse, 0);

      ElementAcc global_lse = (sum_lse == 0.f || sum_lse != sum_lse) ? std::numeric_limits<ElementAcc>::infinity() : logf(sum_lse) + lse_max;
      if (threadIdx.x == 0 and params.ptr_lse != nullptr) {
        gLSE(0) = global_lse;
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kNLsePerThread; ++i) {
        const int split = i * 32 + threadIdx.x;
        if (split < local_split_kv) {
          sLseScale[split] = expf(local_lse[i] - global_lse);
        }
      }
    }
    __syncthreads();

    constexpr int Elements = kHeadDimLatent / MaxThreadsPerBlock;
    const size_t offset_oaccum = kHeadDimLatent * params.split_kv * (get<0>(blk_coord) + kNumHeads * get<2>(blk_coord));
    Tensor gOaccum = make_tensor(make_gmem_ptr(params.ptr_oaccum + offset_oaccum),
                               Shape<Int<kHeadDimLatent>>{}, Stride<_1>{});
    ElementAcc local_val[Elements] = {0};
    for (int split = 0; split < local_split_kv; ++split) {
      ElementAcc lse_scale = sLseScale[split];
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < Elements; ++i) {
        local_val[i] += lse_scale * gOaccum(threadIdx.x + MaxThreadsPerBlock * i);
      }
      gOaccum.data() = gOaccum.data() + kHeadDimLatent;
    }
    auto ptr_o_local = params.ptr_o + (get<0>(blk_coord) + get<2>(blk_coord) * kNumHeads) * kHeadDimLatent;
    Tensor gO = make_tensor(make_gmem_ptr(ptr_o_local), Shape<Int<kHeadDimLatent>>{}, Stride<_1>{});

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < Elements; ++i) {
      gO(threadIdx.x + MaxThreadsPerBlock * i) = static_cast<ElementOut>(local_val[i]);
    }
  }
};

}  // namespace cutlass::fmha::kernel
