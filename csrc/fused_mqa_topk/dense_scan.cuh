// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Dense FP8 MQA scoring of one 8-row block over its whole KV range: DeepGEMM's
// SM100 MQA logits pipeline (contiguous KV, 32 heads, D = 128) with the logits
// store replaced by a per-score policy. Used by the persistent kernel's exact
// recovery rescan (rescan.cuh).
//
// Roles (384 threads): warps 0-7 reduce the 32 heads (4 rows per warpgroup),
// warp 8 loads Q and the weights, warp 9 the KV tiles and their scales, warp 10
// issues the UMMAs, warp 11 is idle.
//
// Policy interface (all called by the math warps):
//   kWeightScale: score = (sum_h kWeightScale * w_h * (s_h + |s_h|)) * scale_kv
//   begin_block(sched, q_block, query_base): after the block's Q / weights
//       landed
//   begin_split(split): before the split's scores
//   score(i, kv_offset, kv_lane, split, score): row query_base + i of the
//       warpgroup
//   end_split(split, num_splits, kv_base): after the split's scores
//   end_block(sched, q_block): before the block's Q stage is released
#pragma once

#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

#include "common.cuh"
#include "constants.h"
#include "deep_gemm_compat.cuh"

namespace fused_mqa_topk::dense {
using namespace deep_gemm;

constexpr uint32_t BLOCK_Q = kBlockRows, SPLIT_KV = kTileKeys;
constexpr uint32_t kNumKVStages = 2, kNumTmemStages = 2;
constexpr uint32_t UMMA_M = 128, UMMA_N = BLOCK_Q * kNumHeads, UMMA_K = 32;
// The rescan scores exactly one 8-row block: the scheduler is built with that
// block as its `sm_idx`, and its grid stride (kNumSMs) exceeds any block count
// (rows < 2^26), so next_q_block yields that block only. A smaller stride would
// also score block + stride, outside the rescan's row bounds and output rows.
constexpr uint32_t kSingleBlockStride = 1u << 31;
using Scheduler =
    dg::SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kSingleBlockStride>;
using SharedStorage =
    dg::MQALogitsSharedStorage<kNumHeads, kHeadDim, BLOCK_Q, SPLIT_KV, 1,
                               kNumKVStages, kNumTmemStages,
                               cutlass::float_e4m3_t, float>;

// The policy's state lives in dynamic shared memory right after the pipeline
// storage. Barrier words are zeroed before init: the persistent kernel reuses
// this memory between rescans.
template <typename Policy>
CUTLASS_DEVICE void score_block(const Scheduler& block, Policy& policy,
                                const cute::TmaDescriptor& tensor_map_q,
                                const cute::TmaDescriptor& tensor_map_kv,
                                const cute::TmaDescriptor& tensor_map_sf_kv,
                                const cute::TmaDescriptor& tensor_map_weights) {
  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = 8;
  constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_Q * kNumHeads * kHeadDim;
  constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = SPLIT_KV * kHeadDim;
  constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = SPLIT_KV * sizeof(float);
  constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * sizeof(float);
  constexpr uint32_t kNumTmemCols =
      utils::get_num_aligned_tmem_cols<UMMA_N * kNumTmemStages>();

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
  }

  extern __shared__ __align__(SharedStorage::kSwizzleAlignment)
      uint8_t smem_buffer[];
  auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
  if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
    const auto init = [](auto& barrier, uint32_t count) {
      *reinterpret_cast<volatile uint64_t*>(&barrier) = 0;
      barrier.init(count);
    };
    init(smem.full_q_barriers[0], 1);
    init(smem.empty_q_barriers[0], kMathThreads + 32);  // + the UMMA warp
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      init(smem.full_kv_barriers[i], 1);
      init(smem.empty_kv_barriers[i], kMathThreads);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumTmemStages; ++i) {
      init(smem.full_tmem_barriers[i], 1);
      init(smem.empty_tmem_barriers[i], kMathThreads);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncwarp();
  if (warp_idx == kSpecWarpStart + 2)
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
  __syncthreads();

  RingPipeline<1> q_pipeline;
  RingPipeline<kNumKVStages> kv_pipeline;
  RingPipeline<kNumTmemStages> tmem_pipeline;

  uint32_t q_block_idx, kv_base, num_kv_splits;
  if (warp_idx == kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
    if (cute::elect_one_sync()) {
      Scheduler scheduler = block;
      while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);
        const uint32_t q_token_base =
            scheduler.get_q_tma_token_base(q_block_idx);
        dg::tma_copy<kHeadDim, BLOCK_Q * kNumHeads, 0>(
            &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
            smem.smem_q[q_stage_idx], 0, q_token_base * kNumHeads);
        dg::tma_copy<kNumHeads, BLOCK_Q, 0>(
            &tensor_map_weights, &smem.full_q_barriers[q_stage_idx],
            smem.smem_weights[q_stage_idx], 0, q_token_base);
        smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(
            SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
      }
    }
    __syncwarp();
  } else if (warp_idx == kSpecWarpStart + 1) {
    cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
    Scheduler scheduler = block;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
#pragma unroll 1
      for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits;
           ++kv_split_idx) {
        if (cute::elect_one_sync()) {
          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
          const uint32_t kv_tma_offset =
              scheduler.get_kv_tma_offset(kv_base, kv_split_idx);
          dg::tma_copy<kHeadDim, SPLIT_KV, 0>(
              &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
              smem.smem_kv[kv_stage_idx], 0, kv_tma_offset);
          dg::tma_copy<SPLIT_KV, 1, 0>(
              &tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
              smem.smem_sf_kv[kv_stage_idx], kv_tma_offset, 0);
          smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(
              SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
        }
        __syncwarp();
      }
    }
  } else if (warp_idx == kSpecWarpStart + 2) {
    cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr_in_smem) == 0);
    auto instr_desc = cute::UMMA::make_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, UMMA_M, UMMA_N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
    Scheduler scheduler = block;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
      smem.full_q_barriers[q_stage_idx].wait(q_phase);
      for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits;
           ++kv_split_idx) {
        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
        if (cute::elect_one_sync()) {
          // One UMMA per math warpgroup: KV rows [128 i, 128 i + 128) x all 256
          // Q rows.
#pragma unroll
          for (uint32_t i = 0; i < 2; ++i) {
            CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
            const uint32_t tmem_addr = tmem_stage_idx * UMMA_N;
            smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
            ptx::tcgen05_after_thread_sync();
#pragma unroll
            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
              auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                       kHeadDim, kHeadDim>(
                  smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
              auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                       kHeadDim, kHeadDim>(
                  smem.smem_q[q_stage_idx], 0, k * UMMA_K);
              ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, k,
                                            runtime_instr_desc);
            }
            ptx::umma_arrive_no_elect(smem.full_tmem_barriers[tmem_stage_idx]);
          }
        }
        __syncwarp();
      }
      smem.empty_q_barriers[q_stage_idx].arrive();
    }
  } else if (warp_idx == kSpecWarpStart + 3) {
    cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<kMathRegs>();
    const uint32_t math_thread_idx = warp_idx * 32 + lane_idx;
    const uint32_t query_base = (warp_idx / 4) * 4;
    float weights[4][kNumHeads];
    Scheduler scheduler = block;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
      smem.full_q_barriers[q_stage_idx].wait(q_phase);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; ++j)
          weights[i][j] = Policy::kWeightScale *
                          ptx::ld_shared(smem.smem_weights[q_stage_idx] +
                                         (query_base + i) * kNumHeads + j);
      policy.begin_block(scheduler, q_block_idx, query_base);
      for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits;
           ++kv_split_idx) {
        policy.begin_split(kv_split_idx);
        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
#pragma unroll
        for (uint32_t kv_half = 0; kv_half < 2; ++kv_half) {
          const uint32_t kv_lane =
              (math_thread_idx % kMathWgThreads) + kv_half * kMathWgThreads;
          const uint32_t kv_offset =
              scheduler.get_logits_col(kv_base, kv_split_idx, kv_lane);
          const float scale_kv =
              ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + kv_lane);
          CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
          smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
          ptx::tcgen05_after_thread_sync();
          // Both KV half-MMAs have committed before the shared stage is freed.
          if (kv_half == 1) {
            cutlass::arch::fence_view_async_shared();
            smem.empty_kv_barriers[kv_stage_idx].arrive();
          }
          float accum_pair[2][kNumHeads];
#pragma unroll
          for (uint32_t pair = 0; pair < 4; pair += 2) {
            // Two query rows at a time keep the live accumulator registers low.
            ptx::tmem_load_32dp32b<2 * kNumHeads>(
                tmem_stage_idx * UMMA_N + (query_base + pair) * kNumHeads,
                reinterpret_cast<uint32_t*>(accum_pair));
            cutlass::arch::fence_view_async_tmem_load();
            if (pair == 2) {
              ptx::tcgen05_before_thread_sync();
              smem.empty_tmem_barriers[tmem_stage_idx].arrive();
            }
#pragma unroll
            for (uint32_t r = 0; r < 2; ++r)
              policy.score(
                  pair + r, kv_offset, kv_lane, kv_split_idx,
                  reduce_heads<true>(accum_pair[r], weights[pair + r]) *
                      scale_kv);
          }
        }
        policy.end_split(kv_split_idx, num_kv_splits, kv_base);
      }
      policy.end_block(scheduler, q_block_idx);
      cutlass::arch::fence_view_async_shared();
      smem.empty_q_barriers[q_stage_idx].arrive();
    }
    cutlass::arch::NamedBarrier(kMathThreads, kBarDenseMath).sync();
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
}

// Invalidates the pipeline's barriers so the next score_block can re-initialise
// the same shared memory (mbarrier.inval before reuse).
CUTLASS_DEVICE void invalidate_barriers(SharedStorage& smem) {
  for (auto& b : smem.full_q_barriers) mbar_inval(&b);
  for (auto& b : smem.empty_q_barriers) mbar_inval(&b);
  for (auto& b : smem.full_kv_barriers) mbar_inval(&b);
  for (auto& b : smem.empty_kv_barriers) mbar_inval(&b);
  for (auto& b : smem.full_tmem_barriers) mbar_inval(&b);
  for (auto& b : smem.empty_tmem_barriers) mbar_inval(&b);
}

}  // namespace fused_mqa_topk::dense
