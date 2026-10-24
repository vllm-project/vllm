// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Dense FP8 MQA scoring of one 8-row block over its whole KV range: DeepGEMM's
// SM100 MQA logits pipeline (contiguous KV, 32 heads, D = 128) with the logits
// store replaced by a per-score policy. Used by the <= 4K resident kernel and
// by the persistent kernel's exact recovery rescan.
//
// Roles (384 threads): warps 0-7 reduce the 32 heads (4 rows per warpgroup),
// warp 8 loads Q and the weights, warp 9 the KV tiles and their scales, warp 10
// issues the UMMAs, warp 11 is idle.
//
// Policy interface (all called by the math warps):
//   kWeightScale, kScoreScale     score = (sum_h kWeightScale * w_h * (s_h +
//   |s_h|)) * kScoreScale * scale_kv begin_block(sched, q_block, query_base)
//   after the block's Q / weights landed begin_split(split, query_base) before
//   the split's scores score(i, kv_offset, kv_lane, split, score)    row
//   query_base + i of the warpgroup end_split(split, num_splits, kv_base) after
//   the split's scores end_block(sched, q_block)                     before the
//   block's Q stage is released
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

#include "deep_gemm_compat.cuh"

namespace fused_mqa_dense {
using namespace deep_gemm;

constexpr uint32_t kNumHeads = 32, kHeadDim = 128, BLOCK_Q = 8, SPLIT_KV = 256;
constexpr uint32_t kNumKVStages = 2, kNumTmemStages = 2;
constexpr uint32_t UMMA_M = 128, UMMA_N = BLOCK_Q * kNumHeads, UMMA_K = 32;
using Scheduler =
    fused_mqa_dg::SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, 4096>;
using SharedStorage = fused_mqa_dg::MQALogitsSharedStorage<
    kNumHeads, kHeadDim, false, BLOCK_Q, SPLIT_KV, 1, kNumKVStages,
    kNumTmemStages, cutlass::float_e4m3_t, float>;

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
    init(smem.empty_q_barriers[0], 256 + 32);
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      init(smem.full_kv_barriers[i], 1);
      init(smem.empty_kv_barriers[i], 256);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumTmemStages; ++i) {
      init(smem.full_tmem_barriers[i], 1);
      init(smem.empty_tmem_barriers[i], 256);
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
  cudaGridDependencySynchronize();

  uint32_t q_block_idx, kv_base, num_kv_splits;
  if (warp_idx == kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<56>();
    if (cute::elect_one_sync()) {
      Scheduler scheduler = block;
      while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);
        const uint32_t q_token_base =
            scheduler.get_q_tma_token_base(q_block_idx);
        fused_mqa_dg::tma_copy<kHeadDim, BLOCK_Q * kNumHeads, 0>(
            &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
            smem.smem_q[q_stage_idx], 0, q_token_base * kNumHeads);
        fused_mqa_dg::tma_copy<kNumHeads, BLOCK_Q, 0>(
            &tensor_map_weights, &smem.full_q_barriers[q_stage_idx],
            smem.smem_weights[q_stage_idx], 0, q_token_base);
        smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(
            SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
      }
    }
    __syncwarp();
  } else if (warp_idx == kSpecWarpStart + 1) {
    cutlass::arch::warpgroup_reg_dealloc<56>();
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
          fused_mqa_dg::tma_copy<kHeadDim, SPLIT_KV, 0>(
              &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
              smem.smem_kv[kv_stage_idx], 0, kv_tma_offset);
          fused_mqa_dg::tma_copy<SPLIT_KV, 1, 0>(
              &tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
              smem.smem_sf_kv[kv_stage_idx], kv_tma_offset, 0);
          smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(
              SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
        }
        __syncwarp();
      }
    }
  } else if (warp_idx == kSpecWarpStart + 2) {
    cutlass::arch::warpgroup_reg_dealloc<56>();
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
    cutlass::arch::warpgroup_reg_dealloc<56>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<224>();
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
        policy.begin_split(kv_split_idx, query_base);
        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
#pragma unroll
        for (uint32_t kv_half = 0; kv_half < 2; ++kv_half) {
          const uint32_t kv_lane = (math_thread_idx % 128) + kv_half * 128;
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
            // relu(s) = (s + |s|) / 2: two FFMA2 chains per row.
            float2 sum_0[2] = {};
            float2 sum_1[2] = {};
#pragma unroll
            for (uint32_t j = 0; j < kNumHeads; j += 4) {
#pragma unroll
              for (uint32_t r = 0; r < 2; ++r) {
                const float* a = accum_pair[r];
                auto transform = [&](uint32_t at, float2 sum) {
                  const auto x = make_float2(a[at], a[at + 1]);
                  const auto abs_x =
                      make_float2(fabsf(a[at]), fabsf(a[at + 1]));
                  const auto w = make_float2(weights[pair + r][at],
                                             weights[pair + r][at + 1]);
                  return __ffma2_rn(__fadd2_rn(x, abs_x), w, sum);
                };
                sum_0[r] = transform(j, sum_0[r]);
                sum_1[r] = transform(j + 2, sum_1[r]);
              }
            }
#pragma unroll
            for (uint32_t r = 0; r < 2; ++r) {
              const auto sum = __fadd2_rn(sum_0[r], sum_1[r]);
              policy.score(pair + r, kv_offset, kv_lane, kv_split_idx,
                           (sum.x + sum.y) * (Policy::kScoreScale * scale_kv));
            }
          }
        }
        policy.end_split(kv_split_idx, num_kv_splits, kv_base);
      }
      policy.end_block(scheduler, q_block_idx);
      cutlass::arch::fence_view_async_shared();
      smem.empty_q_barriers[q_stage_idx].arrive();
    }
    cutlass::arch::NamedBarrier(256, 0).sync();
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
}

}  // namespace fused_mqa_dense
