// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Derived from DeepGEMM commit 891d57b4db1071624b5c8fa0d1e51cb317fa709f;
// see LICENSE.deepseek-deepgemm.

#pragma once

#include <cuda_fp16.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace dsa_litetopk {

using namespace deep_gemm;

inline constexpr uint32_t kEmitChunkBlocks = 256;
inline constexpr uint32_t kEmitLaneSlots = 18;
inline constexpr uint32_t kMathRegisters = 240;
inline constexpr uint32_t kSpecializedRegisters = 24;
inline constexpr uint32_t kUmmaStages = 2;
// A ring slot can only equal this pattern if a +NaN score passed the gate,
// which both gate comparisons reject; the refresher treats it as "empty".
inline constexpr uint32_t kRingSentinel = 0xffffffffu;
// Pace refreshes against the 65,536-token flush window.
inline constexpr uint32_t kRingActiveNs = 2048;
inline constexpr uint32_t kRingIdleNs = 2048;

// Six-byte ABI: cand_val holds low 16 score bits; cand_idx holds high 8 + KV[19:0].
// Scores drop FP32's low 8 bits; distinct scores can tie. The ring packs the
// resulting high24 key with an 8-bit block-in-window coordinate.
using CandidateValue = uint16_t;
constexpr uint32_t kCandidateIndexBits = 20;
constexpr uint32_t kCandidateIndexMask = (1u << kCandidateIndexBits) - 1u;

CUTLASS_DEVICE uint32_t candidate_pack_index(const uint32_t payload,
                                             const uint32_t kv_index) {
  return (kv_index & kCandidateIndexMask) |
         ((payload >> 16) << kCandidateIndexBits);
}

CUTLASS_DEVICE uint32_t candidate_ordered_fp32_code(const float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

CUTLASS_DEVICE uint32_t candidate_fp24_code(const float value) {
  return candidate_ordered_fp32_code(value) >> 8;
}

CUTLASS_DEVICE uint32_t candidate_load_score_code(const CandidateValue value,
                                                  const int32_t packed_idx) {
  return (static_cast<uint32_t>(packed_idx) >> kCandidateIndexBits) << 16 |
         static_cast<uint32_t>(value);
}

CUTLASS_DEVICE void store_candidate(CandidateValue* value_dst,
                                    int32_t* index_dst, const float bq,
                                    const uint32_t kv_index) {
  const uint32_t payload = candidate_fp24_code(bq);
  __stcs(value_dst, static_cast<CandidateValue>(payload));
  __stcs(index_dst,
         static_cast<int32_t>(candidate_pack_index(payload, kv_index)));
}

CUTLASS_DEVICE void store_candidate_payload(CandidateValue* value_dst,
                                            int32_t* index_dst,
                                            const uint32_t payload,
                                            const uint32_t kv_index) {
  __stcs(value_dst, static_cast<CandidateValue>(payload));
  __stcs(index_dst,
         static_cast<int32_t>(candidate_pack_index(payload, kv_index)));
}

// All active lanes address the same epoch/exit word. One atomic read avoids
// serializing 32 shared atomics while retaining an explicit atomic protocol.
CUTLASS_DEVICE uint32_t warp_atomic_load(uint32_t* ptr) {
  const uint32_t mask = __activemask();
  const int leader = __ffs(mask) - 1;
  uint32_t value = 0;
  if (ptx::get_lane_idx() == leader) value = atomicAdd(ptr, 0u);
  return __shfl_sync(mask, value, leader);
}

template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t BLOCK_Q,
          uint32_t BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs, uint32_t kNumSpecializedThreads,
          uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(
    kNumSpecializedThreads + kNumMathThreads,
    1) void sm100_dsa_litetopk(const uint32_t seq_len,
                               const uint32_t seq_len_kv,
                               uint32_t* cu_seq_len_k_start,
                               uint32_t* cu_seq_len_k_end,
                               const float* __restrict__ origin,  // [seq_len]
                               const float* __restrict__ inv_delta,  // [seq_len]
                               int32_t* __restrict__ th_bucket,  // [seq_len]
                               int32_t* __restrict__ bcount,     // [seq_len,
                                                                 // num_buckets]
                               const uint32_t num_buckets, const uint32_t topk,
                               CandidateValue* __restrict__ cand_val,  // [seq_len,
                                                                       // cand_cap]
                               int32_t* __restrict__ cand_idx,  // [seq_len,
                                                                // cand_cap]
                               int32_t* __restrict__ cand_cnt,  // [seq_len]
                               const uint32_t cand_cap,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_q,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_kv,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_kv_scales,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_weights,
                               const int32_t* block_table,
                               const int32_t* page_order) {
  constexpr uint32_t kPageSize = 64;
  const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto warpgroup_idx = warp_idx / 4;
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
  constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
  constexpr uint32_t kNumUmmaStages = kUmmaStages;
  constexpr uint32_t kNumUmmaBuffers = kNumMathWarpGroups * kNumUmmaStages;
  constexpr uint32_t kNumAccumBufs = kNumUmmaBuffers;
  // Frequent gate reloads bound emissions made against a stale threshold.
  constexpr uint32_t kActiveGateStride = 8;
  constexpr uint32_t kActiveEmitChunkBlocks = kEmitChunkBlocks;
  DG_STATIC_ASSERT(
      kNumHeads == 32 && BLOCK_Q == 4 && kNumMathWarps == 8,
      "GLM LiteTopK requires H=32, BLOCK_Q=4 and 8 math warps");
  DG_STATIC_ASSERT(
      BLOCK_KV == 256,
      "the local ring infers the KV tile coordinate from its owner lane");
  DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0,
                   "Invalid threads");

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_kv_scales);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
  }

  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * kHeadDim;
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * sizeof(float);
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE =
      BLOCK_KV * kHeadDim;
  static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE =
      BLOCK_KV * sizeof(float);
  static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE =
      math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u);

  extern __shared__ __align__(512) uint8_t smem_buffer[];
  DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_WEIGHT_SIZE_PER_STAGE % 512 == 0,
                   "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0,
                   "Unaligned TMA swizzling");

  constexpr uint32_t kNumAccumTmemCols = BLOCK_Q * kNumHeads * kNumAccumBufs;
  constexpr uint32_t kNumTmemCols = kNumAccumTmemCols;
  DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

  auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer +
                                            SMEM_Q_SIZE_PER_STAGE * i);
  });
  auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<float*>(smem_buffer +
                                    SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                                    SMEM_WEIGHT_SIZE_PER_STAGE * i);
  });
  auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<__nv_fp8_e4m3*>(
        smem_buffer + (SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                       SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
                       SMEM_KV_SIZE_PER_STAGE * i));
  });
  auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<float*>(smem_buffer +
                                    SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                                    SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
                                    SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
                                    ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
  });

  auto barrier_ptr = reinterpret_cast<Barrier*>(smem_kv_scales[kNumKVStages]);
  auto full_q_barriers =
      utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
  auto empty_q_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return barrier_ptr + (kNumQStages + i); });
  auto full_kv_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return barrier_ptr + (kNumQStages * 2 + i); });
  auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i);
  });
  auto full_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i);
  });
  auto empty_umma_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr +
           (kNumQStages * 2 + kNumKVStages * 2 + kNumAccumBufs + i);
  });

  auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
      barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumAccumBufs * 2 +
      kNumKVStages);
  auto scan_done_flag = reinterpret_cast<uint32_t*>(tmem_ptr_in_smem + 1);
  auto warpq_count = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 4);
  // Reuse queue scratch for gate bits; unsigned atomicMin tightens positive edges.
  auto sparse_gate_bits = reinterpret_cast<uint32_t*>(warpq_count);
  auto emit_smem_records =
      reinterpret_cast<uint32_t*>(warpq_count + kNumMathWarps * BLOCK_Q);
  auto smem_hist = reinterpret_cast<int32_t*>(
      emit_smem_records + kNumMathWarps * BLOCK_Q * kEmitLaneSlots * 32u);
  // Place flush epochs, shadow epochs, and consumed-slot cursors after the histogram.
  auto ring_seq = reinterpret_cast<uint32_t*>(smem_hist + BLOCK_Q * 256);
  auto ring_shadow = ring_seq + kNumMathWarps * BLOCK_Q;
  auto ring_pending =
      reinterpret_cast<int32_t*>(ring_shadow + kNumMathWarps * BLOCK_Q);
  auto ring_stale = ring_pending + BLOCK_Q;
  auto ring_last = ring_stale + BLOCK_Q;
  auto ring_progress = reinterpret_cast<uint8_t*>(ring_last + BLOCK_Q);

  DG_STATIC_ASSERT(
      kNumSpecializedThreads % 128 == 0 and kNumSpecializedThreads >= 64,
      "Invalid threads");
  if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      full_q_barriers[i]->init(1);
      empty_q_barriers[i]->init(kNumMathThreads + 32);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      full_kv_barriers[i]->init(1);
      empty_kv_barriers[i]->init(kNumMathThreads);
    }
    *scan_done_flag = 0;
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == kSpecWarpStart + 1) {
    if (cute::elect_one_sync()) {
#pragma unroll
      for (uint32_t i = 0; i < kNumAccumBufs; ++i) {
        full_umma_barriers[i]->init(1);
        empty_umma_barriers[i]->init(128);
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
  }
  {
    for (uint32_t idx = threadIdx.x;
         idx < kNumMathWarps * BLOCK_Q * kEmitLaneSlots * 32u;
         idx += blockDim.x) {
      emit_smem_records[idx] = kRingSentinel;
    }
    // Seed and harvested suffix records are disjoint and share the bucket scale,
    // so their histogram remains a subset of the row and yields a safe gate.
    for (uint32_t idx = threadIdx.x; idx < BLOCK_Q * num_buckets;
         idx += blockDim.x) {
      const uint32_t r = idx / num_buckets;
      const uint32_t row_q = static_cast<uint32_t>(blockIdx.x) * BLOCK_Q + r;
      smem_hist[idx] =
          row_q < seq_len
              ? __ldcg(bcount + static_cast<uint64_t>(row_q) * num_buckets +
                       (idx - r * num_buckets))
              : 0;
    }
    for (uint32_t idx = threadIdx.x; idx < kNumMathWarps * BLOCK_Q;
         idx += blockDim.x) {
      ring_seq[idx] = 0u;
      ring_shadow[idx] = 0u;
    }
    for (uint32_t idx = threadIdx.x; idx < BLOCK_Q; idx += blockDim.x) {
      ring_pending[idx] = 0;
      ring_last[idx] = 0;
      ring_stale[idx] = blockIdx.x * BLOCK_Q + idx < seq_len ? 0 : 2;
    }
    for (uint32_t idx = threadIdx.x; idx < kNumMathWarps * BLOCK_Q * 32u;
         idx += blockDim.x) {
      ring_progress[idx] = 0u;
    }
  }
  if (threadIdx.x < BLOCK_Q) {
    const uint32_t row_q =
        static_cast<uint32_t>(blockIdx.x) * BLOCK_Q + threadIdx.x;
    const uint32_t row = min(row_q, seq_len - 1);
    const int gate = __ldcg(th_bucket + row);
    ptx::st_shared(sparse_gate_bits + threadIdx.x,
                   __float_as_uint(static_cast<float>(gate + 1)));
  }
  __syncthreads();

  constexpr uint32_t kNumSpecializedRegisters = kSpecializedRegisters;
  constexpr uint32_t kNumMathRegisters = kMathRegisters;

  const uint32_t block_q_idx = blockIdx.x;
  uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
  const auto load_schedule =
      [&](const uint32_t block_q_idx) -> cute::tuple<uint32_t, uint32_t> {
    uint32_t start = cute::numeric_limits<uint32_t>::max();
    uint32_t end = cute::numeric_limits<uint32_t>::min();

#pragma unroll
    for (uint32_t i = 0; i < BLOCK_Q; ++i) {
      const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
      seq_k_start[i] = cu_seq_len_k_start[q_idx];
      seq_k_end[i] = cu_seq_len_k_end[q_idx];
      if (block_q_idx * BLOCK_Q + i >= seq_len) {
        // Padded row of a ragged final q-block: empty, aggregation-neutral.
        seq_k_start[i] = seq_len_kv;
        seq_k_end[i] = 0;
      }
      start = min(start, min(seq_k_start[i], seq_len_kv));
      end = max(end, min(seq_k_end[i], seq_len_kv));
    }
    start = start / 4 * 4;  // TMA alignment for KV scales
    const uint32_t nkv =
        (end > start) ? math::ceil_div(end - start, BLOCK_KV) : 0;
    return {start, nkv};
  };

  const auto get_kv_pipeline =
      [&](const uint32_t& kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
    return {kv_block_idx % kNumKVStages, (kv_block_idx / kNumKVStages) & 1};
  };

  constexpr uint32_t UMMA_M = 128;
  constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
  constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;

  if (warp_idx == kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    if (block_q_idx < num_q_blocks) {
      constexpr uint32_t kKvInnerBytes = kHeadDim;
      if (cute::elect_one_sync()) {
        tma::copy<kKvInnerBytes, BLOCK_Q * kNumHeads, kKvInnerBytes>(
            &tensor_map_q, full_q_barriers[0], smem_q[0], 0,
            block_q_idx * BLOCK_Q * kNumHeads);
        tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights,
            full_q_barriers[0], smem_weights[0], 0, block_q_idx * BLOCK_Q);
        full_q_barriers[0]->arrive_and_expect_tx(
            SMEM_Q_SIZE_PER_STAGE + SMEM_WEIGHT_SIZE_PER_STAGE);
      }
      CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
      uint32_t cached_base = 0xffffffffu, cached_physical = 0;
      for (uint32_t block = 0; block < num_kv_blocks; ++block) {
        const uint32_t first_page = (kv_start + block * BLOCK_KV) / kPageSize;
        const uint32_t base = first_page / 32 * 32;
        if (base != cached_base) {
          const uint32_t page = base + lane_idx;
          cached_physical = page * kPageSize < seq_len_kv
              ? block_table[page_order[page]] : 0;
          cached_base = base;
        }
        uint32_t physical[BLOCK_KV / kPageSize];
#pragma unroll
        for (uint32_t p = 0; p < BLOCK_KV / kPageSize; ++p)
          physical[p] = __shfl_sync(0xffffffffu, cached_physical, first_page - base + p);
        CUTE_TIE_DECL(get_kv_pipeline(block), stage, phase);
        if (cute::elect_one_sync()) {
          empty_kv_barriers[stage]->wait(phase ^ 1);
#pragma unroll
          for (uint32_t p = 0; p < BLOCK_KV / kPageSize; ++p) {
            tma::copy<kKvInnerBytes, kPageSize, kKvInnerBytes, __nv_fp8_e4m3, true>(
                &tensor_map_kv, full_kv_barriers[stage],
                smem_kv[stage] + p * kPageSize * kHeadDim,
                0, 0, 1, physical[p]);
            tma::copy<kPageSize, 1, 0>(&tensor_map_kv_scales,
                full_kv_barriers[stage], smem_kv_scales[stage] + p * kPageSize,
                0, physical[p]);
          }
          full_kv_barriers[stage]->arrive_and_expect_tx(
              SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
        }
        __syncwarp();
      }
    }
  } else if (warp_idx == kSpecWarpStart + 1) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

    auto instr_desc = cute::UMMA::make_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, UMMA_M, UMMA_N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

    if (block_q_idx < num_q_blocks) {
      CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
      full_q_barriers[0]->wait(0);

      for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
           ++kv_block_idx) {
        const uint32_t kvg = kv_block_idx;
        CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);
        full_kv_barriers[kv_stage_idx]->wait(kv_phase);

        DG_STATIC_ASSERT(BLOCK_KV == kNumMathThreads, "Invalid block size");
        DG_STATIC_ASSERT(kHeadDim % UMMA_K == 0, "Invalid head dim");
        // Toggle phase when an accumulator stage is reused, not on every tile.
        const uint32_t umma_stage = kvg % kNumUmmaStages;
        const uint32_t umma_phase = (kvg / kNumUmmaStages) & 1;
#pragma unroll
        for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
          const uint32_t buf = i * kNumUmmaStages + umma_stage;
          empty_umma_barriers[buf]->wait(umma_phase ^ 1);
          ptx::tcgen05_after_thread_sync();
#pragma unroll
          for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
            auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                     kHeadDim, kHeadDim>(
                smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
            auto b_desc =
                mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim,
                                           kHeadDim>(smem_q[0], 0, k * UMMA_K);
            cute::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, buf * UMMA_N, k,
                                           runtime_instr_desc);
          }
          cutlass::arch::umma_arrive(
              reinterpret_cast<uint64_t*>(full_umma_barriers[buf]));
        }
      }
      empty_q_barriers[0]->arrive();
    }
  } else if (warp_idx == kSpecWarpStart + 2 or warp_idx == kSpecWarpStart + 3) {
    // Spare-warp threshold-refresh daemon. Keeping refresh off the math
    // warps avoids placing histogram scan latency on their critical path.
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    if (block_q_idx < num_q_blocks) {
      const uint32_t spare_id = warp_idx - (kSpecWarpStart + 2);  // 0 or 1
      {
        // Count a subset of emitted records to tighten the gate safely.
        // Flush seqlocks reject reads racing a drain, preventing double counts.
        uint32_t pass = 0;
        while (warp_atomic_load(scan_done_flag) == 0) {
          if (ring_stale[spare_id] >= 2 &&
              ring_stale[BLOCK_Q == 4 ? spare_id + 2 : spare_id] >= 2) {
            break;
          }
          const uint32_t local_row =
              BLOCK_Q == 4 ? spare_id + ((pass & 1u) << 1) : spare_id;
          ++pass;
          uint32_t fresh_total = 0;
          if (block_q_idx * BLOCK_Q + local_row < seq_len &&
              ring_stale[local_row] < 2) {
            for (uint32_t w = 0; w < kNumMathWarps; ++w) {
              const uint32_t pair = w * BLOCK_Q + local_row;
              const uint32_t s0 =
                  warp_atomic_load(ring_seq + pair);
              if (s0 != ring_shadow[pair]) {
                // New flush epoch: the drained column is
                // sentinel again and refills from zero.
                ring_progress[pair * 32u + lane_idx] = 0u;
                __syncwarp();
                if (lane_idx == 0) ring_shadow[pair] = s0;
                __syncwarp();
              }
              // Harvest the lane's full staged column each visit.
              uint32_t progress = ring_progress[pair * 32u + lane_idx];
              uint32_t took = 0;
              while (progress < kEmitLaneSlots) {
                const uint32_t record = atomicAdd(emit_smem_records +
                    (pair * kEmitLaneSlots + progress) * 32u + lane_idx, 0u);
                if (record == kRingSentinel) break;
                if (warp_atomic_load(ring_seq + pair) !=
                    s0) {
                  // Raced a flush drain: discard this
                  // read; the epoch reset above replays
                  // the refilled column next visit.
                  break;
                }
                // Decode high24 bucket values. Integer edges remain on the same
                // 0x100 grid, so truncation needs no gate padding. Comparisons
                // route NaN/out-of-range values to the non-publishable top bin.
                uint32_t fbits = record & 0xffffff00u;
                fbits = (fbits & 0x80000000u) ? (fbits ^ 0x80000000u) : ~fbits;
                const float value = __uint_as_float(fbits);
                uint32_t bucket;
                if (value < 0.0f) {
                  bucket = 0u;
                } else if (value < static_cast<float>(num_buckets - 1)) {
                  bucket = static_cast<uint32_t>(value);
                } else {
                  bucket = num_buckets - 1;
                }
                atomicAdd(smem_hist + local_row * num_buckets + bucket, 1);
                ++progress;
                ++took;
              }
              if (took != 0u) {
                ring_progress[pair * 32u + lane_idx] =
                    static_cast<uint8_t>(progress);
              }
              const uint32_t fresh = __reduce_add_sync(0xffffffffu, took);
              if (lane_idx == 0 && fresh != 0u) {
                ring_pending[local_row] += static_cast<int32_t>(fresh);
              }
              fresh_total += fresh;
            }
            __syncwarp();
            // The seed base already carries K records, so each
            // topk/16 batch of fresh evidence can move the
            // boundary.
            const int32_t pend = ring_pending[local_row];
            const int32_t last = ring_last[local_row];
            __syncwarp();
            if (pend - last >= static_cast<int32_t>(topk) / 16) {
              if (lane_idx == 0) ring_last[local_row] = pend;
              __syncwarp();
              // The first prefix reaching K yields a safe integer upper edge:
              // high24 truncation cannot cross these aligned bucket boundaries.
              int carry = 0;
              int found = -1;
              for (uint32_t bbase = 0; bbase < num_buckets && found < 0;
                   bbase += 32) {
                const uint32_t b = bbase + lane_idx;
                const int v = b < num_buckets
                                  ? smem_hist[local_row * num_buckets + b]
                                  : 0;
                const int group_sum = __reduce_add_sync(0xffffffffu, v);
                if (carry + group_sum < static_cast<int>(topk)) {
                  carry += group_sum;
                  continue;
                }
                int prefix = v;
#pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                  const int nsh = __shfl_up_sync(0xffffffffu, prefix, off);
                  if (static_cast<int>(lane_idx) >= off) prefix += nsh;
                }
                const bool hit = carry + prefix >= static_cast<int>(topk);
                const unsigned hm = __ballot_sync(0xffffffffu, hit);
                if (hm) {
                  found = static_cast<int>(bbase) + __ffs(hm) - 1;
                } else {
                  carry += __shfl_sync(0xffffffffu, prefix, 31);
                }
              }
              if (lane_idx == 0) {
                bool published = false;
                // The top bin collects NaN and clamped
                // out-of-range records, so an edge may only
                // be published from a strictly lower bin.
                if (found >= 0 && found < static_cast<int>(num_buckets) - 1) {
                  const uint32_t edge =
                      __float_as_uint(static_cast<float>(found + 1));
                  published =
                      edge < atomicMin(sparse_gate_bits + local_row, edge);
                }
                // A failed warm-start tightening must not retire the daemon.
                ring_stale[local_row] =
                    published ? 0
                              : (pend >= static_cast<int32_t>(topk)
                                     ? ring_stale[local_row] + 1
                                     : ring_stale[local_row]);
              }
              __syncwarp();
            }
          }
          // Quiet pacing. The CTA retires promptly once the math
          // warps finish; bursts skip pacing and chain straight
          // into the next pass.
          if (fresh_total < 64u) {
            uint32_t slept = 0;
            while (slept < kRingIdleNs && warp_atomic_load(scan_done_flag) == 0) {
              __nanosleep(kRingActiveNs);
              slept += kRingActiveNs;
            }
          }
        }
      }
    }
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

    const auto math_thread_idx = warp_idx * 32 + lane_idx;

    auto tmem_load = [](auto num_elems_c, const uint32_t& tmem_addr,
                        float* accum) {
      constexpr int N = decltype(num_elems_c)::value;
      DG_STATIC_ASSERT(N == 32 or N == 64 or N == 128,
                       "Unsupported TMEM load size");
      using Loader = cute::conditional_t<
          N == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
          cute::conditional_t<N == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
                              cute::SM100_TMEM_LOAD_32dp32b128x>>;
      [&]<size_t... Is>(cute::index_sequence<Is...>) {
        Loader::copy(tmem_addr, reinterpret_cast<uint32_t*>(accum)[Is]...);
      }(cute::make_index_sequence<N>{});
      cutlass::arch::fence_view_async_tmem_load();
    };

    // Thresholds and candidates must use the same affine bucket space.
    float weights[BLOCK_Q][kNumHeads];
    float o_reg[BLOCK_Q], inv_reg[BLOCK_Q], vth_reg[BLOCK_Q];
    uint32_t kstart_reg[BLOCK_Q],
        kspan_reg[BLOCK_Q];  // unsigned range-check trick
    const unsigned FULL = 0xffffffffu;

    if (block_q_idx < num_q_blocks) {
      CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
      full_q_barriers[0]->wait(0);

// Weights into registers (once per CTA -- the 2.5-generation win).
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; ++j)
          weights[i][j] = ptx::ld_shared(smem_weights[0] + i * kNumHeads + j);
      }
      // Four warp-uniform 8-bit counters, reset at each chunk.
      uint32_t emit_lane_counts = 0;
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        const uint32_t rq = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
        o_reg[i] = origin[rq];
        inv_reg[i] = inv_delta[rq];
        vth_reg[i] = -o_reg[i] * inv_reg[i];
        o_reg[i] = 0.0f;  // gate closed until the first consume
        kstart_reg[i] = seq_k_start[i];
        kspan_reg[i] =
            seq_k_end[i] > seq_k_start[i] ? seq_k_end[i] - seq_k_start[i] : 0;
      }
      // Fold -inv into weights once to accumulate directly in bucket units.
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; ++j) weights[i][j] *= -inv_reg[i];
      }

      uint32_t rs_max = 0, re_min = 0xffffffffu;
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        rs_max = max(rs_max, kstart_reg[i]);
        re_min = min(re_min, kstart_reg[i] + kspan_reg[i]);
      }

      if constexpr (BLOCK_Q == 4) {
        const float4 initial_gate =
            ptx::ld_shared(reinterpret_cast<const float4*>(sparse_gate_bits));
        o_reg[0] = initial_gate.x;
        o_reg[1] = initial_gate.y;
        o_reg[2 < BLOCK_Q ? 2 : 0] = initial_gate.z;
        o_reg[3 < BLOCK_Q ? 3 : 0] = initial_gate.w;
      } else {
#pragma unroll
        for (uint32_t gi = 0; gi < BLOCK_Q; ++gi)
          o_reg[gi] = __uint_as_float(ptx::ld_shared(sparse_gate_bits + gi));
      }

      for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
           ++kv_block_idx) {
        const uint32_t kvg = kv_block_idx;
        CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);

        if (kv_block_idx != 0 && (kv_block_idx % kActiveGateStride) == 0) {
          if constexpr (BLOCK_Q == 4) {
            const float4 gate = ptx::ld_shared(
                reinterpret_cast<const float4*>(sparse_gate_bits));
            o_reg[0] = gate.x;
            o_reg[1] = gate.y;
            o_reg[2 < BLOCK_Q ? 2 : 0] = gate.z;
            o_reg[3 < BLOCK_Q ? 3 : 0] = gate.w;
          } else {
#pragma unroll
            for (uint32_t gi = 0; gi < BLOCK_Q; ++gi)
              o_reg[gi] =
                  __uint_as_float(ptx::ld_shared(sparse_gate_bits + gi));
          }
        }
        full_kv_barriers[kv_stage_idx]->wait(kv_phase);

        const float scale_kv =
            ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);

        const uint32_t umma_buf =
            warpgroup_idx * kNumUmmaStages + (kvg % kNumUmmaStages);
        const uint32_t umma_wait_phase = (kvg / kNumUmmaStages) & 1;
        const auto tmem_start = umma_buf * UMMA_N;
        full_umma_barriers[umma_buf]->wait(umma_wait_phase);
        ptx::tcgen05_after_thread_sync();

        empty_kv_barriers[kv_stage_idx]->arrive();

        const auto kv_offset =
            kv_start + kv_block_idx * BLOCK_KV + math_thread_idx;
        DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

        uint32_t pass_bits = 0;
        float v_row[BLOCK_Q];
        constexpr uint32_t kTmemRowsEff =
            (128u / kNumHeads) < BLOCK_Q ? (128u / kNumHeads) : BLOCK_Q;
        DG_STATIC_ASSERT(BLOCK_Q % kTmemRowsEff == 0,
                         "BLOCK_Q must divide evenly into TMEM load groups");
#define LITETOPK_SCORE_GATE(i, RC)                                          \
  const float bq = fmaf(scale_kv, sum.x + sum.y, vth_reg[i]);                 \
  v_row[i] = bq;                                                           \
  bool g = __float_as_int(bq) < __float_as_int(o_reg[i]);                    \
  if constexpr (RC) g = g and ((kv_offset - kstart_reg[i]) < kspan_reg[i]); \
  pass_bits |= g ? (1u << i) : 0u;
        const uint32_t kv_base = kv_start + kv_block_idx * BLOCK_KV;
        const bool interior =
            (kv_base >= rs_max) && (kv_base + BLOCK_KV <= re_min);
#define LITETOPK_SCORE_ROWS(RANGE_CHECK)                                       \
  _Pragma("unroll") for (uint32_t pr = 0; pr < BLOCK_Q / kTmemRowsEff; ++pr) { \
    float accum2[kNumHeads * kTmemRowsEff];                                    \
    tmem_load(cute::Int<kNumHeads * kTmemRowsEff>{},                           \
              tmem_start + pr * kTmemRowsEff * kNumHeads, accum2);             \
    if (pr == BLOCK_Q / kTmemRowsEff - 1) {                                    \
      ptx::tcgen05_before_thread_sync();                                       \
      empty_umma_barriers[umma_buf]->arrive();                                 \
    }                                                                          \
    _Pragma("unroll") for (uint32_t k = 0; k < kTmemRowsEff; ++k) {            \
      const uint32_t i = pr * kTmemRowsEff + k;                                \
      const float* accum = accum2 + k * kNumHeads;                             \
      auto sum_0 = make_float2(0, 0);                                          \
      auto sum_1 = make_float2(0, 0);                                          \
      const auto transform = [&](const uint32_t& j, const float2& sum) {       \
        auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));      \
        auto b = make_float2(weights[i][j], weights[i][j + 1]);                \
        return __ffma2_rn(a, b, sum);                                          \
      };                                                                       \
      _Pragma("unroll") for (uint32_t j = 0; j < kNumHeads; j += 4) {          \
        sum_0 = transform(j, sum_0);                                           \
        sum_1 = transform(j + 2, sum_1);                                       \
      }                                                                        \
      auto sum = __fadd2_rn(sum_0, sum_1);                                     \
      LITETOPK_SCORE_GATE(i, RANGE_CHECK)                                      \
    }                                                                          \
  }
        if (interior) {
          LITETOPK_SCORE_ROWS(false)
        } else {
          LITETOPK_SCORE_ROWS(true)
        }
#undef LITETOPK_SCORE_ROWS
#undef LITETOPK_SCORE_GATE

        if (pass_bits != 0) {
#pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            if ((pass_bits >> i) & 1u) {
              const uint32_t candidate_score_bits =
                  candidate_fp24_code(v_row[i]) << 8;
              const uint32_t count = (emit_lane_counts >> (i * 8)) & 0xffu;
              if (count < kEmitLaneSlots) {
                const uint32_t pos =
                    ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + count) * 32u +
                    lane_idx;
                const uint32_t local_block =
                    kv_block_idx % kActiveEmitChunkBlocks;
                const uint32_t record = candidate_score_bits | local_block;
                atomicExch(emit_smem_records + pos, record);
                emit_lane_counts += 1u << (i * 8);
              } else {
                // Preserve local-ring overflow candidates with direct global stores.
                const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                const int out = atomicAdd(cand_cnt + row_q, 1);
                if (out < static_cast<int>(cand_cap)) {
                  const uint64_t out_base =
                      static_cast<uint64_t>(row_q) * cand_cap;
                  store_candidate_payload(&cand_val[out_base + out],
                                         &cand_idx[out_base + out],
                                         candidate_score_bits >> 8, kv_offset);
                }
              }
            }
          }
        }

        if (((kv_block_idx + 1) % kActiveEmitChunkBlocks) == 0 ||
            kv_block_idx + 1 == num_kv_blocks) {
          // Reserve one segment per (warp, row, chunk), then copy private records.
          // Scan two rows in separate 16-bit fields; each sum <= 32*18 prevents carry.
          int my_row_base = 0;
          uint32_t my_row_total = 0;
#pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            const uint32_t count = (emit_lane_counts >> (i * 8)) & 0xffu;
            const uint32_t total = __reduce_add_sync(FULL, count);
            if (lane_idx == i) my_row_total = total;
          }
          if (lane_idx < BLOCK_Q) {
            const uint32_t row_q = block_q_idx * BLOCK_Q + lane_idx;
            if (row_q < seq_len && my_row_total != 0) {
              my_row_base =
                  atomicAdd(cand_cnt + row_q, static_cast<int>(my_row_total));
            }
          }

          const uint32_t count0 = emit_lane_counts & 0xffu;
          const uint32_t count1 = (emit_lane_counts >> 8) & 0xffu;
          const uint32_t count2 = (emit_lane_counts >> 16) & 0xffu;
          const uint32_t count3 = emit_lane_counts >> 24;
          uint32_t inclusive01 = count0 | (count1 << 16);
          uint32_t inclusive23 = count2 | (count3 << 16);
#pragma unroll
          for (int delta = 1; delta < 32; delta <<= 1) {
            const uint32_t other01 = __shfl_up_sync(FULL, inclusive01, delta);
            const uint32_t other23 = __shfl_up_sync(FULL, inclusive23, delta);
            if (lane_idx >= static_cast<uint32_t>(delta)) {
              inclusive01 += other01;
              inclusive23 += other23;
            }
          }

          const uint32_t emit_window_kv_base =
              kv_start + (kv_block_idx / kActiveEmitChunkBlocks) *
                             kActiveEmitChunkBlocks * BLOCK_KV;
#pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            const uint32_t count = (emit_lane_counts >> (i * 8)) & 0xffu;
            const uint32_t inclusive =
                i < 2 ? ((inclusive01 >> ((i & 1u) * 16)) & 0xffffu)
                      : ((inclusive23 >> ((i & 1u) * 16)) & 0xffffu);
            const uint32_t offset = inclusive - count;
            const uint32_t row_q = block_q_idx * BLOCK_Q + i;
            const int out_base = __shfl_sync(FULL, my_row_base, i);

            int copy_count = 0;
            if (row_q < seq_len) {
              copy_count = min(static_cast<int>(count),
                               max(static_cast<int>(cand_cap) - out_base -
                                       static_cast<int>(offset),
                                   0));
            }
            for (int slot = 0; slot < copy_count; ++slot) {
              const uint32_t local_pos =
                  ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + slot) * 32u +
                  lane_idx;
              const uint32_t record =
                  atomicExch(emit_smem_records + local_pos, kRingSentinel);
              const int out = out_base + static_cast<int>(offset) + slot;
              const uint32_t record_payload = (record & 0xffffff00u) >> 8;
              const uint32_t kvo = emit_window_kv_base +
                                  ((record & 0xffu) * BLOCK_KV) + math_thread_idx;
              const uint64_t out_pos =
                  static_cast<uint64_t>(row_q) * cand_cap + out;
              store_candidate_payload(&cand_val[out_pos], &cand_idx[out_pos],
                                     record_payload, kvo);
            }
            // Clear capped slots before the next flush epoch.
            for (int slot = copy_count; slot < static_cast<int>(count);
                 ++slot) {
              atomicExch(emit_smem_records +
                  ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + slot) * 32u + lane_idx,
                  kRingSentinel);
            }
          }
          // Publish the epoch after drain stores and before next-window appends.
          __syncwarp();
          __threadfence_block();
          if (lane_idx < BLOCK_Q) {
            atomicAdd(ring_seq + warp_idx * BLOCK_Q + lane_idx, 1u);
          }
          __threadfence_block();

          emit_lane_counts = 0;
        }
      }

      // Every actual chunk was published at its final KV block.
      empty_q_barriers[0]->arrive();
    }

    // Signal the refresh daemon, then free tensor memory.
    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if (threadIdx.x == 0) {
      __threadfence_block();
      atomicExch(scan_done_flag, 1u);
      // An odd flip count wakes parity waiters; the exit-flag check ensures exit.
      asm volatile("" ::: "memory");
#pragma unroll
      for (int poke = 0; poke < 5; ++poke) {
        full_kv_barriers[0]->arrive();
      }
    }
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
}

}  // namespace dsa_litetopk
// namespace dsa_litetopk
