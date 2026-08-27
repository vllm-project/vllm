// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek

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
inline constexpr uint32_t kGateStride = 64;
inline constexpr uint32_t kWarpQueueCap = 64;
inline constexpr uint32_t kMathRegisters = 240;
inline constexpr uint32_t kSpecializedRegisters = 24;
inline constexpr uint32_t kTmemRows = 4;
inline constexpr uint32_t kUmmaStages = 2;
inline constexpr uint32_t kSparseRefreshNs = 512;
inline constexpr uint32_t kSparseRefreshIdleNs = 2048;
// A ring slot can only equal this pattern if a +NaN score passed the gate,
// which both gate comparisons reject; the refresher treats it as "empty".
inline constexpr uint32_t kRingSentinel = 0xffffffffu;
// Gate updates only need to keep pace with the 65,536-token flush windows,
// so the refresher can poll orders of magnitude slower than the dynamic
// histogram daemon and stay invisible to the math warps' issue ports.
inline constexpr uint32_t kRingActiveNs = 2048;
inline constexpr uint32_t kRingIdleNs = 2048;

// Production candidate ABI: a six-byte global record. cand_val stores the
// low 16 bits of an ascending, sign-aware FP32 high24 score code; cand_idx
// stores its high eight score bits above the exact 20-bit KV index. The local
// ring uses one uint32_t containing high24 score bits and an eight-bit
// block-in-window.
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

// Descending full-width ordering: smaller unsigned keys correspond to larger
// FP32 values.
CUTLASS_DEVICE uint32_t candidate_descending_fp32_code(const float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? bits : (~bits & 0x7fffffffu);
}

// High eight bits of a sign-aware FP16 key.
// Keeping 256 coarse buckets makes the online histogram small enough to live
// beside the score pipeline while retaining much more resolution than the
// sign/exponent byte of a raw FP32 key.
CUTLASS_DEVICE uint32_t candidate_fixed_half_bucket(const float value) {
  const uint32_t fp32_bits = __float_as_uint(value);
  const __half half_value = __float2half(value);
  uint16_t bits = __half_as_ushort(half_value) & 0x7fffu;
  bits |= static_cast<uint16_t>((fp32_bits >> 16) & 0x8000u);
  bits = (bits & 0x8000u) ? bits : static_cast<uint16_t>(~bits & 0x7fffu);
  return static_cast<uint32_t>(bits >> 8);
}

struct OnlineFixedCandidate {
  uint32_t bucket;
  uint32_t payload;
};

CUTLASS_DEVICE OnlineFixedCandidate candidate_online_fixed(const float value) {
  const uint32_t bucket = candidate_fixed_half_bucket(value);
  const uint32_t raw_desc24 = candidate_descending_fp32_code(value) >> 8;
  // Experimental six-byte payload: the coarse bucket is explicit, while
  // the remaining 16 bits retain the raw-score order inside that bucket.
  return {bucket, (bucket << 16) | (raw_desc24 & 0xffffu)};
}

CUTLASS_DEVICE uint32_t candidate_load_score_code(const CandidateValue value,
                                                  const int32_t packed_idx) {
  return (static_cast<uint32_t>(packed_idx) >> kCandidateIndexBits) << 16 |
         static_cast<uint32_t>(value);
}

CUTLASS_DEVICE float candidate_decode_score(const CandidateValue value,
                                            const int32_t packed_idx) {
  const uint32_t code = candidate_load_score_code(value, packed_idx);
  const uint32_t ordered = code << 8;
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
}

CUTLASS_DEVICE int32_t candidate_decode_index(const int32_t packed_idx) {
  return static_cast<int32_t>(static_cast<uint32_t>(packed_idx) &
                              kCandidateIndexMask);
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

CUTLASS_DEVICE void store_candidate_record(CandidateValue* value_dst,
                                           int32_t* index_dst,
                                           const uint32_t payload,
                                           const uint32_t kv_index) {
  store_candidate_payload(value_dst, index_dst, payload, kv_index);
}

template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t BLOCK_Q,
          uint32_t BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs, uint32_t kNumSpecializedThreads,
          uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128,
          bool kOnlineFixedBuckets = false, bool kStaticHotGate = false,
          bool kStaticHotNoHist = false, bool kStaticHotExactGate = false,
          bool kRingRefresh = false>
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
                               const uint32_t refresh_every,
                               const uint32_t num_kv_splits,
                               const uint32_t
                                   probe_group,  // compacted-space group size
                                                 // (pstp-1)*64; 0 = no probe
                                                 // compaction (identity map)
                               const uint64_t
                                   probe_magic,  // ceil(2^42/probe_group):
                                                 // exact div via mul-shift
                               const uint32_t
                                   probe_add_max,  // npage*64 cap for the map
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
                               // fp8: per-token fp32 dequant scales. fp4: the
                               // UE8M0 SF-KV int32 stream (same 1-D shape).
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_kv_scales,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_weights,
                               // fp4 only: per-(token, head) packed UE8M0 SF-Q
                               // int32 stream; fp8 callers pass any map
                               // (unread).
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_sf_q) {
  const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto warpgroup_idx = warp_idx / 4;
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
  constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
  constexpr uint32_t kNumUmmaStages = kUmmaStages;
  constexpr uint32_t kNumUmmaBuffers = kNumMathWarpGroups * kNumUmmaStages;
  // fp4 replaces the per-warpgroup UMMA double buffers with a shared
  // 3-stage TMEM accumulator pool (the SF columns take the freed space);
  constexpr uint32_t kNumAccumBufs = kNumUmmaBuffers;
  // The ring refresher reloads more often: a flooding row keeps emitting
  // at the stale gate for a full stride, so the flood integral scales with
  // the reload cadence. The reload is one ld.shared.v4 per stride.
  constexpr uint32_t kActiveGateStride =
      kOnlineFixedBuckets ? 4u : (kRingRefresh ? 8u : kGateStride);
  constexpr uint32_t kActiveEmitChunkBlocks =
      kOnlineFixedBuckets ? 32u : kEmitChunkBlocks;
  static_assert(
      !(kOnlineFixedBuckets && kStaticHotGate),
      "online fixed buckets and static HOT gate are separate A/B paths");
  static_assert(!kStaticHotNoHist || kStaticHotGate,
                "static HOT no-hist requires the static HOT gate");
  static_assert(!kStaticHotExactGate || kStaticHotNoHist,
                "exact FP24 gate requires the static HOT no-hist path");
  static_assert(
      !kRingRefresh ||
          (kStaticHotNoHist && !kStaticHotExactGate && !kOnlineFixedBuckets),
      "ring refresher v1 covers only the plain bucket-gate no-hist writer");
  DG_STATIC_ASSERT(
      (BLOCK_Q == 4 || BLOCK_Q == 2) && kNumMathWarps == 8,
      "LiteTopK requires BLOCK_Q in {4 (H=32), 2 (H=64)} and 8 math warps");
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

  // fp4 packs two e2m1 elements per byte; SF regions are UE8M0 bytes
  // packed 4-per-int32 and padded to the 128-element UTCCP granule.
  static constexpr uint32_t kKvElemBytesNum = 2u;  // per 2 elems
  static constexpr uint32_t kNumUTCCPAlignedElems = 128;
  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * kHeadDim * kKvElemBytesNum / 2;
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * sizeof(float);
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE =
      BLOCK_KV * kHeadDim * kKvElemBytesNum / 2;
  // fp8: one fp32 dequant scale per token. fp4: one int32 (4x UE8M0) per
  // token, padded to the UTCCP granule.
  static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE =
      BLOCK_KV * sizeof(float);
  static constexpr uint32_t SMEM_SF_Q_SIZE = 0u;
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

  auto smem_sf_q = reinterpret_cast<uint32_t*>(
      reinterpret_cast<uint8_t*>(smem_kv_scales[kNumKVStages]));
  auto barrier_ptr = reinterpret_cast<Barrier*>(
      reinterpret_cast<uint8_t*>(smem_sf_q) + SMEM_SF_Q_SIZE);
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

  auto sf_ready_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr +
           (kNumQStages * 2 + kNumKVStages * 2 + kNumAccumBufs * 2 + i);
  });
  auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
      barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumAccumBufs * 2 +
      kNumKVStages);
  auto scan_done_flag = reinterpret_cast<volatile int*>(tmem_ptr_in_smem + 1);
  auto warpq_count = reinterpret_cast<int32_t*>(tmem_ptr_in_smem + 4);
  // Chunked emit does not use the legacy warp-queue count words. Reuse four
  // of them as a CTA-local Gate4 mailbox. Values are positive float edge
  // bit-patterns, so unsigned atomicMin is exactly a monotonic gate tighten.
  auto sparse_gate_bits = reinterpret_cast<uint32_t*>(warpq_count);
  auto emit_smem_records =
      reinterpret_cast<uint32_t*>(warpq_count + kNumMathWarps * BLOCK_Q);
  auto smem_hist = reinterpret_cast<int32_t*>(
      emit_smem_records + kNumMathWarps * BLOCK_Q * kEmitLaneSlots * 32u);
  // Ring-refresher scratch sits after the refresher's histogram (which
  // reuses the dynamic-path histogram region): per-(math warp, row) flush
  // sequence words, the spare warps' shadow copies, and per-lane
  // consumed-slot cursors.
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
    if constexpr (!kStaticHotNoHist || kRingRefresh) *scan_done_flag = 0;
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
  // One CTA owns each row. The hot-only path emits no sample seeds and scans
  // the complete KV range, so its exact histogram starts at zero in SMEM.
  // LITETOPK_STATIC_HOT_NOHIST_AB removes both this clear and every update;
  // a post-scan kernel rebuilds metadata from the compact candidate slab.
  if constexpr (!kStaticHotNoHist) {
    for (uint32_t idx = threadIdx.x; idx < BLOCK_Q * num_buckets;
         idx += blockDim.x) {
      smem_hist[idx] = 0;
    }
  }
  if constexpr (kRingRefresh) {
    for (uint32_t idx = threadIdx.x;
         idx < kNumMathWarps * BLOCK_Q * kEmitLaneSlots * 32u;
         idx += blockDim.x) {
      emit_smem_records[idx] = kRingSentinel;
    }
    // Preload the seed sample's full bucket histogram as the daemon's
    // refresh base. The sample records are genuine row records in the
    // same (origin, inv) bucket space and the exact-once scan starts after
    // the sampled prefix, so ring harvests never recount them; base plus
    // harvested stays a subset of the row's true records and every
    // publish remains a provable upper bound.
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
    if constexpr (kStaticHotExactGate) {
      // Exact compaction stores the ordered-FP32 pivot edge in this
      // int32 slot. Preserve its bits so the unsigned strict comparison
      // admits only later records from a better high24 class.
      ptx::st_shared(sparse_gate_bits + threadIdx.x,
                     static_cast<uint32_t>(gate));
    } else {
      ptx::st_shared(sparse_gate_bits + threadIdx.x,
                     __float_as_uint(static_cast<float>(gate + 1)));
    }
  }
  __syncthreads();

  constexpr uint32_t kNumSpecializedRegisters = kSpecializedRegisters;
  constexpr uint32_t kNumMathRegisters = kMathRegisters;

  // V1 KV-split scheduling: blockIdx.x = q-block (one per CTA), blockIdx.y =
  // contiguous KV sub-window. Split boundaries are BLOCK_KV-aligned.
  const uint32_t block_q_idx = blockIdx.x;
  const uint32_t kv_split = blockIdx.y;
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
    const uint32_t total_blocks = math::ceil_div(seq_len_kv, BLOCK_KV);
    const uint32_t blocks_per_split =
        math::ceil_div(total_blocks, num_kv_splits);
    const uint32_t split_lo = kv_split * blocks_per_split * BLOCK_KV;
    const uint32_t split_hi =
        min((kv_split + 1) * blocks_per_split * BLOCK_KV, seq_len_kv);
    start = start / 4 * 4;  // TMA alignment for SF KV
    if (start < split_lo) start = split_lo;
    if (end > split_hi) end = split_hi;
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

    if (cute::elect_one_sync()) {
      if (block_q_idx < num_q_blocks) {
        // Q + weights once for this q-block. Inner extents are BYTES
        // for the uint8 maps, so fp4's packed rows halve them.
        constexpr uint32_t kKvInnerBytes = kHeadDim * kKvElemBytesNum / 2;
        tma::copy<kKvInnerBytes, BLOCK_Q * kNumHeads, kKvInnerBytes>(
            &tensor_map_q, full_q_barriers[0], smem_q[0], 0,
            block_q_idx * BLOCK_Q * kNumHeads);
        tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights,
                                         full_q_barriers[0], smem_weights[0], 0,
                                         block_q_idx * BLOCK_Q);
        full_q_barriers[0]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE +
                                                 SMEM_WEIGHT_SIZE_PER_STAGE);

        CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
        for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
             ++kv_block_idx) {
          CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
          empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

          tma::copy<kKvInnerBytes, BLOCK_KV, kKvInnerBytes>(
              &tensor_map_kv, full_kv_barriers[kv_stage_idx],
              smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
          tma::copy<BLOCK_KV, 1, 0>(&tensor_map_kv_scales,
                                    full_kv_barriers[kv_stage_idx],
                                    smem_kv_scales[kv_stage_idx],
                                    kv_start + kv_block_idx * BLOCK_KV, 0);
          full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(
              SMEM_KV_SIZE_PER_STAGE + SMEM_KV_SCALE_SIZE_PER_STAGE);
        }
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
        // Round-robin over kNumUmmaStages accumulators. A stage is
        // reused every kNumUmmaStages tiles, so its phase toggles at
        // that rate, not every tile.
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
      const auto refresh_row = [&](const uint32_t row,
                                   const bool publish_boundary_counts) {
        if (row >= seq_len) return false;
        const uint32_t local_row = row - block_q_idx * BLOCK_Q;
        const int32_t* srow =
            smem_hist + (row - block_q_idx * BLOCK_Q) * num_buckets;
        const int current_gate =
            min(max(static_cast<int>(__uint_as_float(
                        ptx::ld_shared(sparse_gate_bits + local_row))) -
                        1,
                    0),
                static_cast<int>(num_buckets) - 1);
        // Only buckets strictly below the published gate can tighten
        // it. If they contain fewer than topk entries, keep the
        // current gate and avoid scanning the dead upper tail.
        const uint32_t search_buckets = static_cast<uint32_t>(current_gate);
        int found = current_gate;
        int carry = 0;
        int found_lt = 0;
        int found_eq = 0;
        bool done = false;
        for (uint32_t base = 0; base < search_buckets && !done; base += 32) {
          uint32_t b = base + lane_idx;
          int v = 0;
          if (b < search_buckets) {
            // In the single-split fast path smem already contains
            // the initial global histogram plus every scan hit.
            v = srow[b];
          }
          // Histogram entries are nonnegative. Most 32-bucket
          // groups remain strictly below K, so reject them with
          // one REDUX and reserve the five dependent prefix
          // shuffles for the single group that crosses K.
          const int group_sum = __reduce_add_sync(0xffffffffu, v);
          if (carry + group_sum < static_cast<int>(topk)) {
            carry += group_sum;
            continue;
          }
          int prefix = v;
#pragma unroll
          for (int off = 1; off < 32; off <<= 1) {
            int nsh = __shfl_up_sync(0xffffffffu, prefix, off);
            if (static_cast<int>(lane_idx) >= off) prefix += nsh;
          }
          int incl = carry + prefix;
          bool hit = (b < search_buckets) && (incl >= static_cast<int>(topk)) &&
                     (incl - v < static_cast<int>(topk));
          unsigned hm = __ballot_sync(0xffffffffu, hit);
          if (hm) {
            const int hit_lane = __ffs(hm) - 1;
            found = static_cast<int>(base) + hit_lane;
            found_lt = __shfl_sync(0xffffffffu, incl - v, hit_lane);
            found_eq = __shfl_sync(0xffffffffu, v, hit_lane);
            done = true;
          } else {
            carry += __shfl_sync(0xffffffffu, prefix, 31);
          }
        }
        if (!done) {
          // No lower bucket reached K, so the current gate remains
          // the boundary. `carry` is exactly count(bucket<gate).
          found_lt = carry;
          found_eq = srow[found];
        }
        if (lane_idx == 0) {
          // One CTA owns this row. Publish only a genuine tightening;
          // a stale math-warp mailbox read remains conservatively loose.
          const uint32_t edge = __float_as_uint(static_cast<float>(found + 1));
          if (done) {
            th_bucket[row] = found;
            if constexpr (!kStaticHotGate) {
              atomicMin(sparse_gate_bits + local_row, edge);
            }
          }
          if (publish_boundary_counts && num_buckets >= 3) {
            // Reuse the dead histogram row as selector metadata.
            int32_t* meta = bcount + static_cast<uint64_t>(row) * num_buckets;
            meta[0] = ~found;
            meta[1] = found_lt;
            meta[2] = found_eq;
          }
        }
        return done;
      };
      // LITETOPK_STATIC_HOT_AB: the external HOT sample provides a
      // fixed scan-time gate. Avoid every intermediate prefix scan and
      // publish the tight full-scan boundary certificate only once.
      if constexpr (kRingRefresh) {
        // Ring-readback refresher: histogram already-emitted smem
        // ring records (word-atomic reads) and tighten the CTA gate
        // mailbox with zero additions to the math-warp hit path.
        // Counting a SUBSET of emitted records keeps every published
        // edge a provable lower bound on the row's current K-th
        // best, and the per-(math warp, row) flush seqlock discards
        // any read that raced a drain so no record counts twice.
        //
        // The production daemon is always active. Quiet passes use a
        // fixed 2048 ns sleep; keeping this compile-time removes the
        // former launch-time daemon/pacing A/B branches.
        uint32_t pass = 0;
        while (*scan_done_flag == 0) {
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
                  *reinterpret_cast<volatile uint32_t*>(ring_seq + pair);
              if (s0 != ring_shadow[pair]) {
                // New flush epoch: the drained column is
                // sentinel again and refills from zero.
                ring_progress[pair * 32u + lane_idx] = 0u;
                __syncwarp();
                if (lane_idx == 0) ring_shadow[pair] = s0;
                __syncwarp();
              }
              // Deep harvest: drain the lane's whole staged
              // column per visit (up to kEmitLaneSlots) instead
              // of one record per pass. The flood-time daemon
              // bottleneck is harvest bandwidth, not publish
              // cadence; reads and smem atomics only, so the
              // math warps' issue ports stay untouched.
              uint32_t progress = ring_progress[pair * 32u + lane_idx];
              uint32_t took = 0;
              while (progress < kEmitLaneSlots) {
                const uint32_t record = *reinterpret_cast<volatile uint32_t*>(
                    emit_smem_records +
                    (pair * kEmitLaneSlots + progress) * 32u + lane_idx);
                if (record == kRingSentinel) break;
                if (*reinterpret_cast<volatile uint32_t*>(ring_seq + pair) !=
                    s0) {
                  // Raced a flush drain: discard this
                  // read; the epoch reset above replays
                  // the refilled column next visit.
                  break;
                }
                // Invert the truncated fp24 code back to
                // the affine bucket value. Truncation only
                // rounds toward better scores and never
                // crosses an integer boundary (codes and
                // integer edges share the 0x100-aligned
                // grid), so unshifted bins support a
                // pad-free published edge. The comparison
                // ladder (not fmin/fmax) routes NaN and
                // out-of-range scores to the top bin, which
                // the publish guard below never counts.
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
            if (pend - ring_last[local_row] >=
                static_cast<int32_t>(topk) / 16) {
              if (lane_idx == 0) ring_last[local_row] = pend;
              __syncwarp();
              // Smallest bucket prefix holding at least K ring
              // records. The truncated codes and the ordered
              // codes of integers <= 256 both sit on the
              // 0x100-aligned grid, so a counted record's
              // true code is < ord(found + 1) exactly and the
              // bare bucket upper boundary is a provably safe
              // published edge — no truncation pad needed.
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
                // Early warm-start scans that fail to
                // tighten only lack fresh evidence, not
                // convergence; they must not retire the
                // daemon.
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
            while (slept < kRingIdleNs && *scan_done_flag == 0) {
              __nanosleep(kRingActiveNs);
              slept += kRingActiveNs;
            }
          }
        }
      } else if constexpr (kStaticHotNoHist) {
        // Scratch path: no in-scan histogram has to be finalized, so
        // both spare refresh warps can retire immediately.
      } else if constexpr (kStaticHotGate) {
        while (*scan_done_flag == 0) __nanosleep(kSparseRefreshIdleNs);
      } else {
        while (*scan_done_flag == 0) {
          bool tightened = false;
          for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
            tightened |= refresh_row(block_q_idx * BLOCK_Q + r, false);
          __nanosleep(tightened ? kSparseRefreshNs : kSparseRefreshIdleNs);
        }
      }
      if constexpr (!kStaticHotNoHist) {
        for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
          refresh_row(block_q_idx * BLOCK_Q + r, true);
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

    // Bucket comparisons use the affine score space. Raw float bit
    // patterns are nonuniform across exponent ranges and cannot preserve
    // this threshold contract.
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
      // Queue fill counts are warp-uniform: every lane tracks them
      // redundantly in registers, so the hot emit path needs no shared
      // bookkeeping or shuffle broadcast.
      // Four private 8-bit counts. They reset at every fixed-size chunk,
      // bounding the address dispersion of direct global record stores.
      uint32_t emit_lane_counts = 0;
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        const uint32_t rq = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
        if constexpr (kOnlineFixedBuckets) {
          // No sample-derived affine transform in the online A/B
          // path.  The raw score is converted to fixed buckets in
          // the epilogue below; keep these registers compile-time
          // independent of the nullable origin/inv pointers.
          o_reg[i] = 0.0f;
          inv_reg[i] = 1.0f;
          vth_reg[i] = 0.0f;
        } else {
          o_reg[i] = origin[rq];
          inv_reg[i] = inv_delta[rq];
          vth_reg[i] = -o_reg[i] * inv_reg[i];
        }
        o_reg[i] = 0.0f;  // gate closed until the first consume
        kstart_reg[i] = seq_k_start[i];
        kspan_reg[i] =
            seq_k_end[i] > seq_k_start[i] ? seq_k_end[i] - seq_k_start[i] : 0;
      }
      // Fold -inv into the register weights: the whole ReLU-weighted
      // chain then accumulates directly in bucket units. 128 FMULs
      // once per qb, amortized over thousands of kv blocks.
      if constexpr (!kOnlineFixedBuckets) {
#pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++i) {
#pragma unroll
          for (uint32_t j = 0; j < kNumHeads; ++j) weights[i][j] *= -inv_reg[i];
        }
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

        if constexpr (!kStaticHotGate || kRingRefresh) {
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
        uint32_t fixed_payload_row[BLOCK_Q];
        uint32_t fixed_bucket_row[BLOCK_Q];
        constexpr uint32_t kTmemRowsEff =
            (128u / kNumHeads) < BLOCK_Q ? (128u / kNumHeads) : BLOCK_Q;
        DG_STATIC_ASSERT(BLOCK_Q % kTmemRowsEff == 0,
                         "BLOCK_Q must divide evenly into TMEM load groups");
#define LITETOPK_SCORE_GATE(i, RC)                                          \
  bool g;                                                                   \
  if constexpr (kOnlineFixedBuckets) {                                      \
    const float raw_score = scale_kv * (sum.x + sum.y);                     \
    const OnlineFixedCandidate fixed = candidate_online_fixed(raw_score);   \
    fixed_payload_row[i] = fixed.payload;                                   \
    fixed_bucket_row[i] = fixed.bucket;                                     \
    g = fixed.bucket < static_cast<uint32_t>(o_reg[i]);                     \
  } else {                                                                  \
    const float bq = fmaf(scale_kv, sum.x + sum.y, vth_reg[i]);             \
    if constexpr (kStaticHotExactGate) {                                    \
      const uint32_t ordered = candidate_ordered_fp32_code(bq);             \
      v_row[i] = __uint_as_float(ordered);                                  \
      g = ordered < __float_as_uint(o_reg[i]);                              \
    } else {                                                                \
      v_row[i] = bq;                                                        \
      g = __float_as_int(bq) < __float_as_int(o_reg[i]);                    \
    }                                                                       \
  }                                                                         \
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
              uint32_t candidate_score_bits;
              uint32_t candidate_bucket;
              if constexpr (kOnlineFixedBuckets) {
                candidate_score_bits = fixed_payload_row[i] << 8;
                candidate_bucket = fixed_bucket_row[i];
              } else {
                if constexpr (kStaticHotExactGate) {
                  // v_row carries the already-computed
                  // ordered32 key in this specialization.
                  // The ring's low byte belongs exclusively
                  // to local_block; clear the discarded FP32
                  // low bits before packing that coordinate.
                  candidate_score_bits =
                      __float_as_uint(v_row[i]) & 0xffffff00u;
                  candidate_bucket = 0;
                } else {
                  const float x = v_row[i];
                  candidate_score_bits = candidate_fp24_code(x) << 8;
                  const int candidate_braw = static_cast<int>(x);
                  candidate_bucket =
                      static_cast<uint32_t>(max(candidate_braw, 0));
                }
              }
              const uint32_t count = (emit_lane_counts >> (i * 8)) & 0xffu;
              if (count < kEmitLaneSlots) {
                const uint32_t pos =
                    ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + count) * 32u +
                    lane_idx;
                const uint32_t local_block =
                    kv_block_idx % kActiveEmitChunkBlocks;
                const uint32_t record = candidate_score_bits | local_block;
                emit_smem_records[pos] = record;
                emit_lane_counts += 1u << (i * 8);
              } else {
                // A skewed lane can overflow its small local
                // quota without losing candidates. This slow
                // path writes directly to the final buffer.
                const uint32_t row_q = block_q_idx * BLOCK_Q + i;
                const int out = atomicAdd(cand_cnt + row_q, 1);
                if (out < static_cast<int>(cand_cap)) {
                  uint32_t kvo = kv_offset;
                  if (probe_group != 0) {
                    const uint32_t sup = static_cast<uint32_t>(
                        (static_cast<uint64_t>(kvo) * probe_magic) >> 42);
                    kvo += min((sup + 1) * 64u, probe_add_max);
                  }
                  const uint64_t out_base =
                      static_cast<uint64_t>(row_q) * cand_cap;
                  store_candidate_record(&cand_val[out_base + out],
                                         &cand_idx[out_base + out],
                                         candidate_score_bits >> 8, kvo);
                }
              }

              // A passing positive bq is below the float edge
              // (gate + 1), hence already < num_buckets.
              if constexpr (!kStaticHotNoHist) {
                atomicAdd(smem_hist + i * num_buckets +
                              static_cast<int>(candidate_bucket),
                          1);
              }
            }
          }
        }

        if (((kv_block_idx + 1) % kActiveEmitChunkBlocks) == 0 ||
            kv_block_idx + 1 == num_kv_blocks) {
          // Reserve one contiguous final-buffer segment per
          // (source warp, row, chunk), then have each lane copy its
          // private records into that segment. This keeps all
          // collectives and the returning global atomic off the
          // ordinary-hit path while deleting the rectangular global
          // chunk workspace and the post-scan compactor.
          // Keep the four row reservations parallel and early so
          // their L2 round trips overlap the prefix work. Replace
          // four independent 32-lane scans (20 shuffles) with two
          // scans whose 16-bit fields carry two rows each (10
          // shuffles). A row prefix is at most 32*18=576, so fields
          // cannot carry into one another.
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
              const uint32_t record = emit_smem_records[local_pos];
              if constexpr (kRingRefresh) {
                emit_smem_records[local_pos] = kRingSentinel;
              }
              const int out = out_base + static_cast<int>(offset) + slot;
              const uint32_t record_payload = (record & 0xffffff00u) >> 8;
              uint32_t kvo = emit_window_kv_base +
                             ((record & 0xffu) * BLOCK_KV) + math_thread_idx;
              if (probe_group != 0) {
                const uint32_t sup = static_cast<uint32_t>(
                    (static_cast<uint64_t>(kvo) * probe_magic) >> 42);
                kvo += min((sup + 1) * 64u, probe_add_max);
              }
              const uint64_t out_pos =
                  static_cast<uint64_t>(row_q) * cand_cap + out;
              store_candidate_record(&cand_val[out_pos], &cand_idx[out_pos],
                                     record_payload, kvo);
            }
            if constexpr (kRingRefresh) {
              // A capped row copies fewer slots than it
              // holds; the leftovers must not leak into
              // the next flush epoch.
              for (int slot = copy_count; slot < static_cast<int>(count);
                   ++slot) {
                emit_smem_records[((warp_idx * BLOCK_Q + i) * kEmitLaneSlots +
                                   slot) *
                                      32u +
                                  lane_idx] = kRingSentinel;
              }
            }
          }
          if constexpr (kRingRefresh) {
            // Publish the flush epoch: every drain store above
            // is visible before the bump, and the bump is
            // visible before any next-window append.
            __syncwarp();
            __threadfence_block();
            if (lane_idx < BLOCK_Q) {
              ring_seq[warp_idx * BLOCK_Q + lane_idx] += 1u;
            }
            __threadfence_block();
          }

          emit_lane_counts = 0;
        }
      }

      // Every actual chunk was published at its final KV block.
      empty_q_barriers[0]->arrive();
    }

    // Signal the refresh daemon, then free tensor memory.
    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if constexpr (!kStaticHotNoHist || kRingRefresh) {
      if (threadIdx.x == 0) {
        __threadfence_block();
        *scan_done_flag = 1;
        if constexpr (kRingRefresh) {
          // Prompt-wake a daemon parked on a stage-0 parity: odd
          // count so a straggler that missed transient flips
          // still sees the final parity flipped. The in-wait
          // exit-flag recheck makes this an optimization, not a
          // correctness requirement.
          asm volatile("" ::: "memory");
#pragma unroll
          for (int poke = 0; poke < 5; ++poke) {
            full_kv_barriers[0]->arrive();
          }
        }
      }
    }
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
}

// fp4 graft scan: deep_gemm's sm100_fp4_mqa_logits skeleton (persistent
// CTAs with a 3-stage Q pipeline, split TMA warps, inline SF transpose +
// UTCCP + MMA in one issuer warp, KV and SF-KV on a single transaction
// barrier) with the logits store replaced by the bucket-gate no-hist emit.
// The supply side is the dense reference's own choreography, so scan supply
// is dense-parity by construction; the fused side adds only the gate
// compare, the staged-ring append and the window flush.
template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t BLOCK_Q,
          uint32_t BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs, uint32_t kNumSpecializedThreads,
          uint32_t kNumMathThreads, bool kSelfTighten = false,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128>
CUTLASS_GLOBAL __launch_bounds__(
    kNumSpecializedThreads + kNumMathThreads,
    1) void sm100_dsa_litetopk_fp4graft(const uint32_t seq_len,
                                        const uint32_t seq_len_kv,
                                        const uint32_t* __restrict__ cu_seq_len_k_start,
                                        const uint32_t* __restrict__ cu_seq_len_k_end,
                                        // Written by the fused prep (warp +3)
                                        // and read by the math warps within one
                                        // kernel, so neither const nor
                                        // __restrict__ is truthful.
                                        float* origin, float* inv_delta,
                                        int32_t* th_bucket,
                                        CandidateValue* __restrict__ cand_val,
                                        int32_t* __restrict__ cand_idx,
                                        int32_t* __restrict__ cand_cnt,
                                        const uint32_t cand_cap,
                                        const __grid_constant__
                                            cute::TmaDescriptor tensor_map_q,
                                        const __grid_constant__
                                            cute::TmaDescriptor tensor_map_sf_q,
                                        const __grid_constant__
                                            cute::TmaDescriptor tensor_map_kv,
                                        const __grid_constant__
                                            cute::TmaDescriptor
                                                tensor_map_sf_kv,
                                        const __grid_constant__
                                            cute::TmaDescriptor
                                                tensor_map_weights,
                                        // Self-tighten warmstart inputs:
                                        // seed_bcount is seed_prep's exported
                                        // [Q, 256] histogram, seed_topk is K
                                        // for the window-end ring rule.
                                        const uint32_t seed_topk,
                                        int32_t* __restrict__ seed_bcount) {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  const auto sm_idx = blockIdx.x;
  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto warpgroup_idx = warp_idx / 4;
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_sf_q);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
  }

  static constexpr uint32_t kNumTmemStages = 3;
  static constexpr uint32_t kNumUTCCPAlignedElems = 128;
  static constexpr uint32_t UMMA_M = 128;
  static constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;
  static constexpr uint32_t kUmmaKFp4 = 64;
  static constexpr uint32_t kNumSFQ =
      math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems);
  static constexpr uint32_t kNumSFKV =
      math::constexpr_align(BLOCK_KV, kNumUTCCPAlignedElems);
  static constexpr uint32_t kRealNumSFQ = BLOCK_Q * kNumHeads;
  DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0,
                   "Invalid threads");
  DG_STATIC_ASSERT(BLOCK_KV == kNumMathWarpGroups * UMMA_M and
                       BLOCK_KV % kNumUTCCPAlignedElems == 0,
                   "Invalid `BLOCK_KV`");

  static constexpr uint32_t kSwizzleAlignment = 8 * (kHeadDim / 2);
  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * (kHeadDim / 2);
  static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE = kNumSFQ * sizeof(int);
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE = BLOCK_KV * (kHeadDim / 2);
  static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE = kNumSFKV * sizeof(int);
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * sizeof(float);

  extern __shared__ __align__(kSwizzleAlignment) uint8_t smem_buffer[];

  auto smem_q = utils::PatternVisitor([&](const uint32_t& i) {
    return smem_buffer + SMEM_Q_SIZE_PER_STAGE * i;
  });
  auto smem_kv = utils::PatternVisitor([&](const uint32_t& i) {
    return smem_buffer + SMEM_Q_SIZE_PER_STAGE * kNumQStages +
           SMEM_KV_SIZE_PER_STAGE * i;
  });
  const auto smem_sf_ptr =
      smem_buffer + (SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                     SMEM_KV_SIZE_PER_STAGE * kNumKVStages);
  auto smem_sf_q = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<uint32_t*>(smem_sf_ptr +
                                       SMEM_SF_Q_SIZE_PER_STAGE * i);
  });
  auto smem_sf_kv = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<uint32_t*>(smem_sf_ptr +
                                       SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages +
                                       SMEM_SF_KV_SIZE_PER_STAGE * i);
  });
  auto smem_weights = utils::PatternVisitor([&](const uint32_t& i) {
    return reinterpret_cast<float*>(smem_sf_ptr +
                                    SMEM_SF_Q_SIZE_PER_STAGE * kNumQStages +
                                    SMEM_SF_KV_SIZE_PER_STAGE * kNumKVStages +
                                    SMEM_WEIGHT_SIZE_PER_STAGE * i);
  });

  const auto barrier_ptr =
      reinterpret_cast<Barrier*>(smem_weights[kNumQStages]);
  auto full_q_barriers =
      utils::PatternVisitor([&](const uint32_t& i) { return barrier_ptr + i; });
  auto empty_q_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return barrier_ptr + kNumQStages + i; });
  auto full_kv_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return barrier_ptr + kNumQStages * 2 + i; });
  auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr + kNumQStages * 2 + kNumKVStages + i;
  });
  // DeepGEMM's FP8/FP4 1d1d choreography: TMA publishes the raw SF tile
  // through full_kv, a dedicated warp transposes it in shared memory, then
  // all 32 lanes publish the transformed tile to the UMMA issuer here.
  auto with_sf_full_kv_barriers = utils::PatternVisitor([&](const uint32_t& i) {
    return barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + i;
  });
  const auto tmem_barrier_ptr =
      barrier_ptr + kNumQStages * 2 + kNumKVStages * 3;
  auto full_tmem_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return tmem_barrier_ptr + i; });
  auto empty_tmem_barriers = utils::PatternVisitor(
      [&](const uint32_t& i) { return tmem_barrier_ptr + kNumTmemStages + i; });
  auto tmem_ptr_in_smem =
      reinterpret_cast<uint32_t*>(tmem_barrier_ptr + kNumTmemStages * 2);
  // 16B-aligned: the score-bank engine stores one STS.64 per
  // (thread, block) and reads float2; tmem_ptr+1 word is 4 mod 8.
  auto emit_smem_records = reinterpret_cast<uint32_t*>(tmem_ptr_in_smem + 4);
  // Emit region size: the BQ2 engine's score bank (kWin float2 rows per
  // thread) is LARGER than the BQ4 staged ring; the prep/tighten scratch
  // must sit beyond whichever this instantiation uses, or the bank's
  // STS.64 stream clobbers the histograms (the self-tighten recall
  // collapse was exactly this overlap at kWin=32).
  constexpr uint32_t kGraftBankWords = 32u * kNumMathThreads * 2u;
  constexpr uint32_t kGraftRingWords =
      (kNumMathThreads / 32u) * BLOCK_Q * kEmitLaneSlots * 32u;
  constexpr uint32_t kEmitRegionWords =
      (BLOCK_Q == 2 && kGraftBankWords > kGraftRingWords) ? kGraftBankWords
                                                          : kGraftRingWords;
  auto prep_hist = reinterpret_cast<int*>(emit_smem_records + kEmitRegionWords);
  auto prep_done = prep_hist + 256;
  // Self-tightening scratch: per-CTA per-row bucket histogram of drained
  // candidates plus the published (tightened) gate bits, both live only
  // in the kSelfTighten instantiation.
  auto tighten_hist = prep_done + 1;           // [BLOCK_Q][256]
  auto tighten_gate = tighten_hist + 2 * 256;  // [BLOCK_Q]
  // BQ2-only CTA-local candidate cursors.  A q-block is owned by exactly
  // one persistent CTA, so global cand_cnt is needed only at the q-block
  // boundaries; window-local reservations can stay in shared memory.
  auto cta_cand_cnt = tighten_gate + 2;  // [2]

  constexpr uint32_t kNumAccumTmemCols = BLOCK_Q * kNumHeads * kNumTmemStages;
  constexpr uint32_t kNumTmemCols =
      utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQ / 32 +
                                       kNumSFKV / 32>();
  constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
  constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
  DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

  if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      full_q_barriers[i]->init(1);
      empty_q_barriers[i]->init(kNumMathThreads + 32);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      full_kv_barriers[i]->init(1);
      empty_kv_barriers[i]->init(1);
      with_sf_full_kv_barriers[i]->init(32);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumTmemStages; ++i) {
      full_tmem_barriers[i]->init(1);
      empty_tmem_barriers[i]->init(128);
    }
    cutlass::arch::fence_barrier_init();
  }

  if (warp_idx == kSpecWarpStart + 2)
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
  __syncthreads();

  const uint32_t num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);
  uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
  auto load_schedule =
      [&](const uint32_t& q_idx) -> cute::tuple<uint32_t, uint32_t> {
    uint32_t start = cute::numeric_limits<uint32_t>::max();
    uint32_t end = cute::numeric_limits<uint32_t>::min();
#pragma unroll
    for (uint32_t i = 0; i < BLOCK_Q; ++i) {
      const auto row_idx = min(q_idx * BLOCK_Q + i, seq_len - 1);
      seq_k_start[i] = min(cu_seq_len_k_start[row_idx], seq_len_kv);
      seq_k_end[i] = min(cu_seq_len_k_end[row_idx], seq_len_kv);
      if (q_idx * BLOCK_Q + i >= seq_len) {
        // Padded row of a ragged final q-block: empty range so the
        // range check (never elided: rs_max lands at seq_len_kv)
        // rejects every token.
        seq_k_start[i] = seq_len_kv;
        seq_k_end[i] = 0;
      }
      start = min(start, seq_k_start[i]);
      end = max(end, seq_k_end[i]);
    }
    start = start / 4 * 4;  // TMA alignment for SF KV
    const uint32_t nkv =
        (end > start) ? math::ceil_div(end - start, BLOCK_KV) : 0;
    return {start, nkv};
  };

  auto make_pipeline = [](const uint32_t& num_stages) {
    return [iter_idx = 0u,
            num_stages](const uint32_t& step =
                            1) mutable -> cute::tuple<uint32_t, uint32_t> {
      uint32_t current_idx = iter_idx;
      iter_idx += step;
      return {current_idx % num_stages, (current_idx / num_stages) & 1};
    };
  };
  auto advance_q_pipeline = make_pipeline(kNumQStages);
  auto advance_kv_pipeline = make_pipeline(kNumKVStages);
  auto advance_tmem_pipeline = make_pipeline(kNumTmemStages);

  constexpr uint32_t kNumSpecializedRegisters = 56;
  constexpr uint32_t kNumMathRegisters = 224;
  constexpr uint32_t kKvInnerBytes = kHeadDim / 2;

  if (warp_idx == kSpecWarpStart) {
    // TMA warp for Q + SF-Q + weights
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    if (cute::elect_one_sync()) {
      for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
        CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
        empty_q_barriers[q_stage_idx]->wait(q_phase ^ 1);
        tma::copy<kKvInnerBytes, BLOCK_Q * kNumHeads, kKvInnerBytes>(
            &tensor_map_q, full_q_barriers[q_stage_idx], smem_q[q_stage_idx], 0,
            q_idx * BLOCK_Q * kNumHeads);
        tma::copy<BLOCK_Q * kNumHeads, 1, 0>(
            &tensor_map_sf_q, full_q_barriers[q_stage_idx],
            smem_sf_q[q_stage_idx], q_idx * BLOCK_Q * kNumHeads, 0);
        tma::copy<kNumHeads, BLOCK_Q, 0>(
            &tensor_map_weights, full_q_barriers[q_stage_idx],
            smem_weights[q_stage_idx], 0, q_idx * BLOCK_Q);
        full_q_barriers[q_stage_idx]->arrive_and_expect_tx(
            SMEM_Q_SIZE_PER_STAGE + kRealNumSFQ * sizeof(int) +
            SMEM_WEIGHT_SIZE_PER_STAGE);
      }
    }
  } else if (warp_idx == kSpecWarpStart + 1) {
    // TMA warp for KV + SF-KV (one transaction barrier per stage)
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    if (cute::elect_one_sync()) {
      for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
        CUTE_TIE_DECL(load_schedule(q_idx), kv_start, num_kv_blocks);
        for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++kv_idx) {
          CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
          empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);
          tma::copy<kKvInnerBytes, BLOCK_KV, kKvInnerBytes>(
              &tensor_map_kv, full_kv_barriers[kv_stage_idx],
              smem_kv[kv_stage_idx], 0, kv_start + kv_idx * BLOCK_KV);
          tma::copy<BLOCK_KV, 1, 0>(
              &tensor_map_sf_kv, full_kv_barriers[kv_stage_idx],
              reinterpret_cast<uint8_t*>(smem_sf_kv[kv_stage_idx]),
              kv_start + kv_idx * BLOCK_KV, 0);
          full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(
              SMEM_KV_SIZE_PER_STAGE + BLOCK_KV * sizeof(int));
        }
      }
    }
  } else if (warp_idx == kSpecWarpStart + 2) {
    // UMMA warp: inline SF transposes + UTCCP + MMA issue
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

    auto utccp_transpose = [&](uint32_t* smem_ptr) {
      DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128,
                       "Invalid aligned elements");
      uint32_t values[4];
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        values[i] =
            ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
      __syncwarp();
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)),
                       values[i]);
    };

    auto bs_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
        cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
        cutlass::float_ue8m0_t, UMMA_M, UMMA_N, cute::UMMA::Major::K,
        cute::UMMA::Major::K>();
    auto sf_desc = mma::sm100::make_sf_desc(nullptr);

    for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
      CUTE_TIE_DECL(load_schedule(q_idx), kv_start, num_kv_blocks);
      (void)kv_start;
      CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
      full_q_barriers[q_stage_idx]->wait(q_phase);

#pragma unroll
      for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++i) {
        auto smem_ptr = smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
        utccp_transpose(smem_ptr);
        cutlass::arch::fence_view_async_shared();
        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
        if (cute::elect_one_sync())
          cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc,
                                                    kTmemStartColOfSFQ + i * 4);
        __syncwarp();
      }

      for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++kv_idx) {
        CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
        with_sf_full_kv_barriers[kv_stage_idx]->wait(kv_phase);
        ptx::tcgen05_after_thread_sync();

        if (cute::elect_one_sync()) {
#pragma unroll
          for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++i) {
            mma::sm100::replace_smem_desc_addr(
                sf_desc, smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems);
            cute::SM100_UTCCP_4x32dp128bit_1cta::copy(
                sf_desc, kTmemStartColOfSFKV + i * 4);
          }
#pragma unroll
          for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
            CUTE_TIE_DECL(advance_tmem_pipeline(), tmem_stage_idx, tmem_phase);
            const uint32_t tmem_addr = tmem_stage_idx * UMMA_N;
            empty_tmem_barriers[tmem_stage_idx]->wait(tmem_phase ^ 1);
            ptx::tcgen05_after_thread_sync();
#pragma unroll
            for (uint32_t k = 0; k < kHeadDim / kUmmaKFp4; ++k) {
              auto rdesc = mma::sm100::make_runtime_instr_desc_with_sf_id(
                  bs_instr_desc, k * 2, k * 2);
              DG_STATIC_ASSERT(kHeadDim == 128, "Invalid head dim");
              auto a_desc = mma::sm100::make_smem_desc(
                  cute::UMMA::LayoutType::SWIZZLE_64B,
                  smem_kv[kv_stage_idx] + i * UMMA_M * (kHeadDim / 2) +
                      k * kUmmaKFp4 / 2,
                  8 * (kHeadDim / 2), 0);
              auto b_desc = mma::sm100::make_smem_desc(
                  cute::UMMA::LayoutType::SWIZZLE_64B,
                  smem_q[q_stage_idx] + k * kUmmaKFp4 / 2, 8 * (kHeadDim / 2),
                  0);
              ptx::SM100_MMA_MXF4_SS::fma(a_desc, b_desc, tmem_addr, k, rdesc,
                                          kTmemStartColOfSFKV + i * 4,
                                          kTmemStartColOfSFQ);
            }
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
                "cluster.b64 [%0];" ::"r"(cute::cast_smem_ptr_to_uint(
                    full_tmem_barriers[tmem_stage_idx])));
          }
        } else {
          advance_tmem_pipeline(kNumMathWarpGroups);
        }
        cutlass::arch::umma_arrive(
            reinterpret_cast<uint64_t*>(empty_kv_barriers[kv_stage_idx]));
      }
      empty_q_barriers[q_stage_idx]->arrive();
    }
  } else if (warp_idx == kSpecWarpStart + 3) {
    // Dedicated KV-SF transposer, matching DeepGEMM's proven
    // full_barrier -> warp transpose/fence -> with_sf_full_barrier
    // producer chain. This takes the two 128-element transpose tiles per
    // KV block off the UMMA issue warp while leaving Q-SF (once/q-block)
    // local to the issuer.
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    auto utccp_transpose = [&](uint32_t* smem_ptr) {
      DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128,
                       "Invalid aligned elements");
      uint32_t values[4];
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        values[i] =
            ptx::ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
      __syncwarp();
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        ptx::st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)),
                       values[i]);
    };

    for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
      CUTE_TIE_DECL(load_schedule(q_idx), kv_start, num_kv_blocks);
      (void)kv_start;
      for (uint32_t kv_idx = 0; kv_idx < num_kv_blocks; ++kv_idx) {
        CUTE_TIE_DECL(advance_kv_pipeline(), kv_stage_idx, kv_phase);
        full_kv_barriers[kv_stage_idx]->wait(kv_phase);
#pragma unroll
        for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++i) {
          utccp_transpose(smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems);
        }
        cutlass::arch::fence_view_async_shared();
        with_sf_full_kv_barriers[kv_stage_idx]->arrive(0u);
      }
    }
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

    const auto math_warpgroup_idx = warpgroup_idx;
    const auto math_thread_idx = threadIdx.x;

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

    advance_tmem_pipeline(math_warpgroup_idx);

    float weights[BLOCK_Q][kNumHeads];
    float o_reg[BLOCK_Q], vth_reg[BLOCK_Q];
    uint32_t kstart_reg[BLOCK_Q], kspan_reg[BLOCK_Q];
    const unsigned FULL = 0xffffffffu;
    constexpr uint32_t kEmitChunk = kEmitChunkBlocks;

    for (uint32_t q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs) {
      CUTE_TIE_DECL(load_schedule(q_idx), kv_start, num_kv_blocks);
      CUTE_TIE_DECL(advance_q_pipeline(), q_stage_idx, q_phase);
      full_q_barriers[q_stage_idx]->wait(q_phase);

      if constexpr (BLOCK_Q == 2) {
        if (threadIdx.x < BLOCK_Q) {
          const uint32_t row_q = q_idx * BLOCK_Q + threadIdx.x;
          cta_cand_cnt[threadIdx.x] =
              row_q < seq_len ? __ldcg(cand_cnt + row_q) : 0;
        }
        cutlass::arch::NamedBarrier(kNumMathThreads, 4).sync();
#pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++i) {
          const uint32_t rq = min(q_idx * BLOCK_Q + i, seq_len - 1);
          const float inv = inv_delta[rq];
          vth_reg[i] = -origin[rq] * inv;
          o_reg[i] = static_cast<float>(__ldcg(th_bucket + rq) + 1);
          kstart_reg[i] = seq_k_start[i];
          kspan_reg[i] =
              seq_k_end[i] > seq_k_start[i] ? seq_k_end[i] - seq_k_start[i] : 0;
#pragma unroll
          for (uint32_t j = 0; j < kNumHeads; ++j)
            weights[i][j] =
                ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j) *
                -inv;
        }
        // Score-bank + splice engine (H=64/BQ2): the inner kv loop is
        // dense-isomorphic — tmem wait + loads + reduce + one STS.64 of
        // the raw bq pair (scalar STS pairs contend with tmem_load on
        // the MIO pipe: 2x STS.32 measured +1.9ms vs one STS.64). All
        // gating/counting/draining runs in a converged per-window
        // splice over the smem bank; payloads are bit-identical to the
        // staged-ring path (same bq bits -> same fp24 code).
        constexpr uint32_t kWin = kGraftBankWords / (kNumMathThreads * 2u);
        static_assert(kWin == 32, "bank layout and window tied");
        float2* const bank2 =
            reinterpret_cast<float2*>(emit_smem_records) + math_thread_idx;
        int o_bits0 = __float_as_int(o_reg[0]);
        int o_bits1 = __float_as_int(o_reg[1 < BLOCK_Q ? 1 : 0]);
        if constexpr (kSelfTighten) {
          // Warm-start this q-block's histogram from the seed sample
          // counts (ring-warmstart semantics: same (origin, inv)
          // bucket space, exact-once scan after the prefix), publish
          // the seed gate, and sync once.
          for (uint32_t i = threadIdx.x; i < 2u * 256u; i += kNumMathThreads) {
            const uint32_t r = q_idx * BLOCK_Q + i / 256u;
            tighten_hist[i] =
                (seed_bcount != nullptr && r < seq_len)
                    ? __ldcg(seed_bcount + static_cast<uint64_t>(r) * 256u +
                             (i & 255u))
                    : 0;
          }
          if (threadIdx.x < BLOCK_Q)
            tighten_gate[threadIdx.x] = __float_as_int(o_reg[threadIdx.x]);
          cutlass::arch::NamedBarrier(kNumMathThreads, 3).sync();
        }
        const uint32_t row_q0 = q_idx * BLOCK_Q;
        const uint32_t row_q1 = row_q0 + 1;
        const uint32_t rs_max = max(kstart_reg[0], kstart_reg[1]);
        const uint32_t re_min =
            min(kstart_reg[0] + kspan_reg[0], kstart_reg[1] + kspan_reg[1]);
        const auto scan_windows = [&](auto range_check_c) {
          constexpr bool kRangeCheck = decltype(range_check_c)::value != 0;
          for (uint32_t win_start = 0; win_start < num_kv_blocks;
               win_start += kWin) {
            const uint32_t b_end = min(kWin, num_kv_blocks - win_start);
            uint32_t mask0 = 0, mask1 = 0, win_bit = 1u;
            for (uint32_t b = 0; b < b_end; ++b) {
              CUTE_TIE_DECL(advance_tmem_pipeline(kNumMathWarpGroups),
                            tmem_stage_idx, tmem_phase);
              full_tmem_barriers[tmem_stage_idx]->wait(tmem_phase);
              ptx::tcgen05_after_thread_sync();
              const auto tmem_start = tmem_stage_idx * UMMA_N;
              float bqv[BLOCK_Q];
#pragma unroll
              for (uint32_t i = 0; i < BLOCK_Q; ++i) {
                float accum[kNumHeads];
                tmem_load(cute::Int<kNumHeads>{}, tmem_start + i * kNumHeads,
                          accum);
                if (i == BLOCK_Q - 1) {
                  ptx::tcgen05_before_thread_sync();
                  empty_tmem_barriers[tmem_stage_idx]->arrive();
                }
                auto sum_0 = make_float2(0, 0);
                auto sum_1 = make_float2(0, 0);
                const auto transform = [&](const uint32_t& j,
                                           const float2& sum) {
                  auto a =
                      make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));
                  auto b2 = make_float2(weights[i][j], weights[i][j + 1]);
                  return __ffma2_rn(a, b2, sum);
                };
#pragma unroll
                for (uint32_t j = 0; j < kNumHeads; j += 4) {
                  sum_0 = transform(j, sum_0);
                  sum_1 = transform(j + 2, sum_1);
                }
                auto sum = __fadd2_rn(sum_0, sum_1);
                bqv[i] = (sum.x + sum.y) + vth_reg[i];
              }
              bank2[b * kNumMathThreads] =
                  make_float2(bqv[0], bqv[1 < BLOCK_Q ? 1 : 0]);
              // Gate in registers while the scores are still live:
              // costs the measured +0.78ms gate-wiring rung but
              // deletes the splice's full-bank re-read (measured
              // 1.15ms of MIO LDS traffic).
              const uint32_t kvo_g =
                  kv_start + (win_start + b) * BLOCK_KV + math_thread_idx;
              bool g0 = __float_as_int(bqv[0]) < o_bits0;
              bool g1 = __float_as_int(bqv[1 < BLOCK_Q ? 1 : 0]) < o_bits1;
              if constexpr (kRangeCheck) {
                g0 = g0 and (kvo_g - kstart_reg[0]) < kspan_reg[0];
                g1 = g1 and (kvo_g - kstart_reg[1 < BLOCK_Q ? 1 : 0]) <
                                kspan_reg[1 < BLOCK_Q ? 1 : 0];
              }
              mask0 |= g0 ? win_bit : 0u;
              mask1 |= g1 ? win_bit : 0u;
              win_bit <<= 1;
            }

            // Splice: converged (b_end is CTA-uniform), no tmem waits,
            // masks already in registers.
            const uint32_t win_kv_base =
                kv_start + win_start * BLOCK_KV + math_thread_idx;
            const bool win_any = __ballot_sync(FULL, mask0 | mask1) != 0u;
            if (win_any) {
              const uint32_t cnt0 = __popc(mask0);
              const uint32_t cnt1 = __popc(mask1);
              uint32_t inclusive = cnt0 | (cnt1 << 16);
#pragma unroll
              for (int delta = 1; delta < 32; delta <<= 1) {
                const uint32_t other = __shfl_up_sync(FULL, inclusive, delta);
                if (lane_idx >= static_cast<uint32_t>(delta))
                  inclusive += other;
              }
              const uint32_t tot0 = __shfl_sync(FULL, inclusive, 31) & 0xffffu;
              const uint32_t tot1 = __shfl_sync(FULL, inclusive, 31) >> 16;
              const uint32_t excl0 = (inclusive & 0xffffu) - cnt0;
              const uint32_t excl1 = (inclusive >> 16) - cnt1;
              int base01 = 0;
              if (lane_idx < BLOCK_Q) {
                const uint32_t row_q = row_q0 + lane_idx;
                const uint32_t tot = lane_idx == 0 ? tot0 : tot1;
                if (row_q < seq_len && tot != 0) {
                  base01 =
                      atomicAdd(cta_cand_cnt + lane_idx, static_cast<int>(tot));
                }
              }
              const int base0 = __shfl_sync(FULL, base01, 0);
              const int base1 = __shfl_sync(FULL, base01, 1);
              uint32_t rank0 = excl0;
              while (mask0 != 0u) {
                const uint32_t b = __ffs(mask0) - 1u;
                mask0 &= mask0 - 1u;
                const int out = base0 + static_cast<int>(rank0++);
                const float sv0 = bank2[b * kNumMathThreads].x;
                if constexpr (kSelfTighten) {
                  const int bb = static_cast<int>(sv0);
                  atomicAdd(tighten_hist + (bb < 0 ? 0 : (bb > 255 ? 255 : bb)),
                            1);
                }
                if (row_q0 < seq_len && out < static_cast<int>(cand_cap)) {
                  const uint64_t pos =
                      static_cast<uint64_t>(row_q0) * cand_cap + out;
                  store_candidate(cand_val + pos, cand_idx + pos, sv0,
                                  win_kv_base + b * BLOCK_KV);
                }
              }
              uint32_t rank1 = excl1;
              while (mask1 != 0u) {
                const uint32_t b = __ffs(mask1) - 1u;
                mask1 &= mask1 - 1u;
                const int out = base1 + static_cast<int>(rank1++);
                const float sv1 = bank2[b * kNumMathThreads].y;
                if constexpr (kSelfTighten) {
                  const int bb = static_cast<int>(sv1);
                  atomicAdd(
                      tighten_hist + 256 + (bb < 0 ? 0 : (bb > 255 ? 255 : bb)),
                      1);
                }
                if (row_q1 < seq_len && out < static_cast<int>(cand_cap)) {
                  const uint64_t pos =
                      static_cast<uint64_t>(row_q1) * cand_cap + out;
                  store_candidate(cand_val + pos, cand_idx + pos, sv1,
                                  win_kv_base + b * BLOCK_KV);
                }
              }
            }

            if constexpr (kSelfTighten) {
              // Window-end retighten: the ring rule computed inline.
              // The K-th cumulative bucket over ALL drained records
              // (seed candidates were already >= K at their own edge,
              // and this histogram only ADDS suffix records below the
              // current gate, so the published edge keeps certifying
              // >= K records: monotone tightening, recall-safe).
              cutlass::arch::NamedBarrier(kNumMathThreads, 3).sync();
              if (warp_idx == 0) {
#pragma unroll
                for (uint32_t i = 0; i < BLOCK_Q; ++i) {
                  const int* h = tighten_hist + i * 256;
                  int bins[8];
                  int lane_sum = 0;
#pragma unroll
                  for (int k = 0; k < 8; ++k) {
                    bins[k] = h[lane_idx * 8 + k];
                    lane_sum += bins[k];
                  }
                  int incl = lane_sum;
#pragma unroll
                  for (int off = 1; off < 32; off <<= 1) {
                    const int y = __shfl_up_sync(0xffffffffu, incl, off);
                    if (lane_idx >= static_cast<uint32_t>(off)) incl += y;
                  }
                  const int excl = incl - lane_sum;
                  int th_new = 256;
                  int run = excl;
#pragma unroll
                  for (int k = 0; k < 8; ++k) {
                    const int nxt = run + bins[k];
                    if (run < static_cast<int>(seed_topk) &&
                        static_cast<int>(seed_topk) <= nxt)
                      th_new = lane_idx * 8 + k;
                    run = nxt;
                  }
#pragma unroll
                  for (int off = 16; off > 0; off >>= 1)
                    th_new =
                        min(th_new, __shfl_xor_sync(0xffffffffu, th_new, off));
                  if (lane_idx == 0 && th_new < 256) {
                    const int cand_bits =
                        __float_as_int(static_cast<float>(th_new + 1));
                    if (cand_bits < tighten_gate[i])
                      tighten_gate[i] = cand_bits;
                  }
                }
              }
              cutlass::arch::NamedBarrier(kNumMathThreads, 3).sync();
              o_bits0 = tighten_gate[0];
              o_bits1 = tighten_gate[1 < BLOCK_Q ? 1 : 0];
            }
          }
        };
        const bool whole_range_interior =
            kv_start >= rs_max and
            kv_start + num_kv_blocks * BLOCK_KV <= re_min;
        if (__builtin_expect(whole_range_interior, 1)) {
          scan_windows(cute::Int<0>{});
        } else {
          scan_windows(cute::Int<1>{});
        }
      } else {
        uint32_t emit_lane_counts = 0;
#pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++i) {
          const uint32_t rq = min(q_idx * BLOCK_Q + i, seq_len - 1);
          const float inv = inv_delta[rq];
          vth_reg[i] = -origin[rq] * inv;
          o_reg[i] = static_cast<float>(__ldcg(th_bucket + rq) + 1);
          kstart_reg[i] = seq_k_start[i];
          kspan_reg[i] =
              seq_k_end[i] > seq_k_start[i] ? seq_k_end[i] - seq_k_start[i] : 0;
#pragma unroll
          for (uint32_t j = 0; j < kNumHeads; ++j)
            weights[i][j] =
                ptx::ld_shared(smem_weights[q_stage_idx] + i * kNumHeads + j) *
                -inv;
        }
        uint32_t rs_max = 0, re_min = 0xffffffffu;
#pragma unroll
        for (uint32_t i = 0; i < BLOCK_Q; ++i) {
          rs_max = max(rs_max, kstart_reg[i]);
          re_min = min(re_min, kstart_reg[i] + kspan_reg[i]);
        }

        for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
             ++kv_block_idx) {
          CUTE_TIE_DECL(advance_tmem_pipeline(kNumMathWarpGroups),
                        tmem_stage_idx, tmem_phase);
          full_tmem_barriers[tmem_stage_idx]->wait(tmem_phase);
          ptx::tcgen05_after_thread_sync();

          const auto kv_offset =
              kv_start + kv_block_idx * BLOCK_KV + math_thread_idx;
          const auto tmem_start = tmem_stage_idx * UMMA_N;
          uint32_t pass_bits = 0;
          float v_row[BLOCK_Q];
          constexpr uint32_t kTmemRowsEff =
              (128u / kNumHeads) < BLOCK_Q ? (128u / kNumHeads) : BLOCK_Q;
          DG_STATIC_ASSERT(BLOCK_Q % kTmemRowsEff == 0,
                           "BLOCK_Q must divide TMEM load groups");
          const uint32_t kv_base = kv_start + kv_block_idx * BLOCK_KV;
          const bool interior =
              (kv_base >= rs_max) && (kv_base + BLOCK_KV <= re_min);
// Per-row kNumHeads-wide loads with a reused accum[64],
// exactly like the dense reference: the fused 128-wide load
// needs 128 accum regs live on top of the 128 weight regs,
// blowing the 224 budget into local-memory spills (the
// measured +23%% zero-pass tax).
#define GRAFT_SCORE_ROWS(RANGE_CHECK)                                     \
  _Pragma("unroll") for (uint32_t i = 0; i < BLOCK_Q; ++i) {              \
    float accum[kNumHeads];                                               \
    tmem_load(cute::Int<kNumHeads>{}, tmem_start + i * kNumHeads, accum); \
    if (i == BLOCK_Q - 1) {                                               \
      ptx::tcgen05_before_thread_sync();                                  \
      empty_tmem_barriers[tmem_stage_idx]->arrive();                      \
    }                                                                     \
    auto sum_0 = make_float2(0, 0);                                       \
    auto sum_1 = make_float2(0, 0);                                       \
    const auto transform = [&](const uint32_t& j, const float2& sum) {    \
      auto a = make_float2(fmaxf(accum[j], 0), fmaxf(accum[j + 1], 0));   \
      auto b = make_float2(weights[i][j], weights[i][j + 1]);             \
      return __ffma2_rn(a, b, sum);                                       \
    };                                                                    \
    _Pragma("unroll") for (uint32_t j = 0; j < kNumHeads; j += 4) {       \
      sum_0 = transform(j, sum_0);                                        \
      sum_1 = transform(j + 2, sum_1);                                    \
    }                                                                     \
    auto sum = __fadd2_rn(sum_0, sum_1);                                  \
    const float bq = (sum.x + sum.y) + vth_reg[i];                        \
    v_row[i] = bq;                                                        \
    bool g = __float_as_int(bq) < __float_as_int(o_reg[i]);               \
    if constexpr (RANGE_CHECK)                                            \
      g = g and ((kv_offset - kstart_reg[i]) < kspan_reg[i]);             \
    pass_bits |= g ? (1u << i) : 0u;                                      \
  }
          if (interior) {
            GRAFT_SCORE_ROWS(false)
          } else {
            GRAFT_SCORE_ROWS(true)
          }
#undef GRAFT_SCORE_ROWS

          if (pass_bits != 0) {
#pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++i) {
              if ((pass_bits >> i) & 1u) {
                const uint32_t candidate_score_bits =
                    candidate_fp24_code(v_row[i]) << 8;
                const uint32_t count = (emit_lane_counts >> (i * 8)) & 0xffu;
                if (count < kEmitLaneSlots) {
                  const uint32_t pos =
                      ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + count) *
                          32u +
                      lane_idx;
                  const uint32_t local_block = kv_block_idx % kEmitChunk;
                  emit_smem_records[pos] = candidate_score_bits | local_block;
                  emit_lane_counts += 1u << (i * 8);
                } else {
                  const uint32_t row_q = q_idx * BLOCK_Q + i;
                  if (row_q >= seq_len) continue;
                  const int out = atomicAdd(cand_cnt + row_q, 1);
                  if (out < static_cast<int>(cand_cap)) {
                    const uint64_t out_base =
                        static_cast<uint64_t>(row_q) * cand_cap;
                    store_candidate_record(
                        &cand_val[out_base + out], &cand_idx[out_base + out],
                        candidate_score_bits >> 8, kv_offset);
                  }
                }
              }
            }
          }

          if (((kv_block_idx + 1) % kEmitChunk) == 0 ||
              kv_block_idx + 1 == num_kv_blocks) {
            int my_row_base = 0;
            uint32_t my_row_total = 0;
#pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++i) {
              const uint32_t count = (emit_lane_counts >> (i * 8)) & 0xffu;
              const uint32_t total = __reduce_add_sync(FULL, count);
              if (lane_idx == i) my_row_total = total;
            }
            if (lane_idx < BLOCK_Q) {
              const uint32_t row_q = q_idx * BLOCK_Q + lane_idx;
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
                kv_start + (kv_block_idx / kEmitChunk) * kEmitChunk * BLOCK_KV;
#pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++i) {
              const uint32_t count = (emit_lane_counts >> (i * 8)) & 0xffu;
              const uint32_t inclusive =
                  i < 2 ? ((inclusive01 >> ((i & 1u) * 16)) & 0xffffu)
                        : ((inclusive23 >> ((i & 1u) * 16)) & 0xffffu);
              const uint32_t offset = inclusive - count;
              const uint32_t row_q = q_idx * BLOCK_Q + i;
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
                const uint32_t record = emit_smem_records[local_pos];
                const int out = out_base + static_cast<int>(offset) + slot;
                const uint32_t record_payload = (record & 0xffffff00u) >> 8;
                const uint32_t kvo = emit_window_kv_base +
                                     ((record & 0xffu) * BLOCK_KV) +
                                     math_thread_idx;
                const uint64_t out_pos =
                    static_cast<uint64_t>(row_q) * cand_cap + out;
                store_candidate_record(&cand_val[out_pos], &cand_idx[out_pos],
                                       record_payload, kvo);
              }
            }
            emit_lane_counts = 0;
          }
        }
      }
      if constexpr (BLOCK_Q == 2) {
        cutlass::arch::NamedBarrier(kNumMathThreads, 4).sync();
        if (threadIdx.x < BLOCK_Q) {
          const uint32_t row_q = q_idx * BLOCK_Q + threadIdx.x;
          if (row_q < seq_len)
            cand_cnt[row_q] = static_cast<int32_t>(ptx::ld_shared(
                reinterpret_cast<uint32_t*>(cta_cand_cnt) + threadIdx.x));
        }
      }
      empty_q_barriers[q_stage_idx]->arrive();
    }

    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
}

}  // namespace dsa_litetopk
// namespace dsa_litetopk
