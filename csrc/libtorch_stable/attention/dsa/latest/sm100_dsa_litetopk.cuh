// LiteTopK DSA scoring kernel V3 = "hybrid":
//   * scoring loop of DeepGEMM 2.5 (commit 891d57b, the vLLM-pinned version):
//     per-q-block weights held in REGISTERS, per-row 32-element TMEM loads
//     with early UMMA release, tight tcgen05 fencing -- the generation that
//     makes the official kernel fast at large Q;
//   * scheduling of our V1: NON-persistent KV-split. blockIdx.x = q-block,
//     blockIdx.y = KV split window. This is what keeps all 148 SMs busy on the
//     tiny-Q chunks vLLM actually produces at long context (its 512MB logits
//     budget shrinks Q to ~128 at S=1M, where a persistent grid would idle
//     116/148 SMs);
//   * LiteTopK sparse epilogue (batched-vote emit, strided gate reload,
//     warp-local candidate queues) and the spare-warp threshold-refresh daemon
//     (V1 semantics: one q-block per CTA, fixed rows).
//
// Ragged Q (vLLM chunks) handled by forcing an empty KV range on padded rows.

#pragma once

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

// Production candidate ABI: a six-byte global record. cand_val stores the
// low 16 bits of the non-negative FP32 high24 score code; cand_idx stores its
// high eight score bits above the exact 20-bit KV index. The local ring uses
// one uint32_t containing high24 score bits and an eight-bit block-in-window.
using CandidateValue = uint16_t;
constexpr uint32_t kCandidateIndexBits = 20;
constexpr uint32_t kCandidateIndexMask = (1u << kCandidateIndexBits) - 1u;

CUTLASS_DEVICE uint32_t candidate_pack_index(const uint32_t payload,
                                             const uint32_t kv_index) {
  return (kv_index & kCandidateIndexMask) |
         ((payload >> 16) << kCandidateIndexBits);
}

CUTLASS_DEVICE uint32_t candidate_fp24_code(const float value) {
  // The certified boundary path emits every bucket below th directly, so
  // bucket-0 negatives never participate in the boundary rank. Canonicalize
  // them to +0 and retain the monotonic high 24 bits for non-negative values.
  const int32_t signed_bits = __float_as_int(value);
  return static_cast<uint32_t>(max(signed_bits, 0)) >> 8;
}

CUTLASS_DEVICE uint32_t candidate_load_score_code(const CandidateValue value,
                                                  const int32_t packed_idx) {
  return (static_cast<uint32_t>(packed_idx) >> kCandidateIndexBits) << 16 |
         static_cast<uint32_t>(value);
}

CUTLASS_DEVICE float candidate_decode_score(const CandidateValue value,
                                            const int32_t packed_idx) {
  const uint32_t code = candidate_load_score_code(value, packed_idx);
  return __uint_as_float(code << 8);
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
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_kv_scales,
                               const __grid_constant__ cute::TmaDescriptor
                                   tensor_map_weights) {
  const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto warpgroup_idx = warp_idx / 4;
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
  constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
  constexpr uint32_t kNumUmmaStages = kUmmaStages;
  constexpr uint32_t kNumUmmaBuffers = kNumMathWarpGroups * kNumUmmaStages;
  DG_STATIC_ASSERT(BLOCK_Q == 4 && kNumMathWarps == 8,
                   "LiteTopK requires BLOCK_Q=4 and 8 math warps");
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
      BLOCK_Q * kNumHeads * kHeadDim * sizeof(__nv_fp8_e4m3);
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * sizeof(float);
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE =
      BLOCK_KV * kHeadDim * sizeof(__nv_fp8_e4m3);
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

  constexpr uint32_t kNumTmemCols = BLOCK_Q * kNumHeads * kNumUmmaBuffers;
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
           (kNumQStages * 2 + kNumKVStages * 2 + kNumUmmaBuffers + i);
  });

  auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
      barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumUmmaBuffers * 2);
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
      for (uint32_t i = 0; i < kNumUmmaBuffers; ++i) {
        full_umma_barriers[i]->init(1);
        empty_umma_barriers[i]->init(128);
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem);
  }
  // One CTA owns each row. The hot-only path emits no sample seeds and scans
  // the complete KV range, so its exact histogram starts at zero in SMEM.
  for (uint32_t idx = threadIdx.x; idx < BLOCK_Q * num_buckets;
       idx += blockDim.x) {
    smem_hist[idx] = 0;
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
        // Q + weights once for this q-block.
        tma::copy<kHeadDim, BLOCK_Q * kNumHeads, kHeadDim>(
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

          tma::copy<kHeadDim, BLOCK_KV, kHeadDim>(
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
            atomicMin(sparse_gate_bits + local_row, edge);
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
      while (*scan_done_flag == 0) {
        bool tightened = false;
        for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
          tightened |= refresh_row(block_q_idx * BLOCK_Q + r, false);
        __nanosleep(tightened ? kSparseRefreshNs : kSparseRefreshIdleNs);
      }
      for (uint32_t r = spare_id; r < BLOCK_Q; r += 2)
        refresh_row(block_q_idx * BLOCK_Q + r, true);
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
        o_reg[i] = origin[rq];
        inv_reg[i] = inv_delta[rq];
        // GATE4 (user's final form): bucket-space FLOAT end to end.
        // bq = fmaf(scale_kv, sum', c0) -- form-identical to the sign
        // gate; gate = INT compare of bq's BITS vs edge float(g+1)
        // bits (edge >= 1 > 0, so ALL negative bq bit-patterns
        // compare below it and pass: no sign flip needed). cand_val
        // stores bq itself (affine preserves order; select runs in
        // bucket space, indices are the only output -- the exact
        // score is never reconstructed). vth_reg = c0; o_reg is
        // repurposed at consume time to hold the edge float.
        vth_reg[i] = -o_reg[i] * inv_reg[i];
        o_reg[i] = 0.0f;  // gate closed until the first consume
        kstart_reg[i] = seq_k_start[i];
        kspan_reg[i] =
            seq_k_end[i] > seq_k_start[i] ? seq_k_end[i] - seq_k_start[i] : 0;
      }
// Fold -inv into the register weights: the whole ReLU-weighted
// chain then accumulates directly in bucket units. 128 FMULs
// once per qb, amortized over thousands of kv blocks.
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; ++j) weights[i][j] *= -inv_reg[i];
      }
      // Interior-block bounds (warp-uniform): a kv block fully inside
      // every row's [ks, ke) needs no per-element range checks.
      uint32_t rs_max = 0, re_min = 0xffffffffu;
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        rs_max = max(rs_max, kstart_reg[i]);
        re_min = min(re_min, kstart_reg[i] + kspan_reg[i]);
      }

      // The initial exact-subset edge was copied to the shared mailbox
      // before the CTA-wide boot barrier. Thereafter the spare-warp
      // daemon may only decrease it. A racing/stale shared load is a
      // looser edge and therefore recall-safe.
      const float4 initial_gate =
          ptx::ld_shared(reinterpret_cast<const float4*>(sparse_gate_bits));
      o_reg[0] = initial_gate.x;
      o_reg[1] = initial_gate.y;
      o_reg[2] = initial_gate.z;
      o_reg[3] = initial_gate.w;

      for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
           ++kv_block_idx) {
        const uint32_t kvg = kv_block_idx;
        CUTE_TIE_DECL(get_kv_pipeline(kvg), kv_stage_idx, kv_phase);

        if (kv_block_idx != 0 && (kv_block_idx % kGateStride) == 0) {
          const float4 gate =
              ptx::ld_shared(reinterpret_cast<const float4*>(sparse_gate_bits));
          o_reg[0] = gate.x;
          o_reg[1] = gate.y;
          o_reg[2] = gate.z;
          o_reg[3] = gate.w;
        }
        full_kv_barriers[kv_stage_idx]->wait(kv_phase);

        float scale_kv =
            ptx::ld_shared(smem_kv_scales[kv_stage_idx] + math_thread_idx);

        const uint32_t umma_buf =
            warpgroup_idx * kNumUmmaStages + (kvg % kNumUmmaStages);
        const auto tmem_start = umma_buf * UMMA_N;
        full_umma_barriers[umma_buf]->wait((kvg / kNumUmmaStages) & 1);
        ptx::tcgen05_after_thread_sync();

        empty_kv_barriers[kv_stage_idx]->arrive();

        const auto kv_offset =
            kv_start + kv_block_idx * BLOCK_KV + math_thread_idx;
        DG_STATIC_ASSERT(kNumHeads % 8 == 0, "Invalid head");

        uint32_t pass_bits = 0;
        float v_row[BLOCK_Q];

        // P1: row-PAIR TMEM loads (32dp32b64x): half the tcgen05.ld
        // instructions and half the fences on the governor loop; the
        // UMMA release also moves one row earlier.
        // Interior-block gate elision: one warp-uniform branch per
        // block picks a loop body WITHOUT the per-element range
        // checks (SASS: saves 4x VIADD + 4x ISETP per column) for the
        // >99% of blocks fully inside every row's [ks, ke).
        DG_STATIC_ASSERT(BLOCK_Q % kTmemRows == 0,
                         "BLOCK_Q must divide evenly into TMEM load groups");
// GATE4: column cost IDENTICAL to the sign gate (FFMA whose
// addend is c0 instead of th_x, ISETP on bits instead of the
// sign). NaN bq maps to a large positive pattern -> DROPPED
// (old FSETP semantics; recall check is the arbiter).
#define LITETOPK_SCORE_GATE(i, RC)                                          \
  const float bq = fmaf(scale_kv, sum.x + sum.y, vth_reg[i]);               \
  v_row[i] = bq;                                                            \
  bool g = __float_as_int(bq) < __float_as_int(o_reg[i]);                   \
  if constexpr (RC) g = g and ((kv_offset - kstart_reg[i]) < kspan_reg[i]); \
  pass_bits |= g ? (1u << i) : 0u;
        const uint32_t kv_base = kv_start + kv_block_idx * BLOCK_KV;
        const bool interior =
            (kv_base >= rs_max) && (kv_base + BLOCK_KV <= re_min);
#define LITETOPK_SCORE_ROWS(RANGE_CHECK)                                    \
  _Pragma("unroll") for (uint32_t pr = 0; pr < BLOCK_Q / kTmemRows; ++pr) { \
    float accum2[kNumHeads * kTmemRows];                                    \
    tmem_load(cute::Int<kNumHeads * kTmemRows>{},                           \
              tmem_start + pr * kTmemRows * kNumHeads, accum2);             \
    if (pr == BLOCK_Q / kTmemRows - 1) {                                    \
      ptx::tcgen05_before_thread_sync();                                    \
      empty_umma_barriers[umma_buf]->arrive();                              \
    }                                                                       \
    _Pragma("unroll") for (uint32_t k = 0; k < kTmemRows; ++k) {            \
      const uint32_t i = pr * kTmemRows + k;                                \
      const float* accum = accum2 + k * kNumHeads;                          \
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
      LITETOPK_SCORE_GATE(i, RANGE_CHECK)                                   \
    }                                                                       \
  }
        if (interior) {
          LITETOPK_SCORE_ROWS(false)
        } else {
          LITETOPK_SCORE_ROWS(true)
        }
#undef LITETOPK_SCORE_ROWS
#undef LITETOPK_SCORE_GATE

        // redux pruning: inside an active block, one redux.sync.or
        // gives the warp-wide union of hit rows, so the ballot (and
        // its queue bookkeeping) runs only for rows that actually
        // have hits (~1.3 of BLOCK_Q=4 at production density). The
        // cheap VOTE.ANY stays as the outer gate: redux costs more
        // than a vote and must not run on the ~40% inactive blocks.
        // Branches are warp-uniform (no divergence around collectives).
        // Direct per-lane log. Counts reset every chunk, so lanes with
        // different hit histories remain within a small, cache-local
        // group of slot planes. There are no warp collectives, shared
        // queues, or returning atomics on the normal path.
        if (pass_bits != 0) {
#pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            if ((pass_bits >> i) & 1u) {
              const float x = v_row[i];
              // Production record: high24 FP32 score plus the
              // block inside this 256-block window. Flush keeps
              // that code intact and reconstructs the full KVO.
              const uint32_t candidate_score_bits = candidate_fp24_code(x) << 8;
              const int candidate_braw = static_cast<int>(x);
              const uint32_t candidate_bucket =
                  static_cast<uint32_t>(max(candidate_braw, 0));
              const uint32_t count = (emit_lane_counts >> (i * 8)) & 0xffu;
              if (count < kEmitLaneSlots) {
                const uint32_t pos =
                    ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + count) * 32u +
                    lane_idx;
                const uint32_t local_block = kv_block_idx % kEmitChunkBlocks;
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
              atomicAdd(smem_hist + i * num_buckets +
                            static_cast<int>(candidate_bucket),
                        1);
            }
          }
        }

        if (((kv_block_idx + 1) % kEmitChunkBlocks) == 0 ||
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
              kv_start +
              (kv_block_idx / kEmitChunkBlocks) * kEmitChunkBlocks * BLOCK_KV;
#pragma unroll
          for (uint32_t i = 0; i < BLOCK_Q; ++i) {
            const uint32_t count = (emit_lane_counts >> (i * 8)) & 0xffu;
            const uint32_t inclusive =
                i < 2 ? ((inclusive01 >> ((i & 1u) * 16)) & 0xffffu)
                      : ((inclusive23 >> ((i & 1u) * 16)) & 0xffffu);
            const uint32_t offset = inclusive - count;
            const uint32_t row_q = block_q_idx * BLOCK_Q + i;
            const int out_base = __shfl_sync(FULL, my_row_base, i);
            const int copy_count =
                row_q < seq_len
                    ? min(static_cast<int>(count),
                          max(static_cast<int>(cand_cap) - out_base -
                                  static_cast<int>(offset),
                              0))
                    : 0;
            for (int slot = 0; slot < copy_count; ++slot) {
              const uint32_t local_pos =
                  ((warp_idx * BLOCK_Q + i) * kEmitLaneSlots + slot) * 32u +
                  lane_idx;
              const uint32_t record = emit_smem_records[local_pos];
              const int out = out_base + static_cast<int>(offset) + slot;
              const uint64_t out_pos =
                  static_cast<uint64_t>(row_q) * cand_cap + out;
              const uint32_t record_payload = (record & 0xffffff00u) >> 8;
              uint32_t kvo = emit_window_kv_base +
                             ((record & 0xffu) * BLOCK_KV) + math_thread_idx;
              if (probe_group != 0) {
                const uint32_t sup = static_cast<uint32_t>(
                    (static_cast<uint64_t>(kvo) * probe_magic) >> 42);
                kvo += min((sup + 1) * 64u, probe_add_max);
              }
              store_candidate_record(&cand_val[out_pos], &cand_idx[out_pos],
                                     record_payload, kvo);
            }
          }
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
      *scan_done_flag = 1;
    }
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
}

}  // namespace dsa_litetopk
