// SPDX-License-Identifier: MIT
// Experimental derivative of DeepGEMM 2.8 sm100_mqa_logits.cuh.
// Persistent DeepGEMM FP8 MQA producer with fixed-threshold candidate emission.
#pragma once

#include <cutlass/arch/reg_reconfig.h>
#include "handoff.cuh"

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/packing.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/clean_logits.cuh>
#include <deep_gemm/layout/mqa_logits.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>
#include "candidates.cuh"

// Shared SM100 MQA logits core plus contiguous-KV and paged entries
// Both entries use the same q / sf_q / kv / sf_kv / weights TMA signature

namespace fused_mqa_scan {
using namespace deep_gemm;

// Convert runtime valid-token count to `cute::Int` so token loops stay
// compile-time constant
template <uint32_t kBlockQ, uint32_t kCandidate = kBlockQ, typename Fn>
CUTLASS_DEVICE void dispatch_num_block_tokens(const uint32_t& num_block_tokens,
                                              Fn&& fn) {
  if constexpr (kCandidate <= 1) {
    fn(cute::Int<1>{});
  } else if (num_block_tokens >= kCandidate) {
    fn(cute::Int<kCandidate>{});
  } else {
    dispatch_num_block_tokens<kBlockQ, kCandidate - 1>(num_block_tokens,
                                                       static_cast<Fn&&>(fn));
  }
}

// Load heads using power-of-two TMEM chunks.
template <uint32_t kNumHeads, uint32_t kNumLoaded = 0>
CUTLASS_DEVICE void load_tmem_heads_decomposed(const uint32_t& tmem_col,
                                               float* accum) {
  constexpr uint32_t kNumRemaining = kNumHeads - kNumLoaded;
  if constexpr (kNumRemaining > 0) {
    constexpr uint32_t kChunk = kNumRemaining >= 64   ? 64
                                : kNumRemaining >= 32 ? 32
                                : kNumRemaining >= 16 ? 16
                                : kNumRemaining >= 8  ? 8
                                                      : 4;
    ptx::tmem_load_32dp32b<kChunk>(
        tmem_col + kNumLoaded, reinterpret_cast<uint32_t*>(accum + kNumLoaded));
    load_tmem_heads_decomposed<kNumHeads, kNumLoaded + kChunk>(tmem_col, accum);
  }
}

// Warp 0 finds the descending radix bucket. Each lane sums eight adjacent
// bins, the warp locates the group containing rank, and one lane scans only
// that group's eight bins. The selected bin and preceding count are shared
// through ctl after the caller's named barrier.
CUTLASS_DEVICE void select_hist_bucket(int32_t* hist, int32_t* ctl,
                                       uint32_t tid) {
  if (tid >= 32) return;
  constexpr uint32_t FULL = 0xffffffffu;
  int group_count = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j)
    group_count += hist[255 - static_cast<int>(tid) * 8 - j];
  int inclusive = group_count;
#pragma unroll
  for (int shift = 1; shift < 32; shift <<= 1) {
    const int prev = __shfl_up_sync(FULL, inclusive, shift);
    if (tid >= static_cast<uint32_t>(shift)) inclusive += prev;
  }
  const int before = inclusive - group_count;
  const int rank = ctl[0];
  const uint32_t crossing =
      __ballot_sync(FULL, before < rank && inclusive >= rank);
  const int chosen_lane = __ffs(crossing) - 1;
  int chosen = 0;
  int above = before;
  if (static_cast<int>(tid) == chosen_lane) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int bin = 255 - static_cast<int>(tid) * 8 - j;
      if (above + hist[bin] >= rank) {
        chosen = bin;
        break;
      }
      above += hist[bin];
    }
  }
  chosen = __shfl_sync(FULL, chosen, chosen_lane);
  above = __shfl_sync(FULL, above, chosen_lane);
  if (tid == 0) {
    ctl[0] = rank - above;
    ctl[1] = (ctl[1] << 8) | chosen;
  }
}

// One CTA owns all KV tiles of a query block. At a flush boundary, its math
// warps compact one row's global candidate buffer in place through the now
// vacant shared ring. This keeps the live set bounded independently of ISL.
template <uint32_t MATH_THREADS>
CUTLASS_DEVICE void compact_candidates(uint16_t* values,
                                       int32_t* packed_indices,
                                       int32_t* count_ptr, uint32_t* gate_ptr,
                                       int32_t* hist, uint64_t* scratch,
                                       uint32_t tid, uint32_t cap) {
  constexpr int K = 2048;
  auto sync = [] { cutlass::arch::NamedBarrier(MATH_THREADS, 1).sync(); };
  int32_t* ctl = hist + 256;
  const int count = *count_ptr;
  if (tid == 0) {
    ctl[0] = K;
    ctl[1] = 0;
    ctl[2] = 0;
    ctl[3] = 0;
  }
  sync();
  for (int shift = 16; shift >= 8; shift -= 8) {
    if (tid < 256) hist[tid] = 0;
    sync();
    const uint32_t prefix = static_cast<uint32_t>(ctl[1]);
    for (int j = tid; j < count; j += MATH_THREADS) {
      const uint32_t code = fused_candidates::candidate_load_score_code(
          values[j], packed_indices[j]);
      if (shift == 16 || (code >> (shift + 8)) == prefix)
        atomicAdd(hist + ((code >> shift) & 255u), 1);
    }
    sync();
    select_hist_bucket(hist, ctl, tid);
    sync();
  }
  const uint32_t prefix16 = static_cast<uint32_t>(ctl[1]);
  const int coarse_kept = K - ctl[0] + hist[prefix16 & 255u];
  if (coarse_kept <= 3072) {
    const uint32_t coarse_floor = prefix16 << 8;
    for (int j = tid; j < count; j += MATH_THREADS) {
      const uint32_t code = fused_candidates::candidate_load_score_code(
          values[j], packed_indices[j]);
      if (code >= coarse_floor) {
        const int pos = atomicAdd(ctl + 2, 1);
        scratch[pos] =
            (static_cast<uint64_t>(static_cast<uint32_t>(packed_indices[j]))
             << 16) |
            static_cast<uint32_t>(values[j]);
      }
    }
    sync();
    const int kept = ctl[2];
    for (int j = tid; j < kept; j += MATH_THREADS) {
      const uint64_t record = scratch[j];
      values[j] = static_cast<uint16_t>(record);
      packed_indices[j] = static_cast<int32_t>(record >> 16);
    }
    sync();
    if (tid == 0) {
      *count_ptr = kept;
      *gate_ptr = coarse_floor == 0 ? 0 : coarse_floor - 1;
    }
    sync();
    return;
  }
  // Unusually large score ties in one high-16 bucket use the exact pass.
  if (tid < 256) hist[tid] = 0;
  sync();
  for (int j = tid; j < count; j += MATH_THREADS) {
    const uint32_t code = fused_candidates::candidate_load_score_code(
        values[j], packed_indices[j]);
    if ((code >> 8) == prefix16) atomicAdd(hist + (code & 255u), 1);
  }
  sync();
  select_hist_bucket(hist, ctl, tid);
  sync();
  const uint32_t kth = static_cast<uint32_t>(ctl[1]);
  for (int j = tid; j < count; j += MATH_THREADS) {
    const uint32_t code = fused_candidates::candidate_load_score_code(
        values[j], packed_indices[j]);
    if (code > kth) {
      const int pos = atomicAdd(ctl + 2, 1);
      if (pos < K)
        scratch[pos] =
            (static_cast<uint64_t>(static_cast<uint32_t>(packed_indices[j]))
             << 16) |
            static_cast<uint32_t>(values[j]);
    }
  }
  sync();
  const int greater = ctl[2];
  for (int j = tid; j < count; j += MATH_THREADS) {
    const uint32_t code = fused_candidates::candidate_load_score_code(
        values[j], packed_indices[j]);
    if (code == kth) {
      const int pos = atomicAdd(ctl + 3, 1);
      if (greater + pos < K)
        scratch[greater + pos] =
            (static_cast<uint64_t>(static_cast<uint32_t>(packed_indices[j]))
             << 16) |
            static_cast<uint32_t>(values[j]);
    }
  }
  sync();
  for (int j = tid; j < K; j += MATH_THREADS) {
    const uint64_t record = scratch[j];
    values[j] = static_cast<uint16_t>(record);
    packed_indices[j] = static_cast<int32_t>(record >> 16);
  }
  sync();
  if (tid == 0) {
    *count_ptr = K;
    *gate_ptr = kth;
  }
  sync();
}

CUTLASS_DEVICE float candidate_gate_float(const uint32_t code) {
  if (code == 0) return -cute::numeric_limits<float>::infinity();
  const uint32_t ordered = code << 8;
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
}

// Shared device core parameterized by dtype and scheduler geometry/addressing
template <
    uint32_t kNumHeads, uint32_t kHeadDim, bool kIsMXSF,
    bool kIsCompressedLogits, bool kCleanLogits, uint32_t BLOCK_Q,
    uint32_t SPLIT_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
    uint32_t kNumSMs, uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
    typename qk_dtype_t, typename logits_dtype_t, typename reduce_dtype_t,
    typename MakeScheduler, uint32_t kNumMathWarpGroups = kNumMathThreads / 128,
    uint32_t kHandoffInterval = 128, bool kSmemCand = false>
CUTLASS_DEVICE void sm100_mqa_logits_core_impl(
    const uint32_t num_q_tokens, const uint32_t logits_stride,
    logits_dtype_t* logits, const float* origin, const float* inv_delta,
    const int32_t* th_bucket, const int32_t* seed_hist, uint16_t* cand_val,
    int32_t* cand_idx, int32_t* cand_cnt, const uint32_t cand_cap,
    const int64_t cand_row_delta, const cute::TmaDescriptor& tensor_map_q,
    const cute::TmaDescriptor& tensor_map_sf_q,
    const cute::TmaDescriptor& tensor_map_kv,
    const cute::TmaDescriptor& tensor_map_sf_kv,
    const cute::TmaDescriptor& tensor_map_weights,
    const MakeScheduler& make_scheduler) {
  constexpr bool kIsFP4 = cute::is_same_v<qk_dtype_t, cutlass::float_e2m1_t>;

  const auto sm_idx = blockIdx.x;
  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;

  const auto kNegInf = -cute::numeric_limits<logits_dtype_t>::infinity();

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_sf_q);
    cute::prefetch_tma_descriptor(&tensor_map_weights);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
  }

  static constexpr uint32_t kNumTmemStages = 2;
  static_assert(BLOCK_Q == 8 && kNumMathThreads == 256 && !kIsMXSF);
  static constexpr uint32_t kNumUTCCPAlignedElems = 128;
  static constexpr uint32_t UMMA_M = 128;
  static constexpr uint32_t BLOCK_QH = BLOCK_Q * kNumHeads;
  static constexpr uint32_t UMMA_N = math::constexpr_align(BLOCK_QH, 8u);
  static constexpr uint32_t UMMA_K = kIsFP4 ? 64 : 32;
  static constexpr uint32_t kNumSFQ =
      kIsMXSF ? math::constexpr_align(UMMA_N, kNumUTCCPAlignedElems) : 0;
  static constexpr uint32_t kNumSFKV =
      kIsMXSF ? math::constexpr_align(SPLIT_KV, kNumUTCCPAlignedElems) : 0;
  static constexpr uint32_t kNumQKBytesPerToken =
      kIsFP4 ? (kHeadDim / 2) : kHeadDim;
  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE =
      BLOCK_QH * kNumQKBytesPerToken;
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE =
      SPLIT_KV * kNumQKBytesPerToken;
  static constexpr uint32_t SMEM_SF_Q_SIZE_PER_STAGE =
      kIsMXSF ? (BLOCK_QH * sizeof(int)) : 0;
  static constexpr uint32_t SMEM_SF_KV_SIZE_PER_STAGE =
      kIsMXSF ? (kNumSFKV * sizeof(int)) : (SPLIT_KV * sizeof(float));
  static constexpr uint32_t kNumWeightBytesPerRow = math::constexpr_align(
      kNumHeads * static_cast<uint32_t>(sizeof(reduce_dtype_t)), 16u);
  static constexpr uint32_t kNumWeightElementsPerRow =
      kNumWeightBytesPerRow / static_cast<uint32_t>(sizeof(reduce_dtype_t));
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE =
      BLOCK_Q * kNumWeightBytesPerRow;

  DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0,
                   "Invalid threads");
  DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and
                       SPLIT_KV % kNumUTCCPAlignedElems == 0,
                   "Invalid `SPLIT_KV`");
  DG_STATIC_ASSERT(not(kIsCompressedLogits and kCleanLogits),
                   "Compressed logits cannot be cleaned in-kernel");

  using SharedStorage = layout::MQALogitsSharedStorage<
      kNumHeads, kHeadDim, kIsMXSF, BLOCK_Q, SPLIT_KV, kNumQStages,
      kNumKVStages, kNumTmemStages, qk_dtype_t, reduce_dtype_t>;
  extern __shared__ __align__(SharedStorage::kSwizzleAlignment)
      uint8_t smem_buffer[];
  auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
  constexpr uint32_t kRingSlots = 8;
  constexpr uint32_t kRecordBlockMask = 255;
  auto& handoff = *reinterpret_cast<CandidateHandoff<>*>(smem_buffer +
                                                         sizeof(SharedStorage));
  handoff.init(threadIdx.x);

  constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumTmemStages;
  constexpr uint32_t kNumTmemCols =
      utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFQ / 32 +
                                       kNumSFKV / 32>();
  constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
  constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
  DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

  if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      smem.full_q_barriers[i].init(1);
      smem.empty_q_barriers[i].init(kNumMathThreads + 32);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      smem.full_kv_barriers[i].init(1);
      smem.empty_kv_barriers[i].init(kIsMXSF ? 1 : kNumMathThreads);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumTmemStages; ++i) {
      smem.full_tmem_barriers[i].init(1);
      smem.empty_tmem_barriers[i].init(256);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncwarp();

  if (warp_idx == kSpecWarpStart + 2)
    cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, &smem.tmem_ptr_in_smem);
  __syncthreads();

  RingPipeline<kNumQStages> q_pipeline;
  RingPipeline<kNumKVStages> kv_pipeline;
  RingPipeline<kNumTmemStages> tmem_pipeline;

  constexpr uint32_t kNumSpecializedRegisters = 56;
  constexpr uint32_t kNumMathRegisters = 224;

  cudaGridDependencySynchronize();

  // Short contexts (handoff interval <= 8, i.e. <= 16K keys): candidates are
  // dense and the drain was the bottleneck, so math warps emit directly (warp
  // ballot + one shared atomic per row and KV half). Longer contexts keep the
  // ring + drain warps. kHandoffInterval == 0: resident mode (keys <=
  // logits_stride). Every score of the block's 8 rows is stored as an fp24 code
  // in shared memory (hi16 array, then lo8 array, row-major with stride
  // logits_stride) at `logits`; no gate, no candidate emission.
  constexpr bool kResident = kHandoffInterval == 0;
  constexpr bool kDirectEmit = !kResident && kHandoffInterval <= 8;
  // Two drain warps: warp +3 drains math warpgroup 0 (rows 0-3) and warp +0,
  // idle after its single Q load, drains warpgroup 1 (rows 4-7). Rows are
  // disjoint.
  const auto drain_producers = [&](const unsigned p0) {
    auto scheduler = make_scheduler(sm_idx);
    uint32_t q_block_idx, kv_base, num_kv_splits;
    unsigned drain_epoch = 0;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      const unsigned row_base = scheduler.get_logits_row(q_block_idx, 0);
      for (unsigned base = 0; base < num_kv_splits;
           base += kHandoffInterval, ++drain_epoch) {
        for (unsigned p = p0; p < p0 + 4; ++p) {
          handoff.drain(
              p, drain_epoch, 4, row_base + (p / 4) * 4, cand_cnt, cand_cap,
              [&](unsigned row, unsigned slot, unsigned record,
                  unsigned batch_base, unsigned source) {
                const uint32_t physical =
                    kv_base + (batch_base * 2 + (record & 255u)) * 128 +
                    (source % 128);
                const uint64_t off =
                    static_cast<uint64_t>(int64_t(row) + cand_row_delta) *
                        cand_cap +
                    slot;
                fused_candidates::store_candidate_record(
                    cand_val + off, cand_idx + off, record >> 8, physical);
              });
        }
      }
    }
  };

  if (warp_idx == kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    if (cute::elect_one_sync()) {
      auto scheduler = make_scheduler(sm_idx);
      // NOTES: split index for paged scheduler, token offset for contiguous-KV
      // scheduler.
      uint32_t q_block_idx, kv_base, num_kv_splits;
      while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
        CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
        smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);

        const uint32_t q_token_base =
            scheduler.get_q_tma_token_base(q_block_idx);
        tma::copy<kHeadDim, BLOCK_Q * kNumHeads, 0>(
            &tensor_map_q, &smem.full_q_barriers[q_stage_idx],
            smem.smem_q[q_stage_idx], 0, q_token_base * kNumHeads);
        if constexpr (kIsMXSF)
          tma::copy<BLOCK_Q * kNumHeads, 1, 0>(
              &tensor_map_sf_q, &smem.full_q_barriers[q_stage_idx],
              smem.smem_sf_q[q_stage_idx], 0, q_token_base);
        tma::copy<kNumWeightElementsPerRow, BLOCK_Q, 0>(
            &tensor_map_weights, &smem.full_q_barriers[q_stage_idx],
            smem.smem_weights[q_stage_idx], 0, q_token_base);
        smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(
            SMEM_Q_SIZE_PER_STAGE + SMEM_SF_Q_SIZE_PER_STAGE +
            SMEM_WEIGHT_SIZE_PER_STAGE);
      }
    }
    __syncwarp();
    if constexpr (!kDirectEmit && !kResident) drain_producers(4u);
  } else if (warp_idx == kSpecWarpStart + 1) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();

    auto scheduler = make_scheduler(sm_idx);
    uint32_t cached_kv_page_base = 0;
    uint32_t cached_kv_page_coord = 0;
    // NOTES: split index for paged scheduler, token offset for contiguous-KV
    // scheduler.
    uint32_t q_block_idx, kv_base, num_kv_splits;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      cached_kv_page_base = cute::numeric_limits<uint32_t>::max();
#pragma unroll 1
      for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits;
           ++kv_split_idx) {
        if constexpr (decltype(scheduler)::kIsPaged) {
          constexpr uint32_t kPageKV = decltype(scheduler)::kPageKV;
          constexpr uint32_t kNumPagesPerSplit =
              decltype(scheduler)::kNumPagesPerSplit;
          DG_STATIC_ASSERT(kNumPagesPerSplit <= 32,
                           "Split spans more pages than a warp can cache");

          const uint32_t kv_page_base =
              (kv_base + kv_split_idx) * kNumPagesPerSplit;
          if (kv_page_base < cached_kv_page_base or
              kv_page_base + kNumPagesPerSplit > cached_kv_page_base + 32) {
            cached_kv_page_base = (kv_page_base / 32) * 32;
            cached_kv_page_coord = scheduler.get_kv_page_coord_by_page_offset(
                cached_kv_page_base + lane_idx);
          }

          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          if (cute::elect_one_sync())
            smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
          __syncwarp();

          int page_coords[kNumPagesPerSplit];
#pragma unroll
          for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit;
               ++page_idx) {
            const auto src_lane =
                static_cast<int>(kv_page_base - cached_kv_page_base + page_idx);
            page_coords[page_idx] =
                __shfl_sync(0xffffffff, cached_kv_page_coord, src_lane);
          }

          if (cute::elect_one_sync()) {
#pragma unroll
            for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit;
                 ++page_idx) {
              tma::copy<kHeadDim, kPageKV, 0, qk_dtype_t, true>(
                  &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                  smem.smem_kv[kv_stage_idx] +
                      page_idx * kPageKV * kNumQKBytesPerToken,
                  0, 0, 1, page_coords[page_idx]);
              tma::copy<kPageKV, 1, 0>(
                  &tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                  smem.smem_sf_kv[kv_stage_idx] + page_idx * kPageKV, 0,
                  page_coords[page_idx]);
            }
            smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(
                SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
          }
          __syncwarp();
        } else if (cute::elect_one_sync()) {
          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);

          const uint32_t kv_tma_offset =
              scheduler.get_kv_tma_offset(kv_base, kv_split_idx);
          tma::copy<kHeadDim, SPLIT_KV, 0>(
              &tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
              smem.smem_kv[kv_stage_idx], 0, kv_tma_offset);
          tma::copy<SPLIT_KV, 1, 0>(
              &tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
              smem.smem_sf_kv[kv_stage_idx], kv_tma_offset, 0);
          smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(
              SMEM_KV_SIZE_PER_STAGE + SMEM_SF_KV_SIZE_PER_STAGE);
        }
        __syncwarp();
      }
    }
  } else if (warp_idx == kSpecWarpStart + 2) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr_in_smem) == 0);

    auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
      DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128,
                       "Invalid aligned elements");
      uint32_t values[4];
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
      __syncwarp();
      ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2],
                     values[3]);
    };

    auto sf_desc = mma::sm100::make_sf_desc(nullptr);

    auto scheduler = make_scheduler(sm_idx);
    // NOTES: split index for paged scheduler, token offset for contiguous-KV
    // scheduler.
    uint32_t q_block_idx, kv_base, num_kv_splits;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
      smem.full_q_barriers[q_stage_idx].wait(q_phase);

      if constexpr (kIsMXSF) {
#pragma unroll
        for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++i) {
          auto smem_ptr =
              smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
          utccp_required_smem_warp_transpose(smem_ptr);
        }
        cutlass::arch::fence_view_async_shared();
#pragma unroll
        for (uint32_t i = 0; i < kNumSFQ / kNumUTCCPAlignedElems; ++i) {
          auto smem_ptr =
              smem.smem_sf_q[q_stage_idx] + i * kNumUTCCPAlignedElems;
          mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
          if (cute::elect_one_sync())
            cute::SM100_UTCCP_4x32dp128bit_1cta::copy(
                sf_desc, kTmemStartColOfSFQ + i * 4);
          __syncwarp();
        }
      }

      for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits;
           ++kv_split_idx) {
        CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
        smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);

        if constexpr (kIsMXSF) {
#pragma unroll
          for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++i) {
            auto smem_ptr =
                smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
            utccp_required_smem_warp_transpose(smem_ptr);
          }
          cutlass::arch::fence_view_async_shared();
        }

        if (cute::elect_one_sync()) {
          if constexpr (kIsMXSF) {
#pragma unroll
            for (uint32_t i = 0; i < kNumSFKV / kNumUTCCPAlignedElems; ++i) {
              auto smem_ptr =
                  smem.smem_sf_kv[kv_stage_idx] + i * kNumUTCCPAlignedElems;
              mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
              cute::SM100_UTCCP_4x32dp128bit_1cta::copy(
                  sf_desc, kTmemStartColOfSFKV + i * 4);
            }
          }
#pragma unroll
          for (uint32_t i = 0; i < kNumMathWarpGroups; ++i) {
            CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
            uint32_t tmem_addr = tmem_stage_idx * UMMA_N;

            smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
            ptx::tcgen05_after_thread_sync();

            if constexpr (kIsMXSF) {
              DG_STATIC_ASSERT((not kIsFP4 and kHeadDim == 32) or
                                   kHeadDim == 64 or kHeadDim == 128,
                               "Invalid head dim");

              constexpr uint32_t kPackFactor =
                  get_smem_pack_factor<qk_dtype_t>();
              constexpr uint32_t kQKSwizzleMode = kHeadDim / kPackFactor;

              using mma_op_t =
                  cute::conditional_t<kIsFP4, ptx::SM100_MMA_MXF4_SS,
                                      ptx::SM100_MMA_MXF8F6F4_SS>;
              auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
                  qk_dtype_t, qk_dtype_t, float, cutlass::float_ue8m0_t, UMMA_M,
                  UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
#pragma unroll
              for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
                auto runtime_instr_desc =
                    mma::sm100::make_runtime_instr_desc_with_sf_id(
                        instr_desc, k * kPackFactor, k * kPackFactor);
                auto a_desc =
                    mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                               kHeadDim, kQKSwizzleMode>(
                        smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                auto b_desc =
                    mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                               kHeadDim, kQKSwizzleMode>(
                        smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                mma_op_t::fma(a_desc, b_desc, tmem_addr, k, runtime_instr_desc,
                              kTmemStartColOfSFKV + i * 4, kTmemStartColOfSFQ);
              }
            } else {
              auto instr_desc = cute::UMMA::make_instr_desc<
                  cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, UMMA_M,
                  UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
              auto runtime_instr_desc =
                  cute::UMMA::make_runtime_instr_desc(instr_desc);
#pragma unroll
              for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++k) {
                auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K,
                                                         0, kHeadDim, kHeadDim>(
                    smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K,
                                                         0, kHeadDim, kHeadDim>(
                    smem.smem_q[q_stage_idx], 0, k * UMMA_K);
                ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, k,
                                              runtime_instr_desc);
              }
            }

            ptx::umma_arrive_no_elect(smem.full_tmem_barriers[tmem_stage_idx]);
          }
        }
        __syncwarp();
        if constexpr (kIsMXSF)
          cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(
              &smem.empty_kv_barriers[kv_stage_idx]));
      }
      smem.empty_q_barriers[q_stage_idx].arrive();
    }
  } else if (warp_idx == kSpecWarpStart + 3) {
    cutlass::arch::warpgroup_reg_dealloc<kNumSpecializedRegisters>();
    if constexpr (!kDirectEmit && !kResident) drain_producers(0u);
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

    constexpr bool kLoadSeqBounds = true;
    uint32_t seq_k_start[4];
    uint32_t seq_k_end[4];
    const auto math_warpgroup_idx = warp_idx / 4;
    const auto math_thread_idx = warp_idx * 32 + lane_idx;
    DG_STATIC_ASSERT(kNumMathWarpGroups <= kNumTmemStages,
                     "Math warp groups exceed TMEM stages");
    const unsigned query_base = math_warpgroup_idx * 4;

    constexpr bool kIsReduceBF16 = not cute::is_same_v<reduce_dtype_t, float>;
    DG_STATIC_ASSERT(kNumHeads % 4 == 0, "Head count must be a multiple of 4");
    DG_STATIC_ASSERT(8 <= UMMA_N and UMMA_N <= 256, "Invalid UMMA_N for MMA");
    using weights_reg_dtype_t =
        cute::conditional_t<kIsReduceBF16, nv_bfloat162, float>;
    constexpr uint32_t kNumWeightsRegPerToken =
        kIsReduceBF16 ? (kNumHeads / 2) : kNumHeads;
    weights_reg_dtype_t weights[4][kNumWeightsRegPerToken];
    float accum[kNumHeads];

    unsigned produce_epoch = 0;
    auto scheduler = make_scheduler(sm_idx);
    // NOTES: split index for paged scheduler, token offset for contiguous-KV
    // scheduler.
    uint32_t q_block_idx, kv_base, num_kv_splits;
    while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
      CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
      smem.full_q_barriers[q_stage_idx].wait(q_phase);

      const auto process_q_block = [&](auto num_valid_tokens_t) {
        constexpr uint32_t kNumValidTokens = 4;
        uint32_t lane_counts[4] = {};
        uint32_t flush_base = 0;
        float gate_score[4];

#pragma unroll
        for (uint32_t i = 0; i < kNumValidTokens; ++i) {
          const uint32_t row =
              scheduler.get_logits_row(q_block_idx, query_base + i);
          // Use the raw-score threshold estimated from the distributed sample.
          // This threshold remains fixed for the complete scan.
          gate_score[i] = kResident ? 0.0f : origin[row];
          seq_k_start[i] = cute::min(scheduler.cu_seq_len_k_start[row],
                                     scheduler.num_kv_tokens);
          seq_k_end[i] = cute::min(scheduler.cu_seq_len_k_end[row],
                                   scheduler.num_kv_tokens);
          const auto smem_weights_row =
              smem.smem_weights[q_stage_idx] +
              (query_base + i) * kNumWeightElementsPerRow;
          if constexpr (kIsReduceBF16) {
            // Load two bf16 weights at a time as one packed shared u32
            const auto packed_row =
                reinterpret_cast<const uint32_t*>(smem_weights_row);
#pragma unroll
            for (uint32_t j = 0; j < kNumHeads / 2; ++j) {
              const auto packed = ptx::ld_shared(packed_row + j);
              weights[i][j] = ptx::exchange(
                  *reinterpret_cast<const nv_bfloat162*>(&packed), 0);
            }
          } else {
            bool unsafe_weights = false;
#pragma unroll
            for (uint32_t j = 0; j < kNumHeads; ++j) {
              const float weight = ptx::ld_shared(smem_weights_row + j);
              unsafe_weights |= weight != 0.0f && fabsf(weight) < 0x1p-125f;
              // Resident: full weights, the 0.5 ReLU factor is applied per
              // score.
              weights[i][j] = kResident ? weight : weight * 0.5f;
            }
            // Hoist the ReLU factor out of the per-key reduction.
            // Tiny weights can round differently when halved; use
            // the existing GPU recovery with unscaled weights.
            if (!kResident && unsafe_weights) {
              gate_score[i] = __int_as_float(0x7f800000);
              if (math_thread_idx % 128 == 0)
                atomicAdd(cand_cnt + row, static_cast<int>(cand_cap) + 1);
            }
          }
        }

        __shared__ unsigned row_count[BLOCK_Q];
        // Shared candidates, per row [seg0][seg1][seg2][seg3][pool]: each of
        // the row's 4 warps owns a static segment (no atomics; its last slot is
        // a trash slot for lanes without a candidate); a warp past its segment
        // takes slots from the row's shared pool (one atomic per overflowing
        // ballot); past the pool, global slots row*cand_cap + j (count at
        // cand_cap - 1). Unfilled slots get the lowest code.
        const uint32_t pool_size = logits_stride / 6,
                       seg_size = (logits_stride - pool_size) / 4;
        const uint32_t seg_warp = warp_idx % 4;
        // Each array holds the 8 rows plus 256 per-lane trash slots (8 warps x
        // 32 lanes).
        const uint32_t cand_ext = BLOCK_Q * logits_stride + 256;
        __shared__ unsigned pool_cnt[BLOCK_Q];
        uint32_t seg_cnt[kNumValidTokens] = {};
        if constexpr (kDirectEmit) {
          if (math_thread_idx < BLOCK_Q) {
            row_count[math_thread_idx] = 0;
            pool_cnt[math_thread_idx] = 0;
          }
          cutlass::arch::NamedBarrier(kNumMathThreads, 3).sync();
        }
        for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits;
             ++kv_split_idx) {
          if constexpr (!kDirectEmit && !kResident)
            if ((kv_split_idx & (kHandoffInterval - 1)) == 0)
              handoff.acquire(warp_idx, produce_epoch);
          CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
          smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
#pragma unroll
          for (unsigned kv_half = 0; kv_half < 2; ++kv_half) {
            const unsigned kv_lane = (math_thread_idx % 128) + kv_half * 128;
            const unsigned kv_offset =
                scheduler.get_logits_col(kv_base, kv_split_idx, kv_lane);
            const float scale_kv =
                ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + kv_lane);
            CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
            smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
            ptx::tcgen05_after_thread_sync();
            // Both KV half-MMAs have committed before the shared stage is
            // freed.
            if (kv_half == 1) {
              cutlass::arch::fence_view_async_shared();
              smem.empty_kv_barriers[kv_stage_idx].arrive();
            }

            static_assert(!kIsReduceBF16, "FP32-only pair reduction");
            float accum_pair[2][kNumHeads];
#pragma unroll
            for (uint32_t pair = 0; pair < kNumValidTokens; pair += 2) {
              // Load only 2 query rows at once to limit
              // live accumulator registers during score reduction.
              const uint32_t tmem_addr =
                  tmem_stage_idx * UMMA_N + (query_base + pair) * kNumHeads;
              ptx::tmem_load_32dp32b<2 * kNumHeads>(
                  tmem_addr, reinterpret_cast<uint32_t*>(accum_pair));
              cutlass::arch::fence_view_async_tmem_load();
              if (pair + 2 >= kNumValidTokens) {
                ptx::tcgen05_before_thread_sync();
                smem.empty_tmem_barriers[tmem_stage_idx].arrive();
              }

              float2 sum_0[2] = {};
              float2 sum_1[2] = {};
#pragma unroll
              for (uint32_t j = 0; j < kNumHeads; j += 4) {
#pragma unroll
                for (uint32_t r = 0; r < 2; ++r) {
                  if (pair + r >= kNumValidTokens) continue;
                  const uint32_t i = pair + r;
                  auto* a = accum_pair[r];
                  auto transform = [&](uint32_t at, float2 sum) {
                    const auto x = make_float2(a[at], a[at + 1]);
                    const auto abs_x =
                        make_float2(fabsf(a[at]), fabsf(a[at + 1]));
                    const auto w =
                        make_float2(weights[i][at], weights[i][at + 1]);
                    return __ffma2_rn(__fadd2_rn(x, abs_x), w, sum);
                  };
                  sum_0[r] = transform(j, sum_0[r]);
                  sum_1[r] = transform(j + 2, sum_1[r]);
                }
              }
#pragma unroll
              for (uint32_t r = 0; r < 2; ++r) {
                if (pair + r >= kNumValidTokens) continue;
                const uint32_t i = pair + r;
                const auto sum = __fadd2_rn(sum_0[r], sum_1[r]);
                const float reduced = sum.x + sum.y;
                const float bq =
                    reduced * (kResident ? 0.5f * static_cast<float>(scale_kv)
                                         : static_cast<float>(scale_kv));
                const uint32_t row =
                    scheduler.get_logits_row(q_block_idx, query_base + i);
                if constexpr (kResident) {
                  // Select reads only [start, end) of each row, so no masking
                  // here.
                  if (kv_offset < logits_stride) {
                    const uint32_t code =
                        fused_candidates::candidate_fp24_code(bq);
                    const uint32_t at =
                        (query_base + i) * logits_stride + kv_offset;
                    reinterpret_cast<uint16_t*>(logits)[at] =
                        static_cast<uint16_t>(code >> 8);
                    reinterpret_cast<uint8_t*>(
                        logits)[BLOCK_Q * logits_stride * 2 + at] =
                        static_cast<uint8_t>(code);
                  }
                  continue;
                }
                // NaN fails the compare; +inf passes (harmless, it would win
                // the top-k anyway).
                const bool pass = (kSmemCand ? kv_offset - seq_k_start[i] <
                                                   seq_k_end[i] - seq_k_start[i]
                                             : kv_offset >= seq_k_start[i] &&
                                                   kv_offset < seq_k_end[i] &&
                                                   isfinite(bq)) &&
                                  bq > gate_score[i];
                if constexpr (kSmemCand) {
                  // Branch-free: every lane stores; lanes without a candidate
                  // (or past the segment) write their own trash slot after the
                  // rows (never read).
                  const unsigned pass_mask = __ballot_sync(0xffffffffu, pass);
                  const unsigned slot =
                      seg_cnt[i] + __popc(pass_mask & ((1u << lane_idx) - 1u));
                  seg_cnt[i] += __popc(pass_mask);
                  const uint32_t code =
                      fused_candidates::candidate_fp24_code(bq);
                  const uint32_t seg_cap = seg_size;
                  const uint32_t at =
                      pass && slot < seg_cap
                          ? (query_base + i) * logits_stride +
                                seg_warp * seg_size + slot
                          : BLOCK_Q * logits_stride + warp_idx * 32 + lane_idx;
                  auto* cbase = reinterpret_cast<uint8_t*>(logits);
                  reinterpret_cast<uint16_t*>(cbase)[at] =
                      static_cast<uint16_t>(code >> 8);
                  cbase[cand_ext * 2 + at] = static_cast<uint8_t>(code);
                  reinterpret_cast<uint16_t*>(cbase + cand_ext * 3)[at] =
                      static_cast<uint16_t>(kv_offset);
                  if (seg_cnt[i] > seg_cap) {
                    // Warp-uniform: lanes past the segment take row-pool slots.
                    const bool over = pass && slot >= seg_cap;
                    const unsigned over_mask = __ballot_sync(0xffffffffu, over);
                    if (over_mask) {
                      unsigned pool_base = 0;
                      if (lane_idx == 0)
                        pool_base = atomicAdd(&pool_cnt[query_base + i],
                                              __popc(over_mask));
                      pool_base = __shfl_sync(0xffffffffu, pool_base, 0);
                      const unsigned p =
                          pool_base +
                          __popc(over_mask & ((1u << lane_idx) - 1u));
                      if (over && p < pool_size) {
                        const uint32_t pat =
                            (query_base + i) * logits_stride + 4 * seg_size + p;
                        reinterpret_cast<uint16_t*>(cbase)[pat] =
                            static_cast<uint16_t>(code >> 8);
                        cbase[cand_ext * 2 + pat] = static_cast<uint8_t>(code);
                        reinterpret_cast<uint16_t*>(cbase + cand_ext * 3)[pat] =
                            static_cast<uint16_t>(kv_offset);
                      } else if (over && p - pool_size < cand_cap - 1) {
                        // Past the pool (rare): global, val = hi16 code, idx =
                        // lo8 code << 16 | key.
                        const uint64_t off =
                            static_cast<uint64_t>(int64_t(row) +
                                                  cand_row_delta) *
                                cand_cap +
                            (p - pool_size);
                        __stcg(cand_val + off,
                               static_cast<uint16_t>(code >> 8));
                        __stcg(cand_idx + off,
                               static_cast<int32_t>((code & 0xffu) << 16 |
                                                    kv_offset));
                      }
                    }
                  }
                } else if constexpr (kDirectEmit) {
                  const unsigned pass_mask = __ballot_sync(0xffffffffu, pass);
                  if (pass_mask) {
                    // Ballot + one shared atomic per row: global candidate
                    // slots.
                    const unsigned leader = __ffs(pass_mask) - 1;
                    unsigned base = 0;
                    if (lane_idx == leader)
                      base = atomicAdd(&row_count[query_base + i],
                                       __popc(pass_mask));
                    base = __shfl_sync(0xffffffffu, base, leader);
                    const unsigned slot =
                        base + __popc(pass_mask & ((1u << lane_idx) - 1u));
                    if (pass && slot < cand_cap) {
                      const uint64_t off =
                          static_cast<uint64_t>(int64_t(row) + cand_row_delta) *
                              cand_cap +
                          slot;
                      fused_candidates::store_candidate(
                          cand_val + off, cand_idx + off, bq, kv_offset);
                    }
                  }
                } else if (pass) {
                  const uint32_t count = lane_counts[i];
                  if (count < kRingSlots) {
                    const uint32_t pos =
                        ((warp_idx * BLOCK_Q + i) * kRingSlots + count) * 32u +
                        lane_idx;
                    const uint32_t code =
                        fused_candidates::candidate_fp24_code(bq);
                    handoff.records[produce_epoch & 1][warp_idx][i][count]
                                   [lane_idx] =
                        (code << 8) |
                        ((kv_split_idx - flush_base) * 2 + kv_half);
                    lane_counts[i] = count + 1;
                  } else {
                    const int slot = atomicAdd(cand_cnt + row, 1);
                    if (slot < static_cast<int>(cand_cap)) {
                      const uint64_t off =
                          static_cast<uint64_t>(int64_t(row) + cand_row_delta) *
                              cand_cap +
                          slot;
                      fused_candidates::store_candidate(
                          cand_val + off, cand_idx + off, bq, kv_offset);
                    }
                  }
                }
              }
            }
          }  // KV halves; query ownership is independent of the KV half.
          if (!kDirectEmit && !kResident &&
              (((kv_split_idx + 1) & (kHandoffInterval - 1)) == 0 ||
               kv_split_idx + 1 == num_kv_splits)) {
#pragma unroll
            for (unsigned i = 0; i < kNumValidTokens; ++i) {
              handoff.counts[produce_epoch & 1][warp_idx][i][lane_idx] =
                  lane_counts[i];
              lane_counts[i] = 0;
            }
            handoff.publish(warp_idx, produce_epoch, flush_base);
            ++produce_epoch;
            flush_base = kv_split_idx + 1;
          }
        }
        if constexpr (kSmemCand) {
// Seal each warp's segment tail with the lowest code; publish counts and
// spills.
#pragma unroll
          for (uint32_t i = 0; i < kNumValidTokens; ++i) {
            const uint32_t row =
                scheduler.get_logits_row(q_block_idx, query_base + i);
            auto* base = reinterpret_cast<uint8_t*>(logits);
            for (uint32_t slot = min(seg_cnt[i], seg_size) + lane_idx;
                 slot < seg_size; slot += 32) {
              const uint32_t at =
                  (query_base + i) * logits_stride + seg_warp * seg_size + slot;
              reinterpret_cast<uint16_t*>(base)[at] = 0;
              base[cand_ext * 2 + at] = 0;
              reinterpret_cast<uint16_t*>(base + cand_ext * 3)[at] = 0;
            }
            if (lane_idx == 0)
              atomicAdd(&row_count[query_base + i], seg_cnt[i]);
          }
          // After every warp's pool claims: seal each row's unused pool slots
          // and record the global spill count (bit 21 of the published count
          // marks a spilled row).
          cutlass::arch::NamedBarrier(kNumMathThreads, 3).sync();
          for (uint32_t q = 0; q < BLOCK_Q; ++q) {
            auto* base = reinterpret_cast<uint8_t*>(logits);
            for (uint32_t p = pool_cnt[q] + math_thread_idx; p < pool_size;
                 p += kNumMathThreads) {
              const uint32_t at = q * logits_stride + 4 * seg_size + p;
              reinterpret_cast<uint16_t*>(base)[at] = 0;
              base[cand_ext * 2 + at] = 0;
              reinterpret_cast<uint16_t*>(base + cand_ext * 3)[at] = 0;
            }
          }
          if (math_thread_idx < BLOCK_Q) {
            const uint32_t row =
                scheduler.get_logits_row(q_block_idx, math_thread_idx);
            const uint32_t spilled =
                pool_cnt[math_thread_idx] > pool_size
                    ? min(pool_cnt[math_thread_idx] - pool_size, cand_cap - 1)
                    : 0u;
            __stcg(cand_idx +
                       static_cast<uint64_t>(int64_t(row) + cand_row_delta) *
                           cand_cap +
                       cand_cap - 1,
                   static_cast<int32_t>(spilled));
            if (spilled) atomicOr(&row_count[math_thread_idx], 1u << 21);
          }
        }
        if constexpr (kDirectEmit) {
          // Publish row totals (added to any unsafe-weights recovery marker
          // already in cand_cnt). Shared-candidate rows with < 2048 real
          // candidates are pushed past 16384 (recover).
          cutlass::arch::NamedBarrier(kNumMathThreads, 3).sync();
          if (math_thread_idx < BLOCK_Q) {
            // Shared candidates: bits 0-20 = real count, bit 21 = some warp
            // spilled.
            int total = static_cast<int>(row_count[math_thread_idx]);
            atomicAdd(cand_cnt + scheduler.get_logits_row(q_block_idx,
                                                          math_thread_idx),
                      total);
          }
        }
      };

      if constexpr (decltype(scheduler)::kHasPartialBlock)
        dispatch_num_block_tokens<BLOCK_Q>(
            scheduler.get_num_block_tokens(q_block_idx), process_q_block);
      else
        process_q_block(cute::Int<BLOCK_Q>{});

      cutlass::arch::fence_view_async_shared();
      smem.empty_q_barriers[q_stage_idx].arrive();
    }

    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, kNumTmemCols);
  }
}

// Unified contiguous-KV entry for FP8 / MXFP4 / MXFP8.
template <uint32_t kNumHeads, uint32_t kHeadDim, bool kIsMXSF,
          bool kIsCompressedLogits, bool kCleanLogits, bool kUseSchedule,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumQStages,
          uint32_t kNumKVStages, uint32_t kNumSMs,
          uint32_t kNumSpecializedThreads, uint32_t kNumMathThreads,
          typename qk_dtype_t, typename logits_dtype_t,
          typename reduce_dtype_t = float,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128,
          uint32_t kHandoffInterval = 128>
CUTLASS_GLOBAL __launch_bounds__(
    kNumSpecializedThreads + kNumMathThreads,
    1) void sm100_mqa_logits(const uint32_t num_q_tokens,
                             const uint32_t num_kv_tokens,
                             const uint32_t logits_stride,
                             const uint32_t* cu_seq_len_k_start,
                             const uint32_t* cu_seq_len_k_end,
                             const uint32_t* schedule_meta,
                             logits_dtype_t* logits, const float* origin,
                             const float* inv_delta, const int32_t* th_bucket,
                             const int32_t* seed_hist, uint16_t* cand_val,
                             int32_t* cand_idx, int32_t* cand_cnt,
                             const uint32_t cand_cap,
                             const __grid_constant__ cute::TmaDescriptor
                                 tensor_map_q,
                             const __grid_constant__ cute::TmaDescriptor
                                 tensor_map_sf_q,
                             const __grid_constant__ cute::TmaDescriptor
                                 tensor_map_kv,
                             const __grid_constant__ cute::TmaDescriptor
                                 tensor_map_sf_kv,
                             const __grid_constant__ cute::TmaDescriptor
                                 tensor_map_weights) {
  const auto make_scheduler = [&](const uint32_t& sm_idx) {
    return sched::SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs,
                                          kUseSchedule>(
        sm_idx, num_q_tokens, num_kv_tokens, cu_seq_len_k_start,
        cu_seq_len_k_end, schedule_meta);
  };

  sm100_mqa_logits_core_impl<
      kNumHeads, kHeadDim, kIsMXSF, kIsCompressedLogits, kCleanLogits, BLOCK_Q,
      SPLIT_KV, kNumQStages, kNumKVStages, kNumSMs, kNumSpecializedThreads,
      kNumMathThreads, qk_dtype_t, logits_dtype_t, reduce_dtype_t,
      decltype(make_scheduler), kNumMathWarpGroups, kHandoffInterval>(
      num_q_tokens, logits_stride, logits, origin, inv_delta, th_bucket,
      seed_hist, cand_val, cand_idx, cand_cnt, cand_cap, 0, tensor_map_q,
      tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights,
      make_scheduler);
}

}  // namespace fused_mqa_scan
