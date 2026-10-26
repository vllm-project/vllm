// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Exact recovery for the blocks the persistent kernel cannot select from its
// gated candidates (see persistent_kernel.cuh): the block is rescanned densely,
// and every 1024-key segment of fp32 scores is merged into a running per-row
// top set (kept <= kRetainedPerRow, compacted to K).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <math_constants.h>
#include <cstdint>

#include "common.cuh"
#include "constants.h"
#include "dense_scan.cuh"

namespace fused_mqa_topk::rescan {
using dense::Scheduler;

// Keys whose scores are merged into the retained sets at a time.
constexpr uint32_t kSegmentKeys = 1024;
// Per-row retained top set of the exact rescan (scores and key indices).
struct RescanBuffers {
  float* retained;
  int* retained_indices;
};
// Per-row segment histograms: half_bin<6> of a score (1024 bins); the radix
// levels of compact use 256 bins.
constexpr uint32_t kSegmentBins = 1024, kRadixBins = 256;
struct alignas(16) RescanShared {
  float ring[2][8][kSegmentKeys];
  uint32_t hist[8][padded_bins(kSegmentBins)], gates[8], kept[8];
  uint32_t work[padded_bins(kRadixBins)], prefix, rank, emit_counts[8];
  __device__ void init(uint32_t tid) {
    for (uint32_t j = tid; j < sizeof(hist) / sizeof(uint32_t);
         j += kNumThreads)
      reinterpret_cast<uint32_t*>(hist)[j] = 0;
    if (tid < 8) {
      gates[tid] = __float_as_uint(-CUDART_INF_F);
      kept[tid] = 0;
    }
  }
};
__device__ inline void compact_retained(RescanShared& shared,
                                        RescanBuffers buffers, uint32_t K,
                                        uint32_t local, uint32_t row,
                                        uint32_t tid, bool final = false) {
  constexpr uint32_t Full = 0xffffffffu, kThreads = kMathThreads,
                     Items = kRetainedPerRow / kThreads;
  uint32_t lane = tid & 31, count = shared.kept[local];
  uint64_t base = (uint64_t)row * kRetainedPerRow;
  float values[Items];
  int indices[Items];
#pragma unroll
  for (uint32_t i = 0; i < Items; ++i) {
    uint32_t j = tid + i * kThreads;
    values[i] = j < count ? buffers.retained[base + j] : -CUDART_INF_F;
    indices[i] = j < count ? buffers.retained_indices[base + j] : -1;
  }
  // mode 0: values >= gate; 1: codes above threshold; 2: codes equal to it.
  auto emit = [&](float gate, uint32_t threshold, int mode,
                  uint32_t base_offset) {
    const auto passes = [&](float v) {
      return isfinite(v) &&
             (mode == 0 ? v >= gate
                        : (mode == 1 ? ordered_code(v) > threshold
                                     : ordered_code(v) == threshold));
    };
    uint32_t total = 0;
#pragma unroll
    for (uint32_t i = 0; i < Items; ++i) total += passes(values[i]);
    uint32_t warp_total = __reduce_add_sync(Full, total);
    if (lane == 0) shared.emit_counts[tid / 32] = warp_total;
    math_wg_sync();
    uint32_t start = base_offset;
    for (uint32_t w = 0; w < tid / 32; ++w) start += shared.emit_counts[w];
    uint32_t all = 0;
    for (uint32_t w = 0; w < 8; ++w) all += shared.emit_counts[w];
#pragma unroll
    for (uint32_t i = 0; i < Items; ++i) {
      const float v = values[i];
      const bool pass = passes(v);
      uint32_t mask = __ballot_sync(Full, pass),
               pos = start + __popc(mask & ((1u << lane) - 1));
      if (pass && (mode != 2 || pos < K)) {
        buffers.retained[base + pos] = v;
        buffers.retained_indices[base + pos] = indices[i];
      }
      start += __popc(mask);
    }
    math_wg_sync();
    return all;
  };
  if (!final) {
    if (tid == 0) {
      shared.rank = 0;
    }
    math_wg_sync();
    float gate = __uint_as_float(shared.gates[local]);
    uint32_t kept = 0;
#pragma unroll
    for (uint32_t i = 0; i < Items; ++i)
      kept += isfinite(values[i]) && values[i] >= gate;
    kept = __reduce_add_sync(Full, kept);
    if (lane == 0) atomicAdd(&shared.rank, kept);
    math_wg_sync();
    kept = shared.rank;
    if (kept >= K && kept <= kRetainedPerRow - kSegmentKeys) {
      emit(gate, 0, 0, 0);
      math_wg_sync();
      if (tid == 0) shared.kept[local] = kept;
      math_wg_sync();
      return;
    }
  }
  if (tid == 0) {
    shared.prefix = 0;
    shared.rank = K;
  }
  math_wg_sync();
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint32_t j = tid; j < padded_bins(kRadixBins); j += kThreads)
      shared.work[j] = 0;
    math_wg_sync();
    uint32_t prefix = shared.prefix;
#pragma unroll
    for (uint32_t i = 0; i < Items; ++i) {
      uint32_t code = ordered_code(values[i]);
      bool valid = isfinite(values[i]) &&
                   (shift == 24 || (code >> (shift + 8)) == prefix);
      uint32_t mask = __ballot_sync(Full, valid);
      if (valid) {
        uint32_t bin = (code >> shift) & 255u,
                 peers = __match_any_sync(mask, bin);
        if (lane == uint32_t(__ffs(peers) - 1))
          atomicAdd(shared.work + padded_bin(bin), __popc(peers));
      }
    }
    math_wg_sync();
    if (tid < 32) {
      uint32_t sum = 0;
      for (uint32_t j = 0; j < 8; ++j) {
        uint32_t bin = 255 - lane * 8 - j;
        sum += shared.work[padded_bin(bin)];
      }
      const uint32_t inclusive = warp_inclusive_scan(sum, lane);
      uint32_t before = inclusive - sum, rank = shared.rank,
               crossing =
                   __ballot_sync(Full, before < rank && inclusive >= rank),
               chosen_lane = __ffs(crossing) - 1, chosen = 0, above = before;
      if (lane == chosen_lane) {
        for (uint32_t j = 0; j < 8; ++j) {
          uint32_t bin = 255 - lane * 8 - j;
          if (above + shared.work[padded_bin(bin)] >= rank) {
            chosen = bin;
            break;
          }
          above += shared.work[padded_bin(bin)];
        }
      }
      chosen = __shfl_sync(Full, chosen, chosen_lane);
      above = __shfl_sync(Full, above, chosen_lane);
      __syncwarp();  // every lane read shared.rank above
      if (lane == 0) {
        shared.prefix = (prefix << 8) | chosen;
        shared.rank = rank - above;
      }
    }
    math_wg_sync();
  }
  uint32_t threshold = shared.prefix;
  uint32_t greater = emit(0, threshold, 1, 0);
  emit(0, threshold, 2, greater);
  if (tid == 0) {
    shared.kept[local] = K;
    float gate =
        fmaxf(__uint_as_float(shared.gates[local]), ordered_float(threshold));
    atomicExch(shared.gates + local, __float_as_uint(gate));
  }
  math_wg_sync();
}
__device__ inline void merge_segment(RescanShared& shared,
                                     RescanBuffers buffers, uint32_t K,
                                     uint32_t local, uint32_t row,
                                     uint32_t segment, uint32_t first,
                                     uint32_t n, uint32_t tid) {
  constexpr uint32_t Full = 0xffffffffu, Items = kSegmentKeys / 32;
  uint32_t lane = tid, bank = segment & 1;
  float values[Items], gate = __uint_as_float(shared.gates[local]);
  uint32_t count = 0;
#pragma unroll
  for (uint32_t i = 0; i < Items; ++i) {
    uint32_t j = tid + i * 32;
    float v = j < n ? shared.ring[bank][local][j] : -CUDART_INF_F;
    if (v <= gate) v = -CUDART_INF_F;
    values[i] = v;
    bool valid = isfinite(v);
    count += valid;
    uint32_t mask = __ballot_sync(Full, valid);
    if (valid) {
      uint32_t bin = half_bin<6>(v), peers = __match_any_sync(mask, bin);
      if (lane == uint32_t(__ffs(peers) - 1))
        atomicAdd(&shared.hist[local][padded_bin(bin)], __popc(peers));
    }
  }
  count = __reduce_add_sync(Full, count);
  __syncwarp();
  if (count) {
    uint32_t sum = 0;
#pragma unroll
    for (uint32_t j = 0; j < 32; ++j)
      sum += shared.hist[local][padded_bin((31 - lane) * 32 + j)];
    uint32_t inclusive = warp_inclusive_scan(sum, lane);
    uint32_t before = inclusive - sum,
             mask = __ballot_sync(Full, before < K && inclusive >= K);
    if (mask) {
      uint32_t chosen = __ffs(mask) - 1, group = 31 - chosen,
               rank = K - __shfl_sync(Full, before, chosen),
               bin = group * 32 + 31 - lane;
      sum = shared.hist[local][padded_bin(bin)];
      inclusive = warp_inclusive_scan(sum, lane);
      before = inclusive - sum;
      chosen =
          __ffs(__ballot_sync(Full, before < rank && inclusive >= rank)) - 1;
      bin = __shfl_sync(Full, bin, chosen);
      gate = fmaxf(gate, half_bin_floor<6>(bin));
      if (lane == 0) atomicExch(shared.gates + local, __float_as_uint(gate));
    }
    uint32_t written = shared.kept[local];
    uint64_t base = (uint64_t)row * kRetainedPerRow;
#pragma unroll
    for (uint32_t i = 0; i < Items; ++i) {
      float v = values[i];
      bool pass = isfinite(v) && v >= gate;
      uint32_t mask = __ballot_sync(Full, pass),
               pos = written + __popc(mask & ((1u << lane) - 1));
      if (pass) {
        buffers.retained[base + pos] = v;
        buffers.retained_indices[base + pos] = first + tid + i * 32;
      }
      written += __popc(mask);
    }
    __syncwarp();
    if (lane == 0) shared.kept[local] = written;
    __syncwarp();
  }
}

// Streams each row's scores (above its running gate) into the segment ring and
// merges every full segment into the row's retained top set; the final set is
// written to the output.
struct RescanPolicy {
  static constexpr float kWeightScale = 0.5f;
  static constexpr uint32_t kSplitsPerSegment = kSegmentKeys / 256;
  RescanShared& shared;
  RescanBuffers buffers;
  int* out;  // [rows, topk]
  uint32_t topk, query_base, row_base, segment = 0;
  uint32_t seq_k_start[4], seq_k_end[4];
  float gate[4];
  CUTLASS_DEVICE void begin_block(const Scheduler& sched, uint32_t q_block,
                                  uint32_t base) {
    query_base = base;
    row_base = sched.get_logits_row(q_block, 0);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t row = sched.get_logits_row(q_block, base + i);
      gate[i] = -CUDART_INF_F;
      seq_k_start[i] = min(sched.cu_seq_len_k_start[row], sched.num_kv_tokens);
      seq_k_end[i] = min(sched.cu_seq_len_k_end[row], sched.num_kv_tokens);
    }
  }
  CUTLASS_DEVICE void begin_split(uint32_t split) {
    if (split % kSplitsPerSegment == 0) {
      __syncwarp();
      for (uint32_t i = 0; i < 4; ++i)
        gate[i] = __uint_as_float(atomicAdd(shared.gates + query_base + i, 0u));
    }
  }
  CUTLASS_DEVICE void score(uint32_t i, uint32_t kv_offset, uint32_t kv_lane,
                            uint32_t split, float bq) {
    const bool valid = kv_offset >= seq_k_start[i] &&
                       kv_offset < seq_k_end[i] && isfinite(bq) && bq > gate[i];
    shared.ring[segment & 1][query_base + i]
               [(split % kSplitsPerSegment) * 256 + kv_lane] =
        valid ? bq : -CUDART_INF_F;
  }
  CUTLASS_DEVICE void end_split(uint32_t split, uint32_t num_splits,
                                uint32_t kv_base) {
    if ((split + 1) % kSplitsPerSegment != 0 && split + 1 != num_splits) return;
    __threadfence_block();
    math_wg_sync();
    const uint32_t tid = threadIdx.x;
    for (uint32_t local = 0; local < 8; ++local)
      if (shared.kept[local] > kRetainedPerRow - kSegmentKeys)
        compact_retained(shared, buffers, topk, local, row_base + local, tid);
    math_wg_sync();
    const uint32_t local = tid / 32;
    merge_segment(shared, buffers, topk, local, row_base + local, segment,
                  kv_base + (split / kSplitsPerSegment) * kSegmentKeys,
                  (split % kSplitsPerSegment + 1) * 256, tid % 32);
    math_wg_sync();
    if (split + 1 == num_splits)
      for (uint32_t l = 0; l < 8; ++l)
        if (shared.kept[l] > topk)
          compact_retained(shared, buffers, topk, l, row_base + l, tid, true);
    math_wg_sync();
    ++segment;
  }
  CUTLASS_DEVICE void end_block(const Scheduler& sched, uint32_t q_block) {
    const uint32_t local = threadIdx.x / 32, lane = threadIdx.x % 32;
    const uint32_t row = sched.get_logits_row(q_block, local),
                   count = shared.kept[local];
    if (row < sched.num_q_tokens)
      for (uint32_t j = lane; j < topk; j += 32)
        out[(uint64_t)row * topk + j] =
            j < count
                ? buffers.retained_indices[(uint64_t)row * kRetainedPerRow + j]
                : -1;
  }
};

// Rescans one block exactly and writes its output rows of `out`; `buffers`
// rows are addressed by absolute row (callers offset them). Out of line: the
// rare rescan must not add register pressure to the caller's hot loops.
__device__ __noinline__ inline void rescan_block(
    const Scheduler& block, RescanBuffers buffers, int* out, uint32_t topk,
    const cute::TmaDescriptor& tq, const cute::TmaDescriptor& tk,
    const cute::TmaDescriptor& ts, const cute::TmaDescriptor& tw) {
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  auto& shared = *reinterpret_cast<RescanShared*>(smem_buffer +
                                                  sizeof(dense::SharedStorage));
  shared.init(threadIdx.x);
  RescanPolicy policy{shared, buffers, out, topk};
  dense::score_block(block, policy, tq, tk, ts, tw);
}

}  // namespace fused_mqa_topk::rescan
