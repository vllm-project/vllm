// SPDX-License-Identifier: MIT
// Exact recovery for blocks whose gated candidate count left [K, 16384] (or
// whose weights are too small to halve exactly): the block is rescanned
// densely, and every 1024-key segment of scores is merged into a running
// per-row top set (kept <= FallbackRetain, compacted to K).
#pragma once
#include <cutlass/arch/barrier.h>
#include "dense_scan.cuh"
#include <stdint.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_fp16.h>
constexpr unsigned FallbackSegment = 1024, FallbackRetain = 4096;
// output: [rows, K] int32 (K = the row top-k).
struct FallbackBuffers {
  float* retained;
  int* indices;
  int* output;
};
struct alignas(16) FallbackShared {
  float ring[2][8][FallbackSegment];
  unsigned hist[8][1056], groups[8][32], gates[8], kept[8];
  unsigned work[264], prefix, rank, written, added, emit_counts[8];
  __device__ void init(unsigned tid) {
    for (unsigned j = tid; j < 8 * 1056; j += 384)
      reinterpret_cast<unsigned*>(hist)[j] = 0;
    if (tid < 256) reinterpret_cast<unsigned*>(groups)[tid] = 0;
    if (tid < 8) {
      gates[tid] = __float_as_uint(-CUDART_INF_F);
      kept[tid] = 0;
    }
  }
};
__device__ __forceinline__ unsigned fb_code(float x) {
  unsigned u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}
__device__ __forceinline__ float fb_float(unsigned u) {
  return __uint_as_float((u & 0x80000000u) ? (u ^ 0x80000000u) : ~u);
}
__device__ __forceinline__ unsigned fb_half_bin(float x) {
  unsigned u = __half_as_ushort(__float2half_rd(x));
  return ((u & 0x8000u) ? ((~u) & 65535u) : (u ^ 0x8000u)) >> 6;
}
__device__ __forceinline__ float fb_bin_floor(unsigned b) {
  unsigned o = b << 6, u = (o & 0x8000u) ? (o ^ 0x8000u) : ((~o) & 65535u);
  return __half2float(__ushort_as_half(u));
}
__device__ __forceinline__ void fb_sync() {
  asm volatile("" ::: "memory");
  cutlass::arch::NamedBarrier(256, 2).sync();
  asm volatile("" ::: "memory");
}
__device__ void fb_compact(FallbackShared& s, FallbackBuffers b, unsigned K,
                           unsigned local, unsigned row, unsigned tid,
                           bool final = false) {
  constexpr unsigned Full = 0xffffffffu, T = 256, Items = FallbackRetain / T;
  unsigned lane = tid & 31, count = s.kept[local];
  uint64_t base = (uint64_t)row * FallbackRetain;
  float values[Items];
  int indices[Items];
#pragma unroll
  for (unsigned i = 0; i < Items; ++i) {
    unsigned j = tid + i * T;
    values[i] = j < count ? b.retained[base + j] : -CUDART_INF_F;
    indices[i] = j < count ? b.indices[base + j] : -1;
  }
  auto emit = [&](float gate, unsigned threshold, int mode,
                  unsigned base_offset) {
    unsigned total = 0;
#pragma unroll
    for (unsigned i = 0; i < Items; ++i) {
      float v = values[i];
      bool pass =
          isfinite(v) && (mode == 0 ? v >= gate
                                    : (mode == 1 ? fb_code(v) > threshold
                                                 : fb_code(v) == threshold));
      total += pass;
    }
    unsigned warp_total = __reduce_add_sync(Full, total);
    if (lane == 0) s.emit_counts[tid / 32] = warp_total;
    fb_sync();
    unsigned start = base_offset;
    for (unsigned w = 0; w < tid / 32; ++w) start += s.emit_counts[w];
    unsigned all = 0;
    for (unsigned w = 0; w < 8; ++w) all += s.emit_counts[w];
#pragma unroll
    for (unsigned i = 0; i < Items; ++i) {
      float v = values[i];
      bool pass =
          isfinite(v) && (mode == 0 ? v >= gate
                                    : (mode == 1 ? fb_code(v) > threshold
                                                 : fb_code(v) == threshold));
      unsigned mask = __ballot_sync(Full, pass),
               pos = start + __popc(mask & ((1u << lane) - 1));
      if (pass && (mode != 2 || pos < K)) {
        b.retained[base + pos] = v;
        b.indices[base + pos] = indices[i];
      }
      start += __popc(mask);
    }
    fb_sync();
    return all;
  };
  if (!final) {
    if (tid == 0) {
      s.rank = 0;
      s.written = 0;
    }
    fb_sync();
    float gate = __uint_as_float(s.gates[local]);
    unsigned kept = 0;
#pragma unroll
    for (unsigned i = 0; i < Items; ++i)
      kept += isfinite(values[i]) && values[i] >= gate;
    kept = __reduce_add_sync(Full, kept);
    if (lane == 0) atomicAdd(&s.rank, kept);
    fb_sync();
    kept = s.rank;
    if (kept >= K && kept <= FallbackRetain - FallbackSegment) {
      emit(gate, 0, 0, 0);
      fb_sync();
      if (tid == 0) s.kept[local] = kept;
      fb_sync();
      return;
    }
  }
  if (tid == 0) {
    s.prefix = 0;
    s.rank = K;
    s.written = 0;
  }
  fb_sync();
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (unsigned j = tid; j < 264; j += T) s.work[j] = 0;
    fb_sync();
    unsigned prefix = s.prefix;
#pragma unroll
    for (unsigned i = 0; i < Items; ++i) {
      unsigned code = fb_code(values[i]);
      bool valid = isfinite(values[i]) &&
                   (shift == 24 || (code >> (shift + 8)) == prefix);
      unsigned mask = __ballot_sync(Full, valid);
      if (valid) {
        unsigned bin = (code >> shift) & 255u,
                 peers = __match_any_sync(mask, bin);
        if (lane == unsigned(__ffs(peers) - 1))
          atomicAdd(s.work + bin + bin / 32, __popc(peers));
      }
    }
    fb_sync();
    if (tid < 32) {
      unsigned sum = 0;
      for (unsigned j = 0; j < 8; ++j) {
        unsigned bin = 255 - lane * 8 - j;
        sum += s.work[bin + bin / 32];
      }
      unsigned inclusive = sum;
      for (unsigned d = 1; d < 32; d *= 2) {
        unsigned v = __shfl_up_sync(Full, inclusive, d);
        if (lane >= d) inclusive += v;
      }
      unsigned before = inclusive - sum, rank = s.rank,
               crossing =
                   __ballot_sync(Full, before < rank && inclusive >= rank),
               chosen_lane = __ffs(crossing) - 1, chosen = 0, above = before;
      if (lane == chosen_lane) {
        for (unsigned j = 0; j < 8; ++j) {
          unsigned bin = 255 - lane * 8 - j;
          if (above + s.work[bin + bin / 32] >= rank) {
            chosen = bin;
            break;
          }
          above += s.work[bin + bin / 32];
        }
      }
      chosen = __shfl_sync(Full, chosen, chosen_lane);
      above = __shfl_sync(Full, above, chosen_lane);
      __syncwarp();  // every lane read s.rank above
      if (lane == 0) {
        s.prefix = (prefix << 8) | chosen;
        s.rank = rank - above;
      }
    }
    fb_sync();
  }
  unsigned threshold = s.prefix;
  unsigned greater = emit(0, threshold, 1, 0);
  emit(0, threshold, 2, greater);
  if (tid == 0) {
    s.kept[local] = K;
    float gate = fmaxf(__uint_as_float(s.gates[local]), fb_float(threshold));
    atomicExch(s.gates + local, __float_as_uint(gate));
  }
  fb_sync();
}
__device__ void fb_select(FallbackShared& s, FallbackBuffers b, unsigned K,
                          unsigned local, unsigned row, unsigned epoch,
                          unsigned first, unsigned n, unsigned tid,
                          bool final) {
  constexpr unsigned Full = 0xffffffffu, T = 32, Items = FallbackSegment / T;
  unsigned lane = tid, bank = epoch & 1;
  float values[Items], gate = __uint_as_float(s.gates[local]);
  unsigned count = 0;
#pragma unroll
  for (unsigned i = 0; i < Items; ++i) {
    unsigned j = tid + i * T;
    float v = j < n ? s.ring[bank][local][j] : -CUDART_INF_F;
    if (v <= gate) v = -CUDART_INF_F;
    values[i] = v;
    bool valid = isfinite(v);
    count += valid;
    unsigned mask = __ballot_sync(Full, valid);
    if (valid) {
      unsigned bin = fb_half_bin(v), peers = __match_any_sync(mask, bin);
      if (lane == unsigned(__ffs(peers) - 1))
        atomicAdd(&s.hist[local][bin + bin / 32], __popc(peers));
    }
  }
  count = __reduce_add_sync(Full, count);
  __syncwarp();
  if (count) {
    unsigned sum = 0;
#pragma unroll
    for (unsigned j = 0; j < 32; ++j)
      sum += s.hist[local][(31 - lane) * 33 + j];
    unsigned inclusive = sum;
    for (unsigned d = 1; d < 32; d *= 2) {
      unsigned v = __shfl_up_sync(Full, inclusive, d);
      if (lane >= d) inclusive += v;
    }
    unsigned before = inclusive - sum,
             mask = __ballot_sync(Full, before < K && inclusive >= K);
    if (mask) {
      unsigned chosen = __ffs(mask) - 1, group = 31 - chosen,
               rank = K - __shfl_sync(Full, before, chosen),
               bin = group * 32 + 31 - lane;
      sum = s.hist[local][bin + bin / 32];
      inclusive = sum;
      for (unsigned d = 1; d < 32; d *= 2) {
        unsigned v = __shfl_up_sync(Full, inclusive, d);
        if (lane >= d) inclusive += v;
      }
      before = inclusive - sum;
      chosen =
          __ffs(__ballot_sync(Full, before < rank && inclusive >= rank)) - 1;
      bin = __shfl_sync(Full, bin, chosen);
      gate = fmaxf(gate, fb_bin_floor(bin));
      if (lane == 0) atomicExch(s.gates + local, __float_as_uint(gate));
    }
    unsigned written = s.kept[local];
    uint64_t base = (uint64_t)row * FallbackRetain;
#pragma unroll
    for (unsigned i = 0; i < Items; ++i) {
      float v = values[i];
      bool pass = isfinite(v) && v >= gate;
      unsigned mask = __ballot_sync(Full, pass),
               pos = written + __popc(mask & ((1u << lane) - 1));
      if (pass) {
        b.retained[base + pos] = v;
        b.indices[base + pos] = first + tid + i * T;
      }
      written += __popc(mask);
    }
    __syncwarp();
    if (lane == 0) s.kept[local] = written;
    __syncwarp();
  }
}

namespace fused_mqa_fallback {
using fused_mqa_dense::Scheduler;

// Streams each row's scores (above its running gate) into the segment ring and
// merges every full segment into the row's retained top set; the final set is
// written to the output.
struct RescanPolicy {
  static constexpr float kWeightScale = 0.5f, kScoreScale = 1.0f;
  static constexpr uint32_t kSplitsPerSegment = FallbackSegment / 256;
  FallbackShared& stream;
  FallbackBuffers buffers;
  uint32_t topk, query_base, row_base, epoch = 0;
  uint32_t seq_k_start[4], seq_k_end[4];
  float gate[4];
  CUTLASS_DEVICE void begin_block(const Scheduler& s, uint32_t q_block,
                                  uint32_t base) {
    query_base = base;
    row_base = s.get_logits_row(q_block, 0);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t row = s.get_logits_row(q_block, base + i);
      gate[i] = -CUDART_INF_F;
      seq_k_start[i] = min(s.cu_seq_len_k_start[row], s.num_kv_tokens);
      seq_k_end[i] = min(s.cu_seq_len_k_end[row], s.num_kv_tokens);
    }
  }
  CUTLASS_DEVICE void begin_split(uint32_t split, uint32_t) {
    if (split % kSplitsPerSegment == 0) {
      __syncwarp();
      for (uint32_t i = 0; i < 4; ++i)
        gate[i] = __uint_as_float(atomicAdd(stream.gates + query_base + i, 0u));
    }
  }
  CUTLASS_DEVICE void score(uint32_t i, uint32_t kv_offset, uint32_t kv_lane,
                            uint32_t split, float bq) {
    const bool valid = kv_offset >= seq_k_start[i] &&
                       kv_offset < seq_k_end[i] && isfinite(bq) && bq > gate[i];
    stream.ring[epoch & 1][query_base + i]
               [(split % kSplitsPerSegment) * 256 + kv_lane] =
        valid ? bq : -CUDART_INF_F;
  }
  CUTLASS_DEVICE void end_split(uint32_t split, uint32_t num_splits,
                                uint32_t kv_base) {
    if ((split + 1) % kSplitsPerSegment != 0 && split + 1 != num_splits) return;
    __threadfence_block();
    fb_sync();
    const unsigned tid = threadIdx.x;
    for (unsigned local = 0; local < 8; ++local)
      if (stream.kept[local] > FallbackRetain - FallbackSegment)
        fb_compact(stream, buffers, topk, local, row_base + local, tid);
    fb_sync();
    const unsigned local = tid / 32;
    fb_select(stream, buffers, topk, local, row_base + local, epoch,
              kv_base + (split / kSplitsPerSegment) * FallbackSegment,
              (split % kSplitsPerSegment + 1) * 256, tid % 32, false);
    fb_sync();
    if (split + 1 == num_splits)
      for (unsigned l = 0; l < 8; ++l)
        if (stream.kept[l] > topk)
          fb_compact(stream, buffers, topk, l, row_base + l, tid, true);
    fb_sync();
    ++epoch;
  }
  CUTLASS_DEVICE void end_block(const Scheduler& s, uint32_t q_block) {
    const unsigned local = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned row = s.get_logits_row(q_block, local),
                   count = stream.kept[local];
    if (row < s.num_q_tokens)
      for (unsigned j = lane; j < topk; j += 32)
        buffers.output[(uint64_t)row * topk + j] =
            j < count ? buffers.indices[(uint64_t)row * FallbackRetain + j]
                      : -1;
  }
};

// Rescans one block exactly; `buffers` rows are addressed by absolute row
// (callers offset them). Out of line: the rare rescan must not add register
// pressure to the caller's hot loops.
__device__ __noinline__ void rescan_block(
    const Scheduler& block, FallbackBuffers buffers, unsigned topk,
    const cute::TmaDescriptor& tq, const cute::TmaDescriptor& tk,
    const cute::TmaDescriptor& ts, const cute::TmaDescriptor& tw) {
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  auto& stream = *reinterpret_cast<FallbackShared*>(
      smem_buffer + sizeof(fused_mqa_dense::SharedStorage));
  stream.init(threadIdx.x);
  RescanPolicy policy{stream, buffers, topk};
  fused_mqa_dense::score_block(block, policy, tq, tk, ts, tw);
}

}  // namespace fused_mqa_fallback
