// SPDX-License-Identifier: MIT
// <= 4K keys: every score of an 8-row block stays in shared memory as an fp24
// code, so there is no sample, gate or candidate workspace; one warp then
// selects one row.
#pragma once

#include "candidates.cuh"
#include "dense_scan.cuh"
#include "fast_select.cuh"

namespace fused_mqa_resident {
using fused_mqa_dense::BLOCK_Q;
using fused_mqa_dense::Scheduler;
using fused_mqa_dense::SharedStorage;

constexpr uint32_t kResidentKeys = 4096;
// Pipeline storage, then the block's codes: hi16 array then lo8 array,
// row-major, kResidentKeys per row.
constexpr uint32_t kSmemBytes =
    (sizeof(SharedStorage) + BLOCK_Q * kResidentKeys * 3u + 1023u) / 1024u *
    1024u;

// Stores every score as an fp24 code (selection reads only [start, end) of each
// row, so no masking).
struct CodesPolicy {
  static constexpr float kWeightScale = 1.0f, kScoreScale = 0.5f;
  uint8_t* codes;
  uint32_t query_base;
  CUTLASS_DEVICE void begin_block(const Scheduler&, uint32_t, uint32_t base) {
    query_base = base;
  }
  CUTLASS_DEVICE void begin_split(uint32_t, uint32_t) {}
  CUTLASS_DEVICE void score(uint32_t i, uint32_t kv_offset, uint32_t, uint32_t,
                            float score) {
    if (kv_offset < kResidentKeys) {
      const uint32_t code = fused_candidates::candidate_fp24_code(score);
      const uint32_t at = (query_base + i) * kResidentKeys + kv_offset;
      reinterpret_cast<uint16_t*>(codes)[at] = static_cast<uint16_t>(code >> 8);
      codes[BLOCK_Q * kResidentKeys * 2 + at] = static_cast<uint8_t>(code);
    }
  }
  CUTLASS_DEVICE void end_split(uint32_t, uint32_t, uint32_t) {}
  CUTLASS_DEVICE void end_block(const Scheduler&, uint32_t) {}
};

}  // namespace fused_mqa_resident

// <= 4K keys: one CTA per 8-row block, heaviest (latest rows) first.
__global__ __launch_bounds__(384, 1) void fused_mqa_topk_fp8_resident_kernel(
    unsigned rows, unsigned keys, const unsigned* starts_g,
    const unsigned* ends_g, int* output, unsigned topk, bool relative,
    const __grid_constant__ cute::TmaDescriptor tq,
    const __grid_constant__ cute::TmaDescriptor ts,
    const __grid_constant__ cute::TmaDescriptor tk,
    const __grid_constant__ cute::TmaDescriptor tw) {
  using namespace fused_mqa_resident;
  const unsigned vblock = gridDim.x - 1 - blockIdx.x, row_base = vblock * 8;
  // The block's row bounds clamped to the keys; rows past the end see no keys.
  __shared__ unsigned row_bounds[2][8];
  if (threadIdx.x < 8) {
    const unsigned r = row_base + threadIdx.x;
    row_bounds[0][threadIdx.x] = r < rows ? min(starts_g[r], keys) : 0u;
    row_bounds[1][threadIdx.x] =
        r < rows ? max(min(ends_g[r], keys), row_bounds[0][threadIdx.x]) : 0u;
  }
  __syncthreads();
  const unsigned* const starts = row_bounds[0] - row_base;
  const unsigned* const ends = row_bounds[1] - row_base;
  // Rows with <= topk visible keys output all of them (relative indices start
  // at 0).
  const auto write_short = [&](unsigned r, unsigned t, unsigned stride) {
    const unsigned visible = ends[r] - starts[r];
    for (unsigned j = t; j < topk; j += stride)
      output[(uint64_t)r * topk + j] =
          j < visible ? int((relative ? 0u : starts[r]) + j) : -1;
  };
  bool all_short = true;
  for (unsigned r = row_base; r < row_base + 8; ++r)
    all_short &= ends[r] - starts[r] <= topk;
  if (all_short) {
    for (unsigned r = row_base; r < min(row_base + 8, rows); ++r)
      write_short(r, threadIdx.x, 384);
    return;
  }

  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t* const codes = smem + sizeof(SharedStorage);
  CodesPolicy policy{codes, 0};
  fused_mqa_dense::score_block(Scheduler(vblock, rows, keys, starts, ends),
                               policy, tq, tk, ts, tw);
  __syncthreads();
  if (threadIdx.x < 256) cutlass::arch::warpgroup_reg_dealloc<128>();
  __syncthreads();
  if (threadIdx.x >= 256) cutlass::arch::warpgroup_reg_alloc<128>();
  const unsigned warp = threadIdx.x / 32;
  if (warp < 8 && row_base + warp < rows) {
    const unsigned r = row_base + warp, s0 = starts[r], visible = ends[r] - s0;
    if (visible <= topk) {
      write_short(r, threadIdx.x & 31, 32);
    } else {
      int* o = output + (uint64_t)r * topk;
      const uint16_t* hi =
          reinterpret_cast<const uint16_t*>(codes) + warp * kResidentKeys;
      const uint8_t* lo = codes + 8 * kResidentKeys * 2 + warp * kResidentKeys;
      // Scratch (bracket buffer, or the fallback's histogram) reuses the idle
      // Q/KV stage memory.
      static_assert(8 * fast_select::kScratch3Bytes <= sizeof(SharedStorage) &&
                        fast_select::kScratch3Bytes >= fast_select::kBins * 4,
                    "select scratch must fit the Q/KV stages");
      uint8_t* scratch = smem + warp * fast_select::kScratch3Bytes;
      if (!fast_select::select_resident_bracket(hi, lo, s0, visible, o, topk,
                                                scratch))
        fast_select::select_resident(hi + s0, lo + s0, visible, s0, o, topk,
                                     reinterpret_cast<unsigned*>(scratch));
    }
  }
  // Relative indices: selected rows are rewritten in place (L2-hot) once final.
  if (relative) {
    __syncthreads();
    const unsigned quads = topk / 4;
    for (unsigned i = threadIdx.x; i < 8 * quads; i += 384) {
      const unsigned r = row_base + i / quads;
      if (r >= rows || ends[r] - starts[r] <= topk) continue;
      const int s = int(starts_g[r]);
      if (s == 0) continue;
      int4* p =
          reinterpret_cast<int4*>(output + (uint64_t)r * topk) + i % quads;
      int4 v = *p;
      v.x = v.x < 0 ? -1 : v.x - s;
      v.y = v.y < 0 ? -1 : v.y - s;
      v.z = v.z < 0 ? -1 : v.z - s;
      v.w = v.w < 0 ? -1 : v.w - s;
      *p = v;
    }
  }
}
