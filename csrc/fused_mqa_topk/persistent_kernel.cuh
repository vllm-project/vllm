// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Persistent fused FP8 MQA logits + per-row top-K (K = 512, 1024, 2048; SM100),
// DeepGEMM/FA4-style: one CTA per SM, barriers and TMEM initialised once, every
// warp role runs its own loop over blocks claimed (heaviest first) from a
// global counter, and the sample and scan tiles of a block flow through ONE
// continuous Q/KV/TMEM pipeline whose stage/phase state is carried across
// blocks.
//
// Roles (384 threads):
//   warp 8   claimer + Q/weights TMA + drain of math WG1 rows
//   warp 9   KV TMA (sample tiles via the strided map, then the scan tiles)
//   warp 10  UMMA issuer
//   warp 11  linear-term rows + drain of math WG0 rows
//   warps 0-7 math: sample histogram -> gate -> scan/emit -> selection
// A block is rescanned exactly (rescan.cuh) when a row's candidate count
// leaves [K, kCandidatesPerRow], a row's weights are unsafe to scale, or the
// block's candidates exceed the selection scratch (~11K+ per row): the role
// loops pause, the CTA rescans its queued blocks, and the loops resume with
// freshly initialised barriers.
#pragma once
#include <math_constants.h>

#include <cstddef>
#include <cuda_fp8.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

#include "candidates.cuh"
#include "common.cuh"
#include "constants.h"
#include "deep_gemm_compat.cuh"
#include "rescan.cuh"
#include "select.cuh"

namespace fused_mqa_topk::persist {
using namespace deep_gemm;
using rescan::RescanBuffers;
using rescan::RescanShared;
using TBar = cutlass::arch::ClusterTransactionBarrier;

constexpr uint32_t kTerm = 0xffffffffu, kPause = 0xfffffffeu;
// Bytes of a block's Q, of one KV tile and its per-key scales, of a block's
// weights.
constexpr uint32_t kQBytes = kBlockRows * kNumHeads * kHeadDim,
                   kKVBytes = kTileKeys * kHeadDim,
                   kScaleBytes = kTileKeys * sizeof(float),
                   kWBytes = kBlockRows * kNumHeads * sizeof(float);
// KV (and per-key scale) pipeline stages.
constexpr uint32_t kKVStages = 3;
// Linear-term B rows per math warpgroup: 4 query rows x 4 FP8 terms.
constexpr uint32_t kLinRows = 16;
// Terms 2 and 3 of the linear term are stored in units of s * 2^-8.
constexpr float kLinLoScale = 0x1p-8f;
// TMEM: one UMMA per math warpgroup and KV half writes 4 rows x 32 heads plus
// the kLinRows linear-term columns (N = 144). Slots are 160 columns apart (144
// rounded up to a multiple of 32); the 3 slots take 480 of the 512 allocated
// columns.
constexpr uint32_t kUmmaN = 4 * kNumHeads + kLinRows, kSlotCols = 160;
static_assert(kSlotCols >= kUmmaN && kSlotCols % 32 == 0);

// Rebalanced scan: the linear-term rows of each math warpgroup bracket Q, so
// WG0's N=144 UMMA reads B = [lin WG0 | Q rows 0..127] (D = [lin | heads]) and
// WG1's reads B = [Q rows 128..255 | lin WG1] (D = [heads | lin]).
struct KernelSmem {
  alignas(1024) uint8_t lin_lo[kLinRows * kHeadDim];
  cutlass::float_e4m3_t q[kQBytes];
  uint8_t lin_hi[kLinRows * kHeadDim];
  alignas(128) float weights[kBlockRows * kNumHeads];
  float lin_scale[kBlockRows];
  alignas(1024) cutlass::float_e4m3_t kv[kKVStages][kKVBytes];
  alignas(128) float sf_kv[kKVStages][kTileKeys];
};
static_assert(offsetof(KernelSmem, q) == kLinRows * kHeadDim &&
                  offsetof(KernelSmem, lin_hi) == kLinRows * kHeadDim + kQBytes,
              "lin rows must bracket Q");
// Selection scratch starts after lin + Q + lin + weights + row scales, so the
// next block's Q and linear term land during this block's selection.
constexpr uint32_t kSelBase = offsetof(KernelSmem, kv);

// Byte offset of byte `col` of row `row` in a 128B-swizzled K-major tile of
// 128-byte rows (the UMMA B layout of Q and the linear-term rows).
__device__ __forceinline__ uint32_t swizzle128_offset(uint32_t row,
                                                      uint32_t col) {
  return row * 128u + (((col >> 4) ^ (row & 7u)) << 4) + (col & 15u);
}

// q'_r = sum_h (w_rh / 2) q_rh, split into 4 FP8 e4m3 terms: terms 0,1 scale
// s_r, terms 2,3 s_r * 2^-8 (lin_scale[r]); term t of row r is B row (r % 4) *
// 4 + t of lin_lo (rows 0-3) / lin_hi (4-7), 128B-swizzled K-major like Q. One
// warp; lane l owns dims 4l..4l+3.
__device__ __forceinline__ void compute_lin_rows(KernelSmem& sm,
                                                 uint32_t lane) {
  const uint8_t* qbytes = reinterpret_cast<const uint8_t*>(sm.q);
#pragma unroll 1
  for (uint32_t r = 0; r < kBlockRows; ++r) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
    for (uint32_t h = 0; h < kNumHeads; ++h) {
      const uint32_t off = swizzle128_offset(r * kNumHeads + h, lane * 4u);
      const uint32_t packed =
          ptx::ld_shared(reinterpret_cast<const uint32_t*>(qbytes + off));
      const float wv = ptx::ld_shared(sm.weights + r * kNumHeads + h) * 0.5f;
      const __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2(
          static_cast<__nv_fp8x2_storage_t>(packed & 0xffffu), __NV_E4M3);
      const __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(
          static_cast<__nv_fp8x2_storage_t>(packed >> 16), __NV_E4M3);
      const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&lo));
      const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&hi));
      acc[0] = fmaf(wv, f0.x, acc[0]);
      acc[1] = fmaf(wv, f0.y, acc[1]);
      acc[2] = fmaf(wv, f1.x, acc[2]);
      acc[3] = fmaf(wv, f1.y, acc[3]);
    }
    float m = fmaxf(fmaxf(fabsf(acc[0]), fabsf(acc[1])),
                    fmaxf(fabsf(acc[2]), fabsf(acc[3])));
#pragma unroll
    for (uint32_t o = 16; o > 0; o >>= 1)
      m = fmaxf(m, __shfl_xor_sync(kFullMask, m, o));
    int e = -120;  // pow2 scale s >= m / 448, clamped so 1/s stays finite
    if (m > 0.f && isfinite(m)) {
      frexpf(m * (1.0f / 448.0f), &e);
      e = max(e, -120);
    }
    const float s = ldexpf(1.0f, e), inv_s = ldexpf(1.0f, -e);
    float res[4];
#pragma unroll
    for (uint32_t dd = 0; dd < 4; ++dd) res[dd] = acc[dd] * inv_s;
    uint8_t* lin_base = r < kWgRows ? sm.lin_lo : sm.lin_hi;
#pragma unroll
    for (uint32_t tt = 0; tt < 4; ++tt) {
      const __nv_fp8x2_storage_t c01 = __nv_cvt_float2_to_fp8x2(
          make_float2(res[0], res[1]), __NV_SATFINITE, __NV_E4M3);
      const __nv_fp8x2_storage_t c23 = __nv_cvt_float2_to_fp8x2(
          make_float2(res[2], res[3]), __NV_SATFINITE, __NV_E4M3);
      const __half2_raw d01 = __nv_cvt_fp8x2_to_halfraw2(c01, __NV_E4M3);
      const __half2_raw d23 = __nv_cvt_fp8x2_to_halfraw2(c23, __NV_E4M3);
      const float2 g01 =
          __half22float2(*reinterpret_cast<const __half2*>(&d01));
      const float2 g23 =
          __half22float2(*reinterpret_cast<const __half2*>(&d23));
      res[0] -= g01.x;
      res[1] -= g01.y;
      res[2] -= g23.x;
      res[3] -= g23.y;
      // Terms 2, 3 in units of s * 2^-8: the residual after two terms is <=
      // 0.5, so x256 <= 128 (no saturation), and the e4m3 subnormal floor drops
      // from s * 2^-10 to s * 2^-18 per dim.
      if (tt == 1) {
        res[0] *= 1.0f / kLinLoScale;
        res[1] *= 1.0f / kLinLoScale;
        res[2] *= 1.0f / kLinLoScale;
        res[3] *= 1.0f / kLinLoScale;
      }
      const uint32_t off =
          swizzle128_offset((r % kWgRows) * 4u + tt, lane * 4u);
      ptx::st_shared(
          reinterpret_cast<uint32_t*>(lin_base + off),
          static_cast<uint32_t>(c01) | (static_cast<uint32_t>(c23) << 16));
    }
    if (lane == 0) sm.lin_scale[r] = s;
  }
}

// One row's weighted head reduction seeded with its linear-term columns
// (terms 2, 3 in units of s * 2^-8).
__device__ __forceinline__ float reduce_heads_with_lin(const float* s,
                                                       const float* w,
                                                       const float* lin) {
  return reduce_heads<false>(s, w, make_float2(lin[0], lin[1]),
                             __fmul2_rn(make_float2(lin[2], lin[3]),
                                        make_float2(kLinLoScale, kLinLoScale)));
}

// Gate sample state: each math warp histograms its row's sampled scores
// relative to the row's reference `ref` (its max over the first sample tile):
// half_bin<5>(x - ref), kSampleBins bins, 32 per octave of the distance, so
// the bins near the row's top stay fine whatever the scores' common offset
// (absolute bins are ~2 wide at |x| ~ 100, too coarse for top tails that
// narrow).
constexpr uint32_t kSampleShift = 5, kSampleBins = 65536 >> kSampleShift;
struct alignas(16) SampleShared {
  uint32_t hist[kBlockRows][padded_bins(kSampleBins)];
  // Ordered code of each row's max over the first sample tile (0: none).
  uint32_t ref_code[kBlockRows];
  // The first tile's scores, held until the reference is known (in registers
  // they cost the hot loops spills).
  float first[2][kWgRows][kMathThreads];
};

__device__ __forceinline__ void add_sample(uint32_t* hist, float distance) {
  atomicAdd(hist + padded_bin(half_bin<kSampleShift>(distance)), 1u);
}
__device__ __forceinline__ uint32_t sample_count(const uint32_t* hist,
                                                 uint32_t bin) {
  return hist[padded_bin(bin)];
}
// The row reference from its ordered code; 0 when the row had no valid sample
// in the first tile (then the bins are absolute).
__device__ __forceinline__ float sample_ref(uint32_t code) {
  const float ref = ordered_float(code);
  return isfinite(ref) ? ref : 0.0f;
}

// The gate aims kGateSlack16 / 16 above the target count, so that few rows fall
// below the top-k (and into the exact rescan) when the sampled estimate runs
// low. On the B300 captures 17/16 cuts the rescanned blocks by ~85% at about
// the candidate counts of the coarser absolute bins.
constexpr uint32_t kGateSlack16 = 17;

// Gate: the k-th largest sample (k = count * target / visible, count = the
// histogram total) of one row from its histogram; one warp per row. -inf when
// the row sees at most `target` keys or has no samples.
__device__ __noinline__ inline void finish_sample_histogram(
    const uint32_t* hist, float ref, float* gate, uint32_t visible,
    uint32_t target, uint32_t lane) {
  if (visible <= target) {
    if (lane == 0) *gate = -CUDART_INF_F;
    __syncwarp();
    return;
  }
  // Lane l owns kChunksPerLane 32-bin chunks, descending with the lane: chunk
  // kChunks - 1 - kChunksPerLane * l - j for j < kChunksPerLane.
  constexpr uint32_t kChunks = kSampleBins / 32, kChunksPerLane = kChunks / 32;
  uint32_t sum = 0;
#pragma unroll
  for (uint32_t j = 0; j < kChunksPerLane; ++j) {
    const uint32_t c = kChunks - 1 - kChunksPerLane * lane - j;
#pragma unroll 8
    for (uint32_t b = 0; b < 32; ++b) sum += hist[padded_bin(c * 32 + b)];
  }
  const uint32_t incl = warp_inclusive_scan(sum, lane);
  const uint32_t tot = __shfl_sync(kFullMask, incl, 31);
  if (tot == 0) {
    if (lane == 0) *gate = -CUDART_INF_F;
    __syncwarp();
    return;
  }
  const uint32_t rank = max(
      1u, min(tot, (tot * target * kGateSlack16 / 16 + visible - 1) / visible));
  const uint32_t w = __ffs(__ballot_sync(kFullMask, incl >= rank)) - 1;
  uint32_t above = __shfl_sync(kFullMask, incl - sum, w), chosen = 0;
#pragma unroll 1
  for (uint32_t cc = 0; cc < kChunksPerLane; ++cc) {
    const uint32_t c = kChunks - 1 - kChunksPerLane * w - cc;
    // Lane 0 = highest bin of the chunk.
    const uint32_t v =
        warp_inclusive_scan(sample_count(hist, c * 32 + 31 - lane), lane);
    const uint32_t mask = __ballot_sync(kFullMask, above + v >= rank);
    if (mask) {
      chosen = c * 32 + 31 - (__ffs(mask) - 1);
      break;
    }
    above += __shfl_sync(kFullMask, v, 31);
  }
  if (lane == 0) *gate = ref + half_bin_floor<kSampleShift>(chosen);
  __syncwarp();
}

// Candidate records each lane of a math warp buffers per row and handoff
// interval before it falls back to direct global stores.
constexpr uint32_t kRingSlots = 8;
// The drain's prefix sum packs two 16-bit per-row counts into one word.
static_assert(32 * kRingSlots < 65536,
              "per-warp record count must fit 16 bits");

// Handoff ring: [bank][math warp][row][slot][lane] records, per-lane record
// counts and the interval's first scan tile.
struct alignas(16) HandoffRing {
  uint32_t records[2][kMathWarps][kWgRows][kRingSlots][32];
  uint32_t num_records[2][kMathWarps][kWgRows][32];
  uint32_t bases[2][kMathWarps];
};

// samples: gate sample tiles of the block (0 when no row sees more keys than
// the gate target, so every gate is -inf).
struct ClaimHeader {
  uint32_t vblock, all_short, kv_base, splits, slot, samples;
};
struct ClaimEntry : ClaimHeader {
  uint32_t starts[kBlockRows], ends[kBlockRows];
};

// Claim k in a producer role (warps 9-11): waits for the entry, copies its
// header and releases the entry (every lane arrives). False once the claimer
// ends the epoch (kTerm / kPause).
__device__ __forceinline__ bool consume_claim(uint32_t k, TBar* claim_full,
                                              TBar* claim_empty,
                                              const ClaimEntry* entries,
                                              ClaimHeader& claim) {
  claim_full[k & 1].wait((k >> 1) & 1);
  claim = entries[k & 1];
  __syncwarp();
  if (claim.vblock >= kPause) return false;
  claim_empty[k & 1].arrive();
  return true;
}

// Barrier words (static shared, never aliased with data).
struct BarIdx {
  // TMEM slots, shared by the two math warpgroups (WG0 even, WG1 odd
  // positions).
  static constexpr uint32_t kTmemSlots = 3;
  static constexpr uint32_t full_q = 0, empty_q = 1, full_kv = 2,
                            empty_kv = 2 + kKVStages,
                            full_tmem = 2 + 2 * kKVStages,
                            empty_tmem = full_tmem + kTmemSlots,
                            claim_full = empty_tmem + kTmemSlots,
                            claim_empty = claim_full + 2,
                            sel_done = claim_empty + 2, h_full = sel_done + 1,
                            h_empty = h_full + 2 * kMathWarps,
                            full_lin = h_empty + 2 * kMathWarps,
                            total = full_lin + 1;
};

// Expect `bytes` of TMA transactions and arrive `count` times on one barrier.
__device__ __forceinline__ void mbar_arrive_expect_tx(const void* b,
                                                      uint32_t bytes,
                                                      uint32_t count) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(b));
  asm volatile(
      "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(a),
      "r"(bytes)
      : "memory");
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(a),
      "r"(count)
      : "memory");
}

// Candidate slabs: a block's 8 rows live in its CTA slot's 8 slab rows, so row
// r maps to slab row r + slot_row_delta(slot, row_base).
__device__ __forceinline__ int64_t slot_row_delta(uint32_t slot,
                                                  uint32_t row_base) {
  return int64_t(slot) * kBlockRows - int64_t(row_base);
}
// Offset of entry `index` of row `row` in the candidate slabs.
__device__ __forceinline__ uint64_t candidate_offset(uint32_t row,
                                                     int64_t row_delta,
                                                     uint32_t index) {
  return static_cast<uint64_t>(int64_t(row) + row_delta) * kCandidatesPerRow +
         index;
}
// Stores ring record `record` (fp24 code << 8 | tile offset * 2 + KV half,
// tile offset relative to the interval's first tile `interval_base`) of KV
// lane `kv_lane` as entry `index` of row `row`.
__device__ __forceinline__ void store_ring_record(
    uint16_t* vals, int32_t* indices, uint32_t row, int64_t row_delta,
    uint32_t index, uint32_t record, uint32_t kv_base, uint32_t interval_base,
    uint32_t kv_lane) {
  const uint32_t key =
      kv_base + (interval_base * 2 + (record & 255u)) * 128 + kv_lane;
  const uint64_t off = candidate_offset(row, row_delta, index);
  store_candidate_record(vals + off, indices + off, record >> 8, key);
}

// Output row `out` of a row that sees at most top-k keys: its `visible` keys
// in order (0 .. visible - 1, relative to the row start), then -1 padding.
// Threads t, t + stride, ... write the slots.
__device__ __forceinline__ void write_all_keys(int* out, uint32_t topk,
                                               uint32_t visible, uint32_t t,
                                               uint32_t stride) {
  for (uint32_t j = t; j < topk; j += stride)
    out[j] = j < visible ? int(j) : -1;
}

// Rewrites the output rows of the block at row_base in place from absolute key
// indices to indices relative to the row's start (the top_k_per_row_prefill
// convention; -1 padding stays). Rows that start at 0 and rows for which
// skip(local row) holds are left as they are. Threads t, t + stride, ... each
// rewrite 4 slots.
template <class Skip>
__device__ __forceinline__ void make_relative(int* output, uint32_t topk,
                                              uint32_t rows, uint32_t row_base,
                                              const uint32_t* starts_g,
                                              uint32_t t, uint32_t stride,
                                              Skip skip) {
  const uint32_t quads = topk / 4;
  for (uint32_t i = t; i < kBlockRows * quads; i += stride) {
    const uint32_t r = row_base + i / quads;
    if (r >= rows || skip(i / quads)) continue;
    const int s = int(starts_g[r]);
    if (s == 0) continue;
    int4* p = reinterpret_cast<int4*>(output + (uint64_t)r * topk) + i % quads;
    int4 v = *p;
    v.x = v.x < 0 ? -1 : v.x - s;
    v.y = v.y < 0 ? -1 : v.y - s;
    v.z = v.z < 0 ? -1 : v.z - s;
    v.w = v.w < 0 ? -1 : v.w - s;
    *p = v;
  }
}

// Stores one warp's ring records of a handoff interval in its rows' candidate
// slabs. recs = [4 rows][kRingSlots records][32 lanes]; lane l holds n[row]
// records of each row. Each row's slots are reserved with one atomic on
// row_counts[row_base + row]; records past kCandidatesPerRow are dropped (the
// row's count then exceeds the cap, so its block is rescanned exactly).
// store(row, slot, record) writes one record.
// Written for a warpgroup's 4 rows (two 16-bit counts per word in the scan).
static_assert(kWgRows == 4, "store_ring_records packs 4 row counts");
template <class Store>
__device__ __forceinline__ void store_ring_records(const uint32_t* recs,
                                                   const uint32_t (&n)[4],
                                                   int* row_counts,
                                                   uint32_t row_base,
                                                   Store store) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t my_total = 0;
#pragma unroll
  for (uint32_t row = 0; row < 4; ++row) {
    const uint32_t total = __reduce_add_sync(kFullMask, n[row]);
    if (lane == row) my_total = total;
  }
  uint32_t my_start = 0;
  if (lane < 4 && my_total)
    my_start = atomicAdd(row_counts + row_base + lane, my_total);
  // Two 16-bit per-row counts per word: one scan covers two rows.
  uint32_t p01 = n[0] | (n[1] << 16), p23 = n[2] | (n[3] << 16);
#pragma unroll
  for (int dd = 1; dd < 32; dd *= 2) {
    const uint32_t a = __shfl_up_sync(kFullMask, p01, dd),
                   b = __shfl_up_sync(kFullMask, p23, dd);
    if (lane >= dd) {
      p01 += a;
      p23 += b;
    }
  }
  const uint32_t prefix[4] = {p01 & 65535u, p01 >> 16, p23 & 65535u, p23 >> 16};
#pragma unroll
  for (uint32_t row = 0; row < 4; ++row) {
    const uint32_t start = __shfl_sync(kFullMask, my_start, row);
    for (uint32_t j = 0; j < n[row]; ++j) {
      const uint32_t slot = start + prefix[row] - n[row] + j;
      const uint32_t record = recs[(row * kRingSlots + j) * 32 + lane];
      if (slot < kCandidatesPerRow) store(row_base + row, slot, record);
    }
  }
}

// Self-drain, out of line: stores this warp's ring records of the last
// handoff interval. recs = records[bank][warp].
__device__ __noinline__ inline void self_drain_last(
    const uint32_t* __restrict__ recs, uint32_t n0, uint32_t n1, uint32_t n2,
    uint32_t n3, int* __restrict__ row_counts, uint32_t rbase, uint32_t lane_kv,
    uint32_t flush_base, int64_t row_delta, uint16_t* __restrict__ vals,
    int32_t* __restrict__ indices) {
  const uint32_t n[4] = {n0, n1, n2, n3};
  store_ring_records(recs, n, row_counts, rbase,
                     [&](uint32_t row, uint32_t slot, uint32_t record) {
                       store_ring_record(vals, indices, row, row_delta, slot,
                                         record, lane_kv, flush_base, 0);
                     });
  __syncwarp();
}

// Drain warp: waits for producer warp `producer`'s handoff interval, stores its
// records (unless the producer self-drained it) and frees the ring bank.
template <class Store>
__device__ __forceinline__ void drain_handoff(
    HandoffRing& d, TBar* hfull, TBar* hempty, int producer, uint32_t interval,
    uint32_t row_base, int* row_counts, bool skip, Store store) {
  const int lane = threadIdx.x & 31, bank = interval & 1;
  hfull[bank * kMathWarps + producer].wait((interval >> 1) & 1);
  __syncwarp();
  if (skip) {  // self-drained by the producer
    hempty[bank * kMathWarps + producer].arrive();
    return;
  }
  const uint32_t base = d.bases[bank][producer];
  uint32_t n[4];
#pragma unroll
  for (uint32_t row = 0; row < 4; ++row)
    n[row] = d.num_records[bank][producer][row][lane];
  store_ring_records(&d.records[bank][producer][0][0][0], n, row_counts,
                     row_base,
                     [&](uint32_t row, uint32_t slot, uint32_t record) {
                       store(row, slot, record, base, producer * 32 + lane);
                     });
  __syncwarp();
  hempty[bank * kMathWarps + producer].arrive();
}

// Interval: scan tiles per candidate handoff to the drain warps.
template <uint32_t Interval, uint32_t TopK>
__global__
__launch_bounds__(kNumThreads, 1) void fused_mqa_topk_fp8_prefill_kernel(
    uint32_t rows, uint32_t keys, const uint32_t* starts_g,
    const uint32_t* ends_g, int* out, uint16_t* vals, int* indices,
    int* row_counts, RescanBuffers rescan, uint32_t phase_bytes,
    uint32_t sample_target, uint32_t sample_stride, uint32_t sample_groups,
    uint32_t sample_tiles, const float* scales,
    const __grid_constant__ cute::TmaDescriptor tks,
    const __grid_constant__ cute::TmaDescriptor tq,
    const __grid_constant__ cute::TmaDescriptor ts,
    const __grid_constant__ cute::TmaDescriptor tk,
    const __grid_constant__ cute::TmaDescriptor tw, uint32_t nblocks,
    uint32_t* counter, int* defer_next) {
  // Compile-time: a runtime top-k costs ~0.5% in the scan (codegen).
  constexpr uint32_t topk = TopK;
  static_assert(BarIdx::kTmemSlots * kSlotCols <= 512,
                "TMEM slots exceed 512 columns");
  // A ring record's low byte holds (tile - interval start) * 2 + half, and the
  // handoff points use `& (Interval - 1)`.
  static_assert((Interval & (Interval - 1)) == 0 && Interval <= 128,
                "Interval must be a power of two <= 128");
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  auto& sm = *reinterpret_cast<KernelSmem*>(smem_buffer);
  uint8_t* const region = smem_buffer + sizeof(KernelSmem);
  auto& sample_smem = *reinterpret_cast<SampleShared*>(region);
  auto& hdata = *reinterpret_cast<HandoffRing*>(region);

  __shared__ alignas(8) uint64_t bar_words[BarIdx::total];
  __shared__ uint32_t tmem_ptr;
  __shared__ ClaimEntry entries[2];
  __shared__ float gate_s[kBlockRows];
  __shared__ uint32_t sel_recover, sel_staged, sel_short,
      row_offset[kBlockRows + 1];
  __shared__ uint32_t defer_head, unsafe_rows, pause_req, epoch_kind;

  TBar* const bars = reinterpret_cast<TBar*>(bar_words);
  TBar& full_q = bars[BarIdx::full_q];
  TBar& empty_q = bars[BarIdx::empty_q];
  TBar* const full_kv = bars + BarIdx::full_kv;
  // The 32 producer lanes stage the sample scales, so each arrives on full_kv.
  constexpr uint32_t kFullKvArrivals = 32;
  TBar* const empty_kv = bars + BarIdx::empty_kv;
  TBar* const full_tmem = bars + BarIdx::full_tmem;
  TBar* const empty_tmem = bars + BarIdx::empty_tmem;
  TBar* const claim_full = bars + BarIdx::claim_full;
  TBar* const claim_empty = bars + BarIdx::claim_empty;
  TBar& sel_done = bars[BarIdx::sel_done];
  TBar* const h_full = bars + BarIdx::h_full;
  TBar* const h_empty = bars + BarIdx::h_empty;
  TBar* const full_lin = bars + BarIdx::full_lin;

  const uint32_t warp_idx = cutlass::canonical_warp_idx_sync();
  const uint32_t lane = ptx::get_lane_idx();
  constexpr uint32_t kClaimConsumers = 11;  // warps 9, 10, 11, 0-7

  if (warp_idx == 8 && lane == 0) {
    cute::prefetch_tma_descriptor(&tq);
    cute::prefetch_tma_descriptor(&tw);
    cute::prefetch_tma_descriptor(&tk);
    cute::prefetch_tma_descriptor(&ts);
    cute::prefetch_tma_descriptor(&tks);
  }
  if (warp_idx == 9 && cute::elect_one_sync()) {
    defer_head = kTerm;
    pause_req = 0;
  }
  // Epochs: the persistent role loops run until the work runs out (kTerm) or a
  // block needs the exact rescan (kPause); rare blocks are then rescanned
  // exactly (rescan.cuh) and the loops resume with freshly initialised
  // barriers. The first epoch is emitted outside any loop (nesting the role
  // loops in an epoch loop costs ~1.5 percent in the scan: uniform address math
  // is rematerialised every tile). Later epochs only run after a recovery
  // pause.
  __shared__ uint32_t row_bounds[2][kBlockRows];
  bool more = false;
  {
    constexpr uint32_t epoch = 0;
#include "persistent_epoch.inc"
  }
#pragma unroll 1
  for (uint32_t epoch = 1; more; ++epoch) {
#include "persistent_epoch.inc"
  }
  // counter = {next block, finished CTAs}: the last CTA to finish (every claim
  // is done by then) zeroes both, so the caller's workspace is ready for the
  // next launch / graph replay.
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(counter + 1, 1u) == gridDim.x - 1) {
      // Acquire: every other CTA's last claim on counter happens before the
      // reset below.
      __threadfence();
      atomicExch(counter, 0u);
      atomicExch(counter + 1, 0u);
    }
  }
}

}  // namespace fused_mqa_topk::persist
