// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Adapted from SGLang PR #47469
// (https://github.com/sgl-project/sglang/pull/47469, Apache-2.0) at
// 599923b6f48d638d2256abbd0550c8fde141f4f5:
// python/sglang/kernels/jit/csrc/dsa/litetopk_prefill/topk_candidates_fp32.cuh.
// Rows past the slab width are left to the GPU recovery, and wide pivot bins
// take exact_topk instead of truncating.
//
// Top-k over the candidate slab of the prefill DSA indexer: one CTA per row, at
// most 12288 fp32 candidates (all >= the row's sampled threshold T) -> k key
// indices, unordered.
//
// Selection key: y = score - T >= 0, rounded toward -inf to fp16 (`cvt.rz`
// equals `cvt.rm` for y >= 0); the high 12 bits of the 15-bit magnitude index a
// 4096-bin histogram. A block prefix from the top finds the bin holding rank k,
// a second pass emits every element of the higher bins and collects the rank-k
// bin into a 256-slot tie buffer, ranked by exact fp32 score (any copies of the
// k-th value): up to 64 ties with one warp ballot per tie, more with a CTA
// radix select.
//
// Layout: a scores slab plus an index slab. Rows of at most 8192 candidates
// stream by bulk TMA into two resident shared-memory stages; the indices of the
// selected positions are gathered at the end. Rows of up to 12288 candidates
// re-stream their third stage once per pass, and T = -inf rows (short rows that
// keep every key) use the row minimum as T, both on a fallback path.
//
// NaN scores are excluded by construction (the indexer emits finite logits >=
// T). Rows with `count <= k` emit every index. A pivot bin wider than the tie
// buffer takes an exact device-side radix pass over the whole candidate row
// instead of truncating.
#pragma once

#include "device_utils.cuh"
#include "topk_candidates_common.cuh"
#include "exact_topk.cuh"
#include <algorithm>
#include <cstdint>

namespace fused_mqa_topk {

namespace device::ptx {

// Two fp32 -> packed f16x2 (`hi` in the upper half), rounding toward zero.
FMT_DEVICE uint32_t cvt_rz_f16x2(float hi, float lo) {
  uint32_t d;
  asm volatile("cvt.rz.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

// Per-half 0xFFFF / 0x0000 mask of `a >= b` on packed f16x2.
FMT_DEVICE uint32_t set_ge_f16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm volatile("set.ge.u32.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

}  // namespace device::ptx

struct TopKCandidatesFP32Params {
  const float* __restrict__ scores;     // [rows, cap]
  const int32_t* __restrict__ indices;  // [rows, cap]
  const int32_t* __restrict__ count;
  const float* __restrict__ thr;
  int32_t* __restrict__ out;  // [rows, kTopK]
  // The slab width; counts beyond it leave placeholders for the GPU rescan.
  uint32_t cap;
};

// The fp32 candidate top-k of one row: one CTA of 256 threads per row, `run` is
// the kernel body. kTopK_ Selected elements per row (512 / 1024 / 2048).
template <uint32_t kTopK_>
struct TopKCandidatesFP32Impl {
  static constexpr uint32_t kTopK = kTopK_;
  static constexpr uint32_t kBlockSize = 256;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static constexpr uint32_t kMaxCands = 12288;
  static constexpr uint32_t kStageElems = 4096;
  // two resident stage buffers; rows past kResidentElems candidates take a
  // separate path
  static constexpr uint32_t kNumStages = 2;
  static constexpr uint32_t kResidentElems = kNumStages * kStageElems;
  static constexpr uint32_t kNumBins = 4096;
  static constexpr uint32_t kTieCap = 256;  // one tie per thread in radix_ties
  static constexpr uint32_t kStageBytes = kStageElems * sizeof(float);
  static constexpr uint32_t kElemsPerThread =
      kStageElems / kBlockSize;  // per stage
  static constexpr uint32_t kGroups =
      kElemsPerThread / 4;  // groups of 4 elements
  static_assert(kTopK == 512 || kTopK == 1024 || kTopK == 2048);
  static_assert(kElemsPerThread % 4 == 0 &&
                kElemsPerThread <= 16);  // one 16-bit mask per stage
  static_assert(kTopK % (4 * kBlockSize) == 0 || kTopK % 4 == 0);
  static_assert(kMaxCands < 0x10000);  // every counter packs in 16 bits
  static_assert(kTieCap <= kBlockSize);

  struct Smem {
    union {
      alignas(8) uint32_t hist[kNumBins / 2];  // 16-bit counters, bin 2w in the
                                               // low half of word w
      struct {
        int32_t ids[kTopK];         // selected positions in the row
        uint64_t tie_key[kTieCap];  // (order key of the score) << 32 | ~id:
                                    // higher key = taken first
      } sel;
    };
    uint32_t warp_sums[kNumWarps];
    uint32_t pivot_bin;
    uint32_t count_above;
    uint32_t emit_counter;  // packed (ties << 16 | above-pivot)
    float row_min;
    alignas(8) uint64_t full_bar[kNumStages];
    union alignas(128) {
      float data[kNumStages][kStageElems];
      struct {  // radix_ties scratch: the stage buffers are dead after pass 2
        uint32_t hist[2][256];
        uint32_t match[3];  // bin, count above it, count in it
        uint32_t counter, eq_counter;
      } radix;
    };
  };
  static constexpr uint32_t kSmemBytes = sizeof(Smem);
  // the launch-bounds occupancy target: whatever the shared memory allows on a
  // 228 KB SM
  static constexpr uint32_t kOccupancy =
      std::min({228u * 1024u / (kSmemBytes + 1024u), 2048u / kBlockSize,
                65536u / 64u / kBlockSize});
  static_assert(kOccupancy >= 2);

  Smem& smem;
  const uint32_t tx, warp_id, lane_id;
  const float* row;       // scores of this row
  uint32_t n;             // candidates, at most kMaxCands
  float thr;              // T; the row minimum on T = -inf rows
  uint32_t hi_x2, lo_x2;  // pass 2: fp16 codes of the pivot bin's upper and
                          // lower edges, twice

  // ---- per-element helpers
  // -----------------------------------------------------------------

  // Packed fp16 codes of (score - T) for four scores: lo half = element 0 / 2,
  // hi half = 1 / 3.
  FMT_DEVICE void codes_of(const device::AlignedVector<float, 4>& v,
                           uint32_t& c01, uint32_t& c23) const {
    // y >= 0 is the slab contract (every candidate is >= T), so
    // round-toward-zero is round-toward-minus-infinity here; a -0 code is
    // masked by the users
    c01 = device::ptx::cvt_rz_f16x2(v[1] - thr, v[0] - thr);
    c23 = device::ptx::cvt_rz_f16x2(v[3] - thr, v[2] - thr);
  }

  // One hit bit per element of a 4-group from the two per-half masks of
  // `set_ge_f16x2`.
  FMT_DEVICE static uint32_t pack_hits(uint32_t m01, uint32_t m23) {
    // one flag byte per element (0xFF / 0x00), then a signed dot product turns
    // them into 4 bits
    const auto flags = __byte_perm(m01, m23, 0x7531);
    return static_cast<uint32_t>(__dp4a(static_cast<int>(flags),
                                        static_cast<int>(0xF8FCFEFFu),
                                        0));  // -1,-2,-4,-8
  }

  // Valid elements (0..4) of the 4-group starting at stage-local position
  // `pos`.
  FMT_DEVICE static uint32_t valid_in_group(uint32_t n_s, uint32_t pos) {
    return pos >= n_s ? 0u : min(n_s - pos, 4u);
  }

  // ---- stages
  // ------------------------------------------------------------------------------
  // Stage s holds row positions s * kStageElems + [0, stage_len(s)). In a
  // buffer, thread `tx` owns `kGroups` groups of 4 elements: group g is the 16
  // B vector g * kBlockSize + tx.

  FMT_DEVICE uint32_t stage_len(uint32_t s) const {
    return min(n - s * kStageElems, kStageElems);
  }

  // Bulk-copy stage `s` into buffer `buf`. Issuer lane only.
  FMT_DEVICE void load_stage(uint32_t buf, uint32_t s) {
    const uint32_t bytes =
        (stage_len(s) * 4 + 15) / 16 * 16;  // the row has room past `count`
    device::ptx::mbar_arrive_expect_tx(&smem.full_bar[buf], bytes);
    device::ptx::cp_async_bulk_g2s(smem.data[buf], row + s * kStageElems, bytes,
                                   &smem.full_bar[buf]);
  }

  // Pass 1 over stage `s` resident in buffer `buf`: count it into the
  // histogram.
  FMT_DEVICE void hist_stage(uint32_t buf, uint32_t s) {
    const uint32_t n_s = stage_len(s);
#pragma unroll
    for (uint32_t g = 0; g < kGroups; ++g) {
      const uint32_t pos = 4 * (g * kBlockSize + tx);
      if (pos >= n_s) break;
      device::AlignedVector<float, 4> v;
      v.load(smem.data[buf], g * kBlockSize + tx);
      uint32_t c01, c23;
      codes_of(v, c01, c23);
      // fp16 code >> 3 (12 bits), masked: the sign bit of a -0 code must not
      // escape
      const uint32_t bin[4] = {(c01 >> 3) & 0xFFFu, (c01 >> 19) & 0xFFFu,
                               (c23 >> 3) & 0xFFFu, (c23 >> 19) & 0xFFFu};
      const uint32_t nv = valid_in_group(n_s, pos);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        if (i < nv)
          atomicAdd(smem.hist + (bin[i] >> 1), (bin[i] & 1u) ? 0x10000u : 1u);
      }
    }
  }

  // Pass 2 over stage `s` resident in buffer `buf`: one 16-bit hit mask per
  // thread per class (bit 4g + i: element i of group g) -> packed warp scan ->
  // one shared atomic per warp (above-pivot | ties << 16); slots land in
  // arrival order.
  FMT_DEVICE void emit_stage(uint32_t buf, uint32_t s) {
    const uint32_t n_s = stage_len(s);
    const float* data = smem.data[buf];
    uint32_t gt_mask = 0, ge_mask = 0;
#pragma unroll
    for (uint32_t g = 0; g < kGroups; ++g) {
      const uint32_t pos = 4 * (g * kBlockSize + tx);
      if (pos >= n_s) break;
      device::AlignedVector<float, 4> v;
      v.load(data, g * kBlockSize + tx);
      uint32_t c01, c23;
      codes_of(v, c01, c23);
      const uint32_t gt = pack_hits(device::ptx::set_ge_f16x2(c01, hi_x2),
                                    device::ptx::set_ge_f16x2(c23, hi_x2));
      const uint32_t ge = pack_hits(device::ptx::set_ge_f16x2(c01, lo_x2),
                                    device::ptx::set_ge_f16x2(c23, lo_x2));
      const uint32_t valid = (1u << valid_in_group(n_s, pos)) - 1;
      gt_mask |= (gt & valid) << (4 * g);
      ge_mask |= (ge & valid) << (4 * g);
    }
    uint32_t eq_mask = ge_mask & ~gt_mask;
    const uint32_t before = topk_cands::warp_reserve(
        __popc(gt_mask) | __popc(eq_mask) << 16, &smem.emit_counter, lane_id);
    uint32_t gt_slot = before & 0xFFFFu;
    uint32_t eq_slot = before >> 16;
    // above the pivot (dense): predicated stores at compile-time positions, a
    // one-add running slot
#pragma unroll
    for (uint32_t e = 0; e < kElemsPerThread; ++e) {
      const bool hit = (gt_mask >> e) & 1u;
      if (hit)
        smem.sel.ids[gt_slot] =
            s * kStageElems + 4 * ((e / 4) * kBlockSize + tx) + e % 4;
      gt_slot += hit ? 1u : 0u;
    }
    // the pivot bin (sparse)
    for (; eq_mask != 0; eq_mask &= eq_mask - 1, ++eq_slot) {
      const uint32_t e = __ffs(eq_mask) - 1;
      const uint32_t i = 4 * ((e / 4) * kBlockSize + tx) + e % 4;
      const uint32_t id = s * kStageElems + i;
      // order-preserving key of the fp32 score: negatives below positives, -0
      // below +0
      const uint32_t bits = __float_as_uint(data[i]);
      const uint32_t key =
          bits ^ (static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) |
                  0x80000000u);
      if (eq_slot < kTieCap)
        smem.sel.tie_key[eq_slot] = uint64_t{key} << 32 | ~id;
    }
  }

  // ---- selection
  // ---------------------------------------------------------------------------

  // Prefix from the top over the 4096-bin histogram of packed 16-bit counters:
  // lane L of warp w sums chunks j * 32 + L of the warp's range (8 bins each,
  // every LDS.128 of a warp contiguous), one packed shuffle scan gives the lane
  // suffixes, one barrier exchanges the warp totals, and the lane whose chunk
  // holds rank k re-reads its 8 bins. Publishes `pivot_bin` and `count_above`;
  // whole CTA (two barriers).
  FMT_DEVICE void find_pivot() {
    using vec_t = device::AlignedVector<uint32_t, 4>;
    static_assert(kNumWarps == 8 || kNumWarps == 16);
    constexpr uint32_t kVecsPerWarp = 4096 / 8 / kNumWarps;  // 64 or 32
    constexpr uint32_t kChunks = kVecsPerWarp / 32;          // per lane: 2 or 1
    // the four words add without carries between the halves (every count is
    // below 2^16)
    const auto chunk_sum = [](const vec_t& h) {
      const uint32_t s = h[0] + h[1] + h[2] + h[3];
      return (s & 0xFFFFu) + (s >> 16);
    };
    uint32_t local[kChunks], excl[kChunks], total[kChunks];
#pragma unroll
    for (uint32_t j = 0; j < kChunks; ++j) {
      vec_t h;
      h.load(smem.hist, warp_id * kVecsPerWarp + j * 32 + lane_id);
      local[j] = chunk_sum(h);
    }
    // all partial sums are < 2^16 (one row), so two chunks scan in one word
    const uint32_t packed =
        local[0] | (kChunks > 1 ? local[kChunks - 1] << 16 : 0u);
    const uint32_t inc = topk_cands::suffix_sum(packed);  // lanes >= L
    const uint32_t tot = __shfl_sync(device::warp::kFullMask, inc, 0);
    excl[0] = (inc - packed) & 0xFFFFu, total[0] = tot & 0xFFFFu;
    if constexpr (kChunks > 1)
      excl[1] = (inc - packed) >> 16, total[1] = tot >> 16;
    uint32_t warp_total = 0;
#pragma unroll
    for (uint32_t j = 0; j < kChunks; ++j) {
      warp_total += total[j];
    }
    if (lane_id == 0) smem.warp_sums[warp_id] = warp_total;
    __syncthreads();
    uint32_t above = 0;  // everything in the warps above this one
#pragma unroll
    for (uint32_t w = 0; w < kNumWarps; ++w) {
      if (w > warp_id) above += smem.warp_sums[w];
    }
    // chunk order inside the warp: j descending, then lanes descending; bins
    // inside a chunk descend from the high half of word 3 to the low half of
    // word 0
#pragma unroll
    for (int32_t j = kChunks - 1; j >= 0; --j) {
      const uint32_t a = above + excl[j];
      if (a < kTopK && kTopK <= a + local[j]) {
        vec_t h;
        const uint32_t vec = warp_id * kVecsPerWarp + j * 32 + lane_id;
        h.load(smem.hist, vec);
        uint32_t suffix = a;
#pragma unroll
        for (int32_t b = 7; b >= 0; --b) {
          const auto cnt = (b & 1) ? h[b / 2] >> 16 : h[b / 2] & 0xFFFFu;
          if (suffix < kTopK && kTopK <= suffix + cnt) {
            smem.pivot_bin = 8 * vec + static_cast<uint32_t>(b);
            smem.count_above = suffix;
          }
          suffix += cnt;
        }
      }
      above += total[j];
    }
    __syncthreads();
  }

  // Place the `quota` highest of the first `num` tie keys (num <= 32 * kSlots)
  // at `ids[rank]`, rank = number of higher keys. Lane L holds keys L and L +
  // 32; warp w ranks targets w, w + kNumWarps, ... with one ballot per held
  // key. Padding keys are 0, below every real key.
  template <uint32_t kSlots>
  FMT_DEVICE void rank_ties(int32_t* ids, uint32_t num, uint32_t quota) {
    uint64_t held[kSlots];
#pragma unroll
    for (uint32_t s = 0; s < kSlots; ++s) {
      const uint32_t j = s * device::kWarpThreads + lane_id;
      held[s] = j < num ? smem.sel.tie_key[j] : 0;
    }
    constexpr uint32_t kTargets =
        device::div_ceil(kSlots * device::kWarpThreads, kNumWarps);
#pragma unroll
    for (uint32_t i = 0; i < kTargets; ++i) {
      const uint32_t t = i * kNumWarps + warp_id;
      if (t < num) {  // warp-uniform
        const uint64_t key = smem.sel.tie_key[t];
        uint32_t rank = 0;
#pragma unroll
        for (uint32_t s = 0; s < kSlots; ++s) {
          rank += __popc(__ballot_sync(device::warp::kFullMask, held[s] > key));
        }
        if (lane_id == 0 && rank < quota)
          ids[rank] = static_cast<int32_t>(~static_cast<uint32_t>(key));
      }
    }
  }

  // Place the `quota` highest of `num` tie keys (num <= kTieCap: one per
  // thread) at `ids`, any order and any copies of the k-th value: a CTA radix
  // select over the order key, 4 rounds of 8 bits; the copies of the k-th value
  // then claim the remaining slots. Whole CTA.
  FMT_DEVICE void radix_ties(int32_t* ids, uint32_t num, uint32_t quota) {
    auto& r = smem.radix;
    const uint64_t tie = tx < num ? smem.sel.tie_key[tx] : 0;
    const uint32_t key = static_cast<uint32_t>(tie >> 32);
    bool active = tx < num;
    uint32_t remain = quota, total = num;
    if (tx < 256) r.hist[0][tx] = 0;
    if (tx == 0) r.counter = 0, r.eq_counter = 0;
    __syncthreads();
#pragma unroll
    for (uint32_t round = 0; round < 4; ++round) {
      const uint32_t shift = 24 - 8 * round;
      uint32_t* hist = r.hist[round % 2];
      if (active) atomicAdd(&hist[(key >> shift) & 0xFFu], 1u);
      if (tx < 256) r.hist[(round + 1) % 2][tx] = 0;
      __syncthreads();
      // bin tx: inclusive prefix from bin 0, so `total - prefix` elements sit
      // in higher bins
      const uint32_t cnt = tx < 256 ? hist[tx] : 0;
      const uint32_t inc = device::warp::inclusive_sum(cnt, lane_id);
      if (lane_id == device::kWarpThreads - 1) smem.warp_sums[warp_id] = inc;
      __syncthreads();
      if (tx < 256) {
        uint32_t prefix = inc;
        for (uint32_t w = 0; w < warp_id; ++w) {
          prefix += smem.warp_sums[w];
        }
        const uint32_t above = total - prefix;
        if (above < remain && above + cnt >= remain)
          r.match[0] = tx, r.match[1] = above, r.match[2] = cnt;
      }
      __syncthreads();
      const uint32_t bin = r.match[0];
      remain -= r.match[1];
      total = r.match[2];
      if (active) {
        const uint32_t b = (key >> shift) & 0xFFu;
        if (b > bin)
          ids[atomicAdd(&r.counter, 1u)] =
              static_cast<int32_t>(~static_cast<uint32_t>(tie));
        active = b == bin;
      }
    }
    // the survivors hold the k-th value: the first `remain` to claim a slot are
    // taken
    if (active) {
      const uint32_t slot = atomicAdd(&r.eq_counter, 1u);
      if (slot < remain)
        ids[quota - remain + slot] =
            static_cast<int32_t>(~static_cast<uint32_t>(tie));
    }
  }

  // ---- the kernel body
  // ---------------------------------------------------------------------

  FMT_DEVICE void run(const TopKCandidatesFP32Params& params) {
    using namespace device;
    const auto bx = blockIdx.x;
    row = params.scores + int64_t(bx) * params.cap;

    // ---- prologue: warp 0 owns the barriers and the bulk copies, the second
    // warpgroup zeroes the histogram ----
    if (warp_id == 0) {
      if (lane_id == 0) {  // one lane keeps init, fence and every bulk copy in
                           // program order
#pragma unroll
        for (uint32_t s = 0; s < kNumStages; ++s) {
          ptx::mbar_init(&smem.full_bar[s], 1);
        }
        ptx::fence_proxy_async_shared();
        ptx::mbar_arrive_expect_tx(&smem.full_bar[0], kStageBytes);
        cudaGridDependencySynchronize();
        // stage 0 needs no count: the slab row always holds a full stage
        ptx::cp_async_bulk_g2s(smem.data[0], row, kStageBytes,
                               &smem.full_bar[0]);
      }
    } else if (warp_id == 1) {
      if (lane_id == 0) smem.emit_counter = 0;
    } else if (warp_id >= 4) {
      // 8-byte stores: STS.128 does not reach the full shared-memory store
      // bandwidth
      AlignedVector<uint32_t, 2> zeros;
      zeros.fill(0);
      static_loop<sizeof(smem.hist) / 8, kBlockSize - 128>(
          tx - 128, [&](uint32_t i) { zeros.store(smem.hist, i); });
    }
    // every thread loads the per-row scalars itself; the loads overlap the
    // barrier below
    cudaGridDependencySynchronize();
    const uint32_t count = static_cast<uint32_t>(params.count[bx]);
    const bool needs_rescan = count > params.cap;
    n = min(count, params.cap);
    thr = params.thr[bx];
    __syncthreads();

    const auto indices_row = params.indices + int64_t(bx) * params.cap;
    const auto out = params.out + int64_t(bx) * kTopK;
    if (needs_rescan || n <= kTopK) {
      // Truncated slabs are repaired from Q/K by the following GPU launch.
      static_loop<kTopK, kBlockSize>(tx, [&](uint32_t t) {
        out[t] = !needs_rescan && t < n ? indices_row[t] : -1;
      });
      if (tx == 0) {
        ptx::mbar_wait_parity(&smem.full_bar[0],
                              0);  // no CTA exits with a copy in flight
      }
      return cudaTriggerProgrammaticLaunchCompletion();
    }
    const auto num_stages = div_ceil(n, kStageElems);
    __builtin_assume(num_stages == 1 || num_stages == 2 || num_stages == 3);
    // Two stages stay resident. The fallback takes longer rows (the third stage
    // streams into buffer 0 behind a CTA barrier once per pass) and T = -inf
    // rows (short rows that keep every key; T becomes the row minimum).
    const bool resident = n <= kResidentElems && thr != -INFINITY;
    const bool is_issuer = warp_id == 0 && lane_id == 0;
    if (is_issuer) {
      if (num_stages > 1) load_stage(1, 1);
      // the end gathers from the index slab: warm it in L2 meanwhile
      ptx::cp_async_bulk_prefetch_l2(indices_row, (n * 4 + 15) / 16 * 16);
    }

    // ---- pass 1: 4096-bin histogram of the fp16 code's high 12 bits ----
    if (resident) {
#pragma unroll
      for (uint32_t s = 0; s < kNumStages; ++s) {
        if (s >= num_stages) break;
        ptx::mbar_wait_parity(&smem.full_bar[s], 0);
        hist_stage(s, s);
      }
    } else {
      if (thr == -INFINITY) {  // the row minimum, read from global memory
        float m = INFINITY;
        for (uint32_t i = 4 * tx; i < n; i += 4 * kBlockSize) {
          AlignedVector<float, 4> v;
          v.load(row, i / 4);
#pragma unroll
          for (uint32_t j = 0; j < 4; ++j) {
            if (i + j < n) m = fminf(m, v[j]);
          }
        }
        m = warp::reduce_min(m);
        if (lane_id == 0) smem.warp_sums[warp_id] = __float_as_uint(m);
        __syncthreads();
        if (warp_id == 0) {
          m = warp::reduce_min(lane_id < kNumWarps
                                   ? __uint_as_float(smem.warp_sums[lane_id])
                                   : INFINITY);
          if (lane_id == 0) smem.row_min = m;
        }
        __syncthreads();
        thr = smem.row_min;
      }
      ptx::mbar_wait_parity(&smem.full_bar[0], 0);
      hist_stage(0, 0);
      if (num_stages == 3) {
        __syncthreads();  // everyone is done with buffer 0
        if (is_issuer) load_stage(0, 2);
      }
      if (num_stages > 1) {
        ptx::mbar_wait_parity(&smem.full_bar[1], 0);
        hist_stage(1, 1);
      }
      if (num_stages == 3) {
        ptx::mbar_wait_parity(&smem.full_bar[0], 1);
        hist_stage(0, 2);
      }
    }
    __syncthreads();

    find_pivot();
    const uint32_t pivot = smem.pivot_bin;
    const uint32_t count_above = smem.count_above;
    const uint32_t quota =
        kTopK - count_above;  // taken from the pivot bin, >= 1

    // ---- pass 2: emit the bins above the pivot, collect the pivot bin into
    // the tie buffer ---- bin > pivot <=> code >= (pivot + 1) << 3; the codes
    // compare as small positive fp16 values
    const uint32_t code_hi = (pivot + 1) << 3, code_lo = pivot << 3;
    hi_x2 = code_hi << 16 | code_hi, lo_x2 = code_lo << 16 | code_lo;
    if (resident) {
#pragma unroll
      for (uint32_t s = 0; s < kNumStages; ++s) {
        if (s >= num_stages) break;
        emit_stage(s, s);
      }
    } else {
      // buffer 0 holds stage 2 after pass 1 of a 3-stage row
      emit_stage(0, num_stages == 3 ? 2 : 0);
      if (num_stages == 3) {
        __syncthreads();  // everyone is done with buffer 0
        if (is_issuer) load_stage(0, 0);
      }
      if (num_stages > 1) emit_stage(1, 1);
      if (num_stages == 3) {
        ptx::mbar_wait_parity(&smem.full_bar[0],
                              0);  // the third fill of buffer 0
        emit_stage(0, 0);
      }
    }
    cudaTriggerProgrammaticLaunchCompletion();
    __syncthreads();

    if ((smem.emit_counter >> 16) > kTieCap) {
      device::litetopk_recovery::exact_topk<TopKCandidatesFP32Impl>(
          row, indices_row, n, kTopK, out,
          *reinterpret_cast<device::litetopk_recovery::Scratch*>(&smem));
      return;
    }

    // ---- rank the tie buffer by exact score ----
    // Wider pivot bins have already taken the exact recovery path above.
    const uint32_t num_ties = min(smem.emit_counter >> 16, kTieCap);
    const uint32_t taken = min(quota, num_ties);
    // For valid candidate rows this equals kTopK.
    const uint32_t first_hole = count_above + taken;
    int32_t* const tie_ids = smem.sel.ids + count_above;
    for (uint32_t i = taken + tx; i < quota; i += kBlockSize) {
      tie_ids[i] = 0;
    }
    if (num_ties == taken) {  // every tie is taken
      if (tx < num_ties)
        tie_ids[tx] =
            static_cast<int32_t>(~static_cast<uint32_t>(smem.sel.tie_key[tx]));
    } else if (num_ties <= kWarpThreads) {
      rank_ties<1>(tie_ids, num_ties, taken);
    } else if (num_ties <= 2 * kWarpThreads) {
      rank_ties<2>(tie_ids, num_ties, taken);
    } else {
      radix_ties(tie_ids, num_ties, taken);
    }
    __syncthreads();

    // ---- gather: quads of selected positions -> key indices -> out ----
    constexpr uint32_t kQuads = kTopK / 4;
    constexpr uint32_t kPerThread = device::div_ceil(kQuads, kBlockSize);
    AlignedVector<int32_t, 4> res[kPerThread];
#pragma unroll
    for (uint32_t r = 0; r < kPerThread;
         ++r) {  // every load in flight before the first store
      const uint32_t q4 = r * kBlockSize + tx;
      if (q4 < kQuads) {
        AlignedVector<int32_t, 4> ids;
        ids.load(smem.sel.ids, q4);
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          // unsigned compare: an id outside the row (negative included) never
          // reaches the gather
          const auto id = static_cast<uint32_t>(ids[i]);
          res[r][i] = id < n ? indices_row[id] : -1;
        }
      }
    }
#pragma unroll
    for (uint32_t r = 0; r < kPerThread; ++r) {
      const uint32_t q4 = r * kBlockSize + tx;
      if (first_hole < kTopK) {
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          if (4 * q4 + i >= first_hole) res[r][i] = -1;
        }
      }
      if (q4 < kQuads) res[r].store(out, q4);
    }
  }
};

template <typename Impl>
__global__ __launch_bounds__(Impl::kBlockSize, Impl::kOccupancy)  //
    void topk_candidates_fp32_kernel(
        const __grid_constant__ TopKCandidatesFP32Params params) {
  alignas(128) extern __shared__ uint8_t smem_raw[];
  const uint32_t tx = threadIdx.x;
  Impl impl{
      .smem = *reinterpret_cast<typename Impl::Smem*>(smem_raw),
      .tx = tx,
      .warp_id = device::warp::reduce_max(tx / device::kWarpThreads),
      .lane_id = device::warp::get_lane_id(),
      .row = nullptr,
      .n = 0,
      .thr = 0.0f,
      .hi_x2 = 0,
      .lo_x2 = 0,
  };
  impl.run(params);
}

}  // namespace fused_mqa_topk
