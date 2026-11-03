// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Per-row top-K selection of the persistent kernel's candidates, one warp per
// row (K = topk, the row's top-k: 512, 1024 or 2048, a runtime value).
//
// select_staged_row: bracket the K-th largest fp24 code with sampled codes,
// place the codes above the bracket directly, and resolve the bracket with
// histogram levels on register-resident codes.
//
// select_row: the fallback of select_staged_row when its sampled bracket
// misses. Candidate codes (higher code = higher score, staged once in shared
// memory) span a narrow range (gate .. row max), so each level histograms the
// range-relative code into at most kBins bins: collisions are rare, plain
// shared atomics suffice and each lane keeps kUnroll independent elements in
// flight. Levels repeat inside the boundary bin until it is one code wide;
// codes above it are emitted, then ties until K.
#pragma once
#include <cstdint>

#include "candidates.cuh"
#include "common.cuh"
#include "constants.h"

namespace fused_mqa_topk::selection {
constexpr uint32_t kBinsLog2 = 10, kBins = 1u << kBinsLog2;
// Elements each lane keeps in flight.
constexpr uint32_t kUnroll = 8;

// Bin (descending) where the cumulative count first reaches `rank`;
// `above` = elements in higher bins. Lane l owns Bins / 32 bins, descending
// from bin Bins - 1 - l * Bins / 32.
template <uint32_t Bins>
__device__ __forceinline__ uint32_t find_bin_n(const uint32_t* hist,
                                               uint32_t rank, uint32_t& above) {
  const uint32_t lane = threadIdx.x & 31;
  constexpr uint32_t kPerLane = Bins / 32;
  uint32_t sum = 0;
#pragma unroll
  for (uint32_t j = 0; j < kPerLane; ++j)
    sum += hist[Bins - 1 - lane * kPerLane - j];
  const uint32_t inclusive = warp_inclusive_scan(sum, lane);
  const uint32_t before = inclusive - sum;
  const uint32_t winner =
      __ffs(__ballot_sync(kFullMask, before < rank && inclusive >= rank)) - 1;
  uint32_t chosen = 0, acc = before;
  if (lane == winner) {
    for (uint32_t j = 0; j < kPerLane; ++j) {
      const uint32_t bin = Bins - 1 - lane * kPerLane - j, h = hist[bin];
      if (acc + h >= rank) {
        chosen = bin;
        break;
      }
      acc += h;
    }
  }
  above = __shfl_sync(kFullMask, acc, winner);
  return __shfl_sync(kFullMask, chosen, winner);
}

// Requires count >= topk (callers recover rows with fewer candidates).
__device__ inline void select_row(uint32_t* codes, uint32_t count,
                                  const uint16_t* values, const int32_t* packed,
                                  int32_t* out, uint32_t topk, uint32_t* hist) {
  const uint32_t lane = threadIdx.x & 31;
  uint32_t lo = 0xffffffffu, hi = 0;
  uint32_t j = lane;
  for (; j + 32 * (kUnroll - 1) < count; j += 32 * kUnroll) {
    uint16_t v[kUnroll];
    int32_t p[kUnroll];
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      v[u] = values[j + 32 * u];
      p[u] = packed[j + 32 * u];
    }
#pragma unroll
    for (uint32_t u = 0; u < kUnroll; ++u) {
      const uint32_t c = candidate_load_score_code(v[u], p[u]);
      codes[j + 32 * u] = c;
      lo = min(lo, c);
      hi = max(hi, c);
    }
  }
  for (; j < count; j += 32) {
    const uint32_t c = candidate_load_score_code(values[j], packed[j]);
    codes[j] = c;
    lo = min(lo, c);
    hi = max(hi, c);
  }
  lo = __reduce_min_sync(kFullMask, lo);
  hi = __reduce_max_sync(kFullMask, hi);
  __syncwarp();

  // Current bucket: codes in [lo, lo + width). rank = elements still needed
  // from it.
  uint32_t width = hi - lo + 1, rank = topk;
#pragma unroll 1
  while (width > 1) {
    const uint32_t spread = (width - 1) >> kBinsLog2;
    const uint32_t shift = spread ? 32 - __clz(spread) : 0;
    for (uint32_t b = lane; b < kBins; b += 32) hist[b] = 0;
    __syncwarp();
    const uint32_t top = lo + width - 1;
    uint32_t i = lane;
    for (; i + 32 * (kUnroll - 1) < count; i += 32 * kUnroll) {
      uint32_t c[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) c[u] = codes[i + 32 * u];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u)
        if (c[u] >= lo && c[u] <= top)
          atomicAdd(hist + ((c[u] - lo) >> shift), 1u);
    }
    for (; i < count; i += 32) {
      const uint32_t c = codes[i];
      if (c >= lo && c <= top) atomicAdd(hist + ((c - lo) >> shift), 1u);
    }
    __syncwarp();
    uint32_t above;
    const uint32_t chosen = find_bin_n<kBins>(hist, rank, above);
    rank -= above;
    lo += chosen << shift;
    width = min(1u << shift, top - lo + 1);
    __syncwarp();
  }

  // lo is now the K-th largest code: emit codes above it, then ties until K.
  const uint32_t threshold = lo, lt = (1u << lane) - 1u;
  uint32_t pos = 0;
#pragma unroll 1
  for (uint32_t tie = 0; tie < 2; ++tie) {
    for (uint32_t j0 = 0; j0 < count && pos < topk; j0 += 32 * kUnroll) {
      bool pass[kUnroll];
      int32_t p[kUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        const uint32_t e = j0 + 32 * u + lane;
        const uint32_t c = e < count ? codes[e] : 0;
        pass[u] = e < count && (tie ? c == threshold : c > threshold);
        p[u] = pass[u] ? packed[e] : 0;
      }
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        const uint32_t mask = __ballot_sync(kFullMask, pass[u]),
                       at = pos + __popc(mask & lt);
        if (pass[u] && at < topk) out[at] = candidate_decode_index(p[u]);
        pos += __popc(mask);
      }
    }
  }
}

// Register samples per lane for the sampled bracket of select_staged_row.
constexpr uint32_t kSamplesPerLane = 16;
// Largest bracket (elements between the two sampled bounds) that
// select_staged_row resolves in registers; wider brackets fall back to
// select_row.
constexpr uint32_t kMidCap = 1024;
// Bins of the histogram levels that resolve the exact threshold inside a
// bracket (select_staged_row).
constexpr uint32_t kMidBins = 512;

// Bracket element ids are stored as uint16 (midbuf).
static_assert(kCandidatesPerRow <= 65536, "candidate ids must fit 16 bits");
// Per-warp scratch of select_staged_row is kBins words: radix_bracket's 768,
// or the midbuf (kMidCap uint16) plus the kMidBins-bin histogram.
static_assert(768 <= kBins && kMidCap / 2 + kMidBins <= kBins,
              "selection scratch exceeds kBins words");

// r-th largest (1-based) of the kNumSamples * 32 register samples (kNumSamples
// per lane) for two ranks at once
// (r_hi <= r_lo), via a 256-bin hi8 level and two 256-bin lo8 levels. h8: 768
// words of scratch.
template <uint32_t kNumSamples = kSamplesPerLane>
__device__ __forceinline__ void radix_bracket(
    const uint32_t (&smp)[kNumSamples], uint32_t r_hi, uint32_t r_lo,
    uint32_t* h8, uint32_t& h_hi, uint32_t& h_lo) {
  const uint32_t lane = threadIdx.x & 31;
#pragma unroll
  for (uint32_t b = lane; b < 192; b += 32)
    reinterpret_cast<uint4*>(h8)[b] = make_uint4(0, 0, 0, 0);
  __syncwarp();
#pragma unroll
  for (uint32_t u = 0; u < kNumSamples; ++u) atomicAdd(h8 + (smp[u] >> 8), 1u);
  __syncwarp();
  uint32_t ab_hi, ab_lo;
  const uint32_t b_hi = find_bin_n<256>(h8, r_hi, ab_hi);
  const uint32_t b_lo = find_bin_n<256>(h8, r_lo, ab_lo);
  uint32_t* const lh = h8 + 256;
  uint32_t* const ll = h8 + 512;
#pragma unroll
  for (uint32_t u = 0; u < kNumSamples; ++u) {
    const uint32_t b = smp[u] >> 8;
    if (b == b_hi) atomicAdd(lh + (smp[u] & 255u), 1u);
    if (b == b_lo) atomicAdd(ll + (smp[u] & 255u), 1u);
  }
  __syncwarp();
  uint32_t d0, d1;
  h_hi = b_hi << 8 | find_bin_n<256>(lh, r_hi - ab_hi, d0);
  h_lo = b_lo << 8 | find_bin_n<256>(ll, r_lo - ab_lo, d1);
  __syncwarp();  // h8 is reused as the output stage
}

// Brackets the topk-th largest of the `count` staged codes between the hi16
// codes of sample ranks expect -+ margin, where the kPerLane * 32 samples sit
// at evenly spaced quantiles and expect is the topk-th largest's expected rank
// among them.
template <uint32_t kPerLane>
__device__ __forceinline__ void sampled_bracket(const uint32_t* codes,
                                                uint32_t count, uint32_t topk,
                                                uint32_t margin, uint32_t* hist,
                                                uint32_t& h_hi,
                                                uint32_t& h_lo) {
  constexpr uint32_t kSamples = kPerLane * 32;
  const uint32_t lane = threadIdx.x & 31;
  uint32_t smp[kPerLane];
#pragma unroll
  for (uint32_t u = 0; u < kPerLane; ++u)
    smp[u] = codes[((u * 32 + lane) * 2 + 1) * count / (2 * kSamples)] >> 8;
  const uint32_t expect = (topk * kSamples + count / 2) / count;
  const uint32_t r_hi = expect > margin ? expect - margin : 1,
                 r_lo = min(expect + margin, kSamples);
  radix_bracket<kPerLane>(smp, r_hi, r_lo, hist, h_hi, h_lo);
}

// find_bin_n with each lane's kPer contiguous bins loaded as 16-byte vectors
// and searched in registers (needs a 16-byte aligned histogram and kPer % 4 ==
// 0; find_bin_n reads the bins one at a time and has neither requirement).
template <uint32_t Bins>
__device__ __forceinline__ uint32_t find_bin_v(const uint32_t* hist,
                                               uint32_t rank, uint32_t& above) {
  const uint32_t lane = threadIdx.x & 31;
  constexpr uint32_t kPer = Bins / 32, kVec = kPer / 4;
  static_assert(kPer % 4 == 0);
  // lane l owns bins [Bins - (l + 1) * kPer, Bins - l * kPer), highest first.
  const uint4* hv =
      reinterpret_cast<const uint4*>(hist + Bins - (lane + 1) * kPer);
  uint32_t h[kPer];
#pragma unroll
  for (uint32_t v = 0; v < kVec; ++v) {
    const uint4 x = hv[v];
    h[4 * v] = x.x;
    h[4 * v + 1] = x.y;
    h[4 * v + 2] = x.z;
    h[4 * v + 3] = x.w;
  }
  uint32_t sum = 0;
#pragma unroll
  for (uint32_t j = 0; j < kPer; ++j) sum += h[j];
  const uint32_t inclusive = warp_inclusive_scan(sum, lane);
  const uint32_t before = inclusive - sum;
  const uint32_t winner =
      __ffs(__ballot_sync(kFullMask, before < rank && inclusive >= rank)) - 1;
  // Descending walk inside the lane: bin index Bins - lane * kPer - 1 - j  <->
  // h[kPer - 1 - j].
  uint32_t acc = before, chosen = 0, found = 0, acc_at = before;
#pragma unroll
  for (uint32_t j = 0; j < kPer; ++j) {
    const uint32_t x = h[kPer - 1 - j];
    const bool hit = !found && acc + x >= rank;
    if (hit) {
      chosen = Bins - lane * kPer - 1 - j;
      acc_at = acc;
      found = 1;
    }
    acc += x;
  }
  above = __shfl_sync(kFullMask, acc_at, winner);
  return __shfl_sync(kFullMask, chosen, winner);
}

// One row's global-memory candidates, staged to shared memory as
// fp24 codes (codes must be 16-byte aligned, i.e. row offsets rounded up to 4).
// Register budget: ptxas compiles the whole kernel to 168 regs, so every pass
// keeps <= ~32 regs of arrays live (more spills into the scan loop).
// Passes: stage (16-byte global loads of values / packed indices, two chunks of
// 8 in flight per lane, to fp24 codes); hi16 bracket (radix levels over 512
// shared-memory samples); count above / inside the bracket (16-byte shared
// loads); collect (codes above the bracket to out, bracket element ids to
// midbuf); histogram levels on register codes; emit.
// Returns false before writing out; the caller's select_row then restages and
// runs the levels.
__device__ __noinline__ inline bool select_staged_row(
    uint32_t* __restrict__ codes, uint32_t count,
    const uint16_t* __restrict__ values, const int32_t* __restrict__ packed,
    int32_t* __restrict__ out, uint32_t topk, uint32_t* hist) {
  const uint32_t lane = threadIdx.x & 31;
  if (count <= topk) return false;
  const uint32_t chunks = (count + 7) / 8;
  constexpr uint32_t kChunksInFlight = 2;
  const uint4* v4p = reinterpret_cast<const uint4*>(values);
  const uint4* p4p = reinterpret_cast<const uint4*>(packed);
#pragma unroll 1
  for (uint32_t c0i = lane; c0i < chunks; c0i += 32 * kChunksInFlight) {
    uint4 vv[kChunksInFlight], pp0[kChunksInFlight], pp1[kChunksInFlight];
#pragma unroll
    for (uint32_t u = 0; u < kChunksInFlight; ++u) {
      const uint32_t c = min(c0i + 32 * u, chunks - 1);
      vv[u] = __ldcg(v4p + c);
      pp0[u] = __ldcg(p4p + 2 * c);
      pp1[u] = __ldcg(p4p + 2 * c + 1);
    }
#pragma unroll
    for (uint32_t u = 0; u < kChunksInFlight; ++u) {
      const uint32_t c = c0i + 32 * u;
      if (c < chunks) {
        const uint32_t vw[4] = {vv[u].x, vv[u].y, vv[u].z, vv[u].w};
        const uint32_t pw[8] = {pp0[u].x, pp0[u].y, pp0[u].z, pp0[u].w,
                                pp1[u].x, pp1[u].y, pp1[u].z, pp1[u].w};
        uint32_t cw[8];
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k)
          cw[k] = candidate_load_score_code(
              uint16_t(vw[k / 2] >> (16 * (k & 1))), int32_t(pw[k]));
        if (8 * c + 8 <= count) {
          reinterpret_cast<uint4*>(codes)[2 * c] =
              make_uint4(cw[0], cw[1], cw[2], cw[3]);
          reinterpret_cast<uint4*>(codes)[2 * c + 1] =
              make_uint4(cw[4], cw[5], cw[6], cw[7]);
        } else {
#pragma unroll
          for (uint32_t k = 0; k < 8; ++k)
            if (8 * c + k < count) codes[8 * c + k] = cw[k];
        }
      }
    }
  }
  __syncwarp();
  // The bracket is the code range between the sampled ranks expect -+ margin
  // (expect = the K-th largest's expected rank among the samples). Rows up to
  // kFewCandidates use 512 samples, +-40 ranks. Long-context rows (4-6K
  // candidates) would get a bracket of ~16% of the row that way, over kMidCap
  // for 10-20% of the rows at 512K-1M: 1024 samples, +-48 (~3 sd), give ~9%.
  constexpr uint32_t kFewCandidates = 3584, kMargin = 40, kWideMargin = 48;
  uint32_t h_hi, h_lo;
  if (count <= kFewCandidates)
    sampled_bracket<kSamplesPerLane>(codes, count, topk, kMargin, hist, h_hi,
                                     h_lo);
  else
    sampled_bracket<2 * kSamplesPerLane>(codes, count, topk, kWideMargin, hist,
                                         h_hi, h_lo);
  const uint4* c4p = reinterpret_cast<const uint4*>(codes);
  const uint32_t quads = (count + 3) / 4;
  const uint32_t lt = (1u << lane) - 1u;
  uint16_t* midbuf = reinterpret_cast<uint16_t*>(hist);
  uint32_t* h512 = hist + kMidCap / 2;
  // Collect with warp-ballot placement: passing lanes of one step write
  // consecutive slots, so the global stores of the above elements are
  // coalesced. Writes are bounded; the
  // bracket check follows (out may be partially written on failure: the
  // fallback select_row rewrites all K slots).
  uint32_t above = 0, mid = 0;
  constexpr uint32_t kQuadsInFlight = 4;
#pragma unroll 1
  for (uint32_t q0 = 0; q0 < quads; q0 += 32 * kQuadsInFlight) {
    uint4 pp[kQuadsInFlight], cc[kQuadsInFlight];
#pragma unroll
    for (uint32_t u = 0; u < kQuadsInFlight; ++u) {
      const uint32_t q = min(q0 + 32 * u + lane, quads - 1);
      pp[u] = __ldcg(p4p + q);
      cc[u] = c4p[q];
    }
#pragma unroll
    for (uint32_t u = 0; u < kQuadsInFlight; ++u) {
      const uint32_t q = q0 + 32 * u + lane;
      const uint32_t w[4] = {cc[u].x, cc[u].y, cc[u].z, cc[u].w};
      const uint32_t p[4] = {pp[u].x, pp[u].y, pp[u].z, pp[u].w};
#pragma unroll
      for (uint32_t k = 0; k < 4; ++k) {
        const uint32_t e = 4 * q + k, h = w[k] >> 8;
        const bool valid = e < count;
        const bool a = valid && h > h_hi, m = valid && !a && h >= h_lo;
        const uint32_t ma = __ballot_sync(kFullMask, a),
                       mm = __ballot_sync(kFullMask, m);
        const uint32_t at = above + __popc(ma & lt), mt = mid + __popc(mm & lt);
        if (a && at < topk) out[at] = candidate_decode_index(int32_t(p[k]));
        if (m && mt < kMidCap) midbuf[mt] = uint16_t(e);
        above += __popc(ma);
        mid += __popc(mm);
      }
    }
  }
  __syncwarp();
  if (above >= topk || above + mid < topk || mid > kMidCap) return false;
  // Bracket codes per lane (register-resident).
  constexpr uint32_t kMidPerLane = kMidCap / 32;
  uint32_t mc[kMidPerLane];
#pragma unroll
  for (uint32_t u = 0; u < kMidPerLane; ++u) {
    const uint32_t i = lane + 32 * u;
    mc[u] = 0;
    if (i < mid) mc[u] = codes[midbuf[i]];
  }
  uint32_t rank = topk - above, base = h_lo << 8, top = h_hi << 8 | 0xffu;
#pragma unroll 1
  while (top > base) {
    const uint32_t width = top - base + 1, spread = (width - 1) / kMidBins;
    const uint32_t shift = spread ? 32 - __clz(spread) : 0;
#pragma unroll
    for (uint32_t b = lane; b < kMidBins / 4; b += 32)
      reinterpret_cast<uint4*>(h512)[b] = make_uint4(0, 0, 0, 0);
    __syncwarp();
#pragma unroll
    for (uint32_t u = 0; u < kMidPerLane; ++u) {
      if (u * 32 >= mid) break;
      if (lane + 32 * u < mid && mc[u] >= base && mc[u] <= top)
        atomicAdd(h512 + ((mc[u] - base) >> shift), 1u);
    }
    __syncwarp();
    uint32_t above_bin;
    const uint32_t chosen = find_bin_v<kMidBins>(h512, rank, above_bin);
    rank -= above_bin;
    base += chosen << shift;
    top = min(base + (1u << shift) - 1, top);
    __syncwarp();
  }
  const uint32_t th = base;
  {
    // Ballot placement (coalesced stores): greater codes fill [above, K -
    // rank), ties [K - rank, K).
    uint32_t pg = above, pt = topk - rank;
#pragma unroll
    for (uint32_t h = 0; h < 2; ++h) {
      if (h * 16 * 32 >= mid) break;
      int32_t pk[16];
#pragma unroll
      for (uint32_t j = 0; j < 16; ++j) {
        const uint32_t u = h * 16 + j;
        pk[j] = (lane + 32 * u < mid && mc[u] >= th)
                    ? __ldcg(packed + midbuf[lane + 32 * u])
                    : 0;
      }
#pragma unroll
      for (uint32_t j = 0; j < 16; ++j) {
        const uint32_t u = h * 16 + j;
        if (u * 32 >= mid) break;
        const bool v = lane + 32 * u < mid;
        const bool g = v && mc[u] > th, tq = v && mc[u] == th;
        const uint32_t mg = __ballot_sync(kFullMask, g),
                       mt = __ballot_sync(kFullMask, tq);
        const uint32_t ag = pg + __popc(mg & lt), at = pt + __popc(mt & lt);
        if (g) out[ag] = candidate_decode_index(pk[j]);
        if (tq && at < topk) out[at] = candidate_decode_index(pk[j]);
        pg += __popc(mg);
        pt += __popc(mt);
      }
    }
  }
  return true;
}

}  // namespace fused_mqa_topk::selection
