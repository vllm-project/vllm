// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef COOPERATIVE_GRID_TOPK_CUH_
#define COOPERATIVE_GRID_TOPK_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace vllm::cooperative_grid {

constexpr uint32_t kThreads = 1024;
constexpr uint32_t kSamples = 2048;
// Row state (headers, candidate lists) is sized for this many rows.
constexpr uint32_t kMaxRows = 64;
// Shortest row the launcher sends here: shorter rows are faster on the
// cluster path at every row count.
constexpr uint32_t kMinRowLength = 262144;
constexpr uint32_t kBins = 2048;
// Per round: 0 survivors, 1 written above the threshold bin, 2 candidates (or
// ties written), 3 a CTA overflowed, padding to 32 words, then the histogram.
// A launch uses one round unless the cutoff was misjudged (at most kRounds).
constexpr uint32_t kRounds = 6;
constexpr uint32_t kRoundWords = 32 + kBins;
constexpr uint32_t kParityWords = kRounds * kRoundWords;
// Per row: the arrival counter (2 words, padded to 32), then two parities.
constexpr uint32_t kHeaderWords = 32 + 2 * kParityWords;
constexpr size_t kHeaderBytes = kMaxRows * kHeaderWords * 4;
constexpr uint32_t kCand = 16384;  // candidate list (key + index) per row
constexpr size_t kRowBytes = size_t{kCand} * 8;
constexpr size_t kWorkspaceBytes = kMaxRows * kRowBytes;
// Per-row state of the dynamic split (cooperative_grid_dyn.cuh): header words,
// then the row histogram; a grid-wide exit counter follows the rows.
constexpr uint32_t kDynHdr = 32;
constexpr uint32_t kDynRowWords = kDynHdr + kBins;
constexpr size_t kDynBytes = (size_t{kMaxRows} * kDynRowWords + 32) * 4;
// The caller's persistent state: the row headers, the candidate lists (shared
// by the grid path and the dynamic split), then the dynamic split's state.
// Zeroed once and kept across launches.
constexpr size_t kStateBytes = kHeaderBytes + kWorkspaceBytes + kDynBytes;
constexpr uint32_t kStage = 16384;  // survivors a CTA holds
// Dynamic shared memory: the CTA's survivor stage (key + index). kDynSamples
// (launches with 2- or 4-CTA clusters, whose cluster path needs 217 KB of
// static and dynamic shared memory) also puts the samples there instead of in
// static shared memory, so that launch fits.
constexpr size_t kDynSmem = size_t{kStage} * 8;
constexpr size_t kDynSmemSamples = kDynSmem + size_t{kSamples} * 4;
constexpr uint32_t kPre = 2;  // tiles a thread prefetches
// The row is dealt out in tiles of kTile scores (one 16-byte vector per
// thread), round-robin over the row's CTAs, so a region where the top scores
// concentrate (the latest tokens, the start of the context) spreads over many
// CTAs instead of loading one.
constexpr uint32_t kTile = 4 * kThreads;

// Order key of an FP32 score: keys compare as the scores do; NaN is largest.
__device__ __forceinline__ uint32_t okey(uint32_t b) {
  if ((b & 0x7FFFFFFFu) > 0x7F800000u) return 0xFFFFFFFFu;
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Bin of a survivor's distance d above the cutoff, increasing with d: exact
// below 64, then 64 bins an octave.
__device__ __forceinline__ uint32_t lbin(uint32_t d) {
  if (d < 64) return d;
  const uint32_t e = 31 - __clz(d);
  return 64 + (e - 6) * 64 + ((d >> (e - 6)) & 63u);
}

__device__ __forceinline__ unsigned long long ld_acquire64(
    const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long add_acqrel64(
    unsigned long long* p, unsigned long long v) {
  unsigned long long old;
  asm volatile("atom.add.acq_rel.gpu.global.u64 %0, [%1], %2;"
               : "=l"(old)
               : "l"(p), "l"(v)
               : "memory");
  return old;
}
__device__ __forceinline__ unsigned long long add_release64(
    unsigned long long* p, unsigned long long v) {
  unsigned long long old;
  asm volatile("atom.add.release.gpu.global.u64 %0, [%1], %2;"
               : "=l"(old)
               : "l"(p), "l"(v)
               : "memory");
  return old;
}

// Barrier of the row's CTAs: arrive, then wait until the counter reaches
// `target`. Every thread calls it; the CTA's prior writes are visible to the
// others after it. `last`: the arrival that completes the launch (2G) also
// moves the counter to the next generation, low word zero.
__device__ __forceinline__ void row_barrier(unsigned long long* arrivals,
                                            unsigned long long target,
                                            bool last_one,
                                            unsigned long long next_gen) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned long long old = add_release64(arrivals, 1ull);
    if (last_one && old + 1 == target)
      add_release64(arrivals, next_gen - target);
    while (ld_acquire64(arrivals) < target) __nanosleep(20);
  }
  __syncthreads();
}

// CTA-wide max of a and of b (every thread passes one each); red: 2 words.
__device__ __forceinline__ uint2 cta_max2(uint32_t a, uint32_t b,
                                          uint32_t* red) {
  if (threadIdx.x < 2) red[threadIdx.x] = 0;
  __syncthreads();
  a = __reduce_max_sync(0xFFFFFFFFu, a);
  b = __reduce_max_sync(0xFFFFFFFFu, b);
  if (threadIdx.x % 32 == 0) {
    atomicMax(&red[0], a);
    atomicMax(&red[1], b);
  }
  __syncthreads();
  const uint2 m = make_uint2(red[0], red[1]);
  __syncthreads();
  return m;
}

// All threads: in a kBins-bin histogram (2 bins a thread), the bin holding
// the need-th element in bin order (sel[0]) and the count before it
// (sel[1]), and that bin's count (sel[2]). rev: count from the last bin down.
// scan: 32 shared words.
__device__ __forceinline__ void pick_bin(const uint32_t* hist, uint32_t need,
                                         bool rev, uint32_t* sel,
                                         uint32_t* scan) {
  static_assert(kBins == 2 * kThreads);
  const uint32_t tx = threadIdx.x, lane = tx % 32, warp = tx / 32;
  const uint32_t b0 = rev ? kBins - 1 - 2 * tx : 2 * tx,
                 b1 = rev ? b0 - 1 : b0 + 1;
  const uint32_t c0 = hist[b0], c1 = hist[b1], sum = c0 + c1;
  uint32_t incl = sum;
#pragma unroll
  for (uint32_t o = 1; o < 32; o <<= 1) {
    const uint32_t v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) scan[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    const uint32_t w = scan[lane];
    uint32_t wi = w;
#pragma unroll
    for (uint32_t o = 1; o < 32; o <<= 1) {
      const uint32_t v = __shfl_up_sync(0xFFFFFFFFu, wi, o);
      if (lane >= o) wi += v;
    }
    __syncwarp();
    scan[lane] = wi - w;
  }
  __syncthreads();
  const uint32_t run = scan[warp] + incl - sum;
  if (run < need && run + c0 >= need) {
    sel[0] = b0;
    sel[1] = run;
    sel[2] = c0;
  } else if (run + c0 < need && run + sum >= need) {
    sel[0] = b1;
    sel[1] = run + c0;
    sel[2] = c1;
  }
  __syncthreads();
}

// The target-th smallest of n values (val(i) < 2^hb) and the count below it:
// 11-bit radix rounds from bit hb down, stopping once `keep` low bits are
// left (their bin's smallest value is returned). CTA-wide.
template <typename Val>
__device__ __forceinline__ uint2 select_low(Val val, uint32_t n,
                                            uint32_t target, uint32_t hb,
                                            uint32_t keep, uint32_t* hist,
                                            uint32_t* sel, uint32_t* scan) {
  const uint32_t tx = threadIdx.x;
  uint32_t prefix = 0, below = 0, done = 0;
#pragma unroll 1
  while (done + keep < hb) {
    const uint32_t take = min(11u, hb - keep - done), sh = hb - done - take;
    for (uint32_t b = tx; b < kBins; b += kThreads) hist[b] = 0;
    __syncthreads();
    for (uint32_t i = tx; i < n; i += kThreads) {
      const uint32_t d = val(i);
      if ((done ? d >> (sh + take) : 0u) == prefix)
        atomicAdd(&hist[(d >> sh) & ((1u << take) - 1)], 1u);
    }
    __syncthreads();
    pick_bin(hist, target - below, false, sel, scan);
    prefix = (prefix << take) | sel[0];
    below += sel[1];
    done += take;
  }
  return make_uint2(prefix << (hb - done), below);
}

// Slot in list[count...] for the active lanes (warp-wide).
__device__ __forceinline__ uint32_t warp_slot(bool on, uint32_t* count) {
  const uint32_t lane = threadIdx.x % 32, m = __ballot_sync(0xFFFFFFFFu, on);
  uint32_t b = 0;
  if (lane == 0 && m) b = atomicAdd(count, __popc(m));
  b = __shfl_sync(0xFFFFFFFFu, b, 0);
  return b + __popc(m & ((1u << lane) - 1));
}

// Writes the indices of the need largest of m keys (key(i), idx(i)) to
// o[0..need). Up to kRankDirect keys: each ranked against all; more: a radix
// select of the need-th, then one pass. CTA-wide; hist/sel/scan are scratch.
constexpr uint32_t kRankDirect = 256;
template <typename Key, typename Idx>
__device__ __forceinline__ void rank_write(Key key, Idx idx, uint32_t m,
                                           uint32_t need, int32_t* o,
                                           uint32_t* hist, uint32_t* sel,
                                           uint32_t* scan) {
  const uint32_t tx = threadIdx.x;
  if (m <= kRankDirect) {
    for (uint32_t i = tx; i < m; i += kThreads) {
      const uint32_t ki = key(i);
      uint32_t rnk = 0;
      for (uint32_t j = 0; j < m; j++) {
        const uint32_t kj = key(j);
        rnk += kj > ki || (kj == ki && j < i);
      }
      if (rnk < need) o[rnk] = idx(i);
    }
    return;
  }
  const uint2 t = select_low([&](uint32_t i) { return ~key(i); }, m, need, 32,
                             0, hist, sel, scan);
  if (tx < 2) sel[2 + tx] = 0;
  __syncthreads();
  for (uint32_t i0 = 0; i0 < m; i0 += kThreads) {
    const uint32_t i = i0 + tx;
    const uint32_t v = i < m ? ~key(i) : 0xFFFFFFFFu;
    const bool lt = i < m && v < t.x, eq = i < m && v == t.x;
    const uint32_t a = warp_slot(lt, &sel[2]);
    if (lt) o[a] = idx(i);
    const uint32_t e = warp_slot(eq, &sel[3]);
    if (eq && t.y + e < need) o[t.y + e] = idx(i);
  }
}

// Exact TopK over a whole row by one CTA (the last-resort bound of
// redo_rounds()).
__device__ __forceinline__ void whole_row(const uint32_t* inb, uint32_t L,
                                          uint32_t topk, int32_t* o,
                                          uint32_t* hist, uint32_t* sel,
                                          uint32_t* scan) {
  const uint32_t tx = threadIdx.x;
  const auto rk = [&](uint32_t i) { return ~okey(__ldg(inb + i)); };
  const uint2 t = select_low(rk, L, topk, 32, 0, hist, sel, scan);
  if (tx < 2) sel[2 + tx] = 0;
  __syncthreads();
  for (uint32_t i0 = 0; i0 < L; i0 += kThreads) {
    const uint32_t i = i0 + tx;
    const uint32_t v = i < L ? rk(i) : 0xFFFFFFFFu;
    const bool lt = i < L && v < t.x, eqv = i < L && v == t.x;
    const uint32_t a = warp_slot(lt, &sel[2]);
    if (lt) o[a] = static_cast<int32_t>(i);
    const uint32_t e = warp_slot(eqv, &sel[3]);
    if (eqv && t.y + e < topk) o[t.y + e] = static_cast<int32_t>(i);
  }
}

// State of a row's launch handed to redo_rounds().
struct RowCtx {
  const uint32_t* inb;
  const uint4* v4;
  int32_t* o;
  uint32_t* cur;
  unsigned long long* arrivals;
  unsigned long long base, gen;
  uint32_t* ckey;
  int32_t* cidx;
  uint32_t* skey;
  int32_t* sidx;
  uint32_t *samp, *hist, *sel, *scan, *n, *flag;
  uint32_t L, topk, nv, G, g, S, smax, shb, q, cut;
};

// The cutoff at or below the q-th largest sample (every CTA of the row gets
// the same). CTA-wide.
__device__ __forceinline__ uint32_t sample_cut(const RowCtx& c, uint32_t q) {
  const uint32_t keep = c.S == c.L ? 0u : (c.shb > 11 ? c.shb - 11 : 0u);
  const uint32_t* samp = c.samp;
  const uint32_t smax = c.smax;
  uint2 d = select_low([&](uint32_t i) { return smax - samp[i]; }, c.S, q,
                       c.shb, keep, c.hist, c.sel, c.scan);
  // Every thread reads the bin's count before the histogram is reused.
  const bool crowded =
      keep && c.hist[(d.x >> keep) & (kBins - 1)] > max(8u, q / 4);
  __syncthreads();
  if (crowded) {
    d = select_low([&](uint32_t i) { return smax - samp[i]; }, c.S, q, c.shb, 0,
                   c.hist, c.sel, c.scan);
  } else if (keep) {
    d.x |= (1u << keep) - 1;
  }
  return d.x > smax ? 0u : smax - d.x;
}

// Round 0 misjudged its cutoff: the row's CTAs (all of them, taking the same
// decisions from the same row histogram) redo it, then finish the launch.
//  - fewer than TopK survivors: a lower cutoff from the samples (4x the rank);
//  - a CTA overflowed, or the threshold bin holds more than kSamples: the new
//    cutoff is that bin's lowest value (bins get finer each round);
//  - a threshold bin of equal scores (ties) needs no ranking: any of them
//    complete the TopK, written directly (with an overflow, one round keeps
//    the scores above the tie and writes the ties as it filters).
// Inlined although it is off the common path, as whole_row() is: an
// out-of-line call in the grid kernel makes the compiler spill registers
// around it, and the kernel's cluster path, which short rows run, ran up to
// 12% slower on rows read from HBM.
__device__ __forceinline__ void redo_rounds(RowCtx c) {
  const uint32_t tx = threadIdx.x;
  const uint32_t topk = c.topk;
  uint32_t r = 0, B = 0, above = 0, tie = 0, cut = c.cut, ns = *c.n;
  bool tie_round = false, exact_bin = false, give_up = false;
  uint32_t* rs = c.cur;
  const auto filter = [&](uint32_t need, uint32_t* tcount) {
    for (uint32_t b = tx; b < kBins; b += kThreads) c.hist[b] = 0;
    if (tx == 0) {
      *c.n = 0;
      *c.flag = 0;
    }
    __syncthreads();
    const auto emit = [&](bool ok, uint32_t key, uint32_t i) {
      const bool keepit = ok && (tie_round ? key > tie : key >= cut);
      if (__ballot_sync(0xFFFFFFFFu, keepit) != 0) {
        const uint32_t p = warp_slot(keepit, c.n);
        if (keepit) {
          if (!tie_round) atomicAdd(&c.hist[lbin(key - cut)], 1u);
          if (p < kStage) {
            c.skey[p] = key;
            c.sidx[p] = static_cast<int32_t>(i);
          }
        }
      }
      if (tie_round) {
        const bool eq =
            ok && key == tie && !*reinterpret_cast<volatile uint32_t*>(c.flag);
        if (__ballot_sync(0xFFFFFFFFu, eq) != 0) {
          const uint32_t p = warp_slot(eq, tcount);
          if (eq && p < need) c.o[above + p] = static_cast<int32_t>(i);
          if (eq && p >= need) *c.flag = 1;
        }
      }
    };
    for (uint32_t j = 0; j < c.nv; j++) {
      const uint32_t i0 = (c.g + j * c.G) * kTile + 4 * tx;
      const uint4 w = i0 < c.L ? __ldcg(c.v4 + i0 / 4) : make_uint4(0, 0, 0, 0);
      emit(i0 < c.L, okey(w.x), i0);
      emit(i0 + 1 < c.L, okey(w.y), i0 + 1);
      emit(i0 + 2 < c.L, okey(w.z), i0 + 2);
      emit(i0 + 3 < c.L, okey(w.w), i0 + 3);
    }
    __syncthreads();
  };
#pragma unroll 1
  for (;;) {
    rs = c.cur + r * kRoundWords;
    uint32_t* ghist = rs + 32;
    if (r > 0) {
      if (!tie_round) {
        for (uint32_t b = tx; b < kBins; b += kThreads) {
          if (c.hist[b]) atomicAdd(&ghist[b], c.hist[b]);
        }
      }
      if (tx == 0) {
        atomicAdd(&rs[0], ns);
        if (ns > kStage) atomicExch(&rs[3], 1u);
      }
      row_barrier(c.arrivals,
                  c.base + static_cast<unsigned long long>(r + 1) * c.G, false,
                  0);
      if (tie_round)
        break;  // survivors are the scores above the tie: all of them out
    }
    const uint32_t total = __ldcg(&rs[0]);
    const bool ovf = __ldcg(&rs[3]) != 0;
    uint32_t newcut = cut;
    bool redo = false, next_tie = false;
    if (total < topk) {
      // The cutoff kept only `total` < TopK scores: the samples misjudged the
      // row (they can be fooled, e.g. by values placed at the sample
      // positions). One histogram round over the whole row, in log bins of the
      // distance below the largest sample, finds the bin holding the TopK-th
      // score; the new cutoff is that bin's lowest score, which keeps at least
      // TopK.
      redo = true;
      if (r + 2 >= kRounds) {
        newcut = 0;
      } else {
        r++;
        uint32_t* hh = c.cur + r * kRoundWords + 32;
        for (uint32_t b = tx; b < kBins; b += kThreads) c.hist[b] = 0;
        __syncthreads();
        const uint32_t smax = c.smax;
        const auto bin = [&](uint32_t key) {
          return lbin(key >= smax ? 0u : smax - key);
        };
        for (uint32_t j = 0; j < c.nv; j++) {
          const uint32_t i0 = (c.g + j * c.G) * kTile + 4 * tx;
          if (i0 < c.L) {
            const uint4 w = __ldcg(c.v4 + i0 / 4);
            atomicAdd(&c.hist[bin(okey(w.x))], 1u);
            if (i0 + 1 < c.L) atomicAdd(&c.hist[bin(okey(w.y))], 1u);
            if (i0 + 2 < c.L) atomicAdd(&c.hist[bin(okey(w.z))], 1u);
            if (i0 + 3 < c.L) atomicAdd(&c.hist[bin(okey(w.w))], 1u);
          }
        }
        __syncthreads();
        for (uint32_t b = tx; b < kBins; b += kThreads) {
          if (c.hist[b]) atomicAdd(&hh[b], c.hist[b]);
        }
        row_barrier(c.arrivals,
                    c.base + static_cast<unsigned long long>(r + 1) * c.G,
                    false, 0);
        for (uint32_t b = tx; b < kBins; b += kThreads)
          c.hist[b] = __ldcg(hh + b);
        __syncthreads();
        pick_bin(c.hist, topk, false, c.sel, c.scan);
        const uint32_t Bd = c.sel[0];
        __syncthreads();
        // Largest distance in bin Bd (lbin: exact below 64, then 64 bins an
        // octave).
        unsigned long long hi = Bd;
        if (Bd >= 64) {
          const uint32_t sh = (Bd - 64) / 64, m = (Bd - 64) % 64;
          hi = ((65ull + m) << sh) - 1;
        }
        newcut = hi >= smax ? 0u : static_cast<uint32_t>(smax - hi);
        if (newcut >= cut) newcut = 0;
      }
    } else {
      for (uint32_t b = tx; b < kBins; b += kThreads)
        c.hist[b] = __ldcg(ghist + b);
      __syncthreads();
      pick_bin(c.hist, topk, true, c.sel, c.scan);
      B = c.sel[0];
      above = c.sel[1];
      exact_bin = B < 64;  // one value: cut + B
      const uint32_t mB = c.hist[B];
      __syncthreads();
      if (exact_bin && ovf) {
        redo = next_tie = true;
        tie = cut + B;
      } else if (!exact_bin && (ovf || mB > kSamples)) {
        redo = true;
        const uint32_t e = (B - 64) / 64 + 6;
        newcut = cut + ((64u + (B - 64) % 64) << (e - 6));
      }
    }
    if (!redo) break;
    if (r + 1 == kRounds) {
      give_up = true;
      break;
    }
    r++;
    tie_round = next_tie;
    if (!tie_round) cut = newcut;
    filter(topk - above, &(c.cur + r * kRoundWords)[2]);
    ns = *c.n;
  }

  if (!give_up) {
    if (tie_round) {
      // Every survivor is above the tie; the ties were written while filtering.
      for (uint32_t i0 = 0; i0 < ns; i0 += kThreads) {
        const uint32_t i = i0 + tx;
        const bool ok = i < ns;
        const uint32_t pg = warp_slot(ok, &rs[1]);
        if (ok) c.o[pg] = c.sidx[i];
      }
    } else {
      const uint32_t need = topk - above;
      if (tx == 0) *c.flag = 0;
      __syncthreads();
      for (uint32_t i0 = 0; i0 < ns; i0 += kThreads) {
        const uint32_t i = i0 + tx;
        const bool ok = i < ns;
        const uint32_t k = ok ? c.skey[i] : 0u, b = lbin(k - cut);
        const bool gt = ok && b > B;
        const uint32_t pg = warp_slot(gt, &rs[1]);
        if (gt) c.o[pg] = c.sidx[i];
        if (exact_bin) {
          // Equal scores: any `need` of them complete the TopK.
          const bool eq =
              ok && b == B && !*reinterpret_cast<volatile uint32_t*>(c.flag);
          if (__ballot_sync(0xFFFFFFFFu, eq) != 0) {
            const uint32_t pe = warp_slot(eq, &rs[2]);
            if (eq && pe < need) c.o[above + pe] = c.sidx[i];
            if (eq && pe >= need) *c.flag = 1;
          }
        } else {
          const bool eq = ok && b == B;
          const uint32_t pe = warp_slot(eq, &rs[2]);
          if (eq && pe < kCand) {
            c.ckey[pe] = k;
            c.cidx[pe] = c.sidx[i];
          }
        }
      }
    }
  }
  // Ticket: the last CTA ranks the candidates (a bin of distinct values, at
  // most kSamples) and moves the counter to the next generation.
  __syncthreads();
  if (tx == 0) {
    const unsigned long long target =
        c.base + static_cast<unsigned long long>(r + 2) * c.G;
    const unsigned long long old = add_acqrel64(c.arrivals, 1ull);
    *c.flag = old + 1 == target;
    if (*c.flag) add_release64(c.arrivals, ((c.gen + 1) << 32) - target);
  }
  __syncthreads();
  const bool last = *c.flag;
  if (give_up && c.g == 0)
    whole_row(c.inb, c.L, topk, c.o, c.hist, c.sel, c.scan);
  if (!last) return;
  if (!give_up && !tie_round && !exact_bin) {
    const uint32_t m = __ldcg(&rs[2]), need = topk - above;
    for (uint32_t i = tx; i < m; i += kThreads) c.samp[i] = __ldcg(c.ckey + i);
    __syncthreads();
    const uint32_t* lk = c.samp;
    const int32_t* ci = c.cidx;
    rank_write([&](uint32_t i) { return lk[i]; },
               [&](uint32_t i) { return __ldcg(ci + i); }, m, need, c.o + above,
               c.hist, c.sel, c.scan);
    __syncthreads();
  }
  // Every CTA is past the ticket: rounds 1..r are free; clear them for the
  // next launch of this parity (the common path clears only round 0).
  for (uint32_t w = kRoundWords + tx; w < (r + 1) * kRoundWords; w += kThreads)
    c.cur[w] = 0;
}

// One long row: CTA g of the row's G (any G >= 1, not tied to cluster
// boundaries), kThreads threads, at least kDynSmem dynamic shared memory
// (kDynSmemSamples with kDynSamples), every CTA resident (cooperative launch).
// len: the row's length (read once by the caller). headers / ws: the row
// headers and candidate lists of the caller's state.
template <bool kDynSamples>
__device__ __forceinline__ void row_topk(
    const float* __restrict__ logits, uint32_t stride, int32_t len,
    int32_t* __restrict__ out, uint32_t topk, uint8_t* __restrict__ ws,
    uint32_t* __restrict__ headers, uint32_t row, uint32_t g, uint32_t G) {
  extern __shared__ __align__(16) uint32_t s_dyn[];
  __shared__ uint32_t s_hist[kBins];
  __shared__ uint32_t s_sel[4], s_red[2], s_scan[32];
  __shared__ uint32_t s_n, s_flag;

  const uint32_t tx = threadIdx.x;
  const uint32_t* inb = reinterpret_cast<const uint32_t*>(
      logits + static_cast<size_t>(row) * stride);
  int32_t* o = out + static_cast<size_t>(row) * topk;
  uint32_t* hdr = headers + row * kHeaderWords;
  auto* arrivals = reinterpret_cast<unsigned long long*>(hdr);
  uint32_t* ckey = reinterpret_cast<uint32_t*>(ws + row * kRowBytes);
  int32_t* cidx = reinterpret_cast<int32_t*>(ckey + kCand);
  uint32_t* skey = s_dyn;
  int32_t* sidx = reinterpret_cast<int32_t*>(s_dyn + kStage);
  __shared__ uint32_t s_samp_static[kDynSamples ? 1 : kSamples];
  uint32_t* s_samp = kDynSamples ? s_dyn + 2 * kStage : s_samp_static;
  const uint32_t L = static_cast<uint32_t>(max(len, 0));

  // Rows no longer than TopK: every index, then -1.
  if (L <= topk) {
    if (g == 0) {
      for (uint32_t i = tx; i < topk; i += kThreads)
        o[i] = i < L ? static_cast<int32_t>(i) : -1;
    }
    return;
  }
  // Generation of this launch, its parity's state, and the other parity's,
  // which this launch clears for the next one.
  const unsigned long long gen = ld_acquire64(arrivals) >> 32;
  const unsigned long long base = gen << 32;
  uint32_t* cur = hdr + 32 + (gen & 1) * kParityWords;
  uint32_t* nxt = hdr + 32 + ((gen & 1) ^ 1) * kParityWords;
  uint32_t* ghist = cur + 32;
  {
    // Round 0 only: a launch that used more rounds clears them itself.
    const uint32_t per = (kRoundWords + G - 1) / G;
    for (uint32_t w = g * per + tx; w < min(kRoundWords, (g + 1) * per);
         w += kThreads)
      nxt[w] = 0;
  }

  // This CTA's tiles: g, g + G, g + 2·G, ... (see kTile); the first kPre are
  // loaded now, while the cutoff is worked out. nv = the CTA's tile count.
  const uint32_t ntiles = (L + kTile - 1) / kTile;
  const uint32_t nv = g < ntiles ? (ntiles - 1 - g) / G + 1 : 0u;
  const uint4* v4 = reinterpret_cast<const uint4*>(inb);
  uint4 pre[kPre];
#pragma unroll
  for (uint32_t j = 0; j < kPre; j++) {
    const uint32_t i0 = (g + j * G) * kTile + 4 * tx;
    pre[j] = (j < nv && i0 < L) ? __ldcs(v4 + i0 / 4) : make_uint4(0, 0, 0, 0);
  }

  // 1. The cutoff, the same in every CTA of the row.
  const uint32_t S = min(L, kSamples);
  uint32_t mx = 0, nmn = 0;
  for (uint32_t t = tx; t < S; t += kThreads) {
    const uint32_t p =
        S == L ? t
               : static_cast<uint32_t>((static_cast<uint64_t>(2 * t + 1) * L) /
                                       (2 * S));
    const uint32_t k = okey(__ldg(inb + p));
    s_samp[t] = k;
    mx = max(mx, k);
    nmn = max(nmn, ~k);
  }
  const uint2 ex = cta_max2(mx, nmn, s_red);
  const uint32_t smax = ex.x, srange = smax - ~ex.y;
  const uint32_t shb = srange ? 32 - __clz(srange) : 0u;
  uint32_t q;
  if (S == L) {
    q = topk;
  } else {
    const float e = static_cast<float>(topk) * S / static_cast<float>(L);
    q = min(S, static_cast<uint32_t>(e + 5.0f * sqrtf(e) + 5.0f) + 1u);
  }
  const uint32_t keep = S == L ? 0u : (shb > 11 ? shb - 11 : 0u);
  uint2 d = select_low([&](uint32_t i) { return smax - s_samp[i]; }, S, q, shb,
                       keep, s_hist, s_sel, s_scan);
  // Every thread reads the bin's count before the histogram is reused.
  const bool crowded =
      keep && s_hist[(d.x >> keep) & (kBins - 1)] > max(8u, q / 4);
  __syncthreads();
  if (crowded) {
    d = select_low([&](uint32_t i) { return smax - s_samp[i]; }, S, q, shb, 0,
                   s_hist, s_sel, s_scan);
  } else if (keep) {
    d.x |= (1u << keep) - 1;
  }
  const uint32_t cut = d.x > smax ? 0u : smax - d.x;

  // 2. This CTA's slice: survivors >= cut in shared memory, their histogram
  // added to the row's.
  for (uint32_t b = tx; b < kBins; b += kThreads) s_hist[b] = 0;
  if (tx == 0) s_n = 0;
  __syncthreads();
  const auto emit = [&](bool keepit, uint32_t key, uint32_t i) {
    if (__ballot_sync(0xFFFFFFFFu, keepit) == 0) return;
    const uint32_t p = warp_slot(keepit, &s_n);
    if (keepit) {
      atomicAdd(&s_hist[lbin(key - cut)], 1u);
      if (p < kStage) {
        skey[p] = key;
        sidx[p] = static_cast<int32_t>(i);
      }
    }
  };
  for (uint32_t j = 0; j < nv; j++) {
    const uint32_t i0 = (g + j * G) * kTile + 4 * tx;
    uint4 w = make_uint4(0, 0, 0, 0);
    if (j < kPre) {
#pragma unroll
      for (uint32_t jj = 0; jj < kPre; jj++) {
        if (jj == j) w = pre[jj];
      }
    } else if (i0 < L) {
      w = __ldcs(v4 + i0 / 4);
    }
    const uint32_t k0 = okey(w.x), k1 = okey(w.y), k2 = okey(w.z),
                   k3 = okey(w.w);
    emit(i0 < L && k0 >= cut, k0, i0);
    emit(i0 + 1 < L && k1 >= cut, k1, i0 + 1);
    emit(i0 + 2 < L && k2 >= cut, k2, i0 + 2);
    emit(i0 + 3 < L && k3 >= cut, k3, i0 + 3);
  }
  __syncthreads();
  const uint32_t ns = s_n;

  // Add to the row's histogram, then barrier 1.
  for (uint32_t b = tx; b < kBins; b += kThreads) {
    if (s_hist[b]) atomicAdd(&ghist[b], s_hist[b]);
  }
  if (tx == 0) {
    atomicAdd(&cur[0], ns);
    if (ns > kStage) atomicExch(&cur[3], 1u);
  }
  row_barrier(arrivals, base + G, false, 0);
  if (tx == 0) s_flag = __ldcg(&cur[3]) != 0 || __ldcg(&cur[0]) < topk;
  __syncthreads();
  if (s_flag) {
    redo_rounds(RowCtx{inb,    v4,   o,       cur,  arrivals, base,   gen,
                       ckey,   cidx, skey,    sidx, s_samp,   s_hist, s_sel,
                       s_scan, &s_n, &s_flag, L,    topk,     nv,     G,
                       g,      S,    smax,    shb,  q,        cut});
    return;
  }
  uint32_t B = 0;
  {
    for (uint32_t b = tx; b < kBins; b += kThreads)
      s_hist[b] = __ldcg(ghist + b);
    __syncthreads();
    pick_bin(s_hist, topk, true, s_sel, s_scan);
    B = s_sel[0];
    if (s_sel[2] > kSamples) {
      redo_rounds(RowCtx{inb,    v4,   o,       cur,  arrivals, base,   gen,
                         ckey,   cidx, skey,    sidx, s_samp,   s_hist, s_sel,
                         s_scan, &s_n, &s_flag, L,    topk,     nv,     G,
                         g,      S,    smax,    shb,  q,        cut});
      return;
    }
    for (uint32_t i0 = 0; i0 < ns; i0 += kThreads) {
      const uint32_t i = i0 + tx;
      const bool ok = i < ns;
      const uint32_t k = ok ? skey[i] : 0u, b = lbin(k - cut);
      const bool gt = ok && b > B, eq = ok && b == B;
      const uint32_t pg = warp_slot(gt, &cur[1]);
      if (gt) o[pg] = sidx[i];
      const uint32_t pe = warp_slot(eq, &cur[2]);
      if (eq && pe < kCand) {
        ckey[pe] = k;
        cidx[pe] = sidx[i];
      }
    }
  }

  // Ticket: the last CTA of the row ranks the candidates; it also moves the
  // counter to the next generation.
  __syncthreads();
  if (tx == 0) {
    const unsigned long long target = base + 2ull * G;
    const unsigned long long old = add_acqrel64(arrivals, 1ull);
    s_flag = old + 1 == target;
    if (s_flag) add_release64(arrivals, ((gen + 1) << 32) - target);
  }
  __syncthreads();
  if (!s_flag) return;
  const uint32_t above = __ldcg(&cur[1]), m = __ldcg(&cur[2]);
  const uint32_t need = topk - above;
  uint32_t* lk =
      s_samp;  // m <= kSamples (more were redone) fits in the samples' space
  for (uint32_t i = tx; i < m; i += kThreads) lk[i] = __ldcg(ckey + i);
  __syncthreads();
  rank_write([&](uint32_t i) { return lk[i]; },
             [&](uint32_t i) { return __ldcg(cidx + i); }, m, need, o + above,
             s_hist, s_sel, s_scan);
}

}  // namespace vllm::cooperative_grid

#endif  // COOPERATIVE_GRID_TOPK_CUH_
