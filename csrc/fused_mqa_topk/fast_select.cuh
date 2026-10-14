// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include "candidates.cuh"

// One warp selects one row's top-K from its candidate fp24 codes (higher code =
// higher score), staged once in shared memory. Candidate codes span a narrow
// range (gate .. row max), so each level histograms the range-relative code
// into at most kBins bins: collisions are rare, plain shared atomics suffice
// and each lane keeps U independent elements in flight. Levels repeat inside
// the boundary bin until it is one code wide; codes above it are emitted, then
// ties until K.
namespace fast_select {
constexpr unsigned K = 2048, Full = 0xffffffffu, kBins = 1024, U = 8;

// Bin (descending) where the cumulative count first reaches `rank`;
// `above` = elements in higher bins. Lane l owns bins kBins-1-32l .. -31.
__device__ __forceinline__ unsigned find_bin(const unsigned* hist,
                                             unsigned rank, unsigned& above) {
  const unsigned lane = threadIdx.x & 31;
  constexpr unsigned kPerLane = kBins / 32;
  unsigned sum = 0;
#pragma unroll
  for (unsigned j = 0; j < kPerLane; ++j)
    sum += hist[kBins - 1 - lane * kPerLane - j];
  unsigned inclusive = sum;
#pragma unroll
  for (unsigned d = 1; d < 32; d *= 2) {
    const unsigned v = __shfl_up_sync(Full, inclusive, d);
    if (lane >= d) inclusive += v;
  }
  const unsigned before = inclusive - sum;
  const unsigned winner =
      __ffs(__ballot_sync(Full, before < rank && inclusive >= rank)) - 1;
  unsigned chosen = 0, acc = before;
  if (lane == winner) {
    for (unsigned j = 0; j < kPerLane; ++j) {
      const unsigned bin = kBins - 1 - lane * kPerLane - j, h = hist[bin];
      if (acc + h >= rank) {
        chosen = bin;
        break;
      }
      acc += h;
    }
  }
  above = __shfl_sync(Full, acc, winner);
  return __shfl_sync(Full, chosen, winner);
}

// Requires count >= K (callers recover rows with fewer candidates).
__device__ void select_row(unsigned* codes, unsigned count,
                           const uint16_t* values, const int32_t* packed,
                           int32_t* out, unsigned* hist) {
  const unsigned lane = threadIdx.x & 31;
  unsigned lo = 0xffffffffu, hi = 0;
  unsigned j = lane;
  for (; j + 32 * (U - 1) < count; j += 32 * U) {
    uint16_t v[U];
    int32_t p[U];
#pragma unroll
    for (unsigned u = 0; u < U; ++u) {
      v[u] = values[j + 32 * u];
      p[u] = packed[j + 32 * u];
    }
#pragma unroll
    for (unsigned u = 0; u < U; ++u) {
      const unsigned c =
          fused_candidates::candidate_load_score_code(v[u], p[u]);
      codes[j + 32 * u] = c;
      lo = min(lo, c);
      hi = max(hi, c);
    }
  }
  for (; j < count; j += 32) {
    const unsigned c =
        fused_candidates::candidate_load_score_code(values[j], packed[j]);
    codes[j] = c;
    lo = min(lo, c);
    hi = max(hi, c);
  }
  lo = __reduce_min_sync(Full, lo);
  hi = __reduce_max_sync(Full, hi);
  __syncwarp();

  // Current bucket: codes in [lo, lo + width). rank = elements still needed
  // from it.
  unsigned width = hi - lo + 1, rank = K;
#pragma unroll 1
  while (width > 1) {
    const unsigned spread = (width - 1) >> 10;
    const unsigned shift = spread ? 32 - __clz(spread) : 0;
    for (unsigned b = lane; b < kBins; b += 32) hist[b] = 0;
    __syncwarp();
    const unsigned top = lo + width - 1;
    unsigned i = lane;
    for (; i + 32 * (U - 1) < count; i += 32 * U) {
      unsigned c[U];
#pragma unroll
      for (unsigned u = 0; u < U; ++u) c[u] = codes[i + 32 * u];
#pragma unroll
      for (unsigned u = 0; u < U; ++u)
        if (c[u] >= lo && c[u] <= top)
          atomicAdd(hist + ((c[u] - lo) >> shift), 1u);
    }
    for (; i < count; i += 32) {
      const unsigned c = codes[i];
      if (c >= lo && c <= top) atomicAdd(hist + ((c - lo) >> shift), 1u);
    }
    __syncwarp();
    unsigned above;
    const unsigned chosen = find_bin(hist, rank, above);
    rank -= above;
    lo += chosen << shift;
    width = min(1u << shift, top - lo + 1);
    __syncwarp();
  }

  // lo is now the K-th largest code: emit codes above it, then ties until K.
  const unsigned threshold = lo, lt = (1u << lane) - 1u;
  unsigned pos = 0;
#pragma unroll 1
  for (unsigned tie = 0; tie < 2; ++tie) {
    for (unsigned j0 = 0; j0 < count && pos < K; j0 += 32 * U) {
      bool pass[U];
      int32_t p[U];
#pragma unroll
      for (unsigned u = 0; u < U; ++u) {
        const unsigned e = j0 + 32 * u + lane;
        const unsigned c = e < count ? codes[e] : 0;
        pass[u] = e < count && (tie ? c == threshold : c > threshold);
        p[u] = pass[u] ? packed[e] : 0;
      }
#pragma unroll
      for (unsigned u = 0; u < U; ++u) {
        const unsigned mask = __ballot_sync(Full, pass[u]),
                       at = pos + __popc(mask & lt);
        if (pass[u] && at < K)
          out[at] = fused_candidates::candidate_decode_index(p[u]);
        pos += __popc(mask);
      }
    }
  }
}
// Resident rows: fp24 codes live in shared memory as hi16[e] << 8 | lo8[e] for
// the row's visible keys e = 0..count-1 (key index s0 + e). Requires count >=
// K. Same levels as select_row.
__device__ __forceinline__ unsigned resident_code(const uint16_t* hi,
                                                  const uint8_t* lo,
                                                  unsigned e) {
  return unsigned(hi[e]) << 8 | lo[e];
}
struct IdentityKey {
  __device__ __forceinline__ int32_t operator()(unsigned e) const {
    return int32_t(e);
  }
};
template <typename KeyOf = IdentityKey>
__device__ void select_resident(const uint16_t* hi, const uint8_t* lo,
                                unsigned count, unsigned s0, int32_t* out,
                                unsigned* hist, KeyOf key_of = {}) {
  const unsigned lane = threadIdx.x & 31;
  unsigned lo_code = 0xffffffffu, hi_code = 0;
  for (unsigned j = lane; j < count; j += 32) {
    const unsigned c = resident_code(hi, lo, j);
    lo_code = min(lo_code, c);
    hi_code = max(hi_code, c);
  }
  unsigned base = __reduce_min_sync(Full, lo_code);
  const unsigned top0 = __reduce_max_sync(Full, hi_code);
  unsigned width = top0 - base + 1, rank = K;
#pragma unroll 1
  while (width > 1) {
    const unsigned spread = (width - 1) >> 10;
    const unsigned shift = spread ? 32 - __clz(spread) : 0;
    for (unsigned b = lane; b < kBins; b += 32) hist[b] = 0;
    __syncwarp();
    const unsigned top = base + width - 1;
    unsigned i = lane;
    for (; i + 32 * (U - 1) < count; i += 32 * U) {
      unsigned c[U];
#pragma unroll
      for (unsigned u = 0; u < U; ++u) c[u] = resident_code(hi, lo, i + 32 * u);
#pragma unroll
      for (unsigned u = 0; u < U; ++u)
        if (c[u] >= base && c[u] <= top)
          atomicAdd(hist + ((c[u] - base) >> shift), 1u);
    }
    for (; i < count; i += 32) {
      const unsigned c = resident_code(hi, lo, i);
      if (c >= base && c <= top) atomicAdd(hist + ((c - base) >> shift), 1u);
    }
    __syncwarp();
    unsigned above;
    const unsigned chosen = find_bin(hist, rank, above);
    rank -= above;
    base += chosen << shift;
    width = min(1u << shift, top - base + 1);
    __syncwarp();
  }
  const unsigned threshold = base, lt = (1u << lane) - 1u;
  unsigned pos = 0;
#pragma unroll 1
  for (unsigned tie = 0; tie < 2; ++tie) {
    for (unsigned j0 = 0; j0 < count && pos < K; j0 += 32 * U) {
      bool pass[U];
#pragma unroll
      for (unsigned u = 0; u < U; ++u) {
        const unsigned e = j0 + 32 * u + lane;
        const unsigned c = e < count ? resident_code(hi, lo, e) : 0;
        pass[u] = e < count && (tie ? c == threshold : c > threshold);
      }
#pragma unroll
      for (unsigned u = 0; u < U; ++u) {
        const unsigned mask = __ballot_sync(Full, pass[u]),
                       at = pos + __popc(mask & lt);
        if (pass[u] && at < K) out[at] = key_of(s0 + j0 + 32 * u + lane);
        pos += __popc(mask);
      }
    }
  }
}
// Register samples per lane for the sampled bracket selects below.
constexpr unsigned kSamplesPerLane = 16;
// Bracket select (one warp per resident row): thresholds on the hi16 part only,
// 8 elements per 16-byte shared load, per-lane counts with one warp prefix scan
// (no per-element ballots), output staged in shared memory and copied out with
// 16-byte stores. hi_row/lo_row are indexed by key index.
constexpr unsigned kMid3Cap = 1024, kScratch3Bytes = K * 4 + kMid3Cap * 2;

__device__ __forceinline__ unsigned warp_exclusive_scan(unsigned v,
                                                        unsigned& total) {
  const unsigned lane = threadIdx.x & 31;
  unsigned inc = v;
#pragma unroll
  for (unsigned d = 1; d < 32; d *= 2) {
    const unsigned t = __shfl_up_sync(Full, inc, d);
    if (lane >= d) inc += t;
  }
  total = __shfl_sync(Full, inc, 31);
  return inc - v;
}

template <typename Fn>
__device__ __forceinline__ void classify8(const uint4 v, unsigned ebase,
                                          unsigned s0, unsigned end,
                                          unsigned h_hi, unsigned h_lo,
                                          Fn&& fn) {
  const unsigned w[4] = {v.x, v.y, v.z, v.w};
  const bool interior = ebase >= s0 && ebase + 8 <= end;
#pragma unroll
  for (unsigned k = 0; k < 8; ++k) {
    const unsigned h = (w[k / 2] >> (16 * (k & 1))) & 0xffffu, e = ebase + k;
    const bool valid = interior || (e >= s0 && e < end);
    fn(e, valid && h > h_hi, valid && h >= h_lo && h <= h_hi);
  }
}

template <typename KeyOf = IdentityKey>
__device__ bool select_resident_bracket(const uint16_t* hi_row,
                                        const uint8_t* lo_row, unsigned s0,
                                        unsigned count, int32_t* out,
                                        unsigned char* scratch,
                                        KeyOf key_of = {}) {
  const unsigned lane = threadIdx.x & 31, lt = (1u << lane) - 1u;
  // hi16 bracket from 512 strided samples (+-40 sample ranks around the
  // expected K-th).
  unsigned h_hi = 0, h_lo = 0;
  {
    unsigned smp[kSamplesPerLane];
#pragma unroll
    for (unsigned u = 0; u < kSamplesPerLane; ++u)
      smp[u] = hi_row[s0 + (((u * 32 + lane) * 2 + 1) * count >> 10)];
    const unsigned expect = (K * 512 + count / 2) / count;
    const unsigned r_hi = expect > 40 ? expect - 40 : 1,
                   r_lo = min(expect + 40, 512u);
#pragma unroll 1
    for (int bit = 15; bit >= 0; --bit) {
      const unsigned ch = h_hi | (1u << bit), cl = h_lo | (1u << bit);
      unsigned nh = 0, nl = 0;
#pragma unroll
      for (unsigned u = 0; u < kSamplesPerLane; ++u) {
        nh += smp[u] >= ch;
        nl += smp[u] >= cl;
      }
      if (__reduce_add_sync(Full, nh) >= r_hi) h_hi = ch;
      if (__reduce_add_sync(Full, nl) >= r_lo) h_lo = cl;
    }
  }
  const unsigned a0 = s0 & ~7u, end = s0 + count, units = (end - a0 + 7) / 8;
  const uint4* hv = reinterpret_cast<const uint4*>(hi_row + a0);
  unsigned na = 0, nm = 0;
#pragma unroll 2
  for (unsigned unit = lane; unit < units; unit += 32)
    classify8(hv[unit], a0 + unit * 8, s0, end, h_hi, h_lo,
              [&](unsigned, bool a, bool m) {
                na += a;
                nm += m;
              });
  unsigned above, mid;
  unsigned pa = warp_exclusive_scan(na, above),
           pm = warp_exclusive_scan(nm, mid);
  if (above >= K || above + mid < K || mid > kMid3Cap) return false;
  int32_t* stage = reinterpret_cast<int32_t*>(scratch);
  uint16_t* midbuf = reinterpret_cast<uint16_t*>(scratch + K * 4);
#pragma unroll 2
  for (unsigned unit = lane; unit < units; unit += 32)
    classify8(hv[unit], a0 + unit * 8, s0, end, h_hi, h_lo,
              [&](unsigned e, bool a, bool m) {
                if (a) stage[pa++] = key_of(e);
                if (m) midbuf[pm++] = uint16_t(e);
              });
  __syncwarp();
  // Exact K-th code among the bracket (rank r), over the bits where the bracket
  // bounds differ.
  constexpr unsigned R = kMid3Cap / 32;
  unsigned mc[R];
#pragma unroll
  for (unsigned u = 0; u < R; ++u) {
    const unsigned i = u * 32 + lane;
    mc[u] = 0;
    if (u * 32 < mid && i < mid) {
      const unsigned e = midbuf[i];
      mc[u] = unsigned(hi_row[e]) << 8 | lo_row[e];
    }
  }
  const unsigned r = K - above, c_lo = h_lo << 8, c_hi = h_hi << 8 | 0xffu,
                 diff = c_lo ^ c_hi;
  const int top_bit = diff ? 31 - __clz(diff) : -1;
  unsigned t = top_bit >= 0 ? c_lo & ~((2u << top_bit) - 1u) : c_lo;
#pragma unroll 1
  for (int bit = top_bit; bit >= 0; --bit) {
    const unsigned cand = t | (1u << bit);
    unsigned n = 0;
#pragma unroll
    for (unsigned u = 0; u < R; ++u)
      if (u * 32 < mid) n += mc[u] >= cand;
    if (__reduce_add_sync(Full, n) >= r) t = cand;
  }
  unsigned pos = above;
#pragma unroll 1
  for (unsigned tie = 0; tie < 2; ++tie) {
#pragma unroll
    for (unsigned u = 0; u < R; ++u) {
      if (u * 32 >= mid || pos >= K) break;
      const unsigned i = u * 32 + lane;
      const bool pass = i < mid && (tie ? mc[u] == t : mc[u] > t);
      const unsigned mask = __ballot_sync(Full, pass),
                     at = pos + __popc(mask & lt);
      if (pass && at < K) stage[at] = key_of(midbuf[i]);
      pos += __popc(mask);
    }
  }
  __syncwarp();
#pragma unroll 4
  for (unsigned j = lane * 4; j < K; j += 128)
    *reinterpret_cast<int4*>(out + j) =
        *reinterpret_cast<const int4*>(stage + j);
  return true;
}
// Histogram select over any code source: code_of(e) / key_of(e) for e in [0,
// count). Used for rows whose shared candidates overflowed into global memory
// (rare).
template <typename CodeOf, typename KeyOf>
__device__ void select_hist_generic(CodeOf code_of, unsigned count,
                                    int32_t* out, unsigned* hist,
                                    KeyOf key_of) {
  const unsigned lane = threadIdx.x & 31;
  unsigned lo_code = 0xffffffffu, hi_code = 0;
  for (unsigned j = lane; j < count; j += 32) {
    const unsigned c = code_of(j);
    lo_code = min(lo_code, c);
    hi_code = max(hi_code, c);
  }
  unsigned base = __reduce_min_sync(Full, lo_code);
  const unsigned top0 = __reduce_max_sync(Full, hi_code);
  unsigned width = top0 - base + 1, rank = K;
#pragma unroll 1
  while (width > 1) {
    const unsigned spread = (width - 1) >> 10;
    const unsigned shift = spread ? 32 - __clz(spread) : 0;
    for (unsigned b = lane; b < kBins; b += 32) hist[b] = 0;
    __syncwarp();
    const unsigned top = base + width - 1;
    for (unsigned i = lane; i < count; i += 32) {
      const unsigned c = code_of(i);
      if (c >= base && c <= top) atomicAdd(hist + ((c - base) >> shift), 1u);
    }
    __syncwarp();
    unsigned above;
    const unsigned chosen = find_bin(hist, rank, above);
    rank -= above;
    base += chosen << shift;
    width = min(1u << shift, top - base + 1);
    __syncwarp();
  }
  const unsigned threshold = base, lt = (1u << lane) - 1u;
  unsigned pos = 0;
#pragma unroll 1
  for (unsigned tie = 0; tie < 2; ++tie) {
    for (unsigned j0 = 0; j0 < count && pos < K; j0 += 32) {
      const unsigned e = j0 + lane;
      const unsigned c = e < count ? code_of(e) : 0;
      const bool pass = e < count && (tie ? c == threshold : c > threshold);
      const unsigned mask = __ballot_sync(Full, pass),
                     at = pos + __popc(mask & lt);
      if (pass && at < K) out[at] = key_of(e);
      pos += __popc(mask);
    }
  }
}
// Bins of the histogram levels that resolve the exact threshold inside a
// bracket (row_select.cuh).
constexpr unsigned kMidBins = 512;

template <unsigned Bins>
__device__ __forceinline__ unsigned find_bin_n(const unsigned* hist,
                                               unsigned rank, unsigned& above) {
  const unsigned lane = threadIdx.x & 31;
  constexpr unsigned kPerLane = Bins / 32;
  unsigned sum = 0;
#pragma unroll
  for (unsigned j = 0; j < kPerLane; ++j)
    sum += hist[Bins - 1 - lane * kPerLane - j];
  unsigned inclusive = sum;
#pragma unroll
  for (unsigned d = 1; d < 32; d *= 2) {
    const unsigned v = __shfl_up_sync(Full, inclusive, d);
    if (lane >= d) inclusive += v;
  }
  const unsigned before = inclusive - sum;
  const unsigned winner =
      __ffs(__ballot_sync(Full, before < rank && inclusive >= rank)) - 1;
  unsigned chosen = 0, acc = before;
  if (lane == winner) {
    for (unsigned j = 0; j < kPerLane; ++j) {
      const unsigned bin = Bins - 1 - lane * kPerLane - j, h = hist[bin];
      if (acc + h >= rank) {
        chosen = bin;
        break;
      }
      acc += h;
    }
  }
  above = __shfl_sync(Full, acc, winner);
  return __shfl_sync(Full, chosen, winner);
}

}  // namespace fast_select
