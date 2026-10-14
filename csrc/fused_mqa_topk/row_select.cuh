// SPDX-License-Identifier: MIT
// Single-warp top-2048 selection of one row's candidates for the persistent
// kernel. Both selections bracket the 2048-th largest fp24 code with 512
// sampled codes, place the codes above the bracket directly, and resolve the
// bracket with histogram levels on register-resident codes; elements are placed
// by per-lane counts and one warp prefix scan (no per-element ballots).
#pragma once
#include "fast_select.cuh"

namespace row_select {
using namespace fast_select;

// r-th largest (1-based) of the 512 register samples for two ranks at once
// (r_hi <= r_lo), via a 256-bin hi8 level and two 256-bin lo8 levels. h8: 768
// words of scratch.
template <unsigned NS = kSamplesPerLane>
__device__ __forceinline__ void radix_bracket(const unsigned (&smp)[NS],
                                              unsigned r_hi, unsigned r_lo,
                                              unsigned* h8, unsigned& h_hi,
                                              unsigned& h_lo) {
  const unsigned lane = threadIdx.x & 31;
#pragma unroll
  for (unsigned b = lane; b < 192; b += 32)
    reinterpret_cast<uint4*>(h8)[b] = make_uint4(0, 0, 0, 0);
  __syncwarp();
#pragma unroll
  for (unsigned u = 0; u < NS; ++u) atomicAdd(h8 + (smp[u] >> 8), 1u);
  __syncwarp();
  unsigned ab_hi, ab_lo;
  const unsigned b_hi = find_bin_n<256>(h8, r_hi, ab_hi);
  const unsigned b_lo = find_bin_n<256>(h8, r_lo, ab_lo);
  unsigned* const lh = h8 + 256;
  unsigned* const ll = h8 + 512;
#pragma unroll
  for (unsigned u = 0; u < NS; ++u) {
    const unsigned b = smp[u] >> 8;
    if (b == b_hi) atomicAdd(lh + (smp[u] & 255u), 1u);
    if (b == b_lo) atomicAdd(ll + (smp[u] & 255u), 1u);
  }
  __syncwarp();
  unsigned d0, d1;
  h_hi = b_hi << 8 | find_bin_n<256>(lh, r_hi - ab_hi, d0);
  h_lo = b_lo << 8 | find_bin_n<256>(ll, r_lo - ab_lo, d1);
  __syncwarp();  // h8 is reused as the output stage
}

// find_bin_n with each lane's kPer contiguous bins loaded as 16-byte vectors
// and searched in registers.
template <unsigned Bins>
__device__ __forceinline__ unsigned find_bin_v(const unsigned* hist,
                                               unsigned rank, unsigned& above) {
  const unsigned lane = threadIdx.x & 31;
  constexpr unsigned kPer = Bins / 32, kVec = kPer / 4;
  static_assert(kPer % 4 == 0);
  // lane l owns bins [Bins - (l + 1) * kPer, Bins - l * kPer), highest first.
  const uint4* hv =
      reinterpret_cast<const uint4*>(hist + Bins - (lane + 1) * kPer);
  unsigned h[kPer];
#pragma unroll
  for (unsigned v = 0; v < kVec; ++v) {
    const uint4 x = hv[v];
    h[4 * v] = x.x;
    h[4 * v + 1] = x.y;
    h[4 * v + 2] = x.z;
    h[4 * v + 3] = x.w;
  }
  unsigned sum = 0;
#pragma unroll
  for (unsigned j = 0; j < kPer; ++j) sum += h[j];
  unsigned inclusive = sum;
#pragma unroll
  for (unsigned d = 1; d < 32; d *= 2) {
    const unsigned v = __shfl_up_sync(Full, inclusive, d);
    if (lane >= d) inclusive += v;
  }
  const unsigned before = inclusive - sum;
  const unsigned winner =
      __ffs(__ballot_sync(Full, before < rank && inclusive >= rank)) - 1;
  // Descending walk inside the lane: bin index Bins - lane * kPer - 1 - j  <->
  // h[kPer - 1 - j].
  unsigned acc = before, chosen = 0, found = 0, acc_at = before;
#pragma unroll
  for (unsigned j = 0; j < kPer; ++j) {
    const unsigned x = h[kPer - 1 - j];
    const bool hit = !found && acc + x >= rank;
    if (hit) {
      chosen = Bins - lane * kPer - 1 - j;
      acc_at = acc;
      found = 1;
    }
    acc += x;
  }
  above = __shfl_sync(Full, acc_at, winner);
  return __shfl_sync(Full, chosen, winner);
}
// Sampled hi16 bracket (16-step bit search over the 512 register samples).
__device__ __forceinline__ void bit_bracket(
    const unsigned (&smp)[kSamplesPerLane], unsigned r_hi, unsigned r_lo,
    unsigned& h_hi, unsigned& h_lo) {
  h_hi = 0;
  h_lo = 0;
#pragma unroll 1
  for (int bit = 15; bit >= 0; --bit) {
    const unsigned ch = h_hi | (1u << bit), cl = h_lo | (1u << bit);
    unsigned nh = 0, nl = 0;
#pragma unroll
    for (unsigned u = 0; u < kSamplesPerLane; ++u) {
      nh += smp[u] >= ch;
      nl += smp[u] >= cl;
    }
    const unsigned both =
        __reduce_add_sync(Full, nh | nl << 16);  // counts <= 512 fit 16 bits
    if ((both & 0xffffu) >= r_hi) h_hi = ch;
    if ((both >> 16) >= r_lo) h_lo = cl;
  }
}

// <= 16K keys: one row's shared-memory candidates (hi16 / lo8 code arrays and
// the uint16 key array idx_row, read with 16-byte loads). Bracket keys are
// preloaded into registers before the emit. Returns false (out untouched) when
// the bracket misses; the caller then runs select_resident.
__device__ bool select_shared_row(const uint16_t* __restrict__ hi_row,
                                  const uint8_t* __restrict__ lo_row,
                                  const uint16_t* __restrict__ idx_row,
                                  unsigned count, unsigned real,
                                  int32_t* __restrict__ out,
                                  unsigned char* scratch) {
  const unsigned lane = threadIdx.x & 31;
  unsigned h_hi, h_lo;
  {
    unsigned smp[kSamplesPerLane], n_real = 0;
#pragma unroll
    for (unsigned u = 0; u < kSamplesPerLane; ++u) {
      smp[u] = hi_row[(((u * 32 + lane) * 2 + 1) * count >> 10)];
      n_real += smp[u] != 0;
    }
    const unsigned s_real = __reduce_add_sync(Full, n_real);
    if (s_real == 0 || real <= K) return false;
    const unsigned expect = (K * s_real + real / 2) / real;
    const unsigned r_hi = expect > 40 ? expect - 40 : 1,
                   r_lo = min(expect + 40, s_real);
    bit_bracket(smp, r_hi, r_lo, h_hi, h_lo);
    h_lo = max(h_lo, 1u);
  }
  const unsigned units = (count + 7) / 8;
  const uint4* hv = reinterpret_cast<const uint4*>(hi_row);
  const uint4* iv = reinterpret_cast<const uint4*>(idx_row);
  unsigned na = 0, nm = 0;
#pragma unroll 2
  for (unsigned unit = lane; unit < units; unit += 32)
    classify8(hv[unit], unit * 8, 0u, count, h_hi, h_lo,
              [&](unsigned, bool a, bool m) {
                na += a;
                nm += m;
              });
  unsigned above, mid;
  unsigned pa = warp_exclusive_scan(na, above),
           pm = warp_exclusive_scan(nm, mid);
  if (above >= K || above + mid < K || mid > kMid3Cap) return false;
  uint16_t* stage = reinterpret_cast<uint16_t*>(scratch);
  uint16_t* midbuf = reinterpret_cast<uint16_t*>(scratch + K * 2);
  unsigned* hist = reinterpret_cast<unsigned*>(scratch + K * 2 + kMid3Cap * 2);
#pragma unroll 2
  for (unsigned unit = lane; unit < units; unit += 32) {
    const uint4 h4 = hv[unit], k4 = iv[unit];
    const unsigned kw[4] = {k4.x, k4.y, k4.z, k4.w};
    classify8(h4, unit * 8, 0u, count, h_hi, h_lo,
              [&](unsigned e, bool a, bool m) {
                const unsigned k = e & 7u;
                const unsigned key = (kw[k / 2] >> (16 * (k & 1))) & 0xffffu;
                if (a) stage[pa++] = uint16_t(key);
                if (m) midbuf[pm++] = uint16_t(e);
              });
  }
  __syncwarp();
  constexpr unsigned R = kMid3Cap / 32;
  unsigned mc[R];
#pragma unroll
  for (unsigned u = 0; u < R; ++u) {
    const unsigned i = lane + 32 * u;
    mc[u] = 0;
    if (i < mid) {
      const unsigned e = midbuf[i];
      mc[u] = unsigned(hi_row[e]) << 8 | lo_row[e];
    }
  }
  unsigned rank = K - above, base = h_lo << 8, top = h_hi << 8 | 0xffu;
#pragma unroll 1
  while (top > base) {
    const unsigned width = top - base + 1, spread = (width - 1) / kMidBins;
    const unsigned shift = spread ? 32 - __clz(spread) : 0;
#pragma unroll
    for (unsigned b = lane; b < kMidBins / 4; b += 32)
      reinterpret_cast<uint4*>(hist)[b] = make_uint4(0, 0, 0, 0);
    __syncwarp();
#pragma unroll
    for (unsigned u = 0; u < R; ++u) {
      if (u * 32 >= mid) break;
      if (mc[u] >= base && mc[u] <= top)
        atomicAdd(hist + ((mc[u] - base) >> shift), 1u);
    }
    __syncwarp();
    unsigned above_bin;
    const unsigned chosen = find_bin_v<kMidBins>(hist, rank, above_bin);
    rank -= above_bin;
    base += chosen << shift;
    top = min(base + (1u << shift) - 1, top);
    __syncwarp();
  }
  const unsigned th = base;
  {
    unsigned ng = 0, nt = 0, mk[R];
#pragma unroll
    for (unsigned u = 0; u < R; ++u) {
      ng += mc[u] > th;
      nt += mc[u] == th;
      mk[u] = 0;
      if (u * 32 < mid && mc[u] >= th) mk[u] = idx_row[midbuf[lane + 32 * u]];
    }
    unsigned wg, wt;
    unsigned pg = above + warp_exclusive_scan(ng, wg),
             pt = K - rank + warp_exclusive_scan(nt, wt);
#pragma unroll
    for (unsigned u = 0; u < R; ++u) {
      if (u * 32 >= mid) break;
      if (mc[u] > th)
        stage[pg++] = uint16_t(mk[u]);
      else if (mc[u] == th) {
        if (pt < K) stage[pt] = uint16_t(mk[u]);
        ++pt;
      }
    }
  }
  __syncwarp();
#pragma unroll 4
  for (unsigned j = lane * 4; j < K; j += 128) {
    const uint2 v = *reinterpret_cast<const uint2*>(stage + j);
    *reinterpret_cast<int4*>(out + j) = make_int4(
        int(v.x & 0xffffu), int(v.x >> 16), int(v.y & 0xffffu), int(v.y >> 16));
  }
  return true;
}

// > 16K keys: one row's global-memory candidates, staged to shared memory as
// fp24 codes (codes must be 16-byte aligned, i.e. row offsets rounded up to 4).
// Register budget: ptxas compiles the whole kernel to 168 regs, so every pass
// keeps <= ~32 regs of arrays live (more spilled into the scan loop: 290K ->
// 860K cycles/block in a first try).
// Passes: stage (16-byte global loads of values / packed indices, two chunks of
// 8 in flight per lane, to fp24 codes); hi16 bracket (radix levels over 512
// shared-memory samples); count above / inside the bracket (16-byte shared
// loads); collect (codes above the bracket to out, bracket element ids to
// midbuf); histogram levels on register codes; emit.
// Returns false before writing out; the caller's select_row then restages and
// runs the levels.
__device__ __noinline__ bool select_staged_row(
    unsigned* __restrict__ codes, unsigned count,
    const uint16_t* __restrict__ values, const int32_t* __restrict__ packed,
    int32_t* __restrict__ out, unsigned* hist) {
  const unsigned lane = threadIdx.x & 31;
  if (count <= K) return false;
  const unsigned chunks = (count + 7) / 8;
  constexpr unsigned C = 2;
  const uint4* v4p = reinterpret_cast<const uint4*>(values);
  const uint4* p4p = reinterpret_cast<const uint4*>(packed);
#pragma unroll 1
  for (unsigned c0i = lane; c0i < chunks; c0i += 32 * C) {
    uint4 vv[C], pp0[C], pp1[C];
#pragma unroll
    for (unsigned u = 0; u < C; ++u) {
      const unsigned c = min(c0i + 32 * u, chunks - 1);
      vv[u] = __ldcg(v4p + c);
      pp0[u] = __ldcg(p4p + 2 * c);
      pp1[u] = __ldcg(p4p + 2 * c + 1);
    }
#pragma unroll
    for (unsigned u = 0; u < C; ++u) {
      const unsigned c = c0i + 32 * u;
      if (c < chunks) {
        const unsigned vw[4] = {vv[u].x, vv[u].y, vv[u].z, vv[u].w};
        const unsigned pw[8] = {pp0[u].x, pp0[u].y, pp0[u].z, pp0[u].w,
                                pp1[u].x, pp1[u].y, pp1[u].z, pp1[u].w};
        unsigned cw[8];
#pragma unroll
        for (unsigned k = 0; k < 8; ++k)
          cw[k] = fused_candidates::candidate_load_score_code(
              uint16_t(vw[k / 2] >> (16 * (k & 1))), int32_t(pw[k]));
        if (8 * c + 8 <= count) {
          reinterpret_cast<uint4*>(codes)[2 * c] =
              make_uint4(cw[0], cw[1], cw[2], cw[3]);
          reinterpret_cast<uint4*>(codes)[2 * c + 1] =
              make_uint4(cw[4], cw[5], cw[6], cw[7]);
        } else {
#pragma unroll
          for (unsigned k = 0; k < 8; ++k)
            if (8 * c + k < count) codes[8 * c + k] = cw[k];
        }
      }
    }
  }
  __syncwarp();
  unsigned h_hi, h_lo;
  if (count <= 3584) {
    unsigned smp[kSamplesPerLane];
#pragma unroll
    for (unsigned u = 0; u < kSamplesPerLane; ++u)
      smp[u] = codes[(((u * 32 + lane) * 2 + 1) * count >> 10)] >> 8;
    const unsigned expect = (K * 512 + count / 2) / count;
    const unsigned r_hi = expect > 40 ? expect - 40 : 1,
                   r_lo = min(expect + 40, 512u);
    radix_bracket(smp, r_hi, r_lo, hist, h_hi, h_lo);
  } else {
    // Long-context rows (4-6K candidates): +-40 of 512 samples gives a bracket
    // of ~16% of the row, over kMid3Cap for 10-20% of the rows at 512K-1M. 1024
    // samples, +-48 (~3 sd): ~9%.
    unsigned smp[2 * kSamplesPerLane];
#pragma unroll
    for (unsigned u = 0; u < 2 * kSamplesPerLane; ++u)
      smp[u] = codes[(((u * 32 + lane) * 2 + 1) * count >> 11)] >> 8;
    const unsigned expect = (K * 1024 + count / 2) / count;
    const unsigned r_hi = expect > 48 ? expect - 48 : 1,
                   r_lo = min(expect + 48, 1024u);
    radix_bracket<2 * kSamplesPerLane>(smp, r_hi, r_lo, hist, h_hi, h_lo);
  }
  const uint4* c4p = reinterpret_cast<const uint4*>(codes);
  const unsigned quads = (count + 3) / 4;
  const unsigned lt = (1u << lane) - 1u;
  uint16_t* midbuf = reinterpret_cast<uint16_t*>(hist);
  unsigned* h512 = hist + kMid3Cap / 2;
  // Collect with warp-ballot placement: passing lanes of one step write
  // consecutive slots, so the global stores of the above elements are coalesced
  // (per-lane runs were 4x slower: 32 lines/STG). Writes are bounded; the
  // bracket check follows (out may be partially written on failure: the
  // fallback select_row rewrites all K slots).
  unsigned above = 0, mid = 0;
  constexpr unsigned Q = 4;
#pragma unroll 1
  for (unsigned q0 = 0; q0 < quads; q0 += 32 * Q) {
    uint4 pp[Q], cc[Q];
#pragma unroll
    for (unsigned u = 0; u < Q; ++u) {
      const unsigned q = min(q0 + 32 * u + lane, quads - 1);
      pp[u] = __ldcg(p4p + q);
      cc[u] = c4p[q];
    }
#pragma unroll
    for (unsigned u = 0; u < Q; ++u) {
      const unsigned q = q0 + 32 * u + lane;
      const unsigned w[4] = {cc[u].x, cc[u].y, cc[u].z, cc[u].w};
      const unsigned p[4] = {pp[u].x, pp[u].y, pp[u].z, pp[u].w};
#pragma unroll
      for (unsigned k = 0; k < 4; ++k) {
        const unsigned e = 4 * q + k, h = w[k] >> 8;
        const bool valid = e < count;
        const bool a = valid && h > h_hi, m = valid && !a && h >= h_lo;
        const unsigned ma = __ballot_sync(Full, a), mm = __ballot_sync(Full, m);
        const unsigned at = above + __popc(ma & lt), mt = mid + __popc(mm & lt);
        if (a && at < K)
          out[at] = fused_candidates::candidate_decode_index(int32_t(p[k]));
        if (m && mt < kMid3Cap) midbuf[mt] = uint16_t(e);
        above += __popc(ma);
        mid += __popc(mm);
      }
    }
  }
  __syncwarp();
  if (above >= K || above + mid < K || mid > kMid3Cap) return false;
  constexpr unsigned R = kMid3Cap / 32;
  unsigned mc[R];
#pragma unroll
  for (unsigned u = 0; u < R; ++u) {
    const unsigned i = lane + 32 * u;
    mc[u] = 0;
    if (i < mid) mc[u] = codes[midbuf[i]];
  }
  unsigned rank = K - above, base = h_lo << 8, top = h_hi << 8 | 0xffu;
#pragma unroll 1
  while (top > base) {
    const unsigned width = top - base + 1, spread = (width - 1) / kMidBins;
    const unsigned shift = spread ? 32 - __clz(spread) : 0;
#pragma unroll
    for (unsigned b = lane; b < kMidBins / 4; b += 32)
      reinterpret_cast<uint4*>(h512)[b] = make_uint4(0, 0, 0, 0);
    __syncwarp();
#pragma unroll
    for (unsigned u = 0; u < R; ++u) {
      if (u * 32 >= mid) break;
      if (lane + 32 * u < mid && mc[u] >= base && mc[u] <= top)
        atomicAdd(h512 + ((mc[u] - base) >> shift), 1u);
    }
    __syncwarp();
    unsigned above_bin;
    const unsigned chosen = find_bin_v<kMidBins>(h512, rank, above_bin);
    rank -= above_bin;
    base += chosen << shift;
    top = min(base + (1u << shift) - 1, top);
    __syncwarp();
  }
  const unsigned th = base;
  {
    // Ballot placement (coalesced stores): greater codes fill [above, K -
    // rank), ties [K - rank, K).
    unsigned pg = above, pt = K - rank;
#pragma unroll
    for (unsigned h = 0; h < 2; ++h) {
      if (h * 16 * 32 >= mid) break;
      int32_t pk[16];
#pragma unroll
      for (unsigned j = 0; j < 16; ++j) {
        const unsigned u = h * 16 + j;
        pk[j] = (lane + 32 * u < mid && mc[u] >= th)
                    ? __ldcg(packed + midbuf[lane + 32 * u])
                    : 0;
      }
#pragma unroll
      for (unsigned j = 0; j < 16; ++j) {
        const unsigned u = h * 16 + j;
        if (u * 32 >= mid) break;
        const bool v = lane + 32 * u < mid;
        const bool g = v && mc[u] > th, tq = v && mc[u] == th;
        const unsigned mg = __ballot_sync(Full, g),
                       mt = __ballot_sync(Full, tq);
        const unsigned ag = pg + __popc(mg & lt), at = pt + __popc(mt & lt);
        if (g) out[ag] = fused_candidates::candidate_decode_index(pk[j]);
        if (tq && at < K)
          out[at] = fused_candidates::candidate_decode_index(pk[j]);
        pg += __popc(mg);
        pt += __popc(mt);
      }
    }
  }
  return true;
}
}  // namespace row_select
