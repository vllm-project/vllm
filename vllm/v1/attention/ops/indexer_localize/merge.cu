// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Merge of the two locality domains' top-k candidates of the sparse-attention
// indexer into one top-k per row (one CTA per row).
//
// Each domain's top-k ran on its compact score row; c0/c1 hold the selected
// local columns (-1 for none), s0/s1 the compact scores and m0/m1 the
// local-to-request page maps. The kernel keeps the TOP_K largest of the 2 x
// TOP_K candidates (exact K-th largest key by a radix select over 8-bit
// digits; ties are taken in candidate order) and writes their request-local
// token positions to dst, -1 past the number of valid candidates.
//
// Compiled at runtime with NVRTC; the configuration comes from defines:
//   TOP_K           512, 1024 or 2048
//   THREADS         CTA size; each thread loads 2 * TOP_K / THREADS candidates
//                   and stores up to 4 outputs
//   SCORE_F32       1: FP32 scores, 0: BF16 scores
//   LOG2_PAGE_ROWS  log2 of the indexer cache page rows

typedef unsigned int u32;

#define ELEMS (2 * TOP_K / THREADS)
#define PAGE_MASK ((1 << LOG2_PAGE_ROWS) - 1)
#if SCORE_F32
typedef u32 score_bits;
  // Order-preserving keys use all 32 bits of an FP32 score.
  #define KEY_LSB 0
#else
typedef unsigned short score_bits;
  // Keys of BF16 scores live in the upper 16 bits; the select never looks
  // below them.
  #define KEY_LSB 16
#endif
// One 256-bin histogram per pass of 8 key bits.
#define HIST_WORDS (256 * ((32 - KEY_LSB) / 8))

// A thread's candidate columns, loaded as one vector.
struct __align__(4 * ELEMS) Cols {
  int c[ELEMS];
};

extern "C" __global__ __launch_bounds__(THREADS) void indexer_localize_merge(
    const score_bits* __restrict__ s0, const score_bits* __restrict__ s1,
    int s_stride, const int* __restrict__ c0, const int* __restrict__ c1,
    int c_stride, const int* __restrict__ m0, const int* __restrict__ m1,
    int m_stride, int* __restrict__ dst, int dst_stride) {
  __shared__ __align__(16) struct {
    int orow[TOP_K];  // first: 16-byte aligned for the vector reads
    u32 hist[HIST_WORDS];
    u32 rng[3];  // smallest and largest key, valid candidates
    int ctl[3];  // selected digit, keys above it, keys in it
    u32 wtot[THREADS / 32];
  } sm;
  const int tid = threadIdx.x, lane = tid % 32;
  // A shuffle result, which the compiler knows is warp-uniform: the shuffles
  // under `warp == 0` need no convergence handling.
  const int warp = __shfl_sync(0xffffffffu, tid / 32, 0);
  const int row = blockIdx.x;

  // The first half of the CTA loads domain 0's candidates, the second half
  // domain 1's.
  const bool die = tid >= THREADS / 2;
  const int first = (tid % (THREADS / 2)) * ELEMS;
  for (int i = tid; i < HIST_WORDS; i += THREADS) sm.hist[i] = 0;
  if (tid == 0) {
    sm.rng[0] = 0xffffffffu;
    sm.rng[1] = 0;
    sm.rng[2] = 0;
  }
  const Cols cols = *reinterpret_cast<const Cols*>(
      (die ? c1 : c0) + (long long)row * c_stride + first);
  const score_bits* s_row = (die ? s1 : s0) + (long long)row * s_stride;
  const int* m_row = (die ? m1 : m0) + (long long)row * m_stride;
  u32 keys[ELEMS];
  int maps[ELEMS];
  __syncthreads();
  // Order-preserving unsigned keys; the range of this CTA's valid keys.
  u32 k_lo = 0xffffffffu, k_hi = 0;
  int n_ok = 0;
#pragma unroll
  for (int e = 0; e < ELEMS; e++) {
    const int c = max(cols.c[e], 0);
    const u32 bits = (u32)s_row[c] << KEY_LSB;
    keys[e] = bits ^ ((u32)((int)bits >> 31) | 0x80000000u);
    maps[e] = m_row[c >> LOG2_PAGE_ROWS];
    if (cols.c[e] >= 0) {
      k_lo = min(k_lo, keys[e]);
      k_hi = max(k_hi, keys[e]);
      n_ok += 1;
    }
  }
  const u32 w_lo = __reduce_min_sync(0xffffffffu, k_lo);
  const u32 w_hi = __reduce_max_sync(0xffffffffu, k_hi);
  const int w_ok = __reduce_add_sync(0xffffffffu, n_ok);
  if (lane == 0) {
    atomicMin(&sm.rng[0], w_lo);
    atomicMax(&sm.rng[1], w_hi);
    atomicAdd(&sm.rng[2], (u32)w_ok);
  }
  __syncthreads();
  const u32 lo = sm.rng[0], hi = sm.rng[1];
  const int nvalid = (int)sm.rng[2];
  // Radix select of the TOP_K-th largest key over the bits below the common
  // prefix of [lo, hi], 8 bits per pass: pf/pm are the selected prefix and
  // its mask, need the keys still to take within the prefix. With every key
  // equal, top is -1 and there is no pass.
  const int top = 31 - __clz(lo ^ hi);
  u32 pm = top >= 0 ? 0xfffffffeu << top : 0xffffffffu;
  u32 pf = lo & pm;
  int need = TOP_K;
  int n_pass = (top - KEY_LSB + 8) / 8;
  if (nvalid <= TOP_K) {  // keep every valid candidate
    pm = 0;
    pf = 0;
    need = nvalid;
    n_pass = 0;
  }
  for (int p = 0; p < n_pass; p++) {
    const int lsb = max(top - 8 * p - 7, KEY_LSB);
    const u32 dmask = 255u >> (8 - (top - 8 * p - lsb + 1));
    const int hb = p * 256;
    const u32 lead = __shfl_sync(0xffffffffu, keys[0] >> lsb & dmask, 0);
    int n_lead = 0;
#pragma unroll
    for (int e = 0; e < ELEMS; e++) {
      if (cols.c[e] >= 0 && (keys[e] & pm) == pf) {
        const u32 dg = keys[e] >> lsb & dmask;
        if (dg == lead) {
          n_lead += 1;
        } else {
          atomicAdd(&sm.hist[hb + dg], 1u);
        }
      }
    }
    const int n_lead_w = __reduce_add_sync(0xffffffffu, n_lead);
    if (lane == 0 && n_lead_w > 0)
      atomicAdd(&sm.hist[hb + lead], (u32)n_lead_w);
    __syncthreads();
    if (warp == 0) {
      // Warp 0 finds the digit holding the need-th largest key: lane l owns
      // bins 8l..8l+7, suffix sums run from the top digit down.
      u32 bins[8];
      u32 own = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        bins[i] = sm.hist[hb + lane * 8 + i];
        own += bins[i];
      }
      u32 suf = own;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const u32 up = __shfl_down_sync(0xffffffffu, suf, o);
        if (lane + o < 32) suf += up;
      }
      const u32 hit = __ballot_sync(0xffffffffu, (int)suf >= need);
      u32 run = suf - own;
      if (lane == 31 - __clz(hit)) {
#pragma unroll
        for (int j = 7; j >= 0; j--) {
          if ((int)run < need && (int)(run + bins[j]) >= need) {
            sm.ctl[0] = lane * 8 + j;
            sm.ctl[1] = (int)run;
            sm.ctl[2] = (int)bins[j];
          }
          run += bins[j];
        }
      }
    }
    __syncthreads();
    need -= sm.ctl[1];
    pf |= (u32)sm.ctl[0] << lsb;
    pm |= dmask << lsb;
    if (sm.ctl[2] == need) break;
  }
  // Output slots: keys above the selected prefix first, then keys within it
  // in candidate order until need are taken (counts packed 16 + 16 bits).
  u32 n_above = 0, n_in = 0;
#pragma unroll
  for (int e = 0; e < ELEMS; e++) {
    if (cols.c[e] >= 0) {
      if ((keys[e] & pm) > pf) n_above += 1;
      if ((keys[e] & pm) == pf) n_in += 1;
    }
  }
  const u32 mine = n_above | n_in << 16;
  u32 incl = mine;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const u32 prev = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += prev;
  }
  if (lane == 31) sm.wtot[warp] = incl;
  __syncthreads();
  const u32 base =
      __reduce_add_sync(0xffffffffu, lane < warp ? sm.wtot[lane] : 0u) + incl -
      mine;
  int a_run = (int)(base & 0xffff);
  int b_run = (int)(base >> 16);
#pragma unroll
  for (int e = 0; e < ELEMS; e++) {
    const int col = cols.c[e];
    const int val = (maps[e] << LOG2_PAGE_ROWS) + (col & PAGE_MASK);
    if (col >= 0) {
      if ((keys[e] & pm) > pf) {
        sm.orow[a_run + min(b_run, need)] = val;
        a_run += 1;
      } else if ((keys[e] & pm) == pf) {
        if (b_run < need) sm.orow[a_run + b_run] = val;
        b_run += 1;
      }
    }
  }
  __syncthreads();
  const int kept = min(nvalid, TOP_K);
  const int at = tid * 4;
  if (at < TOP_K) {
    int4 out;
    out.x = at < kept ? sm.orow[at] : -1;
    out.y = at + 1 < kept ? sm.orow[at + 1] : -1;
    out.z = at + 2 < kept ? sm.orow[at + 2] : -1;
    out.w = at + 3 < kept ? sm.orow[at + 3] : -1;
    *reinterpret_cast<int4*>(dst + (long long)row * dst_stride + at) = out;
  }
}
