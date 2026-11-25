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
//   SCORE_F32       1: FP32 scores, 0: BF16 scores
//   LOG2_PAGE_ROWS  log2 of the indexer cache page rows (6 or 7)
// The dynamic shared memory size is SMEM_TOTAL below.

typedef unsigned short uint16_t;

#define ELEMS (2 * TOP_K / THREADS)
#define HALF (THREADS / 2)
#define PAGE_MASK ((1 << LOG2_PAGE_ROWS) - 1)
#if SCORE_F32
typedef float score_t;
  // Order-preserving keys use all 32 bits of an FP32 score.
  #define KEY_LSB 0
  #define HIST_WORDS 1024
#else
// A BF16 score; read as its 16 raw bits.
struct __align__(2) score_t {
  unsigned short bits;
};
// Order-preserving keys of BF16 scores live in the upper 16 bits.
  #define KEY_LSB 16
  #define HIST_WORDS 512
#endif
#define HIST_ZERO_ITERS ((HIST_WORDS + THREADS - 1) / THREADS)
#if TOP_K > 4 * THREADS
  #error "each thread stores at most 4 outputs"
#endif

#define SMEM_RNG_OFF (4 * HIST_WORDS)
#define SMEM_CTL_OFF (SMEM_RNG_OFF + 16)
#define SMEM_WTOT_OFF (SMEM_CTL_OFF + 16)
#define SMEM_OROW_OFF (SMEM_WTOT_OFF + 4 * (THREADS / 32))
#define SMEM_TOTAL (((SMEM_OROW_OFF + 4 * TOP_K + 127) / 128) * 128)

__device__ __forceinline__ int make_warp_uniform(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
               : "=r"(result)
               : "r"(x));
  return result;
}

extern "C" __global__ __launch_bounds__(THREADS) void indexer_localize_merge(
    score_t* __restrict__ s0, score_t* __restrict__ s1, int s_stride,
    int* __restrict__ c0, int* __restrict__ c1, int c_stride,
    int* __restrict__ m0, int* __restrict__ m1, int m_stride,
    int* __restrict__ dst, int dst_stride) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  extern __shared__ __align__(1024) char smem_raw[];
  unsigned int* hist = reinterpret_cast<unsigned int*>(smem_raw);
  unsigned int* rng = reinterpret_cast<unsigned int*>(smem_raw + SMEM_RNG_OFF);
  int* ctl = reinterpret_cast<int*>(smem_raw + SMEM_CTL_OFF);
  unsigned int* wtot =
      reinterpret_cast<unsigned int*>(smem_raw + SMEM_WTOT_OFF);
  int* orow = reinterpret_cast<int*>(smem_raw + SMEM_OROW_OFF);

  int row = blockIdx.x;
  // The first half of the CTA loads domain 0's candidates, the second half
  // domain 1's.
  int die = ((tid >= HALF) ? 1 : 0);
  int first = (tid - die * HALF) * ELEMS;
#pragma unroll
  for (int z = 0; z < HIST_ZERO_ITERS; z++) {
    if (z * THREADS + tid < HIST_WORDS) hist[z * THREADS + tid] = 0;
  }
  if (tid == 0) {
    rng[0] = 0xffffffffu;
    rng[1] = 0;
    rng[2] = 0;
  }
  int* c_row = ((die != 0) ? c1 : c0) +
               ((long long)row * (long long)c_stride + (long long)first);
  int cols[ELEMS];
#if ELEMS == 4
  const int4 c4 = *reinterpret_cast<const int4*>(c_row);
  cols[0] = c4.x;
  cols[1] = c4.y;
  cols[2] = c4.z;
  cols[3] = c4.w;
#elif ELEMS == 2
  const int2 c2 = *reinterpret_cast<const int2*>(c_row);
  cols[0] = c2.x;
  cols[1] = c2.y;
#else
  #error "unsupported TOP_K / THREADS"
#endif
  score_t* s_row =
      ((die != 0) ? s1 : s0) + ((long long)row * (long long)s_stride);
  int* m_row = ((die != 0) ? m1 : m0) + ((long long)row * (long long)m_stride);
  unsigned int keys[ELEMS];
  int maps[ELEMS];
  __syncthreads();
#pragma unroll
  for (int e = 0; e < ELEMS; e++) {
    int c = max(cols[e], 0);
#if SCORE_F32
    keys[e] = reinterpret_cast<unsigned int*>(s_row)[c];
#else
    keys[e] = (unsigned int)reinterpret_cast<uint16_t*>(s_row)[c];
#endif
  }
#pragma unroll
  for (int e = 0; e < ELEMS; e++) {
    int c = max(cols[e], 0);
    maps[e] = m_row[c >> LOG2_PAGE_ROWS];
  }
  // Order-preserving unsigned keys; the range of this CTA's valid keys.
  unsigned int k_lo = 0xffffffffu;
  unsigned int k_hi = 0;
  int n_ok = 0;
#pragma unroll
  for (int e = 0; e < ELEMS; e++) {
#if SCORE_F32
    keys[e] = keys[e] ^ ((unsigned int)((int)keys[e] >> 31) | 0x80000000u);
#else
    unsigned int b16 = keys[e] & 0xffff;
    keys[e] = (b16 ^ (0x8000 | -(b16 >> 15) & 0x7fff)) << 16;
#endif
    if (cols[e] >= 0) {
      k_lo = min(k_lo, keys[e]);
      k_hi = max(k_hi, keys[e]);
      n_ok += 1;
    }
  }
  unsigned int w_lo;
  asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;"
               : "=r"(w_lo)
               : "r"(k_lo));
  unsigned int w_hi;
  asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;"
               : "=r"(w_hi)
               : "r"(k_hi));
  int w_ok;
  asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;"
               : "=r"(w_ok)
               : "r"(n_ok));
  if (lane == 0) {
    atomicMin(&rng[0], w_lo);
    atomicMax(&rng[1], w_hi);
    atomicAdd(&rng[2], (unsigned int)w_ok);
  }
  __syncthreads();
  unsigned int lo = rng[0];
  unsigned int hi = rng[1];
  int nvalid = (int)rng[2];
  // Radix select of the TOP_K-th largest key over the bits below the common
  // prefix of [lo, hi], 8 bits per pass: pf/pm are the selected prefix and
  // its mask, need the keys still to take within the prefix.
  int top = 31 - __clz(lo ^ hi);
  unsigned int pm = 0xffffffffu;
  if (top >= 0) pm = 0xfffffffeu << (unsigned int)top;
  unsigned int pf = lo & pm;
  int need = TOP_K;
  int n_pass = (top - KEY_LSB + 8) / 8;
  if (nvalid <= TOP_K) {
    pm = 0;
    pf = 0;
    need = nvalid;
    n_pass = 0;
  }
  if (top < 0) n_pass = 0;
  for (int p = 0; p < n_pass; p++) {
    int lsb = max(top - 8 * p - 7, KEY_LSB);
    unsigned int sh = (unsigned int)lsb;
    unsigned int dmask = 255 >> (unsigned int)(8 - (top - 8 * p - lsb + 1));
    int hb = p * 256;
    unsigned int lead = __shfl_sync(0xFFFFFFFF, keys[0] >> sh & dmask, 0);
    int n_lead = 0;
#pragma unroll
    for (int e = 0; e < ELEMS; e++) {
      if (cols[e] >= 0 && (keys[e] & pm) == pf) {
        unsigned int dg = keys[e] >> sh & dmask;
        if (dg == lead) {
          n_lead += 1;
        } else {
          atomicAdd(&hist[hb + (int)dg], 1);
        }
      }
    }
    int n_lead_w;
    asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;"
                 : "=r"(n_lead_w)
                 : "r"(n_lead));
    if (lane == 0 && n_lead_w > 0) {
      atomicAdd(&hist[hb + (int)lead], (unsigned int)n_lead_w);
    }
    __syncthreads();
    if (warp == 0) {
      // Warp 0 finds the digit holding the need-th largest key: lane l owns
      // bins 8l..8l+7, suffix sums run from the top digit down.
      unsigned int bins[8];
      unsigned int own = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        bins[i] = hist[hb + lane * 8 + i];
        own += bins[i];
      }
      unsigned int suf = own;
      unsigned int up_0 = __shfl_down_sync(0xFFFFFFFF, suf, 1, 32);
      if (lane + 1 < 32) suf += up_0;
      unsigned int up_1 = __shfl_down_sync(0xFFFFFFFF, suf, 2, 32);
      if (lane + 2 < 32) suf += up_1;
      unsigned int up_2 = __shfl_down_sync(0xFFFFFFFF, suf, 4, 32);
      if (lane + 4 < 32) suf += up_2;
      unsigned int up_3 = __shfl_down_sync(0xFFFFFFFF, suf, 8, 32);
      if (lane + 8 < 32) suf += up_3;
      unsigned int up_4 = __shfl_down_sync(0xFFFFFFFF, suf, 16, 32);
      if (lane + 16 < 32) suf += up_4;
      unsigned int hit = __ballot_sync(0xFFFFFFFF, (int)suf >= need);
      unsigned int run = suf - own;
      if (lane == 31 - __clz(hit)) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
          int j = 7 - i;
          if ((int)run < need && (int)(run + bins[j]) >= need) {
            ctl[0] = lane * 8 + j;
            ctl[1] = (int)run;
            ctl[2] = (int)bins[j];
          }
          run += bins[j];
        }
      }
    }
    __syncthreads();
    need = need - ctl[1];
    pf = pf | (unsigned int)ctl[0] << sh;
    pm = pm | dmask << sh;
    if (ctl[2] == need) break;
  }
  // Output slots: keys above the selected prefix first, then keys within it
  // in candidate order until need are taken (counts packed 16 + 16 bits).
  unsigned int n_above = 0;
  unsigned int n_in = 0;
#pragma unroll
  for (int e = 0; e < ELEMS; e++) {
    if (cols[e] >= 0) {
      if ((keys[e] & pm) > pf) n_above += 1;
      if ((keys[e] & pm) == pf) n_in += 1;
    }
  }
  unsigned int mine = n_above | n_in << 16;
  unsigned int incl = mine;
  unsigned int prev_0 = __shfl_up_sync(0xFFFFFFFF, incl, 1, 32);
  if (lane >= 1) incl += prev_0;
  unsigned int prev_1 = __shfl_up_sync(0xFFFFFFFF, incl, 2, 32);
  if (lane >= 2) incl += prev_1;
  unsigned int prev_2 = __shfl_up_sync(0xFFFFFFFF, incl, 4, 32);
  if (lane >= 4) incl += prev_2;
  unsigned int prev_3 = __shfl_up_sync(0xFFFFFFFF, incl, 8, 32);
  if (lane >= 8) incl += prev_3;
  unsigned int prev_4 = __shfl_up_sync(0xFFFFFFFF, incl, 16, 32);
  if (lane >= 16) incl += prev_4;
  if (lane == 31) wtot[warp] = incl;
  __syncthreads();
  unsigned int w_prev = 0;
  if (lane < warp) w_prev = wtot[lane];
  unsigned int w_base;
  asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;"
               : "=r"(w_base)
               : "r"(w_prev));
  unsigned int base = w_base + incl - mine;
  int a_run = (int)(base & 0xffff);
  int b_run = (int)(base >> 16);
#pragma unroll
  for (int e = 0; e < ELEMS; e++) {
    int col = cols[e];
    int val = (maps[e] << LOG2_PAGE_ROWS) + (col & PAGE_MASK);
    if (col >= 0) {
      if ((keys[e] & pm) > pf) {
        orow[a_run + min(b_run, need)] = val;
        a_run += 1;
      } else if ((keys[e] & pm) == pf) {
        if (b_run < need) orow[a_run + b_run] = val;
        b_run += 1;
      }
    }
  }
  __syncthreads();
  int kept = min(nvalid, TOP_K);
  int* d_row = dst + ((long long)row * (long long)dst_stride);
  int at = tid * 4;
  if (at < TOP_K) {
    int words[4];
#pragma unroll
    for (int i = 0; i < 4; i++) words[i] = at + i < kept ? orow[at + i] : -1;
    __builtin_assume(__isGlobal(d_row));
    *reinterpret_cast<int4*>(d_row + at) =
        make_int4(words[0], words[1], words[2], words[3]);
  }
}
