// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Per-step decode build of the die-local sparse-attention indexer, for one or
// several indexer cache groups in one launch.
//
// Per group it writes each decode row's pages split by locality domain into
// compact per-domain block tables (lbt0/lbt1), the per-domain context lengths
// (llen0/llen1), the join map, the per-domain local-to-row page maps
// (inv0/inv1), the 64-entry zero pads after each domain's list (valid ids for
// DeepGEMM's prefetch), the reset of rows num_rows..cap (lengths 0, 64
// zeros), and both domains' DeepGEMM paged-MQA-logits schedules
// ((sms + 1) x 2 int32: a port of DeepGEMM 2.8.1's
// sm100_paged_mqa_logits_metadata with SPLIT_KV = 384, BLOCK_Q = 4 and the
// domain's SM count). Other entries are not written.
//
// Grid (fixed per (group set, num_rows), so CUDA-graph safe), in order: for
// each group one "schedule" block followed by num_rows x nch "split" blocks
// (row-major; one chunk of kChunk pages of one row each); then each group's
// "reset" blocks (kResetRows rows each).
// * Split block: counts its chunk's domain-1 pages, publishes the count in a
//   64-bit flag tagged with the group's epoch, sums all earlier chunks'
//   counts of its row (one warp; they run concurrently), scatters. The row's
//   last chunk also writes the row's lengths and zero pads. No atomics.
// * Schedule block: reads every row's length, last page and that page's
//   domain itself, waits for every chunk flag of every row, derives both
//   domains' lengths from them (the split blocks' formula) and computes both
//   schedules; then it bumps the group's epoch: every block that read the
//   epoch has published its flag by then, so flags never need clearing and
//   graph replays work.
//
// Device state per group (zero-initialized once): state[0] = epoch. No
// standard headers: compiled at runtime with NVRTC.

typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;

#define kThreads 256
#define kPer 4
#define kChunk (kThreads * kPer)
#define kWarps (kThreads / 32)
#define kZPad 64
#define kResetRows 8
#define kSplitKV 384u
#define kBlockQ 4u
#define kMaxGroups 4

struct Group {
  const int* bt;           // [num_rows, bt_stride] block ids
  const int* lens;         // [num_rows] context rows per decode row
  const signed char* dom;  // [num_pages] page -> domain (0/1)
  const int* indices;  // [num_rows] request id per decode row (DeepGEMM varlen
                       // indices)
  int* lbt0;           // [cap, out_stride] die-0 compact block table
  int* lbt1;
  int* inv0;  // [cap, out_stride] die-d local page -> row page
  int* inv1;
  int* jmap;   // [cap, jmap_stride] row page -> (local_col << 1) | die
  int* llen0;  // [cap] die-0 context rows
  int* llen1;
  int* sched0;  // [(sms0 + 1) * 2] die-0 DeepGEMM schedule
  int* sched1;
  u64* flags;  // [cap * nch] chunk counts, tagged with epoch + 1
  u32* state;  // [4] epoch, unused x 3
  i64 bt_stride;
  i64 out_stride;
  i64 jmap_stride;
  i64 dom_q0;  // dom_k > 0: domain(b) = ((dom_q0 + b) / dom_k) & 1 (= the dom
               // table, checked on the host)
  int R;       // page rows
  int width;   // block-table width (pages)
  int cap;     // rows of the output buffers
  int num_rows;
  int nch;          // chunks per row in the grid = ceil(width / kChunk)
  int split_begin;  // the group's schedule block; its split blocks follow
  int reset_begin;  // reset blocks [reset_begin, reset_end)
  int reset_end;
  int dom_k;  // 0: read the dom table
  int pad0;
};

struct Params {
  Group g[kMaxGroups];
  int num_groups;
  int sms0;
  int sms1;
  int rows_max;  // max num_rows over the groups (dynamic smem = 16 * rows_max
                 // bytes)
};

static_assert(sizeof(Group) == 192, "Group layout");
static_assert(sizeof(Params) == 784, "Params layout");

static __device__ __forceinline__ u32 ld_relaxed_u32(const u32* p) {
  u32 v;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}
static __device__ __forceinline__ u64 ld_relaxed_u64(const u64* p) {
  u64 v;
  asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(p)
               : "memory");
  return v;
}
static __device__ __forceinline__ void st_relaxed_u64(u64* p, u64 v) {
  asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v)
               : "memory");
}
static __device__ __forceinline__ u32 ceil_div_u32(u32 a, u32 b) {
  return (a + b - 1) / b;
}
// DeepGEMM: KV/MMA cost scales with Q blocks; epilogue cost with valid tokens
static __device__ __forceinline__ u32 split_cost(u32 nq) {
  return 2u * kBlockQ * ceil_div_u32(nq, kBlockQ) + nq;
}

// The planned die of block b: arithmetic when the host verified it equals the
// table (no dependent load), else the table.
static __device__ __forceinline__ int page_dom(const Group& G, int b) {
  if (G.dom_k > 0) return (int)((((u32)G.dom_q0 + (u32)b) / (u32)G.dom_k) & 1u);
  return (int)G.dom[b];
}

// Waits until the flag carries `tag` (this launch) and returns its count.
static __device__ __forceinline__ u32 wait_flag(const u64* f, u32 tag) {
  u64 x = ld_relaxed_u64(f);
  u32 spins = 0;
  while ((u32)(x >> 32) != tag) {
    __nanosleep(20);
    x = ld_relaxed_u64(f);
    if (++spins == (1u << 26)) __trap();  // a hang would be a bug: fail loudly
  }
  return (u32)x;
}

// Block-wide exclusive scan (int); returns the thread's exclusive prefix,
// `total` = block sum.
static __device__ __forceinline__ int block_excl_scan(int v, int* s_warp,
                                                      int& total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) s_warp[warp] = x;
  __syncthreads();
  int before = 0, tot = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    const int s = s_warp[w];
    before += (w < warp) ? s : 0;
    tot += s;
  }
  __syncthreads();
  total = tot;
  return before + x - v;
}

// In-place inclusive scan of a[0, n) in shared memory, uint32 with wraparound
// (DeepGEMM's prefix arithmetic).
static __device__ void block_incl_scan_u32(u32* a, int n, u32* s_warp) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  u32 carry = 0;
  for (int t0 = 0; t0 < n; t0 += kThreads) {
    const int t = t0 + threadIdx.x;
    u32 x = t < n ? a[t] : 0u;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const u32 y = __shfl_up_sync(0xffffffffu, x, o);
      if (lane >= o) x += y;
    }
    if (lane == 31) s_warp[warp] = x;
    __syncthreads();
    u32 before = 0, tot = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const u32 s = s_warp[w];
      before += (w < warp) ? s : 0u;
      tot += s;
    }
    if (t < n) a[t] = carry + before + x;
    carry += tot;
    __syncthreads();
  }
}

// Both dies' DeepGEMM schedules from per-row die lens in shared memory
// (ln0/ln1, T rows). dsm: 2 * T u32 scratch.
static __device__ void schedules(const Group& G, int sms0, int sms1,
                                 const u32* ln0, const u32* ln1, u32* rs,
                                 u32* pf, int* s_warp) {
  const int tid = threadIdx.x;
  const int T = G.num_rows;
  // Request runs: token t starts a request if t == 0 or indices[t] != indices[t
  // - 1].
  int base = 0;
  for (int t0 = 0; t0 < T; t0 += kThreads) {
    const int t = t0 + tid;
    int is = 0;
    if (t < T) is = (t == 0) || (G.indices[t] != G.indices[t - 1]);
    int tot;
    const int ex = block_excl_scan(is, s_warp, tot);
    if (is) rs[base + ex] = (u32)t;
    base += tot;
  }
  const int nreq = base;
  __syncthreads();
  for (int d = 0; d < 2; ++d) {
    const u32* lens = d ? ln1 : ln0;
    int* sched = d ? G.sched1 : G.sched0;
    const u32 S = (u32)(d ? sms1 : sms0);
    int ovf = 0;
    for (int i = tid; i < nreq; i += kThreads) {
      const u32 e0 = (i + 1 < nreq) ? rs[i + 1] : (u32)T;
      const u32 nq = e0 - rs[i];
      const u64 cost =
          (u64)ceil_div_u32(lens[e0 - 1], kSplitKV) * (u64)split_cost(nq);
      pf[i] = (u32)cost;
      ovf |= cost > 0xffffffffull;
    }
    __syncthreads();
    block_incl_scan_u32(pf, nreq, (u32*)s_warp);
    for (int i = tid + 1; i < nreq; i += kThreads) ovf |= pf[i] < pf[i - 1];
    ovf = __syncthreads_or(ovf);
    if (ovf) {  // DeepGEMM's fallback: token cost
      for (int i = tid; i < nreq; i += kThreads) {
        const u32 e0 = (i + 1 < nreq) ? rs[i + 1] : (u32)T;
        pf[i] = ceil_div_u32(lens[e0 - 1], kSplitKV) * (e0 - rs[i]);
      }
      __syncthreads();
      block_incl_scan_u32(pf, nreq, (u32*)s_warp);
    }
    const u32 total = nreq > 0 ? pf[nreq - 1] : 0u;
    const u32 q = total / S, rem = total % S;
    for (u32 sm = (u32)tid; sm <= S; sm += kThreads) {
      const u32 target = sm * q + (sm < rem ? sm : rem);
      u32 lo = 0, hi = (u32)nreq;
      while (lo < hi) {  // first prefix > target
        const u32 mid = (lo + hi) / 2;
        if (pf[mid] <= target)
          lo = mid + 1;
        else
          hi = mid;
      }
      u32 qt = (u32)T, kv = 0;  // tail sentinel: one past the end
      if (lo < (u32)nreq) {
        const u32 before = lo == 0 ? 0u : pf[lo - 1];
        const u32 e0 = (lo + 1 < (u32)nreq) ? rs[lo + 1] : (u32)T;
        const u32 nq = e0 - rs[lo];
        qt = rs[lo];
        kv = (target - before) / (ovf ? nq : split_cost(nq));
      }
      sched[2 * sm] = (int)qt;
      sched[2 * sm + 1] = (int)kv;
    }
    __syncthreads();
  }
}

// Both dies' schedules for T <= 32 rows: warp d computes die d's, lane t
// holding row t (registers and shuffles only; the same arithmetic as
// schedules()). idx: indices[lane] (lanes < T), ln0/ln1: per-row die lens in
// shared memory.
static __device__ void schedules_warp(const Group& G, int sms0, int sms1,
                                      const u32* ln0, const u32* ln1, int idx) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp >= 2) return;
  const u32 full = 0xffffffffu;
  const int T = G.num_rows;
  const u32* lens = warp ? ln1 : ln0;
  int* sched = warp ? G.sched1 : G.sched0;
  const u32 S = (u32)(warp ? sms1 : sms0);
  const int prev = __shfl_up_sync(full, idx, 1);
  const bool is = lane < T && (lane == 0 || idx != prev);
  const u32 starts =
      __ballot_sync(full, is);  // bit t: token t starts a request
  const int nreq = __popc(starts);
  // lane i < nreq holds request i: its first token, token count, cost
  const bool live = lane < nreq;
  const u32 st = live ? (u32)(__fns(starts, 0, lane + 1)) : (u32)T;
  const u32 upto =
      (live && st < 31u) ? ((2u << st) - 1u) : 0xffffffffu;  // bits 0..st
  const u32 after = live ? (starts & ~upto) : 0u;  // starts after this one
  const u32 en = live ? (after ? (u32)(__ffs(after) - 1) : (u32)T) : (u32)T;
  const u32 nq = en - st;
  const u32 ctx = live ? lens[en - 1] : 0u;
  const u64 c64 =
      live ? (u64)ceil_div_u32(ctx, kSplitKV) * (u64)split_cost(nq) : 0ull;
  u32 pf = (u32)c64;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const u32 y = __shfl_up_sync(full, pf, o);
    if (lane >= o) pf += y;
  }
  const u32 pprev = __shfl_up_sync(full, pf, 1);
  const bool ovf = __any_sync(
      full, (live && c64 > 0xffffffffull) || (live && lane > 0 && pf < pprev));
  if (ovf) {  // DeepGEMM's fallback: token cost
    pf = live ? ceil_div_u32(ctx, kSplitKV) * nq : 0u;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const u32 y = __shfl_up_sync(full, pf, o);
      if (lane >= o) pf += y;
    }
  }
  const u32 total = nreq > 0 ? __shfl_sync(full, pf, nreq - 1) : 0u;
  const u32 q = total / S, rem = total % S;
  for (u32 b0 = 0; b0 <= S; b0 += 32) {
    const u32 sm = b0 + lane;
    const u32 target = sm * q + (sm < rem ? sm : rem);
    u32 lo = 0, hi = (u32)nreq;
#pragma unroll
    for (int it = 0; it < 6; ++it) {  // first prefix > target (nreq <= 32)
      const u32 mid = (lo + hi) / 2;
      const u32 v = __shfl_sync(full, pf, mid & 31);
      if (lo < hi) {
        if (v <= target)
          lo = mid + 1;
        else
          hi = mid;
      }
    }
    const u32 before = __shfl_sync(full, pf, (lo - 1) & 31);
    const u32 qst = __shfl_sync(full, st, lo & 31);
    const u32 qnq = __shfl_sync(full, nq, lo & 31);
    if (sm <= S) {
      u32 qt = (u32)T, kv = 0;  // tail sentinel: one past the end
      if (lo < (u32)nreq) {
        qt = qst;
        kv = (target - (lo == 0 ? 0u : before)) / (ovf ? qnq : split_cost(qnq));
      }
      sched[2 * sm] = (int)qt;
      sched[2 * sm + 1] = (int)kv;
    }
  }
}

// The group's schedule block: per-row die lens from the rows' lengths, last
// pages and chunk flags, then schedules.
static __device__ void schedule_block(const Group& G, int sms0, int sms1,
                                      u32* dsm, int* s_warp, int* s_misc) {
  const int tid = threadIdx.x;
  const int T = G.num_rows, R = G.R;
  u32* ln0 = dsm;
  u32* ln1 = dsm + T;
  u32* rs = dsm + 2 * T;
  u32* pf = dsm + 3 * T;
  u32* s_np = pf;  // [T] pages per row         (pf, rs, ln1 are free until the
                   // lens are known)
  u32* s_last = rs;  // [T] the row's last block id
  u32* s_n1 = ln1;   // [T] domain-1 pages per row (summed from the chunk flags)
  if (tid == 0) s_misc[4] = (int)(ld_relaxed_u32(G.state) + 1u);
  const int idx = (T <= 32 && (tid & 31) < T)
                      ? G.indices[tid & 31]
                      : 0;  // warps 0-1 use it (T <= 32 path)
  for (int r = tid; r < T; r += kThreads) {
    const int L = G.lens[r];
    const int npages = (L + R - 1) / R;
    s_np[r] = (u32)npages;
    s_n1[r] = 0;
    s_last[r] = npages > 0 ? (u32)G.bt[(i64)r * G.bt_stride + npages - 1] : 0u;
  }
  __syncthreads();
  const u32 tag = (u32)s_misc[4];
  // The last pages' domains: issued now, consumed after the flag waits (rows
  // tid, tid + kThreads, ...: first 4).
  int dl[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int r = tid + k * kThreads;
    dl[k] = (r < T && s_np[r] > 0) ? page_dom(G, (int)s_last[r]) : 0;
  }
  // Every chunk flag of every row, spread over the threads.
  const int nfl = T * G.nch;
  for (int i = tid; i < nfl; i += kThreads) {
    const int r = i / G.nch, c = i - (i / G.nch) * G.nch;
    if (c * kChunk < (int)s_np[r])
      atomicAdd(&s_n1[r], wait_flag(G.flags + i, tag));
  }
  __syncthreads();
  for (int r = tid, k = 0; r < T; r += kThreads, ++k) {
    const int d_last =
        k < 4 ? dl[k] : (s_np[r] > 0 ? page_dom(G, (int)s_last[r]) : 0);
    const int L = G.lens[r];
    const int npages = (int)s_np[r];
    const int n1 = (int)s_n1[r];
    const int n0 = npages - n1;
    const int sh = R * npages - L;  // rows missing from the last page
    ln0[r] = (u32)(n0 * R - ((npages > 0 && d_last == 0) ? sh : 0));
    ln1[r] = (u32)(n1 * R - ((npages > 0 && d_last != 0) ? sh : 0));
  }
  __syncthreads();
  // Every block that read the epoch has published (and every flag of this
  // launch has been read): next launch.
  if (tid == 0) atomicAdd(G.state, 1u);
  if (T <= 32)
    schedules_warp(G, sms0, sms1, ln0, ln1, idx);
  else
    schedules(G, sms0, sms1, ln0, ln1, rs, pf, s_warp);
}

static __device__ void split_block(const Group& G, int item, int* s_warp,
                                   int* s_misc) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r = item / G.nch, c = item - (item / G.nch) * G.nch;
  const int R = G.R;
  if (tid == 0) s_misc[4] = (int)(ld_relaxed_u32(G.state) + 1u);
  const i64 rbt = (i64)r * G.bt_stride;
  const int cbase = c * kChunk;
  const int pbase = cbase + tid * kPer;
  // The chunk's block ids, loaded before the row length arrives (entries past
  // the row's pages are in the row of the block table, so in bounds; they are
  // masked out below).
  int bk[kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k)
    bk[k] = (pbase + k < G.width) ? G.bt[rbt + pbase + k] : 0;
  const int L = G.lens[r];
  const int npages = (L + R - 1) / R;
  const int nchr = (npages + kChunk - 1) / kChunk;
  const int last_c = (nchr > 0 ? nchr : 1) - 1;
  if (c > last_c)
    return;  // past the row's pages: nothing to write (block-uniform)

  const i64 rout = (i64)r * G.out_stride;
  const i64 rj = (i64)r * G.jmap_stride;
  bool val[kPer], one[kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    val[k] = pbase + k < npages;
    bk[k] = val[k] ? bk[k] : 0;
  }
  int n1 = 0;
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    const int d = val[k] ? page_dom(G, bk[k]) : 0;
    one[k] = val[k] && d != 0;
    n1 += one[k] ? 1 : 0;
    if (pbase + k == npages - 1)
      s_misc[5] = d;  // the row's last page (only in its last chunk)
  }
  int chunk_n1;
  const int t1 = block_excl_scan(
      n1, s_warp, chunk_n1);  // its barriers also publish s_misc[4], s_misc[5]
  const u32 tag = (u32)s_misc[4];
  const u64* fl = G.flags + (i64)r * G.nch;
  // An empty row's block is not waited for, so it may read the epoch after
  // the schedule block bumped it: it must not publish (its tag would be the
  // next launch's).
  if (tid == 0 && npages > 0)
    st_relaxed_u64(G.flags + (i64)r * G.nch + c,
                   ((u64)tag << 32) | (u64)(u32)chunk_n1);
  // Look-back: the sum of all earlier chunks' counts (they run concurrently, so
  // they are ready about now).
  if (warp == 0) {
    int excl = 0;
    for (int j0 = 0; j0 < c; j0 += 32) {
      const int j = j0 + lane;
      u32 v = j < c ? wait_flag(fl + j, tag) : 0u;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      excl += (int)v;
    }
    if (lane == 0) s_misc[0] = excl;
  }
  __syncthreads();
  const int pre1 = s_misc[0];
  const int pre0 = cbase - pre1;  // earlier chunks are full
  const int nvalid = min(max(npages - cbase, 0), kChunk);
  const int valid_before = min(nvalid, tid * kPer);
  int p1 = pre1 + t1;
  int p0 = pre0 + (valid_before - t1);
  const int twoR = 2 * R;
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    if (!val[k]) continue;
    const int pp = pbase + k;
    if (one[k]) {
      G.lbt1[rout + p1] = bk[k];
      G.inv1[rout + p1] = pp;
      G.jmap[rj + pp] = p1 * twoR + 1;
      ++p1;
    } else {
      G.lbt0[rout + p0] = bk[k];
      G.inv0[rout + p0] = pp;
      G.jmap[rj + pp] = p0 * twoR;
      ++p0;
    }
  }
  if (c == last_c) {  // the row's last chunk: lens and zero pads
    const int n1r = pre1 + chunk_n1;
    const int n0r = npages - n1r;
    if (tid == 0) {
      const bool last_ok = npages > 0;
      const int d_last = last_ok ? s_misc[5] : 0;
      const int sh = R * npages - L;  // rows missing from the last page
      G.llen0[r] = n0r * R - ((last_ok && d_last == 0) ? sh : 0);
      G.llen1[r] = n1r * R - ((last_ok && d_last != 0) ? sh : 0);
    }
    for (int z = tid; z < 2 * kZPad; z += kThreads) {
      const int zz = z & (kZPad - 1);
      if (z < kZPad) {
        if (n0r + zz < G.width) G.lbt0[rout + n0r + zz] = 0;
      } else {
        if (n1r + zz < G.width) G.lbt1[rout + n1r + zz] = 0;
      }
    }
  }
}

static __device__ void reset_block(const Group& G, int item) {
  const int r0 = G.num_rows + item * kResetRows;
  const int r1 = min(r0 + kResetRows, G.cap);
  for (int r = r0; r < r1; ++r) {
    const i64 ro = (i64)r * G.out_stride;
    for (int z = threadIdx.x; z < 2 * kZPad + 2; z += kThreads) {
      if (z < kZPad) {
        if (z < G.width) G.lbt0[ro + z] = 0;
      } else if (z < 2 * kZPad) {
        if (z - kZPad < G.width) G.lbt1[ro + z - kZPad] = 0;
      } else if (z == 2 * kZPad) {
        G.llen0[r] = 0;
      } else {
        G.llen1[r] = 0;
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(kThreads)
    indexer_localize_build(const __grid_constant__ Params p) {
  extern __shared__ u32 dsm[];
  __shared__ int s_warp[kWarps];
  __shared__ int s_misc[8];
  const int b = blockIdx.x;
  for (int gi = 0; gi < p.num_groups; ++gi) {
    const Group& G = p.g[gi];
    if (b == G.split_begin) {
      schedule_block(G, p.sms0, p.sms1, dsm, s_warp, s_misc);
      return;
    }
    if (b > G.split_begin && b <= G.split_begin + G.num_rows * G.nch) {
      split_block(G, b - G.split_begin - 1, s_warp, s_misc);
      return;
    }
    if (b >= G.reset_begin && b < G.reset_end) {
      reset_block(G, b - G.reset_begin);
      return;
    }
  }
}
