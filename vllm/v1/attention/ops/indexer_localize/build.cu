// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Per-step decode build of the die-local sparse-attention indexer, for one or
// several indexer cache groups in one launch.
//
// Per group it writes each decode row's request pages split by locality domain
// into compact per-domain block tables (lbt0/lbt1) followed by 64 zeros (valid
// ids for DeepGEMM's prefetch), the row's per-domain context lengths
// (llen0/llen1), the join map ((local page << 1) | domain per request page),
// the per-domain local-to-request page maps (inv0/inv1), the reset of rows
// num_rows..cap (lengths 0, 64 zeros), and both domains' DeepGEMM
// paged-MQA-logits schedules ((sms + 1) x 2 int32: a port of DeepGEMM 2.8.1's
// sm100_paged_mqa_logits_metadata with SPLIT_KV = 384, BLOCK_Q = 4 and the
// domain's SM count). Other entries are not written.
//
// Requests (DeepGEMM's varlen form): consecutive rows with the same index are
// one request, whose rows see one more key each (speculative decoding).
// DeepGEMM reads the page list of a request's first row and scans as far as
// its last row needs, so every row of a request gets the request's lists, cut
// at its last row's length (the split length); each row keeps its own
// per-domain lengths (its pages are a prefix of the request's).
//
// Grid (fixed per (group set, num_rows), so CUDA-graph safe), in order: for
// each group one "schedule" block followed by num_rows x nch "split" blocks
// (row-major; one chunk of kChunk pages of one row each); then each group's
// "reset" blocks (kResetRows rows each).
// * Split block: counts its chunk's domain-1 pages, publishes the count in a
//   64-bit flag tagged with the group's epoch, sums all earlier chunks'
//   counts of its row (one warp; they run concurrently), scatters. The block
//   holding the row's own last page writes the row's lengths; the request's
//   last chunk writes the zero pads. No atomics.
// * Schedule block: reads every row's split length, last page and that
//   page's domain itself, waits for every chunk flag of every row, derives
//   both domains' request lengths from them (the split blocks' formula) and
//   computes both schedules (DeepGEMM prices a request by its last row);
//   then it bumps the group's epoch: every block that read the epoch has
//   published its flag by then, so flags never need clearing and graph
//   replays work.
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
  const int* indices;      // [num_rows] request id per decode row
  int* lbt0;               // [cap, out_stride] die-0 compact block table
  int* lbt1;
  int* inv0;  // [cap, out_stride] die-d local page -> request page
  int* inv1;
  int* jmap;   // [cap, jmap_stride] request page -> (local page << 1) | die
  int* llen0;  // [cap] die-0 context rows
  int* llen1;
  int* sched0;  // [(sms0 + 1) * 2] die-0 DeepGEMM schedule
  int* sched1;
  u64* flags;  // [cap * nch] chunk counts, tagged with epoch + 1
  u32* state;  // [4] epoch, unused x 3
  i64 bt_stride;
  i64 out_stride;
  i64 jmap_stride;
  i64 dom_q0;  // dom_k > 0: domain(b) = ((dom_q0 + b) / dom_k) & 1
  int R;       // page rows
  int width;   // block-table width (pages)
  int cap;     // rows of the output buffers
  int num_rows;
  int nch;          // chunks per row in the grid = ceil(width / kChunk)
  int split_begin;  // the group's schedule block; its split blocks follow
  int reset_begin;  // reset blocks [reset_begin, reset_end)
  int reset_end;
  int dom_k;  // 0: read the dom table
};

struct Params {
  Group g[kMaxGroups];
  int num_groups;
  int sms0;
  int sms1;
};

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

// The length of the last row of row r's request (rows r, r + 1, ... with r's
// index): the request's split length. The calling warp reads the next 32 rows'
// indices and lengths at a time.
static __device__ __forceinline__ int request_split_len(const Group& G, int r) {
  const int lane = threadIdx.x & 31;
  const int id = G.indices[r];
  int len = G.lens[r];
  for (int e = r;; e += 32) {
    const int t = e + 1 + lane;
    const bool in = t < G.num_rows;
    const int t_id = in ? G.indices[t] : 0;
    const int t_len = in ? G.lens[t] : 0;
    const u32 same = __ballot_sync(0xffffffffu, in && t_id == id);
    // Rows e + 1 .. e + run continue the request.
    const int run = same == 0xffffffffu ? 32 : __ffs(~same) - 1;
    if (run > 0) len = __shfl_sync(0xffffffffu, t_len, run - 1);
    if (run < 32) return len;
  }
}

// Waits until the flag carries `tag` (this launch) and returns its count.
static __device__ __forceinline__ u32 wait_flag(const u64* f, u32 tag) {
  u64 x = ld_relaxed_u64(f);
  while ((u32)(x >> 32) != tag) {
    __nanosleep(20);
    x = ld_relaxed_u64(f);
  }
  return (u32)x;
}

// Block-wide exclusive scan; returns the thread's exclusive prefix, `total` =
// block sum. T is int, or u32 for DeepGEMM's wrapping prefix arithmetic.
template <typename T>
static __device__ __forceinline__ T block_excl_scan(T v, T* s_warp, T& total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  T x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const T y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) s_warp[warp] = x;
  __syncthreads();
  T before = 0, tot = 0;
#pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    const T s = s_warp[w];
    before += (w < warp) ? s : (T)0;
    tot += s;
  }
  __syncthreads();
  total = tot;
  return before + x - v;
}

// In-place inclusive scan of a[0, n) in shared memory.
static __device__ void block_incl_scan(u32* a, int n, u32* s_warp) {
  u32 carry = 0;
  for (int t0 = 0; t0 < n; t0 += kThreads) {
    const int t = t0 + threadIdx.x;
    const u32 v = t < n ? a[t] : 0u;
    u32 tot;
    const u32 before = block_excl_scan(v, s_warp, tot);
    if (t < n) a[t] = carry + before + v;
    carry += tot;
  }
  __syncthreads();
}

// Both dies' DeepGEMM schedules from per-row die lens in shared memory
// (ln0/ln1, T rows). rs, pf: T u32 of shared scratch each (request starts,
// cost prefix sums). DeepGEMM falls back to token costs when the cost sum
// passes 32 bits; it stays far below here (at most one row per SM of a domain,
// each costing at most 9 x ceil(len / 384)), so that fallback is not ported.
static __device__ void schedules(const Group& G, int sms0, int sms1,
                                 const u32* ln0, const u32* ln1, u32* rs,
                                 u32* pf, u32* s_warp) {
  const int tid = threadIdx.x;
  const int T = G.num_rows;
  // Request runs: token t starts one if t == 0 or indices[t] != indices[t - 1].
  u32 nreq = 0;
  for (int t0 = 0; t0 < T; t0 += kThreads) {
    const int t = t0 + tid;
    const u32 is = t < T && (t == 0 || G.indices[t] != G.indices[t - 1]);
    u32 tot;
    const u32 ex = block_excl_scan(is, s_warp, tot);
    if (is) rs[nreq + ex] = (u32)t;
    nreq += tot;
  }
  __syncthreads();
  for (int d = 0; d < 2; ++d) {
    const u32* lens = d ? ln1 : ln0;
    int* sched = d ? G.sched1 : G.sched0;
    const u32 S = (u32)(d ? sms1 : sms0);
    for (u32 i = tid; i < nreq; i += kThreads) {
      const u32 e0 = i + 1 < nreq ? rs[i + 1] : (u32)T;
      pf[i] = ceil_div_u32(lens[e0 - 1], kSplitKV) * split_cost(e0 - rs[i]);
    }
    __syncthreads();
    block_incl_scan(pf, nreq, s_warp);
    const u32 total = pf[nreq - 1];
    const u32 q = total / S, rem = total % S;
    for (u32 sm = tid; sm <= S; sm += kThreads) {
      const u32 target = sm * q + (sm < rem ? sm : rem);
      u32 lo = 0, hi = nreq;
      while (lo < hi) {  // first prefix > target
        const u32 mid = (lo + hi) / 2;
        if (pf[mid] <= target)
          lo = mid + 1;
        else
          hi = mid;
      }
      u32 qt = (u32)T, kv = 0;  // tail sentinel: one past the end
      if (lo < nreq) {
        const u32 before = lo == 0 ? 0u : pf[lo - 1];
        const u32 e0 = lo + 1 < nreq ? rs[lo + 1] : (u32)T;
        qt = rs[lo];
        kv = (target - before) / split_cost(e0 - rs[lo]);
      }
      sched[2 * sm] = (int)qt;
      sched[2 * sm + 1] = (int)kv;
    }
    __syncthreads();
  }
}

// The group's schedule block: per-row die lens of the rows' requests from the
// split lengths, last pages and chunk flags, then schedules (which read the
// lens of each request's last row only).
static __device__ void schedule_block(const Group& G, int sms0, int sms1,
                                      u32* dsm, u32* s_warp, int* s_misc) {
  const int tid = threadIdx.x;
  const int T = G.num_rows, R = G.R;
  u32* ln0 = dsm;
  u32* ln1 = dsm + T;
  u32* rs = dsm + 2 * T;
  u32* pf = dsm + 3 * T;
  // Until the lens are known: (ln0, ln1, rs and pf are free)
  u32* s_split = ln0;  // [T] split length (the request's last row's length)
  u32* s_n1 = ln1;     // [T] domain-1 pages of the split (from the flags)
  u32* s_d1 = rs;      // [T] the domain of the split's last page
  u32* s_id = pf;      // [T] request index, then
  u32* s_np = pf;      // [T] pages of the split
  if (tid == 0) s_misc[4] = (int)(ld_relaxed_u32(G.state) + 1u);
  for (int r = tid; r < T; r += kThreads) s_id[r] = (u32)G.indices[r];
  __syncthreads();
  for (int r = tid; r < T; r += kThreads) {
    int e = r;
    while (e + 1 < T && s_id[e + 1] == s_id[r]) ++e;
    s_split[r] = (u32)G.lens[e];
  }
  __syncthreads();
  for (int r = tid; r < T; r += kThreads) {
    const int npages = ((int)s_split[r] + R - 1) / R;
    s_np[r] = (u32)npages;
    s_n1[r] = 0;
    s_d1[r] = npages > 0 ? page_dom(G, G.bt[r * G.bt_stride + npages - 1]) : 0;
  }
  __syncthreads();
  const u32 tag = (u32)s_misc[4];
  // Every chunk flag of every row, spread over the threads.
  for (int i = tid; i < T * G.nch; i += kThreads) {
    const int r = i / G.nch, c = i % G.nch;
    if (c * kChunk < (int)s_np[r])
      atomicAdd(&s_n1[r], wait_flag(G.flags + i, tag));
  }
  __syncthreads();
  for (int r = tid; r < T; r += kThreads) {
    const int npages = (int)s_np[r], n1 = (int)s_n1[r];
    const int sh = R * npages - (int)s_split[r];  // rows the last page lacks
    ln0[r] = (u32)((npages - n1) * R - (s_d1[r] ? 0 : sh));
    ln1[r] = (u32)(n1 * R - (s_d1[r] ? sh : 0));
  }
  __syncthreads();
  // Every block that read the epoch has published (and every flag of this
  // launch has been read): next launch.
  if (tid == 0) atomicAdd(G.state, 1u);
  schedules(G, sms0, sms1, ln0, ln1, rs, pf, s_warp);
}

static __device__ void split_block(const Group& G, int item, int* s_warp,
                                   int* s_misc) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r = item / G.nch, c = item % G.nch;
  const int R = G.R;
  if (tid == 0) s_misc[4] = (int)(ld_relaxed_u32(G.state) + 1u);
  const i64 rbt = (i64)r * G.bt_stride;
  const int cbase = c * kChunk;
  const int pbase = cbase + tid * kPer;
  // The chunk's block ids, loaded before the lengths arrive (entries past the
  // request's pages are in the row of the block table, so in bounds; they are
  // masked out below).
  int bk[kPer];
#pragma unroll
  for (int k = 0; k < kPer; ++k)
    bk[k] = (pbase + k < G.width) ? G.bt[rbt + pbase + k] : 0;
  // The request's pages up to its last row (the split), and this row's own
  // length (a prefix of them).
  const int L = request_split_len(G, r);
  const int L_row = G.lens[r];
  const int npages = (L + R - 1) / R;
  const int np_row = (L_row + R - 1) / R;
  const int nchr = (npages + kChunk - 1) / kChunk;
  const int last_c = (nchr > 0 ? nchr : 1) - 1;
  // Past the request's pages: nothing to write (block-uniform).
  if (c > last_c) return;

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
  }
  int chunk_n1;
  // Its barriers publish s_misc[4].
  const int t1 = block_excl_scan(n1, s_warp, chunk_n1);
  const u32 tag = (u32)s_misc[4];
  const u64* fl = G.flags + (i64)r * G.nch;
  // An empty request's block is not waited for, so it may read the epoch after
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
#pragma unroll
  for (int k = 0; k < kPer; ++k) {
    if (!val[k]) continue;
    const int pp = pbase + k;
    const int p = one[k] ? p1++ : p0++;
    (one[k] ? G.lbt1 : G.lbt0)[rout + p] = bk[k];
    (one[k] ? G.inv1 : G.inv0)[rout + p] = pp;
    G.jmap[rj + pp] = 2 * p + one[k];
    if (pp == np_row - 1) {  // the row's own last page: its lengths
      const int sh = R * np_row - L_row;  // rows missing from that page
      G.llen0[r] = p0 * R - (one[k] ? 0 : sh);
      G.llen1[r] = p1 * R - (one[k] ? sh : 0);
    }
  }
  if (np_row == 0 && c == 0 && tid == 0) {
    G.llen0[r] = 0;
    G.llen1[r] = 0;
  }
  if (c == last_c) {  // the request's last chunk: zero pads
    const int n1r = pre1 + chunk_n1;
    const int n0r = npages - n1r;
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
  __shared__ u32 s_warp[kWarps];
  __shared__ int s_misc[8];
  const int b = blockIdx.x;
  for (int gi = 0; gi < p.num_groups; ++gi) {
    const Group& G = p.g[gi];
    if (b == G.split_begin) {
      schedule_block(G, p.sms0, p.sms1, dsm, s_warp, s_misc);
      return;
    }
    if (b > G.split_begin && b <= G.split_begin + G.num_rows * G.nch) {
      split_block(G, b - G.split_begin - 1, (int*)s_warp, s_misc);
      return;
    }
    if (b >= G.reset_begin && b < G.reset_end) {
      reset_block(G, b - G.reset_begin);
      return;
    }
  }
}
