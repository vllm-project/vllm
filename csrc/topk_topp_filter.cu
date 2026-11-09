// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// In-place top-k/top-p masking of FP32 logits with per-row device k and p
// (SM90+: TMA bulk copies, thread block clusters).
//
// Each row is handled by one CTA (large batches, persistent over rows) or by a
// thread block cluster of up to 16 CTAs that split the row (small batches):
//   1. stream the row chunk once from HBM through a TMA ring in shared memory:
//      optional sum-exp, and gather candidates above a sampled threshold into
//      a store of shared memory plus a per-CTA overflow region in global
//      memory (L2). On overflow a CTA raises its threshold exactly. When a
//      top-p nucleus extends below the candidates, a second pass gathers down
//      to a threshold sized from the sample's shape and the exact missing mass.
//   2. exact radix selection over the candidates only, with histograms and
//      scalars combined across the cluster: top-k rank, then top-p fixed-point
//      mass, ties broken by lowest index.
//   3. write -inf over the chunk, then scatter the kept candidates back.
// Nuclei beyond the store use a value-space window pass; rows the candidates
// cannot handle at all use an exact multi-pass radix selection over the row.
//
// Semantics: exactly k tokens by logit, ties broken by lowest index (like the
// Triton kernel, while apply_top_k_top_p_pytorch keeps every tie at the k-th
// logit), then top-p on the renormalized top-k distribution, keeping the
// smallest prefix of the (logit desc, index asc) order whose mass reaches p.
// Masses are FP32 exp() rounded to 2^-44 fixed point, so sums are
// deterministic.

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <mutex>

#include "torch_utils.h"

namespace vllm {
namespace topk_topp_filter {

namespace cg = cooperative_groups;
using U64 = unsigned long long;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int STAGE = 4096;                 // elements per streaming round
constexpr int SVECS = STAGE / 4 / THREADS;  // float4 per thread per round
constexpr int MAX_STAGES = 10;
constexpr int MAX_SPLITS = 16;
constexpr unsigned CAP = 6912;  // candidates in shared memory per CTA
constexpr unsigned GCAP =
    32768;  // overflow candidates in global memory per CTA
constexpr unsigned STORE = CAP + GCAP;
constexpr unsigned MAX_CAND_K = STORE * 5 / 8;
constexpr unsigned TARGET_P =
    CAP * 3 / 4;  // minimum expected candidates, top-p only
constexpr unsigned KEEP_P =
    STORE * 5 / 8;  // retained after compaction, top-p only
constexpr unsigned KEY_NEGINF = 0x007fffffu;
constexpr unsigned NO_CANDIDATES = 0xffffffffu;
constexpr int MAX_VOCAB = 1 << 20;
constexpr float MASS_SCALE = 17592186044416.f;  // 2^44
// A row mass is a sum of at most MAX_VOCAB terms of at most MASS_SCALE.
static_assert(double(MASS_SCALE) * MAX_VOCAB <= 18446744073709551616.0);
constexpr int OK = 0, GLOBAL = 1;
constexpr int SBINS = 256;       // sample histogram bins
constexpr float SB_NATS = 32.f;  // sample histogram range below the sample max
enum Weight {
  COUNT,
  LIMBS,
  WIDE
};  // LIMBS: 3x16-bit u32 limbs, at most 2^16 per bin

struct Slot {
  U64 v[3];
};

struct Shared {
  unsigned key[CAP];
  unsigned idx[CAP];
  union {
    U64 wide[256];
    unsigned limb[3][256];
  };
  unsigned count[256];
  U64 gwide[256];  // cluster-combined weights
  unsigned gcount[256];
  unsigned shist[SBINS];  // sample counts below shist_top, top-p-only rows
  unsigned ghist[SBINS];  // shist summed over the cluster
  float shist_top;
  U64 bar[MAX_STAGES];
  U64 scratch[WARPS];
  unsigned wsum[WARPS + 1];
  Slot xin[2][MAX_SPLITS];  // group_gather inputs pushed by each CTA
  U64 xres;
  int xturn;
  unsigned total;
  unsigned prefix, count_at;
  U64 above;
  int found;
};
constexpr int RING_OFFSET = (sizeof(Shared) + 127) / 128 * 128;
constexpr int SHALLOW = 3;  // ring stages when two CTAs share an SM
// Two CTAs with shallow rings, plus 1 KiB reserved each, fit in 228 KiB.
static_assert(2 * (RING_OFFSET + SHALLOW * STAGE * 4 + 1024) <= 228 * 1024);

// Candidate store: the first CAP entries in shared memory, the rest in this
// CTA's global overflow region.
// Dynamic shared memory: Shared, then the TMA ring. Addressing it through the
// symbol keeps shared addresses out of registers.
extern __shared__ __align__(128) unsigned char smem_raw[];

__device__ __forceinline__ Shared& shared_state() {
  return *reinterpret_cast<Shared*>(smem_raw);
}

__device__ __forceinline__ float4* ring_stage(int slot) {
  return reinterpret_cast<float4*>(smem_raw + RING_OFFSET) + slot * (STAGE / 4);
}

// Overflow region: GCAP keys then GCAP indices.
struct Store {
  unsigned* g;
  __device__ unsigned& key(unsigned i) const {
    return i < CAP ? shared_state().key[i] : g[i - CAP];
  }
  __device__ unsigned& idx(unsigned i) const {
    return i < CAP ? shared_state().idx[i] : g[GCAP + i - CAP];
  }
};

// Reductions across the CTAs that share a row.
template <bool CL>
struct Group;
template <>
struct Group<false> {
  __device__ void sync() const { __syncthreads(); }
  __device__ int size() const { return 1; }
  template <class T>
  __device__ T* peer(T* p, int) const {
    return p;
  }
};
template <>
struct Group<true> {
  __device__ void sync() const { cg::this_cluster().sync(); }
  __device__ int size() const { return int(cg::this_cluster().num_blocks()); }
  __device__ int rank() const { return int(cg::this_cluster().block_rank()); }
  template <class T>
  __device__ T* peer(T* p, int r) const {
    return cg::this_cluster().map_shared_rank(p, r);
  }
  // Split barrier: remote writes issued after wait() see every CTA's shared
  // memory writes made before its arrive().
  __device__ void arrive() const {
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  }
  __device__ void wait() const {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  }
  // Fire-and-forget add into CTA r's copy of p (generic atomics would wait).
  __device__ void add(unsigned* p, int r, unsigned v) const {
    asm volatile("red.relaxed.cluster.shared::cluster.add.u32 [%0], %1;" ::"r"(
                     remote(p, r)),
                 "r"(v)
                 : "memory");
  }
  __device__ void add(U64* p, int r, U64 v) const {
    asm volatile("red.relaxed.cluster.shared::cluster.add.u64 [%0], %1;" ::"r"(
                     remote(p, r)),
                 "l"(v)
                 : "memory");
  }
  __device__ unsigned remote(const void* p, int r) const {
    unsigned addr;
    asm("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(addr)
        : "r"(unsigned(__cvta_generic_to_shared(p))), "r"(r));
    return addr;
  }
};

struct Pick {
  unsigned key;
  U64 above;
  unsigned count_at;
  bool found;
};

// Keep (key, idx) iff it ranks at or above the cut in (key desc, idx asc).
struct Cut {
  unsigned key, idx;
  __device__ bool keep(unsigned k, unsigned i) const {
    return k > key || (k == key && i <= idx);
  }
};
constexpr Cut ALL{0u, 0xffffffffu};

__device__ __forceinline__ unsigned ordered(float x) {
  unsigned bits = __float_as_uint(x == 0.f ? 0.f : x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ float from_ordered(unsigned key) {
  return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}

// Clamps before the conversion: a forced logit (1e9) puts d far outside int.
__device__ __forceinline__ int clamp_bin(float d, int bins) {
  return int(fminf(fmaxf(d, 0.f), float(bins - 1)));
}

__device__ __forceinline__ U64 fixed_mass(unsigned key, float maximum) {
  return __float2ull_rn(expf(from_ordered(key) - maximum) * MASS_SCALE);
}

__device__ __forceinline__ float lane(const float4& v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

// Common high bits of all keys in [lo, hi]: returns the number of unknown low
// bits and sets prefix to the shared high bits.
__device__ __forceinline__ int common_start(unsigned lo, unsigned hi,
                                            unsigned& prefix) {
  unsigned diff = lo ^ hi;
  int start = diff ? 32 - __clz(diff) : 0;
  prefix = start >= 32 ? 0u : hi & ~((1u << start) - 1u);
  return start;
}

struct Max {
  template <class T>
  __device__ T operator()(T a, T b) const {
    return a > b ? a : b;
  }
};
struct Min {
  template <class T>
  __device__ T operator()(T a, T b) const {
    return a < b ? a : b;
  }
};
struct Sum {
  template <class T>
  __device__ T operator()(T a, T b) const {
    return a + b;
  }
};

template <class T, class Op>
__device__ T block_reduce(Shared& s, T v, Op op) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = op(v, __shfl_xor_sync(~0u, v, o));
  T* scratch = reinterpret_cast<T*>(s.scratch);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
  __syncthreads();
  v = scratch[0];
#pragma unroll
  for (int i = 1; i < WARPS; ++i) v = op(v, scratch[i]);
  return v;
}

template <class T>
__device__ const T& as(const Slot& slot) {
  static_assert(sizeof(T) <= sizeof(Slot));
  return *reinterpret_cast<const T*>(&slot);
}

// Every CTA's v (uniform within each CTA), by cluster rank, in this CTA's
// shared memory: each CTA pushes v into all peers, then one cluster barrier.
// The two slot sets alternate, so a slot is rewritten only after the next
// gather's barrier, when its owner has read it. Call after a __syncthreads.
template <class T>
__device__ const Slot* group_gather(Shared& s, const T& v) {
  Group<true> g;
  int turn = s.xturn, n = g.size();
  if (threadIdx.x < n)
    *reinterpret_cast<T*>(g.peer(&s.xin[turn][g.rank()], threadIdx.x)) = v;
  g.sync();
  if (threadIdx.x == 0) s.xturn = turn ^ 1;
  return s.xin[turn];
}

template <bool CL, class T, class Op>
__device__ T group_reduce(Shared& s, T v, Op op) {
  static_assert(sizeof(T) <= sizeof(U64));
  v = block_reduce(s, v, op);
  if (!CL) return v;
  const Slot* all = group_gather(s, v);
  T acc = as<T>(all[0]);
  for (int r = 1; r < Group<true>().size(); ++r) acc = op(acc, as<T>(all[r]));
  return acc;
}

__device__ unsigned block_exclusive_scan(Shared& s, unsigned v,
                                         unsigned& total) {
  int lane_id = threadIdx.x & 31, warp = threadIdx.x >> 5;
  unsigned incl = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    unsigned t = __shfl_up_sync(~0u, incl, o);
    if (lane_id >= o) incl += t;
  }
  __syncthreads();
  if (lane_id == 31) s.wsum[warp] = incl;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned acc = 0;
    for (int w = 0; w < WARPS; ++w) {
      unsigned t = s.wsum[w];
      s.wsum[w] = acc;
      acc += t;
    }
    s.wsum[WARPS] = acc;
  }
  __syncthreads();
  total = s.wsum[WARPS];
  return s.wsum[warp] + incl - v;
}

// ---------------------------------------------------------------------------
// Radix selection.

template <Weight W>
__device__ __forceinline__ U64 local_weight(const Shared& s, int b) {
  if (W == COUNT) return s.count[b];
  if (W == WIDE) return s.wide[b];
  return U64(s.limb[0][b]) + (U64(s.limb[1][b]) << 16) +
         (U64(s.limb[2][b]) << 32);
}

// Warp 0 finds the digit bin, scanning from the top, where the cumulative
// weight first reaches target.
template <Weight W, bool CL>
__device__ void scan_bins(Shared& s, int nb, int shift, U64 target, U64 above,
                          unsigned prefix) {
  int lane_id = threadIdx.x;
  U64 w[8], total = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int b = nb - 1 - (lane_id * 8 + j);
    w[j] = b < 0        ? 0
           : !CL        ? local_weight<W>(s, b)
           : W == COUNT ? U64(s.gcount[b])
                        : s.gwide[b];
    total += w[j];
  }
  U64 incl = total;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    U64 t = __shfl_up_sync(~0u, incl, o);
    if (lane_id >= o) incl += t;
  }
  U64 acc = above + incl - total;
  bool hit = acc < target && target <= above + incl;
  if (__ballot_sync(~0u, hit) == 0 && lane_id == 0) s.found = 0;
  if (!hit) return;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int b = nb - 1 - (lane_id * 8 + j);
    if (acc + w[j] >= target) {
      s.prefix = prefix | (unsigned(b) << shift);
      s.above = acc;
      s.count_at = CL ? s.gcount[b] : s.count[b];
      s.found = 1;
      return;
    }
    acc += w[j];
  }
}

// Cluster histograms: every CTA adds its nonzero bins into each CTA's combined
// bins, so the scan reads local shared memory only. open_bins clears this
// CTA's combined bins and arrives; push_bins waits until all are cleared.
__device__ void open_bins(Shared& s, int nb) {
  for (int b = threadIdx.x; b < nb; b += THREADS) {
    s.gcount[b] = 0;
    s.gwide[b] = 0;
  }
  Group<true>().arrive();
}

template <Weight W>
__device__ void push_bins(Shared& s, int nb) {
  Group<true> g;
  g.wait();
  int n = g.size(), rank = g.rank();
  for (int b = threadIdx.x; b < nb; b += THREADS) {
    unsigned count = s.count[b];
    if (!count) continue;
    U64 weight = local_weight<W>(s, b);
    for (int i = 0; i < n; ++i) {
      int r = (rank + i) & (n - 1);
      g.add(&s.gcount[b], r, count);
      if (W != COUNT) g.add(&s.gwide[b], r, weight);
    }
  }
  g.sync();
}

// Finds key K with weight(> K) < target <= weight(>= K) over the visited set.
// visit(f) calls f(select_key, value_key) for each owned element; all
// select keys share `prefix` above bit `start`.
template <Weight W, bool CL, class Visit>
__device__ Pick radix_select(Shared& s, Visit visit, U64 target,
                             unsigned prefix, int start, float maximum,
                             int max_rounds = 8) {
  U64 above = 0;
  int rounds = 0;
  bool found = true;
  do {
    int width = min(8, start), shift = start - width, nb = 1 << width;
    for (int i = threadIdx.x; i < nb; i += THREADS) {
      s.count[i] = 0;
      if (W == WIDE) s.wide[i] = 0;
      if (W == LIMBS) s.limb[0][i] = s.limb[1][i] = s.limb[2][i] = 0;
    }
    if (CL) open_bins(s, nb);
    __syncthreads();
    unsigned high = start >= 32 ? 0u : prefix >> start;
    visit([&](unsigned key, unsigned value) {
      if (start < 32 && (key >> start) != high) return;
      unsigned d = (key >> shift) & unsigned(nb - 1);
      atomicAdd(&s.count[d], 1u);
      if (W == COUNT) return;
      U64 w = fixed_mass(value, maximum);
      if (W == WIDE) {
        atomicAdd(&s.wide[d], w);
        return;
      }
      atomicAdd(&s.limb[0][d], unsigned(w & 0xffff));
      atomicAdd(&s.limb[1][d], unsigned((w >> 16) & 0xffff));
      atomicAdd(&s.limb[2][d], unsigned(w >> 32));
    });
    __syncthreads();
    if (CL) push_bins<W>(s, nb);
    if (threadIdx.x < 32) scan_bins<W, CL>(s, nb, shift, target, above, prefix);
    __syncthreads();
    found = s.found;
    prefix = s.prefix;
    above = s.above;
    start = shift;
    __syncthreads();
    if (threadIdx.x == 0) s.found = 1;
  } while (found && start > 0 && ++rounds < max_rounds);
  return Pick{prefix, above, s.count_at, found};
}

// ---------------------------------------------------------------------------
// Element sources: stored candidates, or the row chunk in global memory.

struct Candidates {
  static constexpr Weight MASS = LIMBS;
  Store st;
  unsigned n, floor;  // only keys above floor belong to the row's candidate set
  static constexpr int UNROLL = 2;
  // Large stores mostly live in L2: then keep more loads in flight.
  template <class F>
  __device__ void values(Cut set, F f) const {
    if (n > CAP) {
      visit<8>(set, f);
    } else {
      visit<UNROLL>(set, f);
    }
  }
  template <int U, class F>
  __device__ void visit(Cut set, F f) const {
    for (unsigned i0 = threadIdx.x; i0 < n; i0 += U * THREADS) {
      unsigned keys[U];
#pragma unroll
      for (int q = 0; q < U; ++q) {
        unsigned i = i0 + q * THREADS;
        keys[q] = i < n ? st.key(i) : 0u;
      }
#pragma unroll
      for (int q = 0; q < U; ++q) {
        unsigned key = keys[q];
        // The index matters only for a tie with the set's boundary key.
        if (key > floor &&
            (key > set.key ||
             (key == set.key && st.idx(i0 + q * THREADS) <= set.idx)))
          f(key, key);
      }
    }
  }
  template <class F>
  __device__ void ties(Cut set, unsigned tie, F f) const {
    for (unsigned i = threadIdx.x; i < n; i += THREADS) {
      unsigned key = st.key(i), id = st.idx(i);
      if (key == tie && set.keep(key, id)) f(~id, key);
    }
  }
};

struct Row {
  static constexpr Weight MASS = WIDE;
  const float* x;
  int begin, end;
  template <class F>
  __device__ void each(F f) const {
    for (int i = begin + threadIdx.x; i < end; i += THREADS)
      f(ordered(x[i]), unsigned(i));
  }
  template <class F>
  __device__ void values(Cut set, F f) const {
    each([&](unsigned key, unsigned id) {
      if (set.keep(key, id)) f(key, key);
    });
  }
  template <class F>
  __device__ void ties(Cut set, unsigned tie, F f) const {
    each([&](unsigned key, unsigned id) {
      if (key == tie && set.keep(key, id)) f(~id, key);
    });
  }
};

// Among ties at pick.key inside set, keep the `rank` lowest indices.
template <bool CL, class Source>
__device__ Cut tie_cut(Shared& s, const Source& src, Cut set, Pick pick,
                       U64 rank, int vocab) {
  Cut cut{pick.key, pick.key == set.key ? set.idx : 0xffffffffu};
  if (rank >= pick.count_at) return cut;
  unsigned prefix;
  int start = common_start(~unsigned(vocab - 1), 0xffffffffu, prefix);
  auto visit = [&](auto f) { src.ties(set, pick.key, f); };
  Pick tie = radix_select<COUNT, CL>(s, visit, rank, prefix, start, 0.f);
  cut.idx = ~tie.key;
  return cut;
}

template <bool CL, class Source>
__device__ Cut select_rank(Shared& s, const Source& src, Cut set, U64 rank,
                           unsigned prefix, int start, int vocab) {
  auto visit = [&](auto f) { src.values(set, f); };
  Pick pick = radix_select<COUNT, CL>(s, visit, rank, prefix, start, 0.f);
  if (!pick.found) return set;
  return tie_cut<CL>(s, src, set, pick, rank - pick.above, vocab);
}

template <bool CL, class Source>
__device__ Cut select_mass(Shared& s, const Source& src, Cut set, U64 target,
                           unsigned prefix, int start, float maximum,
                           int vocab) {
  auto visit = [&](auto f) { src.values(set, f); };
  Pick pick =
      radix_select<Source::MASS, CL>(s, visit, target, prefix, start, maximum);
  if (!pick.found) return set;
  U64 w = fixed_mass(pick.key, maximum);
  U64 need = target - pick.above;
  U64 rank = w ? (need + w - 1) / w : pick.count_at;
  rank = max(U64(1), min(rank, U64(pick.count_at)));
  return tie_cut<CL>(s, src, set, pick, rank, vocab);
}

template <bool CL, class Source>
__device__ U64 set_mass(Shared& s, const Source& src, Cut set, float maximum) {
  U64 total = 0;
  src.values(set, [&](unsigned, unsigned value) {
    total += fixed_mass(value, maximum);
  });
  return group_reduce<CL>(s, total, Sum());
}

__device__ __forceinline__ U64 mass_target(float p, double total) {
  double t = ceil(double(p) * total);
  return t < 1.0 ? 1ull : U64(t);
}

// ---------------------------------------------------------------------------
// TMA ring.

__device__ __forceinline__ unsigned smem_addr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void bar_init(U64* bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_addr(bar))
               : "memory");
}

__device__ __forceinline__ void bar_wait(U64* bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}" ::"r"(smem_addr(bar)),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void tma_load(void* dst, const void* src,
                                         unsigned bytes, U64* bar) {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   smem_addr(bar)),
               "r"(bytes)
               : "memory");
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], "
      "[%1], %2, [%3];" ::"r"(smem_addr(dst)),
      "l"(src), "r"(bytes), "r"(smem_addr(bar))
      : "memory");
}

// FIFO of chunk stages. All threads track the same state; thread 0 issues.
// A chunk may be opened ahead of time (prefetch) while the CTA finishes the
// previous row.
struct Ring {
  int depth;
  int rslot = 0, wslot = 0, inflight = 0;
  unsigned rparity = 0;
  const float* x = nullptr;
  int next = 0, end = 0;
  bool prefetched = false;

  __device__ void open(const float* src, int begin, int stop) {
    if (prefetched) {
      prefetched = false;
      return;
    }
    x = src;
    next = begin;
    end = stop;
  }
  __device__ void prefetch(const float* src, int begin, int stop) {
    open(src, begin, stop);
    pump();
    prefetched = true;
  }
  // Fill every free slot: pump runs after the previous round's barrier, so the
  // slot read last round is free too.
  __device__ void pump() {
    while (inflight < depth && next < end) {
      if (threadIdx.x == 0) {
        unsigned bytes = unsigned(min(STAGE, end - next)) * 4u;
        tma_load(ring_stage(wslot), x + next, bytes,
                 shared_state().bar + wslot);
      }
      wslot = wslot + 1 == depth ? 0 : wslot + 1;
      next += STAGE;
      ++inflight;
    }
  }
  __device__ void consume(int base, int stop, float4 (&v)[SVECS]) {
    bar_wait(shared_state().bar + rslot, rparity);
    const float4* stage = ring_stage(rslot);
    if (base + STAGE <= stop) {
#pragma unroll
      for (int j = 0; j < SVECS; ++j) v[j] = stage[j * THREADS + threadIdx.x];
    } else {
#pragma unroll
      for (int j = 0; j < SVECS; ++j) {
        int i = (j * THREADS + threadIdx.x) * 4;
        v[j] = base + i < stop
                   ? stage[i / 4]
                   : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
      }
    }
    if (++rslot == depth) {
      rslot = 0;
      rparity ^= 1u;
    }
    --inflight;
  }
};

// Up to STAGE values strided over the chunk, four contiguous per slot: token
// order is not random, so the chunk's first stage is a biased sample.
__device__ __forceinline__ void load_sample(const float* x, int begin, int end,
                                            bool vec, float4 (&v)[SVECS]) {
  int step = max(4, (end - begin) / (SVECS * THREADS) / 4 * 4);
#pragma unroll
  for (int j = 0; j < SVECS; ++j) {
    int i = begin + (j * THREADS + threadIdx.x) * step;
    if (vec && i + 3 < end) {
      v[j] = *reinterpret_cast<const float4*>(x + i);
    } else {
      v[j].x = i < end ? x[i] : -INFINITY;
      v[j].y = i + 1 < end ? x[i + 1] : -INFINITY;
      v[j].z = i + 2 < end ? x[i + 2] : -INFINITY;
      v[j].w = i + 3 < end ? x[i + 3] : -INFINITY;
    }
  }
}

__device__ __forceinline__ void load_scalar(const float* x, int end, int base,
                                            float4 (&v)[SVECS]) {
#pragma unroll
  for (int j = 0; j < SVECS; ++j) {
    int i = base + (j * THREADS + threadIdx.x) * 4;
    v[j].x = i < end ? x[i] : -INFINITY;
    v[j].y = i + 1 < end ? x[i + 1] : -INFINITY;
    v[j].z = i + 2 < end ? x[i + 2] : -INFINITY;
    v[j].w = i + 3 < end ? x[i + 3] : -INFINITY;
  }
}

// ---------------------------------------------------------------------------
// Streaming phase.

__device__ __forceinline__ int element_index(int base, int j, int c) {
  return base + (j * THREADS + threadIdx.x) * 4 + c;
}

__device__ __forceinline__ float element(const float4 (&v)[SVECS], int bit) {
  float y = v[0].x;
#pragma unroll
  for (int b = 1; b < SVECS * 4; ++b) y = bit == b ? lane(v[b >> 2], b & 3) : y;
  return y;
}

// Append values in (tv, hi] with one shared atomic per warp; the write loop
// runs over each thread's set bits only, since hits are usually sparse.
// During a stream the store holds raw float bits; they become ordered keys
// once the stream ends. Returns the mask of values that did not fit.
template <bool RANGE = false>
__device__ unsigned append(Shared& s, Store st, const float4 (&v)[SVECS],
                           int base, float tv, float hi = INFINITY) {
  unsigned mask = 0, dropped = 0;
#pragma unroll
  for (int j = 0; j < SVECS; ++j)
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      float y = lane(v[j], c);
      mask |= unsigned(y > tv && (!RANGE || y <= hi)) << (j * 4 + c);
    }
  if (!__any_sync(~0u, mask)) return 0;
  int lane_id = threadIdx.x & 31, n = __popc(mask);
  unsigned excl, total;
  if (__all_sync(~0u, n <= 1)) {
    unsigned hits = __ballot_sync(~0u, n);
    excl = __popc(hits & ((1u << lane_id) - 1u));
    total = __popc(hits);
  } else {
    int incl = n;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int t = __shfl_up_sync(~0u, incl, o);
      if (lane_id >= o) incl += t;
    }
    excl = unsigned(incl - n);
    total = unsigned(__shfl_sync(~0u, incl, 31));
  }
  unsigned wbase = 0;
  if (lane_id == 0) wbase = atomicAdd(&s.total, total);
  unsigned pos = __shfl_sync(~0u, wbase, 0) + excl;
  for (; mask; mask &= mask - 1, ++pos) {
    int bit = __ffs(mask) - 1;
    unsigned value = __float_as_uint(element(v, bit)),
             id = element_index(base, bit >> 2, bit & 3);
    if (pos < CAP) {
      s.key[pos] = value;
      s.idx[pos] = id;
    } else if (pos < STORE) {
      st.g[pos - CAP] = value;
      st.g[GCAP + pos - CAP] = id;
    } else {
      dropped |= 1u << bit;
    }
  }
  return dropped;
}

__device__ __forceinline__ unsigned stored_key(Store st, unsigned i) {
  return ordered(__uint_as_float(st.key(i)));
}

// Raise this CTA's threshold so at most keep_rank - 1 candidates remain,
// including the elements of this round that did not fit. Returns the new key
// threshold.
__device__ __noinline__ unsigned compact(Shared& s, Store st,
                                         const float4 (&v)[SVECS], int base,
                                         unsigned tk, unsigned lost,
                                         unsigned keep_rank,
                                         unsigned min_needed, int& status) {
  unsigned stored = min(s.total, STORE);
  auto dropped = [&](auto f) {
#pragma unroll
    for (int b = 0; b < SVECS * 4; ++b)
      if ((lost >> b) & 1u)
        f(ordered(lane(v[b >> 2], b & 3)), element_index(base, b >> 2, b & 3));
  };
  auto visit = [&](auto f) {
    for (unsigned i = threadIdx.x; i < stored; i += THREADS) {
      unsigned key = stored_key(st, i);
      f(key, key);
    }
    dropped([&](unsigned key, unsigned) { f(key, key); });
  };
  unsigned hi = 0;
  visit([&](unsigned key, unsigned) { hi = max(hi, key); });
  hi = block_reduce(s, hi, Max());
  unsigned prefix;
  int start = common_start(tk + 1, hi, prefix);
  Pick pick =
      radix_select<COUNT, false>(s, visit, keep_rank, prefix, start, 0.f);
  unsigned ntk = pick.key;
  if (!pick.found || pick.above < min_needed) {
    status = GLOBAL;
    __syncthreads();
    if (threadIdx.x == 0) s.total = 0;
    __syncthreads();
    return NO_CANDIDATES;
  }
  unsigned out = 0;
  for (unsigned c0 = 0; c0 < stored; c0 += THREADS * 4) {
    unsigned keys[4], ids[4], n = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      unsigned i = c0 + threadIdx.x * 4 + q;
      if (i < stored && stored_key(st, i) > ntk) {
        keys[n] = st.key(i);
        ids[n] = st.idx(i);
        ++n;
      }
    }
    unsigned total;
    unsigned offset = out + block_exclusive_scan(s, n, total);
    for (unsigned q = 0; q < n; ++q) {
      st.key(offset + q) = keys[q];
      st.idx(offset + q) = ids[q];
    }
    out += total;
  }
  __syncthreads();
  if (threadIdx.x == 0) s.total = out;
  __syncthreads();
  dropped([&](unsigned key, unsigned id) {
    if (key <= ntk) return;
    unsigned pos = atomicAdd(&s.total, 1u);
    st.key(pos) = __float_as_uint(from_ordered(key));
    st.idx(pos) = id;
  });
  __syncthreads();
  return ntk;
}

struct StreamArgs {
  bool gather, sample, need_sum;
  unsigned target, keep_rank, min_needed;
  int vocab, sample_size;
  float p;
};

// Count histogram of this CTA's sample below the sample max, which sizes the
// candidates of top-p-only rows. Clusters add it into every CTA's ghist,
// complete after the next cluster barrier.
template <bool CL>
__device__ void record_sample(Shared& s, const float4 (&v)[SVECS], float top) {
  for (int b = threadIdx.x; b < SBINS; b += THREADS) s.shist[b] = 0;
  if (threadIdx.x == 0) s.shist_top = top;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < SVECS; ++j)
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      float y = lane(v[j], c);
      if (y == -INFINITY) continue;
      atomicAdd(&s.shist[clamp_bin((top - y) * (SBINS / SB_NATS), SBINS)], 1u);
    }
  if (!CL) return;
  __syncthreads();
  Group<true> g;
  int n = g.size(), rank = g.rank();
  for (int b = threadIdx.x; b < SBINS; b += THREADS) {
    unsigned count = s.shist[b];
    if (!count) continue;
    for (int i = 0; i < n; ++i) g.add(&s.ghist[b], (rank + i) & (n - 1), count);
  }
}

template <bool CL>
__device__ const unsigned* sample_counts(const Shared& s) {
  return CL ? s.ghist : s.shist;
}

// Row estimates, from the sample histogram, of the count and of the mass
// (relative to maximum, at bin centers) of the values below tv, cumulative
// from the top. Returns the key `margin` bins below the first bin where the
// mass reaches need, or share of the estimated total when share > 0, and
// NO_CANDIDATES if the count exceeds limit first.
__device__ unsigned sample_cutoff(Shared& s, const unsigned* counts, float tv,
                                  float maximum, float scale, float need,
                                  float share, float limit, int margin) {
  // Warp 0, eight bins per lane from the top.
  constexpr int PER = SBINS / 32;
  if (threadIdx.x < 32) {
    int lane_id = threadIdx.x;
    float top = s.shist_top, w = SB_NATS / SBINS;
    float c[PER], m[PER], lc = 0.f, lm = 0.f;
#pragma unroll
    for (int q = 0; q < PER; ++q) {
      int b = lane_id * PER + q;
      float upper = top - b * w, lower = upper - w;
      c[q] = lower >= tv
                 ? 0.f
                 : counts[b] * scale * (upper > tv ? (tv - lower) / w : 1.f);
      m[q] = c[q] * expf(lower + 0.5f * w - maximum);
      lc += c[q];
      lm += m[q];
    }
    float ic = lc, im = lm;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      float tc = __shfl_up_sync(~0u, ic, o), tm = __shfl_up_sync(~0u, im, o);
      if (lane_id >= o) {
        ic += tc;
        im += tm;
      }
    }
    if (share > 0.f) need = share * __shfl_sync(~0u, im, 31);
    unsigned hits = __ballot_sync(~0u, im >= need || lane_id == 31);
    if (lane_id == 0) s.prefix = NO_CANDIDATES;
    __syncwarp();
    if (lane_id == __ffs(hits) - 1) {
      float cc = ic - lc, cm = im - lm;
      for (int q = 0; q < PER; ++q) {
        cc += c[q];
        cm += m[q];
        if (cc > limit) break;
        int b = lane_id * PER + q;
        if (cm >= need || b == SBINS - 1) {
          s.prefix = ordered(top - (b + margin) * w);
          break;
        }
      }
    }
  }
  __syncthreads();
  unsigned result = s.prefix;
  __syncthreads();
  return result;
}

// A nucleus over a quarter of the row is cheaper through the window path,
// which never stores it.
template <bool CL>
__device__ float candidate_limit(unsigned room, int vocab) {
  return fminf(0.8f * room * Group<CL>().size(), 0.25f * vocab);
}

// Approximate threshold so that about the target count of the row lies above
// it, from the sample of every CTA's first round. Top-p-only rows lower it to
// cover the nucleus the sample predicts, with margin.
template <bool CL>
__device__ __noinline__ unsigned sample_threshold(Shared& s,
                                                  const float4 (&v)[SVECS],
                                                  const StreamArgs& a) {
  // Cleared before the barrier below, which peers pass before adding to it.
  if (CL && a.need_sum)
    for (int b = threadIdx.x; b < SBINS; b += THREADS) s.ghist[b] = 0;
  unsigned lo = 0xffffffffu, hi = 0;
#pragma unroll
  for (int j = 0; j < SVECS; ++j)
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      unsigned key = ordered(lane(v[j], c));
      if (key > KEY_NEGINF) {
        lo = min(lo, key);
        hi = max(hi, key);
      }
    }
  lo = block_reduce(s, lo, Min());
  hi = block_reduce(s, hi, Max());
  if (CL) {
    const Slot* all = group_gather(s, make_uint2(lo, hi));
    for (int r = 0; r < Group<true>().size(); ++r) {
      lo = min(lo, as<uint2>(all[r]).x);
      hi = max(hi, as<uint2>(all[r]).y);
    }
  }
  if (hi <= KEY_NEGINF) return KEY_NEGINF;
  unsigned prefix;
  int start = common_start(lo, hi, prefix);
  if (a.need_sum) record_sample<CL>(s, v, from_ordered(hi));
  U64 scaled = (U64(a.target) * a.sample_size + a.vocab - 1) / a.vocab;
  unsigned rank = unsigned(min(U64(a.sample_size), max(U64(16), scaled)));
  auto visit = [&](auto f) {
#pragma unroll
    for (int j = 0; j < SVECS; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        unsigned key = ordered(lane(v[j], c));
        if (key > KEY_NEGINF) f(key, key);
      }
  };
  Pick pick = radix_select<COUNT, CL>(s, visit, rank, prefix, start, 0.f, 2);
  unsigned tk = pick.found ? max(pick.key - 1, KEY_NEGINF) : KEY_NEGINF;
  if (!CL || !a.need_sum || tk == KEY_NEGINF) return tk;
  // The radix barriers completed the cluster's sample histogram.
  float scale = float(a.vocab) / a.sample_size;
  unsigned lower =
      sample_cutoff(s, sample_counts<CL>(s), INFINITY, from_ordered(hi), scale,
                    0.f, a.p, candidate_limit<CL>(CAP, a.vocab), 4);
  return lower == NO_CANDIDATES ? tk : max(min(tk, lower), KEY_NEGINF);
}

struct Stats {
  float maximum;
  float minimum;  // smallest finite value, only when the sum is requested
  double total;   // sum of exp(x - maximum) over the row, only when requested
  unsigned tk;    // the row's candidates are the keys above tk
  int status;
};

// Float exp: FP64 exp is slow on some GPUs. Each rescaled part keeps an
// independent relative error below 2^-22.
__device__ __forceinline__ double rescale(double sum, float from, float to) {
  return from == -INFINITY ? 0.0 : sum * double(expf(from - to));
}

// The CTAs' stream statistics, combined with one barrier. Each lane rescales
// one CTA's sum, and the butterfly gives every lane of every CTA the same bits.
__device__ Stats combine_stats(Shared& s, Stats st, bool sum) {
  const Slot* all = group_gather(s, st);
  int n = Group<true>().size(), lane_id = threadIdx.x & 31;
  Stats out = as<Stats>(all[0]);
  for (int r = 1; r < n; ++r) {
    const Stats& q = as<Stats>(all[r]);
    out.maximum = fmaxf(out.maximum, q.maximum);
    out.minimum = fminf(out.minimum, q.minimum);
    out.tk = max(out.tk, q.tk);
    out.status = max(out.status, q.status);
  }
  if (!sum) return out;
  const Stats& q = as<Stats>(all[min(lane_id, n - 1)]);
  double t = lane_id < n ? rescale(q.total, q.maximum, out.maximum) : 0.0;
#pragma unroll
  for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(~0u, t, o);
  out.total = t;
  return out;
}

// Not inlined, with the ring state in registers, so the hot loop gets its own
// register allocation.
template <bool CL, bool SUM>
__device__ __forceinline__ Stats stream(Shared& s, Store st, Ring& ring_state,
                                        const float* x, int begin, int end,
                                        bool vec, StreamArgs a) {
  Ring ring = ring_state;
  __syncthreads();
  if (threadIdx.x == 0) {
    s.total = 0;
    s.shist_top = -INFINITY;
  }
  __syncthreads();
  int status = OK;
  unsigned tk = a.gather ? KEY_NEGINF : NO_CANDIDATES;
  float m = -INFINITY, mn = INFINITY, sum = 0.f, comp = 0.f;
  int rounds = max(0, end - begin + STAGE - 1) / STAGE;
  if (vec) {
    ring.open(x, begin, end);
    ring.pump();
  }
  // Clusters sample collectively, so every CTA samples, even an empty chunk.
  if (a.gather && a.sample) {
    float4 v[SVECS];
    load_sample(x, begin, end, vec, v);
    tk = sample_threshold<CL>(s, v, a);
  }
  for (int r = 0; r < rounds; ++r) {
    int base = begin + r * STAGE;
    float4 v[SVECS];
    if (vec) {
      ring.pump();
      ring.consume(base, end, v);
    } else {
      load_scalar(x, end, base, v);
    }
    if (SUM) {
      float mr = -INFINITY;
#pragma unroll
      for (int j = 0; j < SVECS; ++j)
        mr = fmaxf(fmaxf(fmaxf(mr, v[j].x), fmaxf(v[j].y, v[j].z)), v[j].w);
      if (mr > m) {
        float scale = m == -INFINITY ? 0.f : expf(m - mr);
        sum *= scale;
        comp *= scale;
        m = mr;
      }
      if (m != -INFINITY) {
#pragma unroll
        for (int j = 0; j < SVECS; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            float y = lane(v[j], c), e = expf(y - m), t = sum + e;
            comp += sum >= e ? (sum - t) + e : (e - t) + sum;
            sum = t;
            mn = y == -INFINITY ? mn : fminf(mn, y);
          }
      }
    } else if (!a.gather) {
#pragma unroll
      for (int j = 0; j < SVECS; ++j)
        m = fmaxf(fmaxf(fmaxf(m, v[j].x), fmaxf(v[j].y, v[j].z)), v[j].w);
    }
    unsigned lost =
        tk != NO_CANDIDATES ? append(s, st, v, base, from_ordered(tk)) : 0u;
    // The barrier also frees this round's ring slot for reuse.
    if (__syncthreads_or(lost != 0))
      tk = compact(s, st, v, base, tk, lost, a.keep_rank, a.min_needed, status);
  }
  // Candidates become ordered keys. Without the sum the row max is the
  // largest candidate: every candidate set is non-empty unless no value is
  // finite, and compaction failure (GLOBAL) recomputes it.
  unsigned stored = min(s.total, STORE), top = 0;
  for (unsigned i = threadIdx.x; i < stored; i += THREADS) {
    unsigned key = stored_key(st, i);
    st.key(i) = key;
    top = max(top, key);
  }
  if (!SUM && a.gather) m = top ? from_ordered(top) : -INFINITY;
  Stats out{block_reduce(s, m, Max()), INFINITY, 0.0,
            block_reduce(s, tk, Max()), block_reduce(s, status, Max())};
  if (SUM) {
    double part = rescale(double(sum) + double(comp), m, out.maximum);
    out.total = block_reduce(s, part, Sum());
    out.minimum = block_reduce(s, mn, Min());
  }
  ring_state = ring;
  return CL ? combine_stats(s, out, SUM) : out;
}

// ---------------------------------------------------------------------------
// Output.

__device__ void fill_and_scatter(float* x, int begin, int end, bool vec,
                                 const Candidates& src, Cut cut) {
  if (vec) {
    float4 neg = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
    for (int i = begin + threadIdx.x * 4; i < end; i += THREADS * 4)
      *reinterpret_cast<float4*>(x + i) = neg;
  } else {
    for (int i = begin + threadIdx.x; i < end; i += THREADS) x[i] = -INFINITY;
  }
  __syncthreads();
  constexpr int UNROLL = Candidates::UNROLL;
  for (unsigned i0 = threadIdx.x; i0 < src.n; i0 += UNROLL * THREADS) {
    unsigned keys[UNROLL], ids[UNROLL];
#pragma unroll
    for (int q = 0; q < UNROLL; ++q) {
      unsigned i = i0 + q * THREADS;
      keys[q] = i < src.n ? src.st.key(i) : 0u;
      ids[q] = i < src.n ? src.st.idx(i) : 0u;
    }
#pragma unroll
    for (int q = 0; q < UNROLL; ++q)
      if (keys[q] > src.floor && cut.keep(keys[q], ids[q]))
        x[ids[q]] = from_ordered(keys[q]);
  }
}

__device__ __forceinline__ float masked(float y, int i, Cut cut) {
  return cut.keep(ordered(y), unsigned(i)) ? y : -INFINITY;
}

// Read-modify-write of the chunk, four float4 per thread in flight; a vector is
// stored only if one of its values changes.
__device__ __noinline__ void mask_row(float* x, int begin, int end, bool vec,
                                      Cut cut) {
  if (!vec) {
    for (int i = begin + threadIdx.x; i < end; i += THREADS)
      x[i] = masked(x[i], i, cut);
    return;
  }
  constexpr int U = 4;
  for (int i0 = begin + threadIdx.x * 4; i0 < end; i0 += U * THREADS * 4) {
    float4 v[U];
#pragma unroll
    for (int q = 0; q < U; ++q) {
      int i = i0 + q * THREADS * 4;
      if (i < end) v[q] = *reinterpret_cast<const float4*>(x + i);
    }
#pragma unroll
    for (int q = 0; q < U; ++q) {
      int i = i0 + q * THREADS * 4;
      if (i >= end) continue;
      float4 m =
          make_float4(masked(v[q].x, i, cut), masked(v[q].y, i + 1, cut),
                      masked(v[q].z, i + 2, cut), masked(v[q].w, i + 3, cut));
      if (m.x != v[q].x || m.y != v[q].y || m.z != v[q].z || m.w != v[q].w)
        *reinterpret_cast<float4*>(x + i) = m;
    }
  }
}

// ---------------------------------------------------------------------------
// Row processing.

template <bool CL>
__device__ __noinline__ void global_path(Shared& s, float* x, int begin,
                                         int end, int vocab, bool vec,
                                         bool use_k, unsigned kk, bool use_p,
                                         float p, Stats st) {
  Row src{x, begin, end};
  Cut set = ALL;
  if (use_k) set = select_rank<CL>(s, src, ALL, kk, 0u, 32, vocab);
  Cut cut = set;
  if (use_p) {
    double total = use_k ? double(set_mass<CL>(s, src, set, st.maximum))
                         : st.total * MASS_SCALE;
    cut = select_mass<CL>(s, src, set, mass_target(p, total), 0u, 32,
                          st.maximum, vocab);
  }
  __syncthreads();
  mask_row(x, begin, end, vec, cut);
}

// ---------------------------------------------------------------------------
// Window path: the top-p cutoff lies below the candidates (large nucleus).
// A value-space histogram locates the crossing bin, a second pass gathers a
// small window of bins around it and the exact mass above the window, and the
// exact radix selection finishes inside the window.

constexpr int WBINS = 2048;
constexpr float WINDOW_NATS =
    80.f;  // mass below max - 80 cannot move the cutoff

template <class F>
__device__ void for_chunk(Ring& ring, const float* x, int begin, int end,
                          bool vec, F f) {
  int rounds = max(0, end - begin + STAGE - 1) / STAGE;
  if (vec) ring.open(x, begin, end);
  for (int r = 0; r < rounds; ++r) {
    int base = begin + r * STAGE;
    float4 v[SVECS];
    if (vec) {
      ring.pump();
      ring.consume(base, end, v);
    } else {
      load_scalar(x, end, base, v);
    }
#pragma unroll
    for (int j = 0; j < SVECS; ++j)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        int i = element_index(base, j, c);
        if (i < end) f(lane(v[j], c), i);
      }
    __syncthreads();  // frees the ring slot
  }
}

// Linear bins over [lo, hi] for values with key <= tk, bin 0 at the top.
struct Bins {
  float hi, lo, inv;
  unsigned tk;
  __device__ bool in_domain(unsigned key, float y) const {
    return key <= tk && y >= lo;
  }
  __device__ int bin(float y) const { return clamp_bin((hi - y) * inv, WBINS); }
};

// Warp 0: first bin where above + cumulative mass reaches target.
__device__ void crossing_bin(Shared& s, const float* mass, float above,
                             float target) {
  int lane_id = threadIdx.x;
  constexpr int PER = WBINS / 32;
  float local = 0.f;
  for (int q = 0; q < PER; ++q) local += mass[lane_id * PER + q];
  float incl = local;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    float t = __shfl_up_sync(~0u, incl, o);
    if (lane_id >= o) incl += t;
  }
  float acc = above + incl - local;
  unsigned hits = __ballot_sync(~0u, above + incl >= target);
  if (lane_id == 0) s.prefix = WBINS - 1;
  __syncwarp();
  if (!hits || lane_id != __ffs(hits) - 1) return;
  for (int q = 0; q < PER; ++q) {
    acc += mass[lane_id * PER + q];
    if (acc >= target) {
      s.prefix = lane_id * PER + q;
      return;
    }
  }
}

template <bool CL>
__device__ __noinline__ bool window_path(Shared& s, Ring& ring, float* x,
                                         int begin, int end, bool vec,
                                         int vocab, Stats st, U64 above_cands,
                                         U64 target) {
  float hi = from_ordered(st.tk),
        lo = fmaxf(st.minimum, st.maximum - WINDOW_NATS);
  if (!(lo < hi)) return false;
  Bins bins{hi, lo, WBINS / (hi - lo), st.tk};
  float maximum = st.maximum;
  float* mass =
      reinterpret_cast<float*>(s.key);  // histograms alias the candidate buffer
  unsigned* count = s.idx;
  __syncthreads();
  for (int b = threadIdx.x; b < WBINS; b += THREADS) {
    mass[b] = 0.f;
    count[b] = 0;
  }
  __syncthreads();
  for_chunk(ring, x, begin, end, vec, [&](float y, int) {
    unsigned key = ordered(y);
    if (!bins.in_domain(key, y)) return;
    int b = bins.bin(y);
    atomicAdd(&count[b], 1u);
    atomicAdd(&mass[b], expf(y - maximum));
  });
  float* total_mass = mass;
  if (CL) {
    Group<true> g;
    total_mass = mass + WBINS;
    g.sync();
    for (int b = threadIdx.x; b < WBINS; b += THREADS) {
      float t = 0.f;
#pragma unroll 4
      for (int r = 0; r < g.size(); ++r) t += g.peer(mass, r)[b];
      total_mass[b] = t;
    }
    g.sync();
  }
  float above_f = float(double(above_cands) / MASS_SCALE);
  float target_f = float(double(target) / MASS_SCALE);
  if (threadIdx.x < 32) crossing_bin(s, total_mass, above_f, target_f);
  __syncthreads();
  int center = int(s.prefix);
  int j0 = max(0, center - 1), j1 = min(WBINS - 1, center + 1);
  unsigned local = 0;
  for (int b = j0; b <= j1; ++b) local += count[b];
  if (group_reduce<CL>(s, local, Max()) > CAP) {
    j0 = j1 = center;
    if (group_reduce<CL>(s, count[center], Max()) > CAP) return false;
  }
  // Masking below the window during the gather is safe when the float
  // cumulative mass through the window exceeds the target by more than its
  // recursive-summation error bound.
  float max_count = 0.f, elements = 0.f;
  for (int b = threadIdx.x; b <= j1; b += THREADS) {
    max_count = fmaxf(max_count, float(count[b]));
    elements += float(count[b]);
  }
  max_count = group_reduce<CL>(s, max_count, Max());
  elements = group_reduce<CL>(s, elements, Sum());
  if (threadIdx.x == 0) {
    float acc = above_f;
    for (int b = 0; b <= j1; ++b) acc += total_mass[b];
    s.xres = __float_as_uint(acc);
  }
  __syncthreads();
  float through = __uint_as_float(unsigned(s.xres));
  int splits = Group<CL>().size();
  float error = through * 5.97e-8f * (max_count * splits + 2.f * WBINS + 64.f) +
                elements * 1e-13f;
  bool fused = through - error > target_f * (1.f + 1.2e-7f);
  __syncthreads();
  if (threadIdx.x == 0) s.total = 0;
  __syncthreads();
  U64 above = 0;
  for_chunk(ring, x, begin, end, vec, [&](float y, int i) {
    unsigned key = ordered(y);
    bool domain = bins.in_domain(key, y);
    int b = domain ? bins.bin(y) : WBINS;
    if (key > st.tk || b < j0) {
      above += fixed_mass(key, maximum);
    } else if (b <= j1) {
      unsigned pos = atomicAdd(&s.total, 1u);
      s.key[pos] = key;
      s.idx[pos] = unsigned(i);
    } else if (fused && y != -INFINITY) {
      x[i] = -INFINITY;
    }
  });
  above = group_reduce<CL>(s, above, Sum());
  Candidates src{Store{nullptr}, s.total, 0u};
  U64 window = set_mass<CL>(s, src, ALL, maximum);
  // Below the window only masked elements changed, all of them under the cut.
  if (!(above < target && target <= above + window)) return false;
  unsigned klo = 0xffffffffu, khi = 0;
  for (unsigned i = threadIdx.x; i < src.n; i += THREADS) {
    klo = min(klo, s.key[i]);
    khi = max(khi, s.key[i]);
  }
  klo = group_reduce<CL>(s, klo, Min());
  khi = group_reduce<CL>(s, khi, Max());
  unsigned prefix;
  int start = common_start(klo, khi, prefix);
  Cut cut = select_mass<CL>(s, src, ALL, target - above, prefix, start, maximum,
                            vocab);
  __syncthreads();
  if (!fused) {
    mask_row(x, begin, end, vec, cut);
    return true;
  }
  for (unsigned i = threadIdx.x; i < src.n; i += THREADS)
    if (!cut.keep(s.key[i], s.idx[i])) x[s.idx[i]] = -INFINITY;
  return true;
}

// ---------------------------------------------------------------------------
// Second gather: the nucleus extends below the candidates. The sample's shape
// below the candidate threshold, scaled with the stream's exact max and the
// missing mass, sizes a lower threshold; one more pass appends the values in
// between to the store.

// The candidates above the sample's estimate, from the exact max and target
// mass, of the top-p cut with margin; src itself without a sample.
template <bool CL>
__device__ Candidates near_cut(Shared& s, Candidates src, const Stats& st,
                               U64 target, const StreamArgs& a) {
  if (!(s.shist_top > -INFINITY)) return src;
  unsigned floor =
      sample_cutoff(s, sample_counts<CL>(s), INFINITY, st.maximum,
                    float(a.vocab) / a.sample_size,
                    float(double(target) / MASS_SCALE), 0.f, INFINITY, 16);
  src.floor = max(src.floor, floor);
  return src;
}

// Returns the lower key threshold, or NO_CANDIDATES when the estimate does
// not fit the store.
template <bool CL>
__device__ unsigned second_threshold(Shared& s, const Stats& st, U64 missing,
                                     unsigned stored, const StreamArgs& a) {
  if (!(s.shist_top > -INFINITY)) return NO_CANDIDATES;
  unsigned room = STORE - group_reduce<CL>(s, stored, Max());
  float need = 1.25f * float(double(missing) / MASS_SCALE);
  return sample_cutoff(s, sample_counts<CL>(s), from_ordered(st.tk), st.maximum,
                       float(a.vocab) / a.sample_size, need, 0.f,
                       candidate_limit<CL>(room, a.vocab), 2);
}

// Appends this CTA's values with keys in (lo, tk] as keys. Returns false if
// any CTA's store overflowed.
template <bool CL>
__device__ bool second_gather(Shared& s, Store st, Ring& ring, const float* x,
                              int begin, int end, bool vec, unsigned lo,
                              unsigned tk) {
  unsigned before = s.total;
  float low = from_ordered(lo), high = from_ordered(tk);
  bool over = false;
  int rounds = max(0, end - begin + STAGE - 1) / STAGE;
  if (vec) ring.open(x, begin, end);
  for (int r = 0; r < rounds; ++r) {
    int base = begin + r * STAGE;
    float4 v[SVECS];
    if (vec) {
      ring.pump();
      ring.consume(base, end, v);
    } else {
      load_scalar(x, end, base, v);
    }
    over |= __syncthreads_or(append<true>(s, st, v, base, low, high) != 0);
  }
  unsigned stored = min(s.total, STORE);
  for (unsigned i = before + threadIdx.x; i < stored; i += THREADS)
    st.key(i) = stored_key(st, i);
  return !group_reduce<CL>(s, int(over), Max());
}

template <bool CL>
__device__ float row_max(Shared& s, const float* x, int begin, int end) {
  float m = -INFINITY;
  for (int i = begin + threadIdx.x; i < end; i += THREADS) m = fmaxf(m, x[i]);
  return group_reduce<CL>(s, m, Max());
}

__device__ unsigned count_above(Store st, unsigned n, unsigned floor) {
  unsigned c = 0;
  for (unsigned i = threadIdx.x; i < n; i += THREADS) c += st.key(i) > floor;
  return c;
}

template <bool CL>
__device__ Stats stream_row(Shared& s, Store st, Ring& ring, const float* x,
                            int begin, int end, bool vec, StreamArgs a) {
  return a.need_sum ? stream<CL, true>(s, st, ring, x, begin, end, vec, a)
                    : stream<CL, false>(s, st, ring, x, begin, end, vec, a);
}

// `next` is the CTA's next row; its first stages are prefetched once this row
// no longer needs the ring.
template <bool CL>
__device__ void process_row(Shared& s, Store store, Ring& ring, float* x,
                            int begin, int end, int vocab, int sample_size,
                            bool vec, bool use_k, unsigned kk, bool use_p,
                            float p, const float* next) {
  auto prefetch_next = [&] {
    if (vec && next) ring.prefetch(next, begin, end);
  };
  StreamArgs a;
  a.gather = !use_k || kk <= MAX_CAND_K;
  a.need_sum = use_p && !use_k;
  a.keep_rank = use_k ? max(CAP / 2, kk + 1) : KEEP_P;
  a.min_needed = use_k ? kk : 1u;
  // 8x margin for small k, and 2x (over 5 sigma of the sample estimate) for
  // large k, so the expected candidates cover k and one stream suffices.
  a.target =
      use_k ? max(min(CAP / 2, max(1024u, 8 * kk)), min(MAX_CAND_K, 2 * kk))
            : TARGET_P;
  a.sample = vocab > STAGE;
  a.vocab = vocab;
  a.sample_size = sample_size;
  a.p = p;
  Stats st = stream_row<CL>(s, store, ring, x, begin, end, vec, a);
  unsigned n_row = group_reduce<CL>(
      s, count_above(store, min(s.total, STORE), st.tk), Sum());
  bool restarted =
      st.status == OK && a.gather && use_k && n_row < kk && st.tk > KEY_NEGINF;
  if (restarted) {
    a.sample = false;
    st = stream_row<CL>(s, store, ring, x, begin, end, vec, a);
    n_row = group_reduce<CL>(s, count_above(store, min(s.total, STORE), st.tk),
                             Sum());
  }
  if (!a.gather || st.status != OK) {
    if (a.gather && !a.need_sum) st.maximum = row_max<CL>(s, x, begin, end);
    if (st.maximum == -INFINITY) return;
    return global_path<CL>(s, x, begin, end, vocab, vec, use_k, kk, use_p, p,
                           st);
  }
  if (st.maximum == -INFINITY) return;
  bool may_window = use_p && !use_k && st.tk != KEY_NEGINF;
  if (!may_window) prefetch_next();
  bool exact_k =
      !use_k || n_row <= kk;  // candidates already form the top-k set
  if (exact_k && !use_p && st.tk == KEY_NEGINF)
    return;  // only -inf would be masked
  Candidates src{store, min(s.total, STORE), st.tk};
  unsigned prefix;
  int start = common_start(st.tk + 1, ordered(st.maximum), prefix);
  Cut set =
      exact_k ? ALL : select_rank<CL>(s, src, ALL, kk, prefix, start, vocab);
  Cut cut = set;
  if (use_p) {
    double total;
    Candidates sel = src;  // the candidates the mass selection visits
    if (use_k || st.tk == KEY_NEGINF) {
      total = double(set_mass<CL>(s, src, set, st.maximum));
    } else {
      total = st.total * MASS_SCALE;
      U64 target = mass_target(p, total);
      // The candidates above the sample's estimate of the cut usually hold
      // the target mass: then the selection visits only them.
      sel = near_cut<CL>(s, src, st, target, a);
      if (sel.floor == src.floor ||
          set_mass<CL>(s, sel, ALL, st.maximum) < target) {
        U64 cand_mass = set_mass<CL>(s, src, ALL, st.maximum);
        unsigned lower =
            cand_mass < target
                ? second_threshold<CL>(s, st, target - cand_mass, src.n, a)
                : NO_CANDIDATES;
        if (lower != NO_CANDIDATES) {
          if (second_gather<CL>(s, store, ring, x, begin, end, vec, lower,
                                st.tk)) {
            src = Candidates{store, min(s.total, STORE), lower};
            n_row = group_reduce<CL>(s, threadIdx.x ? 0u : src.n, Sum());
            cand_mass = set_mass<CL>(s, src, ALL, st.maximum);
          }
        }
        if (cand_mass < target) {
          if (window_path<CL>(s, ring, x, begin, end, vec, vocab, st, cand_mass,
                              target))
            return;
          return global_path<CL>(s, x, begin, end, vocab, vec, use_k, kk, use_p,
                                 p, st);
        }
        sel = src;
      }
      start = common_start(sel.floor + 1, ordered(st.maximum), prefix);
      prefetch_next();
    }
    cut = select_mass<CL>(s, sel, set, mass_target(p, total), prefix, start,
                          st.maximum, vocab);
  }
  __syncthreads();
  // Large kept sets: a masked read-modify-write beats writing the row twice.
  // One CTA per long row (large batches) gains from it with smaller sets.
  if ((CL || vocab < 16 * STAGE ? 4ull : 8ull) * n_row > unsigned(vocab))
    return mask_row(x, begin, end, vec, cut);
  fill_and_scatter(x, begin, end, vec, src, cut);
}

__device__ void init_ring(Shared& s, int depth) {
  if (threadIdx.x == 0) {
    s.found = 1;
    s.xturn = 0;
    for (int i = 0; i < depth; ++i) bar_init(s.bar + i);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
}

struct Args {
  float* logits;
  int64_t stride;
  const int* ks;
  const float* ps;
  unsigned* workspace;  // GCAP keys then GCAP indices per CTA
  int batch, vocab, depth, splits;
  bool vec;
};

__device__ Store cta_store(const Args& a) {
  return Store{a.workspace + size_t(blockIdx.x) * 2 * GCAP};
}

__device__ bool row_params(const Args& a, int row, bool& use_k, unsigned& kk,
                           bool& use_p, float& p) {
  int k = a.ks ? a.ks[row] : a.vocab;
  p = a.ps ? a.ps[row] : 1.f;
  use_k = k < a.vocab;
  use_p = p < 1.f;
  kk = unsigned(max(k, 1));
  return use_k || use_p;
}

// First row at or after `row`, in this CTA's stride, that needs filtering.
__device__ int next_active(const Args& a, int row) {
  for (; row < a.batch; row += gridDim.x) {
    int k = a.ks ? a.ks[row] : a.vocab;
    float p = a.ps ? a.ps[row] : 1.f;
    if (k < a.vocab || p < 1.f) return row;
  }
  return a.batch;
}

// One CTA per row, persistent over rows.
__global__ void __launch_bounds__(THREADS, 2) row_kernel(Args a) {
  Shared& s = shared_state();
  Ring ring{a.depth};
  Store store = cta_store(a);
  init_ring(s, a.depth);
  int sample_size = min(STAGE, a.vocab);
  int row = next_active(a, blockIdx.x);
  while (row < a.batch) {
    bool use_k, use_p;
    unsigned kk;
    float p;
    row_params(a, row, use_k, kk, use_p, p);
    int next = next_active(a, row + gridDim.x);
    const float* next_x = next < a.batch ? a.logits + next * a.stride : nullptr;
    process_row<false>(s, store, ring, a.logits + row * a.stride, 0, a.vocab,
                       a.vocab, sample_size, a.vec, use_k, kk, use_p, p,
                       next_x);
    __syncthreads();
    row = next;
  }
}

// Clusters of `splits` CTAs, each splitting one row, persistent over rows.
__global__ void __launch_bounds__(THREADS, 2) cluster_kernel(Args a) {
  Shared& s = shared_state();
  Ring ring{a.depth};
  Store store = cta_store(a);
  init_ring(s, a.depth);
  cg::this_cluster().sync();
  int rank = int(cg::this_cluster().block_rank());
  int chunk = ((a.vocab + a.splits - 1) / a.splits + 3) / 4 * 4;
  int begin = min(a.vocab, rank * chunk), end = min(a.vocab, begin + chunk);
  int sample_size = 0;
  for (int r = 0; r < a.splits; ++r)
    sample_size +=
        min(STAGE, max(0, min(a.vocab, (r + 1) * chunk) - r * chunk));
  int clusters = gridDim.x / a.splits;
  for (int row = blockIdx.x / a.splits; row < a.batch; row += clusters) {
    bool use_k, use_p;
    unsigned kk;
    float p;
    if (row_params(a, row, use_k, kk, use_p, p))
      process_row<true>(s, store, ring, a.logits + row * a.stride, begin, end,
                        a.vocab, sample_size, a.vec, use_k, kk, use_p, p,
                        nullptr);
    // Peers finish reading this CTA's shared memory before it is reused, and
    // before it is released at exit.
    cg::this_cluster().sync();
  }
}

// ---------------------------------------------------------------------------
// Host.

struct Plan {
  int sms = 0, max_smem = 0, deep = 0, shallow = SHALLOW, max_splits = 1;
  int clusters[MAX_SPLITS + 1] = {0};  // co-resident clusters per cluster size
};

void check(cudaError_t err, const char* what) {
  STD_TORCH_CHECK(err == cudaSuccess, "top_k_top_p_filter: ", what, ": ",
                  cudaGetErrorString(err));
}

void init_plan(Plan& p, int device) {
  check(cudaDeviceGetAttribute(&p.sms, cudaDevAttrMultiProcessorCount, device),
        "SM count");
  check(cudaDeviceGetAttribute(&p.max_smem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
        "shared memory size");
  for (auto kernel : {row_kernel, cluster_kernel})
    check(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, p.max_smem),
          "shared memory attribute");
  check(cudaFuncSetAttribute(cluster_kernel,
                             cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
        "cluster attribute");
  p.deep = std::min(MAX_STAGES, (p.max_smem - RING_OFFSET) / (STAGE * 4));
  p.clusters[1] = p.sms;
  for (int splits = 2; splits <= MAX_SPLITS; splits *= 2) {
    cudaLaunchConfig_t config = {};
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = splits;
    attr.val.clusterDim.y = attr.val.clusterDim.z = 1;
    config.gridDim = dim3(splits);
    config.blockDim = dim3(THREADS);
    config.dynamicSmemBytes = RING_OFFSET + size_t(p.deep) * STAGE * 4;
    config.attrs = &attr;
    config.numAttrs = 1;
    int clusters = 0;
    if (cudaOccupancyMaxActiveClusters(&clusters, cluster_kernel, &config) !=
        cudaSuccess) {
      cudaGetLastError();
      break;
    }
    if (clusters <= 0) break;
    p.clusters[splits] = clusters;
    p.max_splits = splits;
  }
}

const Plan& plan(int device) {
  constexpr int kMaxDevices = 64;
  static Plan plans[kMaxDevices];
  static std::once_flag flags[kMaxDevices];
  STD_TORCH_CHECK(device >= 0 && device < kMaxDevices,
                  "top_k_top_p_filter: unsupported device index ", device);
  std::call_once(flags[device], init_plan, plans[device], device);
  return plans[device];
}

// Splits per row: clusters stay co-resident and keep a stage per CTA.
int choose_splits(const Plan& pl, int batch, int vocab) {
  auto fits = [&](int splits) {
    return splits <= pl.max_splits && vocab / splits >= STAGE;
  };
  // Deep rings while the CTAs fill at most half the SMs: past that, cluster
  // placement and shared barriers cost more than the shorter chunks save.
  int splits = 1;
  while (fits(splits * 2) && batch <= pl.clusters[splits * 2] &&
         batch * splits * 2 <= pl.sms / 2)
    splits *= 2;
  // Up to one row per SM, small clusters with shallow rings still cut the
  // per-row latency. This pays off only for long rows. Past one row per SM,
  // one CTA per row is faster (B300: 10% to 30% at 160 to 296 rows).
  if (splits == 1 && vocab >= 16 * STAGE && fits(2) && batch <= pl.sms)
    splits = fits(4) && batch * 4 * 8 <= 2 * pl.sms * 7 ? 4 : 2;
  return splits;
}

// At most two CTAs per SM run at once, each with its own overflow store.
int64_t workspace_bytes(int64_t sm_count) {
  return 2 * sm_count * 2 * int64_t(GCAP) * int64_t(sizeof(unsigned));
}

}  // namespace topk_topp_filter
}  // namespace vllm

// Also initializes the device plan, outside the first call of the op.
int64_t top_k_top_p_filter_workspace_size(int64_t device) {
  namespace tf = vllm::topk_topp_filter;
  const torch::stable::accelerator::DeviceGuard device_guard(int(device));
  return tf::workspace_bytes(tf::plan(int(device)).sms);
}

void top_k_top_p_filter(torch::stable::Tensor& logits,
                        const std::optional<torch::stable::Tensor>& k,
                        const std::optional<torch::stable::Tensor>& p,
                        torch::stable::Tensor& workspace) {
  namespace tf = vllm::topk_topp_filter;
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 &&
                      logits.scalar_type() == ScalarType::Float &&
                      logits.stride(1) == 1,
                  "top_k_top_p_filter: logits must be a 2D float32 CUDA tensor "
                  "with contiguous rows");
  const int64_t batch = logits.size(0), vocab = logits.size(1);
  STD_TORCH_CHECK(vocab < tf::MAX_VOCAB,
                  "top_k_top_p_filter: vocab must be below 2^20, got ", vocab);
  if (k)
    STD_TORCH_CHECK(
        k->is_contiguous() && k->scalar_type() == ScalarType::Int &&
            k->numel() == batch,
        "top_k_top_p_filter: k must be a contiguous int32 [batch] tensor");
  if (p)
    STD_TORCH_CHECK(
        p->is_contiguous() && p->scalar_type() == ScalarType::Float &&
            p->numel() == batch,
        "top_k_top_p_filter: p must be a contiguous float32 [batch] tensor");
  if (batch == 0 || (!k && !p)) return;

  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());
  const tf::Plan& pl = tf::plan(logits.get_device_index());
  STD_TORCH_CHECK(
      pl.deep >= pl.shallow,
      "top_k_top_p_filter: not enough shared memory on this device");
  STD_TORCH_CHECK(
      workspace.is_cuda() && workspace.scalar_type() == ScalarType::Byte &&
          workspace.numel() >= tf::workspace_bytes(pl.sms),
      "top_k_top_p_filter: workspace must be a uint8 CUDA tensor of "
      "top_k_top_p_filter_workspace_size(sm_count) bytes");

  const int splits = tf::choose_splits(pl, int(batch), int(vocab));
  const bool deep = batch * splits <= pl.sms;
  tf::Args a;
  a.logits = logits.mutable_data_ptr<float>();
  a.stride = logits.stride(0);
  a.ks = k ? k->const_data_ptr<int>() : nullptr;
  a.ps = p ? p->const_data_ptr<float>() : nullptr;
  a.workspace = reinterpret_cast<unsigned*>(workspace.mutable_data_ptr());
  a.batch = int(batch);
  a.vocab = int(vocab);
  a.depth = deep ? pl.deep : pl.shallow;
  a.splits = splits;
  a.vec = vocab % 4 == 0 && a.stride % 4 == 0 &&
          reinterpret_cast<uintptr_t>(a.logits) % 16 == 0;
  const size_t smem = tf::RING_OFFSET + size_t(a.depth) * tf::STAGE * 4;
  const cudaStream_t stream =
      get_current_cuda_stream(logits.get_device_index());

  cudaLaunchConfig_t config = {};
  cudaLaunchAttribute attr;
  config.blockDim = dim3(tf::THREADS);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  // Both kernels are persistent over rows, and each CTA owns workspace slot
  // blockIdx.x, so the grid stays within the 2 * SMs slots.
  const int64_t ctas = deep ? pl.sms : 2 * pl.sms;
  if (splits == 1) {
    config.gridDim = dim3(std::min<int64_t>(batch, ctas));
    tf::check(cudaLaunchKernelEx(&config, tf::row_kernel, a), "launch");
    return;
  }
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = splits;
  attr.val.clusterDim.y = attr.val.clusterDim.z = 1;
  config.gridDim = dim3(std::min<int64_t>(batch, ctas / splits) * splits);
  config.attrs = &attr;
  config.numAttrs = 1;
  tf::check(cudaLaunchKernelEx(&config, tf::cluster_kernel, a),
            "cluster launch");
}
