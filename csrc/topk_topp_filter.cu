// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// In-place top-k/top-p masking of FP32 logits with per-row device k and p
// (SM90+: thread block clusters).
//
// One launch. Each row is handled by one CTA, or at small batch by a thread
// block cluster of up to 16 CTAs that split the row:
//   1. stream the chunk once: the max of every 16 logits (level 1, shared
//      memory), of every 256 (level 2, pushed into the leader CTA), the row
//      max, and the sum of exp for top-p-only rows. No compaction, no global
//      intermediates.
//   2. the leader selects exactly through the levels. The supergroups at or
//      above a level-2 threshold hold the groups at or above a level-1
//      threshold, and those hold every logit at or above it. A threshold
//      qualifies when the maxima at or above it alone hold k logits (top-k) or
//      the nucleus mass (top-p only). Radix selection picks the highest one at
//      each level, then the exact cut among the gathered candidates.
//   3. write -inf over the chunk, then scatter the kept candidates back.
// Rows with too many candidates (huge nuclei, k in the thousands, mass ties)
// take the exact fallback: radix passes over the row by all CTAs of the row,
// then a masking pass.
//
// Semantics: exactly k tokens by logit, ties broken by lowest index (like the
// Triton kernel, while apply_top_k_top_p_pytorch keeps every tie at the k-th
// logit), then top-p on the renormalized top-k distribution, keeping the
// smallest prefix of the (logit desc, index asc) order whose mass reaches p.
// Masses are FP32 exp() rounded to 2^-44 fixed point, so sums are
// deterministic.

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>

#include "torch_utils.h"

namespace vllm {
namespace topk_topp_filter {
namespace cg = cooperative_groups;
using U64 = unsigned long long;

constexpr int THREADS = 256, WARPS = THREADS / 32;
constexpr int U = 8;                    // float4 loads per thread per round
constexpr int ROUND = THREADS * U * 4;  // elements per streaming round
constexpr int GL = 8192;                // candidate groups
constexpr int MIN_CAP = 4096;  // candidate logits in shared memory, at least
constexpr int MAX_S = 16;      // CTAs per row
constexpr int TIER_P = 64,
              TIER_P2 =
                  1024;  // top-p only rows try their best TIER_P logits first
constexpr int STORE1 =
    512;  // candidates of that tier, outside the level-1 maxima
constexpr float BIN_SCALE = 8.f;  // level bins per nat
constexpr int LAST_BIN = 255;     // the open bin, 32 nats below the max
constexpr float MASS_SCALE = 17592186044416.f;  // 2^44
constexpr unsigned ALL_IDX = 0xffffffffu;
constexpr int MAX_VOCAB = 1 << 20;

extern __shared__ __align__(16) unsigned char smem[];
__device__ __forceinline__ unsigned char* smem_base() { return smem; }

// Keep (key, idx) iff it ranks at or above the cut in (key desc, idx asc).
struct Cut {
  unsigned key, idx;
  __device__ bool keep(unsigned k, unsigned i) const {
    return k > key || (k == key && i <= idx);
  }
};
constexpr Cut ALL{0u, ALL_IDX};

struct Hist {
  unsigned cnt[256],
      limb[4][256];  // counts, and masses in 12-bit limbs (no 64-bit atomics)
};
struct Res {
  U64 above;
  unsigned digit, above_c, at_c, found;
};
struct Pick {
  unsigned key;
  U64 above;  // weight strictly above key
  unsigned above_c, at_c;
  bool found;
};
// The leader's decision, pushed into every CTA of the row.
// Candidate logits in shared memory: cap keys at offset, then cap indices.
struct Store {
  int offset, cap;
  __device__ unsigned* keys() const {
    return reinterpret_cast<unsigned*>(smem_base() + offset);
  }
  __device__ unsigned* idxs() const { return keys() + cap; }
};
struct Pub {
  Store store;
  Cut cut;
  float maximum;
  double total;
  unsigned floor, count;
  int fallback;
};
struct Shared {
  Hist hist;
  Hist* ghist;  // the leader's histogram that clustered passes combine into
  Res res;
  Pub pub;
  float max_r[MAX_S];
  double sum_r[MAX_S];
  U64 slot[2][MAX_S];
  U64 scratch[WARPS];
  unsigned counter, kept;
  int turn;
};

// Byte offsets of the dynamic arrays behind Shared.
struct Layout {
  int l2all, l2loc, slist, glist, store1, region, cap, bytes;
};
struct Args {
  float* logits;
  int64_t stride;
  const int* ks;
  const float* ps;
  int vocab, chunk, ns;
  bool vec;
  Layout ly;
};
struct Row {
  bool use_k, use_p;
  unsigned k;
  float p;
};
// This CTA's part of a row. Indices are row positions.
struct Ctx {
  float* x;
  int begin, end, vocab, chunk, ns, nsl;
  bool vec;
};

__device__ __forceinline__ Shared& sh() {
  return *reinterpret_cast<Shared*>(smem);
}

// f(i, load(i)) for i in [0, n) strided over the CTA, with BATCH loads in
// flight per thread (loops over remote or global memory are otherwise latency
// bound).
template <int BATCH = 4, class Load, class F>
__device__ __forceinline__ void batched(int n, const Load& load, const F& f) {
  for (int base = threadIdx.x; base < n; base += BATCH * THREADS) {
    decltype(load(0)) v[BATCH];
#pragma unroll
    for (int b = 0; b < BATCH; ++b)
      if (base + b * THREADS < n) v[b] = load(base + b * THREADS);
#pragma unroll
    for (int b = 0; b < BATCH; ++b)
      if (base + b * THREADS < n) f(base + b * THREADS, v[b]);
  }
}
template <class T>
__device__ __forceinline__ T* dyn(int offset) {
  return reinterpret_cast<T*>(smem + offset);
}

// The CTAs of a row: one CTA, or the cluster.
template <bool CL>
struct Grp;
template <>
struct Grp<false> {
  __device__ int size() const { return 1; }
  __device__ int rank() const { return 0; }
  __device__ void sync() const { __syncthreads(); }
  __device__ void arrive() const {}
  __device__ void wait() const {}
  template <class T>
  __device__ T* peer(T* p, int) const {
    return p;
  }
};
template <>
struct Grp<true> {
  __device__ int size() const { return int(cg::this_cluster().num_blocks()); }
  __device__ int rank() const { return int(cg::this_cluster().block_rank()); }
  __device__ void sync() const { cg::this_cluster().sync(); }
  __device__ void arrive() const {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  }
  __device__ void wait() const {
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  }
  template <class T>
  __device__ T* peer(T* p, int r) const {
    return cg::this_cluster().map_shared_rank(p, r);
  }
};

__device__ Row row_params(const Args& a, int row);
__device__ Ctx context(const Args& a, int row, int rank, int size);
template <bool CL, bool SUM>
__device__ void stream(Grp<CL> g, const Ctx& c, const Layout& ly);
template <bool CL>
__device__ void select_row(Grp<CL> g, const Ctx& c, const Row& r,
                           const Layout& ly);
template <bool CL>
__device__ Cut fallback(Grp<CL> g, const Ctx& c, const Row& r, const Pub& pub,
                        const Layout& ly);
__device__ void fill(const Ctx& c);
__device__ void mask(const Ctx& c, Cut cut);
__device__ void scatter(const Ctx& c, const Pub& pub, const Layout& ly,
                        bool leader);

// ---------------------------------------------------------------------------
// Kernel: one row per CTA (CL = false) or per cluster.

template <bool CL>
__global__ void __launch_bounds__(THREADS, 2) filter_kernel(Args a) {
  Grp<CL> g;
  g.arrive();  // the matching wait precedes the first access to a peer
  int row = blockIdx.x / g.size();
  Row r = row_params(a, row);
  Ctx c = context(a, row, g.rank(), g.size());
  if (!r.use_k && !r.use_p) return g.wait();
  if (threadIdx.x == 0) {
    sh().ghist = g.peer(dyn<Hist>(a.ly.region), 0);
    sh().turn = 0;
    sh().kept = 0;
  }
  if (r.use_p && !r.use_k)
    stream<CL, true>(g, c, a.ly);
  else
    stream<CL, false>(g, c, a.ly);
  g.sync();
  if (g.rank() == 0) select_row<CL>(g, c, r, a.ly);
  g.sync();
  const Pub pub = sh().pub;
  if (pub.fallback) return mask(c, fallback<CL>(g, c, r, pub, a.ly));
  fill(c);
  __syncthreads();
  scatter(c, pub, a.ly, g.rank() == 0);
}

__device__ Row row_params(const Args& a, int row) {
  int k = a.ks ? a.ks[row] : a.vocab;
  float p = a.ps ? a.ps[row] : 1.f;
  return Row{k < a.vocab, p < 1.f, unsigned(max(k, 1)), p};
}

__device__ Ctx context(const Args& a, int row, int rank, int size) {
  int begin = min(a.vocab, rank * a.chunk), end = min(a.vocab, begin + a.chunk);
  int nsl = (end - begin + 255) / 256;
  return Ctx{a.logits + row * a.stride,
             begin,
             end,
             a.vocab,
             a.chunk,
             a.ns,
             nsl,
             a.vec};
}

// ---------------------------------------------------------------------------
// Phase 1: stream the chunk once.

__device__ void load_round(const Ctx& c, int base, float4 (&v)[U]);
__device__ __forceinline__ float round_sum(const float4 (&v)[U], float m);
__device__ void reduce_round(const Ctx& c, int base, const float (&mj)[U],
                             float* l1, float* l2);
template <class T, class Op>
__device__ T block_reduce(T v, Op op);
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
__device__ __forceinline__ double rescale(double sum, float from, float to) {
  return from == -INFINITY ? 0.0 : sum * double(expf(from - to));
}

template <bool CL, bool SUM>
__device__ void stream(Grp<CL> g, const Ctx& c, const Layout& ly) {
  float* l1 = dyn<float>(ly.region);
  float* l2 = dyn<float>(ly.l2loc);
  float m = -INFINITY;
  double sum = 0.0;  // of exp(x - m): pairwise in FP32 within a round, FP64
                     // across rounds
  float4 v[U];
  if (c.begin < c.end) load_round(c, c.begin, v);
  for (int base = c.begin; base < c.end; base += ROUND) {
    float4 nv[U];
    if (base + ROUND < c.end) load_round(c, base + ROUND, nv);
    float mj[U], mr = -INFINITY;
#pragma unroll
    for (int j = 0; j < U; ++j) {
      mj[j] = fmaxf(fmaxf(v[j].x, v[j].y), fmaxf(v[j].z, v[j].w));
      mr = fmaxf(mr, mj[j]);
    }
    reduce_round(c, base, mj, l1, l2);
    if (SUM && mr > m) sum = rescale(sum, m, mr);
    m = fmaxf(m, mr);
    if (SUM && m != -INFINITY) sum += round_sum(v, m);
#pragma unroll
    for (int j = 0; j < U; ++j) v[j] = nv[j];
  }
  float rmax = block_reduce(m, Max());
  double rsum = SUM ? block_reduce(rescale(sum, m, rmax), Sum()) : 0.0;
  g.wait();
  Shared& s = sh();
  int rank = g.rank();
  if (threadIdx.x == 0) {
    *g.peer(&s.max_r[rank], 0) = rmax;
    *g.peer(&s.sum_r[rank], 0) = rsum;
  }
  if (CL) {
    float* dst = g.peer(dyn<float>(ly.l2all), 0) + rank * (c.chunk / 256);
    for (int i = threadIdx.x; i < c.nsl; i += THREADS) dst[i] = l2[i];
  }
}

// Sum of exp(x - m) over the lane's round. __expf after the exact-ish
// subtraction keeps each term within a few ulp, the pairwise sum within 5
// roundings.
__device__ __forceinline__ float round_sum(const float4 (&v)[U], float m) {
  float e[U];
#pragma unroll
  for (int j = 0; j < U; ++j)
    e[j] = (__expf(v[j].x - m) + __expf(v[j].y - m)) +
           (__expf(v[j].z - m) + __expf(v[j].w - m));
#pragma unroll
  for (int w = U / 2; w; w /= 2)
#pragma unroll
    for (int j = 0; j < w; ++j) e[j] += e[j + w];
  return e[0];
}

// Warp w holds U * 128 consecutive logits, float4 j of lane l at j * 128 + 4 l.
// A group of 16 is 4 lanes of one j, a supergroup of 256 is j pair (2u, 2u +
// 1). Transposing butterflies leave each lane two group maxima, then four lanes
// one supergroup max each.
__device__ void reduce_round(const Ctx& c, int base, const float (&mj)[U],
                             float* l1, float* l2) {
  static_assert(U == 8, "the transposing butterflies assume 8 float4 per lane");
  int lane = threadIdx.x & 31, b0 = lane & 1, b1 = (lane >> 1) & 1;
  int offset = base - c.begin + (threadIdx.x >> 5) * U * 128;
  float r[4], q[2];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float keep = b0 ? mj[2 * i + 1] : mj[2 * i],
          send = b0 ? mj[2 * i] : mj[2 * i + 1];
    r[i] = fmaxf(keep, __shfl_xor_sync(~0u, send, 1));  // j = 2 i + b0
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    float keep = b1 ? r[2 * i + 1] : r[2 * i],
          send = b1 ? r[2 * i] : r[2 * i + 1];
    q[i] = fmaxf(keep, __shfl_xor_sync(~0u, send, 2));  // j = 4 i + 2 b1 + b0
    l1[offset / 16 + (4 * i + 2 * b1 + b0) * 8 + (lane >> 2)] = q[i];
  }
  float z = fmaxf(b0 ? q[1] : q[0],
                  __shfl_xor_sync(~0u, b0 ? q[0] : q[1], 1));  // u = 2 b0 + b1
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) z = fmaxf(z, __shfl_xor_sync(~0u, z, o));
  int sg = offset / 256 + 2 * b0 + b1;
  if (lane < 4 && sg < c.nsl) l2[sg] = z;
}

__device__ __forceinline__ float4 load4(const float* x, int i, int end,
                                        bool vec) {
  if (vec && i + 3 < end) return *reinterpret_cast<const float4*>(x + i);
  float4 v;
  v.x = i < end ? x[i] : -INFINITY;
  v.y = i + 1 < end ? x[i + 1] : -INFINITY;
  v.z = i + 2 < end ? x[i + 2] : -INFINITY;
  v.w = i + 3 < end ? x[i + 3] : -INFINITY;
  return v;
}

__device__ void load_round(const Ctx& c, int base, float4 (&v)[U]) {
  int i = base + (threadIdx.x >> 5) * U * 128 + (threadIdx.x & 31) * 4;
#pragma unroll
  for (int j = 0; j < U; ++j) v[j] = load4(c.x, i + j * 128, c.end, c.vec);
}

// ---------------------------------------------------------------------------
// Phase 2: selection by the leader.

__device__ __forceinline__ unsigned ordered(float x) {
  unsigned bits = __float_as_uint(x == 0.f ? 0.f : x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
__device__ __forceinline__ float from_ordered(unsigned key) {
  return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}
__device__ __forceinline__ U64 fixed_mass(unsigned key, float maximum) {
  return __float2ull_rn(expf(from_ordered(key) - maximum) * MASS_SCALE);
}
__device__ __forceinline__ U64 mass_target(float p, double total) {
  double t = ceil(double(p) * total);
  return t < 1.0 ? 1ull : U64(t);
}

// Value bins of 1/BIN_SCALE nat below the row max, the last one open-ended.
// Bins are monotone in the logit, so "bin <= b" is "logit >= some threshold".
struct Bins {
  float maximum;
  __device__ int of(unsigned key) const {
    float d = (maximum - from_ordered(key)) * BIN_SCALE;
    return d < float(LAST_BIN) ? int(d) : LAST_BIN;
  }
};

template <class Items>
__device__ int level(const Items& items, U64 n, Bins bins);
template <class Items, class Keep>
__device__ int compact(const Items& items, const Keep& keep, uint16_t* list,
                       int cap);
__device__ unsigned gather(const Ctx& c, const Layout& ly, Store st, int ng,
                           Bins bins, int b1, U64* mass, unsigned& lo);
template <bool CL, class Items>
__device__ Cut finish(Grp<CL> g, const Items& items, const Row& r,
                      float maximum, double total, unsigned lo);
__device__ Cut finish_small(const unsigned* keys, const unsigned* idxs,
                            unsigned m, unsigned* sorted, const Row& r,
                            float maximum, double total);
template <bool CL>
__device__ void push_kept(Grp<CL> g, const Ctx& c, const Layout& ly,
                          const Pub& pub);
template <bool MASS = true>
__device__ void clear_hist(Hist& h);

template <bool CL>
__device__ __forceinline__ float l1_value(Grp<CL> g, const Ctx& c,
                                          const Layout& ly, unsigned grp) {
  if (!CL) return dyn<float>(ly.region)[grp];
  unsigned owner = grp / unsigned(c.chunk / 16);
  return g.peer(dyn<float>(ly.region),
                int(owner))[grp - owner * unsigned(c.chunk / 16)];
}

// Maxima grp to grp + 3 (grp a multiple of 4, within one CTA's chunk).
template <bool CL>
__device__ __forceinline__ float4 l1_values(Grp<CL> g, const Ctx& c,
                                            const Layout& ly, unsigned grp) {
  const float4* l1 = reinterpret_cast<const float4*>(dyn<float>(ly.region));
  if (!CL) return l1[grp / 4];
  unsigned owner = grp / unsigned(c.chunk / 16);
  return g.peer(l1, int(owner))[(grp - owner * unsigned(c.chunk / 16)) / 4];
}

// The candidate groups for the best n logits: those whose max reaches bin b1,
// the bin of the n-th best group max, found within the supergroups whose max
// reaches the bin of the n-th best supergroup max. Returns b1 and their number
// (glist holds GL).
template <bool CL>
__device__ int candidate_groups(Grp<CL> g, const Ctx& c, const Layout& ly,
                                U64 n, Bins bins, int& ng) {
  const float* l2 = dyn<float>(ly.l2all);
  uint16_t* slist = dyn<uint16_t>(ly.slist);
  auto l2_items = [&](auto f) {
    for (int i = threadIdx.x; i < c.ns; i += THREADS)
      f(ordered(l2[i]), unsigned(i));
  };
  int b2 = level(l2_items, n, bins);
  int ns = compact(
      l2_items, [=](unsigned key) { return bins.of(key) <= b2; }, slist, c.ns);
  auto l1_items = [&](auto f) {  // 4 maxima per load
    auto group = [&](int j) {
      return ns == c.ns ? 4u * j : slist[j >> 2] * 16u + (j & 3) * 4;
    };
    batched(
        ns * 4, [&](int j) { return l1_values(g, c, ly, group(j)); },
        [&](int j, float4 v) {
          unsigned grp = group(j);
          f(ordered(v.x), grp);
          f(ordered(v.y), grp + 1);
          f(ordered(v.z), grp + 2);
          f(ordered(v.w), grp + 3);
        });
  };
  int b1 = level(l1_items, n, bins);
  ng = compact(
      l1_items, [=](unsigned key) { return bins.of(key) <= b1; },
      dyn<uint16_t>(ly.glist), GL);
  return b1;
}

// Top-k (also with top-p): the logits reaching the bin of the k-th best group
// max. Returns the lowest candidate key.
template <bool CL>
__device__ unsigned candidates(Grp<CL> g, const Ctx& c, const Layout& ly,
                               Store st, U64 k, Bins bins, unsigned& count) {
  int ng;
  int b1 = candidate_groups<CL>(g, c, ly, k, bins, ng);
  unsigned lo = 0;
  count = ng <= GL ? gather(c, ly, st, ng, bins, b1, nullptr, lo) : ~0u;
  return lo;
}

// Top-p only: the best TIER_P logits, then the best TIER_P2, settle every
// nucleus inside them. Their exact mass decides, and the missing mass bounds
// the nucleus size from below (it comes from logits below the tier's lowest).
// Larger nuclei take the fallback (count beyond the store, floor 0).
template <bool CL>
__device__ unsigned nucleus_candidates(Grp<CL> g, const Ctx& c,
                                       const Layout& ly, Store& st, U64 target,
                                       Bins bins, unsigned& count) {
  Store big = st;
  for (int tier = 0; tier < 2; ++tier) {
    int ng;
    U64 mass = 0;
    unsigned lo = 0;
    int b1 = candidate_groups<CL>(g, c, ly, tier ? TIER_P2 : TIER_P, bins, ng);
    st = tier ? big : Store{ly.store1, STORE1};
    count = ng <= GL ? gather(c, ly, st, ng, bins, b1, &mass, lo) : ~0u;
    if (ng <= GL && mass >= target) {
      if (count > unsigned(st.cap) && tier == 0) {
        st = big;
        count = gather(c, ly, st, ng, bins, b1, nullptr, lo);
      }
      return lo;
    }
    U64 unit = max(fixed_mass(lo, bins.maximum), 1ull);
    if (ng > GL ||
        count + (target - mass + unit - 1) / unit > unsigned(TIER_P2))
      break;
  }
  st = big;
  count = ~0u;
  return 0u;
}

template <bool CL>
__device__ void select_row(Grp<CL> g, const Ctx& c, const Row& r,
                           const Layout& ly) {
  Shared& s = sh();
  float maximum = -INFINITY;
  for (int i = 0; i < g.size(); ++i) maximum = fmaxf(maximum, s.max_r[i]);
  double total = 0.0;
  for (int i = 0; i < g.size(); ++i)
    total += rescale(s.sum_r[i], s.max_r[i], maximum);
  U64 target = r.use_k ? r.k : mass_target(r.p, total * double(MASS_SCALE));
  Store st{ly.region, ly.cap};
  Bins bins{maximum};
  unsigned count;
  unsigned lo = r.use_k
                    ? candidates<CL>(g, c, ly, st, target, bins, count)
                    : nucleus_candidates<CL>(g, c, ly, st, target, bins, count);
  Pub pub{st, ALL, maximum, total, lo, count, count > unsigned(st.cap)};
  const unsigned* keys = st.keys();
  const unsigned* idxs = st.idxs();
  auto store_items = [&](auto f) {
    for (unsigned i = threadIdx.x; i < count; i += THREADS) f(keys[i], idxs[i]);
  };
  if (count <= THREADS)
    pub.cut = finish_small(keys, idxs, count, dyn<unsigned>(ly.glist), r,
                           maximum, total);
  else if (!pub.fallback)
    pub.cut = finish(Grp<false>(), store_items, r, maximum, total, lo);
  else
    clear_hist(*dyn<Hist>(ly.region));
  if (CL && !pub.fallback) push_kept(g, c, ly, pub);
  __syncthreads();
  if (threadIdx.x < g.size()) *g.peer(&s.pub, threadIdx.x) = pub;
}

// The kept candidates of the peers' chunks, into the peers' level-1 space (free
// by now), so every CTA scatters its own chunk after its fill, without another
// barrier.
template <bool CL>
__device__ void push_kept(Grp<CL> g, const Ctx& c, const Layout& ly,
                          const Pub& pub) {
  const unsigned* keys = pub.store.keys();
  const unsigned* idxs = pub.store.idxs();
  Store dst{ly.region, ly.cap};
  for (unsigned i = threadIdx.x; i < pub.count; i += THREADS) {
    unsigned key = keys[i], idx = idxs[i];
    int owner = int(idx) / c.chunk;
    if (owner == 0 || !pub.cut.keep(key, idx)) continue;
    unsigned pos = atomicAdd(g.peer(&sh().kept, owner), 1u);
    g.peer(dst.keys(), owner)[pos] = key;
    g.peer(dst.idxs(), owner)[pos] = idx;
  }
}

template <bool CL, class Items>
__device__ bool binned_nucleus(Grp<CL> g, const Ctx& c, const Layout& ly,
                               const Items& items, const Row& r, const Pub& pub,
                               Cut& cut);
template <bool CL, bool MASS, class Items>
__device__ Cut select(Grp<CL> g, const Items& items, Cut within, U64 target,
                      unsigned lo, unsigned hi, float maximum);
template <bool CL, bool MASS>
__device__ Res combine_scan(Grp<CL> g, U64 need);
template <bool MASS>
__device__ __forceinline__ void hist_add(Hist& h, unsigned d, U64 w);
__device__ __forceinline__ unsigned append(unsigned* counter, unsigned n);
template <bool CL>
__device__ U64 group_sum(Grp<CL> g, U64 v);

// Exact fallback over this CTA's chunk of the row, with the histograms combined
// across the cluster: for top-p only the boundary value bin, else radix passes.
template <bool CL>
__device__ Cut fallback(Grp<CL> g, const Ctx& c, const Row& r, const Pub& pub,
                        const Layout& ly) {
  auto row_items = [&](auto f) {
    int n = (c.end - c.begin + 3) / 4;
    batched(
        n, [&](int j) { return load4(c.x, c.begin + 4 * j, c.end, c.vec); },
        [&](int j, float4 v) {
          unsigned i = unsigned(c.begin + 4 * j);
          f(ordered(v.x), i);
          if (i + 1 < unsigned(c.end)) f(ordered(v.y), i + 1);
          if (i + 2 < unsigned(c.end)) f(ordered(v.z), i + 2);
          if (i + 3 < unsigned(c.end)) f(ordered(v.w), i + 3);
        });
  };
  Cut cut;
  if (!r.use_k && binned_nucleus(g, c, ly, row_items, r, pub, cut)) return cut;
  cut = finish(g, row_items, r, pub.maximum, pub.total, pub.floor);
  return cut;
}

// Top-p only, AIR style: the value bin where the row's mass reaches the
// nucleus, then the exact cut among that bin's logits, gathered into the
// leader. False (before any state changes) if the bin does not fit the leader's
// store.
template <bool CL, class Items>
__device__ bool binned_nucleus(Grp<CL> g, const Ctx& c, const Layout& ly,
                               const Items& items, const Row& r, const Pub& pub,
                               Cut& cut) {
  Shared& s = sh();
  Bins bins{pub.maximum};
  U64 target = mass_target(r.p, pub.total * double(MASS_SCALE));
  clear_hist<true>(s.hist);
  __syncthreads();
  items([&](unsigned key, unsigned) {
    int b =
        bins.of(key);  // the open bin stays out: a boundary there fails anyway
    if (b < LAST_BIN)
      hist_add<true>(s.hist, LAST_BIN - b, fixed_mass(key, bins.maximum));
  });
  __syncthreads();
  Res res = combine_scan<CL, true>(g, target);
  if (!res.found || res.at_c > unsigned(ly.cap)) return false;
  int b = LAST_BIN - int(res.digit);
  Store st{ly.region, ly.cap};
  unsigned* counter = g.peer(&s.kept, 0);
  unsigned* keys = g.peer(st.keys(), 0);
  unsigned* idxs = g.peer(st.idxs(), 0);
  items([&](unsigned key, unsigned idx) {
    bool take = bins.of(key) == b;
    unsigned pos = append(counter, take ? 1u : 0u);
    if (!take) return;
    keys[pos] = key;
    idxs[pos] = idx;
  });
  g.sync();
  if (g.rank() == 0) {
    unsigned n = s.kept;
    auto bin_items = [&](auto f) {
      for (unsigned i = threadIdx.x; i < n; i += THREADS)
        f(st.keys()[i], st.idxs()[i]);
    };
    Cut found =
        select<false, true>(Grp<false>(), bin_items, ALL, target - res.above,
                            0u, ordered(bins.maximum), bins.maximum);
    __syncthreads();
    if (threadIdx.x < g.size()) g.peer(&s.pub, threadIdx.x)->cut = found;
  }
  g.sync();
  cut = s.pub.cut;
  return true;
}

// ---------------------------------------------------------------------------
// Exact cut over candidates: top-k, then top-p on the renormalized top-k.

template <bool CL, bool MASS>
__device__ Res combine_scan(Grp<CL> g, U64 need);
template <bool MASS>
__device__ __forceinline__ void hist_add(Hist& h, unsigned d, U64 w);
template <bool CL, bool MASS, class Items, class Map>
__device__ Pick radix(Grp<CL> g, const Items& items, const Map& map, U64 target,
                      unsigned lo, unsigned hi, float maximum, unsigned stop);
template <bool CL, bool MASS, class Items>
__device__ Cut select(Grp<CL> g, const Items& items, Cut within, U64 target,
                      unsigned lo, unsigned hi, float maximum);
template <bool CL, class Items>
__device__ U64 cut_mass(Grp<CL> g, const Items& items, Cut cut, unsigned lo,
                        float maximum);

template <bool CL, class Items>
__device__ Cut finish(Grp<CL> g, const Items& items, const Row& r,
                      float maximum, double total, unsigned lo) {
  unsigned hi = ordered(maximum);
  Cut cut =
      r.use_k ? select<CL, false>(g, items, ALL, r.k, lo, hi, maximum) : ALL;
  if (!r.use_p) return cut;
  double mass = r.use_k ? double(cut_mass(g, items, cut, lo, maximum))
                        : total * double(MASS_SCALE);
  return select<CL, true>(g, items, cut, mass_target(r.p, mass), lo, hi,
                          maximum);
}

template <class T>
__device__ T block_scan(T v, T& total);

// The same cut for at most THREADS candidates: each thread ranks one by
// comparing it with all others, then a prefix sum of the masses in rank order.
__device__ Cut finish_small(const unsigned* keys, const unsigned* idxs,
                            unsigned m, unsigned* sorted, const Row& r,
                            float maximum, double total) {
  unsigned i = threadIdx.x, key = 0, idx = 0, rank = ALL_IDX;
  if (i < m) {
    key = keys[i];
    idx = idxs[i];
    rank = 0;
    for (unsigned j = 0; j < m; ++j) {
      unsigned kj = keys[j], ij = idxs[j];
      rank += kj > key || (kj == key && ij < idx);
    }
  }
  unsigned* skey = sorted;
  unsigned* sidx = sorted + THREADS;
  __syncthreads();
  if (rank < m) {
    skey[rank] = key;
    sidx[rank] = idx;
  }
  __syncthreads();
  unsigned n = r.use_k ? min(r.k, m) : m, cut = n - 1;
  if (r.use_p) {
    U64 sum,
        mass = block_scan(i < n ? fixed_mass(skey[i], maximum) : 0ull, sum);
    U64 target =
        mass_target(r.p, r.use_k ? double(sum) : total * double(MASS_SCALE));
    cut = min(cut, unsigned(__syncthreads_count(i < n && mass < target)));
  }
  return Cut{skey[cut], sidx[cut]};
}

// The smallest prefix of the items within `within` (from the top) whose weight
// (count or mass) reaches target, ties by lowest index.
template <bool CL, bool MASS, class Items>
__device__ Cut select(Grp<CL> g, const Items& items, Cut within, U64 target,
                      unsigned lo, unsigned hi, float maximum) {
  auto eligible = [=](unsigned key, unsigned idx, unsigned& rk) {
    rk = key;
    return key >= lo && within.keep(key, idx);
  };
  Pick pk = radix<CL, MASS>(g, items, eligible, target, lo, hi, maximum, 0);
  if (!pk.found) return within;
  U64 unit = MASS ? fixed_mass(pk.key, maximum) : 1ull;
  U64 need = (target - pk.above + unit - 1) / unit;
  if (need >= pk.at_c)
    return Cut{pk.key, within.key == pk.key ? within.idx : ALL_IDX};
  unsigned key = pk.key;
  auto tie = [=](unsigned k, unsigned idx, unsigned& rk) {
    rk = ~idx;
    return k == key && within.keep(k, idx);
  };
  Pick pi = radix<CL, false>(g, items, tie, need, ~unsigned(MAX_VOCAB - 1), ~0u,
                             maximum, 0);
  return Cut{key, ~pi.key};
}

template <bool CL>
__device__ U64 group_sum(Grp<CL> g, U64 v);

template <bool CL, class Items>
__device__ U64 cut_mass(Grp<CL> g, const Items& items, Cut cut, unsigned lo,
                        float maximum) {
  U64 mass = 0;
  items([&](unsigned key, unsigned idx) {
    if (key >= lo && cut.keep(key, idx)) mass += fixed_mass(key, maximum);
  });
  return group_sum(g, block_reduce(mass, Sum()));
}

// Level threshold: the first bin from the top where the items' count reaches n,
// LAST_BIN if it never does. One histogram pass (the open bin stays out, it
// only matters when nothing else reaches n).
template <class Items>
__device__ int level(const Items& items, U64 n, Bins bins) {
  Shared& s = sh();
  clear_hist<false>(s.hist);
  __syncthreads();
  items([&](unsigned key, unsigned) {
    int b = bins.of(key);
    if (b < LAST_BIN) hist_add<false>(s.hist, LAST_BIN - b, 0ull);
  });
  __syncthreads();
  Res res = combine_scan<false, false>(Grp<false>(), n);
  return res.found ? LAST_BIN - int(res.digit) : LAST_BIN;
}

// ---------------------------------------------------------------------------
// Radix selection with 8-bit digits over the bits where lo and hi differ.

template <bool MASS>
__device__ void scan_hist(const Hist& h, U64 need, Res& out);
__device__ __forceinline__ int common_start(unsigned lo, unsigned hi,
                                            unsigned& prefix) {
  unsigned diff = lo ^ hi;
  int start = diff ? 32 - __clz(diff) : 0;
  prefix = start >= 32 ? 0u : hi & ~((1u << start) - 1u);
  return start;
}

template <bool MASS>
__device__ __forceinline__ void hist_add(Hist& h, unsigned d, U64 w) {
  atomicAdd(&h.cnt[d], 1u);
  if (!MASS) return;
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    unsigned part = unsigned(w >> (12 * l)) & 0xfffu;
    if (part) atomicAdd(&h.limb[l][d], part);
  }
}

template <bool CL, bool MASS, class Items, class Map>
__device__ Pick radix(Grp<CL> g, const Items& items, const Map& map, U64 target,
                      unsigned lo, unsigned hi, float maximum, unsigned stop) {
  Shared& s = sh();
  unsigned prefix;
  int start = common_start(lo, hi, prefix);
  Pick pk{prefix, 0ull, 0u, 0u, true};
  for (;;) {
    int shift = max(start - 8, 0);
    unsigned dmask = (1u << (start - shift)) - 1u;
    clear_hist<MASS>(s.hist);
    __syncthreads();
    items([&](unsigned key, unsigned idx) {
      unsigned rk;
      if (!map(key, idx, rk) ||
          (start < 32 && (rk >> start) != (prefix >> start)))
        return;
      hist_add<MASS>(s.hist, (rk >> shift) & dmask,
                     MASS ? fixed_mass(key, maximum) : 0ull);
    });
    __syncthreads();
    Res res = combine_scan<CL, MASS>(g, target - pk.above);
    if (!res.found) {
      pk.found = false;
      return pk;
    }
    prefix |= res.digit << shift;
    pk.above += res.above;
    pk.above_c += res.above_c;
    pk.at_c = res.at_c;
    start = shift;
    if (start == 0 || (stop && res.at_c <= stop)) break;
  }
  pk.key = prefix;
  return pk;
}

// Combines the CTAs' histograms in the leader, which scans them and pushes the
// result to every CTA.
template <bool CL, bool MASS>
__device__ Res combine_scan(Grp<CL> g, U64 need) {
  Shared& s = sh();
  Hist* gh = CL ? s.ghist : &s.hist;
  if (CL) {
    for (int b = threadIdx.x; b < 256; b += THREADS) {
      if (s.hist.cnt[b]) atomicAdd(&gh->cnt[b], s.hist.cnt[b]);
      if (MASS)
        for (int l = 0; l < 4; ++l)
          if (s.hist.limb[l][b]) atomicAdd(&gh->limb[l][b], s.hist.limb[l][b]);
    }
    g.sync();
  }
  if (g.rank() == 0) {
    if (threadIdx.x < 32) scan_hist<MASS>(*gh, need, s.res);
    if (CL) {
      __syncthreads();
      if (threadIdx.x < g.size()) *g.peer(&s.res, threadIdx.x) = s.res;
      clear_hist<MASS>(*gh);
    }
  }
  g.sync();
  return s.res;
}

__device__ __forceinline__ U64 bin_weight(const Hist& h, int b, bool mass) {
  if (!mass) return h.cnt[b];
  return U64(h.limb[0][b]) + (U64(h.limb[1][b]) << 12) +
         (U64(h.limb[2][b]) << 24) + (U64(h.limb[3][b]) << 36);
}

// Warp 0: the bin, from the top, where the cumulative weight first reaches
// need.
template <bool MASS>
__device__ void scan_hist(const Hist& h, U64 need, Res& out) {
  int lane = threadIdx.x;
  U64 w[8], ws = 0;
  unsigned cn[8], cs = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int b = 255 - lane * 8 - i;
    cn[i] = h.cnt[b];
    w[i] = bin_weight(h, b, MASS);
    ws += w[i];
    cs += cn[i];
  }
  U64 wi = ws;
  unsigned ci = cs;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    U64 tw = __shfl_up_sync(~0u, wi, o);
    unsigned tc = __shfl_up_sync(~0u, ci, o);
    if (lane >= o) {
      wi += tw;
      ci += tc;
    }
  }
  unsigned hit = __ballot_sync(~0u, wi >= need);
  if (!hit) {
    if (lane == 0) out.found = 0;
    return;
  }
  if (lane != __ffs(hit) - 1) return;
  U64 acc = wi - ws;
  unsigned ca = ci - cs;
  for (int i = 0; i < 8; ++i) {
    if (acc + w[i] >= need) {
      out = Res{acc, unsigned(255 - lane * 8 - i), ca, cn[i], 1u};
      return;
    }
    acc += w[i];
    ca += cn[i];
  }
}

// ---------------------------------------------------------------------------
// Candidates and output.

// n slots of a list for each calling thread, one atomic per warp.
__device__ __forceinline__ unsigned append(unsigned* counter,
                                           unsigned n) {  // n < 8
  unsigned active = __activemask(), lt = (1u << (threadIdx.x & 31)) - 1;
  unsigned b0 = __ballot_sync(active, n & 1), b1 = __ballot_sync(active, n & 2),
           b2 = __ballot_sync(active, n & 4);
  unsigned total = __popc(b0) + 2 * __popc(b1) + 4 * __popc(b2);
  if (!total) return 0u;
  int leader = __ffs(active) - 1;
  unsigned base = 0;
  if (int(threadIdx.x & 31) == leader) base = atomicAdd(counter, total);
  return __shfl_sync(active, base, leader) + __popc(b0 & lt) +
         2 * __popc(b1 & lt) + 4 * __popc(b2 & lt);
}

// The indices of the items to keep, in item order per thread: count, scan,
// write (no atomics, whose latency chains dominate long sparse lists). Returns
// their number.
template <class Items, class Keep>
__device__ int compact(const Items& items, const Keep& keep, uint16_t* list,
                       int cap) {
  unsigned n = 0, total;
  items([&](unsigned key, unsigned) { n += keep(key); });
  unsigned pos = block_scan(n, total) - n;
  items([&](unsigned key, unsigned idx) {
    if (!keep(key)) return;
    if (pos < unsigned(cap)) list[pos] = uint16_t(idx);
    ++pos;
  });
  __syncthreads();
  return int(total);
}

// The logits reaching bin b1 of the candidate groups, from global memory (L2),
// their lowest key, and optionally the mass of all of them (also those beyond
// the store).
__device__ unsigned gather(const Ctx& c, const Layout& ly, Store st, int ng,
                           Bins bins, int b1, U64* mass, unsigned& lo) {
  Shared& s = sh();
  unsigned* keys = st.keys();
  unsigned* idxs = st.idxs();
  U64 sum = 0;
  unsigned low = ~0u;
  const uint16_t* glist = dyn<uint16_t>(ly.glist);
  __syncthreads();
  if (threadIdx.x == 0) s.counter = 0;
  __syncthreads();
  auto at = [&](int j) { return glist[j >> 2] * 16 + (j & 3) * 4; };
  batched(
      ng * 4, [&](int j) { return load4(c.x, at(j), c.vocab, c.vec); },
      [&](int j, float4 v) {
        int i = at(j);
        float e[4] = {v.x, v.y, v.z, v.w};
        bool take[4];
        unsigned n = 0;
#pragma unroll
        for (int q = 0; q < 4; ++q)
          n += take[q] = i + q < c.vocab && bins.of(ordered(e[q])) <= b1;
        unsigned pos = append(&s.counter, n);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          unsigned key = ordered(e[q]);
          if (!take[q]) continue;
          low = min(low, key);
          if (mass) sum += fixed_mass(key, bins.maximum);
          if (pos < unsigned(st.cap)) {
            keys[pos] = key;
            idxs[pos] = unsigned(i + q);
          }
          ++pos;
        }
      });
  if (mass) *mass = block_reduce(sum, Sum());
  lo = block_reduce(low, Min());
  return s.counter;
}

__device__ __forceinline__ void store4(float* x, int i, int end, bool vec,
                                       float4 v) {
  if (vec && i + 3 < end) {
    *reinterpret_cast<float4*>(x + i) = v;
    return;
  }
  if (i < end) x[i] = v.x;
  if (i + 1 < end) x[i + 1] = v.y;
  if (i + 2 < end) x[i + 2] = v.z;
  if (i + 3 < end) x[i + 3] = v.w;
}

__device__ void fill(const Ctx& c) {
  float4 ninf = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
#pragma unroll 4
  for (int i = c.begin + threadIdx.x * 4; i < c.end; i += THREADS * 4)
    store4(c.x, i, c.end, c.vec, ninf);
}

__device__ __forceinline__ float masked(float y, unsigned i, Cut cut) {
  return cut.keep(ordered(y), i) ? y : -INFINITY;
}

__device__ void mask(const Ctx& c, Cut cut) {
#pragma unroll 4
  for (int i = c.begin + threadIdx.x * 4; i < c.end; i += THREADS * 4) {
    float4 v = load4(c.x, i, c.end, c.vec);
    v = make_float4(masked(v.x, i, cut), masked(v.y, i + 1, cut),
                    masked(v.z, i + 2, cut), masked(v.w, i + 3, cut));
    store4(c.x, i, c.end, c.vec, v);
  }
}

// The kept logits of this CTA's chunk: the leader's from its store, the peers'
// from the lists the leader pushed.
__device__ void scatter(const Ctx& c, const Pub& pub, const Layout& ly,
                        bool leader) {
  Store own = leader ? pub.store : Store{ly.region, ly.cap};
  const unsigned* keys = own.keys();
  const unsigned* idxs = own.idxs();
  unsigned n = leader ? pub.count : sh().kept;
  for (unsigned i = threadIdx.x; i < n; i += THREADS) {
    unsigned idx = idxs[i];
    if (!leader || (pub.cut.keep(keys[i], idx) && int(idx) < c.end))
      c.x[idx] = from_ordered(keys[i]);
  }
}

// ---------------------------------------------------------------------------
// Reductions.

template <bool MASS>
__device__ void clear_hist(Hist& h) {
  unsigned* w = reinterpret_cast<unsigned*>(&h);
  int words = MASS ? int(sizeof(Hist) / 4) : 256;
  for (int i = threadIdx.x; i < words; i += THREADS) w[i] = 0u;
}

template <class T, class Op>
__device__ T block_reduce(T v, Op op) {
  static_assert(sizeof(T) <= sizeof(U64));
#pragma unroll
  for (int o = 16; o; o >>= 1) v = op(v, __shfl_xor_sync(~0u, v, o));
  T* scratch = reinterpret_cast<T*>(sh().scratch);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
  __syncthreads();
  v = scratch[0];
#pragma unroll
  for (int i = 1; i < WARPS; ++i) v = op(v, scratch[i]);
  return v;
}

// Inclusive prefix sum over the CTA's threads, and the total.
template <class T>
__device__ T block_scan(T v, T& total) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    T t = __shfl_up_sync(~0u, v, o);
    if (lane >= o) v += t;
  }
  T* scratch = reinterpret_cast<T*>(sh().scratch);
  __syncthreads();
  if (lane == 31) scratch[warp] = v;
  __syncthreads();
  T before = 0;
  total = 0;
#pragma unroll
  for (int w = 0; w < WARPS; ++w) {
    before += w < warp ? scratch[w] : T(0);
    total += scratch[w];
  }
  return before + v;
}

// The sum of the CTAs' values (each uniform within its CTA): every CTA pushes
// into every peer, then one barrier. Slot sets alternate, so a slot is
// rewritten only after its owner passed the next barrier.
template <bool CL>
__device__ U64 group_sum(Grp<CL> g, U64 v) {
  if (!CL) return v;
  Shared& s = sh();
  int turn = s.turn;
  if (threadIdx.x < g.size()) *g.peer(&s.slot[turn][g.rank()], threadIdx.x) = v;
  g.sync();
  U64 total = 0;
  for (int i = 0; i < g.size(); ++i) total += s.slot[turn][i];
  __syncthreads();
  if (threadIdx.x == 0) s.turn = turn ^ 1;
  return total;
}

// ---------------------------------------------------------------------------
// Host.

struct Plan {
  int sms = 0, max_smem = 0;
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
  for (auto kernel : {filter_kernel<false>, filter_kernel<true>})
    check(cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, p.max_smem),
          "shared memory attribute");
  check(cudaFuncSetAttribute(filter_kernel<true>,
                             cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
        "cluster attribute");
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

int round_up(int x, int m) { return (x + m - 1) / m * m; }

Layout make_layout(int vocab, int splits, int chunk) {
  int ns = (vocab + 255) / 256, l1_bytes = round_up(chunk, ROUND) / 16 * 4;
  int region = std::max({l1_bytes, MIN_CAP * 8, int(sizeof(Hist))});
  Layout ly;
  int off = round_up(sizeof(Shared), 16);
  ly.l2all = off;
  off += round_up(ns * 4, 16);
  ly.l2loc = ly.l2all;
  if (splits > 1) {
    ly.l2loc = off;
    off += round_up(chunk / 256 * 4, 16);
  }
  ly.slist = off;
  off += round_up(ns * 2, 16);
  ly.glist = off;
  off += GL * 2;
  ly.store1 = off;
  off += STORE1 * 8;
  ly.region = off;
  ly.cap = region / 8;
  ly.bytes = off + region;
  return ly;
}

int active_clusters(int device, int vocab, int splits);

// CTAs per row: clusters while all CTAs stay resident at two per SM, all
// clusters fit at once (else a second wave doubles the time), and each CTA
// streams a round at least.
int choose_splits(const Plan& pl, int device, int64_t rows, int vocab) {
  int splits = 1;
  while (splits < MAX_S && rows * splits * 2 <= int64_t(pl.sms) * 2 &&
         vocab / (splits * 2) >= ROUND &&
         rows <= active_clusters(device, vocab, splits * 2))
    splits *= 2;
  return splits;
}

// Co-resident clusters of the given size, cached per (device, vocab, size).
int active_clusters(int device, int vocab, int splits) {
  static std::mutex mutex;
  static std::map<std::tuple<int, int, int>, int> cache;
  const std::lock_guard<std::mutex> lock(mutex);
  auto [it, fresh] = cache.try_emplace({device, vocab, splits}, 0);
  if (!fresh) return it->second;
  int chunk = round_up((vocab + splits - 1) / splits, 256);
  cudaLaunchConfig_t config = {};
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = unsigned(splits);
  attr.val.clusterDim.y = attr.val.clusterDim.z = 1;
  config.gridDim = dim3(unsigned(splits));
  config.blockDim = dim3(THREADS);
  config.dynamicSmemBytes = make_layout(vocab, splits, chunk).bytes;
  config.attrs = &attr;
  config.numAttrs = 1;
  if (cudaOccupancyMaxActiveClusters(&it->second, filter_kernel<true>,
                                     &config) != cudaSuccess) {
    cudaGetLastError();
    it->second = 0;
  }
  return it->second;
}

}  // namespace topk_topp_filter
}  // namespace vllm

void top_k_top_p_filter(torch::stable::Tensor& logits,
                        const std::optional<torch::stable::Tensor>& k,
                        const std::optional<torch::stable::Tensor>& p) {
  namespace tf = vllm::topk_topp_filter;
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(logits.is_cuda() && logits.dim() == 2 &&
                      logits.scalar_type() == ScalarType::Float &&
                      logits.stride(1) == 1,
                  "top_k_top_p_filter: logits must be a 2D float32 CUDA tensor "
                  "with contiguous rows");
  const int64_t rows = logits.size(0), vocab = logits.size(1);
  STD_TORCH_CHECK(vocab < tf::MAX_VOCAB,
                  "top_k_top_p_filter: vocab must be below 2^20, got ", vocab);
  if (k)
    STD_TORCH_CHECK(
        k->is_contiguous() && k->scalar_type() == ScalarType::Int &&
            k->numel() == rows,
        "top_k_top_p_filter: k must be a contiguous int32 [batch] tensor");
  if (p)
    STD_TORCH_CHECK(
        p->is_contiguous() && p->scalar_type() == ScalarType::Float &&
            p->numel() == rows,
        "top_k_top_p_filter: p must be a contiguous float32 [batch] tensor");
  if (rows == 0 || (!k && !p)) return;

  const int device = logits.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device);
  const tf::Plan& pl = tf::plan(device);
  const int splits = tf::choose_splits(pl, device, rows, int(vocab));
  tf::Args a;
  a.logits = logits.mutable_data_ptr<float>();
  a.stride = logits.stride(0);
  a.ks = k ? k->const_data_ptr<int>() : nullptr;
  a.ps = p ? p->const_data_ptr<float>() : nullptr;
  a.vocab = int(vocab);
  a.chunk = tf::round_up((int(vocab) + splits - 1) / splits, 256);
  a.ns = (int(vocab) + 255) / 256;
  a.vec = vocab % 4 == 0 && a.stride % 4 == 0 &&
          reinterpret_cast<uintptr_t>(a.logits) % 16 == 0;
  a.ly = tf::make_layout(int(vocab), splits, a.chunk);
  STD_TORCH_CHECK(a.ly.bytes <= pl.max_smem, "top_k_top_p_filter: vocab ",
                  vocab, " needs ", a.ly.bytes,
                  " bytes of shared memory, the device has ", pl.max_smem);

  cudaLaunchConfig_t config = {};
  cudaLaunchAttribute attr;
  config.gridDim = dim3(unsigned(rows * splits));
  config.blockDim = dim3(tf::THREADS);
  config.dynamicSmemBytes = a.ly.bytes;
  config.stream = get_current_cuda_stream(device);
  if (splits == 1) {
    tf::check(cudaLaunchKernelEx(&config, tf::filter_kernel<false>, a),
              "launch");
    return;
  }
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = unsigned(splits);
  attr.val.clusterDim.y = attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  tf::check(cudaLaunchKernelEx(&config, tf::filter_kernel<true>, a),
            "cluster launch");
}
