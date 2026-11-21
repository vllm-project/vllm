// Exact top-p cutoffs and kept-token log-sum-exps for bf16 logits in one
// launch (speculative-decoding verification).
//
// A bf16 row takes at most 65,536 distinct values, so its top-p nucleus is
// fixed by a count histogram over the 16-bit order key. Each row is handled by
// one thread-block cluster of C CTAs, C chosen per batch (pick_cluster_size):
// every CTA counts its slice of the row into its own shared-memory histogram
// (u16 counts packed in pairs, two keys converted per instruction). After a
// cluster barrier every CTA turns its non-empty 256-key blocks into per-block
// fixed-point masses and sends them to rank 0 (a single CTA also keeps the
// per-key masses of the eight blocks ending at the max for the descent); after
// a second barrier one warp of rank 0 descends from the max key to the cutoff,
// reading the peers' counts over DSMEM only for the boundary block. Each key's
// mass is computed once, by the CTA that counted it, in a loop specialized per
// block (sign, temperature 1, edge blocks holding the invalid keys). A slice
// over 65,535 tokens (C <= 2) can wrap a u16 count: the counts are checked
// against the slice length and recounted with every wrap recorded if one did.
//
// Temperature: x~ = x / T rounded to nearest (IEEE, tl.div_rn in Triton). The
// kernels that read the logits downstream must scale the same way, so that
// their log p = x~ - lse matches this kernel's masses bit for bit.
//
// Masses use a fixed point: q = floor(exp(x~ - x~max) * 2^40) with an
// exact-argument exp2, so sums are exact integers and the result is
// deterministic. Rows whose cumulative mass lands within the error band of the
// target are redone with 2^62-scaled weights computed in fp32/integer
// arithmetic to ~2^-45 and summed exactly in 128 bits; only rows inside that
// (~1e-13) band fall back to fp64, which is slow on parts like GB300. Both
// fallbacks keep the terms of the integer search's boundary block as they
// compute them, so their descents reuse them instead of recomputing. Keep
// logit >= cutoff (all ties kept); -inf = no masking (T == 0, top_p >= 1, no
// finite logit), +inf for a +inf max.
//
// The kernel also writes lse = log sum exp(x~) over the kept tokens, in fp64.
// The descent already holds their fixed-point mass Z = sum of q over keys >=
// cutoff, and the max key alone contributes 2^40, so lse = x~max + ln(Z *
// 2^-40) with ln(Z * 2^-40) >= 0: the max token's log p = x~max - lse is never
// positive. Rows with top_p >= 1 keep every token and get the full-row value;
// greedy rows (T == 0) get NaN.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>

#include "torch_utils.h"

namespace cg = cooperative_groups;

namespace vllm {
namespace topp_cutoffs {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kKeys = 1 << 16;
constexpr int kWords = kKeys / 2;  // u16 counts packed in pairs: keys 2w, 2w+1
constexpr int kBlocks = 256;       // 256-key blocks: the key's high byte
constexpr int kBatch = 8;          // uint4 loads per thread per batch
constexpr int kMaxWraps = 64;      // u16 count wrap events
// Each wrap takes 65,536 tokens of one key and records <= 3 events.
constexpr int64_t kMaxWrapSlice = (kMaxWraps / 3) * 65536LL;
constexpr int kTop = 8;  // blocks ending at the max whose q one CTA caches
constexpr uint32_t kKeyLo = 0x0080;  // keys below: negative NaNs and -inf
constexpr uint32_t kKeyHi = 0xFF80;  // +inf; keys above: positive NaNs
constexpr float kMassScale = 1099511627776.0f;  // 2^40

template <int C>
struct Smem {
  uint32_t hist[kWords];
  union {
    struct {
      long long mass[C][kBlocks];
      float err[C][kBlocks];
      uint32_t n[C][kBlocks];
    } part;                      // rank 0: block partials of every rank
    double chunk64[kBlocks][8];  // rank 0, fp64 path: 32-key chunk sums
    struct {
      unsigned long long lo[C][kBlocks];
      unsigned long long hi[C][kBlocks];
    } p2;  // rank 0, precise path: 128-bit block partials
  } u;
  unsigned long long band2[C];
  unsigned long long exp2frac[64];  // precise path: kExp2Frac62
  union {
    // C == 1: q of the occupied keys of blocks kb-7..kb for the descent (with
    // more CTAs the descent computes them under its DSMEM count reads).
    long long qtop[kTop][kBlocks];
    // Rank 0, fallback paths: the per-key terms of block b_int (the integer
    // search's boundary block), saved where they are first computed.
    struct {
      unsigned long long w[kBlocks];  // precise weights
      double v[kBlocks];              // fp64 cnt * exp
      uint32_t cnt[kBlocks];          // merged counts
    } kcache;
  } u2;
  int16_t blk_list[kBlocks];  // this CTA's non-empty blocks
  int n_list;
  int b_int;
  double blk_mass64[kBlocks];
  double scan64[2][kBlocks];
  double red64[kWarps];
  uint32_t red_max[kWarps];
  uint32_t red_min[kWarps];
  uint32_t red_key[kWarps];
  // Pushed by every rank to every rank before the first cluster barrier.
  uint32_t peer_max[C];
  uint32_t peer_range[C];  // first | last << 16 block holding any key
  int peer_wraps[C];
  uint32_t wrap_key[kMaxWraps];
  int wrap_delta[kMaxWraps];
  int n_wraps;
  uint32_t n_tok;
  uint32_t n_sum;
  int fp64;
  double above64;
  struct {
    unsigned long long lo[2][8], hi[2][8], x[6];
    int idx[2][8];
  } r128;  // rank 0, precise descent: cross-warp scratch
};

template <int C>
__device__ __forceinline__ void cluster_sync(cg::cluster_group& cluster) {
  if constexpr (C == 1) {
    __syncthreads();
  } else {
    cluster.sync();
  }
}

// Split cluster barrier: every CTA arrives on entry and waits before its first
// write to a peer's shared memory, which must not happen before the peer runs.
template <int C>
__device__ __forceinline__ void cluster_arrive_relaxed() {
  if constexpr (C > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;");
}

template <int C>
__device__ __forceinline__ void cluster_wait() {
  if constexpr (C > 1)
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

template <int C, typename T>
__device__ __forceinline__ T* at_rank(cg::cluster_group& cluster, T* p, int r) {
  if constexpr (C == 1) {
    return p;
  } else {
    return cluster.map_shared_rank(p, r);
  }
}

__device__ __forceinline__ uint32_t bf16_key(uint32_t bits) {
  bits &= 0xFFFFu;
  if (bits == 0x8000u) bits = 0;  // -0.0 keys as +0.0
  return (bits & 0x8000u) ? (bits ^ 0xFFFFu) : (bits ^ 0x8000u);
}

__device__ __forceinline__ float key_value(uint32_t key) {
  const uint32_t bits = key >= 0x8000u ? (key ^ 0x8000u) : (key ^ 0xFFFFu);
  return __uint_as_float(bits << 16);
}

__device__ __forceinline__ bool valid_key(uint32_t key) {
  return key >= kKeyLo && key <= kKeyHi;
}

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float scaled(float v, float temp) {
  return temp == 1.0f ? v : __fdiv_rn(v, temp);  // x / 1 is exact
}

__device__ __forceinline__ void value_mass(float xs, float xs_max, long long& q,
                                           float& err) {
  constexpr float kLog2eHi = 1.44269502162933349609375f;
  constexpr float kLog2eLo = 1.925963033500011e-08f;
  const float d_hi = __fsub_rn(xs, xs_max);
  const float bb = __fsub_rn(d_hi, xs);
  const float d_lo =
      __fadd_rn(__fsub_rn(xs, __fsub_rn(d_hi, bb)), __fsub_rn(-xs_max, bb));
  float a_hi = __fmul_rn(d_hi, kLog2eHi);
  const float a_lo =
      fmaf(d_hi, kLog2eHi, -a_hi) + (d_hi * kLog2eLo + d_lo * kLog2eHi);
  a_hi = fminf(a_hi, 0.0f);
  const float e =
      fminf(ex2_approx(a_hi) * (1.0f + a_lo * 0.6931471805599453f), 1.0f);
  q = static_cast<long long>(e * kMassScale);
  err = e * (4.76837158203125e-07f - a_hi * 5.9604644775390625e-08f);
}

// x bits (high half) of key k0 + i, k0 % 8 == 0: the keys of a 256-key block
// share the sign, so the bits step by -1 (negative) or +1 per key.
__device__ __forceinline__ uint32_t key_bits_base(uint32_t k0) {
  return (k0 < 0x8000u ? (k0 ^ 0xFFFFu) : (k0 ^ 0x8000u)) << 16;
}

__device__ __forceinline__ uint32_t key_bits_step(uint32_t k0) {
  return k0 < 0x8000u ? 0xFFFF0000u : 0x10000u;
}

// One lane's 8 keys k0..k0+7 into the block mass (q * cnt summed exactly),
// its error bound and the valid count, key by key as the generic loop would.
// kUnit: temperature 1 (x / 1 is exact). kEdge: block 0 or 255, which hold
// the invalid keys. kTop: q also goes to rank 0's descent cache.
template <bool kUnit, bool kEdge, bool kTop>
__device__ __forceinline__ void lane_masses(const uint32_t (&cnt)[8],
                                            uint32_t k0, float temp,
                                            float xs_max, long long* qt,
                                            long long& mq, float& er,
                                            uint32_t& n) {
  const uint32_t base = key_bits_base(k0);
  const uint32_t step = key_bits_step(k0);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (!cnt[i]) continue;
    if (kEdge && !valid_key(k0 + i)) {
      if (kTop) qt[i] = 0;
      continue;
    }
    const float x = __uint_as_float(base + i * step);
    long long q;
    float e_err;
    value_mass(kUnit ? x : __fdiv_rn(x, temp), xs_max, q, e_err);
    mq += q * static_cast<long long>(cnt[i]);
    er += e_err * static_cast<float>(cnt[i]);
    n += cnt[i];
    if (kTop) qt[i] = q;
  }
}

template <bool kUnit>
__device__ __forceinline__ void lane_masses_any(
    const uint32_t (&cnt)[8], uint32_t k0, bool edge, bool top, float temp,
    float xs_max, long long* qt, long long& mq, float& er, uint32_t& n) {
  if (edge) {
    if (top) {
      lane_masses<kUnit, true, true>(cnt, k0, temp, xs_max, qt, mq, er, n);
    } else {
      lane_masses<kUnit, true, false>(cnt, k0, temp, xs_max, qt, mq, er, n);
    }
  } else if (top) {
    lane_masses<kUnit, false, true>(cnt, k0, temp, xs_max, qt, mq, er, n);
  } else {
    lane_masses<kUnit, false, false>(cnt, k0, temp, xs_max, qt, mq, er, n);
  }
}

// u16 count wrap of `key` (the word held `old` before this token's add):
// the true counts are the u16 values plus these deltas.
template <int C>
__device__ __noinline__ void record_wrap(Smem<C>& sm, uint32_t key,
                                         uint32_t old) {
  auto push = [&](uint32_t k, int d) {
    const int i = atomicAdd(&sm.n_wraps, 1);
    if (i < kMaxWraps) {
      sm.wrap_key[i] = k;
      sm.wrap_delta[i] = d;
    }
  };
  push(key, 65536);
  if (!(key & 1u)) {
    push(key + 1, -1);  // the carry landed in the odd key's count
    if ((old >> 16) == 0xFFFFu) push(key + 1, 65536);
  }
}

// Keys of both bf16 halves of `w`: x ^ 0x8000 for x >= 0, ~x for x < 0, so
// keys order like the values. -0.0 lands on 0x7FFF (no other value does) and
// is merged into +0.0's 0x8000 after the histogram. NaN and -inf get keys
// outside [kKeyLo, kKeyHi] and are counted but never read.
__device__ __forceinline__ uint32_t pair_keys(uint32_t w) {
  uint32_t s;
  asm("prmt.b32 %0, %1, 0, 0xBB99;" : "=r"(s) : "r"(w));  // per-half sign
  return w ^ (s | 0x80008000u);
}

__device__ __forceinline__ uint32_t max_u16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("max.u16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ uint32_t min_u16x2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("min.u16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ uint32_t* word_at(uint32_t* hist, uint32_t off) {
  return reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(hist) + off);
}

// Two tokens; hi2/lo2 track the packed max/min key.
template <int C>
__device__ __forceinline__ void add_pair(Smem<C>& sm, uint32_t w, uint32_t& hi2,
                                         uint32_t& lo2) {
  const uint32_t k2 = pair_keys(w);
  hi2 = max_u16x2(hi2, k2);
  lo2 = min_u16x2(lo2, k2);
  // Word key >> 1 holds keys 2w (low half) and 2w + 1 (high half).
  atomicAdd(word_at(sm.hist, (k2 << 1) & 0x1FFFCu), (k2 & 1u) * 0xFFFFu + 1u);
  atomicAdd(word_at(sm.hist, (k2 >> 15) & 0x1FFFCu), max(k2 & 0x10000u, 1u));
}

template <int C>
__device__ __forceinline__ void add_one(Smem<C>& sm, uint32_t bits,
                                        uint32_t& hi2, uint32_t& lo2) {
  const uint32_t k = pair_keys(bits & 0xFFFFu) & 0xFFFFu;
  hi2 = max_u16x2(hi2, k * 0x10001u);
  lo2 = min_u16x2(lo2, k * 0x10001u);
  atomicAdd(word_at(sm.hist, (k << 1) & 0x1FFFCu), (k & 1u) * 0xFFFFu + 1u);
}

// Recount path for slices whose u16 counts wrapped: canonical keys (-0.0 as
// +0.0) and every wrap recorded.
template <int C>
__device__ __forceinline__ void add_tracked(Smem<C>& sm, uint32_t bits) {
  const uint32_t key = bf16_key(bits);
  const uint32_t old =
      atomicAdd(&sm.hist[key >> 1], (key & 1u) ? 0x10000u : 1u);
  if (((old >> ((key & 1u) * 16)) & 0xFFFFu) == 0xFFFFu)
    record_wrap(sm, key, old);
}

template <int C, bool kTrack>
__device__ __forceinline__ void add_word(Smem<C>& sm, uint32_t w, uint32_t& hi2,
                                         uint32_t& lo2) {
  if constexpr (kTrack) {
    add_tracked(sm, w);
    add_tracked(sm, w >> 16);
  } else {
    add_pair(sm, w, hi2, lo2);
  }
}

__device__ __forceinline__ void load_batch(uint4 (&v)[kBatch], const uint4* p4,
                                           int b0, int nvec, int tid) {
#pragma unroll
  for (int j = 0; j < kBatch; ++j) {
    const int i = b0 + tid + j * kThreads;
    v[j] = i < nvec ? __ldg(p4 + i) : make_uint4(0, 0, 0, 0);
  }
}

// Counts this CTA's slice into its histogram; `cur` holds the first batch.
template <int C, bool kTrack>
__device__ __forceinline__ void histogram(Smem<C>& sm, uint4 (&cur)[kBatch],
                                          const uint4* p4, int nvec,
                                          const uint16_t* p16, int vec_end,
                                          int end, int tid, uint32_t& hi2,
                                          uint32_t& lo2) {
  for (int b0 = 0; b0 < nvec; b0 += kThreads * kBatch) {
    if (b0 > 0) load_batch(cur, p4, b0, nvec, tid);
#pragma unroll
    for (int j = 0; j < kBatch; ++j) {
      if (b0 + tid + j * kThreads < nvec) {
        add_word<C, kTrack>(sm, cur[j].x, hi2, lo2);
        add_word<C, kTrack>(sm, cur[j].y, hi2, lo2);
        add_word<C, kTrack>(sm, cur[j].z, hi2, lo2);
        add_word<C, kTrack>(sm, cur[j].w, hi2, lo2);
      }
    }
  }
  for (int i = vec_end + tid; i < end; i += kThreads) {
    if constexpr (kTrack) {
      add_tracked(sm, p16[i]);
    } else {
      add_one(sm, p16[i], hi2, lo2);
    }
  }
}

template <int C>
__device__ __forceinline__ void apply_wraps(const Smem<C>* s, int nw,
                                            uint32_t k0, uint32_t (&cnt)[8]) {
  for (int i = 0; i < nw; ++i) {
    const uint32_t k = s->wrap_key[i] - k0;
    if (k < 8u) cnt[k] += s->wrap_delta[i];
  }
}

__device__ __forceinline__ void unpack8(const uint4 w, uint32_t (&cnt)[8]) {
  cnt[0] = w.x & 0xFFFFu;
  cnt[1] = w.x >> 16;
  cnt[2] = w.y & 0xFFFFu;
  cnt[3] = w.y >> 16;
  cnt[4] = w.z & 0xFFFFu;
  cnt[5] = w.z >> 16;
  cnt[6] = w.w & 0xFFFFu;
  cnt[7] = w.w >> 16;
}

// Counts of keys k0..k0+7 (k0 % 8 == 0) summed over the cluster.
template <int C, bool kWrap>
__device__ __forceinline__ void merged_counts8(cg::cluster_group& cluster,
                                               Smem<C>& sm, uint32_t k0,
                                               uint32_t (&cnt)[8]) {
  uint4 w[C];
#pragma unroll
  for (int r = 0; r < C; ++r)
    w[r] = *reinterpret_cast<const uint4*>(at_rank<C>(cluster, sm.hist, r) +
                                           (k0 >> 1));
#pragma unroll
  for (int j = 0; j < 8; ++j) cnt[j] = 0;
#pragma unroll
  for (int r = 0; r < C; ++r) {
    uint32_t c[8];
    unpack8(w[r], c);
    if constexpr (kWrap) {
      const int nw = sm.peer_wraps[r];
      if (nw) apply_wraps<C>(at_rank<C>(cluster, &sm, r), nw, k0, c);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) cnt[j] += c[j];
  }
}

template <int C, bool kWrap>
__device__ __forceinline__ uint32_t merged_count(cg::cluster_group& cluster,
                                                 Smem<C>& sm, uint32_t key) {
  uint32_t words[C];
#pragma unroll
  for (int r = 0; r < C; ++r)
    words[r] = at_rank<C>(cluster, sm.hist, r)[key >> 1];
  const int shift = (key & 1u) * 16;
  uint32_t cnt = 0;
#pragma unroll
  for (int r = 0; r < C; ++r) {
    cnt += (words[r] >> shift) & 0xFFFFu;
    if constexpr (kWrap) {
      const int nw = sm.peer_wraps[r];
      if (nw) {
        const Smem<C>* s = at_rank<C>(cluster, &sm, r);
        for (int i = 0; i < nw; ++i)
          if (s->wrap_key[i] == key) cnt += s->wrap_delta[i];
      }
    }
  }
  return cnt;
}

template <typename T>
__device__ __forceinline__ T warp_sum(T v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

template <typename T>
__device__ __forceinline__ void warp_suffix_256(const T (&v)[8], T (&out)[8],
                                                int lane) {
  T acc = 0;
#pragma unroll
  for (int j = 7; j >= 0; --j) {
    acc += v[j];
    out[j] = acc;
  }
  T incl = acc;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const T t = __shfl_down_sync(0xffffffffu, incl, off);
    if (lane + off < 32) incl += t;
  }
  const T above = incl - acc;
#pragma unroll
  for (int j = 0; j < 8; ++j) out[j] += above;
}

template <typename T>
__device__ __forceinline__ T suffix_sum_256(T (*buf)[kBlocks], T v, int tid) {
  int cur = 0;
  if (tid < kBlocks) buf[0][tid] = v;
  __syncthreads();
  for (int off = 1; off < kBlocks; off <<= 1) {
    if (tid < kBlocks) {
      T x = buf[cur][tid];
      if (tid + off < kBlocks) x += buf[cur][tid + off];
      buf[cur ^ 1][tid] = x;
    }
    cur ^= 1;
    __syncthreads();
  }
  return tid < kBlocks ? buf[cur][tid] : T(0);
}

template <int C>
__device__ __forceinline__ int block_max_int(Smem<C>& sm, int v, int tid) {
  v = __reduce_max_sync(0xffffffffu, v);
  if ((tid & 31) == 0) sm.red_key[tid >> 5] = static_cast<uint32_t>(v + 1);
  __syncthreads();
  int best = -1;
  for (int w = 0; w < kWarps; ++w)
    best = max(best, static_cast<int>(sm.red_key[w]) - 1);
  __syncthreads();
  return best;
}

__device__ __forceinline__ double key_exp64(uint32_t key, float temp,
                                            double xs_max) {
  return exp(static_cast<double>(scaled(key_value(key), temp)) - xs_max);
}

// --- Precise fixed-point path ---------------------------------------------
// Rows the integer search cannot settle are first redone with
// W = 2^62 exp(x~ - x~max), computed in fp32/integer arithmetic to ~2^-45 and
// summed exactly in 128 bits, with a rigorous bound on the accumulated error.
// Only rows still inside that bound (~1e-13 of the mass) need fp64.
__device__ const unsigned long long kExp2Frac62[64] = {
    // 2^62 * 2^(j/64)
    0x4000000000000000ull, 0x40B268F9DE0183BAull, 0x4166C34C5615D0ECull,
    0x421D1461D66F2023ull, 0x42D561B3E6243D8Aull, 0x438FB0CB4F468808ull,
    0x444C0740496D4294ull, 0x450A6ABAA4B77ECDull, 0x45CAE0F1F545EB73ull,
    0x468D6FADBF2DD4F3ull, 0x47521CC5A2E6A9E0ull, 0x4818EE218A3358EEull,
    0x48E1E9B9D588E19Bull, 0x49AD159789F37496ull, 0x4A7A77D47F7B84B1ull,
    0x4B4A169B900C2D00ull, 0x4C1BF828C6DC54B8ull, 0x4CF022C9905BFD32ull,
    0x4DC69CDCEAA72A9Cull, 0x4E9F6CD3967FDBA8ull, 0x4F7A993048D088D7ull,
    0x50582887DCB8A7E1ull, 0x513821818624B40Cull, 0x521A8AD704F3404Full,
    0x52FF6B54D8A89C75ull, 0x53E6C9DA74B29AB5ull, 0x54D0AD5A753E077Cull,
    0x55BD1CDAD49F699Cull, 0x56AC1F752150A563ull, 0x579DBC56B48521BAull,
    0x5891FAC0E95612C8ull, 0x5988E20954889245ull, 0x5A827999FCEF3242ull,
    0x5B7EC8F19468BBC9ull, 0x5C7DD7A3B17DCF75ull, 0x5D7FAD59099F22FEull,
    0x5E8451CFAC061B5Full, 0x5F8BCCDB3D398841ull, 0x6096266533384A2Bull,
    0x61A3666D124BB204ull, 0x62B39508AA836D6Full, 0x63C6BA6455DCD8AEull,
    0x64DCDEC3371793D1ull, 0x65F60A7F79393E2Eull, 0x6712460A8FC24072ull,
    0x683199ED779592CAull, 0x69540EC8F895722Dull, 0x6A79AD55E7F6FD10ull,
    0x6BA27E656B4EB57Aull, 0x6CCE8AE13C57EBDBull, 0x6DFDDBCBED791BABull,
    0x6F307A412F074892ull, 0x70666F76154A7089ull, 0x719FC4B95F452D29ull,
    0x72DC8373BE41A454ull, 0x741CB5281E25EE34ull, 0x75606373EE921C97ull,
    0x76A7980F6CCA15C2ull, 0x77F25CCDEE6D7AE6ull, 0x7940BB9E2CFFD89Dull,
    0x7A92BE8A92436616ull, 0x7BE86FB985689DDCull, 0x7D41D96DB915019Dull,
    0x7E9F06067A4360BAull,
};

struct U128 {
  unsigned long long lo, hi;
};

__device__ __forceinline__ U128 add128(U128 a, U128 b) {
  U128 r;
  r.lo = a.lo + b.lo;
  r.hi = a.hi + b.hi + (r.lo < a.lo);
  return r;
}

__device__ __forceinline__ U128 sub128(U128 a, U128 b) {
  U128 r;
  r.lo = a.lo - b.lo;
  r.hi = a.hi - b.hi - (a.lo < b.lo);
  return r;
}

__device__ __forceinline__ bool ge128(U128 a, U128 b) {
  return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
}

__device__ __forceinline__ U128 mul64x32(unsigned long long a, uint32_t b) {
  return U128{a * b, __umul64hi(a, b)};
}

__device__ __forceinline__ U128 shfl_down128(U128 v, int off) {
  return U128{__shfl_down_sync(0xffffffffu, v.lo, off),
              __shfl_down_sync(0xffffffffu, v.hi, off)};
}

__device__ __forceinline__ U128 warp_sum128(U128 v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = add128(v, shfl_down128(v, off));
  return v;
}

// W = round(2^62 exp(xs - xs_max)) for xs <= xs_max, and a bound on |W -
// exact|.
__device__ __forceinline__ void precise_weight(const unsigned long long* tab,
                                               float xs, float xs_max,
                                               unsigned long long& w,
                                               unsigned long long& err) {
  constexpr float kLog2eHi = 1.44269502162933349609375f;
  constexpr float kLog2eLo = 1.925963033500011e-08f;
  constexpr float kLn2Hi = 0.693147182464599609375f;
  constexpr float kLn2Lo = -1.9046542121259336e-09f;
  // d = xs - xs_max exactly (two-sum).
  const float d_hi = __fsub_rn(xs, xs_max);
  const float bb = __fsub_rn(d_hi, xs);
  const float d_lo =
      __fadd_rn(__fsub_rn(xs, __fsub_rn(d_hi, bb)), __fsub_rn(-xs_max, bb));
  if (!(d_hi >= -44.0f)) {  // 2^62 e^-44 < 0.4
    w = 0;
    err = 1;
    return;
  }
  // a = d log2(e) = a_hi + a_lo, |error| <~ 2^-45.5 |a|.
  const float a_hi = __fmul_rn(d_hi, kLog2eHi);
  const float a_lo =
      fmaf(d_hi, kLog2eHi, -a_hi) + (d_hi * kLog2eLo + d_lo * kLog2eHi);
  // a = k / 64 + g with |g| <= 1/128; a_hi - k / 64 is exact, and g is
  // renormalized (two-sum) since a_lo can exceed what is left of a_hi.
  const int k =
      min(__float2int_rn(a_hi * 64.0f), 0);  // keys above the max: unused
  const float g0 = __fsub_rn(a_hi, static_cast<float>(k) * 0.015625f);
  const float g_hi = __fadd_rn(g0, a_lo);
  const float gb = __fsub_rn(g_hi, g0);
  const float g_lo =
      __fadd_rn(__fsub_rn(g0, __fsub_rn(g_hi, gb)), __fsub_rn(a_lo, gb));
  // e^y - 1 with y = g ln2 = y_hi + y_lo, |y| <= 0.0055: y + y^2/2 in
  // float-float, the rest in fp32 (abs error <~ 2^-48).
  const float y0 = __fmul_rn(g_hi, kLn2Hi);
  const float y1 = fmaf(g_hi, kLn2Hi, -y0) + (g_hi * kLn2Lo + g_lo * kLn2Hi);
  const float y_hi = __fadd_rn(y0, y1);
  const float y_lo = __fsub_rn(y1, __fsub_rn(y_hi, y0));  // Fast2Sum
  const float sq_hi = __fmul_rn(y_hi, y_hi);
  const float sq_lo = fmaf(y_hi, y_hi, -sq_hi);
  const float half_sq = 0.5f * sq_hi;
  const float p_hi = __fadd_rn(y_hi, half_sq);
  const float p_err = __fsub_rn(half_sq, __fsub_rn(p_hi, y_hi));  // Fast2Sum
  const float tail =
      y_lo + 0.5f * sq_lo + y_hi * y_lo +
      sq_hi * y_hi *
          (0.16666667f + y_hi * (0.041666668f + y_hi * 0.008333334f));
  const float p_lo = __fadd_rn(p_err, tail);
  // 2^62 2^((k & 63) / 64) (1 + p), then the 2^(k >> 6) scale, rounded.
  constexpr float k2p64 = 18446744073709551616.0f;
  const long long pf =
      __float2ll_rn(p_hi * k2p64) + __float2ll_rn(p_lo * k2p64);
  const unsigned long long t = tab[k & 63];
  const unsigned long long v = t + static_cast<unsigned long long>(__mul64hi(
                                       static_cast<long long>(t), pf));
  const int sh = -(k >> 6);  // 0..64
  w = sh >= 64 ? 0ull : (sh == 0 ? v : (v + (1ull << (sh - 1))) >> sh);
  err = ((w >> 45) + 1) * static_cast<unsigned long long>(sh + 2) + 2;
}

// Named barrier of the 256 threads (warps 0-7) that run rank 0's precise
// descent.
__device__ __forceinline__ void bar256() {
  asm volatile("bar.sync 1, 256;" ::: "memory");
}

// Inclusive suffix sums of v over threads 0-255; `total` gets the sum of all.
// Each call site passes its own `slot` so no barrier guards the reuse.
template <int C>
__device__ __forceinline__ U128 suffix256_128(Smem<C>& sm, U128 v, int tid,
                                              int slot, U128& total) {
  const int lane = tid & 31;
  const int w = tid >> 5;
  U128 incl = v;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const U128 t = shfl_down128(incl, off);
    if (lane + off < 32) incl = add128(incl, t);
  }
  if (lane == 0) {
    sm.r128.lo[slot][w] = incl.lo;
    sm.r128.hi[slot][w] = incl.hi;
  }
  bar256();
  U128 above{0, 0}, tot{0, 0};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const U128 x{sm.r128.lo[slot][i], sm.r128.hi[slot][i]};
    tot = add128(tot, x);
    if (i > w) above = add128(above, x);
  }
  total = tot;
  return add128(incl, above);
}

template <int C>
__device__ __forceinline__ int max256(Smem<C>& sm, int v, int tid, int slot) {
  v = __reduce_max_sync(0xffffffffu, v);
  if ((tid & 31) == 0) sm.r128.idx[slot][tid >> 5] = v;
  bar256();
  int best = -1;
#pragma unroll
  for (int i = 0; i < 8; ++i) best = max(best, sm.r128.idx[slot][i]);
  return best;
}

// ceil(top_p * total) exactly, for 0 < top_p < 1: top_p = mant * 2^(ex - 24).
__device__ __forceinline__ U128 ceil_scaled(U128 total, float top_p) {
  int ex;
  const float fr = frexpf(top_p, &ex);
  const uint32_t mant = static_cast<uint32_t>(fr * 16777216.0f);
  U128 prod = mul64x32(total.lo, mant);
  prod.hi += total.hi * mant;
  const int sh = 24 - ex;  // >= 24
  if (sh >= 128) return U128{(prod.lo | prod.hi) ? 1ull : 0ull, 0};
  const unsigned long long lo_mask = sh >= 64 ? ~0ull : ((1ull << sh) - 1);
  const bool rem =
      sh >= 64 ? (prod.lo != 0 || (prod.hi & ((1ull << (sh - 64)) - 1)) != 0)
               : (prod.lo & lo_mask) != 0;
  const U128 q =
      sh >= 64 ? U128{prod.hi >> (sh - 64), 0}
               : U128{(prod.lo >> sh) | (prod.hi << (64 - sh)), prod.hi >> sh};
  return add128(q, U128{rem ? 1ull : 0ull, 0});
}

// Returns true when the row was settled (uniform over the cluster).
template <int C, bool kWrap>
__device__ __noinline__ bool precise_path(cg::cluster_group& cluster,
                                          Smem<C>& sm, int rank, float m,
                                          float temp, float top_p, int tid,
                                          float* thresh, double* lse) {
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const float xs_max = scaled(m, temp);
  const int nw = kWrap ? sm.peer_wraps[rank] : 0;
  // 1. Every rank: 128-bit partials of its non-empty blocks (rank 0 zeroed
  //    the partials when it flagged the row). The weights of block b_int go
  //    to rank 0's cache for the descent.
  auto* p2 = &at_rank<C>(cluster, &sm, 0)->u.p2;
  unsigned long long* wcache =
      at_rank<C>(cluster, &sm.u2.kcache.w[0], 0) + lane * 8;
  if (tid == 0) sm.band2[rank] = 0;
  if (tid < 64) sm.exp2frac[tid] = kExp2Frac62[tid];
  __syncthreads();
  unsigned long long band = 0;
  const int n_list = sm.n_list;
  const int b_int = sm.b_int;
  for (int j = warp; j < n_list; j += kWarps) {
    const int b = sm.blk_list[j];
    const uint32_t k0 = static_cast<uint32_t>(b) * kBlocks + lane * 8;
    uint32_t cnt[8];
    unpack8(*reinterpret_cast<const uint4*>(sm.hist + (k0 >> 1)), cnt);
    if (kWrap && nw) apply_wraps<C>(&sm, nw, k0, cnt);
    const bool keep = b == b_int;
    const bool edge = b == 0 || b == kBlocks - 1;
    const uint32_t step = key_bits_step(k0);
    uint32_t bits = key_bits_base(k0);
    U128 acc{0, 0};
#pragma unroll 1  // one key at a time: fewer registers spilled
    for (int t = 0; t < 8; ++t, bits += step) {
      const uint32_t c = cnt[t];
      if (c && (!edge || valid_key(k0 + t))) {
        unsigned long long w, e;
        precise_weight(sm.exp2frac, scaled(__uint_as_float(bits), temp), xs_max,
                       w, e);
        acc = add128(acc, mul64x32(w, c));
        band += e * c;
        if (keep) wcache[t] = w;
      }
    }
    acc = warp_sum128(acc);
    if (lane == 0) {
      p2->lo[rank][b] = acc.lo;
      p2->hi[rank][b] = acc.hi;
    }
  }
  band = warp_sum(band);
  if (lane == 0) atomicAdd(&sm.band2[rank], band);
  __syncthreads();
  if (tid == 0 && rank != 0)
    at_rank<C>(cluster, &sm, 0)->band2[rank] = sm.band2[rank];
  cluster_sync<C>(cluster);
  // 2. Rank 0, warps 0-7: the same descent in 128 bits, one block and then
  //    one key per thread.
  if (rank == 0 && tid < kBlocks) {
    U128 bm{0, 0};
#pragma unroll
    for (int r = 0; r < C; ++r)
      bm = add128(bm, U128{sm.u.p2.lo[r][tid], sm.u.p2.hi[r][tid]});
    U128 total;
    const U128 from = suffix256_128(sm, bm, tid, 0, total);
    unsigned long long band_all = 0;
#pragma unroll
    for (int r = 0; r < C; ++r) band_all += sm.band2[r];
    const U128 target = ceil_scaled(total, top_p);
    const int B = max(max256(sm, ge128(from, target) ? tid : -1, tid, 0), 0);
    if (tid == B) {
      const U128 above = sub128(from, bm);
      sm.r128.x[0] = above.lo;
      sm.r128.x[1] = above.hi;
    }
    const uint32_t key = static_cast<uint32_t>(B) * kBlocks + tid;
    const uint32_t cnt = merged_count<C, kWrap>(cluster, sm, key);
    unsigned long long w = 0;
    if (cnt && valid_key(key)) {
      if (B == b_int) {
        w = sm.u2.kcache.w[tid];
      } else {
        unsigned long long e;
        precise_weight(sm.exp2frac, scaled(key_value(key), temp), xs_max, w, e);
      }
    }
    const U128 kq = cnt ? mul64x32(w, cnt) : U128{0, 0};
    U128 ksum;
    const U128 kfrom = suffix256_128(sm, kq, tid, 1, ksum);  // barrier: above
    const U128 f = add128(U128{sm.r128.x[0], sm.r128.x[1]}, kfrom);
    const int H = max256(sm, (cnt && ge128(f, target)) ? tid : -1, tid, 1);
    if (tid == H) {
      const U128 gt = sub128(f, kq);
      sm.r128.x[2] = f.lo;
      sm.r128.x[3] = f.hi;
      sm.r128.x[4] = gt.lo;
      sm.r128.x[5] = gt.hi;
    }
    bar256();
    const U128 mass_k{sm.r128.x[2], sm.r128.x[3]};
    const U128 mass_gt{sm.r128.x[4], sm.r128.x[5]};
    // Every partial sum and the total are within band_all of exact, so the
    // margin to the target must exceed twice that (plus the ceil).
    const U128 band128{2 * band_all + 2, 0};
    const bool sure = H >= 0 && !ge128(band128, sub128(mass_k, target)) &&
                      !ge128(band128, sub128(target, mass_gt));
    if (tid == 0 && sure) {
      *thresh = key_value(static_cast<uint32_t>(B) * kBlocks + H);
      const double z = static_cast<double>(mass_k.hi) * 4.0 +
                       static_cast<double>(mass_k.lo) * 0x1p-62;
      *lse = static_cast<double>(xs_max) + log(z);
    }
    if (tid < C) at_rank<C>(cluster, &sm, tid)->fp64 = sure ? 0 : 2;
  }
  cluster_sync<C>(cluster);
  return sm.fp64 == 0;
}

// Exact fp64 cutoff for rows within the error band, run by every CTA of the
// cluster: the 32-key chunk sums of the occupied blocks are spread over all
// warps, then rank 0 combines them in the same order as a single CTA would.
// The terms of block b_int are kept for the descent.
template <int C, bool kWrap>
__device__ __noinline__ void fp64_path(cg::cluster_group& cluster, Smem<C>& sm,
                                       int rank, float m, float temp,
                                       float top_p, int tid, float* thresh,
                                       double* lse) {
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (rank == 0) {
    double* c = &sm.u.chunk64[0][0];
    for (int i = tid; i < kBlocks * 8; i += kThreads) c[i] = 0.0;
  }
  int ub_lo = kBlocks, ub_hi = -1;
#pragma unroll
  for (int r = 0; r < C; ++r) {
    ub_lo = min(ub_lo, static_cast<int>(sm.peer_range[r] & 0xFFFFu));
    ub_hi = max(ub_hi, static_cast<int>(sm.peer_range[r] >> 16));
  }
  cluster_sync<C>(cluster);
  const double xs_max = static_cast<double>(scaled(m, temp));
  const int items = max(ub_hi - ub_lo + 1, 0) * 8;  // 8 chunks per block
  const int b_int = sm.b_int;
  double* dst = &at_rank<C>(cluster, &sm, 0)->u.chunk64[0][0];
  double* vcache = at_rank<C>(cluster, &sm.u2.kcache.v[0], 0);
  uint32_t* ccache = at_rank<C>(cluster, &sm.u2.kcache.cnt[0], 0);
  for (int it = rank * kWarps + warp; it < items; it += C * kWarps) {
    const int b = ub_lo + (it >> 3);
    const int k = (it & 7) * 32 + lane;
    const uint32_t key = static_cast<uint32_t>(b) * kBlocks + k;
    const uint32_t cnt = merged_count<C, kWrap>(cluster, sm, key);
    const bool act = cnt && valid_key(key);
    const bool keep = b == b_int;
    if (keep) ccache[k] = cnt;
    if (!__any_sync(0xffffffffu, act)) {
      if (keep) vcache[k] = 0.0;
      continue;
    }
    double v = act ? cnt * key_exp64(key, temp, xs_max) : 0.0;
    if (keep) vcache[k] = v;
    v = warp_sum(v);
    if (lane == 0) dst[b * 8 + (it & 7)] = v;
  }
  cluster_sync<C>(cluster);
  if (rank == 0) {
    if (tid < kBlocks) {
      double t = 0.0;
#pragma unroll
      for (int w = 0; w < 8; ++w) t += sm.u.chunk64[tid][w];
      sm.blk_mass64[tid] = t;
    }
    __syncthreads();
    const double from = suffix_sum_256(
        sm.scan64, tid < kBlocks ? sm.blk_mass64[tid] : 0.0, tid);
    if (tid == 0) sm.red64[0] = from;
    __syncthreads();
    const double target = static_cast<double>(top_p) * sm.red64[0];
    const int B =
        block_max_int(sm, (tid < kBlocks && from >= target) ? tid : -1, tid);
    if (tid == 0) sm.above64 = 0.0;
    __syncthreads();
    if (B + 1 < kBlocks && tid == B + 1) sm.above64 = from;
    __syncthreads();
    double kv = 0.0;
    uint32_t cnt = 0;
    if (tid < kBlocks) {
      if (B == b_int) {
        cnt = sm.u2.kcache.cnt[tid];
        kv = sm.u2.kcache.v[tid];
      } else {
        const uint32_t key = static_cast<uint32_t>(B) * kBlocks + tid;
        cnt = merged_count<C, kWrap>(cluster, sm, key);
        if (cnt && valid_key(key)) kv = cnt * key_exp64(key, temp, xs_max);
      }
    }
    const double kfrom = sm.above64 + suffix_sum_256(sm.scan64, kv, tid);
    const int hit = block_max_int(
        sm, (tid < kBlocks && cnt && kfrom >= target) ? tid : -1, tid);
    if (tid == (hit >= 0 ? hit : 0)) {
      const double z64 = hit >= 0 ? kfrom : sm.red64[0];
      *thresh =
          hit >= 0 ? key_value(static_cast<uint32_t>(B) * kBlocks + hit) : m;
      *lse = xs_max + log(z64);
    }
  }
  // Rank 0 read the peers' counts above.
  if constexpr (C > 1) cluster.sync();
}

// Per-block masses of this CTA's own histogram, sent to rank 0 (exact
// integers: merged sums are order-free). The non-empty blocks are listed first
// so the warps share them evenly; with C == 1, q of the keys in blocks
// kb-7..kb also goes to the descent's cache.
template <int C, bool kWrap>
__device__ __forceinline__ void block_masses(cg::cluster_group& cluster,
                                             Smem<C>& sm, int rank, int warp,
                                             int lane, float temp, float xs_max,
                                             bool masses, int nw, int kb) {
  const int b_lo = static_cast<int>(sm.peer_range[rank] & 0xFFFFu);
  const int nblk = static_cast<int>(sm.peer_range[rank] >> 16) - b_lo + 1;
  if (kWrap && nw) {
    // Recount after a wrap: the counts need the recorded deltas.
    for (int j = warp; j < nblk; j += kWarps) {
      const int b = b_lo + j;
      const uint32_t k0 = static_cast<uint32_t>(b) * kBlocks + lane * 8;
      uint32_t cnt[8];
      unpack8(*reinterpret_cast<const uint4*>(sm.hist + (k0 >> 1)), cnt);
      apply_wraps<C>(&sm, nw, k0, cnt);
      const uint32_t all =
          cnt[0] + cnt[1] + cnt[2] + cnt[3] + cnt[4] + cnt[5] + cnt[6] + cnt[7];
      if (!__any_sync(0xffffffffu, all != 0)) continue;
      if (lane == 0)
        sm.blk_list[atomicAdd(&sm.n_list, 1)] = static_cast<int16_t>(b);
    }
  } else {
    // Four blocks per warp step, one atomic per step. The wrap check sums
    // every count, NaN and -inf included.
    uint32_t lane_tot = 0;
    for (int j0 = warp * 4; j0 < nblk; j0 += kWarps * 4) {
      uint32_t m = 0;
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        uint4 w = make_uint4(0, 0, 0, 0);
        if (j0 + u < nblk)
          w = *reinterpret_cast<const uint4*>(
              sm.hist + ((b_lo + j0 + u) * kBlocks + lane * 8) / 2);
        if constexpr (kWrap)
          lane_tot += (w.x & 0xFFFFu) + (w.x >> 16) + (w.y & 0xFFFFu) +
                      (w.y >> 16) + (w.z & 0xFFFFu) + (w.z >> 16) +
                      (w.w & 0xFFFFu) + (w.w >> 16);
        m |= __any_sync(0xffffffffu, (w.x | w.y | w.z | w.w) != 0) ? 1u << u
                                                                   : 0u;
      }
      if (m) {
        int at = 0;
        if (lane == 0) at = atomicAdd(&sm.n_list, __popc(m));
        at = __shfl_sync(0xffffffffu, at, 0);
        if (lane < 4 && ((m >> lane) & 1u))
          sm.blk_list[at + __popc(m & ((1u << lane) - 1u))] =
              static_cast<int16_t>(b_lo + j0 + lane);
      }
    }
    if constexpr (kWrap) {
      lane_tot = warp_sum(lane_tot);
      if (lane == 0) atomicAdd(&sm.n_sum, lane_tot);
    }
  }
  __syncthreads();
  if (!masses) return;
  long long* dst_mass =
      &at_rank<C>(cluster, &sm.u.part.mass[0][0], 0)[rank * kBlocks];
  float* dst_err =
      &at_rank<C>(cluster, &sm.u.part.err[0][0], 0)[rank * kBlocks];
  uint32_t* dst_n = &at_rank<C>(cluster, &sm.u.part.n[0][0], 0)[rank * kBlocks];
  long long* qtop = at_rank<C>(cluster, &sm.u2.qtop[0][0], 0) + lane * 8;
  const bool unit = temp == 1.0f;
  const int n_list = sm.n_list;
  for (int j = warp; j < n_list; j += kWarps) {
    const int b = sm.blk_list[j];
    const uint32_t k0 = static_cast<uint32_t>(b) * kBlocks + lane * 8;
    uint32_t cnt[8];
    unpack8(*reinterpret_cast<const uint4*>(sm.hist + (k0 >> 1)), cnt);
    if (kWrap && nw) apply_wraps<C>(&sm, nw, k0, cnt);
    const int t = b - (kb - (kTop - 1));
    const bool top = C == 1 && t >= 0 && t < kTop;
    const bool edge = b == 0 || b == kBlocks - 1;
    long long* qt = qtop + t * kBlocks;
    long long mq = 0;
    float er = 0.0f;
    uint32_t n = 0;
    if (unit) {
      lane_masses_any<true>(cnt, k0, edge, top, temp, xs_max, qt, mq, er, n);
    } else {
      lane_masses_any<false>(cnt, k0, edge, top, temp, xs_max, qt, mq, er, n);
    }
    mq = warp_sum(mq);
    er = warp_sum(er);
    n = warp_sum(n);
    if (lane == 0) {
      dst_mass[b] = mq;
      dst_err[b] = er;
      dst_n[b] = n;
    }
  }
}

template <int C, bool kWrap>
__global__ void __launch_bounds__(kThreads, 1)
    topp_hist_cluster_kernel(const __nv_bfloat16* __restrict__ logits,
                             int64_t stride, int vocab,
                             const int64_t* __restrict__ expanded_idx_mapping,
                             const float* __restrict__ temperature,
                             const float* __restrict__ top_p_ptr,
                             float* __restrict__ thresh,
                             double* __restrict__ lse, int force_fallback) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem<C>& sm = *reinterpret_cast<Smem<C>*>(smem_raw);
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = C == 1 ? 0 : static_cast<int>(cluster.block_rank());
  const int64_t row = blockIdx.x / C;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  cluster_arrive_relaxed<C>();

  // The row's first loads go out before the request parameters.
  const __nv_bfloat16* row_ptr = logits + row * stride;
  const int slice = ((vocab + C - 1) / C + 7) & ~7;
  const int start = min(rank * slice, vocab);
  const int end = min(start + slice, vocab);
  const bool aligned =
      (reinterpret_cast<uintptr_t>(row_ptr + start) & 15u) == 0;
  const int nvec = aligned ? (end - start) / 8 : 0;
  const int vec_end = start + nvec * 8;
  const uint4* p4 = reinterpret_cast<const uint4*>(row_ptr + start);
  uint4 cur[kBatch];
  load_batch(cur, p4, 0, nvec, tid);

  const int req = static_cast<int>(expanded_idx_mapping[row]);
  const float temp = temperature[req];
  const float top_p = top_p_ptr[req];
  if (temp == 0.0f) {
    // Uniform across the cluster: no CTA touches DSMEM.
    if (rank == 0 && tid == 0) {
      thresh[row] = -INFINITY;
      lse[row] = NAN;
    }
    return;
  }

  {
    uint4* h4 = reinterpret_cast<uint4*>(sm.hist);
    for (int i = tid; i < kWords / 4; i += kThreads)
      h4[i] = make_uint4(0, 0, 0, 0);
    if (rank == 0) {
      for (int i = tid; i < C * kBlocks; i += kThreads) {
        (&sm.u.part.mass[0][0])[i] = 0;
        (&sm.u.part.err[0][0])[i] = 0.0f;
        (&sm.u.part.n[0][0])[i] = 0;
      }
    }
    if (tid == 0) {
      sm.n_list = 0;
      if constexpr (kWrap) {
        sm.n_wraps = 0;
        sm.n_sum = 0;
      }
    }
  }
  __syncthreads();

  uint32_t hi2 = 0, lo2 = 0xFFFFFFFFu;
  const uint16_t* p16 = reinterpret_cast<const uint16_t*>(row_ptr);
  histogram<C, false>(sm, cur, p4, nvec, p16, vec_end, end, tid, hi2, lo2);
  {
    const uint32_t hi =
        __reduce_max_sync(0xffffffffu, max(hi2 & 0xFFFFu, hi2 >> 16));
    const uint32_t lo =
        __reduce_min_sync(0xffffffffu, min(lo2 & 0xFFFFu, lo2 >> 16));
    if (lane == 0) {
      sm.red_max[warp] = hi;
      sm.red_min[warp] = lo;
    }
  }
  __syncthreads();
  uint32_t raw_hi = __reduce_max_sync(0xffffffffu, sm.red_max[lane]);
  uint32_t raw_lo = __reduce_min_sync(0xffffffffu, sm.red_min[lane]);
  if (tid == 0) {
    // -0.0 counted on 0x7FFF: move it to +0.0's key (the sum stays within the
    // u16 unless the slice is over 65,535 tokens, which the wrap check covers).
    const uint32_t w = sm.hist[0x3FFF];
    if (w >> 16) {
      sm.hist[0x3FFF] = w & 0xFFFFu;
      sm.hist[0x4000] += w >> 16;
    }
    if constexpr (kWrap) sm.n_tok = static_cast<uint32_t>(end - start);
  }
  if (raw_hi == 0x7FFFu) raw_hi = 0x8000u;
  if (raw_lo == 0x7FFFu) raw_lo = 0x8000u;
  uint32_t vmax = raw_hi;  // largest valid key, 0 if none
  if (raw_hi > kKeyHi) {
    // A positive NaN: rescan the slice for the largest non-NaN key (rare).
    uint32_t mk = 0;
    for (int i = start + tid; i < end; i += kThreads) {
      const uint32_t k = bf16_key(p16[i]);
      if (valid_key(k)) mk = max(mk, k);
    }
    mk = __reduce_max_sync(0xffffffffu, mk);
    __syncthreads();
    if (lane == 0) sm.red_max[warp] = mk;
    __syncthreads();
    vmax = __reduce_max_sync(0xffffffffu, sm.red_max[lane]);
  } else if (raw_hi < kKeyLo) {
    vmax = 0;
  }
  cluster_wait<C>();
  if (warp == 0 && lane < C) {
    Smem<C>* dst = at_rank<C>(cluster, &sm, lane);
    dst->peer_max[rank] = vmax;
    dst->peer_range[rank] =
        raw_lo > raw_hi ? kBlocks : (raw_lo >> 8) | ((raw_hi >> 8) << 16);
    dst->peer_wraps[rank] = 0;
  }
  cluster_sync<C>(cluster);

  uint32_t kmax = 0;
#pragma unroll
  for (int r = 0; r < C; ++r) kmax = max(kmax, sm.peer_max[r]);
  const bool finite_max = kmax >= kKeyLo && kmax < kKeyHi;
  const float m = key_value(kmax);
  const float xs_max = scaled(m, temp);

  const bool masses = finite_max && top_p > 0.0f;
  const int kb = static_cast<int>(kmax >> 8);
  if (masses || kWrap)
    block_masses<C, kWrap>(cluster, sm, rank, warp, lane, temp, xs_max, masses,
                           0, kb);
  if constexpr (kWrap) {
    // The u16 counts sum to the token count unless some key took 65,536+
    // tokens of this slice; then recount with the wraps recorded.
    __syncthreads();
    if (sm.n_sum != sm.n_tok) {
      __syncthreads();
      uint4* h4 = reinterpret_cast<uint4*>(sm.hist);
      for (int i = tid; i < kWords / 4; i += kThreads)
        h4[i] = make_uint4(0, 0, 0, 0);
      if (tid == 0) {
        sm.n_wraps = 0;
        sm.n_list = 0;
      }
      __syncthreads();
      uint32_t h2 = 0, l2 = 0;
      load_batch(cur, p4, 0, nvec, tid);
      histogram<C, true>(sm, cur, p4, nvec, p16, vec_end, end, tid, h2, l2);
      __syncthreads();
      const int nw = min(sm.n_wraps, kMaxWraps);
      if (warp == 0 && lane < C)
        at_rank<C>(cluster, &sm, lane)->peer_wraps[rank] = nw;
      block_masses<C, kWrap>(cluster, sm, rank, warp, lane, temp, xs_max,
                             masses, nw, kb);
    }
  }
  cluster_sync<C>(cluster);

  if (rank == 0 && warp == 0) {
    // Warp 0 descends from the top block to the boundary key.
    float result = 0.0f;
    int ambiguous = 0;
    int b_out = -1;
    long long z_int = 0;
    if (!valid_key(kmax)) {
      result = -INFINITY;  // no finite logit
    } else if (kmax == kKeyHi) {
      result = INFINITY;
    } else if (top_p <= 0.0f) {
      result = m;
      z_int = static_cast<long long>(merged_count<C, kWrap>(cluster, sm, kmax))
              << 40;
    } else {
      long long bm[8], from[8];
      float err = 0.0f;
      uint32_t n_valid = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        bm[j] = 0;
#pragma unroll
        for (int r = 0; r < C; ++r) {
          bm[j] += sm.u.part.mass[r][lane * 8 + j];
          err += sm.u.part.err[r][lane * 8 + j];
          n_valid += sm.u.part.n[r][lane * 8 + j];
        }
      }
      warp_suffix_256(bm, from, lane);
      err = warp_sum(err);
      n_valid = warp_sum(n_valid);
      const long long total = __shfl_sync(0xffffffffu, from[0], 0);
      err = __shfl_sync(0xffffffffu, err, 0);
      n_valid = __shfl_sync(0xffffffffu, n_valid, 0);
      if (top_p >= 1.0f) {
        result = -INFINITY;  // every token kept: lse is the full-row value
        z_int = total;
      } else {
        const long long target = static_cast<long long>(
            ceil(static_cast<double>(top_p) * static_cast<double>(total)));
        const long long band = static_cast<long long>(
            2.0 * (static_cast<double>(err) * 1099511627776.0 + n_valid) + 4.0);
        int cand = -1;
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (from[j] >= target) cand = lane * 8 + j;
        const int B = __reduce_max_sync(0xffffffffu, cand);
        b_out = B;
        long long above = 0;  // mass of the blocks above B
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (lane * 8 + j == B + 1) above = from[j];
        above = __shfl_sync(0xffffffffu, above, min((B + 1) >> 3, 31));
        const uint32_t k0 = static_cast<uint32_t>(B) * kBlocks + lane * 8;
        // The masses do not depend on the counts: load or compute them while
        // the peers' counts are in flight.
        long long q8[8];
        const int t = B - (static_cast<int>(kmax >> 8) - (kTop - 1));
        if (C == 1 && t >= 0) {
#pragma unroll
          for (int j = 0; j < 8; ++j) q8[j] = sm.u2.qtop[t][lane * 8 + j];
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            float e_err;
            q8[j] = 0;
            if (valid_key(k0 + j))
              value_mass(scaled(key_value(k0 + j), temp), xs_max, q8[j], e_err);
          }
        }
        uint32_t cnt[8];
        merged_counts8<C, kWrap>(cluster, sm, k0, cnt);
        long long kq[8], kfrom[8];
#pragma unroll
        for (int j = 0; j < 8; ++j)
          kq[j] = cnt[j] ? q8[j] * static_cast<long long>(cnt[j]) : 0;
        warp_suffix_256(kq, kfrom, lane);
        int hit = -1;
        long long mass_k = 0, mass_gt = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (cnt[j] && above + kfrom[j] >= target) {
            hit = lane * 8 + j;
            mass_k = above + kfrom[j];
            mass_gt = mass_k - kq[j];
          }
        }
        const int H = __reduce_max_sync(0xffffffffu, hit);
        const int owner = H >= 0 ? H / 8 : 0;
        mass_k = __shfl_sync(0xffffffffu, mass_k, owner);
        mass_gt = __shfl_sync(0xffffffffu, mass_gt, owner);
        const bool sure =
            H >= 0 && (mass_k - target > band) && (target - mass_gt > band);
        ambiguous = (!sure || force_fallback) ? 1 : 0;
        result = H >= 0 ? key_value(static_cast<uint32_t>(B) * kBlocks + H) : m;
        z_int = H >= 0 ? mass_k : total;
      }
    }
    if (ambiguous) {
      // The precise path's partials reuse this space; peers write after the
      // barrier below.
      __syncwarp();
      uint4* z = reinterpret_cast<uint4*>(&sm.u.p2);
      for (int i = lane; i < static_cast<int>(sizeof(sm.u.p2) / 16); i += 32)
        z[i] = make_uint4(0, 0, 0, 0);
    }
    if (lane < C) {
      Smem<C>* dst = at_rank<C>(cluster, &sm, lane);
      dst->fp64 = ambiguous;
      dst->b_int = b_out;
    }
    if (lane == 0 && !ambiguous) {
      thresh[row] = result;
      if (result == INFINITY || !valid_key(kmax)) {
        lse[row] = result;
      } else {
        // Z = 2^e (1 + f) with f in [0, 1) taken from the exact integer Z -
        // 2^e, so ln(Z * 2^-40) = (e - 40) ln 2 + log1p(f) >= 0.
        const int e = 63 - __clzll(z_int);
        const float f = static_cast<float>(z_int - (1LL << e)) *
                        __int_as_float((127 - e) << 23);
        lse[row] =
            static_cast<double>(xs_max) +
            ((e - 40) * 0.6931471805599453 + static_cast<double>(log1pf(f)));
      }
    }
  }
  // Rank 0's descent read the peers' counts; it also tells them whether the
  // row needs a fallback path.
  cluster_sync<C>(cluster);
  if (sm.fp64) {
    // force_fallback: 1 = through the precise path, 2 = straight to fp64.
    if (force_fallback == 2 ||
        !precise_path<C, kWrap>(cluster, sm, rank, m, temp, top_p, tid,
                                thresh + row, lse + row))
      fp64_path<C, kWrap>(cluster, sm, rank, m, temp, top_p, tid, thresh + row,
                          lse + row);
  }
}

template <int C, bool kWrap>
cudaLaunchConfig_t launch_config(int rows, cudaLaunchAttribute* attrs,
                                 cudaStream_t stream) {
  auto kernel = &topp_hist_cluster_kernel<C, kWrap>;
  if constexpr (C > 8) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed,
                         1);
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sizeof(Smem<C>));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(rows * C);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = sizeof(Smem<C>);
  cfg.stream = stream;
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {C, 1, 1};
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  return cfg;
}

template <int C, bool kWrap>
void launch(const torch::stable::Tensor& logits,
            const torch::stable::Tensor& expanded_idx_mapping,
            const torch::stable::Tensor& temperature,
            const torch::stable::Tensor& top_p, torch::stable::Tensor& thresh,
            torch::stable::Tensor& lse, int force_fallback,
            cudaStream_t stream) {
  cudaLaunchAttribute attrs[1];
  cudaLaunchConfig_t cfg =
      launch_config<C, kWrap>(static_cast<int>(logits.size(0)), attrs, stream);
  const cudaError_t err = cudaLaunchKernelEx(
      &cfg, &topp_hist_cluster_kernel<C, kWrap>,
      reinterpret_cast<const __nv_bfloat16*>(logits.const_data_ptr()),
      logits.stride(0), static_cast<int>(logits.size(1)),
      expanded_idx_mapping.const_data_ptr<int64_t>(),
      temperature.const_data_ptr<float>(), top_p.const_data_ptr<float>(),
      thresh.mutable_data_ptr<float>(), lse.mutable_data_ptr<double>(),
      force_fallback);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "topp_cutoffs_lse launch failed: ", cudaGetErrorString(err));
}

// Clusters of C CTAs that fit on the device at once (one CTA per SM).
template <int C>
int max_active_clusters(int device) {
  constexpr int kMaxDevices = 64;
  static std::atomic<int> cache[kMaxDevices];
  if (device >= 0 && device < kMaxDevices) {
    const int n = cache[device].load(std::memory_order_relaxed);
    if (n > 0) return n;
  }
  cudaLaunchAttribute attrs[1];
  cudaLaunchConfig_t cfg = launch_config<C, false>(C, attrs, nullptr);
  int n = 0;
  cudaOccupancyMaxActiveClusters(&n, &topp_hist_cluster_kernel<C, false>, &cfg);
  n = n > 0 ? n : 1;
  if (device >= 0 && device < kMaxDevices)
    cache[device].store(n, std::memory_order_relaxed);
  return n;
}

// The cluster size with the lowest estimated time: waves of clusters times a
// cluster's latency, ~fixed + 0.064 ns per token of its slice (GB300 stage
// timings; one CTA skips the cluster barriers, more CTAs shorten the
// histogram). C >= 4 slices must fit u16 counts; C <= 2 track wraps.
int pick_cluster_size(int64_t rows, int64_t vocab, int device) {
  constexpr int kSizes[] = {1, 2, 4, 8, 16};
  constexpr int64_t kFixedNs[] = {7700, 10500, 10000, 10400, 12400};
  const int active[] = {
      max_active_clusters<1>(device), max_active_clusters<2>(device),
      max_active_clusters<4>(device), max_active_clusters<8>(device),
      max_active_clusters<16>(device)};
  int best = 0;
  int64_t best_ns = INT64_MAX;
  for (int i = 0; i < 5; ++i) {
    const int c = kSizes[i];
    const int64_t slice = ((vocab + c - 1) / c + 7) & ~7;
    if (c > 2 ? slice > 65535 : slice > kMaxWrapSlice) continue;
    const int64_t waves = (rows + active[i] - 1) / active[i];
    const int64_t ns = waves * (kFixedNs[i] + slice * 64 / 1000);
    if (ns < best_ns) {
      best_ns = ns;
      best = c;
    }
  }
  return best;
}

}  // namespace topp_cutoffs
}  // namespace vllm

void topp_cutoffs_lse(const torch::stable::Tensor& logits,
                      const torch::stable::Tensor& expanded_idx_mapping,
                      const torch::stable::Tensor& temperature,
                      const torch::stable::Tensor& top_p,
                      torch::stable::Tensor& thresh, torch::stable::Tensor& lse,
                      int64_t cluster_size, int64_t force_fallback) {
  using torch::headeronly::ScalarType;
  namespace tc = vllm::topp_cutoffs;
  STD_TORCH_CHECK(logits.is_cuda() && logits.dim() == 2,
                  "logits must be a 2D CUDA tensor");
  STD_TORCH_CHECK(logits.scalar_type() == ScalarType::BFloat16,
                  "logits must be bf16");
  STD_TORCH_CHECK(logits.stride(1) == 1, "logits must have unit vocab stride");
  STD_TORCH_CHECK(expanded_idx_mapping.scalar_type() == ScalarType::Long,
                  "expanded_idx_mapping must be int64");
  STD_TORCH_CHECK(temperature.scalar_type() == ScalarType::Float &&
                      top_p.scalar_type() == ScalarType::Float,
                  "temperature and top_p must be fp32");
  const int64_t rows = logits.size(0);
  STD_TORCH_CHECK(expanded_idx_mapping.numel() == rows,
                  "expanded_idx_mapping must have one entry per row");
  STD_TORCH_CHECK(thresh.scalar_type() == ScalarType::Float &&
                      thresh.numel() == rows && thresh.is_contiguous(),
                  "thresh must be a contiguous fp32 tensor of num_rows");
  STD_TORCH_CHECK(lse.scalar_type() == ScalarType::Double &&
                      lse.numel() == rows && lse.is_contiguous(),
                  "lse must be a contiguous fp64 tensor of num_rows");
  STD_TORCH_CHECK(force_fallback >= 0 && force_fallback <= 2,
                  "force_fallback must be 0, 1 or 2");
  if (rows == 0) return;
  const int device = logits.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device);
  const int64_t vocab = logits.size(1);
  const int64_t c = cluster_size > 0
                        ? cluster_size
                        : tc::pick_cluster_size(rows, vocab, device);
  STD_TORCH_CHECK(c == 1 || c == 2 || c == 4 || c == 8 || c == 16,
                  "cluster_size must be 0 (auto), 1, 2, 4, 8 or 16; vocab ",
                  vocab, " has no valid one");
  const int64_t slice = ((vocab + c - 1) / c + 7) & ~7;
  STD_TORCH_CHECK(c > 2 ? slice <= 65535 : slice <= tc::kMaxWrapSlice, "vocab ",
                  vocab, " is too large for cluster_size ", c);
  const bool wrap = slice > 65535;
  const cudaStream_t stream = get_current_cuda_stream();
  const int force = static_cast<int>(force_fallback);
#define TOPP_LAUNCH(C, W)                                                    \
  tc::launch<C, W>(logits, expanded_idx_mapping, temperature, top_p, thresh, \
                   lse, force, stream)
  switch (c) {
    case 1:
      if (wrap) {
        TOPP_LAUNCH(1, true);
      } else {
        TOPP_LAUNCH(1, false);
      }
      break;
    case 2:
      if (wrap) {
        TOPP_LAUNCH(2, true);
      } else {
        TOPP_LAUNCH(2, false);
      }
      break;
    case 4:
      TOPP_LAUNCH(4, false);
      break;
    case 8:
      TOPP_LAUNCH(8, false);
      break;
    default:
      TOPP_LAUNCH(16, false);
      break;
  }
#undef TOPP_LAUNCH
}
