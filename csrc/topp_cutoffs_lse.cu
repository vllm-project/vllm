// Exact top-p cutoffs and kept-token log-sum-exps for bf16 logits in one
// launch (speculative-decoding verification).
//
// A bf16 row takes at most 65,536 distinct values, so its top-p nucleus is
// fixed by a count histogram over the 16-bit order key. Each row is handled by
// one thread-block cluster of C CTAs: every CTA histograms 1/C of the row into
// its own shared-memory histogram (u16 counts packed in pairs; C >= ceil(V /
// 65535) keeps them from overflowing). After a cluster barrier every CTA turns
// its own histogram into per-block fixed-point masses and sends them to rank 0;
// after a second barrier one warp of rank 0 descends from the max key to the
// cutoff, reading the peers' counts over DSMEM only for the boundary block.
//
// Temperature: x~ = x / T rounded to nearest (IEEE, tl.div_rn in Triton). The
// kernels that read the logits downstream must scale the same way, so that
// their log p = x~ - lse matches this kernel's masses bit for bit.
//
// Masses use a fixed point: q = floor(exp(x~ - x~max) * 2^40) with an
// exact-argument exp2, so sums are exact integers and the result is
// deterministic. Rows whose cumulative mass lands within the error band of the
// target are resolved in fp64. Keep logit >= cutoff (all ties kept); -inf = no
// masking (T == 0, top_p >= 1, no finite logit), +inf for a +inf max.
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

#include <cstdint>

#include "torch_utils.h"

namespace cg = cooperative_groups;

namespace vllm {
namespace topp_cutoffs {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kKeys = 1 << 16;
constexpr int kWords =
    kKeys / 2;                // u16 counts packed in pairs: keys 2w, 2w + 1
constexpr int kBlocks = 256;  // 256-key blocks: the key's high byte
constexpr int kGroups = kThreads / kBlocks;
constexpr int kMaxVecs = 8;  // uint4 loads per thread: slice <= 65535 tokens
constexpr uint32_t kKeyLo = 0x0080;  // keys below: negative NaNs and -inf
constexpr uint32_t kKeyHi = 0xFF80;  // +inf; keys above: positive NaNs
constexpr float kMassScale = 1099511627776.0f;  // 2^40

template <int C>
struct Smem {
  uint32_t hist[kWords];
  // Rank 0 only: per-rank partial sums of every block, written over DSMEM.
  long long part_mass[C][kBlocks];
  float part_err[C][kBlocks];
  uint32_t part_n[C][kBlocks];
  uint8_t occ[kBlocks];
  uint8_t uocc[kBlocks];
  int16_t occ_list[kBlocks];
  double blk_mass64[kBlocks];
  double scan64[2][kBlocks];
  double red64[kWarps];
  uint32_t red_key[kWarps];
  uint32_t occ_cnt[kBlocks / 32];
  double above64;
  uint32_t max_key;
  uint32_t kmax;
  int n_occ;
  int ambiguous;
  float result;
  long long z_int;
  double z64;
};

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

// The scaled logit x~ (see the header).
__device__ __forceinline__ float scaled(float v, float temp) {
  return __fdiv_rn(v, temp);
}

// radix2's _r2_token_mass for x~ - x~max: q and its per-token error bound. The
// difference is carried exactly as d_hi + d_lo (two-sum) and log2(e) as two
// floats, so the exp2 argument a_hi + a_lo is exact to ~2^-47.
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

template <int C>
__device__ __forceinline__ void add_token(Smem<C>& sm, uint32_t bits,
                                          uint32_t& max_key) {
  const uint32_t key = bf16_key(bits);
  if (valid_key(key)) {
    atomicAdd(&sm.hist[key >> 1], (key & 1u) ? 0x10000u : 1u);
    sm.occ[key >> 8] = 1;
    max_key = max(max_key, key);
  }
}

// Counts of the 8 consecutive keys k0..k0+7 (k0 % 8 == 0) in one histogram.
__device__ __forceinline__ void counts8(const uint32_t* h, uint32_t k0,
                                        uint32_t (&cnt)[8]) {
  const uint4 w = *reinterpret_cast<const uint4*>(h + (k0 >> 1));
  cnt[0] = w.x & 0xFFFFu;
  cnt[1] = w.x >> 16;
  cnt[2] = w.y & 0xFFFFu;
  cnt[3] = w.y >> 16;
  cnt[4] = w.z & 0xFFFFu;
  cnt[5] = w.z >> 16;
  cnt[6] = w.w & 0xFFFFu;
  cnt[7] = w.w >> 16;
}

template <int C>
__device__ __forceinline__ void merged_counts8(cg::cluster_group& cluster,
                                               Smem<C>& sm, uint32_t k0,
                                               uint32_t (&cnt)[8]) {
  uint4 w[C];
#pragma unroll
  for (int r = 0; r < C; ++r)
    w[r] = *reinterpret_cast<const uint4*>(cluster.map_shared_rank(sm.hist, r) +
                                           (k0 >> 1));
#pragma unroll
  for (int j = 0; j < 8; ++j) cnt[j] = 0;
#pragma unroll
  for (int r = 0; r < C; ++r) {
    cnt[0] += w[r].x & 0xFFFFu;
    cnt[1] += w[r].x >> 16;
    cnt[2] += w[r].y & 0xFFFFu;
    cnt[3] += w[r].y >> 16;
    cnt[4] += w[r].z & 0xFFFFu;
    cnt[5] += w[r].z >> 16;
    cnt[6] += w[r].w & 0xFFFFu;
    cnt[7] += w[r].w >> 16;
  }
}

template <int C>
__device__ __forceinline__ uint32_t merged_count(cg::cluster_group& cluster,
                                                 Smem<C>& sm, uint32_t key) {
  uint32_t words[C];
#pragma unroll
  for (int r = 0; r < C; ++r)
    words[r] = cluster.map_shared_rank(sm.hist, r)[key >> 1];
  const int shift = (key & 1u) * 16;
  uint32_t cnt = 0;
#pragma unroll
  for (int r = 0; r < C; ++r) cnt += (words[r] >> shift) & 0xFFFFu;
  return cnt;
}

template <typename T>
__device__ __forceinline__ T warp_sum(T v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// Warp-wide suffix sums over 256 values, 8 per lane (lane l holds 8l..8l+7):
// out[j] = sum of values with index >= 8l + j.
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

// Exact fp64 cutoff over every occupied key (rows within the error band).
template <int C>
__device__ float fp64_cutoff(cg::cluster_group& cluster, Smem<C>& sm, float m,
                             float temp, float top_p, int tid) {
  if (tid < kBlocks) {
    uint8_t u = 0;
#pragma unroll
    for (int r = 0; r < C; ++r) u |= cluster.map_shared_rank(sm.occ, r)[tid];
    sm.uocc[tid] = u;
  }
  __syncthreads();
  const int group = tid / kBlocks;
  const int lane_key = tid % kBlocks;
  const double xs_max = static_cast<double>(scaled(m, temp));
  for (int b0 = 0; b0 < kBlocks; b0 += kGroups) {
    const int b = b0 + group;
    double v64 = 0.0;
    if (sm.uocc[b]) {
      const uint32_t key = static_cast<uint32_t>(b) * kBlocks + lane_key;
      const uint32_t cnt = merged_count<C>(cluster, sm, key);
      if (cnt && valid_key(key))
        v64 = cnt *
              exp(static_cast<double>(scaled(key_value(key), temp)) - xs_max);
    }
    v64 = warp_sum(v64);
    if ((tid & 31) == 0) sm.red64[tid >> 5] = v64;
    __syncthreads();
    if (lane_key == 0) {
      double t = 0.0;
      for (int w = 0; w < kBlocks / 32; ++w)
        t += sm.red64[group * (kBlocks / 32) + w];
      sm.blk_mass64[b] = t;
    }
    __syncthreads();
  }
  const double from =
      suffix_sum_256(sm.scan64, tid < kBlocks ? sm.blk_mass64[tid] : 0.0, tid);
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
    const uint32_t key = static_cast<uint32_t>(B) * kBlocks + tid;
    cnt = merged_count<C>(cluster, sm, key);
    if (cnt && valid_key(key))
      kv =
          cnt * exp(static_cast<double>(scaled(key_value(key), temp)) - xs_max);
  }
  const double kfrom = sm.above64 + suffix_sum_256(sm.scan64, kv, tid);
  const int hit = block_max_int(
      sm, (tid < kBlocks && cnt && kfrom >= target) ? tid : -1, tid);
  if (tid == (hit >= 0 ? hit : 0)) sm.z64 = hit >= 0 ? kfrom : sm.red64[0];
  __syncthreads();
  return hit >= 0 ? key_value(static_cast<uint32_t>(B) * kBlocks + hit) : m;
}

template <int C>
__global__ void __launch_bounds__(kThreads, 1)
    topp_hist_cluster_kernel(const __nv_bfloat16* __restrict__ logits,
                             int64_t stride, int vocab,
                             const int32_t* __restrict__ expanded_idx_mapping,
                             const float* __restrict__ temperature,
                             const float* __restrict__ top_p_ptr,
                             float* __restrict__ thresh,
                             double* __restrict__ lse, int force_fp64) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem<C>& sm = *reinterpret_cast<Smem<C>*>(smem_raw);
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int64_t row = blockIdx.x / C;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int req = expanded_idx_mapping[row];
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

  // Issue this thread's loads first; they land while the histogram is zeroed.
  const __nv_bfloat16* row_ptr = logits + row * stride;
  const int slice = ((vocab + C - 1) / C + 7) & ~7;
  const int start = min(rank * slice, vocab);
  const int end = min(start + slice, vocab);
  const bool aligned =
      (reinterpret_cast<uintptr_t>(row_ptr + start) & 15u) == 0;
  const int nvec = aligned ? (end - start) / 8 : 0;
  const int vec_end = start + nvec * 8;
  const uint4* p4 = reinterpret_cast<const uint4*>(row_ptr + start);
  uint4 v[kMaxVecs];
#pragma unroll
  for (int j = 0; j < kMaxVecs; ++j) {
    const int i = tid + j * kThreads;
    v[j] = i < nvec ? __ldg(p4 + i) : make_uint4(0, 0, 0, 0);
  }
  {
    uint4* h4 = reinterpret_cast<uint4*>(sm.hist);
    for (int i = tid; i < kWords / 4; i += kThreads)
      h4[i] = make_uint4(0, 0, 0, 0);
    if (tid < kBlocks) sm.occ[tid] = 0;
    if (rank == 0) {
      for (int i = tid; i < C * kBlocks; i += kThreads) {
        (&sm.part_mass[0][0])[i] = 0;
        (&sm.part_err[0][0])[i] = 0.0f;
        (&sm.part_n[0][0])[i] = 0;
      }
    }
  }
  __syncthreads();

  uint32_t max_key = 0;
#pragma unroll
  for (int j = 0; j < kMaxVecs; ++j) {
    if (tid + j * kThreads < nvec) {
      add_token(sm, v[j].x, max_key);
      add_token(sm, v[j].x >> 16, max_key);
      add_token(sm, v[j].y, max_key);
      add_token(sm, v[j].y >> 16, max_key);
      add_token(sm, v[j].z, max_key);
      add_token(sm, v[j].z >> 16, max_key);
      add_token(sm, v[j].w, max_key);
      add_token(sm, v[j].w >> 16, max_key);
    }
  }
  const uint16_t* p16 = reinterpret_cast<const uint16_t*>(row_ptr);
  for (int i = vec_end + tid; i < end; i += kThreads)
    add_token(sm, p16[i], max_key);
  max_key = __reduce_max_sync(0xffffffffu, max_key);
  if (lane == 0) sm.red_key[warp] = max_key;
  __syncthreads();
  if (tid < 32) {
    uint32_t k = sm.red_key[lane];
    k = __reduce_max_sync(0xffffffffu, k);
    if (lane == 0) sm.max_key = k;
  }
  // This CTA's occupied blocks, compacted.
  if (tid < kBlocks) {
    const bool o = sm.occ[tid] != 0;
    const unsigned ball = __ballot_sync(0xffffffffu, o);
    if (lane == 0) sm.occ_cnt[warp] = __popc(ball);
  }
  cluster.sync();

  if (tid < kBlocks) {
    int before = 0;
    for (int w = 0; w < warp; ++w) before += sm.occ_cnt[w];
    const bool o = sm.occ[tid] != 0;
    const unsigned ball = __ballot_sync(0xffffffffu, o);
    if (o) sm.occ_list[before + __popc(ball & ((1u << lane) - 1))] = tid;
    if (tid == kBlocks - 1) sm.n_occ = before + __popc(ball);
  }
  if (tid == kBlocks) {
    uint32_t k = 0;
#pragma unroll
    for (int r = 0; r < C; ++r)
      k = max(k, *cluster.map_shared_rank(&sm.max_key, r));
    sm.kmax = k;
  }
  __syncthreads();
  const uint32_t kmax = sm.kmax;
  const bool finite_max = kmax >= kKeyLo && kmax < kKeyHi;
  const float m = key_value(kmax);
  const float xs_max = scaled(m, temp);

  // Per-block masses of this CTA's own histogram, one warp per occupied block,
  // sent to rank 0 (exact integers, so the merged sums are order-free).
  if (finite_max && top_p > 0.0f) {
    const int n_occ = sm.n_occ;
    long long* dst_mass =
        &cluster.map_shared_rank(&sm.part_mass[0][0], 0)[rank * kBlocks];
    float* dst_err =
        &cluster.map_shared_rank(&sm.part_err[0][0], 0)[rank * kBlocks];
    uint32_t* dst_n =
        &cluster.map_shared_rank(&sm.part_n[0][0], 0)[rank * kBlocks];
    for (int j = warp; j < n_occ; j += kWarps) {
      const int b = sm.occ_list[j];
      const uint32_t k0 = static_cast<uint32_t>(b) * kBlocks + lane * 8;
      uint32_t cnt[8];
      counts8(sm.hist, k0, cnt);
      long long mq = 0;
      float er = 0.0f;
      uint32_t n = 0;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        if (cnt[t] && valid_key(k0 + t)) {
          long long q;
          float e_err;
          value_mass(scaled(key_value(k0 + t), temp), xs_max, q, e_err);
          mq += q * static_cast<long long>(cnt[t]);
          er += e_err * static_cast<float>(cnt[t]);
          n += cnt[t];
        }
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
  cluster.sync();

  if (rank == 0) {
    // Warp 0 descends from the top block to the boundary key.
    if (warp == 0) {
      float result = 0.0f;
      int ambiguous = 0;
      long long z_int = 0;
      if (!valid_key(kmax)) {
        result = -INFINITY;  // no finite logit
      } else if (kmax == kKeyHi) {
        result = INFINITY;
      } else if (top_p <= 0.0f) {
        result = m;
        z_int = static_cast<long long>(merged_count<C>(cluster, sm, kmax))
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
            bm[j] += sm.part_mass[r][lane * 8 + j];
            err += sm.part_err[r][lane * 8 + j];
            n_valid += sm.part_n[r][lane * 8 + j];
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
              2.0 * (static_cast<double>(err) * 1099511627776.0 + n_valid) +
              4.0);
          int cand = -1;
#pragma unroll
          for (int j = 0; j < 8; ++j)
            if (from[j] >= target) cand = lane * 8 + j;
          const int B = __reduce_max_sync(0xffffffffu, cand);
          long long above = 0;
#pragma unroll
          for (int j = 0; j < 8; ++j)
            if (lane * 8 + j == B + 1) above = from[j];
          above = __shfl_sync(0xffffffffu, warp_sum(above), 0);
          const uint32_t k0 = static_cast<uint32_t>(B) * kBlocks + lane * 8;
          uint32_t cnt[8];
          merged_counts8<C>(cluster, sm, k0, cnt);
          long long kq[8], kfrom[8];
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            kq[j] = 0;
            if (cnt[j] && valid_key(k0 + j)) {
              long long q;
              float e_err;
              value_mass(scaled(key_value(k0 + j), temp), xs_max, q, e_err);
              kq[j] = q * static_cast<long long>(cnt[j]);
            }
          }
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
          ambiguous = (!sure || force_fp64) ? 1 : 0;
          result =
              H >= 0 ? key_value(static_cast<uint32_t>(B) * kBlocks + H) : m;
          z_int = H >= 0 ? mass_k : total;
        }
      }
      if (lane == 0) {
        sm.result = result;
        sm.ambiguous = ambiguous;
        sm.z_int = z_int;
      }
    }
    __syncthreads();
    const bool from_fp64 = sm.ambiguous;
    if (from_fp64) {
      const float r = fp64_cutoff<C>(cluster, sm, m, temp, top_p, tid);
      if (tid == 0) sm.result = r;
    }
    if (tid == 0) {
      const float r = sm.result;
      thresh[row] = r;
      if (r == INFINITY || !valid_key(kmax)) {
        lse[row] = r;
      } else {
        double ln_z;
        if (from_fp64) {
          ln_z = log(sm.z64);
        } else {
          // Z = 2^e (1 + f) with f in [0, 1) taken from the exact integer Z -
          // 2^e, so ln(Z * 2^-40) = (e - 40) ln 2 + log1p(f) >= 0 and stays
          // accurate near 2^40.
          const long long z = sm.z_int;
          const int e = 63 - __clzll(z);
          const float f = static_cast<float>(z - (1LL << e)) *
                          __int_as_float((127 - e) << 23);
          ln_z = (e - 40) * 0.6931471805599453 + static_cast<double>(log1pf(f));
        }
        lse[row] = static_cast<double>(xs_max) + ln_z;
      }
    }
  }
  // Keep every CTA's shared memory alive until rank 0 is done reading it.
  cluster.sync();
}

template <int C>
void launch(const torch::stable::Tensor& logits,
            const torch::stable::Tensor& expanded_idx_mapping,
            const torch::stable::Tensor& temperature,
            const torch::stable::Tensor& top_p, torch::stable::Tensor& thresh,
            torch::stable::Tensor& lse, bool force_fp64, cudaStream_t stream) {
  const int rows = static_cast<int>(logits.size(0));
  const int vocab = static_cast<int>(logits.size(1));
  const size_t smem = sizeof(Smem<C>);
  auto kernel = &topp_hist_cluster_kernel<C>;
  if constexpr (C > 8) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed,
                         1);
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(rows * C);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {C, 1, 1};
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(
      &cfg, kernel,
      reinterpret_cast<const __nv_bfloat16*>(logits.const_data_ptr()),
      logits.stride(0), vocab, expanded_idx_mapping.const_data_ptr<int32_t>(),
      temperature.const_data_ptr<float>(), top_p.const_data_ptr<float>(),
      thresh.mutable_data_ptr<float>(), lse.mutable_data_ptr<double>(),
      force_fp64 ? 1 : 0);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "topp_cutoffs_lse launch failed: ", cudaGetErrorString(err));
}

}  // namespace topp_cutoffs
}  // namespace vllm

void topp_cutoffs_lse(const torch::stable::Tensor& logits,
                      const torch::stable::Tensor& expanded_idx_mapping,
                      const torch::stable::Tensor& temperature,
                      const torch::stable::Tensor& top_p,
                      torch::stable::Tensor& thresh, torch::stable::Tensor& lse,
                      int64_t cluster_size, bool force_fp64) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(logits.is_cuda() && logits.dim() == 2,
                  "logits must be a 2D CUDA tensor");
  STD_TORCH_CHECK(logits.scalar_type() == ScalarType::BFloat16,
                  "logits must be bf16");
  STD_TORCH_CHECK(logits.stride(1) == 1, "logits must have unit vocab stride");
  STD_TORCH_CHECK(expanded_idx_mapping.scalar_type() == ScalarType::Int,
                  "expanded_idx_mapping must be int32");
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
  const int64_t vocab = logits.size(1);
  const int64_t slice = ((vocab + cluster_size - 1) / cluster_size + 7) & ~7;
  STD_TORCH_CHECK(slice <= 65535,
                  "a CTA slice must hold < 65536 tokens; raise cluster_size");
  if (rows == 0) return;
  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  namespace tc = vllm::topp_cutoffs;
  switch (cluster_size) {
    case 4:
      tc::launch<4>(logits, expanded_idx_mapping, temperature, top_p, thresh,
                    lse, force_fp64, stream);
      break;
    case 8:
      tc::launch<8>(logits, expanded_idx_mapping, temperature, top_p, thresh,
                    lse, force_fp64, stream);
      break;
    case 16:
      tc::launch<16>(logits, expanded_idx_mapping, temperature, top_p, thresh,
                     lse, force_fp64, stream);
      break;
    default:
      STD_TORCH_CHECK(false, "cluster_size must be 4, 8 or 16");
  }
}
