// SPDX-License-Identifier: Apache-2.0
#ifndef FLASHINFER_LITEDSA_UNION_CUH_
#define FLASHINFER_LITEDSA_UNION_CUH_
// Group-union + query-major membership builder for grouped (packed) sparse
// prefill attention: G adjacent queries share one attention call over the
// UNION of their top-k lists; a per-query membership bitmask restores EXACT
// per-query attend-sets inside the kernel (LSE-equal to the per-query path).
//
// One 1024-thread CTA per group, smem position bitmap (S <= 1M -> 128KB):
//   1) insert: G*k top-k indices set bits (smem atomicOr) + a summary bit
//      per 1024-position segment;
//   2) sweep: per-thread popc with per-4-word partial sums -> two-level
//      exclusive scan -> ascending emit. The emitted VALUE is converted
//      logical->physical in place via the block table (fused; -1 when the
//      block is out of range). A 4-word-granularity uint16 rank table is a
//      free byproduct of the sweep;
//   3) rank pass: each (query, candidate) finds its union rank via the rank
//      table (<=3 word popcs) and sets bit r in the query's membership row,
//      accumulated in smem and dumped whole (overwrite; no pre-zeroing of
//      the destination needed).
//
// OVERWRITE-WRITE CONTRACT: outputs are caller-owned persistent buffers,
// refilled in place with NO fill/zero pre-pass. u_phys padding beyond
// counts[g] stays STALE — every consumer must bound reads by counts (the
// masked attention kernel's topk_length does). Ported 1:1 from the
// research-stack union_bitmap.cu (bit-exact validated vs the allocating
// builders, incl. dirty-buffer double-overwrite).

#include <cuda_runtime.h>
#include <cstdint>

namespace flashinfer::litedsa {

constexpr int kLiteDSAUnionThreads = 1024;

__global__ void litedsa_union_qm_kernel(
    const int32_t* __restrict__ idx,     // [T, k] logical, -1 invalid
    int32_t* __restrict__ u_phys,        // [ng, cap] PHYSICAL slots out
    int32_t* __restrict__ counts,        // [ng]
    uint32_t* __restrict__ memb_qm,      // [ng, G, cap/32] query-major bits
    const int32_t* __restrict__ req_pg,  // [ng] request id per group
    const int32_t* __restrict__ btab,    // [nreq, bt_blocks] block table
    const int bt_stride, const int bt_blocks, const int bsz, const int k,
    const int G, const int cap, const int s_words) {
  extern __shared__ uint32_t bm[];  // [s_words] position bitmap
  __shared__ uint32_t summary[kLiteDSAUnionThreads / 32];
  __shared__ int warp_sums[kLiteDSAUnionThreads / 32 + 1];
  const int g = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int32_t* bt_row = btab + (long)req_pg[g] * bt_stride;

  {
    uint4* b4 = reinterpret_cast<uint4*>(bm);
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (int i = tid; i < s_words / 4; i += kLiteDSAUnionThreads) b4[i] = z;
    if (tid < kLiteDSAUnionThreads / 32) summary[tid] = 0u;
  }
  __syncthreads();

  const long base = (long)g * G * k;
  const int total = G * k;
  for (int t = tid; t < total; t += kLiteDSAUnionThreads) {
    const int p = idx[base + t];
    if (p >= 0 && p < s_words * 32) {
      // Publish a summary bit only when a bitmap word transitions from
      // entirely empty. This removes the redundant summary atomic from
      // duplicate positions and from later bits in the same word.
      const uint32_t old = atomicOr(&bm[p >> 5], 1u << (p & 31));
      if (old == 0u) atomicOr(&summary[(p >> 10) >> 5], 1u << ((p >> 10) & 31));
    }
  }
  __syncthreads();

  // sweep: thread t owns one 1024-position segment (32 words)
  constexpr int wpt = 32;
  const int w0 = min(tid * wpt, s_words);
  const int w1 = min(w0 + wpt, s_words);
  const bool dirty = (summary[tid >> 5] >> (tid & 31)) & 1u;
  int csum[8];
  int my = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) csum[i] = 0;
  if (dirty) {
    const uint4* s4 = reinterpret_cast<const uint4*>(bm);
    for (int w = w0; w + 4 <= w1; w += 4) {
      const uint4 v4 = s4[w >> 2];
      const int c = __popc(v4.x) + __popc(v4.y) + __popc(v4.z) + __popc(v4.w);
      csum[(w - w0) >> 2] = c;
      my += c;
    }
  }
  int pref = my;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int v = __shfl_up_sync(0xffffffffu, pref, off);
    if (lane >= off) pref += v;
  }
  if (lane == 31) warp_sums[wid + 1] = pref;
  if (tid == 0) warp_sums[0] = 0;
  __syncthreads();
  if (wid == 0) {
    int wv = (lane < kLiteDSAUnionThreads / 32) ? warp_sums[lane + 1] : 0;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, wv, off);
      if (lane >= off) wv += v;
    }
    if (lane < kLiteDSAUnionThreads / 32) warp_sums[lane + 1] = wv;
  }
  __syncthreads();
  int pos = warp_sums[wid] + pref - my;
  if (tid == kLiteDSAUnionThreads - 1)
    counts[g] = warp_sums[kLiteDSAUnionThreads / 32];

  // 4-word-granularity rank prefix table (uint16; union count <= cap <=
  // 32768 fits): shrinks the rank pass's in-segment popc loop to <=3 words
  uint16_t* sub_rank = reinterpret_cast<uint16_t*>(bm + s_words + 16 * 1024);
  {
    int r = pos;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (w0 + (i << 2) < s_words) {
        sub_rank[(w0 >> 2) + i] = (uint16_t)r;
        r += csum[i];
      }
    }
  }

  if (dirty) {
    int ps = pos;
    for (int w = w0; w < w1; ++w) {
      uint32_t m = bm[w];
      while (m) {
        const int b = __ffs(m) - 1;
        m &= m - 1;
        if (ps < cap) {
          const int p = (w << 5) | b;
          const int blk = p / bsz;
          u_phys[(long)g * cap + ps] =
              blk < bt_blocks ? bt_row[blk] * bsz + (p - blk * bsz) : -1;
        }
        ++ps;
      }
    }
  }
  __syncthreads();  // bitmap + sub_rank stable for the rank pass

  // Query-major membership: only the ceil(count/128)*4 words that masked
  // attention can consume are live. Accumulate and dump that prefix; the
  // global tensor keeps its fixed capw row stride and its unused tail may
  // remain stale.
  uint32_t* qm_s = bm + s_words;
  const int capw = cap >> 5;
  const int active_capw =
      (min(warp_sums[kLiteDSAUnionThreads / 32], cap) + 31) >> 5;
  const int active_stride = max(4, (active_capw + 3) & ~3);
  for (int i = tid; i < G * active_stride; i += kLiteDSAUnionThreads)
    qm_s[i] = 0u;
  __syncthreads();
  for (int t = tid; t < total; t += kLiteDSAUnionThreads) {
    const int p = idx[base + t];
    if (p < 0 || p >= s_words * 32) continue;
    const int pword = p >> 5;
    const int sub = pword >> 2;
    int r = sub_rank[sub];
    for (int w = sub << 2; w < pword; ++w) r += __popc(bm[w]);
    r += __popc(bm[pword] & ((1u << (p & 31)) - 1u));
    if (r < cap) {
      const int q = t / k;
      atomicOr(&qm_s[q * active_stride + (r >> 5)], 1u << (r & 31));
    }
  }
  __syncthreads();
  {
    uint4* dst = reinterpret_cast<uint4*>(memb_qm + (long)g * G * capw);
    const uint4* src = reinterpret_cast<const uint4*>(qm_s);
    const int stride4 = active_stride >> 2;
    const int dst_stride4 = capw >> 2;
    for (int i = tid; i < G * stride4; i += kLiteDSAUnionThreads) {
      const int q = i / stride4;
      const int j = i - q * stride4;
      dst[q * dst_stride4 + j] = src[i];
    }
  }
}

// dynamic smem: bitmap + qm rows (64KB for G*capw<=16K words) + rank table
inline int litedsa_union_qm_smem_bytes(int s_words) {
  return s_words * 4 + 64 * 1024 + (s_words / 4) * 2;
}

}  // namespace flashinfer::litedsa

#endif  // FLASHINFER_LITEDSA_UNION_CUH_
