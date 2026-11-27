// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef COOPERATIVE_GRID_DYN_CUH_
#define COOPERATIVE_GRID_DYN_CUH_

#include "cooperative_grid_topk.cuh"

namespace vllm::cooperative_grid {

constexpr uint32_t kChunkTiles = 2;

__device__ __forceinline__ uint32_t ld_acquire32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ uint32_t add_release32(uint32_t* p, uint32_t v) {
  uint32_t o;
  asm volatile("atom.add.release.gpu.global.u32 %0, [%1], %2;"
               : "=r"(o)
               : "l"(p), "r"(v)
               : "memory");
  return o;
}
__device__ __forceinline__ uint32_t add_acqrel32(uint32_t* p, uint32_t v) {
  uint32_t o;
  asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;"
               : "=r"(o)
               : "l"(p), "r"(v)
               : "memory");
  return o;
}

// One CTA joins long row `row` (length len): claims chunks until none are
// left, then takes part in the row's barrier and ticket. Returns at once if
// every chunk is already claimed.
__device__ __forceinline__ void dyn_row_topk(
    const float* __restrict__ logits, uint32_t stride, int32_t len,
    int32_t* __restrict__ out, uint32_t topk, uint8_t* __restrict__ ws,
    uint32_t* __restrict__ dstate, uint32_t row) {
  extern __shared__ __align__(16) uint32_t s_dyn[];
  __shared__ uint32_t s_hist[kBins];
  __shared__ uint32_t s_sel[4], s_red[2], s_scan[32];
  __shared__ uint32_t s_n, s_c, s_flag;

  const uint32_t tx = threadIdx.x;
  uint32_t* st = dstate + row * kDynRowWords;
  uint32_t* ghist = st + kDynHdr;
  const uint32_t L = static_cast<uint32_t>(max(len, 0));
  const uint32_t ntiles = (L + kTile - 1) / kTile,
                 nch = (ntiles + kChunkTiles - 1) / kChunkTiles;

  if (tx == 0) {
    uint32_t c = nch;
    if (ld_acquire32(st) < nch) c = atomicAdd(st, 1u);
    if (c < nch)
      atomicAdd(&st[2], 1u);  // joined: counted before any of its tiles is
    s_c = c;
  }
  __syncthreads();
  uint32_t c = s_c;
  if (c >= nch) return;

  const uint32_t* inb = reinterpret_cast<const uint32_t*>(
      logits + static_cast<size_t>(row) * stride);
  int32_t* o = out + static_cast<size_t>(row) * topk;
  uint32_t* ckey = reinterpret_cast<uint32_t*>(ws + row * kRowBytes);
  int32_t* cidx = reinterpret_cast<int32_t*>(ckey + kCand);
  uint32_t* skey = s_dyn;
  int32_t* sidx = reinterpret_cast<int32_t*>(s_dyn + kStage);
  uint32_t* s_samp = s_dyn + 2 * kStage;
  const uint4* v4 = reinterpret_cast<const uint4*>(inb);

  // First chunk's tiles in flight while the cutoff is worked out.
  uint4 pre[kChunkTiles];
#pragma unroll
  for (uint32_t j = 0; j < kChunkTiles; j++) {
    const uint32_t t = c * kChunkTiles + j, i0 = t * kTile + 4 * tx;
    pre[j] =
        (t < ntiles && i0 < L) ? __ldcs(v4 + i0 / 4) : make_uint4(0, 0, 0, 0);
  }

  // 1. The cutoff (as row_topk in cooperative_grid_topk.cuh; the same in every
  //    CTA of the row).
  const uint32_t S = min(L, kSamples);
  uint32_t mx = 0, nmn = 0;
  for (uint32_t t = tx; t < S; t += kThreads) {
    const uint32_t p =
        S == L ? t
               : static_cast<uint32_t>((static_cast<uint64_t>(2 * t + 1) * L) /
                                       (2 * S));
    const uint32_t k = okey(__ldg(inb + p));
    s_samp[t] = k;
    mx = max(mx, k);
    nmn = max(nmn, ~k);
  }
  const uint2 ex = cta_max2(mx, nmn, s_red);
  const uint32_t smax = ex.x, srange = smax - ~ex.y;
  const uint32_t shb = srange ? 32 - __clz(srange) : 0u;
  uint32_t q;
  if (S == L) {
    q = topk;
  } else {
    const float e = static_cast<float>(topk) * S / static_cast<float>(L);
    q = min(S, static_cast<uint32_t>(e + 5.0f * sqrtf(e) + 5.0f) + 1u);
  }
  const uint32_t keep = S == L ? 0u : (shb > 11 ? shb - 11 : 0u);
  uint2 d = select_low([&](uint32_t i) { return smax - s_samp[i]; }, S, q, shb,
                       keep, s_hist, s_sel, s_scan);
  const bool crowded =
      keep && s_hist[(d.x >> keep) & (kBins - 1)] > max(8u, q / 4);
  __syncthreads();
  if (crowded) {
    d = select_low([&](uint32_t i) { return smax - s_samp[i]; }, S, q, shb, 0,
                   s_hist, s_sel, s_scan);
  } else if (keep) {
    d.x |= (1u << keep) - 1;
  }
  const uint32_t cut = d.x > smax ? 0u : smax - d.x;

  // 2. Claimed chunks: survivors >= cut in shared memory, their histogram.
  for (uint32_t b = tx; b < kBins; b += kThreads) s_hist[b] = 0;
  if (tx == 0) s_n = 0;
  __syncthreads();
  const auto emit = [&](bool keepit, uint32_t key, uint32_t i) {
    if (__ballot_sync(0xFFFFFFFFu, keepit) == 0) return;
    const uint32_t p = warp_slot(keepit, &s_n);
    if (keepit) {
      atomicAdd(&s_hist[lbin(key - cut)], 1u);
      if (p < kStage) {
        skey[p] = key;
        sidx[p] = static_cast<int32_t>(i);
      }
    }
  };
  uint32_t mine = 0;
  bool first = true;
#pragma unroll 1
  for (;;) {
#pragma unroll
    for (uint32_t j = 0; j < kChunkTiles; j++) {
      const uint32_t t = c * kChunkTiles + j, i0 = t * kTile + 4 * tx;
      if (t >= ntiles) break;
      uint4 w = make_uint4(0, 0, 0, 0);
      if (first) {
#pragma unroll
        for (uint32_t jj = 0; jj < kChunkTiles; jj++) {
          if (jj == j) w = pre[jj];
        }
      } else if (i0 < L) {
        w = __ldcs(v4 + i0 / 4);
      }
      const uint32_t k0 = okey(w.x), k1 = okey(w.y), k2 = okey(w.z),
                     k3 = okey(w.w);
      emit(i0 < L && k0 >= cut, k0, i0);
      emit(i0 + 1 < L && k1 >= cut, k1, i0 + 1);
      emit(i0 + 2 < L && k2 >= cut, k2, i0 + 2);
      emit(i0 + 3 < L && k3 >= cut, k3, i0 + 3);
      mine++;
    }
    first = false;
    __syncthreads();
    if (tx == 0) s_c = atomicAdd(st, 1u);
    __syncthreads();
    c = s_c;
    if (c >= nch) break;
  }
  const uint32_t ns = s_n;

  // Add to the row's histogram; barrier: every tile of the row done.
  for (uint32_t b = tx; b < kBins; b += kThreads) {
    if (s_hist[b]) atomicAdd(&ghist[b], s_hist[b]);
  }
  if (tx == 0) {
    atomicAdd(&st[4], ns);
    if (ns > kStage) atomicExch(&st[7], 1u);
  }
  __syncthreads();
  if (tx == 0) {
    add_release32(&st[1], mine);
    while (ld_acquire32(&st[1]) < ntiles) __nanosleep(20);
    s_flag = __ldcg(&st[7]) != 0 || __ldcg(&st[4]) < topk;
  }
  __syncthreads();
  bool fb = s_flag;
  uint32_t B = 0;
  if (!fb) {
    for (uint32_t b = tx; b < kBins; b += kThreads)
      s_hist[b] = __ldcg(ghist + b);
    __syncthreads();
    pick_bin(s_hist, topk, true, s_sel, s_scan);
    B = s_sel[0];
    fb = s_sel[2] > kSamples;  // the same in every CTA (same histogram)
  }
  // The row's verdict, the same in every CTA that joined: 1 done here, 2 the
  // row's own cluster redoes it with the cluster algorithm (nothing written).
  if (tx == 0) atomicExch(&st[8], fb ? 2u : 1u);
  {
    if (!fb) {
      for (uint32_t i0 = 0; i0 < ns; i0 += kThreads) {
        const uint32_t i = i0 + tx;
        const bool ok = i < ns;
        const uint32_t k = ok ? skey[i] : 0u, b = lbin(k - cut);
        const bool gt = ok && b > B, eq = ok && b == B;
        const uint32_t pg = warp_slot(gt, &st[5]);
        if (gt) o[pg] = sidx[i];
        const uint32_t pe = warp_slot(eq, &st[6]);
        if (eq && pe < kCand) {
          ckey[pe] = k;
          cidx[pe] = sidx[i];
        }
      }
    }
  }

  // Ticket: the last CTA that joined ranks the candidates.
  __syncthreads();
  if (tx == 0) {
    const uint32_t old = add_acqrel32(&st[3], 1u);
    s_flag = old + 1 == ld_acquire32(&st[2]);
  }
  __syncthreads();
  if (!s_flag || fb) return;
  const uint32_t above = __ldcg(&st[5]), m = __ldcg(&st[6]);
  const uint32_t need = topk - above;
  uint32_t* lk = s_samp;
  for (uint32_t i = tx; i < m; i += kThreads) lk[i] = __ldcg(ckey + i);
  __syncthreads();
  rank_write([&](uint32_t i) { return lk[i]; },
             [&](uint32_t i) { return __ldcg(cidx + i); }, m, need, o + above,
             s_hist, s_sel, s_scan);
}

}  // namespace vllm::cooperative_grid

#endif  // COOPERATIVE_GRID_DYN_CUH_
