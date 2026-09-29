// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Paged prefill attention v2 INT4 per-token-head kernel for AMD RDNA3
// (gfx1100). Cache-only: all K/V read from INT4 paged cache. RHT fused.
//
// Uses bf16 WMMA for BOTH Q×K and P×V (identical compute path to INT8).
// K nibbles dequanted to fp16 in loader — same as INT8's int8→fp16.
// No Q quantization overhead. Advantage over INT8: reads HALF the K/V bytes.
//
// Layout:
//   K/V cache (uint8): [num_blocks, block_size, num_kv_heads, head_size//2]
//   k/v_scale_cache (fp32): [num_blocks, block_size, num_kv_heads]
//     (bits 0-3 = zero-point, bits 4-31 = float32 scale)

#include <cstdint>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "paged_prefill_attn_rdna3.cuh"

namespace vllm {
namespace prefill_attn_rdna3_v2_int4 {

#if defined(USE_ROCM)

using vllm::prefill_attn_rdna3::bf16_t;
using vllm::prefill_attn_rdna3::to_T;
using vllm::prefill_attn_rdna3::v8fp32;
using vllm::prefill_attn_rdna3::wmma_mma;
using vllm::prefill_attn_rdna3::WmmaNative;

constexpr int K_TILE = 16;
constexpr int M_PER_WAVE = 16;

__device__ __forceinline__ float wave16_max(float v) {
  v = fmaxf(v, __shfl_xor(v, 1));
  v = fmaxf(v, __shfl_xor(v, 2));
  v = fmaxf(v, __shfl_xor(v, 4));
  v = fmaxf(v, __shfl_xor(v, 8));
  return v;
}
__device__ __forceinline__ float wave16_sum(float v) {
  v += __shfl_xor(v, 1);
  v += __shfl_xor(v, 2);
  v += __shfl_xor(v, 4);
  v += __shfl_xor(v, 8);
  return v;
}

__device__ __forceinline__ float extract_scale(float raw) {
  int bits;
  __builtin_memcpy(&bits, &raw, 4);
  int sb = bits & ~0xF;
  float s;
  __builtin_memcpy(&s, &sb, 4);
  return s;
}
__device__ __forceinline__ int extract_zp(float raw) {
  int bits;
  __builtin_memcpy(&bits, &raw, 4);
  return bits & 0xF;
}

// ---- RHT butterfly (forward Q, inverse output) ----------------------------

template <typename E, int FRAGS>
__device__ __forceinline__ void rht_forward_q(E (*frags)[16], const float* d1) {
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh)
  #pragma unroll
    for (int k = 0; k < 16; ++k)
      frags[dh][k] = (E)((float)frags[dh][k] * d1[dh * 16 + k]);
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    int dist = 1 << j;
  #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh)
  #pragma unroll
      for (int b = 0; b < 16; b += 2 * dist)
  #pragma unroll
        for (int o = 0; o < dist; ++o) {
          float a = (float)frags[dh][b + o],
                b2 = (float)frags[dh][b + o + dist];
          frags[dh][b + o] = (E)(a + b2);
          frags[dh][b + o + dist] = (E)(a - b2);
        }
  }
  #pragma unroll
  for (int j = 0; (1 << j) < FRAGS; ++j) {
    int dd = 1 << j;
  #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh) {
      int p = dh ^ dd;
      if (dh < p) {
  #pragma unroll
        for (int k = 0; k < 16; ++k) {
          float a = (float)frags[dh][k], b = (float)frags[p][k];
          frags[dh][k] = (E)(a + b);
          frags[p][k] = (E)(a - b);
        }
      }
    }
  }
}

template <int FRAGS>
__device__ __forceinline__ void rht_inverse_output(v8fp32 (&acc)[FRAGS],
                                                   const float* d1, int ll,
                                                   float inv_hs) {
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    int dist = 1 << j;
    bool lo = ((ll >> j) & 1) == 0;
  #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh)
  #pragma unroll
      for (int i = 0; i < 8; ++i) {
        float p = __shfl_xor(acc[dh][i], dist);
        acc[dh][i] = lo ? (acc[dh][i] + p) : (p - acc[dh][i]);
      }
  }
  #pragma unroll
  for (int j = 0; (1 << j) < FRAGS; ++j) {
    int dd = 1 << j;
  #pragma unroll
    for (int dh = 0; dh < FRAGS; ++dh) {
      int p = dh ^ dd;
      if (dh < p)
  #pragma unroll
        for (int i = 0; i < 8; ++i) {
          float a = acc[dh][i], b = acc[p][i];
          acc[dh][i] = a + b;
          acc[p][i] = a - b;
        }
    }
  }
  #pragma unroll
  for (int dh = 0; dh < FRAGS; ++dh) {
    float ss = d1[dh * 16 + ll] * inv_hs;
  #pragma unroll
    for (int i = 0; i < 8; ++i) acc[dh][i] *= ss;
  }
}

// ---- Scale pre-loader: first K_TILE threads pre-load scales+zp ----------
// Eliminates redundant global reads (DC threads per K position all read the
// same scale). Also removes if(dh==0) divergence in the K/V loaders.

template <int HEAD_SIZE>
__device__ __forceinline__ void preload_scales(
    const float* __restrict__ sc_cache, const int* __restrict__ bt, int si,
    int kh, int sn, int bound, int bs, int mbps, int64_t ssb, int64_t sss,
    int64_t ssh, float* __restrict__ sc_lds, int* __restrict__ zp_lds,
    int tid) {
  if (tid < K_TILE) {
    int ak = sn + tid;
    bool v = ak < bound;
    if (v) {
      int lb = ak / bs, sl = ak - lb * bs;
      int pb = bt[si * mbps + lb];
      float raw = sc_cache[pb * ssb + sl * sss + kh * ssh];
      sc_lds[tid] = extract_scale(raw);
      zp_lds[tid] = extract_zp(raw);
    } else {
      sc_lds[tid] = 0.0f;
      zp_lds[tid] = 0;
    }
  }
}

// ---- K loader: nibble→center→fp16 (reads HALF bytes vs INT8) --------------
// Scales+zp already in LDS from preload_scales.

template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_k_int4(T* __restrict__ K,
                                            const uint8_t* __restrict__ kc,
                                            const int* __restrict__ bt,
                                            const int* __restrict__ zp_lds,
                                            int si, int kh, int sn, int bound,
                                            int bs, int mbps, int64_t skb,
                                            int64_t sks, int64_t skh, int tid) {
  constexpr int DC = HEAD_SIZE / 16, X = 16 / sizeof(T);
  int ki = tid / DC, dh = tid % DC;
  int ak = sn + ki;
  bool v = ak < bound;
  int lb = ak / bs, sl = ak - lb * bs;
  int pb = v ? bt[si * mbps + lb] : 0;
  int zp = v ? zp_lds[ki] : 0;

  const uint8_t* src = kc + (int64_t)pb * skb + (int64_t)sl * sks +
                       (int64_t)kh * skh + (int64_t)(dh * 8);
  uint8_t pk[8];
  if (v)
    *(uint64_t*)pk = *(const uint64_t*)src;
  else
    *(uint64_t*)pk = 0;

  T dq[16];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    dq[2 * i] = to_T<T>((float)((int)(pk[i] & 0xF) - zp));
    dq[2 * i + 1] = to_T<T>((float)((int)((pk[i] >> 4) & 0xF) - zp));
  }
  *(int4*)&K[(dh * 2 + 0) * (K_TILE * X) + ki * X] = *(int4*)&dq[0];
  *(int4*)&K[(dh * 2 + 1) * (K_TILE * X) + ki * X] = *(int4*)&dq[8];
}

// ---- V loader: nibble→center→fp16 (transposed) ---------------------------
// Scales+zp already in LDS from preload_scales.

template <typename T, int HEAD_SIZE>
__device__ __forceinline__ void load_v_int4(T* __restrict__ V,
                                            const uint8_t* __restrict__ vc,
                                            const int* __restrict__ bt,
                                            const int* __restrict__ zp_lds,
                                            int si, int kh, int sn, int bound,
                                            int bs, int mbps, int64_t svb,
                                            int64_t svs, int64_t svh, int tid) {
  constexpr int DC = HEAD_SIZE / 16;
  int vk = max(0, min(K_TILE, bound - sn));
  int ms = tid / DC, md = tid % DC;
  bool v = ms < vk;
  int ak = sn + ms;
  int lb = ak / bs, sl = ak - lb * bs;
  int pb = v ? bt[si * mbps + lb] : 0;
  int zp = v ? zp_lds[ms] : 0;

  uint8_t pk[8];
  if (v) {
    const uint8_t* src = vc + (int64_t)pb * svb + (int64_t)sl * svs +
                         (int64_t)kh * svh + (int64_t)(md * 8);
    *(uint64_t*)pk = *(const uint64_t*)src;
  } else
    *(uint64_t*)pk = 0;

  int db = md * 16;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    V[(db + 2 * i) * K_TILE + ms] = to_T<T>((float)((int)(pk[i] & 0xF) - zp));
    V[(db + 2 * i + 1) * K_TILE + ms] =
        to_T<T>((float)((int)((pk[i] >> 4) & 0xF) - zp));
  }
}

// ---- attn_step: bf16 WMMA (identical to INT8 kernel) ----------------------

template <typename T, int HEAD_SIZE, int X, bool CAUSAL>
__device__ __forceinline__ void attn_step(
    const T* K, const T* V, T* Pw, const float* ksc, const float* vsc,
    typename WmmaNative<T>::v16 (&qf)[HEAD_SIZE / 16],
    v8fp32 (&oa)[HEAD_SIZE / 16], float (&ms)[8], float (&ls)[8], int wqa,
    int sn, int vqc, int vkc, float sm, int lane, int ll, int lh) {
  using V16 = typename WmmaNative<T>::v16;
  constexpr int F = HEAD_SIZE / 16;

  // Q×K dot product: accumulate WMMA across all head frags.
  // For F>8 (HS=256), unroll 1 forces register recycling so the compiler
  // spills qf/oa to scratch instead of keeping 16 frags live simultaneously.
  // Scratch bandwidth penalty is ~1-2% of total WMMA compute.
  v8fp32 sa = {0, 0, 0, 0, 0, 0, 0, 0};
  if constexpr (F <= 8) {
  #pragma unroll
    for (int dh = 0; dh < F; ++dh) {
      V16 bf;
      int4 lo = *(const int4*)&K[(dh * 2 + 0) * (K_TILE * X) + ll * X];
      int4 hi = *(const int4*)&K[(dh * 2 + 1) * (K_TILE * X) + ll * X];
      __builtin_memcpy(&bf, &lo, 16);
      __builtin_memcpy(((char*)&bf) + 16, &hi, 16);
      sa = wmma_mma(qf[dh], bf, sa);
    }
  } else {
  #pragma unroll 1
    for (int dh = 0; dh < F; ++dh) {
      V16 bf;
      int4 lo = *(const int4*)&K[(dh * 2 + 0) * (K_TILE * X) + ll * X];
      int4 hi = *(const int4*)&K[(dh * 2 + 1) * (K_TILE * X) + ll * X];
      __builtin_memcpy(&bf, &lo, 16);
      __builtin_memcpy(((char*)&bf) + 16, &hi, 16);
      sa = wmma_mma(qf[dh], bf, sa);
    }
  }

  float kc = ksc[ll];
  int ak = sn + ll;
  bool ks = ll < vkc;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    int mr = 2 * i + lh;
    bool mq = mr < vqc;
    bool keep = mq && ks;
    if constexpr (CAUSAL) keep = keep && (ak <= wqa + mr);
    sa[i] = keep ? sa[i] * sm * kc : -INFINITY;
  }

  float mi[8], mn[8], al[8], pi[8], li[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    mi[i] = wave16_max(sa[i]);
    mn[i] = fmaxf(ms[i], mi[i]);
    al[i] = (ms[i] == -INFINITY) ? 0.f : __expf(ms[i] - mn[i]);
    pi[i] = (mn[i] == -INFINITY) ? 0.f : __expf(sa[i] - mn[i]);
    li[i] = wave16_sum(pi[i]);
    ls[i] = ls[i] * al[i] + li[i];
    ms[i] = mn[i];
  }
  if constexpr (F <= 8) {
  #pragma unroll
    for (int dh = 0; dh < F; ++dh)
  #pragma unroll
      for (int i = 0; i < 8; ++i) oa[dh][i] *= al[i];
  } else {
  #pragma unroll 1
    for (int dh = 0; dh < F; ++dh)
  #pragma unroll
      for (int i = 0; i < 8; ++i) oa[dh][i] *= al[i];
  }

  float vc = vsc[ll];
  #pragma unroll
  for (int i = 0; i < 8; ++i) pi[i] *= vc;

  #pragma unroll
  for (int i = 0; i < 8; ++i) Pw[(2 * i + lh) * K_TILE + ll] = to_T<T>(pi[i]);
  V16 pf;
  int4 pl = *(const int4*)&Pw[ll * K_TILE];
  int4 ph = *(const int4*)&Pw[ll * K_TILE + 8];
  __builtin_memcpy(&pf, &pl, 16);
  __builtin_memcpy(((char*)&pf) + 16, &ph, 16);

  if constexpr (F <= 8) {
  #pragma unroll
    for (int dh = 0; dh < F; ++dh) {
      V16 vf;
      int4 vl = *(const int4*)&V[(dh * 16 + ll) * K_TILE];
      int4 vh = *(const int4*)&V[(dh * 16 + ll) * K_TILE + 8];
      __builtin_memcpy(&vf, &vl, 16);
      __builtin_memcpy(((char*)&vf) + 16, &vh, 16);
      oa[dh] = wmma_mma(pf, vf, oa[dh]);
    }
  } else {
  #pragma unroll 1
    for (int dh = 0; dh < F; ++dh) {
      V16 vf;
      int4 vl = *(const int4*)&V[(dh * 16 + ll) * K_TILE];
      int4 vh = *(const int4*)&V[(dh * 16 + ll) * K_TILE + 8];
      __builtin_memcpy(&vf, &vl, 16);
      __builtin_memcpy(((char*)&vf) + 16, &vh, 16);
      oa[dh] = wmma_mma(pf, vf, oa[dh]);
    }
  }
}

// ---- Main kernel ----------------------------------------------------------

// See the INT8 sibling: declaring the real HEAD_SIZE-thread bound lets the
// compiler use up to 256 VGPRs instead of the 1024-default-conservative 192,
// cutting the per-thread scratch spill (1088 B at HS=256) on long prefill.
template <typename T, int HEAD_SIZE>
__global__ void __launch_bounds__(HEAD_SIZE) paged_prefill_attn_kernel_v2_int4(
    T* __restrict__ out, const T* __restrict__ q,
    const uint8_t* __restrict__ k_cache, const uint8_t* __restrict__ v_cache,
    const float* __restrict__ k_scale_cache,
    const float* __restrict__ v_scale_cache,
    const float* __restrict__ rht_signs, const int* __restrict__ block_table,
    const int* __restrict__ cu_seqlens_q, const int* __restrict__ seq_lens,
    int num_query_heads, int num_kv_heads, int block_size,
    int max_blocks_per_seq, float sm_scale, bool causal, int64_t sq0,
    int64_t sq1, int64_t skb, int64_t sks, int64_t skh, int64_t svb,
    int64_t svs, int64_t svh, int64_t ssb, int64_t sss, int64_t ssh,
    int64_t svvb, int64_t svvs, int64_t svvh, int64_t so0, int64_t so1) {
  using V16 = typename WmmaNative<T>::v16;
  using E = typename WmmaNative<T>::elem;
  constexpr int F = HEAD_SIZE / 16, X = 16 / sizeof(T);
  constexpr int TH = HEAD_SIZE, NW = TH / 32, BM = NW * M_PER_WAVE;

  int si = blockIdx.x, hi = blockIdx.y, qi = blockIdx.z;
  int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  int ll = lane & 15, lh = lane >> 4;

  int qst = cu_seqlens_q[si], ql = cu_seqlens_q[si + 1] - qst;
  int sl = seq_lens[si], cl = sl - ql;
  int qts = qi * BM;
  if (qts >= ql) return;
  int wqo = wid * M_PER_WAVE, mqp = qts + wqo + ll;
  bool vq = mqp < ql;
  int vqw = max(0, min(M_PER_WAVE, ql - qts - wqo));
  int wqa = cl + qts + wqo;
  int kvh = hi / (num_query_heads / num_kv_heads);

  __shared__ T K_lds[F * 2 * K_TILE * X];
  __shared__ T V_lds[HEAD_SIZE * K_TILE];
  __shared__ T P_lds[NW][M_PER_WAVE * K_TILE];
  __shared__ float ksl[K_TILE], vsl[K_TILE];
  __shared__ float d1[HEAD_SIZE];
  __shared__ int kzp[K_TILE], vzp[K_TILE];  // pre-loaded zero-points
  T* Pw = &P_lds[wid][0];

  if (tid < HEAD_SIZE) d1[tid] = rht_signs[tid];
  __syncthreads();

  V16 qf[F];
  if (vq) {
    const T* qr = q + (int64_t)(qst + mqp) * sq0 + (int64_t)hi * sq1;
  #pragma unroll
    for (int dh = 0; dh < F; ++dh)
      __builtin_memcpy(&qf[dh], qr + dh * 16, sizeof(V16));
  } else {
  #pragma unroll
    for (int dh = 0; dh < F; ++dh)
  #pragma unroll
      for (int k = 0; k < 16; ++k) qf[dh][k] = (E)0;
  }

  rht_forward_q<E, F>(reinterpret_cast<E(*)[16]>(qf), d1);

  float ms[8], ls[8];
  v8fp32 oa[F];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    ms[i] = -INFINITY;
    ls[i] = 0.f;
  }
  #pragma unroll
  for (int dh = 0; dh < F; ++dh) oa[dh] = (v8fp32){0, 0, 0, 0, 0, 0, 0, 0};

  for (int sn = 0; sn < cl; sn += K_TILE) {
    preload_scales<HEAD_SIZE>(k_scale_cache, block_table, si, kvh, sn, cl,
                              block_size, max_blocks_per_seq, ssb, sss, ssh,
                              ksl, kzp, tid);
    preload_scales<HEAD_SIZE>(v_scale_cache, block_table, si, kvh, sn, cl,
                              block_size, max_blocks_per_seq, svvb, svvs, svvh,
                              vsl, vzp, tid);
    load_k_int4<T, HEAD_SIZE>(K_lds, k_cache, block_table, kzp, si, kvh, sn, cl,
                              block_size, max_blocks_per_seq, skb, sks, skh,
                              tid);
    load_v_int4<T, HEAD_SIZE>(V_lds, v_cache, block_table, vzp, si, kvh, sn, cl,
                              block_size, max_blocks_per_seq, svb, svs, svh,
                              tid);
    __syncthreads();
    attn_step<T, HEAD_SIZE, X, false>(K_lds, V_lds, Pw, ksl, vsl, qf, oa, ms,
                                      ls, wqa, sn, vqw, min(K_TILE, cl - sn),
                                      sm_scale, lane, ll, lh);
    __syncthreads();
  }

  int l2e = causal ? min(sl, cl + qts + max(0, min(BM, ql - qts))) : sl;
  for (int sn = cl; sn < l2e; sn += K_TILE) {
    preload_scales<HEAD_SIZE>(k_scale_cache, block_table, si, kvh, sn, sl,
                              block_size, max_blocks_per_seq, ssb, sss, ssh,
                              ksl, kzp, tid);
    preload_scales<HEAD_SIZE>(v_scale_cache, block_table, si, kvh, sn, sl,
                              block_size, max_blocks_per_seq, svvb, svvs, svvh,
                              vsl, vzp, tid);
    load_k_int4<T, HEAD_SIZE>(K_lds, k_cache, block_table, kzp, si, kvh, sn, sl,
                              block_size, max_blocks_per_seq, skb, sks, skh,
                              tid);
    load_v_int4<T, HEAD_SIZE>(V_lds, v_cache, block_table, vzp, si, kvh, sn, sl,
                              block_size, max_blocks_per_seq, svb, svs, svh,
                              tid);
    __syncthreads();
    attn_step<T, HEAD_SIZE, X, true>(K_lds, V_lds, Pw, ksl, vsl, qf, oa, ms, ls,
                                     wqa, sn, vqw, min(K_TILE, l2e - sn),
                                     sm_scale, lane, ll, lh);
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    float li = 1.f / (ls[i] + 1e-10f);
  #pragma unroll
    for (int dh = 0; dh < F; ++dh) oa[dh][i] *= li;
  }
  rht_inverse_output<F>(oa, d1, ll, 1.f / (float)HEAD_SIZE);

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    int aqp = qts + wqo + 2 * i + lh;
    if (aqp >= ql) continue;
    T* or_ = out + (int64_t)(qst + aqp) * so0 + (int64_t)hi * so1;
  #pragma unroll
    for (int dh = 0; dh < F; ++dh) or_[dh * 16 + ll] = to_T<T>(oa[dh][i]);
  }
}

// ---- Launcher -------------------------------------------------------------

template <typename T, int HS>
void launch_int4(T* out, const T* q, const uint8_t* kc, const uint8_t* vc,
                 const float* ksc, const float* vsc, const float* rht,
                 const int* bt, const int* cq, const int* sl, int ns, int nqh,
                 int nkh, int bs, int mbps, int mql, float sm, bool ca,
                 int64_t sq0, int64_t sq1, int64_t skb, int64_t sks,
                 int64_t skh, int64_t svb, int64_t svs, int64_t svh,
                 int64_t ssb, int64_t sss, int64_t ssh, int64_t svvb,
                 int64_t svvs, int64_t svvh, int64_t so0, int64_t so1,
                 cudaStream_t st) {
  constexpr int NW = HS / 32, BM = NW * M_PER_WAVE;
  dim3 g(ns, nqh, (mql + BM - 1) / BM);
  paged_prefill_attn_kernel_v2_int4<T, HS><<<g, dim3(HS), 0, st>>>(
      out, q, kc, vc, ksc, vsc, rht, bt, cq, sl, nqh, nkh, bs, mbps, sm, ca,
      sq0, sq1, skb, sks, skh, svb, svs, svh, ssb, sss, ssh, svvb, svvs, svvh,
      so0, so1);
}

  #define INST(T, HS)                                                        \
    template void launch_int4<T, HS>(                                        \
        T*, const T*, const uint8_t*, const uint8_t*, const float*,          \
        const float*, const float*, const int*, const int*, const int*, int, \
        int, int, int, int, int, float, bool, int64_t, int64_t, int64_t,     \
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,       \
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t);
INST(half, 128) INST(bf16_t, 128) INST(half, 256) INST(bf16_t, 256)
  #undef INST

#endif

}  // namespace prefill_attn_rdna3_v2_int4
}  // namespace vllm

// ---- Torch entry ----------------------------------------------------------

#if defined(USE_ROCM)
void paged_prefill_attn_rdna3_int4(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_cache,
    torch::Tensor v_cache, torch::Tensor k_scale_cache,
    torch::Tensor v_scale_cache, torch::Tensor rht_signs,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, int64_t max_query_len, double sm_scale,
    bool causal) {
  using namespace vllm::prefill_attn_rdna3_v2_int4;
  int ns = seq_lens.size(0), nqh = q.size(1), nkh = k_scale_cache.size(2);
  int bs = k_cache.size(1), mbps = block_table.size(1);
  auto st = at::cuda::getCurrentCUDAStream().stream();
  #define L(T, HS)                                                            \
    launch_int4<T, HS>(                                                       \
        (T*)out.data_ptr(), (const T*)q.data_ptr(),                           \
        (const uint8_t*)k_cache.data_ptr(),                                   \
        (const uint8_t*)v_cache.data_ptr(),                                   \
        (const float*)k_scale_cache.data_ptr(),                               \
        (const float*)v_scale_cache.data_ptr(),                               \
        (const float*)rht_signs.data_ptr(),                                   \
        (const int*)block_table.data_ptr(),                                   \
        (const int*)cu_seqlens_q.data_ptr(), (const int*)seq_lens.data_ptr(), \
        ns, nqh, nkh, bs, mbps, (int)max_query_len, (float)sm_scale, causal,  \
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1),       \
        k_cache.stride(2), v_cache.stride(0), v_cache.stride(1),              \
        v_cache.stride(2), k_scale_cache.stride(0), k_scale_cache.stride(1),  \
        k_scale_cache.stride(2), v_scale_cache.stride(0),                     \
        v_scale_cache.stride(1), v_scale_cache.stride(2), out.stride(0),      \
        out.stride(1), st)
  int hs = q.size(2);
  if (q.dtype() == at::kHalf) {
    if (hs == 128) {
      L(half, 128);
    } else if (hs == 256) {
      L(half, 256);
    } else {
      TORCH_CHECK(false, "INT4 prefill: unsupported head_size ", hs);
    }
  } else {
    if (hs == 128) {
      L(bf16_t, 128);
    } else if (hs == 256) {
      L(bf16_t, 256);
    } else {
      TORCH_CHECK(false, "INT4 prefill: unsupported head_size ", hs);
    }
  }
  #undef L
}
#else
void paged_prefill_attn_rdna3_int4(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_cache,
    torch::Tensor v_cache, torch::Tensor k_scale_cache,
    torch::Tensor v_scale_cache, torch::Tensor rht_signs,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, int64_t max_query_len, double sm_scale,
    bool causal) {
  TORCH_CHECK(false, "requires ROCm");
}
#endif
