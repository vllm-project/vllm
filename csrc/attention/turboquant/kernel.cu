// SPDX-License-Identifier: Apache-2.0
#include "kernel.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

__device__ __forceinline__ float shfl_xor_f(float x, int mask) {
  return __shfl_xor_sync(0xffffffffu, x, mask);
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int d = 16; d; d >>= 1) x += shfl_xor_f(x, d);
  return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int d = 16; d; d >>= 1) x = fmaxf(x, shfl_xor_f(x, d));
  return x;
}

struct DynamicSplitPlan {
  int splits;
  int pps;
};

__device__ __forceinline__ DynamicSplitPlan
make_dynamic_split_plan(int chunks, int max_splits, int one_wave) {
  if (chunks <= 0) return {0, 1};
  int splits = max_splits;
  const int two_chunk_cap = chunks / 2 > 0 ? chunks / 2 : 1;
  if (splits > two_chunk_cap) splits = two_chunk_cap;
  const int four_chunk_cap = chunks / 4 > 0 ? chunks / 4 : 1;
  const int warp_full_cap =
      one_wave > four_chunk_cap ? one_wave : four_chunk_cap;
  if (splits > warp_full_cap) splits = warp_full_cap;
  const int pps = (chunks + splits - 1) / splits;
  splits = (chunks + pps - 1) / pps;
  return {splits, pps};
}

// Unnormalized Walsh-Hadamard transform of width D across one warp; each lane
// holds D/32 contiguous elements. Butterfly stages over distinct index bits
// commute, so intra-lane stages (low bits) followed by shuffle stages (lane
// bits) reproduce the Sylvester Hadamard product exactly for D in {64,128,256}.
template <int D>
__device__ __forceinline__ void fwht(float (&x)[D / 32], int lane) {
  constexpr int V = D / 32;
#pragma unroll
  for (int s = 1; s < V; s <<= 1) {
#pragma unroll
    for (int j = 0; j < V; ++j) {
      if ((j & s) == 0) {
        const float a = x[j] + x[j + s];
        const float b = x[j] - x[j + s];
        x[j] = a;
        x[j + s] = b;
      }
    }
  }
#pragma unroll
  for (int bit = 1; bit < 32; bit <<= 1) {
#pragma unroll
    for (int j = 0; j < V; ++j) {
      const float y = shfl_xor_f(x[j], bit);
      x[j] = (lane & bit) ? (y - x[j]) : (x[j] + y);
    }
  }
}

// ---------------------------------------------------------------------------
// Decode: one CTA per (split, kvh*npacks, b). Each warp owns a logical
// 32-token chunk, independent of the physical cache page template.
// The q rotation (FWHT folding the normalized Hadamard and the D^-0.5 scale)
// is fused: each CTA rotates its own head pack straight into shared memory.
// ---------------------------------------------------------------------------
template <int D, int PAGE_SIZE, int NP4, int ACTIVE_HEADS = NP4 * 4>
__global__ void __launch_bounds__(128)
    decode_kernel(const __nv_bfloat16* __restrict__ q,
                  const uint8_t* __restrict__ kv_cache,
                  const int* __restrict__ page_table,
                  const int* __restrict__ seq_lens,
                  const float* __restrict__ centroids,
                  __nv_bfloat16* __restrict__ out, float* __restrict__ part_acc,
                  float2* __restrict__ part_ml, const int HQ, const int HKV,
                  const int page_table_stride, const int npacks,
                  const int max_splits, const int one_wave) {
  constexpr int PP = NP4 * 4;
  constexpr int CODE_BYTES = D / 2;  // K4/V4 code bytes per token
  constexpr int NV4 = D / 32;        // 16B K-code loads per token
  constexpr int NWRD = D / 8;        // 4B K-code words per token
  constexpr int VD = D / 32;         // V dims owned per lane
  constexpr int VB = D / 64;         // V code bytes owned per lane
  const int split = blockIdx.x;
  const int kvh = blockIdx.y / npacks;
  const int pack = blockIdx.y - kvh * npacks;
  const int b = blockIdx.z;
  const int G = HQ / HKV;
  const int seq = seq_lens[b];
  const int chunks_b = (seq + 31) >> 5;
  const DynamicSplitPlan dynamic_plan =
      make_dynamic_split_plan(chunks_b, max_splits, one_wave);
  if (split >= dynamic_plan.splits) return;
  const int chunk_lo = split * dynamic_plan.pps;
  const int chunk_hi = min(chunk_lo + dynamic_plan.pps, chunks_b);

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nw = blockDim.x >> 5;
  const int h0 = kvh * G + pack * 8;
  const int nvalid = min(ACTIVE_HEADS, G - pack * 8);

  __shared__ float lut[16][32];
  __shared__ float4 qs4[D][NP4];
  __shared__ float4 wsm4[4][32][NP4];
  __shared__ __align__(16) float comb_v[4][PP][D];
  __shared__ float2 comb_ml[4][PP];
  __shared__ float comb_w[PP][4];
  __shared__ float2 comb_total_ml[PP];

  for (int i = threadIdx.x; i < 512; i += blockDim.x)
    lut[i >> 5][i & 31] = centroids[i >> 5];
  if (nvalid < ACTIVE_HEADS) {
    for (int i = threadIdx.x; i < ACTIVE_HEADS * D; i += blockDim.x) {
      const int p = i / D;
      const int d = i & (D - 1);
      if (p >= nvalid) reinterpret_cast<float*>(&qs4[d][0])[p] = 0.0f;
    }
  }
  for (int h = warp; h < nvalid; h += nw) {
    float x[VD];
    const __nv_bfloat16* src = q + ((size_t)b * HQ + h0 + h) * D + lane * VD;
#pragma unroll
    for (int j = 0; j < VD; ++j) x[j] = __bfloat162float(src[j]);
    fwht<D>(x, lane);
#pragma unroll
    for (int j = 0; j < VD; ++j)
      reinterpret_cast<float*>(&qs4[lane * VD + j][0])[h] = x[j] * (1.f / D);
  }
  __syncthreads();

  float m[PP], ll[PP], zl[PP], A[PP][VD];
#pragma unroll
  for (int p = 0; p < ACTIVE_HEADS; ++p) {
    m[p] = -CUDART_INF_F;
    ll[p] = 0.0f;
    zl[p] = 0.0f;
#pragma unroll
    for (int j = 0; j < VD; ++j) A[p][j] = 0.0f;
  }

  for (int chunk = chunk_lo + warp; chunk < chunk_hi; chunk += nw) {
    const int token0 = chunk * 32;
    const int pg0 = token0 / PAGE_SIZE;
    const int row0 = token0 - pg0 * PAGE_SIZE;
    const int blk0 = page_table[b * page_table_stride + pg0];
    const size_t rec0 = (size_t)blk0 * HKV + kvh;
    size_t rec1 = rec0;
    if constexpr (PAGE_SIZE == 16) {
      if (token0 + 16 < seq) {
        const int blk1 = page_table[b * page_table_stride + pg0 + 1];
        rec1 = (size_t)blk1 * HKV + kvh;
      }
    }
    const size_t rec = (PAGE_SIZE == 16 && lane >= 16) ? rec1 : rec0;
    const int row = (PAGE_SIZE == 16) ? (lane & 15) : (row0 + lane);
    const uint8_t* record = kv_cache + rec * (PAGE_SIZE * (D + 6));
    const uint4* kp =
        reinterpret_cast<const uint4*>(record + (size_t)row * CODE_BYTES);
    uint4 kv4[NV4];
#pragma unroll
    for (int i = 0; i < NV4; ++i) kv4[i] = kp[i];
    const int ntok = min(32, seq - token0);
    const float kn = __half2float(
        *reinterpret_cast<const __half*>(record + PAGE_SIZE * D + row * 2));
    const float vsc = __half2float(*reinterpret_cast<const __half*>(
        record + PAGE_SIZE * (D + 2) + row * 2));
    const float vzr = __half2float(*reinterpret_cast<const __half*>(
        record + PAGE_SIZE * (D + 4) + row * 2));

    // Two independent accumulator sets break the D/2-deep serial FMA
    // dependency chains (n2 and per-head dots) in half.
    float4 dot[NP4], dotb[NP4];
#pragma unroll
    for (int g = 0; g < NP4; ++g) {
      dot[g] = make_float4(0.f, 0.f, 0.f, 0.f);
      dotb[g] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float n2 = 1e-16f, n2b = 0.f;

    unsigned w32[NWRD];
#pragma unroll
    for (int i = 0; i < NV4; ++i) {
      w32[4 * i + 0] = kv4[i].x;
      w32[4 * i + 1] = kv4[i].y;
      w32[4 * i + 2] = kv4[i].z;
      w32[4 * i + 3] = kv4[i].w;
    }
#pragma unroll
    for (int wi = 0; wi < NWRD; wi += 2) {
      const unsigned wva = w32[wi];
      const unsigned wvb = w32[wi + 1];
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const float ca = lut[(wva >> (4 * n)) & 0xFu][lane];
        n2 = fmaf(ca, ca, n2);
#pragma unroll
        for (int g = 0; g < NP4; ++g) {
          const float4 qv = qs4[wi * 8 + n][g];
          if (4 * g + 0 < ACTIVE_HEADS) dot[g].x = fmaf(ca, qv.x, dot[g].x);
          if (4 * g + 1 < ACTIVE_HEADS) dot[g].y = fmaf(ca, qv.y, dot[g].y);
          if (4 * g + 2 < ACTIVE_HEADS) dot[g].z = fmaf(ca, qv.z, dot[g].z);
          if (4 * g + 3 < ACTIVE_HEADS) dot[g].w = fmaf(ca, qv.w, dot[g].w);
        }
      }
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const float cb = lut[(wvb >> (4 * n)) & 0xFu][lane];
        n2b = fmaf(cb, cb, n2b);
#pragma unroll
        for (int g = 0; g < NP4; ++g) {
          const float4 qv = qs4[(wi + 1) * 8 + n][g];
          if (4 * g + 0 < ACTIVE_HEADS) dotb[g].x = fmaf(cb, qv.x, dotb[g].x);
          if (4 * g + 1 < ACTIVE_HEADS) dotb[g].y = fmaf(cb, qv.y, dotb[g].y);
          if (4 * g + 2 < ACTIVE_HEADS) dotb[g].z = fmaf(cb, qv.z, dotb[g].z);
          if (4 * g + 3 < ACTIVE_HEADS) dotb[g].w = fmaf(cb, qv.w, dotb[g].w);
        }
      }
    }
    n2 += n2b;
#pragma unroll
    for (int g = 0; g < NP4; ++g) {
      dot[g].x += dotb[g].x;
      dot[g].y += dotb[g].y;
      dot[g].z += dotb[g].z;
      dot[g].w += dotb[g].w;
    }
    const float mult = kn * rsqrtf(n2);
    const bool tval = lane < ntok;
    float sc[PP];
#pragma unroll
    for (int g = 0; g < NP4; ++g) {
      sc[4 * g + 0] = tval ? dot[g].x * mult : -CUDART_INF_F;
      sc[4 * g + 1] = tval ? dot[g].y * mult : -CUDART_INF_F;
      sc[4 * g + 2] = tval ? dot[g].z * mult : -CUDART_INF_F;
      sc[4 * g + 3] = tval ? dot[g].w * mult : -CUDART_INF_F;
    }
#pragma unroll
    for (int p = 0; p < ACTIVE_HEADS; ++p) {
      float pm = sc[p];
#pragma unroll
      for (int off = 16; off; off >>= 1)
        pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, off));
      const float mn = fmaxf(m[p], pm);
      const float alpha = __expf(m[p] - mn);
      const float w = __expf(sc[p] - mn);
      ll[p] = ll[p] * alpha + w;
      zl[p] = zl[p] * alpha + w * vzr;
      reinterpret_cast<float*>(&wsm4[warp][lane][0])[p] = w * vsc;
#pragma unroll
      for (int j = 0; j < VD; ++j) A[p][j] *= alpha;
      m[p] = mn;
    }
    __syncwarp();
    // V phase: lane owns dims VD*lane .. VD*lane+VD-1 (bytes VB*lane ..)
#pragma unroll 4
    for (int t = 0; t < 32; ++t) {
      const size_t vrec = (PAGE_SIZE == 16 && t >= 16) ? rec1 : rec0;
      const int vrow = (PAGE_SIZE == 16) ? (t & 15) : (row0 + t);
      const uint8_t* vrecord = kv_cache + vrec * (PAGE_SIZE * (D + 6));
      const uint8_t* vb = vrecord + PAGE_SIZE * CODE_BYTES +
                          (size_t)vrow * CODE_BYTES + (size_t)lane * VB;
      unsigned v;
      if constexpr (VB == 1) {
        v = *vb;
      } else if constexpr (VB == 2) {
        v = *reinterpret_cast<const unsigned short*>(vb);
      } else {
        v = *reinterpret_cast<const unsigned*>(vb);
      }
      float f[VD];
#pragma unroll
      for (int j = 0; j < VD; ++j)
        f[j] =
            __int_as_float(0x4B000000u | ((v >> (4 * j)) & 0xFu)) - 8388608.f;
#pragma unroll
      for (int g = 0; g < NP4; ++g) {
        const float4 wt4 = wsm4[warp][t][g];
        const float wt[4] = {wt4.x, wt4.y, wt4.z, wt4.w};
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          if (4 * g + u < ACTIVE_HEADS) {
#pragma unroll
            for (int j = 0; j < VD; ++j)
              A[4 * g + u][j] = fmaf(wt[u], f[j], A[4 * g + u][j]);
          }
        }
      }
    }
    __syncwarp();
  }

  // fold per-lane l/z partials, publish per-warp partials to smem
#pragma unroll
  for (int p = 0; p < ACTIVE_HEADS; ++p) {
    float s0 = ll[p];
    float s1 = zl[p];
#pragma unroll
    for (int off = 16; off; off >>= 1) {
      s0 += __shfl_xor_sync(0xffffffffu, s0, off);
      s1 += __shfl_xor_sync(0xffffffffu, s1, off);
    }
    if constexpr (VD == 2) {
      reinterpret_cast<float2*>(&comb_v[warp][p][0])[lane] =
          make_float2(A[p][0] + s1, A[p][1] + s1);
    } else if constexpr (VD == 4) {
      reinterpret_cast<float4*>(&comb_v[warp][p][0])[lane] =
          make_float4(A[p][0] + s1, A[p][1] + s1, A[p][2] + s1, A[p][3] + s1);
    } else {
      float4* dst = reinterpret_cast<float4*>(&comb_v[warp][p][0]) + 2 * lane;
      dst[0] =
          make_float4(A[p][0] + s1, A[p][1] + s1, A[p][2] + s1, A[p][3] + s1);
      dst[1] =
          make_float4(A[p][4] + s1, A[p][5] + s1, A[p][6] + s1, A[p][7] + s1);
    }
    if (lane == 0) comb_ml[warp][p] = make_float2(m[p], s0);
  }
  __syncthreads();

  // D=256 otherwise repeats the same cross-warp exponentials for every
  // output dimension.  Compute the softmax merge weights once per head.
  if constexpr (D == 256) {
    if (threadIdx.x < nvalid) {
      const int p = threadIdx.x;
      float M = -CUDART_INF_F;
      for (int w = 0; w < nw; ++w) M = fmaxf(M, comb_ml[w][p].x);
      float L = 0.f;
      for (int w = 0; w < nw; ++w) {
        const float e = __expf(comb_ml[w][p].x - M);
        comb_w[p][w] = e;
        L = fmaf(e, comb_ml[w][p].y, L);
      }
      comb_total_ml[p] = make_float2(M, L);
    }
    __syncthreads();
    for (int p = 0; p < nvalid; ++p) {
      const float2 total = comb_total_ml[p];
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float O = 0.f;
        for (int w = 0; w < nw; ++w) O = fmaf(comb_w[p][w], comb_v[w][p][d], O);
        if (max_splits == 1) {
          out[((size_t)b * HQ + h0 + p) * D + d] =
              __float2bfloat16(O / total.y);
        } else {
          const size_t idx = ((size_t)b * HQ + h0 + p) * max_splits + split;
          part_acc[idx * D + d] = O;
          if (d == 0) part_ml[idx] = total;
        }
      }
    }
  } else {
    for (int p = 0; p < nvalid; ++p) {
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float M = -CUDART_INF_F;
        for (int w = 0; w < nw; ++w) M = fmaxf(M, comb_ml[w][p].x);
        float L = 0.f;
        float O = 0.f;
        for (int w = 0; w < nw; ++w) {
          const float2 mlw = comb_ml[w][p];
          const float e = __expf(mlw.x - M);
          L = fmaf(e, mlw.y, L);
          O = fmaf(e, comb_v[w][p][d], O);
        }
        if (max_splits == 1) {
          out[((size_t)b * HQ + h0 + p) * D + d] = __float2bfloat16(O / L);
        } else {
          const size_t idx = ((size_t)b * HQ + h0 + p) * max_splits + split;
          part_acc[idx * D + d] = O;
          if (d == 0) part_ml[idx] = make_float2(M, L);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Merge splits. grid B*HQ, block D (one thread per dim).
// ---------------------------------------------------------------------------
template <int D>
__global__ void merge_kernel(const float* __restrict__ part_acc,
                             const float2* __restrict__ part_ml,
                             const int* __restrict__ seq_lens,
                             __nv_bfloat16* __restrict__ out, const int HQ,
                             const int max_splits, const int one_wave) {
  const int bh = blockIdx.x;
  const int b = bh / HQ;
  const int seq = seq_lens[b];
  const int chunks = (seq + 31) >> 5;
  const DynamicSplitPlan dynamic_plan =
      make_dynamic_split_plan(chunks, max_splits, one_wave);
  const int nv = dynamic_plan.splits;
  const int d = threadIdx.x;
  const float* acc = part_acc + (size_t)bh * max_splits * D;
  const float2* ml = part_ml + (size_t)bh * max_splits;
  __shared__ float weights[256];
  __shared__ float inv_l;
  if (d < 32) {
    float M = -CUDART_INF_F;
    for (int s = d; s < nv; s += 32) M = fmaxf(M, ml[s].x);
    M = warp_max(M);
    float L = 0.f;
    for (int s = d; s < nv; s += 32) {
      const float e = __expf(ml[s].x - M);
      weights[s] = e;
      L = fmaf(e, ml[s].y, L);
    }
    L = warp_sum(L);
    if (d == 0) inv_l = 1.f / L;
  }
  __syncthreads();
  float O = 0.f;
  for (int s = 0; s < nv; ++s) O = fmaf(weights[s], acc[s * D + d], O);
  out[(size_t)bh * D + d] = __float2bfloat16(O * inv_l);
}

// ---------------------------------------------------------------------------
template <int D>
static void launch_simt(const void* q, const void* kv_cache,
                        const void* page_table, const void* seq_lens,
                        const void* rotation, const void* centroids, void* out,
                        float* workspace, int B, int HQ, int HKV,
                        int page_table_stride, int page_size,
                        const DecodePlan& pl, cudaStream_t stream) {
  (void)rotation;
  float* part_acc = nullptr;
  float2* part_ml = nullptr;
  if (pl.splits > 1) {
    part_acc = workspace;
    part_ml =
        reinterpret_cast<float2*>(part_acc + (size_t)B * HQ * pl.splits * D);
  }

  dim3 dgrid(pl.splits, HKV * pl.npacks, B);
  const int threads = 32 * pl.nwarps;
#define TQ_SIMT_LAUNCH(PS, HP, AH)                                       \
  decode_kernel<D, PS, HP, AH><<<dgrid, threads, 0, stream>>>(           \
      reinterpret_cast<const __nv_bfloat16*>(q),                         \
      reinterpret_cast<const uint8_t*>(kv_cache),                        \
      reinterpret_cast<const int*>(page_table),                          \
      reinterpret_cast<const int*>(seq_lens),                            \
      reinterpret_cast<const float*>(centroids),                         \
      reinterpret_cast<__nv_bfloat16*>(out), part_acc, part_ml, HQ, HKV, \
      page_table_stride, pl.npacks, pl.splits, pl.one_wave)

  if (pl.np4 == 1) {
    if (pl.G <= 2) {
      switch (page_size) {
        case 16:
          TQ_SIMT_LAUNCH(16, 1, 2);
          break;
        case 32:
          TQ_SIMT_LAUNCH(32, 1, 2);
          break;
        case 64:
          TQ_SIMT_LAUNCH(64, 1, 2);
          break;
        default:
          TQ_SIMT_LAUNCH(128, 1, 2);
          break;
      }
    } else {
      switch (page_size) {
        case 16:
          TQ_SIMT_LAUNCH(16, 1, 4);
          break;
        case 32:
          TQ_SIMT_LAUNCH(32, 1, 4);
          break;
        case 64:
          TQ_SIMT_LAUNCH(64, 1, 4);
          break;
        default:
          TQ_SIMT_LAUNCH(128, 1, 4);
          break;
      }
    }
  } else {
    switch (page_size) {
      case 16:
        TQ_SIMT_LAUNCH(16, 2, 8);
        break;
      case 32:
        TQ_SIMT_LAUNCH(32, 2, 8);
        break;
      case 64:
        TQ_SIMT_LAUNCH(64, 2, 8);
        break;
      default:
        TQ_SIMT_LAUNCH(128, 2, 8);
        break;
    }
  }
#undef TQ_SIMT_LAUNCH

  if (pl.splits > 1) {
    merge_kernel<D><<<B * HQ, D, 0, stream>>>(
        part_acc, part_ml, reinterpret_cast<const int*>(seq_lens),
        reinterpret_cast<__nv_bfloat16*>(out), HQ, pl.splits, pl.one_wave);
  }
}

void turboquant_decode_launch(const void* q, const void* kv_cache,
                              const void* page_table, const void* seq_lens,
                              const void* rotation, const void* centroids,
                              void* out, float* workspace, int B, int HQ,
                              int HKV, int page_table_stride, int page_size,
                              int head_dim, const DecodePlan& pl,
                              cudaStream_t stream) {
  switch (head_dim) {
    case 64:
      launch_simt<64>(q, kv_cache, page_table, seq_lens, rotation, centroids,
                      out, workspace, B, HQ, HKV, page_table_stride, page_size,
                      pl, stream);
      break;
    case 128:
      launch_simt<128>(q, kv_cache, page_table, seq_lens, rotation, centroids,
                       out, workspace, B, HQ, HKV, page_table_stride, page_size,
                       pl, stream);
      break;
    default:
      launch_simt<256>(q, kv_cache, page_table, seq_lens, rotation, centroids,
                       out, workspace, B, HQ, HKV, page_table_stride, page_size,
                       pl, stream);
      break;
  }
}
