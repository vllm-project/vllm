// SPDX-License-Identifier: Apache-2.0
// TurboQuant K4V4-NC paged GQA decode, q_len=1, D in {64,128,256}, H100.
//
// Design:
//  - decode kernel: block = (batch, kv_head, q_pack, split).
//      * fused one-warp FWHT rotates the CTA's own q pack into smem fp16
//        (folds the normalized Hadamard and D^-0.5; no separate rot kernel).
//      * cp.async double-buffered loads of contiguous per-(page,head)
//        PAGE_SIZE*(D/2)-byte code planes + 64B metadata rows.
//      * K is dequantized register-direct into m16n8k16 B fragments inside
//        the QK loop (shuffle LUT of the 16 centroids); no decoded-K smem
//        plane exists, cutting ~2*D*64 bytes of shared memory per block.
//      * V bytes -> half2 nibble dequant to padded smem (row stride 2D+16
//        bytes -> conflict-free ldmatrix).
//      * K row scale norm/centroid_norm folded into the score post-mma
//        (per-token scalar), log2(e) folded in for exp2f softmax.
//      * m16n8k16 tensor-core QK^T (warp-redundant, D/16 k-steps) and PV
//        (D split across 4 warps -> D/4 dims each), FA2-style online softmax,
//        C->A fragment reuse.
//  - merge kernel: log-sum-exp combine of split partials.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr float LOG2E = 1.4426950408889634f;

// Compile-time shared-memory layout for a head dimension D.
template <int D>
struct TQLayout {
  static constexpr int KV_STRIDE = 2 * D + 16;  // decoded row stride (pad 16B)
  static constexpr int CODE_STRIDE = D / 2 + 16;  // code row stride (pad 16B)
  static constexpr int P_STRIDE = 144;            // P staging row (128B + 16B)
  static constexpr int OFF_Q = 0;
  static constexpr int OFF_V = OFF_Q + 16 * KV_STRIDE;
  static constexpr int OFF_CK = OFF_V + 64 * KV_STRIDE;
  static constexpr int OFF_CV = OFF_CK + 2 * 64 * CODE_STRIDE;
  static constexpr int OFF_NORM = OFF_CV + 2 * 64 * CODE_STRIDE;
  static constexpr int OFF_SCALE = OFF_NORM + 2 * 64 * 2;
  static constexpr int OFF_ZERO = OFF_SCALE + 2 * 64 * 2;
  static constexpr int OFF_P = OFF_ZERO + 2 * 64 * 2;    // fp16 P [16][64]
  static constexpr int OFF_RED = OFF_P + 16 * P_STRIDE;  // f32 [2][4w][16 rows]
  static constexpr int OFF_R = OFF_RED + 512;
  static constexpr int SMEM_BYTES = OFF_R + 64 * 4;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst),
               "l"(src));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldmx4(uint32_t a, uint32_t& r0, uint32_t& r1,
                                      uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ void ldmx4t(uint32_t a, uint32_t& r0, uint32_t& r1,
                                       uint32_t& r2, uint32_t& r3) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(a));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float x, float y) {
  __half2 h = __floats2half2_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&h);
}

template <int D>
__device__ __forceinline__ void tq_fwht(float (&x)[D / 32], int lane) {
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
      const float y = __shfl_xor_sync(0xffffffffu, x[j], bit);
      x[j] = (lane & bit) ? (y - x[j]) : (x[j] + y);
    }
  }
}

template <int D, int PAGE_SIZE>
__global__ void __launch_bounds__(128, (D >= 256) ? 2 : 3) tq_decode_kernel(
    const __nv_bfloat16* __restrict__ qg, const uint8_t* __restrict__ kv_cache,
    const int* __restrict__ page_table, const int* __restrict__ seq_lens,
    const float* __restrict__ centroids, __nv_bfloat16* __restrict__ out,
    float* __restrict__ part_o, float* __restrict__ part_ml, int B, int HQ,
    int HKV, int G, int NPACKS, int SPLITS, int page_table_stride) {
  using L = TQLayout<D>;
  constexpr int CODE_BYTES = D / 2;  // K4/V4 code bytes per token
  constexpr int CCH = D / 32;        // 16B cp.async chunks per code row
  constexpr int QCH = D / 8;         // 16B chunks per fp16 q row
  constexpr int KSTEPS = D / 16;     // QK^T k iterations
  constexpr int KW = D / 16;         // 4B code words per half-row decoder
  constexpr int NJ2 = D / 64;        // 16-dim PV column groups per warp
  constexpr int NO = D / 32;         // n8 output blocks per warp
  extern __shared__ uint8_t sm[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int bx = blockIdx.x;
  const int pack = bx % NPACKS;
  const int kvh = (bx / NPACKS) % HKV;
  const int b = bx / (NPACKS * HKV);
  const int split = blockIdx.y;

  const int seq = seq_lens[b];
  const int ntiles_total = (seq + 63) >> 6;
  const int tiles_per_split = (ntiles_total + SPLITS - 1) / SPLITS;
  const int tile0 = split * tiles_per_split;
  const int tile1 = min(ntiles_total, tile0 + tiles_per_split);

  const int hp = min(16, G - pack * 16);
  const int head0 = kvh * G + pack * 16;

  if (tile0 >= tile1) {  // empty split: write neutral partials
    if (part_o != nullptr) {
      size_t obase = ((size_t)(split * B + b) * HQ + head0) * D;
      for (int i = tid; i < hp * D; i += 128) part_o[obase + i] = 0.f;
      size_t mlb = ((size_t)(split * B + b) * HQ + head0) * 2;
      for (int i = tid; i < hp; i += 128) {
        part_ml[mlb + 2 * i] = -1e30f;
        part_ml[mlb + 2 * i + 1] = 0.f;
      }
    }
    return;
  }

  const int ntiles = tile1 - tile0;
  float* red = reinterpret_cast<float*>(sm + L::OFF_RED);
  float* r_sm = reinterpret_cast<float*>(sm + L::OFF_R);
  // lanes 0-15 hold the centroid table for shuffle-based dequant
  const float my_c = centroids[lane & 15];
  const __half* nsm = reinterpret_cast<const __half*>(sm + L::OFF_NORM);
  const __half* ssm = reinterpret_cast<const __half*>(sm + L::OFF_SCALE);
  const __half* zsm = reinterpret_cast<const __half*>(sm + L::OFF_ZERO);

  auto issue_tile = [&](int t) {
    if (t < ntiles) {
      const int tile = tile0 + t;
      const int buf = t & 1;
      for (int i = tid; i < 64 * CCH; i += 128) {
        const int dst_row = i / CCH;
        const int ch = i % CCH;
        const int token = min(tile * 64 + dst_row, seq - 1);
        const int pg = token / PAGE_SIZE;
        const int row = token - pg * PAGE_SIZE;
        const int blk = page_table[b * page_table_stride + pg];
        const size_t rec = (size_t)blk * HKV + kvh;
        const uint8_t* record = kv_cache + rec * (PAGE_SIZE * (D + 6));
        const size_t so = (size_t)row * CODE_BYTES + ch * 16;
        const int doff =
            dst_row * L::CODE_STRIDE + ch * 16 + buf * (64 * L::CODE_STRIDE);
        cp_async16(smem_u32(sm + L::OFF_CK + doff), record + so);
        cp_async16(smem_u32(sm + L::OFF_CV + doff),
                   record + PAGE_SIZE * CODE_BYTES + so);
      }
      if (tid < 24) {
        const int f = tid >> 3;
        const int j = tid & 7;
        const int token = min(tile * 64 + j * 8, ((seq - 1) >> 3) << 3);
        const int pg = token / PAGE_SIZE;
        const int row = token - pg * PAGE_SIZE;
        const int blk = page_table[b * page_table_stride + pg];
        const size_t rec = (size_t)blk * HKV + kvh;
        const uint8_t* record = kv_cache + rec * (PAGE_SIZE * (D + 6));
        const __half* srcp =
            reinterpret_cast<const __half*>(record + PAGE_SIZE * (D + 2 * f));
        const int off = ((f == 0)   ? L::OFF_NORM
                         : (f == 1) ? L::OFF_SCALE
                                    : L::OFF_ZERO) +
                        buf * 128 + j * 16;
        cp_async16(smem_u32(sm + off), srcp + row);
      }
    }
    cp_commit();
  };

  issue_tile(0);
  issue_tile(1);

  // Fused q rotation: zero-pad rows >= hp, then FWHT the CTA's own heads
  // (one warp per head, round-robin) straight into the fp16 q plane.
  for (int i = tid; i < 16 * QCH; i += 128) {
    const int row = i / QCH;
    const int ch = i % QCH;
    if (row >= hp)
      *reinterpret_cast<uint4*>(sm + L::OFF_Q + row * L::KV_STRIDE + ch * 16) =
          make_uint4(0u, 0u, 0u, 0u);
  }
  for (int h = warp; h < hp; h += 4) {
    constexpr int V = D / 32;
    float x[V];
    const __nv_bfloat16* src = qg + ((size_t)b * HQ + head0 + h) * D + lane * V;
#pragma unroll
    for (int j = 0; j < V; ++j) x[j] = __bfloat162float(src[j]);
    tq_fwht<D>(x, lane);
    __half* dst =
        reinterpret_cast<__half*>(sm + L::OFF_Q + h * L::KV_STRIDE) + lane * V;
#pragma unroll
    for (int j = 0; j < V; ++j) dst[j] = __float2half(x[j] * (1.f / D));
  }
  __syncthreads();

  uint32_t qa[KSTEPS][4];
  {
    const int r = ((lane >> 3) & 1) * 8 + (lane & 7);
    const uint32_t base =
        smem_u32(sm + L::OFF_Q + r * L::KV_STRIDE + (lane >> 4) * 16);
#pragma unroll
    for (int ks = 0; ks < KSTEPS; ks++)
      ldmx4(base + ks * 32, qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3]);
  }

  float O[NO][4];
#pragma unroll
  for (int j = 0; j < NO; j++)
#pragma unroll
    for (int e = 0; e < 4; e++) O[j][e] = 0.f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;

  const uint32_t vmat_base =
      smem_u32(sm + L::OFF_V) +
      (((lane >> 3) & 1) * 8 + (lane & 7)) * L::KV_STRIDE + warp * (D / 2) +
      (lane >> 4) * 16;
  const uint32_t pmat_base =
      smem_u32(sm + L::OFF_P) +
      (((lane >> 3) & 1) * 8 + (lane & 7)) * L::P_STRIDE + (lane >> 4) * 16;
  const int r_local = lane >> 1;
  const int hh = lane & 1;
  const int drow = warp * 16 + r_local;
  const int row0 = lane >> 2;
  const int row1 = row0 + 8;
  const int nA = lane >> 2;  // fragment n index (token within 8-group)
  const int kb = lane & 3;   // fragment k byte selector

  for (int t = 0; t < ntiles; t++) {
    const int buf = t & 1;
    cp_wait<1>();
    __syncthreads();

    {  // decode V half-row (D/4 code bytes -> D/2 dims)
      const uint8_t* crow =
          sm + L::OFF_CV + buf * (64 * L::CODE_STRIDE) + drow * L::CODE_STRIDE;
      const __half2 s2 = __half2half2(ssm[buf * 64 + drow]);
      const __half2 z2 = __half2half2(zsm[buf * 64 + drow]);
      const __half2 c1024 = __float2half2_rn(1024.f);
#pragma unroll
      for (int i = 0; i < KW; i++) {
        const int cst = hh * KW + ((i + 2 * r_local + 4 * hh) & (KW - 1));
        const uint32_t code =
            *reinterpret_cast<const uint32_t*>(crow + cst * 4);
        uint32_t w[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
          // fp16 bias trick: 0x6400|n == (half)(1024+n), exact for n in [0,15]
          const uint32_t bt = code >> (8 * e);
          uint32_t raw = 0x64006400u | (bt & 0xFu) | ((bt & 0xF0u) << 12);
          __half2 hv = *reinterpret_cast<__half2*>(&raw);
          hv = __hfma2(__hsub2(hv, c1024), s2, z2);
          w[e] = *reinterpret_cast<uint32_t*>(&hv);
        }
        *reinterpret_cast<uint4*>(sm + L::OFF_V + drow * L::KV_STRIDE +
                                  cst * 16) =
            make_uint4(w[0], w[1], w[2], w[3]);
      }
    }
    __syncthreads();

    // QK^T: warp w owns tokens [16w, 16w+16). K is dequantized straight
    // into B fragments from the staged code bytes (shuffle centroid LUT);
    // per-lane csq partials cover dims {16ks+2kb, +1, +8, +9} over all ks.
    float C[2][4];
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) C[j][e] = 0.f;
    float csq0 = 0.f, csq1 = 0.f;
    {
      const uint8_t* ckA = sm + L::OFF_CK + buf * (64 * L::CODE_STRIDE) +
                           (warp * 16 + nA) * L::CODE_STRIDE;
      const uint8_t* ckB = ckA + 8 * L::CODE_STRIDE;
#pragma unroll
      for (int ks = 0; ks < KSTEPS; ks++) {
        const uint2 wA = *reinterpret_cast<const uint2*>(ckA + ks * 8);
        const uint2 wB = *reinterpret_cast<const uint2*>(ckB + ks * 8);
        const uint32_t bl0 = (wA.x >> (8 * kb)) & 0xFFu;
        const uint32_t bh0 = (wA.y >> (8 * kb)) & 0xFFu;
        const uint32_t bl1 = (wB.x >> (8 * kb)) & 0xFFu;
        const uint32_t bh1 = (wB.y >> (8 * kb)) & 0xFFu;
        const float c0 = __shfl_sync(0xffffffffu, my_c, bl0 & 0xFu);
        const float c1 = __shfl_sync(0xffffffffu, my_c, bl0 >> 4);
        const float c2 = __shfl_sync(0xffffffffu, my_c, bh0 & 0xFu);
        const float c3 = __shfl_sync(0xffffffffu, my_c, bh0 >> 4);
        const float c4 = __shfl_sync(0xffffffffu, my_c, bl1 & 0xFu);
        const float c5 = __shfl_sync(0xffffffffu, my_c, bl1 >> 4);
        const float c6 = __shfl_sync(0xffffffffu, my_c, bh1 & 0xFu);
        const float c7 = __shfl_sync(0xffffffffu, my_c, bh1 >> 4);
        csq0 = fmaf(c0, c0, csq0);
        csq0 = fmaf(c1, c1, csq0);
        csq0 = fmaf(c2, c2, csq0);
        csq0 = fmaf(c3, c3, csq0);
        csq1 = fmaf(c4, c4, csq1);
        csq1 = fmaf(c5, c5, csq1);
        csq1 = fmaf(c6, c6, csq1);
        csq1 = fmaf(c7, c7, csq1);
        const uint32_t b0 = pack_h2(c0, c1);
        const uint32_t b1 = pack_h2(c2, c3);
        const uint32_t b2 = pack_h2(c4, c5);
        const uint32_t b3 = pack_h2(c6, c7);
        mma16816(C[0], qa[ks], b0, b1);
        mma16816(C[1], qa[ks], b2, b3);
      }
    }
    csq0 += __shfl_xor_sync(0xffffffffu, csq0, 1);
    csq0 += __shfl_xor_sync(0xffffffffu, csq0, 2);
    csq1 += __shfl_xor_sync(0xffffffffu, csq1, 1);
    csq1 += __shfl_xor_sync(0xffffffffu, csq1, 2);
    if (kb == 0) {
      const float n0 = __half2float(nsm[buf * 64 + warp * 16 + nA]);
      const float n1 = __half2float(nsm[buf * 64 + warp * 16 + 8 + nA]);
      r_sm[warp * 16 + nA] = n0 * rsqrtf(csq0 + 1e-16f) * LOG2E;
      r_sm[warp * 16 + 8 + nA] = n1 * rsqrtf(csq1 + 1e-16f) * LOG2E;
    }
    __syncwarp();

    const int token_base = (tile0 + t) * 64;
    const int vcnt = min(seq, token_base + 64) - token_base;
    float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int c0 = warp * 16 + j * 8 + 2 * (lane & 3);
      const float2 rr = *reinterpret_cast<const float2*>(&r_sm[c0]);
      C[j][0] = (c0 < vcnt) ? C[j][0] * rr.x : -1e30f;
      C[j][1] = (c0 + 1 < vcnt) ? C[j][1] * rr.y : -1e30f;
      C[j][2] = (c0 < vcnt) ? C[j][2] * rr.x : -1e30f;
      C[j][3] = (c0 + 1 < vcnt) ? C[j][3] * rr.y : -1e30f;
      mx0 = fmaxf(mx0, fmaxf(C[j][0], C[j][1]));
      mx1 = fmaxf(mx1, fmaxf(C[j][2], C[j][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    if ((lane & 3) == 0) {
      red[warp * 16 + row0] = mx0;
      red[warp * 16 + row1] = mx1;
    }
    __syncthreads();
    issue_tile(t + 2);  // all warps done with CK[buf]: safe to refill
    const float mn0 = fmaxf(m0, fmaxf(fmaxf(red[row0], red[16 + row0]),
                                      fmaxf(red[32 + row0], red[48 + row0])));
    const float mn1 = fmaxf(m1, fmaxf(fmaxf(red[row1], red[16 + row1]),
                                      fmaxf(red[32 + row1], red[48 + row1])));
    const float a0 = exp2f(m0 - mn0);
    const float a1 = exp2f(m1 - mn1);
    m0 = mn0;
    m1 = mn1;
    float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      C[j][0] = exp2f(C[j][0] - mn0);
      C[j][1] = exp2f(C[j][1] - mn0);
      C[j][2] = exp2f(C[j][2] - mn1);
      C[j][3] = exp2f(C[j][3] - mn1);
      rs0 += C[j][0] + C[j][1];
      rs1 += C[j][2] + C[j][3];
    }
    rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
    rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
    rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
    rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
    if ((lane & 3) == 0) {
      red[64 + warp * 16 + row0] = rs0;
      red[64 + warp * 16 + row1] = rs1;
    }
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int col = warp * 16 + j * 8 + 2 * (lane & 3);
      *reinterpret_cast<uint32_t*>(sm + L::OFF_P + row0 * L::P_STRIDE +
                                   col * 2) = pack_h2(C[j][0], C[j][1]);
      *reinterpret_cast<uint32_t*>(sm + L::OFF_P + row1 * L::P_STRIDE +
                                   col * 2) = pack_h2(C[j][2], C[j][3]);
    }
    __syncthreads();
    l0 = l0 * a0 + ((red[64 + row0] + red[80 + row0]) +
                    (red[96 + row0] + red[112 + row0]));
    l1 = l1 * a1 + ((red[64 + row1] + red[80 + row1]) +
                    (red[96 + row1] + red[112 + row1]));
#pragma unroll
    for (int jn = 0; jn < NO; jn++) {
      O[jn][0] *= a0;
      O[jn][1] *= a0;
      O[jn][2] *= a1;
      O[jn][3] *= a1;
    }
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      uint32_t pa[4];
      ldmx4(pmat_base + ks * 32, pa[0], pa[1], pa[2], pa[3]);
#pragma unroll
      for (int jn2 = 0; jn2 < NJ2; jn2++) {
        uint32_t v0, v1, v2, v3;
        ldmx4t(vmat_base + ks * (16 * L::KV_STRIDE) + jn2 * 32, v0, v1, v2, v3);
        mma16816(O[2 * jn2], pa, v0, v1);
        mma16816(O[2 * jn2 + 1], pa, v2, v3);
      }
    }
  }

  const int cbase = warp * (D / 4) + 2 * (lane & 3);
  if (part_o == nullptr) {
    const float inv0 = 1.f / l0;
    const float inv1 = 1.f / l1;
    if (row0 < hp) {
      __nv_bfloat16* orow = out + ((size_t)b * HQ + head0 + row0) * D;
#pragma unroll
      for (int jn = 0; jn < NO; jn++) {
        __nv_bfloat162 v =
            __floats2bfloat162_rn(O[jn][0] * inv0, O[jn][1] * inv0);
        *reinterpret_cast<uint32_t*>(orow + cbase + 8 * jn) =
            *reinterpret_cast<uint32_t*>(&v);
      }
    }
    if (row1 < hp) {
      __nv_bfloat16* orow = out + ((size_t)b * HQ + head0 + row1) * D;
#pragma unroll
      for (int jn = 0; jn < NO; jn++) {
        __nv_bfloat162 v =
            __floats2bfloat162_rn(O[jn][2] * inv1, O[jn][3] * inv1);
        *reinterpret_cast<uint32_t*>(orow + cbase + 8 * jn) =
            *reinterpret_cast<uint32_t*>(&v);
      }
    }
  } else {
    const size_t base = (size_t)(split * B + b) * HQ + head0;
    if (row0 < hp) {
      float* po = part_o + (base + row0) * D;
#pragma unroll
      for (int jn = 0; jn < NO; jn++)
        *reinterpret_cast<float2*>(po + cbase + 8 * jn) =
            make_float2(O[jn][0], O[jn][1]);
    }
    if (row1 < hp) {
      float* po = part_o + (base + row1) * D;
#pragma unroll
      for (int jn = 0; jn < NO; jn++)
        *reinterpret_cast<float2*>(po + cbase + 8 * jn) =
            make_float2(O[jn][2], O[jn][3]);
    }
    if (warp == 0 && (lane & 3) == 0) {
      if (row0 < hp) {
        part_ml[(base + row0) * 2] = m0;
        part_ml[(base + row0) * 2 + 1] = l0;
      }
      if (row1 < hp) {
        part_ml[(base + row1) * 2] = m1;
        part_ml[(base + row1) * 2 + 1] = l1;
      }
    }
  }
}

template <int D>
__global__ void tq_merge_kernel(const float* __restrict__ part_o,
                                const float* __restrict__ part_ml,
                                __nv_bfloat16* __restrict__ out, int S,
                                int BHQ) {
  const int bh = blockIdx.x;
  const int d = threadIdx.x;
  float M = -1e30f, L = 0.f, acc = 0.f;
  for (int s = 0; s < S; s++) {
    const float m = part_ml[((size_t)s * BHQ + bh) * 2];
    const float l = part_ml[((size_t)s * BHQ + bh) * 2 + 1];
    if (l <= 0.f) continue;
    const float o = part_o[((size_t)s * BHQ + bh) * D + d];
    if (m > M) {
      const float sc = exp2f(M - m);
      acc *= sc;
      L *= sc;
      M = m;
    }
    const float w = exp2f(m - M);
    acc += w * o;
    L += w * l;
  }
  out[(size_t)bh * D + d] = __float2bfloat16(acc / L);
}

template <int D, int PAGE_SIZE>
void launch_decode_page(const void* q, const void* kv_cache, const int* pt,
                        const int* sl, const float* cen, void* out,
                        void* part_o, void* part_ml, int B, int HQ, int HKV,
                        int G, int npacks, int splits, int page_table_stride,
                        cudaStream_t stream) {
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(tq_decode_kernel<D, PAGE_SIZE>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         TQLayout<D>::SMEM_BYTES);
    init = true;
  }
  dim3 grid(B * HKV * npacks, splits);
  tq_decode_kernel<D, PAGE_SIZE>
      <<<grid, 128, TQLayout<D>::SMEM_BYTES, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(q),
          reinterpret_cast<const uint8_t*>(kv_cache), pt, sl, cen,
          reinterpret_cast<__nv_bfloat16*>(out),
          reinterpret_cast<float*>(part_o), reinterpret_cast<float*>(part_ml),
          B, HQ, HKV, G, npacks, splits, page_table_stride);
}

template <int D>
void tq_launch_dim(const void* q, const void* kv_cache, const int* pt,
                   const int* sl, const float* rot, const float* cen, void* out,
                   void* part_o, void* part_ml, int B, int HQ, int HKV,
                   int page_table_stride, int page_size, int splits,
                   cudaStream_t stream) {
  (void)rot;
  const int G = HQ / HKV;
  const int npacks = (G + 15) / 16;

  switch (page_size) {
    case 16:
      launch_decode_page<D, 16>(q, kv_cache, pt, sl, cen, out, part_o, part_ml,
                                B, HQ, HKV, G, npacks, splits,
                                page_table_stride, stream);
      break;
    case 32:
      launch_decode_page<D, 32>(q, kv_cache, pt, sl, cen, out, part_o, part_ml,
                                B, HQ, HKV, G, npacks, splits,
                                page_table_stride, stream);
      break;
    case 64:
      launch_decode_page<D, 64>(q, kv_cache, pt, sl, cen, out, part_o, part_ml,
                                B, HQ, HKV, G, npacks, splits,
                                page_table_stride, stream);
      break;
    default:
      launch_decode_page<D, 128>(q, kv_cache, pt, sl, cen, out, part_o, part_ml,
                                 B, HQ, HKV, G, npacks, splits,
                                 page_table_stride, stream);
      break;
  }

  if (splits > 1)
    tq_merge_kernel<D><<<B * HQ, D, 0, stream>>>(
        reinterpret_cast<const float*>(part_o),
        reinterpret_cast<const float*>(part_ml),
        reinterpret_cast<__nv_bfloat16*>(out), splits, B * HQ);
}

}  // namespace

extern "C" void turboquant_mma_decode_launch(
    const void* q, const void* kv_cache, const int* pt, const int* sl,
    const float* rot, const float* cen, void* out, void* part_o, void* part_ml,
    int B, int HQ, int HKV, int page_table_stride, int page_size, int head_dim,
    int splits, cudaStream_t stream) {
  switch (head_dim) {
    case 64:
      tq_launch_dim<64>(q, kv_cache, pt, sl, rot, cen, out, part_o, part_ml, B,
                        HQ, HKV, page_table_stride, page_size, splits, stream);
      break;
    case 128:
      tq_launch_dim<128>(q, kv_cache, pt, sl, rot, cen, out, part_o, part_ml, B,
                         HQ, HKV, page_table_stride, page_size, splits, stream);
      break;
    default:
      tq_launch_dim<256>(q, kv_cache, pt, sl, rot, cen, out, part_o, part_ml, B,
                         HQ, HKV, page_table_stride, page_size, splits, stream);
      break;
  }
}
