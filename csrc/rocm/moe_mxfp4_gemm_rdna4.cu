// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MXFP4 x FP8 (W4A8) fused-MoE GEMM for RDNA4 (gfx12). Milestone 1: the dense
// RDNA4 kernels (SWMMAC for <= 8 rows, skinny WMMA for 16/32/64 rows) re-used
// per sorted token block, with the expert routing contract of
// moe_mxfp4_gemm_rdna3.cu:
//
//   a_fp8   [rows_a, K]         uint8 E4M3, per-row scale a_scale [rows_a] f32
//   c       [M*top_k, N]        fp16/bf16, row = sorted token slot
//   b_q     [E, K/32, N, 4]     uint32 fragment-major (mxfp4_moe_repack_rdna4)
//   b_scale [E, K/32, N]        uint8 E8M0
//   sorted_token_ids [blocks*block_size_m] i32, slots in [0, M*top_k);
//   entries >= M*top_k are padding. expert_ids[block], num_tokens_post_padded.
//   A row for a slot is slot / top_k (gate_up: top_k = k; down: 1).
//   Epilogue: v *= topk_weights[slot] if mul_topk_weight; v += bias[e][n].
//
// Every block covers the whole K range for its rows x 16 columns (KW waves
// split K and reduce through LDS), so results are stored directly: no output
// zero-fill and no atomics. Padded slots are never written.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdlib>

#include "qdq_mxfp4_rdna4.cuh"

#if defined(__gfx1200__) || defined(__gfx1201__)
  #define __HIP__MXFP4_RDNA4__
#endif

namespace vllm {
namespace moe_mxfp4_rdna4 {

constexpr int kGroup = 32;
using bf16_t = __hip_bfloat16;

template <typename T>
__device__ __forceinline__ T from_f(float v);
template <>
__device__ __forceinline__ half from_f(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ bf16_t from_f(float v) {
  return __float2bfloat16(v);
}
template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f(half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_f(bf16_t v) {
  return __bfloat162float(v);
}

struct MoeArgs {
  const uint8_t* aq;
  const float* ascale;
  const uint32_t* b_q;      // [E][K/32][N][4]
  const uint8_t* b_scales;  // [E][K/32][N]
  const float* topk_w;      // nullptr unless mul_topk_weight
  const int32_t* sorted_token_ids;
  const int32_t* expert_ids;
  const int32_t* num_tokens_post_padded;
  int total_slots;  // M * top_k
  int top_k;
  int N, K;
  size_t w_stride, s_stride;  // per-expert strides (elements)
  const uint8_t* b_base;      // [E][N] per-column E5M2 fold base (hybrid)
};

#if defined(__HIP__MXFP4_RDNA4__) || !defined(__HIP_DEVICE_COMPILE__)
using v8f32 = float __attribute__((ext_vector_type(8)));
using v2i32 = int __attribute__((ext_vector_type(2)));
using v4i32 = int __attribute__((ext_vector_type(4)));
using mxfp4_rdna4::dequant_mxfp4_8_e4m3;
using mxfp4_rdna4::dequant_mxfp4_8_e5m2;
using mxfp4_rdna4::mxfp4_e5m2_tables;
using mxfp4_rdna4::mxfp4_e8m0_to_f32;

__device__ __forceinline__ v8f32 wmma_fp8(v2i32 a, v2i32 b, v8f32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

// ---------------------------------------------------------------------------
// block_size_m == 8: one SWMMAC (16x16x32 fp8, 2:4 sparse A) per E8M0 group,
// eight gathered token rows packed into the 16 sparse rows (see
// mxfp4_swmmac_rdna4.cu for the row-pair mapping and index constants).
// grid = (ceil(N/16), token blocks), block = 32*KW.
// ---------------------------------------------------------------------------
template <typename T, int KW, int PF>
__global__ void __launch_bounds__(32 * KW)
    moe_swmmac_kernel(MoeArgs ar, T* __restrict__ c,
                      const T* __restrict__ bias) {
  const int blk = blockIdx.y;
  if (blk * 8 >= ar.num_tokens_post_padded[0]) return;
  const int e = ar.expert_ids[blk];
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int lo = lane & 15;
  const int hi = lane >> 4;
  const int n_col = blockIdx.x * 16 + lo;
  const bool n_ok = n_col < ar.N;
  const int K = ar.K, N = ar.N;
  const int a_r = lo >> 1;
  const int slot = ar.sorted_token_ids[blk * 8 + a_r];
  // Padding rows read row 0 (valid memory); their results are dropped at the
  // store, and out-of-range columns read column N-1 for the same reason, so
  // every load is unconditional (no execz regions, loads stay in flight).
  const int a_row = slot < ar.total_slots ? slot / ar.top_k : 0;
  const int n_ld = min(n_col, N - 1);
  const bool odd = (lo & 1) != 0;
  const int idx = odd ? (int)0xEEEEEEEEu : (int)0x44444444u;
  // Even lanes take the low halves of each dword pair, odd lanes the high.
  const uint32_t a_sel = odd ? 0x07060302u : 0x05040100u;
  const uint32_t* bq = ar.b_q + (size_t)e * ar.w_stride;
  const uint8_t* bs = ar.b_scales + (size_t)e * ar.s_stride;
  const int ngroups = K / kGroup;
  const int g_begin = (ngroups * wave) / KW;
  const int g_end = (ngroups * (wave + 1)) / KW;

  // Raw loads only: the row-pair perm and the E8M0 conversion happen at
  // consume time so a refill never waits on its own loads.
  auto load = [&](int g, uint4& a, unsigned long long& b, uint32_t& sc) {
    g = min(g, ngroups - 1);  // tail refills re-read the last group
    const int k_g = g * kGroup;
    __builtin_memcpy(&a, ar.aq + (size_t)a_row * K + k_g + 16 * hi, 16);
    const auto* bp = reinterpret_cast<const unsigned long long*>(
        bq + ((size_t)g * N + n_ld) * 4 + 2 * hi);
    b = __builtin_nontemporal_load(bp);
    sc = bs[(size_t)g * N + n_ld];
  };

  v8f32 acc = {0, 0, 0, 0, 0, 0, 0, 0};
  uint4 a_r_[PF];
  unsigned long long b_r[PF];
  uint32_t w_r[PF];
  #pragma unroll
  for (int d = 0; d < PF; d++) load(g_begin + d, a_r_[d], b_r[d], w_r[d]);
  for (int g0 = g_begin; g0 < g_end; g0 += PF) {
  #pragma unroll
    for (int d = 0; d < PF; d++) {
      const int g = g0 + d;
      if (g >= g_end) break;
      uint32_t l0, h0, l1, h1;
      dequant_mxfp4_8_e4m3((uint32_t)b_r[d], l0, h0);
      dequant_mxfp4_8_e4m3((uint32_t)(b_r[d] >> 32), l1, h1);
      v4i32 b;
      b[0] = (int)l0;
      b[1] = (int)h0;
      b[2] = (int)l1;
      b[3] = (int)h1;
      v2i32 a;
      a[0] = (int)__builtin_amdgcn_perm(a_r_[d].y, a_r_[d].x, a_sel);
      a[1] = (int)__builtin_amdgcn_perm(a_r_[d].w, a_r_[d].z, a_sel);
      v8f32 grp = {0, 0, 0, 0, 0, 0, 0, 0};
      grp = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, grp, idx);
      const float ws = mxfp4_e8m0_to_f32(w_r[d]);
  #pragma unroll
      for (int i = 0; i < 8; i++) acc[i] = fmaf(grp[i], ws, acc[i]);
      load(g + PF, a_r_[d], b_r[d], w_r[d]);
    }
  }
  float part[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) part[i] = acc[2 * i] + acc[2 * i + 1];
  if constexpr (KW > 1) {
    __shared__ float red[KW][32][4];
  #pragma unroll
    for (int i = 0; i < 4; i++) red[wave][lane][i] = part[i];
    __syncthreads();
    if (wave != 0) return;
  #pragma unroll
    for (int w = 1; w < KW; w++)
  #pragma unroll
      for (int i = 0; i < 4; i++) part[i] += red[w][lane][i];
  }
  if (!n_ok) return;
  const float bv = bias ? to_f<T>(bias[(size_t)e * N + n_col]) : 0.0f;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    const int r = 4 * hi + i;  // logical row within the block
    const int s = ar.sorted_token_ids[blk * 8 + r];
    if (s >= ar.total_slots) continue;
    const int arow = s / ar.top_k;
    float v = part[i] * ar.ascale[arow];
    if (ar.topk_w) v *= ar.topk_w[s];
    c[(size_t)s * N + n_col] = from_f<T>(v + bv);
  }
}

// ---------------------------------------------------------------------------
// block_size_m in {16, 32, 64}: skinny WMMA, MT = block_size_m / 16 row tiles
// of gathered token rows, B-only prefetch ring, A one group ahead (the
// register-pressure layout from the dense kernel). grid = (ceil(N/16), token
// blocks), block = 32*KW.
// ---------------------------------------------------------------------------
template <typename T, int MT, int KW, int PF>
__global__ void __launch_bounds__(32 * KW)
    moe_wmma_kernel(MoeArgs ar, T* __restrict__ c, const T* __restrict__ bias) {
  const int blk = blockIdx.y;
  if (blk * 16 * MT >= ar.num_tokens_post_padded[0]) return;
  const int e = ar.expert_ids[blk];
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int lo = lane & 15;
  const int hi = lane >> 4;
  const int n_col = blockIdx.x * 16 + lo;
  const bool n_ok = n_col < ar.N;
  const int K = ar.K, N = ar.N;
  const int n_ld = min(n_col, N - 1);
  const uint32_t* bq = ar.b_q + (size_t)e * ar.w_stride;
  const uint8_t* bs = ar.b_scales + (size_t)e * ar.s_stride;
  const int ngroups = K / kGroup;
  const int g_begin = (ngroups * wave) / KW;
  const int g_end = (ngroups * (wave + 1)) / KW;

  // Padding rows read row 0 and out-of-range columns read column N-1; both
  // are discarded at the store, so no load is predicated (see the SWMMAC
  // kernel). The ring holds PF groups of A and B, each slot refilled right
  // after its WMMAs so no load ever lands on a live register.
  int a_row[MT];
  #pragma unroll
  for (int t = 0; t < MT; t++) {
    const int s = ar.sorted_token_ids[blk * 16 * MT + 16 * t + lo];
    a_row[t] = s < ar.total_slots ? s / ar.top_k : 0;
  }

  struct Slot {
    uint4 a[MT];
    unsigned long long b;
    uint32_t sc;
  };
  Slot ring[PF];
  auto load = [&](int g, Slot& sl) {
    g = min(g, ngroups - 1);
    const int k_g = g * kGroup;
  #pragma unroll
    for (int t = 0; t < MT; t++)
      __builtin_memcpy(&sl.a[t], ar.aq + (size_t)a_row[t] * K + k_g + 16 * hi,
                       16);
    const auto* bp = reinterpret_cast<const unsigned long long*>(
        bq + ((size_t)g * N + n_ld) * 4 + 2 * hi);
    sl.b = __builtin_nontemporal_load(bp);
    sl.sc = bs[(size_t)g * N + n_ld];
  };

  v8f32 acc[MT];
  #pragma unroll
  for (int t = 0; t < MT; t++) acc[t] = {0, 0, 0, 0, 0, 0, 0, 0};
  #pragma unroll
  for (int d = 0; d < PF; d++) load(g_begin + d, ring[d]);

  for (int g0 = g_begin; g0 < g_end; g0 += PF) {
  #pragma unroll
    for (int d = 0; d < PF; d++) {
      const int g = g0 + d;
      if (g >= g_end) break;
      const Slot sl = ring[d];
      const float ws = mxfp4_e8m0_to_f32(sl.sc);
      v2i32 bf[2];
  #pragma unroll
      for (int st = 0; st < 2; st++) {
        uint32_t blo, bhi;
        dequant_mxfp4_8_e4m3((uint32_t)(st == 0 ? sl.b : (sl.b >> 32)), blo,
                             bhi);
        bf[st][0] = (int)blo;
        bf[st][1] = (int)bhi;
      }
  #pragma unroll
      for (int t = 0; t < MT; t++) {
        v8f32 grp = {0, 0, 0, 0, 0, 0, 0, 0};
        v2i32 af0, af1;
        af0[0] = (int)sl.a[t].x;
        af0[1] = (int)sl.a[t].y;
        af1[0] = (int)sl.a[t].z;
        af1[1] = (int)sl.a[t].w;
        grp = wmma_fp8(af0, bf[0], grp);
        grp = wmma_fp8(af1, bf[1], grp);
  #pragma unroll
        for (int q = 0; q < 8; q++) acc[t][q] = fmaf(grp[q], ws, acc[t][q]);
      }
      load(g + PF, ring[d]);
    }
  }

  if constexpr (KW > 1) {
    __shared__ float red[KW][MT][32][8];
  #pragma unroll
    for (int t = 0; t < MT; t++)
  #pragma unroll
      for (int i = 0; i < 8; i++) red[wave][t][lane][i] = acc[t][i];
    __syncthreads();
    if (wave != 0) return;
  #pragma unroll
    for (int w = 1; w < KW; w++)
  #pragma unroll
      for (int t = 0; t < MT; t++)
  #pragma unroll
        for (int i = 0; i < 8; i++) acc[t][i] += red[w][t][lane][i];
  }
  if (!n_ok) return;
  const float bv = bias ? to_f<T>(bias[(size_t)e * N + n_col]) : 0.0f;
  // gfx12 accumulator: lane -> n = lo, slot i -> row 8*hi + i of the tile.
  #pragma unroll
  for (int t = 0; t < MT; t++) {
  #pragma unroll
    for (int i = 0; i < 8; i++) {
      const int r = 16 * t + 8 * hi + i;
      const int s = ar.sorted_token_ids[blk * 16 * MT + r];
      if (s >= ar.total_slots) continue;
      const int arow = s / ar.top_k;
      float v = acc[t][i] * ar.ascale[arow];
      if (ar.topk_w) v *= ar.topk_w[s];
      c[(size_t)s * N + n_col] = from_f<T>(v + bv);
    }
  }
}
// ---------------------------------------------------------------------------
// block_size_m == 256: PREFILL "hybrid" kernel (after FlyDSL's RDNA4 hybrid
// strategy). One workgroup = one 256-slot token block x BN columns (BN = 64
// or 128) of its expert; 8 waves, wave w owns rows [32w, 32w+32) (2 row
// tiles) x all BN/16 column tiles. Per K-stage (4 E8M0 groups = 128 K) the
// expert's B fragments are loaded once (one or two 16-byte fragments per
// thread, nontemporal), decoded ONCE to E5M2 with the E8M0 scale folded into
// the perm table (shift = e8m0 - base[n], base = column max - 13 keeps every
// shift in E5M2's normal range; 2^(base-127) is applied in the epilogue),
// written to a ping-pong LDS arena laid out [group][K-half][col][16 B]
// (conflict-free for both the decode stores and the ds_load_b128 fragment
// reads) and shared by the 8 M-waves: 2*BN WMMAs per wave per barrier,
// 8/BN v_perm per WMMA. A rows are gathered through sorted_token_ids straight
// into registers one group ahead. No load is predicated: out-of-range
// columns/rows are clamped to valid addresses and discarded at the store, so
// every load stays in flight (no execz regions).
// grid = (token blocks, ceil(N/BN)), block = 256.
// ---------------------------------------------------------------------------
__device__ __forceinline__ v8f32 wmma_fp8_bf8(v2i32 a, v2i32 b, v8f32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(a, b, c);
}

template <typename T, int BN>
__global__ void __launch_bounds__(256)
    moe_hybrid_kernel(MoeArgs ar, T* __restrict__ c,
                      const T* __restrict__ bias) {
  constexpr int BM = 256, GS = 4,
                CT = BN / 16;           // rows, groups/stage, col tiles
  constexpr int BFT = (BN * GS) / 256;  // B fragments per thread/stage
  constexpr int DG = 256 / BN;          // groups covered per fragment pass
  const int blk = blockIdx.x;
  if (blk * BM >= ar.num_tokens_post_padded[0]) return;
  const int e = ar.expert_ids[blk];
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int lo = lane & 15;
  const int hi = lane >> 4;
  const int K = ar.K, N = ar.N;
  const int G = K / kGroup;
  const int n0 = blockIdx.y * BN;
  const uint32_t* bq = ar.b_q + (size_t)e * ar.w_stride;
  const uint8_t* bs = ar.b_scales + (size_t)e * ar.s_stride;
  const uint8_t* bb = ar.b_base + (size_t)e * N;

  // Decode duty: thread -> column dc, groups dg0 + f*DG (f < BFT).
  const int dc = tid & (BN - 1);
  const int dg0 = tid / BN;
  const int dcol = min(n0 + dc, N - 1);
  const int dbase = bb[dcol];

  int a_row[2];
  #pragma unroll
  for (int t = 0; t < 2; t++) {
    const int s = ar.sorted_token_ids[blk * BM + 32 * wave + 16 * t + lo];
    a_row[t] = (s < ar.total_slots) ? s / ar.top_k : 0;  // clamp, mask at store
  }
  const uint8_t* a_ptr[2] = {ar.aq + (size_t)a_row[0] * K + 16 * hi,
                             ar.aq + (size_t)a_row[1] * K + 16 * hi};

  __shared__ __align__(16) uint8_t lds[2][GS][2][BN][16];

  uint4 nb[BFT];
  uint32_t nsc[BFT];
  auto load_b_stage = [&](int st) {
  #pragma unroll
    for (int f = 0; f < BFT; f++) {
      const int g = min(st * GS + dg0 + f * DG, G - 1);
      const auto* p =
          reinterpret_cast<const v4i32*>(bq + ((size_t)g * N + dcol) * 4);
      const v4i32 v = __builtin_nontemporal_load(p);
      nb[f] =
          uint4{(uint32_t)v[0], (uint32_t)v[1], (uint32_t)v[2], (uint32_t)v[3]};
      nsc[f] = bs[(size_t)g * N + dcol];
    }
  };
  auto decode_stage = [&](int buf) {
  #pragma unroll
    for (int f = 0; f < BFT; f++) {
      uint32_t tl, th;
      mxfp4_e5m2_tables((int)nsc[f] - dbase, tl, th);
      uint32_t w[4] = {nb[f].x, nb[f].y, nb[f].z, nb[f].w};
      uint32_t d[8];
  #pragma unroll
      for (int o = 0; o < 4; o++)
        dequant_mxfp4_8_e5m2(w[o], tl, th, d[2 * o], d[2 * o + 1]);
      const int dg = dg0 + f * DG;
      *reinterpret_cast<uint4*>(&lds[buf][dg][0][dc][0]) =
          uint4{d[0], d[1], d[2], d[3]};
      *reinterpret_cast<uint4*>(&lds[buf][dg][1][dc][0]) =
          uint4{d[4], d[5], d[6], d[7]};
    }
  };
  // A ring: 8 groups (two K-stages) in flight per row tile, refilled in place
  // right after use so no copy ever forces an early s_wait_loadcnt. Static
  // slot indices come from unrolling two stages per loop iteration.
  constexpr int AR = 2 * GS;
  uint4 a_r[AR][2];
  auto load_a = [&](int g, int slot) {
    const int gg = min(g, G - 1);
  #pragma unroll
    for (int t = 0; t < 2; t++)
      __builtin_memcpy(&a_r[slot][t], a_ptr[t] + gg * kGroup, 16);
  };

  v8f32 acc[2][CT];
  #pragma unroll
  for (int t = 0; t < 2; t++)
  #pragma unroll
    for (int ct = 0; ct < CT; ct++) acc[t][ct] = {0, 0, 0, 0, 0, 0, 0, 0};

  const int nstages = (G + GS - 1) / GS;
  load_b_stage(0);
  #pragma unroll
  for (int q = 0; q < AR; q++) load_a(q, q);
  decode_stage(0);
  __syncthreads();

  for (int st0 = 0; st0 < nstages; st0 += 2) {
  #pragma unroll
    for (int h = 0; h < 2; h++) {
      const int st = st0 + h;
      if (st >= nstages) break;
      const int buf = st & 1;
      const bool has_next = st + 1 < nstages;
      if (has_next) load_b_stage(st + 1);
      // B-fragment lookahead: group gi+1's LDS reads are issued before group
      // gi's WMMAs (static double buffer, gi unrolled). Group 0 of a stage is
      // read right after the barrier.
      uint4 bwr[2][CT];
  #pragma unroll
      for (int ct = 0; ct < CT; ct++)
        bwr[0][ct] =
            *reinterpret_cast<const uint4*>(&lds[buf][0][hi][16 * ct + lo][0]);
  #pragma unroll
      for (int gi = 0; gi < GS; gi++) {
        const int g = st * GS + gi;
        if (g >= G) break;
        if (gi + 1 < GS) {
  #pragma unroll
          for (int ct = 0; ct < CT; ct++)
            bwr[(gi + 1) & 1][ct] = *reinterpret_cast<const uint4*>(
                &lds[buf][gi + 1][hi][16 * ct + lo][0]);
        }
        const int slot = h * GS + gi;
        v2i32 af0[2], af1[2];
  #pragma unroll
        for (int t = 0; t < 2; t++) {
          af0[t][0] = (int)a_r[slot][t].x;
          af0[t][1] = (int)a_r[slot][t].y;
          af1[t][0] = (int)a_r[slot][t].z;
          af1[t][1] = (int)a_r[slot][t].w;
        }
  #pragma unroll
        for (int ct = 0; ct < CT; ct++) {
          const uint4 bw = bwr[gi & 1][ct];
          v2i32 bf0, bf1;
          bf0[0] = (int)bw.x;
          bf0[1] = (int)bw.y;
          bf1[0] = (int)bw.z;
          bf1[1] = (int)bw.w;
  #pragma unroll
          for (int t = 0; t < 2; t++) {
            acc[t][ct] = wmma_fp8_bf8(af0[t], bf0, acc[t][ct]);
            acc[t][ct] = wmma_fp8_bf8(af1[t], bf1, acc[t][ct]);
          }
        }
        load_a(g + AR, slot);  // registers of this slot are dead now: refill
      }
      if (has_next) decode_stage(buf ^ 1);
      __syncthreads();
    }
  }

  // Epilogue: lane -> col 16*ct + lo, slot i -> row 8*hi + i of each row tile.
  #pragma unroll
  for (int ct = 0; ct < CT; ct++) {
    const int n = n0 + 16 * ct + lo;
    if (n >= N) continue;
    const float bsc = __uint_as_float((uint32_t)bb[n] << 23);  // 2^(base-127)
    const float bv = bias ? to_f<T>(bias[(size_t)e * N + n]) : 0.0f;
  #pragma unroll
    for (int t = 0; t < 2; t++) {
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int r = 32 * wave + 16 * t + 8 * hi + i;
        const int s = ar.sorted_token_ids[blk * BM + r];
        if (s >= ar.total_slots) continue;
        float v = acc[t][ct][i] * bsc * ar.ascale[s / ar.top_k];
        if (ar.topk_w) v *= ar.topk_w[s];
        c[(size_t)s * N + n] = from_f<T>(v + bv);
      }
    }
  }
}

#else
template <typename T, int KW, int PF>
__global__ void moe_swmmac_kernel(MoeArgs, T*, const T*) {}
template <typename T, int MT, int KW, int PF>
__global__ void moe_wmma_kernel(MoeArgs, T*, const T*) {}
template <typename T, int BN>
__global__ void moe_hybrid_kernel(MoeArgs, T*, const T*) {}
#endif

// KW (waves per block, K split): start high for latency hiding and back off
// as the grid grows: blocks*KW capped at ~4096 waves (128 per CU on 32 CUs).
// VLLM_MXFP4_RDNA4_MOE_KW overrides.
inline int moe_kw(long blocks, int cap) {
  static const int ov = [] {
    const char* e = std::getenv("VLLM_MXFP4_RDNA4_MOE_KW");
    return e ? atoi(e) : 0;
  }();
  if (ov == 1 || ov == 2 || ov == 4 || ov == 8) return ov;
  int kw = cap;
  while (kw > 1 && blocks * kw > 4096) kw /= 2;
  return kw;
}

template <typename T>
void launch(const MoeArgs& ar, T* c, const T* bias, int block_size_m,
            int num_token_blocks, cudaStream_t st) {
  const long col_tiles = (ar.N + 15) / 16;
  const dim3 grd((unsigned)col_tiles, (unsigned)num_token_blocks);
  const long blocks = col_tiles * (long)num_token_blocks;
#define L_SW(KW_) \
  moe_swmmac_kernel<T, KW_, 8><<<grd, 32 * (KW_), 0, st>>>(ar, c, bias)
#define L_WM(MT_, KW_)                               \
  moe_wmma_kernel<T, MT_, KW_, ((MT_) >= 2 ? 4 : 8)> \
      <<<grd, 32 * (KW_), 0, st>>>(ar, c, bias)
#define SW_KW(kw)     \
  do {                \
    if (kw == 1)      \
      L_SW(1);        \
    else if (kw == 2) \
      L_SW(2);        \
    else if (kw == 4) \
      L_SW(4);        \
    else              \
      L_SW(8);        \
  } while (0)
#define WM_KW(MT_, kw) \
  do {                 \
    if (kw == 1)       \
      L_WM(MT_, 1);    \
    else if (kw == 2)  \
      L_WM(MT_, 2);    \
    else if (kw == 4)  \
      L_WM(MT_, 4);    \
    else               \
      L_WM(MT_, 8);    \
  } while (0)
  if (block_size_m == 256) {
    // VLLM_MXFP4_RDNA4_MOE_BN = 64 | 128 (default 64) selects the hybrid
    // N-tile.
    static const int bn = [] {
      const char* e = std::getenv("VLLM_MXFP4_RDNA4_MOE_BN");
      return (e && atoi(e) == 128) ? 128 : 64;
    }();
    const dim3 g2((unsigned)num_token_blocks, (unsigned)((ar.N + bn - 1) / bn));
    if (bn == 64)
      moe_hybrid_kernel<T, 64><<<g2, 256, 0, st>>>(ar, c, bias);
    else
      moe_hybrid_kernel<T, 128><<<g2, 256, 0, st>>>(ar, c, bias);
  } else if (block_size_m == 8) {
    const int kw = moe_kw(blocks, 8);
    SW_KW(kw);
  } else if (block_size_m == 16) {
    const int kw = moe_kw(blocks, 8);
    WM_KW(1, kw);
  } else if (block_size_m == 32) {
    const int kw = moe_kw(blocks, 8);
    WM_KW(2, kw);
  } else {  // 64
    const int kw = moe_kw(blocks, 2);
    WM_KW(4, kw);
  }
#undef WM_KW
#undef SW_KW
#undef L_WM
#undef L_SW
}

}  // namespace moe_mxfp4_rdna4
}  // namespace vllm

// [E, K/8, N] uint32 -> [E, K/32, N, 4] fragment-major (one 16-byte load per
// lane per E8M0 group). Scales stay [E, K/32, N] (checkpoint [E, N, K/32] is
// a plain permute on the Python side).
torch::Tensor mxfp4_moe_repack_rdna4(torch::Tensor const& b_q) {
  TORCH_CHECK(b_q.dim() == 3 && b_q.size(1) % 4 == 0,
              "mxfp4_moe_repack_rdna4: expected [E, K/8, N] with K % 32 == 0");
  const int64_t E = b_q.size(0), G = b_q.size(1) / 4, N = b_q.size(2);
  return b_q.contiguous().view({E, G, 4, N}).permute({0, 1, 3, 2}).contiguous();
}

// Per-column E5M2 fold base for the hybrid kernel: base = max_g(e8m0) - 13,
// clamped to >= 1, as uint8 [E, N] from scales [E, K/32, N].
torch::Tensor mxfp4_moe_fold_base_rdna4(torch::Tensor const& b_scales) {
  TORCH_CHECK(b_scales.dim() == 3 && b_scales.scalar_type() == at::kByte,
              "mxfp4_moe_fold_base_rdna4: expected uint8 [E, K/32, N]");
  const auto si = b_scales.to(at::kInt);
  const auto base = (si.amax(1) - 13).clamp_min(1);
  // Lossless iff every group scale is within [base-13, base+13] (the shifted
  // E2M1 magnitudes 0.5..6 stay E5M2 normals). Measured spans: <= 10 (dense
  // Qwen3.8-27B), <= 8 (gpt-oss-120b experts), so this never fires on real
  // checkpoints; callers that hit it must use block_size_m <= 64.
  TORCH_CHECK((si.amin(1) - base >= -13).all().item<bool>(),
              "mxfp4_moe_fold_base_rdna4: an E8M0 column span exceeds the "
              "E5M2 fold range; use block_size_m <= 64 for this weight");
  return base.to(at::kByte).contiguous();
}

void moe_mxfp4_gemm_rdna4(
    torch::Tensor const& a_fp8, torch::Tensor const& a_scale, torch::Tensor& c,
    torch::Tensor const& b_q, torch::Tensor const& b_scales,
    torch::Tensor const& topk_weights, torch::Tensor const& sorted_token_ids,
    torch::Tensor const& expert_ids,
    torch::Tensor const& num_tokens_post_padded, int64_t top_k,
    int64_t block_size_m, bool mul_topk_weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& b_base) {
  using namespace vllm::moe_mxfp4_rdna4;
  TORCH_CHECK(a_fp8.is_cuda() && a_fp8.dim() == 2 &&
                  a_fp8.scalar_type() == at::kByte && a_fp8.is_contiguous(),
              "a_fp8 must be a contiguous uint8 [rows, K]");
  TORCH_CHECK(a_scale.numel() == a_fp8.size(0),
              "a_scale must have one entry per A row");
  TORCH_CHECK(
      c.dim() == 2 && c.is_contiguous() &&
          (c.scalar_type() == at::kHalf || c.scalar_type() == at::kBFloat16),
      "c must be contiguous fp16/bf16 [M*top_k, N]");
  TORCH_CHECK(
      b_q.dim() == 4 && b_q.size(3) == 4 && b_q.is_contiguous(),
      "b_q must be fragment-major [E, K/32, N, 4] from mxfp4_moe_repack_rdna4");
  TORCH_CHECK(b_scales.dim() == 3 && b_scales.is_contiguous() &&
                  b_scales.scalar_type() == at::kByte,
              "b_scales must be contiguous uint8 [E, K/32, N]");
  const int K = (int)a_fp8.size(1), N = (int)b_q.size(2);
  TORCH_CHECK(K % kGroup == 0 && b_q.size(1) == K / kGroup &&
                  b_scales.size(1) == K / kGroup && b_scales.size(2) == N,
              "K/32 mismatch between a_fp8, b_q and b_scales");
  TORCH_CHECK(c.size(1) == N, "c must have N columns");
  TORCH_CHECK(block_size_m == 8 || block_size_m == 16 || block_size_m == 32 ||
                  block_size_m == 64 || block_size_m == 256,
              "block_size_m must be 8, 16, 32, 64 or 256");
  TORCH_CHECK(top_k >= 1, "top_k must be >= 1");
  TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt &&
                  expert_ids.scalar_type() == at::kInt &&
                  num_tokens_post_padded.scalar_type() == at::kInt,
              "routing tensors must be int32");
  const int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);
  TORCH_CHECK(expert_ids.size(0) >= num_token_blocks,
              "expert_ids must have one entry per token block");
  if (bias) {
    TORCH_CHECK(bias->dim() == 2 && bias->size(0) == b_q.size(0) &&
                    bias->size(1) == N && bias->is_contiguous() &&
                    bias->scalar_type() == c.scalar_type(),
                "bias must be contiguous [E, N] with c's dtype");
  }
  if (mul_topk_weight) {
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat &&
                    topk_weights.numel() >= c.size(0),
                "topk_weights must be fp32 with one entry per slot");
  }
  const at::cuda::OptionalCUDAGuard guard(a_fp8.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  MoeArgs ar;
  ar.aq = reinterpret_cast<const uint8_t*>(a_fp8.data_ptr());
  ar.ascale = a_scale.data_ptr<float>();
  ar.b_q = reinterpret_cast<const uint32_t*>(b_q.data_ptr());
  ar.b_scales = reinterpret_cast<const uint8_t*>(b_scales.data_ptr());
  ar.topk_w = mul_topk_weight ? topk_weights.data_ptr<float>() : nullptr;
  ar.sorted_token_ids = sorted_token_ids.data_ptr<int32_t>();
  ar.expert_ids = expert_ids.data_ptr<int32_t>();
  ar.num_tokens_post_padded = num_tokens_post_padded.data_ptr<int32_t>();
  ar.total_slots = (int)c.size(0);
  ar.top_k = (int)top_k;
  ar.N = N;
  ar.K = K;
  // Hybrid (block_size_m 256) folds the E8M0 scale into E5M2 weights around a
  // per-column base = max - 13 (mxfp4_moe_fold_base_rdna4). Fold is lossless
  // while every group scale is within [base-13, base+13]; if a checkpoint ever
  // violates that (none measured do), fall back to the E4M3 64-row kernel.
  torch::Tensor base_t;
  if (block_size_m == 256) {
    // A supplied b_base is trusted (validated once by mxfp4_moe_fold_base_rdna4
    // at weight load); computing it here per call costs two reductions.
    base_t = b_base ? *b_base : mxfp4_moe_fold_base_rdna4(b_scales);
    TORCH_CHECK(base_t.dim() == 2 && base_t.size(0) == b_q.size(0) &&
                    base_t.size(1) == N && base_t.is_contiguous() &&
                    base_t.scalar_type() == at::kByte,
                "b_base must be contiguous uint8 [E, N]");
    ar.b_base = reinterpret_cast<const uint8_t*>(base_t.data_ptr());
  } else {
    ar.b_base = nullptr;
  }
  ar.w_stride = (size_t)b_q.size(1) * N * 4;
  ar.s_stride = (size_t)b_scales.size(1) * N;
  if (num_token_blocks == 0) return;
  if (c.scalar_type() == at::kHalf) {
    launch<half>(
        ar, reinterpret_cast<half*>(c.data_ptr()),
        bias ? reinterpret_cast<const half*>(bias->data_ptr()) : nullptr,
        (int)block_size_m, num_token_blocks, stream);
  } else {
    launch<bf16_t>(
        ar, reinterpret_cast<bf16_t*>(c.data_ptr()),
        bias ? reinterpret_cast<const bf16_t*>(bias->data_ptr()) : nullptr,
        (int)block_size_m, num_token_blocks, stream);
  }
}
