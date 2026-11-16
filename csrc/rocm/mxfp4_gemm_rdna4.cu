// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MXFP4 dense GEMM for RDNA4 (gfx12). Port of mxfp4_gemm_rdna3.cu.
//
// What changes vs the RDNA3 kernel, and why:
//
//  1. DTYPE. RDNA3 has no FP8 matrix path, so it upcasts E2M1 -> bf16/fp16 and
//     runs W4A16 through v_wmma_f32_16x16x16_bf16. RDNA4 has native FP8 WMMA,
//     so we decode E2M1 -> OCP E4M3 (exact: all 8 magnitudes fit) and run
//     W4A8 through v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12. Activations are
//     dynamically quantized to E4M3 with one fp32 scale per row (per-token).
//
//  2. SCALE PLACEMENT. RDNA3 folds the E8M0 scale into the dequantized element
//     as an exponent add. E4M3 has a 4-bit exponent (bias 7), so that would
//     overflow; here the scale stays out of the element and is applied as one
//     fp32 FMA per 32-group on the accumulator. Because the accumulator's N
//     index is constant per lane, that is a single scalar per lane per group.
//
//  3. FRAGMENT / ACCUMULATOR LAYOUT. gfx11 WMMA takes 16-element A/B operands
//     duplicated across wave halves; gfx12 takes 8 elements (v2i32 for FP8)
//     with no duplication. The accumulator also re-maps: gfx11 is m = 2*i + hi,
//     gfx12 is m = 8*hi + i (verified on-device, not assumed).
//
//  4. ATOMICS. The RDNA3 K-split epilogue emulates packed fp16/bf16 atomic add
//     with an atomicCAS retry loop. gfx12 has the real instructions
//     (global_atomic_pk_add_f16 / _bf16), used directly below.
//
// Layouts (match the RDNA3 WMMA path):
//   a        [M, K]     fp16 / bf16
//   b_q      [K/32, N, 4] uint32 (fragment-major, from mxfp4_repack_rdna4):
//            b_q[g][n][o] holds K codes 32g+8o .. 32g+8o+7 of column n, nibble
//            i = K+i; one 16-byte load gives a column's whole E8M0 group
//   b_scale  [K/32, N]  uint8 E8M0
//   out      [M, N]     same dtype as a

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdlib>
#include <type_traits>

#include "qdq_mxfp4_rdna4.cuh"

// Defined in mxfp4_swmmac_rdna4.cu (same op; SWMMAC path for M <= 8).
void mxfp4_swmmac_skinny_launch(const uint8_t* aq, const float* ascale,
                                const uint32_t* b_q, const uint8_t* b_scales,
                                void* c, bool out_bf16, int M, int N, int K,
                                void* stream);

#if defined(__gfx1200__) || defined(__gfx1201__)
  #define __HIP__MXFP4_RDNA4__
#endif

namespace vllm {
namespace mxfp4_rdna4 {

using bf16_t = __hip_bfloat16;

constexpr int kGroup = 32;    // MXFP4 E8M0 block
constexpr int kGemvMaxM = 8;  // below this the matrix engine loses to GEMV

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

#if defined(__HIP__MXFP4_RDNA4__) || !defined(__HIP_DEVICE_COMPILE__)

using v8f32 = float __attribute__((ext_vector_type(8)));
using v2i32 = int __attribute__((ext_vector_type(2)));

__device__ __forceinline__ v8f32 wmma_fp8(v2i32 a, v2i32 b, v8f32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

// ---------------------------------------------------------------------------
// Skinny WMMA path (9 <= M <= 64), after the upstream wvSplitK design. One
// block per 16-column tile covers ALL M rows (MT <= 4 row tiles), so every
// weight byte is read exactly once per launch and the B stream can bypass the
// caches (nontemporal). KW waves each own a contiguous K slice and reduce
// through LDS at the end (no global atomics, no extra launches); each wave
// keeps PF groups of B/A/scale loads in flight in a register ring. Per group a
// lane loads its 8-byte B word pair (octets 2*hi, 2*hi+1) and 16 A bytes per
// row tile, and issues 2*MT WMMAs.
template <typename T, int MT, int KW, int PF>
__global__ void __launch_bounds__(32 * KW)
    gemm_mxfp4_wmma_fp8_skinny(const uint8_t* __restrict__ aq,
                               const float* __restrict__ ascale,
                               const uint32_t* __restrict__ b_q,
                               const uint8_t* __restrict__ b_scales,
                               T* __restrict__ c, int M, int N, int K) {
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lo = lane & 15;
  const int hi = lane >> 4;
  const int n_col = blockIdx.x * 16 + lo;
  const bool n_ok = n_col < N;
  const int ngroups = K / kGroup;
  const int g_begin = (ngroups * wave) / KW;
  const int g_end = (ngroups * (wave + 1)) / KW;

  bool a_ok[MT];
  #pragma unroll
  for (int t = 0; t < MT; t++) a_ok[t] = (16 * t + lo) < M;

  struct Slot {
    unsigned long long b;
    uint4 a[MT];
    float ws;
  };
  Slot ring[PF];
  auto load = [&](int g, Slot& sl) {
    const int k_g = g * kGroup;
    sl.b = 0ull;
    sl.ws = 0.0f;
    if (n_ok) {
      const auto* bp = reinterpret_cast<const unsigned long long*>(
          b_q + ((size_t)g * N + n_col) * 4 + 2 * hi);
      sl.b = __builtin_nontemporal_load(bp);
      sl.ws = mxfp4_e8m0_to_f32(b_scales[(size_t)g * N + n_col]);
    }
  #pragma unroll
    for (int t = 0; t < MT; t++) {
      uint4 aw = {0u, 0u, 0u, 0u};
      if (a_ok[t])
        __builtin_memcpy(&aw, aq + (size_t)(16 * t + lo) * K + k_g + 16 * hi,
                         16);
      sl.a[t] = aw;
    }
  };

  v8f32 acc[MT];
  #pragma unroll
  for (int t = 0; t < MT; t++) acc[t] = {0, 0, 0, 0, 0, 0, 0, 0};

  #pragma unroll
  for (int d = 0; d < PF; d++)
    if (g_begin + d < g_end) load(g_begin + d, ring[d]);

  for (int g0 = g_begin; g0 < g_end; g0 += PF) {
  #pragma unroll
    for (int d = 0; d < PF; d++) {
      const int g = g0 + d;
      if (g >= g_end) break;
      const Slot& sl = ring[d];
      v8f32 grp[MT];
  #pragma unroll
      for (int t = 0; t < MT; t++) grp[t] = {0, 0, 0, 0, 0, 0, 0, 0};
  #pragma unroll
      for (int s = 0; s < 2; s++) {  // step s, half hi -> K-octet 2*hi + s
        uint32_t blo, bhi;
        dequant_mxfp4_8_e4m3((uint32_t)(s == 0 ? sl.b : (sl.b >> 32)), blo,
                             bhi);
        v2i32 bf;
        bf[0] = (int)blo;
        bf[1] = (int)bhi;
  #pragma unroll
        for (int t = 0; t < MT; t++) {
          v2i32 af;
          af[0] = (int)(s == 0 ? sl.a[t].x : sl.a[t].z);
          af[1] = (int)(s == 0 ? sl.a[t].y : sl.a[t].w);
          grp[t] = wmma_fp8(af, bf, grp[t]);
        }
      }
      const float ws = sl.ws;
  #pragma unroll
      for (int t = 0; t < MT; t++)
  #pragma unroll
        for (int i = 0; i < 8; i++) acc[t][i] = fmaf(grp[t][i], ws, acc[t][i]);
      if (g + PF < g_end) load(g + PF, ring[d]);
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
  // gfx12 accumulator: lane -> n = lo, slot i -> m = 8*hi + i.
  #pragma unroll
  for (int t = 0; t < MT; t++) {
  #pragma unroll
    for (int i = 0; i < 8; i++) {
      const int m_row = 16 * t + 8 * hi + i;
      if (m_row >= M) continue;
      c[(size_t)m_row * N + n_col] = from_f<T>(acc[t][i] * ascale[m_row]);
    }
  }
}

// ---------------------------------------------------------------------------
// Section 1: general (prefill) path. 16M x 16N tile, one wave, full K.
// ---------------------------------------------------------------------------
// WM x WN waves per block, each wave owning one 16x16 WMMA tile, so the block
// tile is (16*WM) x (16*WN). WM=WN=1 is the original single-wave kernel.
template <typename T, int WM, int WN, bool KSPLIT>
__global__ void gemm_mxfp4_wmma_fp8_direct(const uint8_t* __restrict__ aq,
                                           const float* __restrict__ ascale,
                                           const uint32_t* __restrict__ b_q,
                                           const uint8_t* __restrict__ b_scales,
                                           T* __restrict__ c, int M, int N,
                                           int K, float* __restrict__ c_f32,
                                           int k_per_split) {
  const int wave = threadIdx.x >> 5;
  const int m_tile = blockIdx.y * (16 * WM) + 16 * (wave / WN);
  const int n_tile = blockIdx.x * (16 * WN) + 16 * (wave % WN);
  if (m_tile >= M || n_tile >= N) return;

  const int lane = threadIdx.x & 31;
  const int lo = lane & 15;  // A: row | B: col | acc: N index
  const int hi = lane >> 4;  // which half of the 16-wide K step

  const int n_col = n_tile + lo;
  const int a_row = m_tile + lo;
  const bool a_ok = a_row < M;
  const bool n_ok = n_col < N;

  v8f32 acc = {0, 0, 0, 0, 0, 0, 0, 0};

  // One iteration per E8M0 group (32 K) = two 16-wide WMMA steps.
  // K-split: block z owns [k_begin, k_end). Host picks k_per_split as a
  // multiple of kGroup; the last split absorbs any remainder.
  const int k_begin = KSPLIT ? (int)blockIdx.z * k_per_split : 0;
  const int k_end =
      KSPLIT ? ((blockIdx.z == gridDim.z - 1) ? K : k_begin + k_per_split) : K;
  for (int k_g = k_begin; k_g < k_end; k_g += kGroup) {
    v8f32 grp = {0, 0, 0, 0, 0, 0, 0, 0};
    // K-octet assignment: WMMA step s, lane half hi covers octet 2*hi + s.
    // A and B agree, so the K sum is unchanged, and each lane's two B words
    // (fragment-major layout) and 16 A bytes are each one contiguous load.
    uint2 bw = {0u, 0u};
    if (n_ok)
      __builtin_memcpy(
          &bw, b_q + ((size_t)(k_g / kGroup) * N + n_col) * 4 + 2 * hi, 8);
    uint4 aw = {0u, 0u, 0u, 0u};
    if (a_ok) __builtin_memcpy(&aw, aq + (size_t)a_row * K + k_g + 16 * hi, 16);
  #pragma unroll
    for (int s = 0; s < 2; s++) {
      v2i32 af;
      af[0] = (int)(s == 0 ? aw.x : aw.z);
      af[1] = (int)(s == 0 ? aw.y : aw.w);
      const uint32_t qa = s == 0 ? bw.x : bw.y;
      uint32_t blo, bhi;
      dequant_mxfp4_8_e4m3(qa, blo, bhi);
      v2i32 bf;
      bf[0] = (int)blo;
      bf[1] = (int)bhi;
      grp = wmma_fp8(af, bf, grp);
    }
    // E8M0 is constant across the group and across this lane's 8 outputs,
    // because the accumulator's N index is fixed per lane.
    const float wsc =
        n_ok ? mxfp4_e8m0_to_f32(b_scales[(size_t)(k_g / kGroup) * N + n_col])
             : 0.0f;
  #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = fmaf(grp[i], wsc, acc[i]);
  }

  if (!n_ok) return;
  // gfx12 accumulator: lane -> n = lo, slot i -> m = 8*hi + i.
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const int m_row = m_tile + 8 * hi + i;
    if (m_row >= M) continue;
    // Per-token scale distributes over the K sum, so apply it before the
    // split reduction. gridDim.z > 1 reduces into a zeroed fp32 workspace via
    // native global_atomic_add_f32 (no CAS). gfx12 has packed f16/bf16
    // atomics too, but this lane owns one n and 8 m rows, so a packed pair
    // would need a lane shuffle plus an odd-N tail path.
    const float v = acc[i] * ascale[m_row];
    if constexpr (KSPLIT)
      unsafeAtomicAdd(c_f32 + (size_t)m_row * N + n_col, v);
    else
      c[(size_t)m_row * N + n_col] = from_f<T>(v);
  }
}

template <typename T, int WM, int WN, int TM, int TN, bool KSPLIT>
__global__ void __launch_bounds__(32 * WM * WN)
    gemm_mxfp4_wmma_fp8_tiled(const uint8_t* __restrict__ aq,
                              const float* __restrict__ ascale,
                              const uint32_t* __restrict__ b_q,
                              const uint8_t* __restrict__ b_scales,
                              T* __restrict__ c, int M, int N, int K,
                              float* __restrict__ c_f32, int k_per_split) {
  // LDS variant, after the gfx11 GPTQ/MXFP4 tiled kernels: WM x WN waves per
  // block, each wave owning a TM x TN grid of 16x16 WMMA tiles (TM=TN=2 is
  // their 32x32-per-wave shape), so a group costs 2*TM*TN WMMAs per barrier
  // instead of 2 and the barrier is amortised.
  //
  // The packed B words of one 32-wide E8M0 group (4 per column) are loaded
  // and decoded ONCE per block into LDS as [octet][n] -> 8 E4M3 bytes; decode
  // threads write consecutive n and the WMMA fragment reads (lane -> n,
  // octet = 2*step + half) are 8-byte loads over consecutive n, so both sides
  // are bank-conflict free without padding. Two buffers: group g+1's global
  // loads are issued before group g's WMMAs and decoded into the other buffer
  // after them, one __syncthreads per group covering both hazards.
  constexpr int BM = 16 * WM * TM;
  constexpr int BN = 16 * WN * TN;
  constexpr int NT = 32 * WM * WN;
  constexpr int WORDS =
      (BN + NT - 1) / NT;  // 16-byte B columns per thread / group

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int lo = lane & 15;  // A: row | B: col | acc: N index
  const int hi = lane >> 4;  // which half of the 16-wide K step
  const int bn0 = blockIdx.x * BN;
  const int wm_base = blockIdx.y * BM + 16 * TM * (wave / WN);
  const int wn_loc = 16 * TN * (wave % WN);  // wave's first column in block
  const int wn_base = bn0 + wn_loc;

  // No per-wave early return: every thread joins the block barriers and the
  // shared decode; out-of-range work is masked.
  int a_row[TM];
  bool a_ok[TM];
  #pragma unroll
  for (int tm = 0; tm < TM; tm++) {
    a_row[tm] = wm_base + 16 * tm + lo;
    a_ok[tm] = a_row[tm] < M;
  }
  int n_col[TN];
  bool n_ok[TN];
  #pragma unroll
  for (int tn = 0; tn < TN; tn++) {
    n_col[tn] = wn_base + 16 * tn + lo;
    n_ok[tn] = n_col[tn] < N;
  }

  __shared__ uint2 b_lds[2][4][BN];

  const int k_begin = KSPLIT ? (int)blockIdx.z * k_per_split : 0;
  const int k_end =
      KSPLIT ? ((blockIdx.z == gridDim.z - 1) ? K : k_begin + k_per_split) : K;
  const int ngroups = (k_end - k_begin) / kGroup;

  uint4 qa_nx[WORDS];
  v2i32 a_nx[TM][2];
  float wsc_nx[TN];

  auto load_group = [&](int g) {
    const int k_g = k_begin + g * kGroup;
  #pragma unroll
    for (int w = 0; w < WORDS; w++) {
      const int cc = tid + w * NT;
      uint4 v = {0u, 0u, 0u, 0u};
      if (cc < BN && bn0 + cc < N)
        __builtin_memcpy(&v, b_q + ((size_t)(k_g / kGroup) * N + bn0 + cc) * 4,
                         16);
      qa_nx[w] = v;
    }
  #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      // Step st, half hi covers K-octet 2*hi + st (matches the B read below),
      // so both steps' A bytes are one 16-byte load.
      uint4 aw = {0u, 0u, 0u, 0u};
      if (a_ok[tm])
        __builtin_memcpy(&aw, aq + (size_t)a_row[tm] * K + k_g + 16 * hi, 16);
      a_nx[tm][0][0] = (int)aw.x;
      a_nx[tm][0][1] = (int)aw.y;
      a_nx[tm][1][0] = (int)aw.z;
      a_nx[tm][1][1] = (int)aw.w;
    }
  #pragma unroll
    for (int tn = 0; tn < TN; tn++)
      wsc_nx[tn] = n_ok[tn]
                       ? mxfp4_e8m0_to_f32(
                             b_scales[(size_t)(k_g / kGroup) * N + n_col[tn]])
                       : 0.0f;
  };
  auto store_group = [&](int buf) {
  #pragma unroll
    for (int w = 0; w < WORDS; w++) {
      const int cc = tid + w * NT;
      if (cc < BN) {
        const uint32_t q[4] = {qa_nx[w].x, qa_nx[w].y, qa_nx[w].z, qa_nx[w].w};
  #pragma unroll
        for (int o = 0; o < 4; o++) {
          uint32_t blo, bhi;
          dequant_mxfp4_8_e4m3(q[o], blo, bhi);
          b_lds[buf][o][cc] = make_uint2(blo, bhi);
        }
      }
    }
  };

  v8f32 acc[TM][TN];
  #pragma unroll
  for (int tm = 0; tm < TM; tm++)
  #pragma unroll
    for (int tn = 0; tn < TN; tn++) acc[tm][tn] = {0, 0, 0, 0, 0, 0, 0, 0};
  v2i32 a_cur[TM][2];
  float wsc_cur[TN];

  if (ngroups > 0) {
    load_group(0);
    store_group(0);
  #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
      a_cur[tm][0] = a_nx[tm][0];
      a_cur[tm][1] = a_nx[tm][1];
    }
  #pragma unroll
    for (int tn = 0; tn < TN; tn++) wsc_cur[tn] = wsc_nx[tn];
  }
  __syncthreads();

  for (int g = 0; g < ngroups; g++) {
    const int buf = g & 1;
    const bool has_next = (g + 1 < ngroups);
    if (has_next) load_group(g + 1);  // in flight during this group's WMMAs

    v8f32 grp[TM][TN];
  #pragma unroll
    for (int tm = 0; tm < TM; tm++)
  #pragma unroll
      for (int tn = 0; tn < TN; tn++) grp[tm][tn] = {0, 0, 0, 0, 0, 0, 0, 0};
  #pragma unroll
    for (int st = 0; st < 2; st++) {
  #pragma unroll
      for (int tn = 0; tn < TN; tn++) {
        const uint2 bw = b_lds[buf][2 * hi + st][wn_loc + 16 * tn + lo];
        v2i32 bf;
        bf[0] = (int)bw.x;
        bf[1] = (int)bw.y;
  #pragma unroll
        for (int tm = 0; tm < TM; tm++)
          grp[tm][tn] = wmma_fp8(a_cur[tm][st], bf, grp[tm][tn]);
      }
    }
    // E8M0 is constant across the group and across a lane's 8 outputs of a
    // tile, because the accumulator's N index is fixed per lane.
  #pragma unroll
    for (int tm = 0; tm < TM; tm++)
  #pragma unroll
      for (int tn = 0; tn < TN; tn++)
  #pragma unroll
        for (int i = 0; i < 8; i++)
          acc[tm][tn][i] = fmaf(grp[tm][tn][i], wsc_cur[tn], acc[tm][tn][i]);

    if (has_next) {
      store_group(buf ^ 1);
  #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        a_cur[tm][0] = a_nx[tm][0];
        a_cur[tm][1] = a_nx[tm][1];
      }
  #pragma unroll
      for (int tn = 0; tn < TN; tn++) wsc_cur[tn] = wsc_nx[tn];
    }
    __syncthreads();
  }

  // Past the last barrier. gfx12 accumulator: lane -> n = lo, slot i -> m =
  // 8*hi + i. Per-token scale distributes over the K sum, so it is applied
  // before the split reduction (native global_atomic_add_f32, no CAS).
  #pragma unroll
  for (int tn = 0; tn < TN; tn++) {
    if (!n_ok[tn]) continue;
  #pragma unroll
    for (int tm = 0; tm < TM; tm++) {
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int m_row = wm_base + 16 * tm + 8 * hi + i;
        if (m_row >= M) continue;
        const float v = acc[tm][tn][i] * ascale[m_row];
        if constexpr (KSPLIT)
          unsafeAtomicAdd(c_f32 + (size_t)m_row * N + n_col[tn], v);
        else
          c[(size_t)m_row * N + n_col[tn]] = from_f<T>(v);
      }
    }
  }
}

#else  // non-RDNA4 device pass: keep symbols so the TU links everywhere.

template <typename T, int WM, int WN, int TM, int TN, bool KSPLIT>
__global__ void gemm_mxfp4_wmma_fp8_tiled(const uint8_t*, const float*,
                                          const uint32_t*, const uint8_t*, T*,
                                          int, int, int, float*, int) {}

template <typename T, int MT, int KW, int PF>
__global__ void gemm_mxfp4_wmma_fp8_skinny(const uint8_t*, const float*,
                                           const uint32_t*, const uint8_t*, T*,
                                           int, int, int) {}

template <typename T, int WM, int WN, bool KSPLIT>
__global__ void gemm_mxfp4_wmma_fp8_direct(const uint8_t*, const float*,
                                           const uint32_t*, const uint8_t*, T*,
                                           int, int, int, float*, int) {}

#endif  // __HIP__MXFP4_RDNA4__

#if defined(__HIP__MXFP4_RDNA4__) || !defined(__HIP_DEVICE_COMPILE__)
template <typename T, int M_MAX>
__global__ void gemv_mxfp4_rdna4(const T* __restrict__ a,
                                 const uint32_t* __restrict__ b_q,
                                 const uint8_t* __restrict__ b_scales,
                                 T* __restrict__ c, int M, int N, int K) {
  const int n = blockIdx.x * blockDim.y + threadIdx.y;  // one wave per column
  if (n >= N) return;
  const int lane = threadIdx.x;  // 0..31
  const int ngroups = K / kGroup;

  float acc[M_MAX];
  #pragma unroll
  for (int m = 0; m < M_MAX; m++) acc[m] = 0.0f;

  for (int g = lane; g < ngroups; g += 32) {
    uint4 bw;
    __builtin_memcpy(&bw, b_q + ((size_t)g * N + n) * 4, 16);
    const float wsc = mxfp4_e8m0_to_f32(b_scales[(size_t)g * N + n]);
    const uint32_t q[4] = {bw.x, bw.y, bw.z, bw.w};
  #pragma unroll
    for (int o = 0; o < 4; o++) {
      const int k0 = g * kGroup + o * 8;
      // E2M1 -> E4M3 via v_perm (shared decode), then v_cvt_f32_fp8 per
      // element: far fewer VALU ops than a table lookup per nibble.
      uint32_t blo, bhi;
      dequant_mxfp4_8_e4m3(q[o], blo, bhi);
      const float dq[8] = {__builtin_amdgcn_cvt_f32_fp8((int)blo, 0) * wsc,
                           __builtin_amdgcn_cvt_f32_fp8((int)blo, 1) * wsc,
                           __builtin_amdgcn_cvt_f32_fp8((int)blo, 2) * wsc,
                           __builtin_amdgcn_cvt_f32_fp8((int)blo, 3) * wsc,
                           __builtin_amdgcn_cvt_f32_fp8((int)bhi, 0) * wsc,
                           __builtin_amdgcn_cvt_f32_fp8((int)bhi, 1) * wsc,
                           __builtin_amdgcn_cvt_f32_fp8((int)bhi, 2) * wsc,
                           __builtin_amdgcn_cvt_f32_fp8((int)bhi, 3) * wsc};
      for (int m = 0; m < M; m++) {
        const T* ar = a + (size_t)m * K + k0;
        float s = 0.0f;
  #pragma unroll
        for (int i = 0; i < 8; i++) s = fmaf(to_f<T>(ar[i]), dq[i], s);
        acc[m] += s;
      }
    }
  }

  for (int m = 0; m < M; m++) {
    float s = acc[m];
    for (int off = 16; off; off >>= 1) s += __shfl_xor(s, off);
    if (lane == 0) c[(size_t)m * N + n] = from_f<T>(s);
  }
}
#else
template <typename T, int M_MAX>
__global__ void gemv_mxfp4_rdna4(const T*, const uint32_t*, const uint8_t*, T*,
                                 int, int, int) {}
#endif

// Tile config override for tuning sweeps: VLLM_MXFP4_RDNA4_TILE =
// 1 -> 16x16 (1 wave), 2 -> 32x32 (4 waves), 3 -> 64x32 (8 waves).
// Unset / 0 picks by shape.
inline int tile_cfg_override() {
  static const int v = [] {
    const char* e = std::getenv("VLLM_MXFP4_RDNA4_TILE");
    return e ? atoi(e) : 0;
  }();
  return v;
}

template <typename T>
__global__ void f32_to_t(const float* __restrict__ src, T* __restrict__ dst,
                         size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    dst[i] = from_f<T>(src[i]);
}

// VLLM_MXFP4_RDNA4_KSPLIT=n forces n splits (clamped so each split keeps at
// least one E8M0 group). Unset / 0 picks by occupancy.
inline int ksplit_override() {
  static const int v = [] {
    const char* e = std::getenv("VLLM_MXFP4_RDNA4_KSPLIT");
    return e ? atoi(e) : 0;
  }();
  return v;
}

// K-split is off by default. Measured on gfx1201 (K=N=4096): at M=16 the
// kernel takes ~71us whether it runs 256 waves over all of K or 1024 waves
// over K/4, so the skinny-M floor is not serial-K latency and splitting only
// adds the workspace zero, atomics and convert pass (M=32: 73 -> 93us). Kept
// behind VLLM_MXFP4_RDNA4_KSPLIT for experiments; the non-split instantiation
// is unchanged code.
// VLLM_MXFP4_RDNA4_LDS=1 forces the LDS-staged/double-buffered kernel, =0 the
// direct one. Unset picks by M: measured on gfx1201 (K=N=4096) the LDS
// variant wins at M<=32 (M=16: 71 -> 48us) but with one 16x16 tile per wave
// it has only two WMMAs per barrier and halves throughput from M=64 up
// (M=2048: 48 -> 25 TF/s). Amortising that needs >1 tile per wave.
inline int lds_override() {
  static const int v = [] {
    const char* e = std::getenv("VLLM_MXFP4_RDNA4_LDS");
    return e ? atoi(e) : -1;
  }();
  return v;
}
inline bool use_lds_kernel(int M) {
  const int ov = lds_override();
  return ov < 0 ? (M <= 32) : (ov != 0);
}

// VLLM_MXFP4_RDNA4_SKINNY=0 keeps the fp32-activation GEMV for M<=8; unset/1
// uses the W4A8 SWMMAC kernel in mxfp4_swmmac_rdna4.cu.
inline bool use_swmmac_skinny() {
  static const bool v = [] {
    const char* e = std::getenv("VLLM_MXFP4_RDNA4_SKINNY");
    return !e || atoi(e) != 0;
  }();
  return v;
}

// Skinny WMMA (9 <= M <= 64): VLLM_MXFP4_RDNA4_SKWMMA=0 disables it;
// VLLM_MXFP4_RDNA4_SKW = waves per tile (1/2/4), VLLM_MXFP4_RDNA4_SPF = ring
// depth (2/4/8). Defaults are the measured best (see launch_wmma).
inline int env_int(const char* name, int dflt) {
  const char* e = std::getenv(name);
  return e ? atoi(e) : dflt;
}
inline bool skinny_wmma_enabled() {
  static const bool v = env_int("VLLM_MXFP4_RDNA4_SKWMMA", 1) != 0;
  return v;
}
// Measured on gfx1201 (K=N=4096): KW=4 wins for <=2 row tiles (M=16/32:
// 0.040/0.051 -> 0.023/0.024 ms), KW=2 for 3-4 row tiles (M=64: 0.054 ->
// 0.038 ms; KW=4 there is 0.040-0.044 from ring x row-tile register pressure).
inline int skinny_kw(int mt) {
  static const int ov = env_int("VLLM_MXFP4_RDNA4_SKW", 0);
  if (ov == 1 || ov == 2 || ov == 4) return ov;
  return mt <= 2 ? 4 : 2;
}
inline int skinny_pf() {
  static const int v = [] {
    const int k = env_int("VLLM_MXFP4_RDNA4_SPF", 4);
    return (k == 2 || k == 8) ? k : 4;
  }();
  return v;
}

inline int choose_ksplit(int M, int N, int K, int WM, int WN) {
  (void)M;
  (void)N;
  (void)WM;
  (void)WN;
  const int max_by_k = K / kGroup;
  const int ov = ksplit_override();
  if (ov > 1) return ov < max_by_k ? ov : max_by_k;
  return 1;
}

}  // namespace mxfp4_rdna4
}  // namespace vllm

template <typename TT>
void launch_wmma(const uint8_t* aqp, const float* asp, const uint32_t* bq,
                 const uint8_t* bs, TT* cp, int M, int N, int K,
                 const torch::Tensor& a, cudaStream_t stream) {
  using namespace vllm::mxfp4_rdna4;
  // Tile configs (VLLM_MXFP4_RDNA4_TILE): 1 16x16/1w, 2 32x32/4w, 3 64x32/8w
  // (one 16x16 tile per wave; LDS or direct by use_lds_kernel), 4 64x64/4w,
  // 5 128x64/8w (32x32 per wave, LDS only), 6 32x16/2w (LDS only).
  if (M <= 64 && skinny_wmma_enabled()) {
    const int mt = (M + 15) / 16;
    const int kw = skinny_kw(mt), pf = skinny_pf();
    const dim3 grd((N + 15) / 16);
#define SKW_LAUNCH(MT_, KW_, PF_)               \
  gemm_mxfp4_wmma_fp8_skinny<TT, MT_, KW_, PF_> \
      <<<grd, 32 * (KW_), 0, stream>>>(aqp, asp, bq, bs, cp, M, N, K)
#define SKW_PF(MT_, KW_)       \
  do {                         \
    if (pf == 2)               \
      SKW_LAUNCH(MT_, KW_, 2); \
    else if (pf == 8)          \
      SKW_LAUNCH(MT_, KW_, 8); \
    else                       \
      SKW_LAUNCH(MT_, KW_, 4); \
  } while (0)
#define SKW_KW(MT_)   \
  do {                \
    if (kw == 1)      \
      SKW_PF(MT_, 1); \
    else if (kw == 2) \
      SKW_PF(MT_, 2); \
    else              \
      SKW_PF(MT_, 4); \
  } while (0)
    switch (mt) {
      case 1:
        SKW_KW(1);
        break;
      case 2:
        SKW_KW(2);
        break;
      case 3:
        SKW_KW(3);
        break;
      default:
        SKW_KW(4);
        break;
    }
#undef SKW_KW
#undef SKW_PF
#undef SKW_LAUNCH
    return;
  }
  int cfg = tile_cfg_override();
  if (cfg == 0) cfg = 1;
  int WMv = 1, WNv = 1, TMv = 1, TNv = 1;
  bool lds = use_lds_kernel(M);
  switch (cfg) {
    case 2:
      WMv = 2;
      WNv = 2;
      break;
    case 3:
      WMv = 4;
      WNv = 2;
      break;
    case 4:
      WMv = 2;
      WNv = 2;
      TMv = 2;
      TNv = 2;
      lds = true;
      break;
    case 5:
      WMv = 4;
      WNv = 2;
      TMv = 2;
      TNv = 2;
      lds = true;
      break;
    case 6:
      WMv = 2;
      WNv = 1;
      lds = true;
      break;
    default:
      break;
  }
  const int splits = choose_ksplit(M, N, K, WMv, WNv);
  const int k_per = (splits > 1) ? ((K / splits) / kGroup) * kGroup : K;
  const int BM = 16 * WMv * TMv, BN = 16 * WNv * TNv;
  dim3 grd((N + BN - 1) / BN, (M + BM - 1) / BM, splits);
  const int nthreads = 32 * WMv * WNv;
  torch::Tensor ws;
  float* wsp = nullptr;
  if (splits > 1) {
    ws = torch::zeros({M, N}, a.options().dtype(torch::kFloat32));
    wsp = ws.data_ptr<float>();
  }
#define L_LDS(WM_, WN_, TM_, TN_)                                            \
  do {                                                                       \
    if (splits > 1)                                                          \
      gemm_mxfp4_wmma_fp8_tiled<TT, WM_, WN_, TM_, TN_, true>                \
          <<<grd, nthreads, 0, stream>>>(aqp, asp, bq, bs, cp, M, N, K, wsp, \
                                         k_per);                             \
    else                                                                     \
      gemm_mxfp4_wmma_fp8_tiled<TT, WM_, WN_, TM_, TN_, false>               \
          <<<grd, nthreads, 0, stream>>>(aqp, asp, bq, bs, cp, M, N, K, wsp, \
                                         k_per);                             \
  } while (0)
#define L_DIR(WM_, WN_)                                                      \
  do {                                                                       \
    if (splits > 1)                                                          \
      gemm_mxfp4_wmma_fp8_direct<TT, WM_, WN_, true>                         \
          <<<grd, nthreads, 0, stream>>>(aqp, asp, bq, bs, cp, M, N, K, wsp, \
                                         k_per);                             \
    else                                                                     \
      gemm_mxfp4_wmma_fp8_direct<TT, WM_, WN_, false>                        \
          <<<grd, nthreads, 0, stream>>>(aqp, asp, bq, bs, cp, M, N, K, wsp, \
                                         k_per);                             \
  } while (0)
  switch (cfg) {
    case 2:
      if (lds)
        L_LDS(2, 2, 1, 1);
      else
        L_DIR(2, 2);
      break;
    case 3:
      if (lds)
        L_LDS(4, 2, 1, 1);
      else
        L_DIR(4, 2);
      break;
    case 4:
      L_LDS(2, 2, 2, 2);
      break;
    case 5:
      L_LDS(4, 2, 2, 2);
      break;
    case 6:
      L_LDS(2, 1, 1, 1);
      break;
    default:
      if (lds)
        L_LDS(1, 1, 1, 1);
      else
        L_DIR(1, 1);
      break;
  }
#undef L_LDS
#undef L_DIR
  if (splits > 1) {
    const size_t n_el = (size_t)M * N;
    f32_to_t<TT>
        <<<(unsigned)((n_el + 255) / 256), 256, 0, stream>>>(wsp, cp, n_el);
  }
}

// ---------------------------------------------------------------------------
// Torch entry point
// ---------------------------------------------------------------------------
// [K/8, N] uint32 (8 E2M1 codes per word along K) -> fragment-major
// [K/32, N, 4]: out[g][n][o] = in[4g + o][n]. Done once at weight load; every
// RDNA4 kernel then reads a column's whole E8M0 group with one 16-byte load.
torch::Tensor mxfp4_repack_rdna4(torch::Tensor const& b_q) {
  TORCH_CHECK(b_q.dim() == 2 && b_q.size(0) % 4 == 0,
              "mxfp4_repack_rdna4: expected [K/8, N] with K % 32 == 0");
  const int64_t G = b_q.size(0) / 4, N = b_q.size(1);
  return b_q.contiguous().view({G, 4, N}).permute({0, 2, 1}).contiguous();
}

// `a` is the original fp16/bf16 activation, used by the GEMV path which keeps
// full precision. `a_fp8`/`a_scale` are its per-token E4M3 quantization from
// vLLM's scaled_fp8_quant, and drive the WMMA path.
torch::Tensor mxfp4_gemm_rdna4(torch::Tensor const& a,
                               torch::Tensor const& a_fp8,
                               torch::Tensor const& a_scale,
                               torch::Tensor const& b_q,
                               torch::Tensor const& b_scales) {
  using namespace vllm::mxfp4_rdna4;
  TORCH_CHECK(a.is_cuda() && b_q.is_cuda() && b_scales.is_cuda());
  TORCH_CHECK(a.dim() == 2 && b_q.dim() == 3 && b_scales.dim() == 2);
  TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
              "mxfp4_gemm_rdna4: activation must be fp16 or bf16");

  const int M = a.size(0), K = a.size(1), N = b_q.size(1);
  TORCH_CHECK(K % kGroup == 0, "K must be a multiple of 32");
  TORCH_CHECK(
      b_q.size(0) == K / kGroup && b_q.size(2) == 4 && b_q.is_contiguous(),
      "b_q must be the fragment-major [K/32, N, 4] from mxfp4_repack_rdna4");
  TORCH_CHECK(b_scales.size(0) == K / kGroup, "b_scales must be [K/32, N]");
  TORCH_CHECK(a_fp8.size(0) == M && a_fp8.size(1) == K, "a_fp8 must be [M, K]");
  TORCH_CHECK(a_scale.numel() == M, "a_scale must have M entries");

  const at::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto c = torch::empty({M, N}, a.options());

  const auto* bq = reinterpret_cast<const uint32_t*>(b_q.data_ptr());
  const auto* bs = reinterpret_cast<const uint8_t*>(b_scales.data_ptr());

#define DISPATCH(T, TT)                                                     \
  if (M <= kGemvMaxM && use_swmmac_skinny()) {                              \
    mxfp4_swmmac_skinny_launch(                                             \
        reinterpret_cast<const uint8_t*>(a_fp8.data_ptr()),                 \
        reinterpret_cast<const float*>(a_scale.data_ptr()), bq, bs,         \
        c.data_ptr(), std::is_same<TT, __hip_bfloat16>::value, M, N, K,     \
        reinterpret_cast<void*>(static_cast<cudaStream_t>(stream)));        \
  } else if (M <= kGemvMaxM) {                                              \
    dim3 blk(32, 4);                                                        \
    dim3 grd((N + 3) / 4);                                                  \
    gemv_mxfp4_rdna4<TT, kGemvMaxM><<<grd, blk, 0, stream>>>(               \
        reinterpret_cast<const TT*>(a.data_ptr()), bq, bs,                  \
        reinterpret_cast<TT*>(c.data_ptr()), M, N, K);                      \
  } else {                                                                  \
    launch_wmma<TT>(reinterpret_cast<const uint8_t*>(a_fp8.data_ptr()),     \
                    reinterpret_cast<const float*>(a_scale.data_ptr()), bq, \
                    bs, reinterpret_cast<TT*>(c.data_ptr()), M, N, K, a,    \
                    stream);                                                \
  }

  if (a.scalar_type() == at::kHalf) {
    DISPATCH(half, half)
  } else {
    DISPATCH(bf16_t, __hip_bfloat16)
  }
#undef DISPATCH

  return c;
}
