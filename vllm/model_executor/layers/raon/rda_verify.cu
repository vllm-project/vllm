// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// RAON KDA speculative-verify fused step (RAON_KDA_SGL_FUSED_SPEC).
//
// One launch replaces the spec branch of KimiK3DeltaAttention._forward
// (raon kda.py, has_spec_decode with no prefill / plain decode rows):
//   causal_conv1d_update(num_accepted_tokens=...)  -> _scale_packed_v (RDA)
//   -> fused_recurrent_kda(ssm_state_indices=[N, T]) -> o_norm (gated RMSNorm)
// for a packed ragged batch of 1..kTMax query tokens per request (draft k=7 + 1
// bonus = 8; zero-draft decode rows relabelled by kda_spec_state.py have 1).
//
// Cache contract (vLLM 0.28.0 spec KDA, also FlashInfer v0.7.1
// packed_fused_kda_decode, kda_decode.py:666-672):
//   * recurrent state is READ from slot idx[n, num_accepted[n]-1]
//     (fused_recurrent.py:196-202) and token t's state is WRITTEN to slot
//     idx[n, t] when that slot is > 0 (fused_recurrent.py:265-279);
//   * the conv cache of request n is the single rolling window in slot
//     idx[n, 0], time-major [slots, S >= T+2, 3P]; history is read at columns
//     o, o+1, o+2 with o = num_accepted-1 and rewritten as
//     [col o+1, col o+2, x_0 .. x_{L-1}] at columns 0..L+1
//     (causal_conv1d.py:860-956).
//
// Design:
//   * one block per (head, request); the 128x128 fp32 state stays in
//     registers across the whole token chain, loaded once up front and stored
//     once per token checkpoint (SGLang kda_decode_mtp.py:194-208
//     TILES_PER_PASS residency; FlashInfer
//     fused_kda_decode_multitoken.py:20-23, :271-292 state load issued before
//     the front end);
//   * the K axis of a value row spans 8 lanes (16 contiguous k per lane, so
//     state traffic is float4-vectorized and a row is one 512 B segment), 4
//     rows per warp reduce concurrently with 3-step butterflies (SGLang
//     P2_LANES_RESIDENT=8, kda_decode_mtp.py:40-45);
//   * lane kg owns k = j4*32 + kg*4 + m, so each float4 instruction of a row
//     group covers one whole 128 B line of state (FlashInfer
//     fused_kda_decode_multitoken.py:20-23); the q/k/decay smem tiles use the
//     same mapping and are conflict free.  (SGLang's lane-major padded tile,
//     kda_decode_mtp.py:63-80, was tried first: 16 B pieces 64 B apart per
//     instruction left the checkpoint stores half-sector, ~25% slower at
//     N=256);
//   * front end first: conv+SiLU for q/k/v, decay, beta for every token into
//     smem before the recurrence (SGLang phase 1 kda_decode_mtp.py:280-600,
//     FlashInfer stage 1 :294-363); each conv channel's window is read and
//     rewritten by the one thread that owns the channel, so the in-place
//     rolling update needs no barrier (FlashInfer :322-349 aliasing rule);
//   * at large grids one block owns all 128 value rows; below that the rows
//     of a head are split over a kSplit-block thread-block cluster (value rows
//     are independent under the delta rule); the ranks rendezvous after
//     reading the accepted conv history and before the partitioned
//     rolling-window stores, and exchange per-token sums of squares for the
//     RMSNorm through distributed shared memory (FlashInfer
//     fused_kda_decode_multitoken.py:24-33, :255-260, :322-335, :528-560);
//   * recurrence output is rounded to bf16 before the gated RMSNorm, and the
//     norm runs in the same block (FlashInfer :492-495 "Match the composed vLLM
//     path"; SGLang APPLY_ONORM kda_decode_mtp.py:720-750; PR 43659).
//
// RAON specifics: any local head count (grid axis), RDA value scale folded at
// the post-conv seam (fp32 ratio, or computed here from raw delta logits with
// the aten logsigmoid chain of raon kda.py _rda_scale_v_kernel), gate lower
// bound (raon config: -5), sigmoid-gated o_norm.
//
// Numerics: bf16 rounding is kept at the unfused chain's seams (conv output,
// RDA-scaled v, recurrence output) and the math uses the same formulas
// (expf/log1pf, q/sqrt(sum+1e-6)), but reduction order and FMA contraction
// differ from the Triton kernels, so the result is allclose, not bitwise.
// Built WITHOUT --use_fast_math.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace raon_kda_spec {

constexpr int kD = 128;  // head_dim (K = V = 128)
constexpr int kW = 4;    // conv width
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kLanesK = 8;          // lanes spanning one value row's K axis
constexpr int kVec = kD / kLanesK;  // 16 contiguous k per lane
constexpr int kRowGroups =
    32 / kLanesK;  // 4 rows reduced concurrently per warp
// Lane kg of a row group owns k = j4 * (4 * kLanesK) + kg * 4 + m (j4 < kVec/4,
// m < 4): one float4 per (lane, j4), and the 8 lanes of a row cover 128
// contiguous bytes per instruction, so every state load/store instruction
// writes whole 128 B lines (FlashInfer fused_kda_decode_multitoken.py:20-23,
// "k positions jj*(4*KQ) + kq*4 + m ... state traffic stays fully
// coalesced").  The same mapping on the linear q/k/decay smem tiles is
// bank-conflict free (8 lanes x 16 B contiguous, row groups broadcast).
constexpr int kJ4Stride = 4 * kLanesK;  // floats between a lane's float4 chunks
constexpr int kQkTok = kD;              // linear [token][k] smem tiles

// Compile-time tuning knobs (benchmarked in the study; defaults are the shipped
// choice).
#ifndef RAON_KDA_SPEC_MIN_BLOCKS
  #define RAON_KDA_SPEC_MIN_BLOCKS 2
#endif
#ifndef RAON_KDA_SPEC_QK_REGS
  #define RAON_KDA_SPEC_QK_REGS \
    1  // 1: q/k/decay of a token cached in registers; 0: read from smem
#endif
#ifndef RAON_KDA_SPEC_STREAM_CS
  #define RAON_KDA_SPEC_STREAM_CS \
    1  // 1: ld/st.global.cs for state traffic; 0: default caching
#endif
#ifndef RAON_KDA_SPEC_FAST_FRONT
  #define RAON_KDA_SPEC_FAST_FRONT \
    0  // 1: __expf/__fdividef in the conv/gate front end
#endif

__device__ __forceinline__ int qk_pos(int k) { return k; }

__device__ __forceinline__ float bf(const __nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ float bf16_round(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ float sigmoidf_(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_front(float x) {
#if RAON_KDA_SPEC_FAST_FRONT
  return __fdividef(x, 1.0f + __expf(-x));
#else
  return x / (1.0f + expf(-x));
#endif
}

__device__ __forceinline__ float exp_front(float x) {
#if RAON_KDA_SPEC_FAST_FRONT
  return __expf(x);
#else
  return expf(x);
#endif
}

__device__ __forceinline__ float sigmoid_front(float x) {
#if RAON_KDA_SPEC_FAST_FRONT
  return __fdividef(1.0f, 1.0f + __expf(-x));
#else
  return sigmoidf_(x);
#endif
}

// aten / raon _rda_scale_v_kernel logsigmoid: min(x, 0) - log1p(exp(-|x|)).
__device__ __forceinline__ float logsigmoidf_(float x) {
  return fminf(x, 0.0f) - log1pf(expf(-fabsf(x)));
}

__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
  for (int m = kLanesK / 2; m >= 1; m >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, m, 32);
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
  return v;
}

// ratio_mode: 0 = fp32 ratio, 1 = bf16 raw delta logits, 2 = fp32 raw delta
// logits. Grid (H * kSplit, N); with kSplit > 1 launched as (kSplit, 1, 1)
// clusters.
template <int kTMax, int kSplit>
__global__
__launch_bounds__(kThreads, (kSplit == 4 ? 4 : RAON_KDA_SPEC_MIN_BLOCKS)) void kda_spec_fused_kernel(
    const __nv_bfloat16* __restrict__ x,
    int64_t x_row,                     // [T, 3P] pre-conv q|k|v
    const float* __restrict__ conv_w,  // [3P, 4] fp32
    const float* __restrict__ conv_b,  // [3P] fp32 or nullptr
    __nv_bfloat16* __restrict__ conv_pool, int64_t cp_slot,
    int64_t cp_col,  // [slots, S, 3P]
    const __nv_bfloat16* __restrict__ g,
    int64_t g_row,                      // [T, P] raw forget gate
    const float* __restrict__ A_log,    // [H]
    const float* __restrict__ dt_bias,  // [P]
    const __nv_bfloat16* __restrict__ beta,
    int64_t beta_row,  // [T, H] raw logits
    const void* __restrict__ ratio, int64_t ratio_row,
    int ratio_mode,  // [T, H]
    const __nv_bfloat16* __restrict__ onorm_g,
    int64_t og_row,                                    // [T, P] o_norm gate
    const float* __restrict__ onorm_w,                 // [128]
    float* __restrict__ state, int64_t st_slot,        // [slots, H, 128, 128]
    const int* __restrict__ idx, int64_t idx_row,      // [N, Tcols]
    const int* __restrict__ qsl,                       // [N + 1]
    const int* __restrict__ accepted,                  // [N]
    __nv_bfloat16* __restrict__ out, int64_t out_row,  // [T, P]
    int P, float scale, float lower_bound, float eps) {
  constexpr int kRows = kD / kSplit;  // value rows this block owns
  constexpr int kRowsPerThread = kRows / (kWarps * kRowGroups);  // 4 / 2 / 1
  static_assert(
      kRowsPerThread >= 1 && kRowsPerThread * kWarps * kRowGroups == kRows,
      "rows");
  static_assert(kTMax <= kWarps, "one warp per token");
  const int h = blockIdx.x / kSplit;
  const int rank = blockIdx.x % kSplit;
  const int row0 = rank * kRows;
  const int n = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Every rank of a cluster shares (h, n), so all ranks take the same exits.
  const int bos = qsl[n];
  const int L = qsl[n + 1] - bos;
  if (L <= 0) return;
  if (L > kTMax)
    __trap();  // host contract: query length <= idx columns <= kTMax
  int a = accepted[n] - 1;
  a = a < 0 ? 0 : (a > kTMax - 1 ? kTMax - 1 : a);
  const int conv_slot = idx[n * idx_row];
  const int src_slot = idx[n * idx_row + a];
  const int hk = h * kD;

  if (conv_slot <= 0 || src_slot <= 0) {
    // Null row: zero its outputs, leave both caches untouched.
    for (int i = tid; i < L * kRows; i += kThreads) {
      out[(int64_t)(bos + i / kRows) * out_row + hk + row0 + (i % kRows)] =
          __float2bfloat16(0.0f);
    }
    return;
  }

  __shared__ __align__(16) float sQraw[kTMax * kD];
  __shared__ __align__(16) float sKraw[kTMax * kD];
  __shared__ __align__(16) float sV[kTMax * kD];
  __shared__ __align__(16) float sO[kTMax * kRows];
  __shared__ __align__(16) float sQ[kTMax * kQkTok];
  __shared__ __align__(16) float sK[kTMax * kQkTok];
  __shared__ __align__(16) float sDec[kTMax * kQkTok];
  __shared__ float sBeta[kTMax];
  __shared__ float sRatio[kTMax];
  __shared__ float
      sPart[kTMax];  // per-token sum of squares of this rank's rows

  // ---- recurrent state -> registers, issued before the front end ----------
  const int rg = lane / kLanesK;
  const int kg = lane % kLanesK;
  const int k0 = kg * 4;
  float s[kRowsPerThread][kVec];
  {
    const float* src = state + (int64_t)src_slot * st_slot + (int64_t)hk * kD;
#pragma unroll
    for (int b = 0; b < kRowsPerThread; ++b) {
      const int row =
          row0 + warp * (kRowsPerThread * kRowGroups) + b * kRowGroups + rg;
      const float4* p = reinterpret_cast<const float4*>(src + row * kD + k0);
#pragma unroll
      for (int j4 = 0; j4 < kVec / 4; ++j4) {
#if RAON_KDA_SPEC_STREAM_CS
        const float4 v = __ldcs(p + j4 * kLanesK);
#else
        const float4 v = p[j4 * kLanesK];
#endif
        s[b][4 * j4 + 0] = v.x;
        s[b][4 * j4 + 1] = v.y;
        s[b][4 * j4 + 2] = v.z;
        s[b][4 * j4 + 3] = v.w;
      }
    }
  }

  int dsts[kTMax];  // checkpoint slots, read once up front
#pragma unroll
  for (int t = 0; t < kTMax; ++t) dsts[t] = t < L ? idx[n * idx_row + t] : 0;

  // ---- per-token scalars: beta sigmoid, RDA ratio --------------------------
  if (tid < L) {
    const int t = tid;
    const float b = bf(beta[(int64_t)(bos + t) * beta_row + h]);
    sBeta[t] = sigmoidf_(b);
    float r;
    if (ratio_mode == 0) {
      r = static_cast<const float*>(ratio)[(int64_t)(bos + t) * ratio_row + h];
    } else {
      const float d = ratio_mode == 1
                          ? bf(static_cast<const __nv_bfloat16*>(
                                ratio)[(int64_t)(bos + t) * ratio_row + h])
                          : static_cast<const float*>(
                                ratio)[(int64_t)(bos + t) * ratio_row + h];
      r = expf(logsigmoidf_(b + d) - logsigmoidf_(b));
    }
    sRatio[t] = r;
  }

  // ---- front end: 3*128 conv channels + 128 decay channels, 2 per thread ---
  // History is read for both tasks first; under a cluster no rank may rewrite
  // the shared rolling window before every rank holds its history.
  // Every global input of the front end is issued here, next to the state
  // load: one exposed memory round trip per block instead of one per phase.
  __nv_bfloat16 c0[2], c1[2], c2[2];
  __nv_bfloat16
      xs[2][kTMax];  // q|k|v channel inputs, or raw forget gate (decay task)
#pragma unroll
  for (int task_i = 0; task_i < 2; ++task_i) {
    const int task = tid + task_i * kThreads;
    const int ci = task / kD;
    const int d = task % kD;
    if (ci < 3) {
      const int64_t chan = (int64_t)ci * P + hk + d;
      const __nv_bfloat16* win =
          conv_pool + (int64_t)conv_slot * cp_slot + chan;
      c0[task_i] = win[(int64_t)(a + 0) * cp_col];
      c1[task_i] = win[(int64_t)(a + 1) * cp_col];
      c2[task_i] = win[(int64_t)(a + 2) * cp_col];
#pragma unroll
      for (int t = 0; t < kTMax; ++t) {
        if (t < L) xs[task_i][t] = x[(int64_t)(bos + t) * x_row + chan];
      }
    } else {
#pragma unroll
      for (int t = 0; t < kTMax; ++t) {
        if (t < L) xs[task_i][t] = g[(int64_t)(bos + t) * g_row + hk + d];
      }
    }
  }
  // o_norm gate and weight of this warp's token, also up front.
  constexpr int kPerLane = kRows / 32;
  float og[kPerLane], ow[kPerLane];
  if (warp < L) {
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int v = row0 + i * 32 + lane;
      og[i] = bf(onorm_g[(int64_t)(bos + warp) * og_row + hk + v]);
      ow[i] = onorm_w[v];
    }
  }
  if constexpr (kSplit > 1) {
    cooperative_groups::this_cluster().sync();
  } else {
    __syncthreads();  // sRatio / sBeta
  }
#pragma unroll
  for (int task_i = 0; task_i < 2; ++task_i) {
    const int task = tid + task_i * kThreads;
    const int ci = task / kD;  // 0 q, 1 k, 2 v, 3 decay
    const int d = task % kD;
    if (ci < 3) {
      const int64_t chan = (int64_t)ci * P + hk + d;
      // Conv-cache bytes are partitioned over the cluster ranks.
      const bool mine = kSplit == 1 || (task * kSplit) / (3 * kD) == rank;
      __nv_bfloat16* win = conv_pool + (int64_t)conv_slot * cp_slot + chan;
      const float4 w = *reinterpret_cast<const float4*>(conv_w + chan * kW);
      const float bias = conv_b == nullptr ? 0.0f : conv_b[chan];
      float h0 = bf(c0[task_i]), h1 = bf(c1[task_i]), h2 = bf(c2[task_i]);
      if (mine) {
        win[0] = c1[task_i];
        win[cp_col] = c2[task_i];
      }
#pragma unroll
      for (int t = 0; t < kTMax; ++t) {
        if (t >= L) break;
        const __nv_bfloat16 xr = xs[task_i][t];
        if (mine) win[(int64_t)(t + 2) * cp_col] = xr;
        const float xv = bf(xr);
        float acc = bias;
        acc += h0 * w.x;
        acc += h1 * w.y;
        acc += h2 * w.z;
        acc += xv * w.w;
        acc = silu_front(acc);
        const float y = bf16_round(acc);  // conv output seam (bf16 out)
        if (ci == 0) {
          sQraw[t * kD + d] = y;
        } else if (ci == 1) {
          sKraw[t * kD + d] = y;
        } else {
          sV[t * kD + d] = bf16_round(y * sRatio[t]);  // RDA seam (bf16)
        }
        h0 = h1;
        h1 = h2;
        h2 = xv;
      }
    } else {
      const float exp_a = expf(A_log[h]);
      const float dtb = dt_bias[hk + d];
#pragma unroll
      for (int t = 0; t < kTMax; ++t) {
        if (t >= L) break;
        const float gr = bf(xs[task_i][t]) + dtb;
        sDec[t * kQkTok + qk_pos(d)] =
            exp_front(lower_bound * sigmoid_front(exp_a * gr));
      }
    }
  }
  __syncthreads();

  // ---- q/k L2 norm, one warp per token -------------------------------------
  if (warp < L) {
    const int t = warp;
    float q[4], k[4];
    float sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q[i] = sQraw[t * kD + lane * 4 + i];
      k[i] = sKraw[t * kD + lane * 4 + i];
      sq += q[i] * q[i];
      sk += k[i] * k[i];
    }
    sq = warp_sum(sq);
    sk = warp_sum(sk);
    const float nq = sqrtf(sq + 1e-6f);
    const float nk = sqrtf(sk + 1e-6f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sQ[t * kQkTok + qk_pos(lane * 4 + i)] = q[i] / nq * scale;
      sK[t * kQkTok + qk_pos(lane * 4 + i)] = k[i] / nk;
    }
  }
  __syncthreads();

  // ---- register-resident delta-rule recurrence over the token chain --------
#pragma unroll 1
  for (int t = 0; t < L; ++t) {
    const float* pq = sQ + t * kQkTok + k0;
    const float* pk = sK + t * kQkTok + k0;
    const float* pd = sDec + t * kQkTok + k0;
#if RAON_KDA_SPEC_QK_REGS
    float rq[kVec], rk[kVec], rd[kVec];
  #pragma unroll
    for (int j4 = 0; j4 < kVec / 4; ++j4) {
      const float4 a4 = *reinterpret_cast<const float4*>(pq + kJ4Stride * j4);
      const float4 b4 = *reinterpret_cast<const float4*>(pk + kJ4Stride * j4);
      const float4 c4 = *reinterpret_cast<const float4*>(pd + kJ4Stride * j4);
      rq[4 * j4 + 0] = a4.x;
      rq[4 * j4 + 1] = a4.y;
      rq[4 * j4 + 2] = a4.z;
      rq[4 * j4 + 3] = a4.w;
      rk[4 * j4 + 0] = b4.x;
      rk[4 * j4 + 1] = b4.y;
      rk[4 * j4 + 2] = b4.z;
      rk[4 * j4 + 3] = b4.w;
      rd[4 * j4 + 0] = c4.x;
      rd[4 * j4 + 1] = c4.y;
      rd[4 * j4 + 2] = c4.z;
      rd[4 * j4 + 3] = c4.w;
    }
  #define RQ(j) rq[j]
  #define RK(j) rk[j]
  #define RD(j) rd[j]
#else
  #define RQ(j) pq[((j) >> 2) * kJ4Stride + ((j) & 3)]
  #define RK(j) pk[((j) >> 2) * kJ4Stride + ((j) & 3)]
  #define RD(j) pd[((j) >> 2) * kJ4Stride + ((j) & 3)]
#endif
    const float bt = sBeta[t];
#pragma unroll
    for (int b = 0; b < kRowsPerThread; ++b) {
      const int lrow =
          warp * (kRowsPerThread * kRowGroups) + b * kRowGroups + rg;
      float shk = 0.0f;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        s[b][j] *= RD(j);
        shk += s[b][j] * RK(j);
      }
      shk = group_sum(shk);
      const float vn = (sV[t * kD + row0 + lrow] - shk) * bt;
      float shq = 0.0f;
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        s[b][j] += vn * RK(j);
        shq += s[b][j] * RQ(j);
      }
      shq = group_sum(shq);
      if (kg == 0)
        sO[t * kRows + lrow] =
            bf16_round(shq);  // recurrence output seam (bf16)
    }
#undef RQ
#undef RK
#undef RD
    int dst = 0;
#pragma unroll
    for (int u = 0; u < kTMax; ++u)
      dst = u == t ? dsts[u] : dst;  // register select
    if (dst > 0) {
      float* dp = state + (int64_t)dst * st_slot + (int64_t)hk * kD;
#pragma unroll
      for (int b = 0; b < kRowsPerThread; ++b) {
        const int row =
            row0 + warp * (kRowsPerThread * kRowGroups) + b * kRowGroups + rg;
        float4* p = reinterpret_cast<float4*>(dp + row * kD + k0);
#pragma unroll
        for (int j4 = 0; j4 < kVec / 4; ++j4) {
          const float4 v4 = make_float4(s[b][4 * j4 + 0], s[b][4 * j4 + 1],
                                        s[b][4 * j4 + 2], s[b][4 * j4 + 3]);
#if RAON_KDA_SPEC_STREAM_CS
          __stcs(p + j4 * kLanesK, v4);
#else
          p[j4 * kLanesK] = v4;
#endif
        }
      }
    }
  }
  __syncthreads();

  // ---- gated RMSNorm epilogue, one warp per token ---------------------------
  float ss = 0.0f;
  if (warp < L) {
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const float o = sO[warp * kRows + i * 32 + lane];
      ss += o * o;
    }
    ss = warp_sum(ss);
    if (lane == 0) sPart[warp] = ss;
  }
  if constexpr (kSplit > 1) {
    auto cluster = cooperative_groups::this_cluster();
    cluster.sync();
    if (warp < L) {
      ss = 0.0f;
#pragma unroll
      for (int r = 0; r < kSplit; ++r)
        ss += cluster.map_shared_rank(sPart, r)[warp];
    }
    cluster.sync();  // peers stop reading this rank's sPart before it exits
  }
  if (warp < L) {
    const int t = warp;
    const float rstd = 1.0f / sqrtf(ss / kD + eps);
    __nv_bfloat16* op = out + (int64_t)(bos + t) * out_row + hk + row0;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int v = i * 32 + lane;
      const float y = sO[t * kRows + v] * rstd * ow[i];
      op[v] = __float2bfloat16(y * sigmoidf_(og[i]));
    }
  }
}

}  // namespace raon_kda_spec

// --------------------------------------------------------------------------
// torch entry point
// --------------------------------------------------------------------------
void run(torch::Tensor x,          // [T, 3P] bf16, last dim contiguous
         torch::Tensor conv_w,     // [3P, 4] fp32 contiguous
         torch::Tensor conv_b,     // [3P] fp32 or empty
         torch::Tensor conv_pool,  // [slots, S, 3P] bf16
         torch::Tensor g,          // [T, H, 128] bf16 (row-strided ok)
         torch::Tensor A_log,      // [H] fp32
         torch::Tensor dt_bias,    // [P] fp32
         torch::Tensor beta,       // [T, H] bf16
         torch::Tensor ratio,      // [T, H] fp32 ratio / bf16|fp32 delta
         int64_t ratio_mode,
         torch::Tensor onorm_g,   // [T, H, 128] bf16
         torch::Tensor onorm_w,   // [128] fp32
         torch::Tensor state,     // [slots, H, 128, 128] fp32
         torch::Tensor idx,       // [>=N, Tcols] int32
         torch::Tensor qsl,       // [N + 1] int32
         torch::Tensor accepted,  // [>=N] int32
         torch::Tensor out,       // [T, H, 128] bf16
         int64_t N, double scale, double lower_bound, double eps,
         int64_t split) {
  using namespace raon_kda_spec;
  const int H = static_cast<int>(state.size(1));
  const int P = H * kD;
  TORCH_CHECK(x.stride(1) == 1 && conv_pool.stride(2) == 1,
              "inner dims must be contiguous");
  TORCH_CHECK(conv_w.is_contiguous() && conv_w.size(1) == kW,
              "conv_w [3P, 4] contiguous");
  TORCH_CHECK(g.stride(2) == 1 && g.stride(1) == kD,
              "g [T, H, 128] inner-contiguous");
  TORCH_CHECK(onorm_g.stride(2) == 1 && onorm_g.stride(1) == kD,
              "onorm_g inner-contiguous");
  TORCH_CHECK(out.stride(2) == 1 && out.stride(1) == kD,
              "out inner-contiguous");
  TORCH_CHECK(beta.stride(1) == 1 && ratio.stride(1) == 1,
              "beta/ratio head-contiguous");
  TORCH_CHECK(state.stride(3) == 1 && state.stride(2) == kD &&
                  state.stride(1) == kD * kD && state.stride(0) % 4 == 0,
              "state [slots, H, 128, 128] with 16B-aligned slots");
  TORCH_CHECK(idx.stride(1) == 1 && idx.dtype() == torch::kInt32,
              "idx int32 [N, T]");
  TORCH_CHECK(qsl.is_contiguous() && qsl.dtype() == torch::kInt32, "qsl int32");
  TORCH_CHECK(accepted.is_contiguous() && accepted.dtype() == torch::kInt32,
              "accepted int32");
  const int tcols = static_cast<int>(idx.size(1));
  TORCH_CHECK(tcols >= 1 && tcols <= 8, "1..8 verify positions supported, got ",
              tcols);
  TORCH_CHECK(conv_pool.size(1) >= tcols + kW - 2,
              "conv pool narrower than T+2");
  TORCH_CHECK(split == 1 || split == 2 || split == 4,
              "split must be 1, 2 or 4");
  if (N == 0) return;
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const float* bias_ptr = conv_b.numel() ? conv_b.data_ptr<float>() : nullptr;
  auto launch = [&](auto tmax_tag, auto split_tag) {
    constexpr int kTMax = decltype(tmax_tag)::value;
    constexpr int kSplit = decltype(split_tag)::value;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(H * kSplit, static_cast<unsigned>(N));
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = kSplit;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = kSplit > 1 ? 1 : 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, kda_spec_fused_kernel<kTMax, kSplit>,
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), x.stride(0),
        (const float*)conv_w.data_ptr<float>(), bias_ptr,
        reinterpret_cast<__nv_bfloat16*>(conv_pool.data_ptr()),
        conv_pool.stride(0), conv_pool.stride(1),
        reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()), g.stride(0),
        (const float*)A_log.data_ptr<float>(),
        (const float*)dt_bias.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr()), beta.stride(0),
        (const void*)ratio.data_ptr(), ratio.stride(0),
        static_cast<int>(ratio_mode),
        reinterpret_cast<const __nv_bfloat16*>(onorm_g.data_ptr()),
        onorm_g.stride(0), (const float*)onorm_w.data_ptr<float>(),
        state.data_ptr<float>(), state.stride(0),
        (const int*)idx.data_ptr<int>(), idx.stride(0),
        (const int*)qsl.data_ptr<int>(), (const int*)accepted.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), P,
        static_cast<float>(scale), static_cast<float>(lower_bound),
        static_cast<float>(eps)));
  };
  auto by_split = [&](auto tmax_tag) {
    if (split == 4) {
      launch(tmax_tag, std::integral_constant<int, 4>{});
    } else if (split == 2) {
      launch(tmax_tag, std::integral_constant<int, 2>{});
    } else {
      launch(tmax_tag, std::integral_constant<int, 1>{});
    }
  };
  if (tcols <= 2) {
    by_split(std::integral_constant<int, 2>{});
  } else if (tcols <= 4) {
    by_split(std::integral_constant<int, 4>{});
  } else {
    by_split(std::integral_constant<int, 8>{});
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run,
        "RAON KDA speculative-verify fused step (conv+RDA+recurrence+o_norm)");
}
