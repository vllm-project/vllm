// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HY V4 iHC boundary: optional post step, then pre (gate projection, gates and
// gated channel reduction) and the following RMSNorm.
//
//   residual_out[t, j, :] = bf16(post[t, j] * x[t, :] + residual[t, j, :])
//   mixes[t, n] = sum_k residual_out[t, k] * W[n, k]      (k over hc * hidden)
//   pre/post gates from mixes and the RMS of residual_out; hidden =
//   rms_norm(bf16(sum_j pre[t, j] * residual_out[t, j, :]), norm_weight)
//
// The two-kernel path follows TensorRT-LLM's fused hyper-connection kernels
// (cpp/tensorrt_llm/kernels/mhcKernels). Kernel 1 applies the post step and
// computes per-split sums of squares and the 8 gate projections on tensor
// cores (mma.sync m16n8k16, N = 8 = number of gates). The FP32 weight is split
// into three BF16 parts (hi + mid + lo), so the products equal FP32 products
// and the projection matches an FP32 GEMM up to summation order. Kernel 2 runs
// one CTA per token: it sums the splits in a fixed order (deterministic),
// computes the gates, the gated reduction and the RMSNorm. Both kernels use
// programmatic dependent launch.
//
// Rounding: the post step is one FMA, so the new residual is the exact result
// rounded to BF16 (the unfused Triton kernel contracts only some channels), and
// the BF16 residual feeds the statistics. The gated reduction then either stays
// in FP32 into the norm (round_before_norm = false, FMA accumulation, as the
// HPC library's fused kernel) or is rounded to BF16 first (round_before_norm =
// true, no FMA, as the unfused path). The norm rounds once.
//
// With fp8_out, the finish kernel also quantizes the BF16 hidden state to FP8
// with one UE8M0 scale per 128 channels, in the packed layout of
// per_token_group_quant_fp8_packed_for_deepgemm (bitwise equal to it).
//
// Up to 48 tokens with the norm fused and round_before_norm false (decode), a
// single kernel does all of it (ihc_small_kernel): a cluster of 8 or 16 CTAs
// splits hidden into slices, the gate projection runs on CUDA cores in FP32,
// and one exchange over the cluster's distributed shared memory gives the
// gates and both RMS normalizations. The two-kernel path is latency bound at
// these sizes.
//
// Above 1024 tokens with a post step, B200 / B300 run the statistics in
// ihc_stats_tma_kernel instead: TMA loads, the post step in shared memory and
// tcgen05 MMAs in separate warps keep the copy at HBM bandwidth. Its finish
// kernel issues the residual loads before the gates.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "torch_utils.h"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <tuple>

namespace vllm::hy_v4_ihc {

constexpr int kHc = 4;
constexpr int kNumMixes = 2 * kHc;        // MMA N
constexpr int kNumStats = kNumMixes + 1;  // mixes + sum of squares
constexpr int kTileM = 16;                // tokens per CTA (MMA M)
constexpr int kWarps = 8;
constexpr int kBlock = kWarps * 32;
constexpr int kChunk = 32;  // k elements per warp iteration (two k16 steps)
constexpr int kVec = 8;     // BF16 elements per 16-byte access
constexpr int kMaxRowIters = 4;
constexpr int kMaxHidden = kBlock * kVec * kMaxRowIters;
// Splits of the k range, largest first. Each warp's slice (4 * hidden /
// (splits * 8)) must be a whole number of 32-element chunks inside one hc
// channel, so the usable counts depend on the hidden size: 96 / 48 / 24 for
// 6144, 64 / 32 for 4096 and 8192, 80 / 40 / 20 for 5120, 56 / 28 for 7168.
// 1 fits every supported hidden size.
constexpr int kSplitChoices[] = {96, 80, 64, 56, 48, 40, 32, 28,
                                 24, 20, 16, 8,  4,  2,  1};
constexpr int kMaxSplits = 96;
constexpr int kStatGroup = 16;    // threads per statistic in the finish kernel
constexpr int kQuantGroup = 128;  // channels per FP8 scale
constexpr float kFp8Max = 448.f;
// Small-M path: a cluster of 8 or 16 CTAs splits hidden into slices; each CTA
// owns its slice of all hc channels for up to kMaxClusterTokens tokens.
constexpr int kSmallHidden = 6144;
constexpr int kMaxSmallTokens = 48;
constexpr int kMaxClusterTokens = 4;
constexpr int kSmallBlock = 384;
constexpr int kSmallWarps = kSmallBlock / 32;
constexpr int kMaxSliceVecs = kSmallBlock / kHc;  // one thread per 8 k elements
constexpr int kMaxTokenWarps = 3;  // warps per token in the second phase
constexpr int kNumGram = kHc * (kHc + 1) / 2;
constexpr int kSmallStats = kNumMixes + kNumGram;  // per token

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t const (&a)[4],
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void unpack8(uint4 raw, float (&f)[8]) {
  auto const* p = reinterpret_cast<__nv_bfloat162 const*>(&raw);
#pragma unroll
  for (int v = 0; v < 4; v++) {
    float2 const x = __bfloat1622float2(p[v]);
    f[2 * v] = x.x;
    f[2 * v + 1] = x.y;
  }
}

__device__ __forceinline__ uint4 pack8(float const (&f)[8]) {
  uint4 raw;
  auto* p = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
  for (int v = 0; v < 4; v++) {
    p[v] = __float22bfloat162_rn(make_float2(f[2 * v], f[2 * v + 1]));
  }
  return raw;
}

// Grid: ceil(M / 16) * kSplits; the splits of a token tile are adjacent. The
// split count is a template parameter: with a runtime count the post variant
// exceeds 64 registers and spills.
// A CTA owns 16 tokens and one k range; its 8 warps split that range, and each
// warp's slice stays inside one hc channel.
//
// MMA fragment layout: thread (g = lane / 4, q = lane % 4) owns the logical k
// positions {2q, 2q + 1, 2q + 8, 2q + 9} of a k16 step, in both A (tokens g and
// g + 8) and B (mix g). The MMA sums over k, so k may be permuted as long as A
// and B agree: mapping those positions to the contiguous elements [8q, 8q + 8)
// of a 32-element chunk (4 for each of two k16 steps) turns every fragment load
// into one 16-byte load per row.
template <bool kHasPost, int kSplits>
__launch_bounds__(kBlock, 4) __global__
    void ihc_stats_kernel(__nv_bfloat16 const* __restrict__ residual_in,
                          __nv_bfloat16 const* __restrict__ x,
                          float const* __restrict__ post,
                          float const* __restrict__ weight,
                          __nv_bfloat16* __restrict__ residual_out,
                          float* __restrict__ partials, int num_tokens,
                          int hidden) {
  cudaGridDependencySynchronize();
  int const tile = blockIdx.x / kSplits;
  int const split = blockIdx.x % kSplits;
  int const tid = threadIdx.x;
  int const warp = tid / 32;
  int const lane = tid % 32;
  int const g = lane / 4;
  int const q = lane % 4;
  int64_t const k_total = static_cast<int64_t>(kHc) * hidden;
  int64_t const warp_k = k_total / (kSplits * kWarps);
  int64_t const k_lo = (static_cast<int64_t>(split) * kWarps + warp) * warp_k;
  int const channel = static_cast<int>(k_lo / hidden);
  int const h_lo =
      static_cast<int>(k_lo - channel * static_cast<int64_t>(hidden));

  int const tok0 = tile * kTileM + g;
  int const tok1 = tok0 + 8;
  bool const ok0 = tok0 < num_tokens;
  bool const ok1 = tok1 < num_tokens;
  float gate0 = 0.f;
  float gate1 = 0.f;
  if constexpr (kHasPost) {
    gate0 = ok0 ? post[tok0 * kHc + channel] : 0.f;
    gate1 = ok1 ? post[tok1 * kHc + channel] : 0.f;
  }

  float acc[4] = {0.f, 0.f, 0.f, 0.f};  // tokens g / g + 8, mixes 2q / 2q + 1
  float sq0 = 0.f;
  float sq1 = 0.f;
  int64_t const row0 = tok0 * k_total + k_lo;
  int64_t const row1 = tok1 * k_total + k_lo;
  int64_t const x_row0 = static_cast<int64_t>(tok0) * hidden + h_lo;
  int64_t const x_row1 = static_cast<int64_t>(tok1) * hidden + h_lo;
  float const* w_row = weight + g * k_total + k_lo;
  uint4 const zero = make_uint4(0, 0, 0, 0);

#pragma unroll 2
  for (int64_t kk = 0; kk < warp_k; kk += kChunk) {
    int64_t const e = kk + q * 8;
    uint4 raw0 =
        ok0 ? *reinterpret_cast<uint4 const*>(residual_in + row0 + e) : zero;
    uint4 raw1 =
        ok1 ? *reinterpret_cast<uint4 const*>(residual_in + row1 + e) : zero;
    float r0[8];
    float r1[8];
    if constexpr (kHasPost) {
      float x0[8];
      float x1[8];
      unpack8(ok0 ? *reinterpret_cast<uint4 const*>(x + x_row0 + e) : zero, x0);
      unpack8(ok1 ? *reinterpret_cast<uint4 const*>(x + x_row1 + e) : zero, x1);
      unpack8(raw0, r0);
      unpack8(raw1, r1);
#pragma unroll
      for (int v = 0; v < 8; v++) {
        r0[v] = fmaf(gate0, x0[v], r0[v]);
        r1[v] = fmaf(gate1, x1[v], r1[v]);
      }
      raw0 = pack8(r0);
      raw1 = pack8(r1);
      if (ok0) {
        *reinterpret_cast<uint4*>(residual_out + row0 + e) = raw0;
      }
      if (ok1) {
        *reinterpret_cast<uint4*>(residual_out + row1 + e) = raw1;
      }
    }
    // The BF16-rounded residual feeds the statistics.
    unpack8(raw0, r0);
    unpack8(raw1, r1);
#pragma unroll
    for (int v = 0; v < 8; v++) {
      sq0 = fmaf(r0[v], r0[v], sq0);
      sq1 = fmaf(r1[v], r1[v], sq1);
    }
    // A fragments of the two k16 steps: elements [0, 4) and [4, 8).
    auto const* p0 = reinterpret_cast<uint32_t const*>(&raw0);
    auto const* p1 = reinterpret_cast<uint32_t const*>(&raw1);
    uint32_t const a_lo[4] = {p0[0], p1[0], p0[1], p1[1]};
    uint32_t const a_hi[4] = {p0[2], p1[2], p0[3], p1[3]};
    // B fragments: this thread's 8 weights of mix g as hi / mid / lo parts.
    float w[8];
    *reinterpret_cast<float4*>(&w[0]) =
        __ldg(reinterpret_cast<float4 const*>(w_row + e));
    *reinterpret_cast<float4*>(&w[4]) =
        __ldg(reinterpret_cast<float4 const*>(w_row + e + 4));
    uint32_t b[3][4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      __nv_bfloat162 const hi = __floats2bfloat162_rn(w[2 * i], w[2 * i + 1]);
      float2 const hi_f = __bfloat1622float2(hi);
      float const rest0 = w[2 * i] - hi_f.x;
      float const rest1 = w[2 * i + 1] - hi_f.y;
      __nv_bfloat162 const mid = __floats2bfloat162_rn(rest0, rest1);
      float2 const mid_f = __bfloat1622float2(mid);
      __nv_bfloat162 const lo =
          __floats2bfloat162_rn(rest0 - mid_f.x, rest1 - mid_f.y);
      b[0][i] = *reinterpret_cast<uint32_t const*>(&hi);
      b[1][i] = *reinterpret_cast<uint32_t const*>(&mid);
      b[2][i] = *reinterpret_cast<uint32_t const*>(&lo);
    }
#pragma unroll
    for (int part = 2; part >= 0; part--) {  // smallest part first
      mma_bf16(acc, a_lo, b[part][0], b[part][1]);
      mma_bf16(acc, a_hi, b[part][2], b[part][3]);
    }
  }

  // CTA reduction of the MMA accumulators and sums of squares.
  __shared__ float s_part[kWarps][kTileM][kNumStats];
  s_part[warp][g][2 * q] = acc[0];
  s_part[warp][g][2 * q + 1] = acc[1];
  s_part[warp][g + 8][2 * q] = acc[2];
  s_part[warp][g + 8][2 * q + 1] = acc[3];
  sq0 += __shfl_xor_sync(0xffffffff, sq0, 1);
  sq0 += __shfl_xor_sync(0xffffffff, sq0, 2);
  sq1 += __shfl_xor_sync(0xffffffff, sq1, 1);
  sq1 += __shfl_xor_sync(0xffffffff, sq1, 2);
  if (q == 0) {
    s_part[warp][g][kNumMixes] = sq0;
    s_part[warp][g + 8][kNumMixes] = sq1;
  }
  __syncthreads();
  if (tid < kTileM * kNumStats) {
    int const t = tid / kNumStats;
    int const n = tid % kNumStats;
    int const tok = tile * kTileM + t;
    if (tok < num_tokens) {
      float v = 0.f;
#pragma unroll
      for (int w_idx = 0; w_idx < kWarps; w_idx++) {
        v += s_part[w_idx][t][n];
      }
      partials[(static_cast<int64_t>(split) * num_tokens + tok) * kNumStats +
               n] = v;
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

// Quantizes this thread's 8 BF16 values; 16 consecutive threads hold one
// 128-channel group. Same arithmetic as
// per_token_group_quant_8bit_packed_register_kernel: scale = 2^ceil(log2(
// max(amax, eps) / 448)), values clamped to +-448 and rounded to nearest.
// The scale byte goes to [h / 512, tok] (int32 column-major, 4 per int32).
__device__ __forceinline__ void quantize8(uint4 raw, int lane, int64_t tok,
                                          int h, uint8_t* __restrict__ fp8_out,
                                          uint8_t* __restrict__ scale_out,
                                          int scale_stride, int hidden) {
  float y[kVec];
  unpack8(raw, y);
  float amax = 1e-10f;
#pragma unroll
  for (int v = 0; v < kVec; v++) {
    amax = fmaxf(amax, fabsf(y[v]));
  }
  unsigned const mask = 0xffffu << (lane & 16);
#pragma unroll
  for (int off = 8; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(mask, amax, off));
  }
  uint32_t const bits =
      __float_as_uint(fmaxf(__fdiv_rn(amax, kFp8Max), 1e-10f));
  uint32_t const exp = ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u);
  float const inv = __uint_as_float((254u - exp) << 23);  // exact 1 / scale
  uint2 q;
  auto* p = reinterpret_cast<__nv_fp8x2_storage_t*>(&q);
#pragma unroll
  for (int v = 0; v < kVec / 2; v++) {
    float2 const f =
        make_float2(fminf(fmaxf(y[2 * v] * inv, -kFp8Max), kFp8Max),
                    fminf(fmaxf(y[2 * v + 1] * inv, -kFp8Max), kFp8Max));
    p[v] = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
  }
  *reinterpret_cast<uint2*>(fp8_out + tok * hidden + h) = q;
  if ((lane & 15) == 0) {
    int const group = h / kQuantGroup;
    scale_out[((group / 4) * static_cast<int64_t>(scale_stride) + tok) * 4 +
              group % 4] = static_cast<uint8_t>(exp);
  }
}

// One CTA per token: gates, gated reduction and RMSNorm. Each thread keeps its
// slice of the reduced row in registers, so the hidden state is written once.
// With kPrefetch > 0, hidden == kPrefetch * kBlock * kVec and the residual
// loads are issued before the partials are read, so their latency overlaps
// the gate computation.
template <bool kFuseNorm, bool kRoundBeforeNorm, bool kFp8Out, int kPrefetch>
__launch_bounds__(kBlock, kPrefetch > 0 ? 4 : 1) __global__
    void ihc_finish_kernel(
        __nv_bfloat16 const* __restrict__ residual,
        float const* __restrict__ partials, float const* __restrict__ hc_scale,
        float const* __restrict__ hc_base,
        __nv_bfloat16 const* __restrict__ norm_weight,
        float* __restrict__ post_out, __nv_bfloat16* __restrict__ hidden_out,
        uint8_t* __restrict__ fp8_out, uint8_t* __restrict__ scale_out,
        int scale_stride, int num_tokens, int hidden, int num_splits,
        float magnitude, float hc_eps, float rms_eps, float var_eps) {
  cudaGridDependencySynchronize();
  int64_t const tok = blockIdx.x;
  int const tid = threadIdx.x;
  int const warp = tid / 32;
  int const lane = tid % 32;
  int64_t const k_total = static_cast<int64_t>(kHc) * hidden;
  __shared__ float s_stat[kNumStats];
  __shared__ float s_pre[kHc];
  __shared__ float s_red[kWarps];
  constexpr int kIters = kPrefetch > 0 ? kPrefetch : kMaxRowIters;
  uint4 raw[kPrefetch > 0 ? kPrefetch : 1][kHc];
  if constexpr (kPrefetch > 0) {
#pragma unroll
    for (int it = 0; it < kPrefetch; it++) {
#pragma unroll
      for (int j = 0; j < kHc; j++) {
        raw[it][j] = *reinterpret_cast<uint4 const*>(
            residual + tok * k_total + j * static_cast<int64_t>(hidden) +
            (it * kBlock + tid) * kVec);
      }
    }
  }
  // 16 threads per statistic: thread i of a group takes splits i, i + 16, ...
  // (all loads in flight at once), then a butterfly reduction within the
  // group. The order depends only on num_splits, so the result is
  // deterministic.
  // Whole warps take part, so the shuffles are uniform; the 10th group idles.
  if (warp < (kNumStats * kStatGroup + 31) / 32) {
    int const n = tid / kStatGroup;
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < kMaxSplits / kStatGroup; i++) {
      int const s = tid % kStatGroup + i * kStatGroup;
      if (n < kNumStats && s < num_splits) {
        v += partials[(static_cast<int64_t>(s) * num_tokens + tok) * kNumStats +
                      n];
      }
    }
#pragma unroll
    for (int off = kStatGroup / 2; off > 0; off >>= 1) {
      v += __shfl_xor_sync(0xffffffff, v, off);
    }
    if (n < kNumStats && tid % kStatGroup == 0) {
      s_stat[n] = v;
    }
  }
  __syncthreads();
  if (tid < kHc) {
    float const rstd =
        rsqrtf(s_stat[kNumMixes] / static_cast<float>(k_total) + rms_eps);
    float const pre = s_stat[tid] * rstd * hc_scale[0] + hc_base[tid];
    s_pre[tid] = 1.f / (1.f + expf(-pre)) + hc_eps;
    float const post =
        s_stat[kHc + tid] * rstd * hc_scale[1] + hc_base[kHc + tid];
    post_out[tok * kHc + tid] =
        magnitude * (1.f / (1.f + expf(-post))) + hc_eps;
  }
  __syncthreads();

  float row[kIters][kVec];
  float sumsq = 0.f;
#pragma unroll
  for (int it = 0; it < kIters; it++) {
    int const h = (it * kBlock + tid) * kVec;
    if (h >= hidden) {
      break;
    }
#pragma unroll
    for (int v = 0; v < kVec; v++) {
      row[it][v] = 0.f;
    }
#pragma unroll
    for (int j = 0; j < kHc; j++) {
      float r[kVec];
      if constexpr (kPrefetch > 0) {
        unpack8(raw[it][j], r);
      } else {
        unpack8(*reinterpret_cast<uint4 const*>(
                    residual + tok * k_total +
                    j * static_cast<int64_t>(hidden) + h),
                r);
      }
#pragma unroll
      for (int v = 0; v < kVec; v++) {
        row[it][v] = kRoundBeforeNorm
                         ? __fadd_rn(row[it][v], __fmul_rn(s_pre[j], r[v]))
                         : fmaf(s_pre[j], r[v], row[it][v]);
      }
    }
#pragma unroll
    for (int v = 0; v < kVec; v++) {
      if constexpr (kRoundBeforeNorm) {
        row[it][v] = __bfloat162float(__float2bfloat16_rn(row[it][v]));
      }
      sumsq = fmaf(row[it][v], row[it][v], sumsq);
    }
  }
  float rnorm = 1.f;
  if constexpr (kFuseNorm) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sumsq += __shfl_xor_sync(0xffffffff, sumsq, off);
    }
    if (lane == 0) {
      s_red[warp] = sumsq;
    }
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; w++) {
      total += s_red[w];
    }
    rnorm = rsqrtf(total / static_cast<float>(hidden) + var_eps);
  }
#pragma unroll
  for (int it = 0; it < kIters; it++) {
    int const h = (it * kBlock + tid) * kVec;
    if (h >= hidden) {
      break;
    }
    if constexpr (kFuseNorm) {
      float nw[kVec];
      unpack8(*reinterpret_cast<uint4 const*>(norm_weight + h), nw);
#pragma unroll
      for (int v = 0; v < kVec; v++) {
        row[it][v] = row[it][v] * rnorm * nw[v];
      }
    }
    uint4 const out = pack8(row[it]);
    *reinterpret_cast<uint4*>(hidden_out + tok * hidden + h) = out;
    if constexpr (kFp8Out) {
      quantize8(out, lane, tok, h, fp8_out, scale_out, scale_stride, hidden);
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

// Stats kernel for long prefills on SM100-family GPUs (B200, B300): TMA, the
// fifth-generation tensor cores (tcgen05) and warp specialization, so the loads
// never wait for the math. Persistent, one CTA per SM, 8 warps:
//   warp 0 (lane 0)  TMA producer: for each 128-channel chunk it loads the
//                    residual [64 tokens x 128], the x tile and the weight tile
//                    [24 x 128] into a 4-stage ring;
//   warps 4-7        post step in place in shared memory (r = bf16(post * x +
//                    r)), sum of squares of the BF16 result, then the epilogue;
//   warp 1 (lane 0)  tcgen05.mma M=64 N=24 K=16 on the updated tile, with the
//                    accumulators in tensor memory;
//   warp 2 (lane 0)  TMA store of the updated tile to residual_out.
// ihc_split_weight_kernel first splits the FP32 weight into BF16 hi / mid / lo
// rows (0-7 / 8-15 / 16-23), so one MMA computes all three parts and the
// projection matches an FP32 GEMM up to summation order. tcgen05 truncates its
// FP32 additions, so chunks go round-robin to kAcc accumulators to keep each
// accumulation chain short; the accumulator set is double-buffered per work
// unit. A work unit is (64-token tile, hc channel, 1 / ksplits of the hidden
// channels); it writes split (channel * ksplits + slice) of the partials.
namespace tma {
constexpr int kTileM = 64;
constexpr int kChunkK = 128;
constexpr int kBoxK = 64;  // 128-byte rows, 128-byte swizzle
constexpr int kMmaN = 3 * kNumMixes;
constexpr int kStages = 4;
constexpr int kAcc = 8;
constexpr int kTileBytes = kTileM * kChunkK * 2;
constexpr int kBoxBytes = kTileM * kBoxK * 2;
constexpr int kWBoxBytes = kMmaN * kBoxK * 2;
constexpr int kWStageBytes = 2 * kWBoxBytes;
constexpr int kStageTx = 2 * kTileBytes + kWStageBytes;
constexpr int kSmemX = kStages * kTileBytes;
constexpr int kSmemW = 2 * kStages * kTileBytes;
constexpr int kSmemBytes = kSmemW + kStages * kWStageBytes + 1024;  // + align
constexpr int kBufCols = kAcc * 32;
constexpr int kTmemCols = 2 * kBufCols;
constexpr int kMaxKSplits = kMaxSplits / kHc;
constexpr int kMinTokens = 1024;  // the two-kernel path is faster up to here
// tcgen05 instruction descriptor, kind::f16: D FP32, A and B BF16, K-major.
constexpr uint32_t kIdesc = (1u << 4) | (1u << 7) | (1u << 10) |
                            (static_cast<uint32_t>(kMmaN >> 3) << 17) |
                            (static_cast<uint32_t>(kTileM >> 4) << 24);
}  // namespace tma

// Splits the FP32 weight [8, K] into BF16 rows hi [0, 8), mid [8, 16) and lo
// [16, 24) with hi + mid + lo == weight (finite, normal-range values), the B
// operand of the TMA stats kernel.
// Four weights per thread.
__launch_bounds__(kBlock) __global__
    void ihc_split_weight_kernel(float const* __restrict__ weight,
                                 __nv_bfloat16* __restrict__ split,
                                 int64_t count) {
  cudaGridDependencySynchronize();
  int64_t const i =
      (static_cast<int64_t>(blockIdx.x) * kBlock + threadIdx.x) * 4;
  if (i < count) {
    float4 const w = *reinterpret_cast<float4 const*>(weight + i);
    float const v[4] = {w.x, w.y, w.z, w.w};
    __nv_bfloat16 parts[3][4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      __nv_bfloat16 const hi = __float2bfloat16_rn(v[j]);
      float const rest = v[j] - __bfloat162float(hi);
      __nv_bfloat16 const mid = __float2bfloat16_rn(rest);
      parts[0][j] = hi;
      parts[1][j] = mid;
      parts[2][j] = __float2bfloat16_rn(rest - __bfloat162float(mid));
    }
#pragma unroll
    for (int part = 0; part < 3; part++) {
      *reinterpret_cast<uint2*>(split + part * count + i) =
          *reinterpret_cast<uint2 const*>(parts[part]);
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

#if defined(__CUDA_ARCH__) &&                           \
    (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030) && \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) ||          \
     defined(__CUDA_ARCH_SPECIFIC__))
  #define HY_V4_IHC_TCGEN05 1
#endif

#ifdef HY_V4_IHC_TCGEN05
__device__ __forceinline__ uint32_t smem_u32(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar),
               "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar)
               : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t done;
  do {
    asm volatile(
        "{\n.reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
  } while (!done);
}

__device__ __forceinline__ void tma_load(uint32_t dst, CUtensorMap const* map,
                                         uint32_t bar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(bar), "r"(c0), "r"(c1)
      : "memory");
}

__device__ __forceinline__ void tma_store(CUtensorMap const* map, uint32_t src,
                                          int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], "
      "[%1];" ::"l"(reinterpret_cast<uint64_t>(map)),
      "r"(src), "r"(c0), "r"(c1)
      : "memory");
}

// K-major operand, 128-byte swizzle: 8-row groups 1024 bytes apart.
__device__ __forceinline__ uint64_t smem_desc(uint32_t addr) {
  return static_cast<uint64_t>((addr >> 4) & 0x3FFF) | (1ull << 16) |
         (64ull << 32) | (1ull << 46) | (2ull << 61);
}

__device__ __forceinline__ void umma(uint32_t d_tmem, uint64_t a, uint64_t b,
                                     uint32_t accumulate) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(
          d_tmem),
      "l"(a), "l"(b), "r"(tma::kIdesc), "r"(accumulate));
}

__device__ __forceinline__ void umma_commit(uint32_t bar) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
      "[%0];" ::"r"(bar)
      : "memory");
}

// M=64 accumulator: row r lives in TMEM lane (r / 16) * 32 + r % 16. Lanes
// 0-15 of the warp read column c, lanes 16-31 column c + 1.
__device__ __forceinline__ uint32_t tmem_ld(uint32_t taddr) {
  uint32_t v;
  asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x1.b32 {%0}, [%1], 1;"
               : "=r"(v)
               : "r"(taddr));
  return v;
}
#endif  // HY_V4_IHC_TCGEN05

__launch_bounds__(kBlock, 1) __global__
    void ihc_stats_tma_kernel(const __grid_constant__ CUtensorMap tm_res,
                              const __grid_constant__ CUtensorMap tm_x,
                              const __grid_constant__ CUtensorMap tm_w,
                              const __grid_constant__ CUtensorMap tm_out,
                              float const* __restrict__ post,
                              float* __restrict__ partials, int num_tokens,
                              int hidden, int ksplits) {
#ifdef HY_V4_IHC_TCGEN05
  using namespace tma;
  constexpr int kTileM = tma::kTileM;  // not the two-kernel path's tile
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full_bar[kStages];
  __shared__ __align__(8) uint64_t ready_bar[kStages];
  __shared__ __align__(8) uint64_t empty_bar[kStages];
  __shared__ __align__(8) uint64_t tfull_bar[2];
  __shared__ __align__(8) uint64_t tempty_bar[2];
  __shared__ uint32_t tmem_base;
  __shared__ float s_sq[2][2][kTileM];

  // Swizzle atoms need 1024-byte alignment.
  uint8_t* const smem =
      smem_raw + ((1024 - (smem_u32(smem_raw) & 1023)) & 1023);
  uint32_t const sbase = smem_u32(smem);
  int const warp = threadIdx.x / 32;
  int const lane = threadIdx.x % 32;
  if (threadIdx.x == 0) {
    for (int s = 0; s < kStages; s++) {
      mbar_init(smem_u32(&full_bar[s]), 1);
      mbar_init(smem_u32(&ready_bar[s]), 128);
      mbar_init(smem_u32(&empty_bar[s]), 2);
    }
    for (int b = 0; b < 2; b++) {
      mbar_init(smem_u32(&tfull_bar[b]), 1);
      mbar_init(smem_u32(&tempty_bar[b]), 128);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 1) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], "
        "%1;" ::"r"(smem_u32(&tmem_base)),
        "r"(kTmemCols));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  uint32_t const tmem = tmem_base;

  cudaGridDependencySynchronize();

  int const chunks = hidden / kChunkK / ksplits;  // per work unit
  int const per_tile = kHc * ksplits;
  int const units = (num_tokens + kTileM - 1) / kTileM * per_tile;
  int const used_acc = chunks < kAcc ? chunks : kAcc;

  if (warp == 0 && lane == 0) {
    uint32_t it = 0;
    for (int u = blockIdx.x; u < units; u += gridDim.x) {
      int const row0 = u / per_tile * kTileM;
      int const channel = u % per_tile / ksplits;
      int const c0 = u % ksplits * chunks;
      for (int c = c0; c < c0 + chunks; c++, it++) {
        int const s = it % kStages;
        mbar_wait(smem_u32(&empty_bar[s]), ((it / kStages) & 1) ^ 1);
        uint32_t const bar = smem_u32(&full_bar[s]);
        mbar_expect_tx(bar, kStageTx);
        int const xcol = c * kChunkK;
        int const rcol = channel * hidden + xcol;
        uint32_t const r = sbase + s * kTileBytes;
        uint32_t const x = sbase + kSmemX + s * kTileBytes;
        uint32_t const w = sbase + kSmemW + s * kWStageBytes;
        tma_load(r, &tm_res, bar, rcol, row0);
        tma_load(r + kBoxBytes, &tm_res, bar, rcol + kBoxK, row0);
        tma_load(x, &tm_x, bar, xcol, row0);
        tma_load(x + kBoxBytes, &tm_x, bar, xcol + kBoxK, row0);
        tma_load(w, &tm_w, bar, rcol, 0);
        tma_load(w + kWBoxBytes, &tm_w, bar, rcol + kBoxK, 0);
      }
    }
  } else if (warp == 1 && lane == 0) {
    uint32_t it = 0;
    uint32_t i = 0;
    for (int u = blockIdx.x; u < units; u += gridDim.x, i++) {
      int const buf = i & 1;
      mbar_wait(smem_u32(&tempty_bar[buf]), ((i >> 1) & 1) ^ 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      for (int c = 0; c < chunks; c++, it++) {
        int const s = it % kStages;
        mbar_wait(smem_u32(&ready_bar[s]), (it / kStages) & 1);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        uint32_t const a = sbase + s * kTileBytes;
        uint32_t const b = sbase + kSmemW + s * kWStageBytes;
        uint32_t const d = tmem + buf * kBufCols + c % kAcc * 32;
  #pragma unroll
        for (int h = 0; h < 2; h++) {
  #pragma unroll
          for (int k = 0; k < 4; k++) {
            umma(d, smem_desc(a + h * kBoxBytes + k * 32),
                 smem_desc(b + h * kWBoxBytes + k * 32),
                 c >= kAcc || h != 0 || k != 0);
          }
        }
        umma_commit(smem_u32(&empty_bar[s]));
      }
      umma_commit(smem_u32(&tfull_bar[buf]));
    }
  } else if (warp == 2 && lane == 0) {
    uint32_t it = 0;
    for (int u = blockIdx.x; u < units; u += gridDim.x) {
      int const row0 = u / per_tile * kTileM;
      int const channel = u % per_tile / ksplits;
      int const c0 = u % ksplits * chunks;
      for (int c = c0; c < c0 + chunks; c++, it++) {
        int const s = it % kStages;
        mbar_wait(smem_u32(&ready_bar[s]), (it / kStages) & 1);
        int const rcol = channel * hidden + c * kChunkK;
        uint32_t const r = sbase + s * kTileBytes;
        tma_store(&tm_out, r, rcol, row0);
        tma_store(&tm_out, r + kBoxBytes, rcol + kBoxK, row0);
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        mbar_arrive(smem_u32(&empty_bar[s]));
      }
    }
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
  } else if (warp >= 4) {
    // Thread t owns row t % 64 of the 64-channel half t / 64 of each chunk.
    int const t = threadIdx.x - 128;
    int const row = t % kTileM;
    int const half = t / kTileM;
    int const ew = warp - 4;  // TMEM lanes [32 * ew, 32 * ew + 16)
    uint32_t it = 0;
    uint32_t i = 0;
    for (int u = blockIdx.x; u < units; u += gridDim.x, i++) {
      int const tile = u / per_tile;
      int const channel = u % per_tile / ksplits;
      int const split = u % per_tile;
      int const tok = tile * kTileM + row;
      float const gate = tok < num_tokens ? post[tok * kHc + channel] : 0.f;
      // Four accumulators per chunk, then the chunk sums into the total.
      float sq_total = 0.f;
      for (int c = 0; c < chunks; c++, it++) {
        int const s = it % kStages;
        mbar_wait(smem_u32(&full_bar[s]), (it / kStages) & 1);
        uint8_t* const rb =
            smem + s * kTileBytes + half * kBoxBytes + row * 128;
        uint8_t const* const xb =
            smem + kSmemX + s * kTileBytes + half * kBoxBytes + row * 128;
        float sq[4] = {0.f, 0.f, 0.f, 0.f};
  #pragma unroll
        for (int j = 0; j < 8; j++) {
          int const off = (j ^ (row & 7)) << 4;  // 128-byte swizzle
          float r[kVec];
          float xv[kVec];
          unpack8(*reinterpret_cast<uint4 const*>(rb + off), r);
          unpack8(*reinterpret_cast<uint4 const*>(xb + off), xv);
  #pragma unroll
          for (int v = 0; v < kVec; v++) {
            r[v] = fmaf(gate, xv[v], r[v]);
          }
          uint4 const out = pack8(r);
          *reinterpret_cast<uint4*>(rb + off) = out;
          unpack8(out, r);
  #pragma unroll
          for (int v = 0; v < kVec; v++) {
            sq[v % 4] = fmaf(r[v], r[v], sq[v % 4]);
          }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbar_arrive(smem_u32(&ready_bar[s]));
        sq_total += (sq[0] + sq[1]) + (sq[2] + sq[3]);
      }
      int const buf = i & 1;
      s_sq[buf][half][row] = sq_total;
      asm volatile("bar.sync 1, 128;" ::: "memory");
      mbar_wait(smem_u32(&tfull_bar[buf]), (i >> 1) & 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      uint32_t const taddr =
          tmem + buf * kBufCols + (static_cast<uint32_t>(ew * 32) << 16);
      constexpr int kCols = kMmaN / 2;  // accumulator columns per lane
      float acc[kCols] = {};
      for (int a = 0; a < used_acc; a++) {
        uint32_t v[kCols];
  #pragma unroll
        for (int q = 0; q < kCols; q++) {
          v[q] = tmem_ld(taddr + a * 32 + 2 * q);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  #pragma unroll
        for (int q = 0; q < kCols; q++) {
          acc[q] += __uint_as_float(v[q]);
        }
      }
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      mbar_arrive(smem_u32(&tempty_bar[buf]));
      // The lane holds mixes 2 * q + lane / 16 (q < 4) of token row r16, as
      // hi / mid / lo parts in acc[q], acc[4 + q] and acc[8 + q].
      int const r16 = ew * 16 + (lane & 15);
      int const tok_out = tile * kTileM + r16;
      if (tok_out < num_tokens) {
        float* const out =
            partials +
            (static_cast<int64_t>(split) * num_tokens + tok_out) * kNumStats;
  #pragma unroll
        for (int q = 0; q < kHc; q++) {
          out[2 * q + lane / 16] = (acc[8 + q] + acc[4 + q]) + acc[q];
        }
        if (lane < 16) {
          out[kNumMixes] = s_sq[buf][0][r16] + s_sq[buf][1][r16];
        }
      }
    }
  }

  __syncthreads();
  if (warp == 1) {
    __syncwarp();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem),
        "r"(tma::kTmemCols));
  }
  cudaTriggerProgrammaticLaunchCompletion();
#endif  // HY_V4_IHC_TCGEN05
}

// Reduce-scatter over groups of kLanes lanes: afterwards lane l holds the
// group sums of elements [i * N / kLanes, (i + 1) * N / kLanes) in
// v[0, N / kLanes), i = l % kLanes. Each element is summed by the same
// butterfly whatever N is.
template <int N, int kLanes = 32>
__device__ __forceinline__ void warp_reduce_scatter(float (&v)[N], int lane) {
  static_assert(N % kLanes == 0);
#pragma unroll
  for (int o = kLanes / 2, half = N / 2; o > 0; o >>= 1, half >>= 1) {
    bool const upper = lane & o;
#pragma unroll
    for (int i = 0; i < half; i++) {
      float const send = upper ? v[i] : v[i + half];
      float const keep = upper ? v[i + half] : v[i];
      v[i] = keep + __shfl_xor_sync(0xffffffff, send, o);
    }
  }
}

__device__ __forceinline__ uint32_t smem_addr(void const* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Address of the same shared-memory location in CTA `rank` of the cluster.
__device__ __forceinline__ uint32_t cluster_addr(uint32_t addr, int rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(out)
               : "r"(addr), "r"(rank));
  return out;
}

// Stores v at a peer's shared-memory address and counts its bytes on the
// peer's mbarrier.
__device__ __forceinline__ void store_to_peer(uint32_t addr, float v,
                                              uint32_t bar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, "
      "[%2];" ::"r"(addr),
      "r"(__float_as_uint(v)), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint32_t bar) {
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta."
        "b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
        : "=r"(done)
        : "r"(bar)
        : "memory");
  }
}

// Grid: ceil(M / kTokens) clusters of kCluster CTAs; CTA rank r of a cluster
// owns hidden slice [r * slice, (r + 1) * slice) of all hc channels for the
// cluster's kTokens tokens.
//   1. thread (channel c, vector v): post step and its 8 x 8 block of the gate
//      projection in FP32 for every token. The weight block is loaded before
//      the dependency wait and stays in registers for all tokens.
//   2. thread (token t, vector v): the token's 4 channels from shared memory,
//      kept in registers, give the 10 terms of the 4 x 4 Gram matrix
//      G[j][k] = sum_h r_j[h] r_k[h]. Its trace is the sum of squares of the
//      residual, and sum_jk pre_j pre_k G[j][k] the sum of squares of the
//      gated reduction, so one exchange gives both RMS normalizations.
//   3. CTA partials (8 mixes + 10 Gram terms per token) are pushed to every
//      CTA of the cluster with st.async; each CTA sums them in rank order.
//   4. thread (token t, vector v): gates and norm scale from the totals, gated
//      reduction from registers, RMSNorm, FP8 quantization.
// The gated reduction stays in FP32 into the norm (round_before_norm false).
template <int kCluster, int kTokens, bool kHasPost, bool kFp8Out>
__launch_bounds__(kSmallBlock, 1) __global__ void ihc_small_kernel(
    __nv_bfloat16 const* __restrict__ residual_in,
    __nv_bfloat16 const* __restrict__ x, float const* __restrict__ post,
    float const* __restrict__ weight, float const* __restrict__ hc_scale,
    float const* __restrict__ hc_base,
    __nv_bfloat16 const* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ residual_out, float* __restrict__ post_out,
    __nv_bfloat16* __restrict__ hidden_out, uint8_t* __restrict__ fp8_out,
    uint8_t* __restrict__ scale_out, int scale_stride, int num_tokens,
    int hidden, float magnitude, float hc_eps, float rms_eps, float var_eps) {
  constexpr int kMixStats = kTokens * kNumMixes;
  constexpr int kMixPad = (kMixStats + 31) / 32 * 32;
  constexpr int kStats = kTokens * kSmallStats;
  int const rank =
      static_cast<int>(cooperative_groups::this_cluster().block_rank());
  int const tok_base = blockIdx.x / kCluster * kTokens;
  int const tid = threadIdx.x;
  int const warp = tid / 32;
  int const lane = tid % 32;
  int const slice = hidden / kCluster;
  int const vecs = slice / kVec;
  int const token_warps = (vecs + 31) / 32;
  int64_t const k_total = static_cast<int64_t>(kHc) * hidden;

  __shared__ __align__(16)
      __nv_bfloat16 s_res[kTokens][kHc][kMaxSliceVecs * kVec];
  __shared__ float s_mix_warp[kSmallWarps][kMixStats];
  __shared__ float s_gram_warp[kTokens][kMaxTokenWarps][kNumGram];
  __shared__ float s_recv[kCluster][kStats];  // [source rank][stat]
  __shared__ float s_total[kStats];
  __shared__ __align__(8) uint64_t s_bar;

  uint32_t const bar = smem_addr(&s_bar);
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
        "r"(kCluster * kStats * 4));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // Peers push only after every CTA's mbarrier is initialized (waited on
  // before the push).
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

  bool const act1 = tid < kHc * vecs;
  int const c = tid / vecs;
  int const v1 = tid - c * vecs;
  int const h1 = rank * slice + v1 * kVec;
  int const t2 = warp / token_warps;
  int const tw = warp - t2 * token_warps;
  int const v2 = tw * 32 + lane;
  bool const act2 = t2 < kTokens && v2 < vecs;
  int const h2 = rank * slice + v2 * kVec;

  // Static parameters, loaded before the dependency wait.
  float w[kNumMixes][kVec];
  if (act1) {
    float const* w_ptr = weight + static_cast<int64_t>(c) * hidden + h1;
#pragma unroll
    for (int n = 0; n < kNumMixes; n++) {
      *reinterpret_cast<float4*>(&w[n][0]) =
          __ldg(reinterpret_cast<float4 const*>(w_ptr + n * k_total));
      *reinterpret_cast<float4*>(&w[n][4]) =
          __ldg(reinterpret_cast<float4 const*>(w_ptr + n * k_total + 4));
    }
  } else {
#pragma unroll
    for (int n = 0; n < kNumMixes; n++) {
#pragma unroll
      for (int e = 0; e < kVec; e++) {
        w[n][e] = 0.f;
      }
    }
  }
  float pre_scale = 0.f;
  float post_scale = 0.f;
  float pre_base[kHc];
  float post_base = 0.f;
  if (act2 || tid < kTokens * kHc) {
    pre_scale = hc_scale[0];
    post_scale = hc_scale[1];
#pragma unroll
    for (int j = 0; j < kHc; j++) {
      pre_base[j] = hc_base[j];
    }
    post_base = hc_base[kHc + tid % kHc];
  }
  uint4 const zero = make_uint4(0, 0, 0, 0);
  uint4 const nw_raw =
      act2 ? __ldg(reinterpret_cast<uint4 const*>(norm_weight + h2)) : zero;

  cudaGridDependencySynchronize();

  uint4 raw[kTokens];
  uint4 x_raw[kTokens];
  float gate[kTokens];
#pragma unroll
  for (int t = 0; t < kTokens; t++) {
    int64_t const tok = tok_base + t;
    bool const ok = act1 && tok < num_tokens;
    raw[t] = ok ? *reinterpret_cast<uint4 const*>(residual_in +
                                                  (tok * kHc + c) * hidden + h1)
                : zero;
    if constexpr (kHasPost) {
      x_raw[t] =
          ok ? *reinterpret_cast<uint4 const*>(x + tok * hidden + h1) : zero;
      gate[t] = ok ? post[tok * kHc + c] : 0.f;
    }
  }

  float acc[kMixPad];
#pragma unroll
  for (int i = 0; i < kMixPad; i++) {
    acc[i] = 0.f;
  }
#pragma unroll
  for (int t = 0; t < kTokens; t++) {
    int64_t const tok = tok_base + t;
    float r[kVec];
    if constexpr (kHasPost) {
      float xf[kVec];
      unpack8(raw[t], r);
      unpack8(x_raw[t], xf);
#pragma unroll
      for (int e = 0; e < kVec; e++) {
        r[e] = fmaf(gate[t], xf[e], r[e]);
      }
      raw[t] = pack8(r);
      if (act1 && tok < num_tokens) {
        *reinterpret_cast<uint4*>(residual_out + (tok * kHc + c) * hidden +
                                  h1) = raw[t];
      }
    }
    if (act1) {
      *reinterpret_cast<uint4*>(&s_res[t][c][v1 * kVec]) = raw[t];
    }
    // The BF16-rounded residual feeds the statistics.
    unpack8(raw[t], r);
    float* a = &acc[t * kNumMixes];
#pragma unroll
    for (int e = 0; e < kVec; e++) {
#pragma unroll
      for (int n = 0; n < kNumMixes; n++) {
        a[n] = fmaf(r[e], w[n][e], a[n]);
      }
    }
  }
  warp_reduce_scatter(acc, lane);
#pragma unroll
  for (int i = 0; i < kMixPad / 32; i++) {
    int const idx = lane * (kMixPad / 32) + i;
    if (idx < kMixStats) {
      s_mix_warp[warp][idx] = acc[i];
    }
  }
  __syncthreads();

  float r4[kHc][kVec];
  float gram[32];
#pragma unroll
  for (int i = 0; i < 32; i++) {
    gram[i] = 0.f;
  }
  if (act2) {
#pragma unroll
    for (int j = 0; j < kHc; j++) {
      unpack8(*reinterpret_cast<uint4 const*>(&s_res[t2][j][v2 * kVec]), r4[j]);
    }
    int idx = 0;
#pragma unroll
    for (int j = 0; j < kHc; j++) {
#pragma unroll
      for (int k = j; k < kHc; k++, idx++) {
#pragma unroll
        for (int e = 0; e < kVec; e++) {
          gram[idx] = fmaf(r4[j][e], r4[k][e], gram[idx]);
        }
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < kHc; j++) {
#pragma unroll
      for (int e = 0; e < kVec; e++) {
        r4[j][e] = 0.f;
      }
    }
  }
  warp_reduce_scatter(gram, lane);
  if (lane < kNumGram && t2 < kTokens) {
    s_gram_warp[t2][tw][lane] = gram[0];
  }
  __syncthreads();

  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (tid < kStats) {
    int const t = tid / kSmallStats;
    int const n = tid % kSmallStats;
    float v = 0.f;
    if (n < kNumMixes) {
      for (int i = 0; i < blockDim.x / 32; i++) {
        v += s_mix_warp[i][t * kNumMixes + n];
      }
    } else {
      for (int i = 0; i < token_warps; i++) {
        v += s_gram_warp[t][i][n - kNumMixes];
      }
    }
    uint32_t const dst = smem_addr(&s_recv[rank][tid]);
#pragma unroll
    for (int r = 0; r < kCluster; r++) {
      store_to_peer(cluster_addr(dst, r), v, cluster_addr(bar, r));
    }
  }
  mbarrier_wait(bar);
  if (tid < kStats) {
    float v = 0.f;
#pragma unroll
    for (int r = 0; r < kCluster; r++) {
      v += s_recv[r][tid];
    }
    s_total[tid] = v;
  }
  __syncthreads();

  // Gram upper triangle: (0,0) (0,1) (0,2) (0,3) (1,1) (1,2) (1,3) (2,2)
  // (2,3) (3,3).
  if (rank == 0 && tid < kTokens * kHc) {
    int64_t const tok = tok_base + tid / kHc;
    if (tok < num_tokens) {
      float const* st = &s_total[tid / kHc * kSmallStats];
      float const* g = st + kNumMixes;
      float const rstd = rsqrtf(
          (g[0] + g[4] + g[7] + g[9]) / static_cast<float>(k_total) + rms_eps);
      float const pst = st[kHc + tid % kHc] * rstd * post_scale + post_base;
      post_out[tok * kHc + tid % kHc] =
          magnitude * (1.f / (1.f + expf(-pst))) + hc_eps;
    }
  }
  int64_t const tok2 = tok_base + t2;
  if (act2 && tok2 < num_tokens) {
    float const* st = &s_total[t2 * kSmallStats];
    float const* g = st + kNumMixes;
    float const rstd = rsqrtf(
        (g[0] + g[4] + g[7] + g[9]) / static_cast<float>(k_total) + rms_eps);
    float p[kHc];
#pragma unroll
    for (int j = 0; j < kHc; j++) {
      float const pre = st[j] * rstd * pre_scale + pre_base[j];
      p[j] = 1.f / (1.f + expf(-pre)) + hc_eps;
    }
    float const sumsq =
        p[0] * p[0] * g[0] + p[1] * p[1] * g[4] + p[2] * p[2] * g[7] +
        p[3] * p[3] * g[9] +
        2.f * (p[0] * p[1] * g[1] + p[0] * p[2] * g[2] + p[0] * p[3] * g[3] +
               p[1] * p[2] * g[5] + p[1] * p[3] * g[6] + p[2] * p[3] * g[8]);
    float const rnorm = rsqrtf(sumsq / static_cast<float>(hidden) + var_eps);
    float row[kVec];
    float nw[kVec];
    unpack8(nw_raw, nw);
#pragma unroll
    for (int e = 0; e < kVec; e++) {
      row[e] = 0.f;
#pragma unroll
      for (int j = 0; j < kHc; j++) {
        row[e] = fmaf(p[j], r4[j][e], row[e]);
      }
      row[e] = row[e] * rnorm * nw[e];
    }
    uint4 const out = pack8(row);
    *reinterpret_cast<uint4*>(hidden_out + tok2 * hidden + h2) = out;
    if constexpr (kFp8Out) {
      quantize8(out, lane, tok2, h2, fp8_out, scale_out, scale_stride, hidden);
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

__device__ __forceinline__ void cp_async16(void* dst, void const* src) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr(dst)),
      "l"(src)
      : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

// Waits until at most n committed groups are pending; n is a constant once
// the caller's loop is unrolled.
__device__ __forceinline__ void cp_async_wait(int n) {
  switch (n) {
    case 0:
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      break;
    case 1:
      asm volatile("cp.async.wait_group 1;" ::: "memory");
      break;
    case 2:
      asm volatile("cp.async.wait_group 2;" ::: "memory");
      break;
    case 3:
      asm volatile("cp.async.wait_group 3;" ::: "memory");
      break;
    case 4:
      asm volatile("cp.async.wait_group 4;" ::: "memory");
      break;
    case 5:
      asm volatile("cp.async.wait_group 5;" ::: "memory");
      break;
    case 6:
      asm volatile("cp.async.wait_group 6;" ::: "memory");
      break;
    default:
      asm volatile("cp.async.wait_group 7;" ::: "memory");
      break;
  }
}

// float2 elementwise ops: __ffma2_rn/__fmul2_rn are sm_100-family only.
__device__ __forceinline__ float2 ffma2_rn(float2 a, float2 b, float2 c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
  return __ffma2_rn(a, b, c);
#else
  return make_float2(__fmaf_rn(a.x, b.x, c.x), __fmaf_rn(a.y, b.y, c.y));
#endif
}

__device__ __forceinline__ float2 fmul2_rn(float2 a, float2 b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1100
  return __fmul2_rn(a, b);
#else
  return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
#endif
}

// Non-coherent loads the compiler keeps in place (volatile): issued before
// the dependency wait, they overlap the previous kernel.
__device__ __forceinline__ float2 ldg_early(float const* ptr) {
  float2 v;
  asm volatile("ld.global.nc.v2.f32 {%0, %1}, [%2];"
               : "=f"(v.x), "=f"(v.y)
               : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint4 ldg_early(void const* ptr) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}

// A fragment rows g and g + 8 of the gate projection: the BF16 high part of
// mix g's weights and the BF16 remainder (w - hi rounded), so the projection
// carries 16 mantissa bits of each weight.
__device__ __forceinline__ void split_weights(float2 lo_k, float2 hi_k,
                                              uint32_t (&a)[4]) {
  __nv_bfloat162 const h0 = __floats2bfloat162_rn(lo_k.x, lo_k.y);
  __nv_bfloat162 const h1 = __floats2bfloat162_rn(hi_k.x, hi_k.y);
  float2 const h0f = __bfloat1622float2(h0);
  float2 const h1f = __bfloat1622float2(h1);
  __nv_bfloat162 const r0 =
      __floats2bfloat162_rn(lo_k.x - h0f.x, lo_k.y - h0f.y);
  __nv_bfloat162 const r1 =
      __floats2bfloat162_rn(hi_k.x - h1f.x, hi_k.y - h1f.y);
  a[0] = *reinterpret_cast<uint32_t const*>(&h0);
  a[1] = *reinterpret_cast<uint32_t const*>(&r0);
  a[2] = *reinterpret_cast<uint32_t const*>(&h1);
  a[3] = *reinterpret_cast<uint32_t const*>(&r1);
}

// Cooperative kernel for 49 to 256 tokens (HY V4 hidden only).
// CTA (tile, s) owns tokens [kTile * tile, + kTile) and hidden columns
// [256 s, 256 (s + 1)) of all hc channels; the 24 CTAs of a tile cover the
// hidden size. The whole grid must be resident (the host checks occupancy).
//   1. Thread (slot, vector v) copies the 4 channels (and x) of token
//      8 i + slot of each group i into shared memory with cp.async, one
//      commit group per group; as a group lands it applies the post step and
//      sums the token's 10 Gram terms (one warp per token). The gate
//      projection of the tile runs on tensor cores from shared memory (A:
//      the weights as 8 mixes x {high, remainder} BF16 rows, held in
//      registers from before the dependency wait). The CTA's 18 statistics
//      per token go to `partials`, each float4 tagged with this call's epoch.
//   2. Each CTA polls its tile's 24 partials until all carry the epoch and
//      sums them in a fixed order. The epoch comes from a per-tile counter in
//      `tickets` that every call advances by 24, so the workspace persists
//      across calls and calls must not run concurrently on two streams.
//   3. Gates and norm scale, then thread (slot, v): gated reduction,
//      RMSNorm, FP8 quantization.
struct CoopLayout {
  static constexpr int kSliceH = 256;
  static constexpr int kSlices = kSmallHidden / kSliceH;
  static constexpr int kVecs = kSliceH / kVec;  // 32: one warp per token
  static constexpr int kThreads = 256;
  static constexpr int kWarps = kThreads / 32;
  static constexpr int kSlots = kWarps;  // tokens per group
  static constexpr int kStepsW = kHc * kSliceH / 16 / kWarps;  // k16 / warp
  static constexpr int kChStride = kSliceH + 16;
  static constexpr int kTokStride = kHc * kChStride + 8;
  // float4 partials per CTA whatever kTile, so a (tile, slice) slot only ever
  // holds tags from that tile's own counter.
  static constexpr int kPartStride = 24 * kSmallStats / 3;
};

template <int kTile, bool kHasPost>
constexpr int coop_smem() {
  using L = CoopLayout;
  return kTile * (L::kTokStride + (kHasPost ? L::kSliceH : 0)) * 2 +
         L::kWarps * kTile * kSmallStats * 4;
}

template <int kTile, bool kHasPost, bool kFp8Out>
__launch_bounds__(CoopLayout::kThreads, 2) __global__ void ihc_coop_kernel(
    __nv_bfloat16 const* __restrict__ residual_in,
    __nv_bfloat16 const* __restrict__ x, float const* __restrict__ post,
    float const* __restrict__ weight, float const* __restrict__ hc_scale,
    float const* __restrict__ hc_base,
    __nv_bfloat16 const* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ residual_out, float* __restrict__ post_out,
    __nv_bfloat16* __restrict__ hidden_out, uint8_t* __restrict__ fp8_out,
    uint8_t* __restrict__ scale_out, int scale_stride,
    float* __restrict__ partials, unsigned long long* __restrict__ tickets,
    int num_tokens, int hidden, float magnitude, float hc_eps, float rms_eps,
    float var_eps) {
  using L = CoopLayout;
  constexpr int kSliceH = L::kSliceH;
  constexpr int kSlices = L::kSlices;
  constexpr int kSlots = L::kSlots;
  constexpr int kThreads = L::kThreads;
  constexpr int kWarpsC = L::kWarps;
  constexpr int kStepsW = L::kStepsW;
  constexpr int kChStride = L::kChStride;
  constexpr int kTokStride = L::kTokStride;
  constexpr int kGroups = kTile / kSlots;
  constexpr int kStats = kTile * kSmallStats;
  static_assert(kTile % kSlots == 0 && kGroups <= 7 && kStepsW % 2 == 0);
  int const tile = blockIdx.x / kSlices;
  int const sl = blockIdx.x % kSlices;
  int const tok_base = tile * kTile;
  int const tok_end = min(num_tokens, tok_base + kTile);
  int const tid = threadIdx.x;
  int const warp = tid / 32;
  int const lane = tid % 32;
  int const g = lane / 4;
  int const q = lane % 4;
  int const slot = warp;
  int const v = lane;
  int const h = sl * kSliceH + v * kVec;
  int64_t const k_total = static_cast<int64_t>(kHc) * hidden;

  extern __shared__ __align__(16) uint8_t s_dyn[];
  auto* s_res = reinterpret_cast<__nv_bfloat16*>(s_dyn);  // [t][j][kSliceH]
  auto* s_x = s_res + kTile * kTokStride;                 // [t][kSliceH]
  // Per-warp partial statistics [warp][t][stat].
  auto* s_wpart =
      reinterpret_cast<float*>(s_x + (kHasPost ? kTile * kSliceH : 0));
  __shared__ __align__(16) float s_post[kTile][kHc];
  __shared__ __align__(16) float s_total[kStats];
  __shared__ float s_gate[kTile][kHc];  // pre gates, then times rnorm
  __shared__ float s_hc[2 + 2 * kHc];

  // Static parameters, before the dependency wait. A fragment of k16 step i
  // of this warp: k elements {2q, 2q + 1} and {2q + 8, 2q + 9} of mix g.
  float2 w_raw[kStepsW][2];
#pragma unroll
  for (int i = 0; i < kStepsW; i++) {
    int const kc = (warp * kStepsW + i) * 16;
    float const* w_ptr = weight + g * k_total +
                         static_cast<int64_t>(kc / kSliceH) * hidden +
                         sl * kSliceH + kc % kSliceH + 2 * q;
    w_raw[i][0] = ldg_early(w_ptr);
    w_raw[i][1] = ldg_early(w_ptr + 8);
  }
  uint4 const nw_raw = ldg_early(norm_weight + h);
  if (tid < 2) {
    s_hc[tid] = hc_scale[tid];
  } else if (tid < 2 + 2 * kHc) {
    s_hc[tid] = hc_base[tid - 2];
  }

  cudaGridDependencySynchronize();
  // The dependent kernel still waits for this grid to complete in its
  // dependency wait; launching it now hides its launch latency.
  cudaTriggerProgrammaticLaunchCompletion();
  // This call's epoch for the tile: every call adds kSlices to its tiles'
  // counters, and calls on a stream do not overlap past the dependency wait.
  __shared__ unsigned s_epoch;
  if (tid == 0) {
    s_epoch =
        static_cast<unsigned>(atomicAdd(&tickets[tile], 1ull) / kSlices) + 1u;
  }

#pragma unroll
  for (int gi = 0; gi < kGroups; gi++) {
    int const t = gi * kSlots + slot;
    int64_t const tok = tok_base + t;
    if (tok < tok_end) {
#pragma unroll
      for (int j = 0; j < kHc; j++) {
        cp_async16(&s_res[t * kTokStride + j * kChStride + v * kVec],
                   residual_in + (tok * kHc + j) * hidden + h);
      }
      if constexpr (kHasPost) {
        cp_async16(&s_x[t * kSliceH + v * kVec], x + tok * hidden + h);
      }
    }
    if constexpr (kHasPost) {
      if (gi == 0 && tid < kTile && tok_base + tid < tok_end) {
        cp_async16(&s_post[tid][0], post + (tok_base + tid) * kHc);
      }
    }
    cp_async_commit();
  }

  uint32_t const res_base = smem_addr(s_res);
  int const m = lane / 8;
  constexpr int kTri[kHc] = {0, 4, 7, 9};
  uint32_t a[kStepsW][4];
#pragma unroll
  for (int gi = 0; gi < kGroups; gi++) {
    int const t = gi * kSlots + slot;
    int64_t const tok = tok_base + t;
    cp_async_wait(kGroups - 1 - gi);
    if (kHasPost && gi == 0) {
      __syncthreads();  // s_post
    }
    // Post step and the token's Gram terms over this CTA's columns: warp
    // `slot` owns token t.
    float2 gram[kNumGram];
#pragma unroll
    for (int i = 0; i < kNumGram; i++) {
      gram[i] = make_float2(0.f, 0.f);
    }
    if (tok < tok_end) {
      float2 r[kHc][kVec / 2];
      [[maybe_unused]] uint4 xr;
      if constexpr (kHasPost) {
        xr = *reinterpret_cast<uint4 const*>(&s_x[t * kSliceH + v * kVec]);
      }
#pragma unroll
      for (int j = 0; j < kHc; j++) {
        auto* sp = reinterpret_cast<uint4*>(
            &s_res[t * kTokStride + j * kChStride + v * kVec]);
        uint4 raw = *sp;
        auto* rp = reinterpret_cast<__nv_bfloat162*>(&raw);
        if constexpr (kHasPost) {
          auto const* xp = reinterpret_cast<__nv_bfloat162 const*>(&xr);
          float const gate = s_post[t][j];
          float2 const gate2 = make_float2(gate, gate);
#pragma unroll
          for (int e = 0; e < kVec / 2; e++) {
            rp[e] = __float22bfloat162_rn(ffma2_rn(
                gate2, __bfloat1622float2(xp[e]), __bfloat1622float2(rp[e])));
          }
          *sp = raw;
          *reinterpret_cast<uint4*>(residual_out + (tok * kHc + j) * hidden +
                                    h) = raw;
        }
        // The BF16-rounded residual feeds the statistics.
#pragma unroll
        for (int e = 0; e < kVec / 2; e++) {
          r[j][e] = __bfloat1622float2(rp[e]);
        }
      }
#pragma unroll
      for (int j = 0; j < kHc; j++) {
#pragma unroll
        for (int kk = j; kk < kHc; kk++) {
#pragma unroll
          for (int e = 0; e < kVec / 2; e++) {
            gram[kTri[j] + kk - j] =
                ffma2_rn(r[j][e], r[kk][e], gram[kTri[j] + kk - j]);
          }
        }
      }
    }
    float gsum[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
      gsum[i] = i < kNumGram ? gram[i].x + gram[i].y : 0.f;
    }
    warp_reduce_scatter<16, 16>(gsum, lane);
    gsum[0] += __shfl_xor_sync(0xffffffff, gsum[0], 16);
    if (lane < kNumGram) {
      s_wpart[(warp * kTile + t) * kSmallStats + kNumMixes + lane] = gsum[0];
    }
    __syncthreads();
    // Gate projection of this group's 8 tokens on tensor cores while later
    // groups are in flight.
    if (gi == 0) {
#pragma unroll
      for (int i = 0; i < kStepsW; i++) {
        split_weights(w_raw[i][0], w_raw[i][1], a[i]);
      }
    }
    int const tm = gi * kSlots + lane % 8;
    float acc[2][4] = {};
#pragma unroll
    for (int i = 0; i < kStepsW; i += 2) {
      int const kc = (warp * kStepsW + i + m / 2) * 16 + (m % 2) * 8;
      uint32_t b[4];
      ldmatrix_x4(b, res_base + 2 * (tm * kTokStride +
                                     kc / kSliceH * kChStride + kc % kSliceH));
      mma_bf16(acc[0], a[i], b[0], b[1]);
      mma_bf16(acc[1], a[i + 1], b[2], b[3]);
    }
    // Rows g / g + 8: mix g's high part and remainder; columns: tokens 2q,
    // 2q + 1 of the group.
#pragma unroll
    for (int c = 0; c < 2; c++) {
      s_wpart[(warp * kTile + gi * kSlots + 2 * q + c) * kSmallStats + g] =
          (acc[0][c] + acc[1][c]) + (acc[0][2 + c] + acc[1][2 + c]);
    }
  }
  __syncthreads();
  // Partials as float4 {3 statistics, epoch tag}; a 16-byte store lands
  // whole, so a reader that sees the tag sees the statistics.
  static_assert(kStats % 3 == 0);
  constexpr int kChunks = kStats / 3;
  unsigned const epoch = s_epoch;
  static_assert(kChunks <= L::kPartStride);
  float4* my_part = reinterpret_cast<float4*>(partials) +
                    static_cast<int64_t>(blockIdx.x) * L::kPartStride;
  for (int c = tid; c < kChunks; c += kThreads) {
    float val[3];
#pragma unroll
    for (int u = 0; u < 3; u++) {
      int const i = 3 * c + u;
      int const t = i / kSmallStats;
      int const n = i % kSmallStats;
      if (n < kNumMixes) {
        val[u] = 0.f;
#pragma unroll
        for (int wi = 0; wi < kWarpsC; wi++) {
          val[u] += s_wpart[wi * kStats + i];
        }
      } else {
        val[u] = s_wpart[(t % kSlots) * kStats + i];  // warp t % kSlots
      }
    }
    float4 const out =
        make_float4(val[0], val[1], val[2], __uint_as_float(epoch));
    asm volatile("st.relaxed.gpu.global.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(
                     &my_part[c]),
                 "f"(out.x), "f"(out.y), "f"(out.z), "f"(out.w)
                 : "memory");
  }

  auto const* tile_part = reinterpret_cast<float4 const*>(partials) +
                          static_cast<int64_t>(tile) * kSlices * L::kPartStride;
  for (int c = tid; c < kChunks; c += kThreads) {
    // All slices' loads in flight at once.
    float4 p4[kSlices];
    bool ready = false;
    while (!ready) {
      ready = true;
#pragma unroll
      for (int sl2 = 0; sl2 < kSlices; sl2++) {
        asm volatile("ld.relaxed.gpu.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                     : "=f"(p4[sl2].x), "=f"(p4[sl2].y), "=f"(p4[sl2].z),
                       "=f"(p4[sl2].w)
                     : "l"(&tile_part[sl2 * L::kPartStride + c])
                     : "memory");
      }
#pragma unroll
      for (int sl2 = 0; sl2 < kSlices; sl2++) {
        ready &= __float_as_uint(p4[sl2].w) == epoch;
      }
    }
    float sum[3] = {0.f, 0.f, 0.f};
#pragma unroll
    for (int sl2 = 0; sl2 < kSlices; sl2++) {
      sum[0] += p4[sl2].x;
      sum[1] += p4[sl2].y;
      sum[2] += p4[sl2].z;
    }
#pragma unroll
    for (int u = 0; u < 3; u++) {
      s_total[3 * c + u] = sum[u];
    }
  }
  __syncthreads();

  // Gates: thread (t, n), n < 4 pre and n >= 4 post. Gram upper triangle:
  // (0,0) (0,1) (0,2) (0,3) (1,1) (1,2) (1,3) (2,2) (2,3) (3,3).
  if (tid < kTile * 2 * kHc) {
    int const t = tid / (2 * kHc);
    int const n = tid % (2 * kHc);
    float const* st = &s_total[t * kSmallStats];
    float const* gm = st + kNumMixes;
    float const rstd =
        rsqrtf((gm[0] + gm[4] + gm[7] + gm[9]) / static_cast<float>(k_total) +
               rms_eps);
    float const z = st[n] * rstd * s_hc[n / kHc] + s_hc[2 + n];
    float const sig = 1.f / (1.f + expf(-z));
    if (n < kHc) {
      s_gate[t][n] = sig + hc_eps;
    } else if (sl == 0 && tok_base + t < tok_end) {
      post_out[(tok_base + t) * kHc + n - kHc] = magnitude * sig + hc_eps;
    }
  }
  __syncthreads();
  float scaled = 0.f;
  if (tid < kTile * kHc) {
    int const t = tid / kHc;
    float const* gm = &s_total[t * kSmallStats + kNumMixes];
    float const* pg = s_gate[t];
    float const sumsq = pg[0] * pg[0] * gm[0] + pg[1] * pg[1] * gm[4] +
                        pg[2] * pg[2] * gm[7] + pg[3] * pg[3] * gm[9] +
                        2.f * (pg[0] * pg[1] * gm[1] + pg[0] * pg[2] * gm[2] +
                               pg[0] * pg[3] * gm[3] + pg[1] * pg[2] * gm[5] +
                               pg[1] * pg[3] * gm[6] + pg[2] * pg[3] * gm[8]);
    float const rnorm =
        rsqrtf(fmaxf(sumsq, 0.f) / static_cast<float>(hidden) + var_eps);
    scaled = pg[tid % kHc] * rnorm;
  }
  __syncthreads();
  if (tid < kTile * kHc) {
    s_gate[tid / kHc][tid % kHc] = scaled;
  }
  __syncthreads();

  float nw[kVec];
  unpack8(nw_raw, nw);
#pragma unroll
  for (int gi = 0; gi < kGroups; gi++) {
    int const t = gi * kSlots + slot;
    int64_t const tok = tok_base + t;
    if (tok < tok_end) {
      float2 row[kVec / 2];
#pragma unroll
      for (int e = 0; e < kVec / 2; e++) {
        row[e] = make_float2(0.f, 0.f);
      }
#pragma unroll
      for (int jj = 0; jj < kHc; jj++) {
        uint4 const raw = *reinterpret_cast<uint4 const*>(
            &s_res[t * kTokStride + jj * kChStride + v * kVec]);
        auto const* rp = reinterpret_cast<__nv_bfloat162 const*>(&raw);
        float const p = s_gate[t][jj];
        float2 const p2 = make_float2(p, p);
#pragma unroll
        for (int e = 0; e < kVec / 2; e++) {
          row[e] = ffma2_rn(p2, __bfloat1622float2(rp[e]), row[e]);
        }
      }
      uint4 out;
      auto* op = reinterpret_cast<__nv_bfloat162*>(&out);
#pragma unroll
      for (int e = 0; e < kVec / 2; e++) {
        op[e] = __float22bfloat162_rn(
            fmul2_rn(row[e], make_float2(nw[2 * e], nw[2 * e + 1])));
      }
      *reinterpret_cast<uint4*>(hidden_out + tok * hidden + h) = out;
      if constexpr (kFp8Out) {
        quantize8(out, lane, tok, h, fp8_out, scale_out, scale_stride, hidden);
      }
    }
  }
}

using StatsKernel = void (*)(__nv_bfloat16 const*, __nv_bfloat16 const*,
                             float const*, float const*, __nv_bfloat16*, float*,
                             int, int);

template <int kSplits>
StatsKernel stats_kernel(bool has_post) {
  return has_post ? ihc_stats_kernel<true, kSplits>
                  : ihc_stats_kernel<false, kSplits>;
}

inline StatsKernel pick_stats_kernel(int splits, bool has_post) {
  switch (splits) {
    case 96:
      return stats_kernel<96>(has_post);
    case 80:
      return stats_kernel<80>(has_post);
    case 64:
      return stats_kernel<64>(has_post);
    case 56:
      return stats_kernel<56>(has_post);
    case 48:
      return stats_kernel<48>(has_post);
    case 40:
      return stats_kernel<40>(has_post);
    case 32:
      return stats_kernel<32>(has_post);
    case 28:
      return stats_kernel<28>(has_post);
    case 24:
      return stats_kernel<24>(has_post);
    case 20:
      return stats_kernel<20>(has_post);
    case 16:
      return stats_kernel<16>(has_post);
    case 8:
      return stats_kernel<8>(has_post);
    case 4:
      return stats_kernel<4>(has_post);
    case 2:
      return stats_kernel<2>(has_post);
    default:
      return stats_kernel<1>(has_post);
  }
}

using FinishKernel = void (*)(__nv_bfloat16 const*, float const*, float const*,
                              float const*, __nv_bfloat16 const*, float*,
                              __nv_bfloat16*, uint8_t*, uint8_t*, int, int, int,
                              int, float, float, float, float);

template <bool kFp8Out, int kPrefetch = 0>
inline FinishKernel pick_finish_kernel(bool fuse_norm, bool round_before_norm) {
  if (fuse_norm) {
    return round_before_norm
               ? ihc_finish_kernel<true, true, kFp8Out, kPrefetch>
               : ihc_finish_kernel<true, false, kFp8Out, kPrefetch>;
  }
  return round_before_norm
             ? ihc_finish_kernel<false, true, kFp8Out, kPrefetch>
             : ihc_finish_kernel<false, false, kFp8Out, kPrefetch>;
}

inline bool split_fits(int64_t hidden, int splits) {
  int64_t const k_total = kHc * hidden;
  if (k_total % (static_cast<int64_t>(splits) * kWarps * kChunk) != 0) {
    return false;
  }
  // Each warp's k slice must stay inside one hc channel.
  return hidden % (k_total / (splits * kWarps)) == 0;
}

// Enough CTAs to fill the GPU at small token counts; fewer splits (less
// partial traffic) once the token tiles alone fill it. Tuned on GB300 at
// hidden 6144.
inline int choose_splits(int64_t num_tokens, int64_t hidden) {
  int const want = num_tokens <= 16     ? 96
                   : num_tokens <= 128  ? 48
                   : num_tokens <= 512  ? 24
                   : num_tokens <= 1024 ? 16
                   : num_tokens <= 8192 ? 8
                                        : 4;
  for (int splits : kSplitChoices) {
    if (splits <= want && split_fits(hidden, splits)) {
      return splits;
    }
  }
  return 0;
}

inline bool aligned16(void const* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
}

// Launch with programmatic dependent launch: the kernel's prologue may overlap
// the previous kernel's tail; it waits in cudaGridDependencySynchronize.
template <typename Kernel, typename... Args>
void launch_pdl(Kernel kernel, int grid, int block, int cluster, int smem,
                cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(grid);
  config.blockDim = dim3(block);
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  attrs[1].id = cudaLaunchAttributeClusterDimension;
  attrs[1].val.clusterDim.x = cluster;
  attrs[1].val.clusterDim.y = 1;
  attrs[1].val.clusterDim.z = 1;
  config.attrs = attrs;
  config.numAttrs = cluster > 1 ? 2 : 1;
  cudaError_t const err = cudaLaunchKernelEx(&config, kernel, args...);
  STD_TORCH_CHECK(err == cudaSuccess, "hy_v4_ihc_boundary: launch failed: ",
                  cudaGetErrorString(err));
}

using SmallKernel = void (*)(__nv_bfloat16 const*, __nv_bfloat16 const*,
                             float const*, float const*, float const*,
                             float const*, __nv_bfloat16 const*, __nv_bfloat16*,
                             float*, __nv_bfloat16*, uint8_t*, uint8_t*, int,
                             int, int, float, float, float, float);

template <int kCluster, int kTokens>
SmallKernel small_kernel(bool has_post, bool fp8_out) {
  if (has_post) {
    return fp8_out ? ihc_small_kernel<kCluster, kTokens, true, true>
                   : ihc_small_kernel<kCluster, kTokens, true, false>;
  }
  return fp8_out ? ihc_small_kernel<kCluster, kTokens, false, true>
                 : ihc_small_kernel<kCluster, kTokens, false, false>;
}

template <int kCluster>
SmallKernel small_kernel(int tokens, bool has_post, bool fp8_out) {
  switch (tokens) {
    case 1:
      return small_kernel<kCluster, 1>(has_post, fp8_out);
    case 2:
      return small_kernel<kCluster, 2>(has_post, fp8_out);
    case 3:
      return small_kernel<kCluster, 3>(has_post, fp8_out);
    default:
      return small_kernel<kCluster, 4>(has_post, fp8_out);
  }
}

struct SmallConfig {
  int cluster;  // CTAs per cluster, 8 or 16
  int tokens;   // tokens per cluster
};

inline SmallKernel pick_small_kernel(SmallConfig cfg, bool has_post,
                                     bool fp8_out) {
  return cfg.cluster == 16 ? small_kernel<16>(cfg.tokens, has_post, fp8_out)
                           : small_kernel<8>(cfg.tokens, has_post, fp8_out);
}

inline int small_threads(int64_t hidden, SmallConfig cfg) {
  int const vecs = static_cast<int>(hidden / cfg.cluster / kVec);
  return std::max((kHc * vecs + 31) / 32 * 32,
                  cfg.tokens * ((vecs + 31) / 32) * 32);
}

// The configurations are tuned for HY V4's hidden size; smaller ones (HY V4
// Flash) split into narrower slices with fewer threads. Slices stay whole FP8
// scale groups, and fp8_out must not change the configuration (it would change
// the BF16 result's summation order).
inline bool small_fits(int64_t hidden, SmallConfig cfg) {
  return hidden <= kSmallHidden && hidden % (cfg.cluster * kQuantGroup) == 0 &&
         small_threads(hidden, cfg) <= kSmallBlock;
}

// Clusters of a small-path variant resident at once, cached per device,
// variant and block size. 16-CTA clusters need the non-portable size opt-in;
// only a few clusters fit per GPC.
inline int max_active_clusters(SmallConfig cfg, bool has_post, bool fp8_out,
                               int threads, int device) {
  constexpr int kMaxDevices = 64;
  constexpr int kVariants = 2 * kMaxClusterTokens * 4;
  constexpr int kBlockSizes = kSmallWarps + 1;
  static std::atomic<int> cache[kMaxDevices][kVariants][kBlockSizes] = {};
  int const variant =
      ((cfg.cluster == 16) * kMaxClusterTokens + cfg.tokens - 1) * 4 +
      has_post * 2 + fp8_out;
  STD_TORCH_CHECK(device >= 0 && device < kMaxDevices,
                  "hy_v4_ihc_boundary: device index out of range");
  std::atomic<int>& slot = cache[device][variant][threads / 32];
  int clusters = slot.load(std::memory_order_relaxed);
  if (clusters > 0) {
    return clusters;
  }
  SmallKernel const kernel = pick_small_kernel(cfg, has_post, fp8_out);
  cudaError_t err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "hy_v4_ihc_boundary: ", cudaGetErrorString(err));
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(cfg.cluster);
  config.blockDim = dim3(threads);
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cfg.cluster;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  err = cudaOccupancyMaxActiveClusters(&clusters, kernel, &config);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "hy_v4_ihc_boundary: ", cudaGetErrorString(err));
  slot.store(clusters, std::memory_order_relaxed);
  return clusters;
}

// Small clusters of few tokens spread the work (mostly weight loads) over many
// SMs, but all clusters must be resident at once: a second wave doubles the
// time. Tuned on B300 at hidden 6144. {0, 0} if no configuration fits.
inline SmallConfig choose_small_config(int64_t num_tokens, int64_t hidden,
                                       bool has_post, bool fp8_out,
                                       int device) {
  SmallConfig const want = num_tokens <= 6    ? SmallConfig{16, 1}
                           : num_tokens <= 24 ? SmallConfig{16, 2}
                           : num_tokens <= 32 ? SmallConfig{8, 3}
                                              : SmallConfig{8, 4};
  for (SmallConfig cfg :
       {want, SmallConfig{16, 2}, SmallConfig{16, 3}, SmallConfig{8, 3},
        SmallConfig{16, 4}, SmallConfig{8, 4}, SmallConfig{8, 2},
        SmallConfig{16, 1}, SmallConfig{8, 1}}) {
    if (!small_fits(hidden, cfg)) {
      continue;
    }
    int64_t const clusters = (num_tokens + cfg.tokens - 1) / cfg.tokens;
    if (clusters <= max_active_clusters(cfg, has_post, fp8_out,
                                        small_threads(hidden, cfg), device)) {
      return cfg;
    }
  }
  return {0, 0};
}

constexpr int kMaxCoopTokens = 256;
constexpr int kMaxCoopTiles = 12;

// Tokens per tile: 8 up to 96 tokens, then 16 and 24, so the grid stays at
// most 12 tiles x 24 CTAs (one wave at two CTAs per SM).
inline int coop_tile(int64_t num_tokens) {
  return num_tokens <= 96 ? 8 : num_tokens <= 192 ? 16 : 24;
}

using CoopKernel = decltype(&ihc_coop_kernel<8, true, true>);

template <int kTile>
CoopKernel coop_kernel(bool has_post, bool fp8_out) {
  if (has_post) {
    return fp8_out ? ihc_coop_kernel<kTile, true, true>
                   : ihc_coop_kernel<kTile, true, false>;
  }
  return fp8_out ? ihc_coop_kernel<kTile, false, true>
                 : ihc_coop_kernel<kTile, false, false>;
}

inline CoopKernel pick_coop_kernel(int tile, bool has_post, bool fp8_out) {
  return tile == 8    ? coop_kernel<8>(has_post, fp8_out)
         : tile == 16 ? coop_kernel<16>(has_post, fp8_out)
                      : coop_kernel<24>(has_post, fp8_out);
}

inline int coop_smem_bytes(int tile, bool has_post) {
  if (has_post) {
    return tile == 8    ? coop_smem<8, true>()
           : tile == 16 ? coop_smem<16, true>()
                        : coop_smem<24, true>();
  }
  return tile == 8    ? coop_smem<8, false>()
         : tile == 16 ? coop_smem<16, false>()
                      : coop_smem<24, false>();
}

// The cooperative kernel's tile counters and tagged partials must outlive a
// call, so each device keeps one zeroed workspace. It is allocated on the first
// call outside stream capture (the profile run); nullptr until then.
struct CoopWorkspace {
  unsigned long long* tickets;
  float* partials;
};

inline CoopWorkspace const* coop_workspace(int device, cudaStream_t stream) {
  constexpr int kMaxDevices = 64;
  static std::atomic<CoopWorkspace*> cache[kMaxDevices] = {};
  static std::mutex mutex;
  STD_TORCH_CHECK(device >= 0 && device < kMaxDevices,
                  "hy_v4_ihc_boundary: device index out of range");
  CoopWorkspace* ws = cache[device].load(std::memory_order_acquire);
  if (ws != nullptr) {
    return ws;
  }
  cudaStreamCaptureStatus status;
  cudaError_t err = cudaStreamIsCapturing(stream, &status);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "hy_v4_ihc_boundary: ", cudaGetErrorString(err));
  if (status != cudaStreamCaptureStatusNone) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(mutex);
  ws = cache[device].load(std::memory_order_acquire);
  if (ws != nullptr) {
    return ws;
  }
  size_t const ticket_bytes = kMaxCoopTiles * sizeof(unsigned long long);
  size_t const partial_bytes = static_cast<size_t>(kMaxCoopTiles) *
                               CoopLayout::kSlices * CoopLayout::kPartStride *
                               sizeof(float4);
  void* mem = nullptr;
  err = cudaMalloc(&mem, ticket_bytes + partial_bytes);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "hy_v4_ihc_boundary: ", cudaGetErrorString(err));
  // Zero tags never match an epoch, which starts at 1.
  err = cudaMemset(mem, 0, ticket_bytes + partial_bytes);
  STD_TORCH_CHECK(err == cudaSuccess,
                  "hy_v4_ihc_boundary: ", cudaGetErrorString(err));
  ws = new CoopWorkspace{
      static_cast<unsigned long long*>(mem),
      reinterpret_cast<float*>(static_cast<char*>(mem) + ticket_bytes)};
  cache[device].store(ws, std::memory_order_release);
  return ws;
}

// Whether `ctas` CTAs of the cooperative kernel are resident at once, as its
// CTAs wait on each other. Also opts the variant into its shared memory size.
// Cached per device and variant.
inline bool coop_fits(int ctas, int tile, bool has_post, bool fp8_out,
                      int device) {
  constexpr int kMaxDevices = 64;
  static std::atomic<int> cache[kMaxDevices][3][4] = {};
  std::atomic<int>& slot = cache[device][tile / 8 - 1][has_post * 2 + fp8_out];
  int resident = slot.load(std::memory_order_relaxed);
  if (resident == 0) {
    CoopKernel const kernel = pick_coop_kernel(tile, has_post, fp8_out);
    int const smem = coop_smem_bytes(tile, has_post);
    cudaError_t err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    STD_TORCH_CHECK(err == cudaSuccess,
                    "hy_v4_ihc_boundary: ", cudaGetErrorString(err));
    int per_sm = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, kernel, CoopLayout::kThreads, smem);
    STD_TORCH_CHECK(err == cudaSuccess,
                    "hy_v4_ihc_boundary: ", cudaGetErrorString(err));
    int sms = 0;
    err = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    STD_TORCH_CHECK(err == cudaSuccess,
                    "hy_v4_ihc_boundary: ", cudaGetErrorString(err));
    // -1 records "none fit" so the query is not repeated.
    resident = per_sm * sms > 0 ? per_sm * sms : -1;
    slot.store(resident, std::memory_order_relaxed);
  }
  return ctas <= resident;
}

// The TMA stats path: whole 128-column chunks on SM 10.0 / 10.3 (B200, B300),
// the targets ihc_stats_tma_kernel is compiled for.
inline bool tma_supported(int64_t hidden, int device) {
  constexpr int kMaxDevices = 64;
  static std::atomic<int> cache[kMaxDevices] = {};  // 0 unknown, 1 yes, -1 no
  if (hidden % tma::kChunkK != 0) {
    return false;
  }
  STD_TORCH_CHECK(device >= 0 && device < kMaxDevices,
                  "hy_v4_ihc_boundary: device index out of range");
  int state = cache[device].load(std::memory_order_relaxed);
  if (state == 0) {
    int major = 0;
    int minor = 0;
    cudaError_t err = cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, device);
    if (err == cudaSuccess) {
      err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                   device);
    }
    STD_TORCH_CHECK(err == cudaSuccess,
                    "hy_v4_ihc_boundary: ", cudaGetErrorString(err));
    state = major == 10 && (minor == 0 || minor == 3) ? 1 : -1;
    if (state == 1) {
      err = cudaFuncSetAttribute(ihc_stats_tma_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 tma::kSmemBytes);
      STD_TORCH_CHECK(err == cudaSuccess,
                      "hy_v4_ihc_boundary: ", cudaGetErrorString(err));
    }
    cache[device].store(state, std::memory_order_relaxed);
  }
  return state == 1;
}

inline int sm_count(int device) {
  constexpr int kMaxDevices = 64;
  static std::atomic<int> cache[kMaxDevices] = {};
  int sms = cache[device].load(std::memory_order_relaxed);
  if (sms == 0) {
    cudaError_t const err =
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    STD_TORCH_CHECK(err == cudaSuccess,
                    "hy_v4_ihc_boundary: ", cudaGetErrorString(err));
    cache[device].store(sms, std::memory_order_relaxed);
  }
  return sms;
}

// Splits of each channel's chunks: minimizes waves x chunks per work unit,
// with 4% per split for the extra partials and per-unit epilogues.
inline int tma_ksplits(int64_t num_tokens, int64_t hidden, int sms) {
  int64_t const tiles = (num_tokens + tma::kTileM - 1) / tma::kTileM;
  int const chunks = static_cast<int>(hidden / tma::kChunkK);
  int best_splits = 1;
  double best_cost = 0.;
  for (int splits = 1; splits <= tma::kMaxKSplits; splits++) {
    if (chunks % splits != 0) {
      continue;
    }
    int64_t const waves = (tiles * kHc * splits + sms - 1) / sms;
    double const cost =
        static_cast<double>(waves) * (chunks / splits) * (1. + 0.04 * splits);
    if (splits == 1 || cost < best_cost) {
      best_cost = cost;
      best_splits = splits;
    }
  }
  return best_splits;
}

inline PFN_cuTensorMapEncodeTiled_v12000 tensor_map_encoder() {
  static PFN_cuTensorMapEncodeTiled_v12000 const fn = [] {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    cudaError_t const err = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &status);
    STD_TORCH_CHECK(err == cudaSuccess && ptr != nullptr,
                    "hy_v4_ihc_boundary: cuTensorMapEncodeTiled unavailable");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();
  return fn;
}

// Row-major BF16 [rows, cols] read and written in [box_rows, 64] boxes.
inline CUtensorMap bf16_tensor_map(void const* ptr, int64_t rows, int64_t cols,
                                   int box_rows) {
  CUtensorMap map;
  cuuint64_t const dims[2] = {static_cast<cuuint64_t>(cols),
                              static_cast<cuuint64_t>(rows)};
  cuuint64_t const strides[1] = {static_cast<cuuint64_t>(cols) * 2};
  cuuint32_t const box[2] = {tma::kBoxK, static_cast<cuuint32_t>(box_rows)};
  cuuint32_t const elem_strides[2] = {1, 1};
  CUresult const res = tensor_map_encoder()(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims,
      strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(res == CUDA_SUCCESS,
                  "hy_v4_ihc_boundary: cuTensorMapEncodeTiled failed: ",
                  static_cast<int>(res));
  return map;
}

}  // namespace vllm::hy_v4_ihc

// residual: [M, 4, H] BF16. x: [M, H] BF16 and post: [M, 4] FP32 (both or
// neither). weight: [8, 4 * H] FP32. Returns hidden [M,
// H] BF16, post gates [M, 4] FP32, with a post step the new residual [M, 4, H]
// BF16, and with fp8_out the FP8 hidden [M, H] and its scales; tensors not
// produced are empty. Allocating here costs less host time than in Python.
// round_before_norm: see the rounding note at the top of the file.
std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor,
           torch::stable::Tensor, torch::stable::Tensor>
hy_v4_ihc_boundary(torch::stable::Tensor const& residual,
                   std::optional<torch::stable::Tensor> const& x,
                   std::optional<torch::stable::Tensor> const& post,
                   torch::stable::Tensor const& weight,
                   torch::stable::Tensor const& hc_scale,
                   torch::stable::Tensor const& hc_base,
                   std::optional<torch::stable::Tensor> const& norm_weight,
                   double magnitude, double hc_eps, double rms_eps,
                   double variance_eps, bool round_before_norm, bool fp8_out) {
  using namespace vllm::hy_v4_ihc;
  using torch::headeronly::ScalarType;
  bool const has_post = x.has_value();
  bool const fuse_norm = norm_weight.has_value();
  STD_TORCH_CHECK(post.has_value() == has_post,
                  "hy_v4_ihc_boundary: x and post go together");
  STD_TORCH_CHECK(residual.dim() == 3 && residual.size(1) == kHc,
                  "hy_v4_ihc_boundary: residual must be [tokens, 4, hidden]");
  int64_t const num_tokens = residual.size(0);
  int64_t const hidden = residual.size(2);
  STD_TORCH_CHECK(hidden % 64 == 0 && hidden <= kMaxHidden,
                  "hy_v4_ihc_boundary: hidden must be a multiple of 64 and at "
                  "most 8192");
  STD_TORCH_CHECK(!fp8_out || hidden % 512 == 0,
                  "hy_v4_ihc_boundary: fp8_out needs hidden % 512 == 0");
  STD_TORCH_CHECK(residual.scalar_type() == ScalarType::BFloat16 &&
                      residual.is_contiguous(),
                  "hy_v4_ihc_boundary: residual must be contiguous BF16");
  STD_TORCH_CHECK(weight.scalar_type() == ScalarType::Float &&
                      weight.is_contiguous() && weight.dim() == 2 &&
                      weight.size(0) == kNumMixes &&
                      weight.size(1) == kHc * hidden,
                  "hy_v4_ihc_boundary: weight must be contiguous FP32 [8, 4 * "
                  "hidden]");
  STD_TORCH_CHECK(hc_scale.scalar_type() == ScalarType::Float &&
                      hc_scale.numel() == 2 &&
                      hc_base.scalar_type() == ScalarType::Float &&
                      hc_base.numel() == kNumMixes,
                  "hy_v4_ihc_boundary: hc_scale [2] and hc_base [8] must be "
                  "FP32");
  if (has_post) {
    STD_TORCH_CHECK(x->scalar_type() == ScalarType::BFloat16 &&
                        x->is_contiguous() && x->numel() == num_tokens * hidden,
                    "hy_v4_ihc_boundary: x must be contiguous BF16 [tokens, "
                    "hidden]");
    STD_TORCH_CHECK(post->scalar_type() == ScalarType::Float &&
                        post->is_contiguous() &&
                        post->numel() == num_tokens * kHc,
                    "hy_v4_ihc_boundary: post must be contiguous FP32 "
                    "[tokens, 4]");
  }
  if (fuse_norm) {
    STD_TORCH_CHECK(norm_weight->scalar_type() == ScalarType::BFloat16 &&
                        norm_weight->is_contiguous() &&
                        norm_weight->numel() == hidden,
                    "hy_v4_ihc_boundary: norm_weight must be contiguous BF16 "
                    "[hidden]");
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      residual.get_device_index());
  auto hidden_out = torch::stable::new_empty(residual, {num_tokens, hidden});
  auto post_out =
      torch::stable::new_empty(residual, {num_tokens, kHc}, ScalarType::Float);
  auto residual_out = torch::stable::new_empty(
      residual, {has_post ? num_tokens : 0, kHc, hidden});
  // DeepGEMM's layout: scale [m, k / 512] with strides (1, m padded to 4).
  int64_t const scale_m = fp8_out ? (num_tokens + 3) / 4 * 4 : 0;
  auto fp8_hidden = torch::stable::new_empty(
      residual, {fp8_out ? num_tokens : 0, hidden}, ScalarType::Float8_e4m3fn);
  auto fp8_scale = torch::stable::transpose(
      torch::stable::new_empty(residual, {fp8_out ? hidden / 512 : 0, scale_m},
                               ScalarType::Int),
      0, 1);
  fp8_scale = torch::stable::narrow(fp8_scale, 0, 0, fp8_out ? num_tokens : 0);
  if (num_tokens == 0) {
    return {hidden_out, post_out, residual_out, fp8_hidden, fp8_scale};
  }
  // 16-byte vector accesses.
  STD_TORCH_CHECK(aligned16(residual.data_ptr()) &&
                      aligned16(weight.data_ptr()) &&
                      (!has_post || aligned16(x->data_ptr())) &&
                      (!fuse_norm || aligned16(norm_weight->data_ptr())),
                  "hy_v4_ihc_boundary: tensors must be 16-byte aligned");
  int const device = residual.get_device_index();
  cudaStream_t const stream = get_current_cuda_stream(device);
  auto const* res_in =
      reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr());
  auto* res_out =
      has_post
          ? reinterpret_cast<__nv_bfloat16*>(residual_out.mutable_data_ptr())
          : nullptr;
  int const m = static_cast<int>(num_tokens);
  int const h = static_cast<int>(hidden);

  SmallConfig const small =
      fuse_norm && !round_before_norm && num_tokens <= kMaxSmallTokens
          ? choose_small_config(num_tokens, hidden, has_post, fp8_out, device)
          : SmallConfig{0, 0};
  if (small.cluster > 0) {
    int const clusters =
        static_cast<int>((num_tokens + small.tokens - 1) / small.tokens);
    launch_pdl(
        pick_small_kernel(small, has_post, fp8_out), clusters * small.cluster,
        small_threads(hidden, small), small.cluster, 0, stream, res_in,
        has_post ? reinterpret_cast<__nv_bfloat16 const*>(x->data_ptr())
                 : nullptr,
        has_post ? reinterpret_cast<float const*>(post->data_ptr()) : nullptr,
        reinterpret_cast<float const*>(weight.data_ptr()),
        reinterpret_cast<float const*>(hc_scale.data_ptr()),
        reinterpret_cast<float const*>(hc_base.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(norm_weight->data_ptr()),
        res_out, reinterpret_cast<float*>(post_out.mutable_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(hidden_out.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(fp8_hidden.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(fp8_scale.mutable_data_ptr()),
        static_cast<int>(scale_m), m, h, static_cast<float>(magnitude),
        static_cast<float>(hc_eps), static_cast<float>(rms_eps),
        static_cast<float>(variance_eps));
    return {hidden_out, post_out, residual_out, fp8_hidden, fp8_scale};
  }

  if (fuse_norm && !round_before_norm && hidden == kSmallHidden) {
    // Looked up on every such call so that the eager profile run allocates
    // the workspace before CUDA graph capture.
    CoopWorkspace const* ws = coop_workspace(device, stream);
    int const tile = coop_tile(num_tokens);
    int const ctas =
        static_cast<int>((num_tokens + tile - 1) / tile) * CoopLayout::kSlices;
    if (ws != nullptr && num_tokens <= kMaxCoopTokens &&
        coop_fits(ctas, tile, has_post, fp8_out, device)) {
      launch_pdl(
          pick_coop_kernel(tile, has_post, fp8_out), ctas, CoopLayout::kThreads,
          1, coop_smem_bytes(tile, has_post), stream, res_in,
          has_post ? reinterpret_cast<__nv_bfloat16 const*>(x->data_ptr())
                   : nullptr,
          has_post ? reinterpret_cast<float const*>(post->data_ptr()) : nullptr,
          reinterpret_cast<float const*>(weight.data_ptr()),
          reinterpret_cast<float const*>(hc_scale.data_ptr()),
          reinterpret_cast<float const*>(hc_base.data_ptr()),
          reinterpret_cast<__nv_bfloat16 const*>(norm_weight->data_ptr()),
          res_out, reinterpret_cast<float*>(post_out.mutable_data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(hidden_out.mutable_data_ptr()),
          reinterpret_cast<uint8_t*>(fp8_hidden.mutable_data_ptr()),
          reinterpret_cast<uint8_t*>(fp8_scale.mutable_data_ptr()),
          static_cast<int>(scale_m), ws->partials, ws->tickets, m, h,
          static_cast<float>(magnitude), static_cast<float>(hc_eps),
          static_cast<float>(rms_eps), static_cast<float>(variance_eps));
      return {hidden_out, post_out, residual_out, fp8_hidden, fp8_scale};
    }
  }

  if (has_post && num_tokens > tma::kMinTokens &&
      tma_supported(hidden, device)) {
    int64_t const k_total = kHc * hidden;
    auto weight_split = torch::stable::new_empty(
        residual, {tma::kMmaN, k_total}, ScalarType::BFloat16);
    int64_t const weight_count = kNumMixes * k_total;
    launch_pdl(
        ihc_split_weight_kernel,
        static_cast<int>((weight_count / 4 + kBlock - 1) / kBlock), kBlock, 1,
        0, stream, reinterpret_cast<float const*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(weight_split.mutable_data_ptr()),
        weight_count);
    int const ksplits = tma_ksplits(num_tokens, hidden, sm_count(device));
    int const num_splits = kHc * ksplits;
    auto partials = torch::stable::new_empty(
        residual, {num_splits, num_tokens, kNumStats}, ScalarType::Float);
    auto* partials_ptr = reinterpret_cast<float*>(partials.mutable_data_ptr());
    int const units = static_cast<int>((num_tokens + tma::kTileM - 1) /
                                       tma::kTileM * num_splits);
    launch_pdl(ihc_stats_tma_kernel, std::min(units, sm_count(device)), kBlock,
               1, tma::kSmemBytes, stream,
               bf16_tensor_map(res_in, num_tokens, k_total, tma::kTileM),
               bf16_tensor_map(x->data_ptr(), num_tokens, hidden, tma::kTileM),
               bf16_tensor_map(weight_split.data_ptr(), tma::kMmaN, k_total,
                               tma::kMmaN),
               bf16_tensor_map(res_out, num_tokens, k_total, tma::kTileM),
               reinterpret_cast<float const*>(post->data_ptr()), partials_ptr,
               m, h, ksplits);
    // HY V4's hidden size issues its residual loads before the gates.
    constexpr int kPrefetch = kSmallHidden / (kBlock * kVec);
    bool const prefetch = hidden == kSmallHidden;
    launch_pdl(
        fp8_out ? (prefetch
                       ? pick_finish_kernel<true, kPrefetch>(fuse_norm,
                                                             round_before_norm)
                       : pick_finish_kernel<true>(fuse_norm, round_before_norm))
                : (prefetch ? pick_finish_kernel<false, kPrefetch>(
                                  fuse_norm, round_before_norm)
                            : pick_finish_kernel<false>(fuse_norm,
                                                        round_before_norm)),
        m, kBlock, 1, 0, stream, static_cast<__nv_bfloat16 const*>(res_out),
        static_cast<float const*>(partials_ptr),
        reinterpret_cast<float const*>(hc_scale.data_ptr()),
        reinterpret_cast<float const*>(hc_base.data_ptr()),
        fuse_norm
            ? reinterpret_cast<__nv_bfloat16 const*>(norm_weight->data_ptr())
            : nullptr,
        reinterpret_cast<float*>(post_out.mutable_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(hidden_out.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(fp8_hidden.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(fp8_scale.mutable_data_ptr()),
        static_cast<int>(scale_m), m, h, num_splits,
        static_cast<float>(magnitude), static_cast<float>(hc_eps),
        static_cast<float>(rms_eps), static_cast<float>(variance_eps));
    return {hidden_out, post_out, residual_out, fp8_hidden, fp8_scale};
  }

  int const num_splits = choose_splits(num_tokens, hidden);
  STD_TORCH_CHECK(num_splits > 0, "hy_v4_ihc_boundary: unsupported hidden");
  auto partials = torch::stable::new_empty(
      residual, {num_splits, num_tokens, kNumStats}, ScalarType::Float);
  auto* partials_ptr = reinterpret_cast<float*>(partials.mutable_data_ptr());
  int const tiles = static_cast<int>((num_tokens + kTileM - 1) / kTileM);

  launch_pdl(
      pick_stats_kernel(num_splits, has_post), tiles * num_splits, kBlock, 1, 0,
      stream, res_in,
      has_post ? reinterpret_cast<__nv_bfloat16 const*>(x->data_ptr())
               : nullptr,
      has_post ? reinterpret_cast<float const*>(post->data_ptr()) : nullptr,
      reinterpret_cast<float const*>(weight.data_ptr()), res_out, partials_ptr,
      m, h);
  launch_pdl(fp8_out ? pick_finish_kernel<true>(fuse_norm, round_before_norm)
                     : pick_finish_kernel<false>(fuse_norm, round_before_norm),
             m, kBlock, 1, 0, stream,
             has_post ? static_cast<__nv_bfloat16 const*>(res_out) : res_in,
             static_cast<float const*>(partials_ptr),
             reinterpret_cast<float const*>(hc_scale.data_ptr()),
             reinterpret_cast<float const*>(hc_base.data_ptr()),
             fuse_norm ? reinterpret_cast<__nv_bfloat16 const*>(
                             norm_weight->data_ptr())
                       : nullptr,
             reinterpret_cast<float*>(post_out.mutable_data_ptr()),
             reinterpret_cast<__nv_bfloat16*>(hidden_out.mutable_data_ptr()),
             reinterpret_cast<uint8_t*>(fp8_hidden.mutable_data_ptr()),
             reinterpret_cast<uint8_t*>(fp8_scale.mutable_data_ptr()),
             static_cast<int>(scale_m), m, h, num_splits,
             static_cast<float>(magnitude), static_cast<float>(hc_eps),
             static_cast<float>(rms_eps), static_cast<float>(variance_eps));
  return {hidden_out, post_out, residual_out, fp8_hidden, fp8_scale};
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("hy_v4_ihc_boundary", TORCH_BOX(&hy_v4_ihc_boundary));
}
