// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HY V4 iHC boundary: optional post step, then pre (gate projection, gates and
// gated channel reduction) and the following RMSNorm, in two kernels.
//
//   residual_out[t, j, :] = bf16(post[t, j] * x[t, :] + residual[t, j, :])
//   mixes[t, n] = sum_k residual_out[t, k] * W[n, k]      (k over hc * hidden)
//   pre/post gates from mixes and the RMS of residual_out; hidden =
//   rms_norm(bf16(sum_j pre[t, j] * residual_out[t, j, :]), norm_weight)
//
// The structure follows TensorRT-LLM's fused hyper-connection kernels
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

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
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
constexpr int kStatGroup = 16;  // threads per statistic in the finish kernel

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

// One CTA per token: gates, gated reduction and RMSNorm. Each thread keeps its
// slice of the reduced row in registers, so the hidden state is written once.
template <bool kFuseNorm, bool kRoundBeforeNorm>
__launch_bounds__(kBlock) __global__
    void ihc_finish_kernel(__nv_bfloat16 const* __restrict__ residual,
                           float const* __restrict__ partials,
                           float const* __restrict__ hc_scale,
                           float const* __restrict__ hc_base,
                           __nv_bfloat16 const* __restrict__ norm_weight,
                           float* __restrict__ post_out,
                           __nv_bfloat16* __restrict__ hidden_out,
                           int num_tokens, int hidden, int num_splits,
                           float magnitude, float hc_eps, float rms_eps,
                           float var_eps) {
  cudaGridDependencySynchronize();
  int64_t const tok = blockIdx.x;
  int const tid = threadIdx.x;
  int const warp = tid / 32;
  int const lane = tid % 32;
  int64_t const k_total = static_cast<int64_t>(kHc) * hidden;
  __shared__ float s_stat[kNumStats];
  __shared__ float s_pre[kHc];
  __shared__ float s_red[kWarps];
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

  float row[kMaxRowIters][kVec];
  float sumsq = 0.f;
#pragma unroll
  for (int it = 0; it < kMaxRowIters; it++) {
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
      unpack8(
          *reinterpret_cast<uint4 const*>(residual + tok * k_total +
                                          j * static_cast<int64_t>(hidden) + h),
          r);
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
  for (int it = 0; it < kMaxRowIters; it++) {
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
    *reinterpret_cast<uint4*>(hidden_out + tok * hidden + h) = pack8(row[it]);
  }
  cudaTriggerProgrammaticLaunchCompletion();
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
                              __nv_bfloat16*, int, int, int, float, float,
                              float, float);

inline FinishKernel pick_finish_kernel(bool fuse_norm, bool round_before_norm) {
  if (fuse_norm) {
    return round_before_norm ? ihc_finish_kernel<true, true>
                             : ihc_finish_kernel<true, false>;
  }
  return round_before_norm ? ihc_finish_kernel<false, true>
                           : ihc_finish_kernel<false, false>;
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
void launch_pdl(Kernel kernel, int grid, cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kBlock);
  config.stream = stream;
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  cudaError_t const err = cudaLaunchKernelEx(&config, kernel, args...);
  STD_TORCH_CHECK(err == cudaSuccess, "hy_v4_ihc_boundary: launch failed: ",
                  cudaGetErrorString(err));
}

}  // namespace vllm::hy_v4_ihc

// residual: [M, 4, H] BF16. x: [M, H] BF16 and post: [M, 4] FP32 (both or
// neither). weight: [8, 4 * H] FP32. Returns hidden [M, H] BF16, post gates
// [M, 4] FP32 and, with a post step, the new residual [M, 4, H] BF16 (else an
// empty tensor). Allocating here costs less host time than in Python.
// round_before_norm: see the rounding note at the top of the file.
std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
hy_v4_ihc_boundary(torch::stable::Tensor const& residual,
                   std::optional<torch::stable::Tensor> const& x,
                   std::optional<torch::stable::Tensor> const& post,
                   torch::stable::Tensor const& weight,
                   torch::stable::Tensor const& hc_scale,
                   torch::stable::Tensor const& hc_base,
                   std::optional<torch::stable::Tensor> const& norm_weight,
                   double magnitude, double hc_eps, double rms_eps,
                   double variance_eps, bool round_before_norm) {
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
  if (num_tokens == 0) {
    return {hidden_out, post_out, residual_out};
  }
  // 16-byte vector accesses.
  STD_TORCH_CHECK(aligned16(residual.data_ptr()) &&
                      aligned16(weight.data_ptr()) &&
                      (!has_post || aligned16(x->data_ptr())) &&
                      (!fuse_norm || aligned16(norm_weight->data_ptr())),
                  "hy_v4_ihc_boundary: tensors must be 16-byte aligned");
  int const num_splits = choose_splits(num_tokens, hidden);
  STD_TORCH_CHECK(num_splits > 0, "hy_v4_ihc_boundary: unsupported hidden");

  cudaStream_t const stream =
      get_current_cuda_stream(residual.get_device_index());
  auto partials = torch::stable::new_empty(
      residual, {num_splits, num_tokens, kNumStats}, ScalarType::Float);
  auto const* res_in =
      reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr());
  auto* res_out =
      has_post
          ? reinterpret_cast<__nv_bfloat16*>(residual_out.mutable_data_ptr())
          : nullptr;
  auto* partials_ptr = reinterpret_cast<float*>(partials.mutable_data_ptr());
  int const m = static_cast<int>(num_tokens);
  int const h = static_cast<int>(hidden);
  int const tiles = static_cast<int>((num_tokens + kTileM - 1) / kTileM);

  launch_pdl(
      pick_stats_kernel(num_splits, has_post), tiles * num_splits, stream,
      res_in,
      has_post ? reinterpret_cast<__nv_bfloat16 const*>(x->data_ptr())
               : nullptr,
      has_post ? reinterpret_cast<float const*>(post->data_ptr()) : nullptr,
      reinterpret_cast<float const*>(weight.data_ptr()), res_out, partials_ptr,
      m, h);
  launch_pdl(
      pick_finish_kernel(fuse_norm, round_before_norm), m, stream,
      has_post ? static_cast<__nv_bfloat16 const*>(res_out) : res_in,
      static_cast<float const*>(partials_ptr),
      reinterpret_cast<float const*>(hc_scale.data_ptr()),
      reinterpret_cast<float const*>(hc_base.data_ptr()),
      fuse_norm
          ? reinterpret_cast<__nv_bfloat16 const*>(norm_weight->data_ptr())
          : nullptr,
      reinterpret_cast<float*>(post_out.mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(hidden_out.mutable_data_ptr()), m, h,
      num_splits, static_cast<float>(magnitude), static_cast<float>(hc_eps),
      static_cast<float>(rms_eps), static_cast<float>(variance_eps));
  return {hidden_out, post_out, residual_out};
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("hy_v4_ihc_boundary", TORCH_BOX(&hy_v4_ihc_boundary));
}
