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
};

#if defined(__HIP__MXFP4_RDNA4__) || !defined(__HIP_DEVICE_COMPILE__)
using v8f32 = float __attribute__((ext_vector_type(8)));
using v2i32 = int __attribute__((ext_vector_type(2)));
using v4i32 = int __attribute__((ext_vector_type(4)));
using mxfp4_rdna4::dequant_mxfp4_8_e4m3;
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
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lo = lane & 15;
  const int hi = lane >> 4;
  const int n_col = blockIdx.x * 16 + lo;
  const bool n_ok = n_col < ar.N;
  const int K = ar.K, N = ar.N;
  const int a_r = lo >> 1;
  const int slot = ar.sorted_token_ids[blk * 8 + a_r];
  const bool a_ok = slot < ar.total_slots;
  const int a_row = a_ok ? slot / ar.top_k : 0;
  const bool odd = (lo & 1) != 0;
  const int idx = odd ? (int)0xEEEEEEEEu : (int)0x44444444u;
  const uint32_t* bq = ar.b_q + (size_t)e * ar.w_stride;
  const uint8_t* bs = ar.b_scales + (size_t)e * ar.s_stride;
  const int ngroups = K / kGroup;
  const int g_begin = (ngroups * wave) / KW;
  const int g_end = (ngroups * (wave + 1)) / KW;

  auto load = [&](int g, v2i32& a, unsigned long long& b, float& ws) {
    const int k_g = g * kGroup;
    a = {0, 0};
    if (a_ok) {
      uint4 w;
      __builtin_memcpy(&w, ar.aq + (size_t)a_row * K + k_g + 16 * hi, 16);
      if (!odd) {
        a[0] = (int)((w.x & 0xFFFFu) | (w.y << 16));
        a[1] = (int)((w.z & 0xFFFFu) | (w.w << 16));
      } else {
        a[0] = (int)((w.x >> 16) | (w.y & 0xFFFF0000u));
        a[1] = (int)((w.z >> 16) | (w.w & 0xFFFF0000u));
      }
    }
    b = 0ull;
    ws = 0.0f;
    if (n_ok) {
      const auto* bp = reinterpret_cast<const unsigned long long*>(
          bq + ((size_t)g * N + n_col) * 4 + 2 * hi);
      b = __builtin_nontemporal_load(bp);
      ws = mxfp4_e8m0_to_f32(bs[(size_t)g * N + n_col]);
    }
  };

  v8f32 acc = {0, 0, 0, 0, 0, 0, 0, 0};
  v2i32 a_r_[PF];
  unsigned long long b_r[PF];
  float w_r[PF];
  #pragma unroll
  for (int d = 0; d < PF; d++)
    if (g_begin + d < g_end) load(g_begin + d, a_r_[d], b_r[d], w_r[d]);
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
      v8f32 grp = {0, 0, 0, 0, 0, 0, 0, 0};
      grp = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a_r_[d], b, grp,
                                                             idx);
      const float ws = w_r[d];
  #pragma unroll
      for (int i = 0; i < 8; i++) acc[i] = fmaf(grp[i], ws, acc[i]);
      if (g + PF < g_end) load(g + PF, a_r_[d], b_r[d], w_r[d]);
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
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lo = lane & 15;
  const int hi = lane >> 4;
  const int n_col = blockIdx.x * 16 + lo;
  const bool n_ok = n_col < ar.N;
  const int K = ar.K, N = ar.N;
  const uint32_t* bq = ar.b_q + (size_t)e * ar.w_stride;
  const uint8_t* bs = ar.b_scales + (size_t)e * ar.s_stride;
  const int ngroups = K / kGroup;
  const int g_begin = (ngroups * wave) / KW;
  const int g_end = (ngroups * (wave + 1)) / KW;

  int a_row[MT];
  bool a_ok[MT];
  #pragma unroll
  for (int t = 0; t < MT; t++) {
    const int s = ar.sorted_token_ids[blk * 16 * MT + 16 * t + lo];
    a_ok[t] = s < ar.total_slots;
    a_row[t] = a_ok[t] ? s / ar.top_k : 0;
  }

  struct Slot {
    unsigned long long b;
    float ws;
  };
  Slot ring[PF];
  auto load_b = [&](int g, Slot& sl) {
    sl.b = 0ull;
    sl.ws = 0.0f;
    if (n_ok) {
      const auto* bp = reinterpret_cast<const unsigned long long*>(
          bq + ((size_t)g * N + n_col) * 4 + 2 * hi);
      sl.b = __builtin_nontemporal_load(bp);
      sl.ws = mxfp4_e8m0_to_f32(bs[(size_t)g * N + n_col]);
    }
  };
  auto load_a = [&](int g, uint4(&a)[MT]) {
    const int k_g = g * kGroup;
  #pragma unroll
    for (int t = 0; t < MT; t++) {
      uint4 aw = {0u, 0u, 0u, 0u};
      if (a_ok[t])
        __builtin_memcpy(&aw, ar.aq + (size_t)a_row[t] * K + k_g + 16 * hi, 16);
      a[t] = aw;
    }
  };

  v8f32 acc[MT];
  #pragma unroll
  for (int t = 0; t < MT; t++) acc[t] = {0, 0, 0, 0, 0, 0, 0, 0};
  #pragma unroll
  for (int d = 0; d < PF; d++)
    if (g_begin + d < g_end) load_b(g_begin + d, ring[d]);
  uint4 a_cur[MT], a_nx[MT];
  if (g_begin < g_end) load_a(g_begin, a_cur);

  for (int g0 = g_begin; g0 < g_end; g0 += PF) {
  #pragma unroll
    for (int d = 0; d < PF; d++) {
      const int g = g0 + d;
      if (g >= g_end) break;
      const bool has_next = (g + 1 < g_end);
      if (has_next) load_a(g + 1, a_nx);
      const Slot sl = ring[d];
      if (g + PF < g_end) load_b(g + PF, ring[d]);
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
        af0[0] = (int)a_cur[t].x;
        af0[1] = (int)a_cur[t].y;
        af1[0] = (int)a_cur[t].z;
        af1[1] = (int)a_cur[t].w;
        grp = wmma_fp8(af0, bf[0], grp);
        grp = wmma_fp8(af1, bf[1], grp);
  #pragma unroll
        for (int q = 0; q < 8; q++) acc[t][q] = fmaf(grp[q], sl.ws, acc[t][q]);
      }
      if (has_next) {
  #pragma unroll
        for (int t = 0; t < MT; t++) a_cur[t] = a_nx[t];
      }
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
#else
template <typename T, int KW, int PF>
__global__ void moe_swmmac_kernel(MoeArgs, T*, const T*) {}
template <typename T, int MT, int KW, int PF>
__global__ void moe_wmma_kernel(MoeArgs, T*, const T*) {}
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
  moe_swmmac_kernel<T, KW_, 4><<<grd, 32 * (KW_), 0, st>>>(ar, c, bias)
#define L_WM(MT_, KW_) \
  moe_wmma_kernel<T, MT_, KW_, 4><<<grd, 32 * (KW_), 0, st>>>(ar, c, bias)
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
  if (block_size_m == 8) {
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

void moe_mxfp4_gemm_rdna4(
    torch::Tensor const& a_fp8, torch::Tensor const& a_scale, torch::Tensor& c,
    torch::Tensor const& b_q, torch::Tensor const& b_scales,
    torch::Tensor const& topk_weights, torch::Tensor const& sorted_token_ids,
    torch::Tensor const& expert_ids,
    torch::Tensor const& num_tokens_post_padded, int64_t top_k,
    int64_t block_size_m, bool mul_topk_weight,
    const std::optional<torch::Tensor>& bias) {
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
                  block_size_m == 64,
              "block_size_m must be 8, 16, 32 or 64");
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
