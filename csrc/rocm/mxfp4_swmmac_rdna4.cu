// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MXFP4 x FP8 SWMMAC skinny GEMM for RDNA4 (gfx12), M <= 8. Companion to
// mxfp4_gemm_rdna4.cu (same torch op, same packed layouts, shared decode in
// qdq_mxfp4_rdna4.cuh); split out so the sparse-matrix path can evolve on its
// own. See the kernel comment for the row-pair mapping.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdlib>

#include "qdq_mxfp4_rdna4.cuh"

#if defined(__gfx1200__) || defined(__gfx1201__)
  #define __HIP__MXFP4_RDNA4__
#endif

namespace vllm {
namespace mxfp4_rdna4_swmmac {

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

#if defined(__HIP__MXFP4_RDNA4__) || !defined(__HIP_DEVICE_COMPILE__)
using v8f32 = float __attribute__((ext_vector_type(8)));
using v2i32 = int __attribute__((ext_vector_type(2)));
using v4i32 = int __attribute__((ext_vector_type(4)));
using mxfp4_rdna4::dequant_mxfp4_8_e4m3;
using mxfp4_rdna4::mxfp4_e8m0_to_f32;

// ---------------------------------------------------------------------------
// SWMMAC skinny path (M <= 8), after vLLM's gfx12 skinny GEMM (PR #45559).
// v_swmmac_f32_16x16x32_fp8 takes a 2:4-sparse 16x32 A stored as 16x16 plus
// a 2-bit-per-element index. Eight dense rows are mapped onto the 16 sparse
// rows: sparse row 2r carries K pairs {0,1} of each quad of row r (index
// 0x44444444), row 2r+1 carries pairs {2,3} (0xEEEEEEEE); both multiply the
// same dense 32x16 B and the epilogue sums the two partial rows. One
// instruction thus covers a whole 32-wide E8M0 group and twice the B of a
// WMMA step. Layout verified on-device against an exact integer reference.
//
// A single wave walking all of K is latency-bound (the first version took the
// same ~44us at N=4096 and N=11008), so: KW waves per 16-column tile each
// take a contiguous slice of the K groups and reduce through LDS, and each
// wave keeps PF groups of A/B/scale loads in flight in a register ring.
template <typename T, int KW, int PF>
__global__ void __launch_bounds__(32 * KW)
    gemm_mxfp4_swmmac_skinny(const uint8_t* __restrict__ aq,
                             const float* __restrict__ ascale,
                             const uint32_t* __restrict__ b_q,
                             const uint8_t* __restrict__ b_scales,
                             T* __restrict__ c, int M, int N, int K) {
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int lo = lane & 15;  // B: column | A: sparse row | acc: N index
  const int hi = lane >> 4;  // K half of the 32-wide group
  const int n_col = blockIdx.x * 16 + lo;
  const bool n_ok = n_col < N;
  const int a_row = lo >> 1;
  const bool a_ok = a_row < M;
  const bool odd = (lo & 1) != 0;
  const int idx = odd ? (int)0xEEEEEEEEu : (int)0x44444444u;
  const int ngroups = K / kGroup;
  const int g_begin = (ngroups * wave) / KW;
  const int g_end = (ngroups * (wave + 1)) / KW;

  auto load = [&](int g, v2i32& a, uint32_t& q0, uint32_t& q1, float& ws) {
    const int k_g = g * kGroup;
    a = {0, 0};
    if (a_ok) {
      uint4 w;
      __builtin_memcpy(&w, aq + (size_t)a_row * K + k_g + 16 * hi, 16);
      if (!odd) {  // pairs {0,1} of each quad
        a[0] = (int)((w.x & 0xFFFFu) | (w.y << 16));
        a[1] = (int)((w.z & 0xFFFFu) | (w.w << 16));
      } else {  // pairs {2,3}
        a[0] = (int)((w.x >> 16) | (w.y & 0xFFFF0000u));
        a[1] = (int)((w.z >> 16) | (w.w & 0xFFFF0000u));
      }
    }
    q0 = q1 = 0u;
    ws = 0.0f;
    if (n_ok) {
      // Fragment-major B [K/32][N][4]: octets 2*hi, 2*hi+1 are adjacent.
      uint2 bw;
      __builtin_memcpy(&bw, b_q + ((size_t)g * N + n_col) * 4 + 2 * hi, 8);
      q0 = bw.x;
      q1 = bw.y;
      ws = mxfp4_e8m0_to_f32(b_scales[(size_t)(k_g / kGroup) * N + n_col]);
    }
  };

  v8f32 acc = {0, 0, 0, 0, 0, 0, 0, 0};
  v2i32 a_r[PF];
  uint32_t q0_r[PF], q1_r[PF];
  float w_r[PF];
  #pragma unroll
  for (int d = 0; d < PF; d++)
    if (g_begin + d < g_end)
      load(g_begin + d, a_r[d], q0_r[d], q1_r[d], w_r[d]);

  for (int g0 = g_begin; g0 < g_end; g0 += PF) {
  #pragma unroll
    for (int d = 0; d < PF; d++) {
      const int g = g0 + d;
      if (g >= g_end) break;
      uint32_t l0, h0, l1, h1;
      dequant_mxfp4_8_e4m3(q0_r[d], l0, h0);
      dequant_mxfp4_8_e4m3(q1_r[d], l1, h1);
      v4i32 b;
      b[0] = (int)l0;
      b[1] = (int)h0;
      b[2] = (int)l1;
      b[3] = (int)h1;
      v8f32 grp = {0, 0, 0, 0, 0, 0, 0, 0};
      grp =
          __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a_r[d], b, grp, idx);
      const float ws = w_r[d];
  #pragma unroll
      for (int i = 0; i < 8; i++) acc[i] = fmaf(grp[i], ws, acc[i]);
      if (g + PF < g_end) load(g + PF, a_r[d], q0_r[d], q1_r[d], w_r[d]);
    }
  }

  // Sparse rows 8*hi+2i and 8*hi+2i+1 are the two halves of logical row 4*hi+i.
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
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    const int m_row = 4 * hi + i;
    if (m_row >= M) continue;
    c[(size_t)m_row * N + n_col] = from_f<T>(part[i] * ascale[m_row]);
  }
}

#else
template <typename T, int KW, int PF>
__global__ void gemm_mxfp4_swmmac_skinny(const uint8_t*, const float*,
                                         const uint32_t*, const uint8_t*, T*,
                                         int, int, int) {}
#endif

}  // namespace mxfp4_rdna4_swmmac
}  // namespace vllm

// Host launcher used by mxfp4_gemm_rdna4.cu's dispatch. `c` is fp16 or bf16
// per `out_bf16`; `stream` is the current HIP stream.
// VLLM_MXFP4_RDNA4_SKINNY_KW = waves per 16-column tile (1, 2 or 4; default 4).
static int skinny_kw() {
  static const int v = [] {
    const char* e = std::getenv("VLLM_MXFP4_RDNA4_SKINNY_KW");
    const int k = e ? atoi(e) : 4;
    return (k == 1 || k == 2) ? k : 4;
  }();
  return v;
}

template <typename T>
static void launch_t(const uint8_t* aq, const float* ascale,
                     const uint32_t* b_q, const uint8_t* b_scales, T* c, int M,
                     int N, int K, cudaStream_t st) {
  using namespace vllm::mxfp4_rdna4_swmmac;
  const dim3 grd((N + 15) / 16);
  switch (skinny_kw()) {
    case 1:
      gemm_mxfp4_swmmac_skinny<T, 1, 4>
          <<<grd, 32, 0, st>>>(aq, ascale, b_q, b_scales, c, M, N, K);
      break;
    case 2:
      gemm_mxfp4_swmmac_skinny<T, 2, 4>
          <<<grd, 64, 0, st>>>(aq, ascale, b_q, b_scales, c, M, N, K);
      break;
    default:
      gemm_mxfp4_swmmac_skinny<T, 4, 4>
          <<<grd, 128, 0, st>>>(aq, ascale, b_q, b_scales, c, M, N, K);
      break;
  }
}

// Host launcher used by mxfp4_gemm_rdna4.cu's dispatch. `c` is fp16 or bf16
// per `out_bf16`; `stream` is the current HIP stream.
void mxfp4_swmmac_skinny_launch(const uint8_t* aq, const float* ascale,
                                const uint32_t* b_q, const uint8_t* b_scales,
                                void* c, bool out_bf16, int M, int N, int K,
                                void* stream) {
  auto st = reinterpret_cast<cudaStream_t>(stream);
  if (out_bf16)
    launch_t<__hip_bfloat16>(aq, ascale, b_q, b_scales,
                             reinterpret_cast<__hip_bfloat16*>(c), M, N, K, st);
  else
    launch_t<half>(aq, ascale, b_q, b_scales, reinterpret_cast<half*>(c), M, N,
                   K, st);
}
