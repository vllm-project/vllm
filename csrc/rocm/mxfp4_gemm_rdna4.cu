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
//   b_q      [K/8, N]   uint32, 8 E2M1 codes per word, nibble i = K+i
//   b_scale  [K/32, N]  uint8 E8M0
//   out      [M, N]     same dtype as a

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "qdq_mxfp4_rdna4.cuh"

#if defined(__gfx1200__) || defined(__gfx1201__)
  #define __HIP__MXFP4_RDNA4__
#endif

namespace vllm {
namespace mxfp4_rdna4 {

using bf16_t = __hip_bfloat16;

constexpr int kGroup = 32;  // MXFP4 E8M0 block
constexpr float kE4M3Max = 448.0f;
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

// Software OCP E4M3 encode (round-to-nearest-even, saturating). The activation
// quantize kernel is memory bound, so this is not on the critical path.
__device__ __forceinline__ uint8_t enc_e4m3(float v) {
  uint32_t s = __float_as_uint(v) & 0x80000000u;
  float a = __builtin_fabsf(v);
  if (!(a > 0.0f)) return (uint8_t)(s >> 24);             // +-0 / NaN -> 0
  if (a >= kE4M3Max) return (uint8_t)((s >> 24) | 0x7E);  // saturate to max
  int e;
  float m = frexpf(a, &e);  // a = m * 2^e, m in [0.5, 1)
  int E = e - 1 + 7;        // unbiased (e-1) + E4M3 bias
  int M;
  if (E <= 0) {                            // subnormal: fixed exponent 2^-6
    M = (int)__builtin_rintf(a * 512.0f);  // a / 2^-6 * 8
    if (M >= 8) {
      E = 1;
      M -= 8;
    } else
      E = 0;
  } else {
    M = (int)__builtin_rintf((m * 2.0f - 1.0f) * 8.0f);
    if (M == 8) {
      M = 0;
      E += 1;
    }
    // OCP E4M3 max normal is E=15,M=6 (448); only E=15,M=7 is NaN. Saturating
    // the whole E==15 binade would flatten every value in [256,448) onto 448.
    if (E > 15 || (E == 15 && M > 6)) return (uint8_t)((s >> 24) | 0x7E);
  }
  return (uint8_t)((s >> 24) | (E << 3) | M);
}

// ---------------------------------------------------------------------------
// Activation quantize: a[M,K] -> aq[M,K] (E4M3) + ascale[M] (fp32), per row.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void quantize_a_e4m3(const T* __restrict__ a,
                                uint8_t* __restrict__ aq,
                                float* __restrict__ ascale, int M, int K) {
  const int row = blockIdx.x;
  if (row >= M) return;
  const T* ar = a + (size_t)row * K;

  float amax = 0.0f;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    amax = fmaxf(amax, __builtin_fabsf(to_f<T>(ar[k])));

  __shared__ float red[32];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  for (int off = 16; off; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off));
  if (lane == 0) red[wave] = amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) t = fmaxf(t, red[i]);
    red[31] = (t > 0.0f) ? (t / kE4M3Max) : 1.0f;
    ascale[row] = red[31];
  }
  __syncthreads();

  const float inv = 1.0f / red[31];
  uint8_t* qr = aq + (size_t)row * K;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    qr[k] = enc_e4m3(to_f<T>(ar[k]) * inv);
}

#if defined(__HIP__MXFP4_RDNA4__) || !defined(__HIP_DEVICE_COMPILE__)

using v8f32 = float __attribute__((ext_vector_type(8)));
using v2i32 = int __attribute__((ext_vector_type(2)));

__device__ __forceinline__ v8f32 wmma_fp8(v2i32 a, v2i32 b, v8f32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

// ---------------------------------------------------------------------------
// Section 1: general (prefill) path. 16M x 16N tile, one wave, full K.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void gemm_mxfp4_wmma_fp8_16x16_1w(
    const uint8_t* __restrict__ aq, const float* __restrict__ ascale,
    const uint32_t* __restrict__ b_q, const uint8_t* __restrict__ b_scales,
    T* __restrict__ c, int M, int N, int K) {
  const int m_tile = blockIdx.y * 16;
  const int n_tile = blockIdx.x * 16;
  if (m_tile >= M || n_tile >= N) return;

  const int lane = threadIdx.x;  // 0..31
  const int lo = lane & 15;      // A: row | B: col | acc: N index
  const int hi = lane >> 4;      // which half of the 16-wide K step

  const int n_col = n_tile + lo;
  const int a_row = m_tile + lo;
  const bool a_ok = a_row < M;
  const bool n_ok = n_col < N;

  v8f32 acc = {0, 0, 0, 0, 0, 0, 0, 0};

  // One iteration per E8M0 group (32 K) = two 16-wide WMMA steps.
  for (int k_g = 0; k_g < K; k_g += kGroup) {
    v8f32 grp = {0, 0, 0, 0, 0, 0, 0, 0};
  #pragma unroll
    for (int s = 0; s < 2; s++) {
      const int k_tile = k_g + s * 16;

      v2i32 af = {0, 0};
      if (a_ok)
        __builtin_memcpy(&af, aq + (size_t)a_row * K + k_tile + 8 * hi, 8);

      v2i32 bf = {0, 0};
      if (n_ok) {
        const uint32_t qa = b_q[(size_t)((k_tile >> 3) + hi) * N + n_col];
        uint32_t blo, bhi;
        dequant_mxfp4_8_e4m3(qa, blo, bhi);
        bf[0] = (int)blo;
        bf[1] = (int)bhi;
      }
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
    c[(size_t)m_row * N + n_col] = from_f<T>(acc[i] * ascale[m_row]);
  }
}

#else  // non-RDNA4 device pass: keep symbols so the TU links everywhere.

template <typename T>
__global__ void gemm_mxfp4_wmma_fp8_16x16_1w(const uint8_t*, const float*,
                                             const uint32_t*, const uint8_t*,
                                             T*, int, int, int) {}

#endif  // __HIP__MXFP4_RDNA4__

// ---------------------------------------------------------------------------
// Section 2: skinny GEMV (M <= 8). Weight-bound, so no FP8 conversion: decode
// E2M1 straight to fp32 magnitudes and fold the E8M0 scale per group.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float e2m1_mag(uint32_t nib) {
  // {0,.5,1,1.5,2,3,4,6}, sign from bit 3.
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float v = mag[nib & 7u];
  return (nib & 8u) ? -v : v;
}

template <typename T, int M_MAX>
__global__ void gemv_mxfp4_rdna4(const T* __restrict__ a,
                                 const uint32_t* __restrict__ b_q,
                                 const uint8_t* __restrict__ b_scales,
                                 T* __restrict__ c, int M, int N, int K) {
  const int n = blockIdx.x * blockDim.y + threadIdx.y;  // one wave per column
  if (n >= N) return;
  const int lane = threadIdx.x;  // 0..31
  const int k_words = K >> 3;    // uint32 words along K

  float acc[M_MAX];
#pragma unroll
  for (int m = 0; m < M_MAX; m++) acc[m] = 0.0f;

  for (int w = lane; w < k_words; w += 32) {
    const uint32_t qa = b_q[(size_t)w * N + n];
    const int k0 = w << 3;
    const float wsc =
        mxfp4_e8m0_to_f32(b_scales[(size_t)(k0 / kGroup) * N + n]);

    float dq[8];
#pragma unroll
    for (int i = 0; i < 8; i++) dq[i] = e2m1_mag((qa >> (4 * i)) & 0xFu) * wsc;

    for (int m = 0; m < M; m++) {
      const T* ar = a + (size_t)m * K + k0;
      float s = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; i++) s = fmaf(to_f<T>(ar[i]), dq[i], s);
      acc[m] += s;
    }
  }

  for (int m = 0; m < M; m++) {
    float s = acc[m];
    for (int off = 16; off; off >>= 1) s += __shfl_xor(s, off);
    if (lane == 0) c[(size_t)m * N + n] = from_f<T>(s);
  }
}

}  // namespace mxfp4_rdna4
}  // namespace vllm

// ---------------------------------------------------------------------------
// Torch entry point
// ---------------------------------------------------------------------------
torch::Tensor mxfp4_gemm_rdna4(torch::Tensor const& a, torch::Tensor const& b_q,
                               torch::Tensor const& b_scales) {
  using namespace vllm::mxfp4_rdna4;
  TORCH_CHECK(a.is_cuda() && b_q.is_cuda() && b_scales.is_cuda());
  TORCH_CHECK(a.dim() == 2 && b_q.dim() == 2 && b_scales.dim() == 2);
  TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
              "mxfp4_gemm_rdna4: activation must be fp16 or bf16");

  const int M = a.size(0), K = a.size(1), N = b_q.size(1);
  TORCH_CHECK(K % kGroup == 0, "K must be a multiple of 32");
  TORCH_CHECK(b_q.size(0) == K / 8, "b_q must be [K/8, N]");
  TORCH_CHECK(b_scales.size(0) == K / kGroup, "b_scales must be [K/32, N]");

  const at::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto c = torch::empty({M, N}, a.options());

  const auto* bq = reinterpret_cast<const uint32_t*>(b_q.data_ptr());
  const auto* bs = reinterpret_cast<const uint8_t*>(b_scales.data_ptr());

#define DISPATCH(T, TT)                                               \
  if (M <= kGemvMaxM) {                                               \
    dim3 blk(32, 4);                                                  \
    dim3 grd((N + 3) / 4);                                            \
    gemv_mxfp4_rdna4<TT, kGemvMaxM><<<grd, blk, 0, stream>>>(         \
        reinterpret_cast<const TT*>(a.data_ptr()), bq, bs,            \
        reinterpret_cast<TT*>(c.data_ptr()), M, N, K);                \
  } else {                                                            \
    auto aq = torch::empty({M, K}, a.options().dtype(torch::kUInt8)); \
    auto as = torch::empty({M}, a.options().dtype(torch::kFloat32));  \
    quantize_a_e4m3<TT><<<M, 256, 0, stream>>>(                       \
        reinterpret_cast<const TT*>(a.data_ptr()),                    \
        reinterpret_cast<uint8_t*>(aq.data_ptr()),                    \
        reinterpret_cast<float*>(as.data_ptr()), M, K);               \
    dim3 grd((N + 15) / 16, (M + 15) / 16);                           \
    gemm_mxfp4_wmma_fp8_16x16_1w<TT><<<grd, 32, 0, stream>>>(         \
        reinterpret_cast<const uint8_t*>(aq.data_ptr()),              \
        reinterpret_cast<const float*>(as.data_ptr()), bq, bs,        \
        reinterpret_cast<TT*>(c.data_ptr()), M, N, K);                \
  }

  if (a.scalar_type() == at::kHalf) {
    DISPATCH(half, half)
  } else {
    DISPATCH(bf16_t, __hip_bfloat16)
  }
#undef DISPATCH

  return c;
}
