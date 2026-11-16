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

#include <cstdlib>

#include "qdq_mxfp4_rdna4.cuh"

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

template <typename T, int WM, int WN, bool KSPLIT>
__global__ void __launch_bounds__(32 * WM * WN)
    gemm_mxfp4_wmma_fp8_tiled(const uint8_t* __restrict__ aq,
                              const float* __restrict__ ascale,
                              const uint32_t* __restrict__ b_q,
                              const uint8_t* __restrict__ b_scales,
                              T* __restrict__ c, int M, int N, int K,
                              float* __restrict__ c_f32, int k_per_split) {
  // Block tile (16*WM) x (16*WN); each wave owns one 16x16 WMMA tile.
  //
  // The packed B words of one 32-wide E8M0 group (4 words per column) are
  // loaded and decoded ONCE per block and staged in LDS, so the WM waves that
  // share an N column no longer each re-decode it. LDS layout is
  // [octet][n] -> 8 E4M3 bytes: the decode threads write consecutive n and the
  // WMMA B-fragment reads (lane -> n, octet = 2*step + half) are 8-byte loads
  // over consecutive n, so both sides are bank-conflict free without padding.
  //
  // Two LDS buffers: group g+1's global loads (B words, A fragments, E8M0)
  // are issued before group g's WMMAs and decoded into the other buffer after
  // them, with one __syncthreads per group covering both hazards.
  constexpr int BN = 16 * WN;
  constexpr int NT = 32 * WM * WN;
  constexpr int WORDS = (BN * 4 + NT - 1) / NT;  // B words per thread / group

  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  const int lo = lane & 15;  // A: row | B: col | acc: N index
  const int hi = lane >> 4;  // which half of the 16-wide K step
  const int bn0 = blockIdx.x * BN;
  const int m_tile = blockIdx.y * (16 * WM) + 16 * (wave / WN);
  const int n_tile = bn0 + 16 * (wave % WN);
  const int n_loc =
      16 * (wave % WN) + lo;  // this lane's column inside the block
  // No per-wave early return: every thread takes part in the block barriers
  // and the shared decode. Out-of-range work is masked instead.
  const bool tile_ok = (m_tile < M) && (n_tile < N);
  const int n_col = n_tile + lo;
  const int a_row = m_tile + lo;
  const bool a_ok = tile_ok && a_row < M;
  const bool n_ok = tile_ok && n_col < N;

  __shared__ uint2 b_lds[2][4][BN];

  const int k_begin = KSPLIT ? (int)blockIdx.z * k_per_split : 0;
  const int k_end =
      KSPLIT ? ((blockIdx.z == gridDim.z - 1) ? K : k_begin + k_per_split) : K;
  const int ngroups = (k_end - k_begin) / kGroup;

  uint32_t qa_nx[WORDS];
  v2i32 a_nx[2];
  float wsc_nx = 0.0f;

  auto load_group = [&](int g) {
    const int k_g = k_begin + g * kGroup;
  #pragma unroll
    for (int w = 0; w < WORDS; w++) {
      const int idx = tid + w * NT;
      uint32_t v = 0;
      if (idx < BN * 4) {
        const int cc = idx % BN, o = idx / BN;
        const int n = bn0 + cc;
        if (n < N) v = b_q[(size_t)((k_g >> 3) + o) * N + n];
      }
      qa_nx[w] = v;
    }
  #pragma unroll
    for (int s = 0; s < 2; s++) {
      v2i32 af = {0, 0};
      if (a_ok)
        __builtin_memcpy(&af, aq + (size_t)a_row * K + k_g + 16 * s + 8 * hi,
                         8);
      a_nx[s] = af;
    }
    wsc_nx =
        n_ok ? mxfp4_e8m0_to_f32(b_scales[(size_t)(k_g / kGroup) * N + n_col])
             : 0.0f;
  };
  auto store_group = [&](int buf) {
  #pragma unroll
    for (int w = 0; w < WORDS; w++) {
      const int idx = tid + w * NT;
      if (idx < BN * 4) {
        const int cc = idx % BN, o = idx / BN;
        uint32_t blo, bhi;
        dequant_mxfp4_8_e4m3(qa_nx[w], blo, bhi);
        b_lds[buf][o][cc] = make_uint2(blo, bhi);
      }
    }
  };

  v8f32 acc = {0, 0, 0, 0, 0, 0, 0, 0};
  v2i32 a_cur[2] = {{0, 0}, {0, 0}};
  float wsc_cur = 0.0f;

  if (ngroups > 0) {
    load_group(0);
    store_group(0);
    a_cur[0] = a_nx[0];
    a_cur[1] = a_nx[1];
    wsc_cur = wsc_nx;
  }
  __syncthreads();

  for (int g = 0; g < ngroups; g++) {
    const int buf = g & 1;
    const bool has_next = (g + 1 < ngroups);
    if (has_next) load_group(g + 1);  // in flight during this group's WMMAs

    v8f32 grp = {0, 0, 0, 0, 0, 0, 0, 0};
  #pragma unroll
    for (int s = 0; s < 2; s++) {
      const uint2 bw = b_lds[buf][2 * s + hi][n_loc];
      v2i32 bf;
      bf[0] = (int)bw.x;
      bf[1] = (int)bw.y;
      grp = wmma_fp8(a_cur[s], bf, grp);
    }
    // E8M0 is constant across the group and across this lane's 8 outputs,
    // because the accumulator's N index is fixed per lane.
  #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = fmaf(grp[i], wsc_cur, acc[i]);

    if (has_next) {
      store_group(buf ^ 1);
      a_cur[0] = a_nx[0];
      a_cur[1] = a_nx[1];
      wsc_cur = wsc_nx;
    }
    __syncthreads();
  }

  if (!n_ok) return;  // past the last barrier, so returning here is safe
  // gfx12 accumulator: lane -> n = lo, slot i -> m = 8*hi + i.
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const int m_row = m_tile + 8 * hi + i;
    if (m_row >= M) continue;
    // Per-token scale distributes over the K sum, so apply it before the
    // split reduction. gridDim.z > 1 reduces into a zeroed fp32 workspace via
    // native global_atomic_add_f32 (no CAS).
    const float v = acc[i] * ascale[m_row];
    if constexpr (KSPLIT)
      unsafeAtomicAdd(c_f32 + (size_t)m_row * N + n_col, v);
    else
      c[(size_t)m_row * N + n_col] = from_f<T>(v);
  }
}

#else  // non-RDNA4 device pass: keep symbols so the TU links everywhere.

template <typename T, int WM, int WN, bool KSPLIT>
__global__ void gemm_mxfp4_wmma_fp8_tiled(const uint8_t*, const float*,
                                          const uint32_t*, const uint8_t*, T*,
                                          int, int, int, float*, int) {}

template <typename T, int WM, int WN, bool KSPLIT>
__global__ void gemm_mxfp4_wmma_fp8_direct(const uint8_t*, const float*,
                                           const uint32_t*, const uint8_t*, T*,
                                           int, int, int, float*, int) {}

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

// ---------------------------------------------------------------------------
// Torch entry point
// ---------------------------------------------------------------------------
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
  TORCH_CHECK(a.dim() == 2 && b_q.dim() == 2 && b_scales.dim() == 2);
  TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16,
              "mxfp4_gemm_rdna4: activation must be fp16 or bf16");

  const int M = a.size(0), K = a.size(1), N = b_q.size(1);
  TORCH_CHECK(K % kGroup == 0, "K must be a multiple of 32");
  TORCH_CHECK(b_q.size(0) == K / 8, "b_q must be [K/8, N]");
  TORCH_CHECK(b_scales.size(0) == K / kGroup, "b_scales must be [K/32, N]");
  TORCH_CHECK(a_fp8.size(0) == M && a_fp8.size(1) == K, "a_fp8 must be [M, K]");
  TORCH_CHECK(a_scale.numel() == M, "a_scale must have M entries");

  const at::cuda::OptionalCUDAGuard guard(a.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto c = torch::empty({M, N}, a.options());

  const auto* bq = reinterpret_cast<const uint32_t*>(b_q.data_ptr());
  const auto* bs = reinterpret_cast<const uint8_t*>(b_scales.data_ptr());

#define LAUNCH_ONE(KERN, TT, WM, WN, SPLIT)                      \
  KERN<TT, WM, WN, SPLIT><<<grd, 32 * (WM) * (WN), 0, stream>>>( \
      aqp, asp, bq, bs, cp, M, N, K, wsp, k_per)

#define LAUNCH_TILED(TT, WM, WN)                                               \
  do {                                                                         \
    const int splits = choose_ksplit(M, N, K, WM, WN);                         \
    const int k_per = (splits > 1) ? ((K / splits) / kGroup) * kGroup : K;     \
    dim3 grd((N + 16 * (WN) - 1) / (16 * (WN)),                                \
             (M + 16 * (WM) - 1) / (16 * (WM)), splits);                       \
    torch::Tensor ws;                                                          \
    float* wsp = nullptr;                                                      \
    if (splits > 1) {                                                          \
      ws = torch::zeros({M, N}, a.options().dtype(torch::kFloat32));           \
      wsp = ws.data_ptr<float>();                                              \
    }                                                                          \
    const bool lds = use_lds_kernel(M);                                        \
    if (lds && splits > 1)                                                     \
      LAUNCH_ONE(gemm_mxfp4_wmma_fp8_tiled, TT, WM, WN, true);                 \
    else if (lds)                                                              \
      LAUNCH_ONE(gemm_mxfp4_wmma_fp8_tiled, TT, WM, WN, false);                \
    else if (splits > 1)                                                       \
      LAUNCH_ONE(gemm_mxfp4_wmma_fp8_direct, TT, WM, WN, true);                \
    else                                                                       \
      LAUNCH_ONE(gemm_mxfp4_wmma_fp8_direct, TT, WM, WN, false);               \
    if (splits > 1) {                                                          \
      const size_t n_el = (size_t)M * N;                                       \
      f32_to_t<TT>                                                             \
          <<<(unsigned)((n_el + 255) / 256), 256, 0, stream>>>(wsp, cp, n_el); \
    }                                                                          \
  } while (0)

#define DISPATCH(T, TT)                                                   \
  if (M <= kGemvMaxM) {                                                   \
    dim3 blk(32, 4);                                                      \
    dim3 grd((N + 3) / 4);                                                \
    gemv_mxfp4_rdna4<TT, kGemvMaxM><<<grd, blk, 0, stream>>>(             \
        reinterpret_cast<const TT*>(a.data_ptr()), bq, bs,                \
        reinterpret_cast<TT*>(c.data_ptr()), M, N, K);                    \
  } else {                                                                \
    const auto* aqp = reinterpret_cast<const uint8_t*>(a_fp8.data_ptr()); \
    const auto* asp = reinterpret_cast<const float*>(a_scale.data_ptr()); \
    auto* cp = reinterpret_cast<TT*>(c.data_ptr());                       \
    int cfg = tile_cfg_override();                                        \
    if (cfg == 0) cfg = (M >= 256) ? 3 : ((M >= 32) ? 2 : 1);             \
    if (cfg == 3) {                                                       \
      LAUNCH_TILED(TT, 4, 2);                                             \
    } else if (cfg == 2) {                                                \
      LAUNCH_TILED(TT, 2, 2);                                             \
    } else {                                                              \
      LAUNCH_TILED(TT, 1, 1);                                             \
    }                                                                     \
  }

  if (a.scalar_type() == at::kHalf) {
    DISPATCH(half, half)
  } else {
    DISPATCH(bf16_t, __hip_bfloat16)
  }
#undef DISPATCH
#undef LAUNCH_TILED
#undef LAUNCH_ONE

  return c;
}
