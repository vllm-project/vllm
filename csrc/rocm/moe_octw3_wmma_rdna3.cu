// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Prefill path for the Octave-W3 MoE on RDNA3 (gfx1100): WMMA over tiles of
// one expert's tokens. moe_octw3_rdna3.cu (fdot2, 4 tokens per block) stays
// the decode path; it re-decodes the expert weights every 4 tokens, which
// dominates once an expert sees tens of tokens.
//
// Kept in its own translation unit like q_gemm_rdna3_wmma.cu.
//
// The block Hadamard on the input is applied once per row by
// octw3_rotate_rdna3 (which also folds SiLU-and-mul for the down projection)
// instead of inside the GEMM, so the GEMM is a plain gather-A x dequant-B
// product. Weights are dequantised to fp16 (s * level) into LDS.
//
// Weight layout as in moe_octw3_rdna3.cu: w [E, K/32, 3, N] int32,
// scales [E, K/32, N] fp16.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__HIPCC__) && defined(__gfx1100__)
  #define __HIP__RDNA3__
#endif

namespace vllm {
namespace moe_octw3_wmma_rdna3 {

constexpr uint32_t LV_LO = 0x4e600cd8u;
constexpr uint32_t LV_HI_POS = 0x403d3a33u;
constexpr uint32_t LV_HI_NEG = 0xc0bdbab3u;

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

using v16fp16 = _Float16 __attribute__((ext_vector_type(16)));
using v8fp32 = float __attribute__((ext_vector_type(8)));

// Same code layout and lookup as moe_octw3_rdna3.cu.
__forceinline__ __device__ void octw3_codes(uint32_t x0, uint32_t x1,
                                            uint32_t x2, uint32_t (&y)[8]) {
  constexpr uint32_t M3 = 0x07070707u;
  y[0] = x0 & M3;
  y[1] = (x0 >> 3) & M3;
  y[2] = x1 & M3;
  y[3] = (x1 >> 3) & M3;
  y[4] = x2 & M3;
  y[5] = (x2 >> 3) & M3;
  y[6] = ((x0 >> 6) & 0x03030303u) | ((x1 >> 4) & 0x04040404u);
  y[7] = ((x2 >> 6) & 0x03030303u) | ((x1 >> 5) & 0x04040404u);
}

__forceinline__ __device__ void octw3_lookup(uint32_t y, uint32_t& w01,
                                             uint32_t& w23) {
  const uint32_t lo = __builtin_amdgcn_perm(LV_LO, LV_LO, y);
  const uint32_t hi = __builtin_amdgcn_perm(LV_HI_NEG, LV_HI_POS, y);
  w01 = __builtin_amdgcn_perm(hi, lo, 0x05010400u);
  w23 = __builtin_amdgcn_perm(hi, lo, 0x07030602u);
}

// y[part * QPER + q] with compile-time register indices only (a runtime
// index into y spills it to scratch).
template <int QPER>
__forceinline__ __device__ void pick_codes(const uint32_t (&y)[8], int part,
                                           uint32_t (&o)[QPER]) {
  #pragma unroll
  for (int q = 0; q < QPER; ++q) o[q] = y[q];
  #pragma unroll
  for (int p = 1; p < 8 / QPER; ++p)
  #pragma unroll
    for (int q = 0; q < QPER; ++q)
      if (part == p) o[q] = y[p * QPER + q];
}

__forceinline__ __device__ half silu_mul(half g, half u) {
  const float gf = __half2float(g);
  return __float2half_rn(
      __half2float(__float2half_rn(gf / (1.0f + __expf(-gf)))) *
      __half2float(u));
}

__forceinline__ __device__ void atomic_add_pk4_f16(half* addr, half2 v01,
                                                   half2 v23) {
  unsigned long long* addr_u = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_u;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(addr_u, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

// y[r] = Hb x[r] / sqrt(had), x[r] = silu(gate) * up when silu. Each thread
// owns 8 consecutive columns: Hadamard bits 0-2 in registers, the rest
// across lanes, so a block of up to 256 stays inside one wave.
__global__ void __launch_bounds__(256)
    rotate_kernel(const half* __restrict__ x, half* __restrict__ y, int rows,
                  int k, int x_stride, bool silu, int had) {
  const int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
  const int chunks = k / 8;
  if (idx >= (int64_t)rows * chunks) return;
  const int r = idx / chunks;
  const int c = (idx % chunks) * 8;
  const half* xr = x + (int64_t)r * x_stride + c;

  float v[8];
  const uint4 u = *reinterpret_cast<const uint4*>(xr);
  const half* h = reinterpret_cast<const half*>(&u);
  if (silu) {
    const uint4 ug = *reinterpret_cast<const uint4*>(xr + k);
    const half* hu = reinterpret_cast<const half*>(&ug);
  #pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __half2float(silu_mul(h[i], hu[i]));
  } else {
  #pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __half2float(h[i]);
  }

  #pragma unroll
  for (int b = 1; b < 8; b <<= 1) {
  #pragma unroll
    for (int i = 0; i < 8; ++i)
      if (b < had && !(i & b)) {
        const float p = v[i], q = v[i | b];
        v[i] = p + q;
        v[i | b] = p - q;
      }
  }
  const int lane = threadIdx.x & 31;
  for (int b = 1; b * 8 < had; b <<= 1) {
  #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float p = __shfl_xor(v[i], b, 32);
      v[i] = (lane & b) ? p - v[i] : v[i] + p;
    }
  }

  const float rnorm = had > 0 ? rsqrtf((float)had) : 1.0f;
  uint4 o;
  half* ho = reinterpret_cast<half*>(&o);
  #pragma unroll
  for (int i = 0; i < 8; ++i) ho[i] = __float2half_rn(v[i] * rnorm);
  *reinterpret_cast<uint4*>(y + (int64_t)r * k + c) = o;
}

// C[out_row, n] (+)= sum_k A[a_row, k] * W_e[k, n] for one BM-token block of
// moe_align_block_size output (all rows share expert e) and BN columns.
// a_row = token_id / a_div. Without output_topk the row is stored as is;
// with it, rows are scaled by topk_weights and atomically summed into
// token_id / output_topk.
//
// WM x WN waves, each owning a (BM/WM) x (BN/WN) patch of 16x16 WMMA tiles.
// K advances one 32-wide group per step: A and the dequantised weights go
// through double-buffered LDS, with the next group's global loads issued
// before the current group's WMMAs.
template <int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32)
    gemm_kernel(const half* __restrict__ a, half* __restrict__ c,
                const uint32_t* __restrict__ w, const half* __restrict__ scales,
                const float* __restrict__ topk_weights,
                const int32_t* __restrict__ sorted_token_ids,
                const int32_t* __restrict__ expert_ids,
                const int32_t* __restrict__ num_tokens_post_padded,
                int num_valid, int size_n, int size_k, int a_div,
                int output_topk) {
  constexpr int NT = WM * WN * 32;
  constexpr int TM = BM / WM / 16;
  constexpr int TN = BN / WN / 16;
  constexpr int KP = 32 + 8;  // padded LDS row of one group
  constexpr int CP = BN + 4;
  constexpr int P = NT / BN;  // threads per weight column
  constexpr int QPER = 8 / P;
  static_assert(NT % BN == 0 && (P == 1 || P == 2 || P == 4));

  const int t = threadIdx.x;
  const int wave = t >> 5;
  const int lane = t & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;
  const int wm = wave % WM;
  const int wn = wave / WM;

  // N tiles of one token block are adjacent in launch order so they share
  // its A rows in L2.
  const int m_tile = blockIdx.y;
  const int m_base = m_tile * BM;
  const int n_base = blockIdx.x * BN;
  if (m_base >= num_tokens_post_padded[0]) return;
  const int e = expert_ids[m_tile];
  if (e < 0) return;

  const int groups = size_k / 32;
  const uint32_t* w_e = w + (int64_t)e * groups * 3 * size_n;
  const half* s_e = scales + (int64_t)e * groups * size_n;

  constexpr int AB_BYTES = 2 * (BM + BN) * KP * 2;
  constexpr int C_BYTES = BM * CP * 4;
  __shared__ __attribute__((
      aligned(16))) char smem[AB_BYTES > C_BYTES ? AB_BYTES : C_BYTES];
  half(*sa)[BM * KP] = reinterpret_cast<half(*)[BM * KP]>(smem);
  half(*sb)[BN * KP] =
      reinterpret_cast<half(*)[BN * KP]>(smem + 2 * BM * KP * 2);
  __shared__ int32_t s_tok[BM];

  if (t < BM) s_tok[t] = sorted_token_ids[m_base + t];
  __syncthreads();

  // A: BM rows x four 16-byte chunks per group.
  constexpr int A_CHUNKS = BM * 4;
  constexpr int A_PER_T = (A_CHUNKS + NT - 1) / NT;
  const half* a_src[A_PER_T];
  int a_dst[A_PER_T];
  #pragma unroll
  for (int i = 0; i < A_PER_T; ++i) {
    const int idx = t + i * NT;
    const int row = idx >> 2;
    a_src[i] = nullptr;
    a_dst[i] = -1;
    if (idx < A_CHUNKS) {
      a_dst[i] = row * KP + (idx & 3) * 8;
      const int tok = s_tok[row];
      if (tok < num_valid)
        a_src[i] = a + (int64_t)(tok / a_div) * size_k + (idx & 3) * 8;
    }
  }

  // Weights: thread -> (column, part); a part covers 32 / P codes.
  const int bcol = t % BN;
  const int bpart = t / BN;
  const int n_glob = n_base + bcol;
  const bool bvalid = n_glob < size_n;

  uint4 ra[A_PER_T];
  uint32_t rw0, rw1, rw2;
  half rs;

  auto gload = [&](int g) {
  #pragma unroll
    for (int i = 0; i < A_PER_T; ++i) {
      ra[i] = make_uint4(0, 0, 0, 0);
      if (a_src[i]) ra[i] = *reinterpret_cast<const uint4*>(a_src[i] + g * 32);
    }
    if (bvalid) {
      const uint32_t* wp = w_e + (int64_t)g * 3 * size_n + n_glob;
      rw0 = wp[0];
      rw1 = wp[size_n];
      rw2 = wp[2 * size_n];
      rs = s_e[(int64_t)g * size_n + n_glob];
    } else {
      rw0 = rw1 = rw2 = 0;
      rs = __float2half_rn(0.0f);
    }
  };
  auto lstore = [&](int buf) {
  #pragma unroll
    for (int i = 0; i < A_PER_T; ++i)
      if (a_dst[i] >= 0) *reinterpret_cast<uint4*>(&sa[buf][a_dst[i]]) = ra[i];
    uint32_t y[8];
    octw3_codes(rw0, rw1, rw2, y);
    uint32_t ys[QPER];
    pick_codes<QPER>(y, bpart, ys);
    const half2 s2 = __half2half2(rs);
    uint32_t wq[QPER * 2];
  #pragma unroll
    for (int q = 0; q < QPER; ++q) {
      uint32_t w01, w23;
      octw3_lookup(ys[q], w01, w23);
      wq[2 * q] = __builtin_bit_cast(
          uint32_t, __hmul2(__builtin_bit_cast(half2, w01), s2));
      wq[2 * q + 1] = __builtin_bit_cast(
          uint32_t, __hmul2(__builtin_bit_cast(half2, w23), s2));
    }
    uint4* dst =
        reinterpret_cast<uint4*>(&sb[buf][bcol * KP + bpart * QPER * 4]);
  #pragma unroll
    for (int i = 0; i < QPER / 2; ++i)
      dst[i] =
          make_uint4(wq[4 * i], wq[4 * i + 1], wq[4 * i + 2], wq[4 * i + 3]);
  };

  v8fp32 acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
  #pragma unroll
    for (int j = 0; j < TN; ++j)
  #pragma unroll
      for (int r = 0; r < 8; ++r) acc[i][j][r] = 0.0f;

  gload(0);
  lstore(0);
  __syncthreads();
  for (int g = 0; g < groups; ++g) {
    const int buf = g & 1;
    if (g + 1 < groups) gload(g + 1);
  #pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      v16fp16 af[TM], bf[TN];
  #pragma unroll
      for (int i = 0; i < TM; ++i)
        af[i] = *reinterpret_cast<const v16fp16*>(
            &sa[buf][((wm * TM + i) * 16 + lane_lo) * KP + ks * 16]);
  #pragma unroll
      for (int j = 0; j < TN; ++j)
        bf[j] = *reinterpret_cast<const v16fp16*>(
            &sb[buf][((wn * TN + j) * 16 + lane_lo) * KP + ks * 16]);
  #pragma unroll
      for (int i = 0; i < TM; ++i)
  #pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[i], bf[j],
                                                                 acc[i][j]);
    }
    if (g + 1 < groups) lstore(buf ^ 1);
    __syncthreads();
  }

  // C fragment: lane holds column lane_lo, rows 2 * r + lane_hi. Restage
  // through LDS so each thread writes four adjacent columns.
  float* sc = reinterpret_cast<float*>(smem);
  #pragma unroll
  for (int i = 0; i < TM; ++i)
  #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = (wn * TN + j) * 16 + lane_lo;
  #pragma unroll
      for (int r = 0; r < 8; ++r)
        sc[((wm * TM + i) * 16 + 2 * r + lane_hi) * CP + col] = acc[i][j][r];
    }
  __syncthreads();
  for (int idx = t; idx < BM * BN / 4; idx += NT) {
    const int row = idx / (BN / 4);
    const int col = (idx % (BN / 4)) * 4;
    const int tok = s_tok[row];
    if (tok >= num_valid || n_base + col >= size_n) continue;
    const float* v = &sc[row * CP + col];
    float tw = 1.0f;
    if (topk_weights != nullptr) tw = topk_weights[tok];
    const half2 v01 = __floats2half2_rn(v[0] * tw, v[1] * tw);
    const half2 v23 = __floats2half2_rn(v[2] * tw, v[3] * tw);
    if (output_topk > 0) {
      atomic_add_pk4_f16(
          c + (int64_t)(tok / output_topk) * size_n + n_base + col, v01, v23);
    } else {
      uint2 u;
      u.x = __builtin_bit_cast(uint32_t, v01);
      u.y = __builtin_bit_cast(uint32_t, v23);
      *reinterpret_cast<uint2*>(c + (int64_t)tok * size_n + n_base + col) = u;
    }
  }
}

#else  // non-RDNA3: empty stubs for symbol parity

__global__ void rotate_kernel(const half*, half*, int, int, int, bool, int) {}

template <int BM, int BN, int WM, int WN>
__global__ void gemm_kernel(const half*, half*, const uint32_t*, const half*,
                            const float*, const int32_t*, const int32_t*,
                            const int32_t*, int, int, int, int, int) {}

#endif  // __HIP__RDNA3__ || !__HIP_DEVICE_COMPILE__

}  // namespace moe_octw3_wmma_rdna3
}  // namespace vllm

// y [rows, K] = Hb x / sqrt(had), x = a [rows, K] or, with silu_mul,
// silu(a[:, :K]) * a[:, K:2K]. had in {0, 8, ..., 256}; K / 8 must be a
// multiple of had / 8.
void octw3_rotate_rdna3(torch::Tensor x, torch::Tensor y, bool silu_mul,
                        int64_t had) {
  TORCH_CHECK(
      x.scalar_type() == torch::kHalf && y.scalar_type() == torch::kHalf,
      "octw3_rotate_rdna3 runs in float16");
  TORCH_CHECK(x.dim() == 2 && y.dim() == 2 && y.is_contiguous() &&
                  x.stride(1) == 1 && x.size(0) == y.size(0),
              "x [rows, K or 2K] with unit column stride, y [rows, K]");
  const int rows = y.size(0);
  const int k = y.size(1);
  TORCH_CHECK(x.size(1) == (silu_mul ? 2 * k : k), "x has the wrong width");
  TORCH_CHECK(k % 8 == 0 && x.stride(0) % 8 == 0,
              "K and the row stride must be multiples of 8");
  TORCH_CHECK(had == 0 || (had >= 8 && had <= 256 && (had & (had - 1)) == 0 &&
                           (k / 8) % (had / 8) == 0),
              "unsupported Hadamard block ", had, " for K = ", k);
  if (rows == 0) return;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t n = (int64_t)rows * (k / 8);
  vllm::moe_octw3_wmma_rdna3::
      rotate_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
          (const half*)x.data_ptr(), (half*)y.data_ptr(), rows, k,
          (int)x.stride(0), silu_mul, (int)had);
}

// Inputs:
//   a           [rows_a, K] half, already rotated (octw3_rotate_rdna3)
//   c           output_topk == 0: [M*top_k, N], every row written;
//               output_topk > 0:  [M, N], pre-zeroed, rows summed into
//   w, scales   [E, K/32, 3, N] int32, [E, K/32, N] half
//   topk_weights [M*top_k] float32 or empty
//   sorted_token_ids, expert_ids, num_tokens_post_padded:
//               moe_align_block_size with block_size_m (32 or 64)
//   a_div       A row of token id t is t / a_div
void moe_octw3_wmma_gemm_rdna3(
    torch::Tensor a, torch::Tensor c, torch::Tensor w, torch::Tensor scales,
    torch::Tensor topk_weights, torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded,
    int64_t a_div, int64_t output_topk, int64_t block_size_m) {
  TORCH_CHECK(a.scalar_type() == torch::kHalf &&
                  c.scalar_type() == torch::kHalf &&
                  scales.scalar_type() == torch::kHalf,
              "Octave-W3 MoE runs in float16");
  TORCH_CHECK(
      a.dim() == 2 && a.is_contiguous() && c.dim() == 2 && c.is_contiguous(),
      "a and c must be contiguous 2D");
  TORCH_CHECK(w.dim() == 4 && w.size(2) == 3, "w must be [E, K/32, 3, N]");
  const int size_k = (int)w.size(1) * 32;
  const int size_n = (int)w.size(3);
  TORCH_CHECK(a.size(1) == size_k, "a must have K = ", size_k, " columns");
  TORCH_CHECK(c.size(1) == size_n && size_n % 4 == 0,
              "c must have N columns, N a multiple of 4");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int num_valid = (int)(a.size(0) * a_div);
  const int num_m_tiles = (int)(sorted_token_ids.size(0) / block_size_m);
  const float* tw =
      topk_weights.numel() > 0 ? topk_weights.data_ptr<float>() : nullptr;

#define OCTW3_WMMA_LAUNCH(BM, BN, WM, WN)                                 \
  vllm::moe_octw3_wmma_rdna3::gemm_kernel<BM, BN, WM, WN>                 \
      <<<dim3((size_n + BN - 1) / BN, num_m_tiles), WM * WN * 32, 0,      \
         stream>>>((const half*)a.data_ptr(), (half*)c.data_ptr(),        \
                   (const uint32_t*)w.data_ptr<int32_t>(),                \
                   (const half*)scales.data_ptr(), tw,                    \
                   sorted_token_ids.data_ptr<int32_t>(),                  \
                   expert_ids.data_ptr<int32_t>(),                        \
                   num_tokens_post_padded.data_ptr<int32_t>(), num_valid, \
                   size_n, size_k, (int)a_div, (int)output_topk)

  switch (block_size_m) {
    case 32:
      OCTW3_WMMA_LAUNCH(32, 64, 2, 2);
      break;
    case 64:
      OCTW3_WMMA_LAUNCH(64, 64, 2, 2);
      break;
    default:
      TORCH_CHECK(false, "block_size_m must be 32 or 64, got ", block_size_m);
  }
#undef OCTW3_WMMA_LAUNCH
}
