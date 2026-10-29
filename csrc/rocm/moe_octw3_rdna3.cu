// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused MoE kernel for Octave-W3 experts on RDNA3 (gfx1100).
//
// Each expert weight is W ~= (s * LV[c]) . Hb^T: 3-bit codes c over the eight
// Lloyd-Max levels of the unit normal, one fp16 scale s per output row and
// group of 32 inputs, and a block Hadamard Hb along K. The kernel rotates its
// input (y = Q . (Hb x)) while staging it in LDS, so the dequant is a pure
// codebook lookup with no zero point.
//
// Same routing, tiling and atomic write-back as moe_q_gemm_rdna3.cu: 256
// threads, a 256-wide K tile, 4 output columns per thread.
//
// Kernel layout (built at load time from the planar checkpoint format):
//   w [E, K/32, 3, N] int32: three words per (group, column). A code byte
//     holds sign (bit 2) and magnitude index (bits 0-1); see octw3_codes().
//   s [E, K/32, N] fp16.

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
namespace moe_octw3_rdna3 {

#define OCTW3_BLOCK_K 256
#define OCTW3_THREADS 256

// fp16 bit patterns of the four magnitudes (0.2451, 0.7560, 1.3439, 2.1520)
// split into low bytes, positive high bytes and negative high bytes.
constexpr uint32_t LV_LO = 0x4e600cd8u;
constexpr uint32_t LV_HI_POS = 0x403d3a33u;
constexpr uint32_t LV_HI_NEG = 0xc0bdbab3u;

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

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

// Eight 4-code vectors (code k of vector q is element 4q + k of the group)
// out of the three words of one (group, column).
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

// Four codes -> two half2 weights (elements 0,1 and 2,3).
__forceinline__ __device__ void octw3_lookup(uint32_t y, half2& w01,
                                             half2& w23) {
  const uint32_t lo = __builtin_amdgcn_perm(LV_LO, LV_LO, y);
  const uint32_t hi = __builtin_amdgcn_perm(LV_HI_NEG, LV_HI_POS, y);
  const uint32_t a = __builtin_amdgcn_perm(hi, lo, 0x05010400u);
  const uint32_t b = __builtin_amdgcn_perm(hi, lo, 0x07030602u);
  w01 = __builtin_bit_cast(half2, a);
  w23 = __builtin_bit_cast(half2, b);
}

template <int BLOCK_SIZE_M>
__global__ void __launch_bounds__(OCTW3_THREADS)
    moe_octw3_kernel(const half* __restrict__ a, half* __restrict__ c,
                     const uint32_t* __restrict__ w,
                     const half* __restrict__ scales,
                     const float* __restrict__ topk_weights,
                     const int32_t* __restrict__ sorted_token_ids,
                     const int32_t* __restrict__ expert_ids,
                     const int32_t* __restrict__ num_tokens_post_padded,
                     const int size_m, const int size_n, const int size_k,
                     const int top_k, const bool mul_topk_weight,
                     const int output_topk, const int a_stride,
                     const bool a_silu_mul, const int had) {
  const int t = threadIdx.x;
  const int token_block = blockIdx.x;
  const int offset_n = blockIdx.y * OCTW3_THREADS * 4;
  const int offset_k = blockIdx.z * OCTW3_BLOCK_K;
  const int end_k = min(offset_k + OCTW3_BLOCK_K, size_k);
  const int n = offset_n + t * 4;

  if (token_block * BLOCK_SIZE_M >= num_tokens_post_padded[0]) return;
  const int expert_id = expert_ids[token_block];
  if (expert_id == -1) return;

  const int groups = size_k / 32;
  const uint32_t* w_e = w + (int64_t)expert_id * groups * 3 * size_n;
  const half* s_e = scales + (int64_t)expert_id * groups * size_n;

  constexpr int LDS_PAD = 8;
  __shared__ half block_a[BLOCK_SIZE_M][OCTW3_BLOCK_K + LDS_PAD];
  __shared__ float rot[BLOCK_SIZE_M][OCTW3_BLOCK_K];

  const int offset_m_base = token_block * BLOCK_SIZE_M;
  const int lane_had = had < 32 ? had : 32;
  const float rnorm = had > 0 ? rsqrtf((float)had) : 1.0f;

  float v[BLOCK_SIZE_M];
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    const int token_row = sorted_token_ids[offset_m_base + m] / top_k;
    float x = 0.0f;
    if (offset_k + t < end_k && token_row < size_m) {
      const half* a_row = a + (int64_t)token_row * a_stride + offset_k + t;
      x = __half2float(a_silu_mul ? silu_mul(a_row[0], a_row[size_k])
                                  : a_row[0]);
    }
    // Hb x within the wave: Sylvester butterflies over the lane bits.
    for (int b = 1; b < lane_had; b <<= 1) {
      const float p = __shfl_xor(x, b, 32);
      x = (t & b) ? p - x : x + p;
    }
    v[m] = x;
  }
  if (had > 32) {
    // The remaining Hadamard bits are wave bits: combine across waves.
  #pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m) rot[m][t] = v[m];
    __syncthreads();
    const int nb = had >> 5;
    const int wb = (t >> 5) & (nb - 1);
    const int base = t - (wb << 5);
  #pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m) {
      float x = 0.0f;
      for (int j = 0; j < nb; ++j) {
        const float p = rot[m][base + (j << 5)];
        x += (__popc(wb & j) & 1) ? -p : p;
      }
      v[m] = x;
    }
  }
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m)
    block_a[m][t] = __float2half_rn(v[m] * rnorm);
  __syncthreads();

  if (n >= size_n) return;

  const uint32_t* w_ptr = w_e + (int64_t)(offset_k / 32) * 3 * size_n + n;
  const half* s_ptr = s_e + (int64_t)(offset_k / 32) * size_n + n;

  float acc[BLOCK_SIZE_M][4];
  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m)
  #pragma unroll
    for (int j = 0; j < 4; ++j) acc[m][j] = 0.0f;

  for (int k = offset_k; k < end_k; k += 32) {
    const uint4 x0 = *reinterpret_cast<const uint4*>(w_ptr);
    const uint4 x1 = *reinterpret_cast<const uint4*>(w_ptr + size_n);
    const uint4 x2 = *reinterpret_cast<const uint4*>(w_ptr + 2 * size_n);
    const uint2 s4 = *reinterpret_cast<const uint2*>(s_ptr);
    w_ptr += 3 * size_n;
    s_ptr += size_n;

    uint32_t y[4][8];
    octw3_codes(x0.x, x1.x, x2.x, y[0]);
    octw3_codes(x0.y, x1.y, x2.y, y[1]);
    octw3_codes(x0.z, x1.z, x2.z, y[2]);
    octw3_codes(x0.w, x1.w, x2.w, y[3]);

    float part[BLOCK_SIZE_M][4];
  #pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m)
  #pragma unroll
      for (int j = 0; j < 4; ++j) part[m][j] = 0.0f;

    const int a_off = k - offset_k;
  #pragma unroll
    for (int q = 0; q < 8; ++q) {
      half2 wq[4][2];
  #pragma unroll
      for (int j = 0; j < 4; ++j) octw3_lookup(y[j][q], wq[j][0], wq[j][1]);
  #pragma unroll
      for (int m = 0; m < BLOCK_SIZE_M; ++m) {
        const half2* a2 =
            reinterpret_cast<const half2*>(&block_a[m][a_off + 4 * q]);
        const half2 a01 = a2[0], a23 = a2[1];
  #pragma unroll
        for (int j = 0; j < 4; ++j) {
          part[m][j] = __builtin_amdgcn_fdot2(wq[j][0], a01, part[m][j], false);
          part[m][j] = __builtin_amdgcn_fdot2(wq[j][1], a23, part[m][j], false);
        }
      }
    }

    const half2 s01 = __builtin_bit_cast(half2, s4.x);
    const half2 s23 = __builtin_bit_cast(half2, s4.y);
    const float sf[4] = {__low2float(s01), __high2float(s01), __low2float(s23),
                         __high2float(s23)};
  #pragma unroll
    for (int m = 0; m < BLOCK_SIZE_M; ++m)
  #pragma unroll
      for (int j = 0; j < 4; ++j) acc[m][j] = fmaf(sf[j], part[m][j], acc[m][j]);
  }

  #pragma unroll
  for (int m = 0; m < BLOCK_SIZE_M; ++m) {
    const int32_t token_id = sorted_token_ids[offset_m_base + m];
    if (token_id / top_k >= size_m) continue;
    if (mul_topk_weight && topk_weights != nullptr) {
      const float tw = topk_weights[token_id];
  #pragma unroll
      for (int j = 0; j < 4; ++j) acc[m][j] *= tw;
    }
    const int64_t out_row = output_topk > 0 ? (int64_t)(token_id / output_topk)
                                            : (int64_t)token_id;
    half* out = c + out_row * size_n + n;
    atomic_add_pk4_f16(out, __floats2half2_rn(acc[m][0], acc[m][1]),
                       __floats2half2_rn(acc[m][2], acc[m][3]));
  }
}

#else  // non-RDNA3: empty stub for symbol parity

template <int BLOCK_SIZE_M>
__global__ void moe_octw3_kernel(const half*, half*, const uint32_t*,
                                 const half*, const float*, const int32_t*,
                                 const int32_t*, const int32_t*, const int,
                                 const int, const int, const int, const bool,
                                 const int, const int, const bool, const int) {}

#endif  // __HIP__RDNA3__ || !__HIP_DEVICE_COMPILE__

template <int BLOCK_SIZE_M>
void launch(const half* a, half* c, const uint32_t* w, const half* s,
            const float* topk_weights, const int32_t* sorted_token_ids,
            const int32_t* expert_ids, const int32_t* num_tokens_post_padded,
            int num_token_blocks, int size_m, int size_n, int size_k,
            int top_k, bool mul_topk_weight, int output_topk, int a_stride,
            bool a_silu_mul, int had, cudaStream_t stream) {
  const dim3 grid(num_token_blocks,
                  (size_n + OCTW3_THREADS * 4 - 1) / (OCTW3_THREADS * 4),
                  (size_k + OCTW3_BLOCK_K - 1) / OCTW3_BLOCK_K);
  moe_octw3_kernel<BLOCK_SIZE_M><<<grid, OCTW3_THREADS, 0, stream>>>(
      a, c, w, s, topk_weights, sorted_token_ids, expert_ids,
      num_tokens_post_padded, size_m, size_n, size_k, top_k, mul_topk_weight,
      output_topk, a_stride, a_silu_mul, had);
}

}  // namespace moe_octw3_rdna3
}  // namespace vllm

// Inputs:
//   a                      [M, K] or [M*top_k, K] (or 2*K: [gate | up])  half
//   c                      [M*top_k, N] or [M, N] with output_topk     half,
//                          pre-zeroed
//   w                      [E, K/32, 3, N]  int32
//   scales                 [E, K/32, N]     half
//   topk_weights           [M*top_k] or empty  float32
//   sorted_token_ids, expert_ids, num_tokens_post_padded: moe_align_block_size
//   had                    Hadamard block along K (0, 32, 64, 128 or 256)
void moe_octw3_gemm_rdna3(torch::Tensor a, torch::Tensor c, torch::Tensor w,
                          torch::Tensor scales, torch::Tensor topk_weights,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t top_k,
                          int64_t block_size_m, bool mul_topk_weight,
                          int64_t output_topk, int64_t had) {
  TORCH_CHECK(a.is_cuda() && c.is_cuda() && w.is_cuda() && scales.is_cuda(),
              "all tensors must be on the GPU");
  TORCH_CHECK(a.scalar_type() == torch::kHalf &&
                  c.scalar_type() == torch::kHalf &&
                  scales.scalar_type() == torch::kHalf,
              "Octave-W3 MoE runs in float16");
  TORCH_CHECK(a.dim() == 2 && c.dim() == 2, "a and c must be 2D");
  TORCH_CHECK(w.dim() == 4 && w.size(2) == 3, "w must be [E, K/32, 3, N]");
  TORCH_CHECK(scales.dim() == 3, "scales must be [E, K/32, N]");
  TORCH_CHECK(a.stride(1) == 1, "a must be contiguous in its last dim");

  const int size_k = (int)w.size(1) * 32;
  const int size_n = (int)w.size(3);
  const bool a_silu_mul = a.size(1) == 2 * size_k;
  TORCH_CHECK(a.size(1) == size_k || a_silu_mul,
              "a must have K or 2*K columns, K = ", size_k);
  TORCH_CHECK(size_n % 4 == 0, "N must be a multiple of 4");
  TORCH_CHECK(had == 0 || had == 32 || had == 64 || had == 128 || had == 256,
              "had must be 0, 32, 64, 128 or 256");
  TORCH_CHECK(had == 0 || size_k % had == 0, "K must be a multiple of had");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);
  const float* tw =
      topk_weights.numel() > 0 ? topk_weights.data_ptr<float>() : nullptr;

#define OCTW3_LAUNCH(BM)                                                      \
  vllm::moe_octw3_rdna3::launch<BM>(                                          \
      (const half*)a.data_ptr(), (half*)c.data_ptr(),                         \
      (const uint32_t*)w.data_ptr<int32_t>(),                                 \
      (const half*)scales.data_ptr(), tw, sorted_token_ids.data_ptr<int32_t>(), \
      expert_ids.data_ptr<int32_t>(),                                         \
      num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks,           \
      (int)a.size(0), size_n, size_k, (int)top_k, mul_topk_weight,            \
      (int)output_topk, (int)a.stride(0), a_silu_mul, (int)had, stream)

  switch (block_size_m) {
    case 1:
      OCTW3_LAUNCH(1);
      break;
    case 2:
      OCTW3_LAUNCH(2);
      break;
    case 4:
      OCTW3_LAUNCH(4);
      break;
    case 8:
      OCTW3_LAUNCH(8);
      break;
    default:
      TORCH_CHECK(false, "block_size_m must be 1, 2, 4 or 8, got ",
                  block_size_m);
  }
#undef OCTW3_LAUNCH
}
