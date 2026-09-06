// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MXFP4 W4A16 GEMM for RDNA3 (gfx1100), WMMA path. Forked from the
// GPTQ WMMA kernel (q_gemm_rdna3_wmma.cu): identical f16/bf16 tiling, only the
// B-tile dequant differs (E2M1 + E8M0 group-32, no zeros). Weights are repacked
// Python-side into the [K/8, N] uint32 layout the GPTQ kernel reads.

#include <cstdint>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "qdq_mxfp4_rdna3.cuh"

#if defined(__HIPCC__) && defined(__gfx1100__)
  #define __HIP__RDNA3__
#endif

namespace vllm {
namespace mxfp4_rdna3_wmma {

using bf16_t = __hip_bfloat16;
using mxfp4_rdna3::dequant_mxfp4_8_bf16;
using mxfp4_rdna3::dequant_mxfp4_8_f32;
using mxfp4_rdna3::dequant_mxfp4_8_fp16;
using mxfp4_rdna3::dequant_mxfp4_8_lut;
using mxfp4_rdna3::mxfp4_e2m1_to_bf16_bits;
using mxfp4_rdna3::mxfp4_e2m1_to_fp16_bits;
using mxfp4_rdna3::mxfp4_e8m0_bias;

// K-split heuristic: more blocks along gridDim.z → more resident waves. Mirror
// the GPTQ kernel's K-only split (good enough for the v1 path).
__host__ __device__ static inline int compute_mxfp4_k_split(int size_k) {
  if (size_k >= 4096) return 4;
  if (size_k >= 2048) return 2;
  return 1;
}

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

// Native AMDGPU vector types expected by the WMMA built-ins.
using v16fp16 = _Float16 __attribute__((ext_vector_type(16)));
using v16bf16 = __bf16 __attribute__((ext_vector_type(16)));
using v8fp32 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8fp32 wmma_mma(v16fp16 a, v16fp16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ v8fp32 wmma_mma(v16bf16 a, v16bf16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

template <typename T>
struct WmmaNative;
template <>
struct WmmaNative<half> {
  using elem = _Float16;
  using v16 = v16fp16;
};
template <>
struct WmmaNative<bf16_t> {
  using elem = __bf16;
  using v16 = v16bf16;
};

template <typename FROM, typename TO>
__device__ __forceinline__ TO bitcast_elem(FROM x) {
  static_assert(sizeof(FROM) == sizeof(TO),
                "bitcast_elem requires equal-sized types");
  TO r;
  __builtin_memcpy(&r, &x, sizeof(TO));
  return r;
}

// Packed atomic add for the K-split (gridDim.z > 1) epilogue: CAS-32 retry on
// a uint32 word covering two adjacent fp16/bf16 output cells.
__device__ __forceinline__ void atomic_add_pk_f16(half2* addr, half2 val) {
  uint32_t* a32 = reinterpret_cast<uint32_t*>(addr);
  uint32_t old = *a32, assumed;
  do {
    assumed = old;
    half2 cur = bitcast_elem<uint32_t, half2>(assumed);
    half2 sum = __hadd2(cur, val);
    old = atomicCAS(a32, assumed, bitcast_elem<half2, uint32_t>(sum));
  } while (assumed != old);
}
__device__ __forceinline__ void atomic_add_pk_bf16(__hip_bfloat162* addr,
                                                   __hip_bfloat162 val) {
  uint32_t* a32 = reinterpret_cast<uint32_t*>(addr);
  uint32_t old = *a32, assumed;
  do {
    assumed = old;
    __hip_bfloat162 cur = bitcast_elem<uint32_t, __hip_bfloat162>(assumed);
    __hip_bfloat162 sum = __hadd2(cur, val);
    old = atomicCAS(a32, assumed, bitcast_elem<__hip_bfloat162, uint32_t>(sum));
  } while (assumed != old);
}

#endif  // helpers guard

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

// v1: 16M x 16N tile, 1 wave, full K. Fragment layout matches the GPTQ v1
// kernel; only the B-tile dequant differs. Handles all M (fallback for M<32).
template <typename T>
__global__ void gemm_mxfp4_wmma_kernel_16x16_1w(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint8_t* __restrict__ b_scales_e8m0, T* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int m_tile = blockIdx.y * 16;
  const int n_tile = blockIdx.x * 16;
  if (m_tile >= size_m || n_tile >= size_n) return;

  const int lane = threadIdx.x;   // 0..31
  const int lane_lo = lane & 15;  // row index within fragment
  const int lane_hi = lane >> 4;  // 0 or 1

  v8fp32 c_acc = {0, 0, 0, 0, 0, 0, 0, 0};

  const int groupsize = size_k / groups;  // == 32 for MXFP4
  constexpr uint32_t mant_bits = std::is_same<T, half>::value ? 10u : 7u;

  const int k_per_split = size_k / gridDim.z;
  const int k_start = blockIdx.z * k_per_split;
  const int k_end = k_start + k_per_split;

  __shared__ T b_lds[16][16];
  __shared__ uint16_t w_lut[16];  // E2M1 magnitude bits for T, filled once
  if (lane < 16)
    w_lut[lane] = std::is_same<T, half>::value ? mxfp4_e2m1_to_fp16_bits(lane)
                                               : mxfp4_e2m1_to_bf16_bits(lane);
  __syncthreads();

  for (int k_tile = k_start; k_tile < k_end; k_tile += 16) {
    // ---- Dequant 16x16 B tile into LDS ----
    // 32 lanes = 16 N cols × 2 K-octets per col.
    const int my_n = lane_lo;
    const int my_k_octet = lane_hi;  // 0 → K[0..7], 1 → K[8..15]
    const int actual_n = n_tile + my_n;

    if (actual_n < size_n) {
      const int qk_row = (k_tile / 8) + my_k_octet;
      const uint32_t qa = b_q[qk_row * size_n + actual_n];

      // group_size == 32, k_tile steps by 16 → both octets of a tile share the
      // same group, and group = k_tile / groupsize is exact.
      const int g = k_tile / groupsize;
      const uint32_t s8 = b_scales_e8m0[g * size_n + actual_n];
      const int32_t bias_u16 = mxfp4_e8m0_bias(s8, mant_bits);

      const int k_base = my_k_octet * 8;

      T dq[8];
      dequant_mxfp4_8_lut<T>(qa, bias_u16, w_lut, dq);
  #pragma unroll
      for (int i = 0; i < 8; i++) b_lds[k_base + i][my_n] = dq[i];
    }

    // Single-wave block: no __syncthreads needed.
    V16 a_frag, b_frag;
    const int m_row = m_tile + lane_lo;

    if (m_row < size_m) {
      const T* a_row = a + m_row * size_k;
      static_assert(sizeof(a_frag) == 32, "V16 must be 32 bytes (16 × 2)");
      __builtin_memcpy(&a_frag, a_row + k_tile, sizeof(a_frag));
    } else {
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag[i] = (E)0;
    }

  #pragma unroll
    for (int i = 0; i < 16; i++) {
      b_frag[i] = bitcast_elem<T, E>(b_lds[i][lane_lo]);
    }

    c_acc = wmma_mma(a_frag, b_frag, c_acc);
  }

  // ---- Store C ---- (lane = N column, slot i → row m = 2*i + lane_hi)
  if (gridDim.z > 1) {
    const bool is_even_lane = (lane_lo & 1) == 0;
    const int out_n_pair = n_tile + lane_lo;
  #pragma unroll
    for (int i = 0; i < 8; i++) {
      float other_f = __shfl_xor(c_acc[i], 1);
      if (!is_even_lane) continue;
      const int out_m = m_tile + 2 * i + lane_hi;
      if (out_m >= size_m || out_n_pair >= size_n) continue;
      T* dst = c + out_m * size_n + out_n_pair;
      if constexpr (std::is_same<T, half>::value) {
        half2 packed =
            __halves2half2(__float2half_rn(c_acc[i]), __float2half_rn(other_f));
        atomic_add_pk_f16(reinterpret_cast<half2*>(dst), packed);
      } else {
        __hip_bfloat162 packed;
        packed.x = __float2bfloat16(c_acc[i]);
        packed.y = __float2bfloat16(other_f);
        atomic_add_pk_bf16(reinterpret_cast<__hip_bfloat162*>(dst), packed);
      }
    }
  } else {
    const int out_n = n_tile + lane_lo;
    if (out_n < size_n) {
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int out_m = m_tile + 2 * i + lane_hi;
        if (out_m < size_m) {
          T* dst = c + out_m * size_n + out_n;
          if constexpr (std::is_same<T, half>::value) {
            *dst = __float2half_rn(c_acc[i]);
          } else {
            *dst = __float2bfloat16(c_acc[i]);
          }
        }
      }
    }
  }
}

#else  // non-RDNA3 device pass: empty kernel for symbol parity.
template <typename T>
__global__ void gemm_mxfp4_wmma_kernel_16x16_1w(const T*, const uint32_t*,
                                                const uint8_t*, T*, const int,
                                                const int, const int,
                                                const int) {}
#endif

template <typename T>
void launch_gemm_mxfp4_wmma_16x16_1w(const T* a, const uint32_t* b_q,
                                     const uint8_t* b_scales_e8m0, T* c,
                                     int size_m, int size_n, int size_k,
                                     int groups, cudaStream_t stream) {
  int k_split = compute_mxfp4_k_split(size_k);
  // Each K-segment must be a whole number of 16-wide tiles, else the k_tile
  // loop would drop the tail of K. Fall back to no split otherwise.
  while (k_split > 1 &&
         (size_k % k_split != 0 || (size_k / k_split) % 16 != 0)) {
    k_split /= 2;
  }
  dim3 block(32);
  dim3 grid((size_n + 15) / 16, (size_m + 15) / 16, k_split);
  gemm_mxfp4_wmma_kernel_16x16_1w<T><<<grid, block, 0, stream>>>(
      a, b_q, b_scales_e8m0, c, size_m, size_n, size_k, groups);
}

// ===========================================================================
// Multi-wave kernels for the prefill regime (large M). Same WMMA tiling /
// double-buffering / K-split as the GPTQ kernels; only the B-tile dequant
// differs (E2M1 + cached E8M0 exponent bias, no zeros). group_size == 32.
// ===========================================================================
#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

// ---- 32x16_2w: 2 waves, 32M x 16N, double-buffered LDS (32 <= M < 64) ----
template <typename T>
__global__ void gemm_mxfp4_wmma_kernel_32x16_2w(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint8_t* __restrict__ b_scales_e8m0, T* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int m_tile = blockIdx.y * 32;
  const int n_tile = blockIdx.x * 16;
  if (m_tile >= size_m || n_tile >= size_n) return;

  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  v8fp32 c_acc = {0, 0, 0, 0, 0, 0, 0, 0};
  const int groupsize = size_k / groups;
  constexpr uint32_t mant_bits = std::is_same<T, half>::value ? 10u : 7u;

  const int k_per_split = size_k / gridDim.z;
  const int k_start = blockIdx.z * k_per_split;
  const int k_end = k_start + k_per_split;

  __shared__ T b_lds[2][16][16];
  __shared__ uint16_t w_lut[16];  // E2M1 magnitude bits for T, filled once
  if (tid < 16)
    w_lut[tid] = std::is_same<T, half>::value ? mxfp4_e2m1_to_fp16_bits(tid)
                                              : mxfp4_e2m1_to_bf16_bits(tid);
  __syncthreads();

  auto dequant_into = [&](int buf, int k_tile) {
    if (wave_id != 0) return;
    const int my_n = lane_lo;
    const int actual_n = n_tile + my_n;
    if (actual_n >= size_n) return;
    const int qk_row = (k_tile / 8) + lane_hi;
    const uint32_t qa = b_q[qk_row * size_n + actual_n];
    const int g = k_tile / groupsize;
    const uint32_t s8 = b_scales_e8m0[g * size_n + actual_n];
    const int32_t bias = mxfp4_e8m0_bias(s8, mant_bits);
    const int k_base = lane_hi * 8;
    T dq[8];
    dequant_mxfp4_8_lut<T>(qa, bias, w_lut, dq);
  #pragma unroll
    for (int i = 0; i < 8; i++) b_lds[buf][k_base + i][my_n] = dq[i];
  };

  dequant_into(0, k_start);
  __syncthreads();

  int cur_buf = 0;
  for (int k_tile = k_start; k_tile < k_end; k_tile += 16) {
    const int next_buf = 1 - cur_buf;
    if (k_tile + 16 < k_end) dequant_into(next_buf, k_tile + 16);

    const int m_row = m_tile + wave_id * 16 + lane_lo;
    V16 a_frag, b_frag;
    if (m_row < size_m) {
      static_assert(sizeof(a_frag) == 32, "V16 must be 32 bytes");
      __builtin_memcpy(&a_frag, a + m_row * size_k + k_tile, sizeof(a_frag));
    } else {
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag[i] = (E)0;
    }
  #pragma unroll
    for (int i = 0; i < 16; i++)
      b_frag[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo]);

    c_acc = wmma_mma(a_frag, b_frag, c_acc);
    __syncthreads();
    cur_buf = next_buf;
  }

  const int m_tile_wave = m_tile + wave_id * 16;
  if (gridDim.z > 1) {
    const bool is_even_lane = (lane_lo & 1) == 0;
    const int out_n_pair = n_tile + lane_lo;
  #pragma unroll
    for (int i = 0; i < 8; i++) {
      float other_f = __shfl_xor(c_acc[i], 1);
      if (!is_even_lane) continue;
      const int out_m = m_tile_wave + 2 * i + lane_hi;
      if (out_m >= size_m || out_n_pair >= size_n) continue;
      T* dst = c + out_m * size_n + out_n_pair;
      if constexpr (std::is_same<T, half>::value) {
        half2 packed =
            __halves2half2(__float2half_rn(c_acc[i]), __float2half_rn(other_f));
        atomic_add_pk_f16(reinterpret_cast<half2*>(dst), packed);
      } else {
        __hip_bfloat162 packed;
        packed.x = __float2bfloat16(c_acc[i]);
        packed.y = __float2bfloat16(other_f);
        atomic_add_pk_bf16(reinterpret_cast<__hip_bfloat162*>(dst), packed);
      }
    }
  } else {
    const int out_n = n_tile + lane_lo;
    if (out_n < size_n) {
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m < size_m) {
          T* dst = c + out_m * size_n + out_n;
          if constexpr (std::is_same<T, half>::value)
            *dst = __float2half_rn(c_acc[i]);
          else
            *dst = __float2bfloat16(c_acc[i]);
        }
      }
    }
  }
}

// ---- 64x64_4w: 4 waves, 64M x 64N, K=16/iter (64 <= M < 128) ----
template <typename T>
__global__ void gemm_mxfp4_wmma_kernel_64x64_4w(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint8_t* __restrict__ b_scales_e8m0, T* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int m_tile = blockIdx.y * 64;
  const int n_tile = blockIdx.x * 64;
  if (m_tile >= size_m || n_tile >= size_n) return;

  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  v8fp32 c_acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc3 = {0, 0, 0, 0, 0, 0, 0, 0};

  const int groupsize = size_k / groups;
  constexpr uint32_t mant_bits = std::is_same<T, half>::value ? 10u : 7u;

  const int k_per_split = size_k / gridDim.z;
  const int k_start = blockIdx.z * k_per_split;
  const int k_end = k_start + k_per_split;

  __shared__ T b_lds[2][16][64];
  __shared__ uint16_t w_lut[16];  // E2M1 magnitude bits for T, filled once
  if (threadIdx.x < 16)
    w_lut[threadIdx.x] = std::is_same<T, half>::value
                             ? mxfp4_e2m1_to_fp16_bits(threadIdx.x)
                             : mxfp4_e2m1_to_bf16_bits(threadIdx.x);
  __syncthreads();

  auto dequant_into = [&](int buf, int k_tile) {
    const int my_n_in_tile = wave_id * 16 + lane_lo;
    const int actual_n = n_tile + my_n_in_tile;
    if (actual_n >= size_n) return;
    const int qk_row = (k_tile / 8) + lane_hi;
    const uint32_t qa = b_q[qk_row * size_n + actual_n];
    const int g = k_tile / groupsize;
    const uint32_t s8 = b_scales_e8m0[g * size_n + actual_n];
    const int32_t bias = mxfp4_e8m0_bias(s8, mant_bits);
    const int k_base = lane_hi * 8;
    T dq[8];
    dequant_mxfp4_8_lut<T>(qa, bias, w_lut, dq);
  #pragma unroll
    for (int i = 0; i < 8; i++) b_lds[buf][k_base + i][my_n_in_tile] = dq[i];
  };

  dequant_into(0, k_start);
  __syncthreads();

  int cur_buf = 0;
  for (int k_tile = k_start; k_tile < k_end; k_tile += 16) {
    const int next_buf = 1 - cur_buf;
    if (k_tile + 16 < k_end) dequant_into(next_buf, k_tile + 16);

    const int m_row = m_tile + wave_id * 16 + lane_lo;
    V16 a_frag, b_frag0, b_frag1, b_frag2, b_frag3;
    if (m_row < size_m) {
      static_assert(sizeof(a_frag) == 32, "V16 must be 32 bytes");
      __builtin_memcpy(&a_frag, a + m_row * size_k + k_tile, sizeof(a_frag));
    } else {
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag[i] = (E)0;
    }
  #pragma unroll
    for (int i = 0; i < 16; i++) {
      b_frag0[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo + 0]);
      b_frag1[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo + 16]);
      b_frag2[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo + 32]);
      b_frag3[i] = bitcast_elem<T, E>(b_lds[cur_buf][i][lane_lo + 48]);
    }
    c_acc0 = wmma_mma(a_frag, b_frag0, c_acc0);
    c_acc1 = wmma_mma(a_frag, b_frag1, c_acc1);
    c_acc2 = wmma_mma(a_frag, b_frag2, c_acc2);
    c_acc3 = wmma_mma(a_frag, b_frag3, c_acc3);
    __syncthreads();
    cur_buf = next_buf;
  }

  const int m_tile_wave = m_tile + wave_id * 16;
  auto store_acc = [&](const v8fp32& acc, int n_base) {
    if (gridDim.z > 1) {
      const bool is_even_lane = (lane_lo & 1) == 0;
      const int out_n_pair = n_base + lane_lo;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        float other_f = __shfl_xor(acc[i], 1);
        if (!is_even_lane) continue;
        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m >= size_m || out_n_pair >= size_n) continue;
        T* dst = c + out_m * size_n + out_n_pair;
        if constexpr (std::is_same<T, half>::value) {
          half2 packed =
              __halves2half2(__float2half_rn(acc[i]), __float2half_rn(other_f));
          atomic_add_pk_f16(reinterpret_cast<half2*>(dst), packed);
        } else {
          __hip_bfloat162 packed;
          packed.x = __float2bfloat16(acc[i]);
          packed.y = __float2bfloat16(other_f);
          atomic_add_pk_bf16(reinterpret_cast<__hip_bfloat162*>(dst), packed);
        }
      }
    } else {
      const int out_n = n_base + lane_lo;
      if (out_n >= size_n) return;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m < size_m) {
          T* dst = c + out_m * size_n + out_n;
          if constexpr (std::is_same<T, half>::value)
            *dst = __float2half_rn(acc[i]);
          else
            *dst = __float2bfloat16(acc[i]);
        }
      }
    }
  };
  store_acc(c_acc0, n_tile + 0);
  store_acc(c_acc1, n_tile + 16);
  store_acc(c_acc2, n_tile + 32);
  store_acc(c_acc3, n_tile + 48);
}

// ---- 128x64_k32: 8 waves, 128M x 64N, K=32/iter, cached group bias (M >= 128)
// ----
template <typename T>
__global__ void gemm_mxfp4_wmma_kernel_128x64_k32(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint8_t* __restrict__ b_scales_e8m0, T* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups) {
  using E = typename WmmaNative<T>::elem;
  using V16 = typename WmmaNative<T>::v16;

  const int m_tile = blockIdx.y * 128;
  const int n_tile = blockIdx.x * 64;
  if (m_tile >= size_m || n_tile >= size_n) return;

  const int tid = threadIdx.x;
  const int wave_id = tid >> 5;
  const int lane = tid & 31;
  const int lane_lo = lane & 15;
  const int lane_hi = lane >> 4;

  v8fp32 c_acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
  v8fp32 c_acc3 = {0, 0, 0, 0, 0, 0, 0, 0};

  const int groupsize = size_k / groups;
  constexpr uint32_t mant_bits = std::is_same<T, half>::value ? 10u : 7u;

  const int k_per_split = size_k / gridDim.z;
  const int k_start = blockIdx.z * k_per_split;
  const int k_end = k_start + k_per_split;

  __shared__ T b_lds[2][64][34];  // +2 pad to break LDS bank conflicts

  const int dq_wave4 = wave_id & 3;
  const int dq_k_half = (wave_id >= 4) ? 1 : 0;
  const int dq_n = dq_wave4 * 16 + lane_lo;
  const int dq_an = n_tile + dq_n;
  const bool dq_ok = (dq_an < size_n);
  const int dq_oct = lane_hi + dq_k_half * 2;
  const int dq_kb = dq_oct * 8;
  int cached_g = -1;
  int32_t cached_bias = 0;

  auto dequant_into = [&](int buf, int k_tile) __attribute__((always_inline)) {
    if (!dq_ok) return;
    const int g = k_tile / groupsize;
    if (g != cached_g) {
      cached_g = g;
      const uint32_t s8 = b_scales_e8m0[g * size_n + dq_an];
      cached_bias = mxfp4_e8m0_bias(s8, mant_bits);
    }
    const int qk_row = (k_tile / 8) + dq_oct;
    const uint32_t qa = b_q[qk_row * size_n + dq_an];
    if constexpr (std::is_same<T, half>::value) {
      half dq[8];
      dequant_mxfp4_8_fp16(qa, cached_bias, dq);
  #pragma unroll
      for (int i = 0; i < 8; i++) b_lds[buf][dq_n][dq_kb + i] = dq[i];
    } else {
      bf16_t dq[8];
      dequant_mxfp4_8_bf16(qa, cached_bias, dq);
  #pragma unroll
      for (int i = 0; i < 8; i++) b_lds[buf][dq_n][dq_kb + i] = dq[i];
    }
  };

  dequant_into(0, k_start);
  __syncthreads();

  int cur_buf = 0;
  const int m_row = m_tile + wave_id * 16 + lane_lo;
  const T* a_row_ptr = (m_row < size_m) ? (a + m_row * size_k) : nullptr;

  for (int k_tile = k_start; k_tile < k_end; k_tile += 32) {
    const int next_buf = 1 - cur_buf;
    if (k_tile + 32 < k_end) dequant_into(next_buf, k_tile + 32);

    V16 a_frag_lo, a_frag_hi;
    V16 b_frag0, b_frag1, b_frag2, b_frag3;
    if (a_row_ptr) {
      __builtin_memcpy(&a_frag_lo, a_row_ptr + k_tile, sizeof(V16));
      __builtin_memcpy(&a_frag_hi, a_row_ptr + k_tile + 16, sizeof(V16));
    } else {
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag_lo[i] = (E)0;
  #pragma unroll
      for (int i = 0; i < 16; i++) a_frag_hi[i] = (E)0;
    }

    static_assert(sizeof(V16) == 32, "V16 must be 32 bytes for memcpy");
    __builtin_memcpy(&b_frag0, &b_lds[cur_buf][lane_lo + 0][0], 32);
    __builtin_memcpy(&b_frag1, &b_lds[cur_buf][lane_lo + 16][0], 32);
    __builtin_memcpy(&b_frag2, &b_lds[cur_buf][lane_lo + 32][0], 32);
    __builtin_memcpy(&b_frag3, &b_lds[cur_buf][lane_lo + 48][0], 32);
    c_acc0 = wmma_mma(a_frag_lo, b_frag0, c_acc0);
    c_acc1 = wmma_mma(a_frag_lo, b_frag1, c_acc1);
    c_acc2 = wmma_mma(a_frag_lo, b_frag2, c_acc2);
    c_acc3 = wmma_mma(a_frag_lo, b_frag3, c_acc3);

    __builtin_memcpy(&b_frag0, &b_lds[cur_buf][lane_lo + 0][16], 32);
    __builtin_memcpy(&b_frag1, &b_lds[cur_buf][lane_lo + 16][16], 32);
    __builtin_memcpy(&b_frag2, &b_lds[cur_buf][lane_lo + 32][16], 32);
    __builtin_memcpy(&b_frag3, &b_lds[cur_buf][lane_lo + 48][16], 32);
    c_acc0 = wmma_mma(a_frag_hi, b_frag0, c_acc0);
    c_acc1 = wmma_mma(a_frag_hi, b_frag1, c_acc1);
    c_acc2 = wmma_mma(a_frag_hi, b_frag2, c_acc2);
    c_acc3 = wmma_mma(a_frag_hi, b_frag3, c_acc3);

    __syncthreads();
    cur_buf = next_buf;
  }

  const int m_tile_wave = m_tile + wave_id * 16;
  auto store_acc = [&](const v8fp32& acc, int n_base) {
    if (gridDim.z > 1) {
      const bool is_even_lane = (lane_lo & 1) == 0;
      const int out_n_pair = n_base + lane_lo;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        float other_f = __shfl_xor(acc[i], 1);
        if (!is_even_lane) continue;
        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m >= size_m || out_n_pair >= size_n) continue;
        T* dst = c + out_m * size_n + out_n_pair;
        if constexpr (std::is_same<T, half>::value) {
          half2 packed =
              __halves2half2(__float2half_rn(acc[i]), __float2half_rn(other_f));
          atomic_add_pk_f16(reinterpret_cast<half2*>(dst), packed);
        } else {
          __hip_bfloat162 packed;
          packed.x = __float2bfloat16(acc[i]);
          packed.y = __float2bfloat16(other_f);
          atomic_add_pk_bf16(reinterpret_cast<__hip_bfloat162*>(dst), packed);
        }
      }
    } else {
      const int out_n = n_base + lane_lo;
      if (out_n >= size_n) return;
  #pragma unroll
      for (int i = 0; i < 8; i++) {
        const int out_m = m_tile_wave + 2 * i + lane_hi;
        if (out_m < size_m) {
          T* dst = c + out_m * size_n + out_n;
          if constexpr (std::is_same<T, half>::value)
            *dst = __float2half_rn(acc[i]);
          else
            *dst = __float2bfloat16(acc[i]);
        }
      }
    }
  };
  store_acc(c_acc0, n_tile + 0);
  store_acc(c_acc1, n_tile + 16);
  store_acc(c_acc2, n_tile + 32);
  store_acc(c_acc3, n_tile + 48);
}

#else  // non-RDNA3 device pass: empty kernels for symbol parity.
template <typename T>
__global__ void gemm_mxfp4_wmma_kernel_32x16_2w(const T*, const uint32_t*,
                                                const uint8_t*, T*, const int,
                                                const int, const int,
                                                const int) {}
template <typename T>
__global__ void gemm_mxfp4_wmma_kernel_64x64_4w(const T*, const uint32_t*,
                                                const uint8_t*, T*, const int,
                                                const int, const int,
                                                const int) {}
template <typename T>
__global__ void gemm_mxfp4_wmma_kernel_128x64_k32(const T*, const uint32_t*,
                                                  const uint8_t*, T*, const int,
                                                  const int, const int,
                                                  const int) {}
#endif

// K-split that keeps each segment a whole multiple of `step` 16-wide tiles.
__host__ static inline int mxfp4_k_split_aligned(int size_k, int step) {
  int s = compute_mxfp4_k_split(size_k);
  while (s > 1 && (size_k % s != 0 || (size_k / s) % step != 0)) s /= 2;
  return s;
}

// Dispatcher: pick the widest tile that fits M, falling back to v1 for M < 32.
template <typename T>
void launch_gemm_mxfp4_wmma(const T* a, const uint32_t* b_q,
                            const uint8_t* b_scales_e8m0, T* c, int size_m,
                            int size_n, int size_k, int groups,
                            cudaStream_t stream) {
  if (size_m >= 128 && size_n >= 64) {
    const int k_split = mxfp4_k_split_aligned(size_k, 32);
    dim3 block(256);
    dim3 grid((size_n + 63) / 64, (size_m + 127) / 128, k_split);
    gemm_mxfp4_wmma_kernel_128x64_k32<T><<<grid, block, 0, stream>>>(
        a, b_q, b_scales_e8m0, c, size_m, size_n, size_k, groups);
    return;
  }
  if (size_m >= 64 && size_n >= 64) {
    const int k_split = mxfp4_k_split_aligned(size_k, 16);
    dim3 block(128);
    dim3 grid((size_n + 63) / 64, (size_m + 63) / 64, k_split);
    gemm_mxfp4_wmma_kernel_64x64_4w<T><<<grid, block, 0, stream>>>(
        a, b_q, b_scales_e8m0, c, size_m, size_n, size_k, groups);
    return;
  }
  if (size_m >= 32) {
    const int k_split = mxfp4_k_split_aligned(size_k, 16);
    dim3 block(64);
    dim3 grid((size_n + 15) / 16, (size_m + 31) / 32, k_split);
    gemm_mxfp4_wmma_kernel_32x16_2w<T><<<grid, block, 0, stream>>>(
        a, b_q, b_scales_e8m0, c, size_m, size_n, size_k, groups);
    return;
  }
  launch_gemm_mxfp4_wmma_16x16_1w<T>(a, b_q, b_scales_e8m0, c, size_m, size_n,
                                     size_k, groups, stream);
}

// Scalar GEMV for decode (small M): WMMA wastes 15/16 of its compute at M=1,
// so this reads each weight column once and does M scalar dots
// (bandwidth-bound, the 4-bit read wins). 256 threads, 4 N cols each; gridDim.z
// splits K (atomic).
#define MXFP4_BLOCK_KN 256
#define MXFP4_THREADS 256

#if defined(__HIP__RDNA3__) || !defined(__HIP_DEVICE_COMPILE__)

__device__ __forceinline__ float mxfp4_to_f(half v) { return __half2float(v); }
__device__ __forceinline__ float mxfp4_to_f(bf16_t v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ void mxfp4_atomic_add_pk4(half* addr, half2 v01,
                                                     half2 v23) {
  unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *a;
  while (true) {
    union {
      unsigned long long u;
      half2 h2[2];
    } cur, sum;
    cur.u = old;
    sum.h2[0] = __hadd2(cur.h2[0], v01);
    sum.h2[1] = __hadd2(cur.h2[1], v23);
    unsigned long long prev = atomicCAS(a, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}
__device__ __forceinline__ void mxfp4_atomic_add_pk4(bf16_t* addr,
                                                     __hip_bfloat162 v01,
                                                     __hip_bfloat162 v23) {
  unsigned long long* a = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *a;
  while (true) {
    union {
      unsigned long long u;
      __hip_bfloat162 b2[2];
    } cur, sum;
    cur.u = old;
    sum.b2[0] = __hadd2(cur.b2[0], v01);
    sum.b2[1] = __hadd2(cur.b2[1], v23);
    unsigned long long prev = atomicCAS(a, old, sum.u);
    if (prev == old) break;
    old = prev;
  }
}

template <typename T, int M_COUNT>
__global__ void gemm_mxfp4_scalar_rdna3(
    const T* __restrict__ a, const uint32_t* __restrict__ b_q,
    const uint8_t* __restrict__ b_scales_e8m0, T* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups) {
  const int t = threadIdx.x;
  const int offset_n = blockIdx.x * MXFP4_BLOCK_KN * 4;
  const int offset_m = blockIdx.y * M_COUNT;
  const int offset_k = blockIdx.z * MXFP4_BLOCK_KN;
  const int end_k = min(offset_k + MXFP4_BLOCK_KN, size_k);
  const int n = offset_n + t * 4;

  constexpr int LDS_PAD = 8;
  __shared__ T block_a[M_COUNT][MXFP4_BLOCK_KN + LDS_PAD];
  // E2M1 signed-magnitude LUT (exact fp32): one lookup replaces the per-nibble
  // bit decode; the E8M0 scale folds in once per group as a multiply. Decode is
  // compute-bound on gfx1100, so this is ~2.8x over the arithmetic decode.
  __shared__ float w_lut[16];
  if (t < 16) {
    const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    w_lut[t] = (t & 0x8 ? -1.f : 1.f) * mag[t & 0x7];
  }
  if (offset_k + t < end_k) {
  #pragma unroll
    for (int m = 0; m < M_COUNT; ++m) {
      block_a[m][t] = (offset_m + m < size_m)
                          ? a[(offset_m + m) * size_k + offset_k + t]
                          : (T)0;
    }
  }
  __syncthreads();
  if (n >= size_n) return;

  const int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = (group + 1) * groupsize;

  int qk = offset_k / 8;
  const uint32_t* b_ptr = b_q + qk * size_n + n;

  // Per-group E8M0 scale as a pure bit construct: 2^(s8-127) is the fp32 whose
  // exponent field is s8 (IEEE bias 127), i.e. uint_as_float(s8 << 23).
  float gscale[4];
  auto refresh = [&](int g) {
  #pragma unroll
    for (int col = 0; col < 4; ++col)
      gscale[col] =
          __uint_as_float((uint32_t)b_scales_e8m0[g * size_n + n + col] << 23);
  };
  refresh(group);

  float block_c[M_COUNT][4];
  #pragma unroll
  for (int m = 0; m < M_COUNT; ++m)
  #pragma unroll
    for (int col = 0; col < 4; ++col) block_c[m][col] = 0.0f;

  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      refresh(group);
    }
    int4 b_w[4];
  #pragma unroll
    for (int j = 0; j < 4; ++j) b_w[j] = *(const int4*)(b_ptr + j * size_n);
    b_ptr += 4 * size_n;

  #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int a_off = (k - offset_k) + 8 * j;
      uint32_t wcol[4];
      __builtin_memcpy(wcol, &b_w[j], sizeof(int4));
  #pragma unroll
      for (int col = 0; col < 4; ++col) {
        const uint32_t qa = wcol[col];
  #pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
          const T* ap = &block_a[m][a_off];
          float s = 0.0f;
  #pragma unroll
          for (int i = 0; i < 8; ++i)
            s += w_lut[(qa >> (4 * i)) & 0xFu] * mxfp4_to_f(ap[i]);
          block_c[m][col] += gscale[col] * s;
        }
      }
    }
    k += 32;
  }

  #pragma unroll
  for (int m = 0; m < M_COUNT; ++m) {
    if (offset_m + m >= size_m) continue;
    T* out = c + (offset_m + m) * size_n + n;
    if constexpr (std::is_same<T, half>::value) {
      half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                 __float2half_rn(block_c[m][1]));
      half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                 __float2half_rn(block_c[m][3]));
      mxfp4_atomic_add_pk4(out, r01, r23);
    } else {
      __hip_bfloat162 r01, r23;
      r01.x = __float2bfloat16(block_c[m][0]);
      r01.y = __float2bfloat16(block_c[m][1]);
      r23.x = __float2bfloat16(block_c[m][2]);
      r23.y = __float2bfloat16(block_c[m][3]);
      mxfp4_atomic_add_pk4(out, r01, r23);
    }
  }
}

#else  // non-RDNA3 stub
template <typename T, int M_COUNT>
__global__ void gemm_mxfp4_scalar_rdna3(const T*, const uint32_t*,
                                        const uint8_t*, T*, const int,
                                        const int, const int, const int) {}
#endif

template <typename T, int M_COUNT>
void launch_mxfp4_scalar(const T* a, const uint32_t* b_q,
                         const uint8_t* b_scales_e8m0, T* c, int size_m,
                         int size_n, int size_k, int groups,
                         cudaStream_t stream) {
  dim3 block(MXFP4_BLOCK_KN);
  dim3 grid((size_n + MXFP4_BLOCK_KN * 4 - 1) / (MXFP4_BLOCK_KN * 4),
            (size_m + M_COUNT - 1) / M_COUNT,
            (size_k + MXFP4_BLOCK_KN - 1) / MXFP4_BLOCK_KN);
  gemm_mxfp4_scalar_rdna3<T, M_COUNT><<<grid, block, 0, stream>>>(
      a, b_q, b_scales_e8m0, c, size_m, size_n, size_k, groups);
}

template <typename T>
void launch_gemm_mxfp4_scalar(const T* a, const uint32_t* b_q,
                              const uint8_t* b_scales_e8m0, T* c, int size_m,
                              int size_n, int size_k, int groups,
                              cudaStream_t stream) {
  if (size_m == 1)
    launch_mxfp4_scalar<T, 1>(a, b_q, b_scales_e8m0, c, size_m, size_n, size_k,
                              groups, stream);
  else if (size_m <= 3)
    launch_mxfp4_scalar<T, 2>(a, b_q, b_scales_e8m0, c, size_m, size_n, size_k,
                              groups, stream);
  else if (size_m <= 7)
    launch_mxfp4_scalar<T, 4>(a, b_q, b_scales_e8m0, c, size_m, size_n, size_k,
                              groups, stream);
  else
    launch_mxfp4_scalar<T, 8>(a, b_q, b_scales_e8m0, c, size_m, size_n, size_k,
                              groups, stream);
}

}  // namespace mxfp4_rdna3_wmma
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
//
// Inputs:
//   a              [M, K]        half or bfloat16
//   b_q_weight     [K/8, N]      uint32 (E2M1, 8 sequential K nibbles per word,
//                                 repacked from compressed-tensors [N, K/2])
//   b_scales_e8m0  [K/32, N]     uint8  (E8M0 block scale, group_size = 32)
//
// Output:
//   c              [M, N]        same dtype as a
//
// Requirements: N % 16 == 0, K % 32 == 0.
torch::Tensor mxfp4_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                               torch::Tensor b_scales_e8m0) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(b_scales_e8m0.is_cuda(),
              "b_scales_e8m0 must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D [M, K]");
  TORCH_CHECK(b_q_weight.dim() == 2, "b_q_weight must be 2D [K/8, N]");
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf || a.scalar_type() == torch::kBFloat16,
      "a must be half or bfloat16");
  TORCH_CHECK(b_q_weight.scalar_type() == torch::kUInt32 ||
                  b_q_weight.scalar_type() == torch::kInt32,
              "b_q_weight must be (u)int32");
  TORCH_CHECK(b_scales_e8m0.scalar_type() == torch::kUInt8,
              "b_scales_e8m0 must be uint8 (E8M0)");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(1);
  int groups = (int)b_scales_e8m0.size(0);

  TORCH_CHECK(b_q_weight.size(0) * 8 == size_k,
              "b_q_weight first dim must be K/8");
  TORCH_CHECK(b_scales_e8m0.size(1) == size_n,
              "b_scales_e8m0 last dim must be N");
  TORCH_CHECK(groups * 32 == size_k, "E8M0 group count must be K/32");
  TORCH_CHECK(size_n % 16 == 0, "WMMA path requires N % 16 == 0");
  TORCH_CHECK(size_k % 32 == 0, "MXFP4 path requires K % 32 == 0");

  auto opts = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  // Zero-init: K-split (gridDim.z > 1) accumulates atomically into c.
  at::Tensor c = torch::zeros({size_m, size_n}, opts);

  // M=1 (single-token decode) -> scalar GEMV; M>=2 (batched decode / prefill)
  // -> WMMA. The scalar path is O(M) per weight read, so it degrades fast with
  // batch; the WMMA 16x16 tile is flat ~constant up to M=16 and far faster
  // (e.g. M=8: 205us scalar -> ~49us WMMA on a 7900 XTX).
  // The LUT decode made the scalar GEMV fast enough to beat the WMMA 16-tile up
  // to M~8 (1.3-1.6x at M=4/8 on gfx1100); above that the matrix engine wins.
  const bool use_scalar = size_m <= 8;
  if (a.scalar_type() == torch::kHalf) {
    namespace mx = vllm::mxfp4_rdna3_wmma;
    auto* ap = (const half*)a.data_ptr();
    auto* bq = (const uint32_t*)b_q_weight.data_ptr();
    auto* bs = (const uint8_t*)b_scales_e8m0.data_ptr();
    auto* cp = (half*)c.data_ptr();
    if (use_scalar)
      mx::launch_gemm_mxfp4_scalar<half>(ap, bq, bs, cp, size_m, size_n, size_k,
                                         groups, stream);
    else
      mx::launch_gemm_mxfp4_wmma<half>(ap, bq, bs, cp, size_m, size_n, size_k,
                                       groups, stream);
  } else {
    using bf16_t = vllm::mxfp4_rdna3_wmma::bf16_t;
    namespace mx = vllm::mxfp4_rdna3_wmma;
    auto* ap = (const bf16_t*)a.data_ptr();
    auto* bq = (const uint32_t*)b_q_weight.data_ptr();
    auto* bs = (const uint8_t*)b_scales_e8m0.data_ptr();
    auto* cp = (bf16_t*)c.data_ptr();
    if (use_scalar)
      mx::launch_gemm_mxfp4_scalar<bf16_t>(ap, bq, bs, cp, size_m, size_n,
                                           size_k, groups, stream);
    else
      mx::launch_gemm_mxfp4_wmma<bf16_t>(ap, bq, bs, cp, size_m, size_n, size_k,
                                         groups, stream);
  }

  return c;
}
