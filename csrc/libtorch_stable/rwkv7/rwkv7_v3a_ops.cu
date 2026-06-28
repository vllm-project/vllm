// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from BlinkDL/Albatross faster3a_2605/cuda at commit
// 5e941fb1eeb7f735a562fb5bbb30fad19adc825b. Source:
// https://github.com/BlinkDL/Albatross/tree/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/faster3a_2605/cuda
// Upstream license: Apache-2.0
// (https://github.com/BlinkDL/Albatross/blob/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/LICENSE).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <climits>
#include <vector>

using dtype = at::Half;
namespace wmma = nvcuda::wmma;

namespace {

constexpr int LN_THREADS = 256;
constexpr int LN_SMALL_THREADS = 1024;
constexpr int LN_SMALL512_THREADS = 512;
constexpr int LN_SMALL_C = 4096;

inline int64_t ceil_div(int64_t n, int64_t d) { return (n + d - 1) / d; }

inline void check_cublas(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what,
              " failed with cublas status ", static_cast<int>(status));
}

inline void check_cublaslt(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what,
              " failed with cublasLt status ", static_cast<int>(status));
}

template <int Act>
__device__ __forceinline__ float apply_act(float x) {
  if constexpr (Act == 1) {
    return tanhf(x);
  } else {
    return 1.0f / (1.0f + expf(-x));
  }
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__device__ __forceinline__ float bf16_bits_to_float_dev(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = static_cast<uint32_t>(bits) << 16;
  return v.f;
}

template <int Threads>
__device__ __forceinline__ float block_sum_t(float x) {
  __shared__ float partial[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  x = warp_sum(x);
  if (lane == 0) {
    partial[warp] = x;
  }
  __syncthreads();
  x = (threadIdx.x < (Threads / 32)) ? partial[lane] : 0.0f;
  if (warp == 0) {
    x = warp_sum(x);
  }
  if (threadIdx.x == 0) {
    partial[0] = x;
  }
  __syncthreads();
  return partial[0];
}

__global__ void emb_ln0_bf16_to_f16_kernel(int V, int C,
                                           const uint16_t* __restrict__ emb,
                                           const uint16_t* __restrict__ weight,
                                           const uint16_t* __restrict__ bias,
                                           dtype* __restrict__ out, float eps) {
  // Precision path: bf16 inputs -> fp32 two-pass stats/affine -> fp16 output.
  const int tok = blockIdx.x;
  const int tid = threadIdx.x;
  if (tok >= V) {
    return;
  }
  const uint16_t* er = emb + static_cast<int64_t>(tok) * C;
  float sum = 0.0f;
  for (int c = tid; c < C; c += blockDim.x) {
    sum += bf16_bits_to_float_dev(er[c]);
  }
  const float mean = block_sum_t<256>(sum) / static_cast<float>(C);
  float var = 0.0f;
  for (int c = tid; c < C; c += blockDim.x) {
    const float d = bf16_bits_to_float_dev(er[c]) - mean;
    var += d * d;
  }
  const float rstd =
      rsqrtf(block_sum_t<256>(var) / static_cast<float>(C) + eps);
  dtype* yr = out + static_cast<int64_t>(tok) * C;
  for (int c = tid; c < C; c += blockDim.x) {
    const float x = bf16_bits_to_float_dev(er[c]);
    const float w = bf16_bits_to_float_dev(weight[c]);
    const float b = bf16_bits_to_float_dev(bias[c]);
    yr[c] = static_cast<dtype>((x - mean) * rstd * w + b);
  }
}

__global__ void add_f16_kernel(const dtype* __restrict__ x,
                               const dtype* __restrict__ y,
                               dtype* __restrict__ out, int64_t n_pairs) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_pairs) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[i]);
    const float2 yv = __half22float2(reinterpret_cast<const __half2*>(y)[i]);
    reinterpret_cast<__half2*>(out)[i] =
        __floats2half2_rn(xv.x + yv.x, xv.y + yv.y);
  }
}

__global__ void advance_i32_kernel(int* __restrict__ x, int amount, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] += amount;
  }
}

template <int ChunkK, int Warps>
__global__ __launch_bounds__(128, 2) void linear_f16_m1_splitk_partial_kernel(
    int K, int N, const dtype* __restrict__ x, const dtype* __restrict__ weight,
    float* __restrict__ partial) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = (blockIdx.x * Warps + warp) * 32 + lane;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  const int k0 = blockIdx.y * ChunkK;
  const int k1 = min(k0 + ChunkK, K);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int k = k0; k < k1; ++k) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(
        weight + static_cast<int64_t>(k) * N + n));
    acc0 = fmaf(xv, wv.x, acc0);
    acc1 = fmaf(xv, wv.y, acc1);
  }
  reinterpret_cast<float2*>(partial + static_cast<int64_t>(blockIdx.y) *
                                          N)[pair] = make_float2(acc0, acc1);
}

__global__ void linear_f16_m1_splitk_reduce_kernel(
    int chunks, int N, const float* __restrict__ partial,
    dtype* __restrict__ y) {
  const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int c = 0; c < chunks; ++c) {
    const float2 v = reinterpret_cast<const float2*>(
        partial + static_cast<int64_t>(c) * N)[pair];
    acc0 += v.x;
    acc1 += v.y;
  }
  reinterpret_cast<__half2*>(y)[pair] = __floats2half2_rn(acc0, acc1);
}

__global__ void linear_f16_m1_splitk_reduce_warp_kernel(
    int chunks, int N, const float* __restrict__ partial,
    dtype* __restrict__ y) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = blockIdx.x * 4 + warp;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int c = lane; c < chunks; c += 32) {
    const float2 v = reinterpret_cast<const float2*>(
        partial + static_cast<int64_t>(c) * N)[pair];
    acc0 += v.x;
    acc1 += v.y;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
    acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
  }
  if (lane == 0) {
    reinterpret_cast<__half2*>(y)[pair] = __floats2half2_rn(acc0, acc1);
  }
}

template <int ChunkK, int Warps>
__global__ __launch_bounds__(128, 2) void linear_f16_rows_splitk_partial_kernel(
    int K, int N, int chunks, const dtype* __restrict__ x,
    const dtype* __restrict__ weight, float* __restrict__ partial) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = (blockIdx.x * Warps + warp) * 32 + lane;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  const int chunk = blockIdx.y;
  const int m = blockIdx.z;
  const int k0 = chunk * ChunkK;
  const int k1 = min(k0 + ChunkK, K);
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int k = k0; k < k1; ++k) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(
        weight + static_cast<int64_t>(k) * N + n));
    acc0 = fmaf(xv, wv.x, acc0);
    acc1 = fmaf(xv, wv.y, acc1);
  }
  reinterpret_cast<float2*>(
      partial + (static_cast<int64_t>(m) * chunks + chunk) * N)[pair] =
      make_float2(acc0, acc1);
}

__global__ void linear_f16_rows_splitk_reduce_kernel(
    int chunks, int N, const float* __restrict__ partial,
    dtype* __restrict__ y) {
  const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int c = 0; c < chunks; ++c) {
    const float2 v = reinterpret_cast<const float2*>(
        partial + (static_cast<int64_t>(m) * chunks + c) * N)[pair];
    acc0 += v.x;
    acc1 += v.y;
  }
  reinterpret_cast<__half2*>(y + static_cast<int64_t>(m) * N)[pair] =
      __floats2half2_rn(acc0, acc1);
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_kernel(
    int M, int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t, dtype* __restrict__ y) {
  const int n = blockIdx.x;
  const int m = blockIdx.y;
  if (m >= M || n >= N) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = weight_t + static_cast<int64_t>(n) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const float2 xv =
        __half22float2(*reinterpret_cast<const __half2*>(x_row + (k2 << 1)));
    const float2 wv =
        __half22float2(*reinterpret_cast<const __half2*>(w_row + (k2 << 1)));
    acc = fmaf(xv.x, wv.x, acc);
    acc = fmaf(xv.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) =
        __float2half_rn(acc);
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_ntile_kernel(
    int M, int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv =
        __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(
            weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv =
        __half2float(*reinterpret_cast<const __half*>(x_row + K - 1));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv,
                      __half2float(*reinterpret_cast<const __half*>(
                          weight_t + static_cast<int64_t>(n) * K + K - 1)),
                      acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) =
            __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_ntile_scalar_kernel(
    int M, int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv,
                      __half2float(*reinterpret_cast<const __half*>(
                          weight_t + static_cast<int64_t>(n) * K + k)),
                      acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) =
            __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int RowTile, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_rows_f16_kernel(
    int M, int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m0 = blockIdx.y * RowTile;
  float acc[RowTile][OutTile];
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      acc[r][j] = 0.0f;
    }
  }
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    float2 wv[OutTile];
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      wv[j] = (n < N) ? __half22float2(*reinterpret_cast<const __half2*>(
                            weight_orig + static_cast<int64_t>(n) * K + k))
                      : make_float2(0.0f, 0.0f);
    }
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(
            x + static_cast<int64_t>(m) * K + k));
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          acc[r][j] = fmaf(xv.x, wv[j].x, acc[r][j]);
          acc[r][j] = fmaf(xv.y, wv[j].y, acc[r][j]);
        }
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float wv = __half2float(*reinterpret_cast<const __half*>(
            weight_orig + static_cast<int64_t>(n) * K + K - 1));
#pragma unroll
        for (int r = 0; r < RowTile; ++r) {
          const int m = m0 + r;
          if (m < M) {
            const float xv = __half2float(*reinterpret_cast<const __half*>(
                x + static_cast<int64_t>(m) * K + K - 1));
            acc[r][j] = fmaf(xv, wv, acc[r][j]);
          }
        }
      }
    }
  }
  __shared__ float partial[Threads / 32][RowTile][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float v = warp_sum(acc[r][j]);
      if (lane == 0) {
        partial[warp][r][j] = v;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          const int n = n0 + j;
          if (n < N) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < Threads / 32; ++w) {
              sum += partial[w][r][j];
            }
            *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) =
                __float2half_rn(sum);
          }
        }
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_row1_exact_f16_kernel(
    int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  for (int k2 = threadIdx.x; k2 < (K >> 1); k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(
          weight_orig + static_cast<int64_t>(n0 + j) * K + k));
      acc[j] = fmaf(xv.x, wv.x, acc[j]);
      acc[j] = fmaf(xv.y, wv.y, acc[j]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      y[n0 + j] = __float2half_rn(sum);
    }
  }
}

template <int Threads, int OutTile>
__global__
__launch_bounds__(Threads, 1) void linear_orig_row1_exact4_f16_kernel(
    int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 =
        __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const dtype* wj = weight_orig + static_cast<int64_t>(n0 + j) * K + k;
      const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(wj));
      const float2 w1 =
          __half22float2(*reinterpret_cast<const __half2*>(wj + 2));
      acc[j] = fmaf(x0.x, w0.x, acc[j]);
      acc[j] = fmaf(x0.y, w0.y, acc[j]);
      acc[j] = fmaf(x1.x, w1.x, acc[j]);
      acc[j] = fmaf(x1.y, w1.y, acc[j]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      y[n0 + j] = __float2half_rn(sum);
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_row2_exact_f16_kernel(
    int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  for (int k2 = threadIdx.x; k2 < (K >> 1); k2 += Threads) {
    const int k = k2 << 1;
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 =
        __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(
          weight_orig + static_cast<int64_t>(n0 + j) * K + k));
      acc0[j] = fmaf(x0.x, wv.x, acc0[j]);
      acc0[j] = fmaf(x0.y, wv.y, acc0[j]);
      acc1[j] = fmaf(x1.x, wv.x, acc1[j]);
      acc1[j] = fmaf(x1.y, wv.y, acc1[j]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (lane == 0) {
      partial[warp][0][j] = v0;
      partial[warp][1][j] = v1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum0 += partial[w][0][j];
        sum1 += partial[w][1][j];
      }
      const int n = n0 + j;
      y[n] = __float2half_rn(sum0);
      y[N + n] = __float2half_rn(sum1);
    }
  }
}

template <int Threads, int OutTile>
__global__
__launch_bounds__(Threads, 1) void linear_orig_row2_exact4_f16_kernel(
    int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x00 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x01 =
        __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
    const float2 x10 =
        __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
    const float2 x11 =
        __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 2));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const dtype* wj = weight_orig + static_cast<int64_t>(n0 + j) * K + k;
      const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(wj));
      const float2 w1 =
          __half22float2(*reinterpret_cast<const __half2*>(wj + 2));
      acc0[j] = fmaf(x00.x, w0.x, acc0[j]);
      acc0[j] = fmaf(x00.y, w0.y, acc0[j]);
      acc0[j] = fmaf(x01.x, w1.x, acc0[j]);
      acc0[j] = fmaf(x01.y, w1.y, acc0[j]);
      acc1[j] = fmaf(x10.x, w0.x, acc1[j]);
      acc1[j] = fmaf(x10.y, w0.y, acc1[j]);
      acc1[j] = fmaf(x11.x, w1.x, acc1[j]);
      acc1[j] = fmaf(x11.y, w1.y, acc1[j]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (lane == 0) {
      partial[warp][0][j] = v0;
      partial[warp][1][j] = v1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum0 += partial[w][0][j];
        sum1 += partial[w][1][j];
      }
      const int n = n0 + j;
      y[n] = __float2half_rn(sum0);
      y[N + n] = __float2half_rn(sum1);
    }
  }
}

__global__ __launch_bounds__(32, 8) void linear_orig_wmma16_f16_kernel(
    int M, int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * 16;
  const int m0 = blockIdx.y * 16;
  __shared__ __half a_tile[16 * 16];
  __shared__ __half b_tile[16 * 16];
  __shared__ float c_tile[16 * 16];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += 16) {
    for (int idx = threadIdx.x; idx < 16 * 16; idx += 32) {
      const int r = idx >> 4;
      const int kk = idx & 15;
      const int m = m0 + r;
      a_tile[idx] = (m < M && k0 + kk < K)
                        ? *reinterpret_cast<const __half*>(
                              x + static_cast<int64_t>(m) * K + k0 + kk)
                        : __float2half(0.0f);
      const int n = n0 + r;
      b_tile[r * 16 + kk] =
          (n < N && k0 + kk < K)
              ? *reinterpret_cast<const __half*>(
                    weight_orig + static_cast<int64_t>(n) * K + k0 + kk)
              : __float2half(0.0f);
    }
    __syncwarp();
    wmma::load_matrix_sync(a_frag, a_tile, 16);
    wmma::load_matrix_sync(b_frag, b_tile, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncwarp();
  }

  wmma::store_matrix_sync(c_tile, c_frag, 16, wmma::mem_row_major);
  __syncwarp();
  for (int idx = threadIdx.x; idx < 16 * 16; idx += 32) {
    const int r = idx >> 4;
    const int j = idx & 15;
    const int m = m0 + r;
    const int n = n0 + j;
    if (m < M && n < N) {
      *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) =
          __float2half_rn(c_tile[idx]);
    }
  }
}

template <int Threads, int OutTile, int Act>
__global__
__launch_bounds__(Threads, 2) void linear_t_act_f16_ntile_scalar_kernel(
    int M, int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = apply_act<Act>(
        __half2float(*reinterpret_cast<const __half*>(x_row + k)));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv,
                      __half2float(*reinterpret_cast<const __half*>(
                          weight_t + static_cast<int64_t>(n) * K + k)),
                      acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) =
            __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile, int Act>
__global__ __launch_bounds__(Threads, 2) void linear_t_act_f16_ntile_kernel(
    int M, int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t, dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    xv.x = apply_act<Act>(xv.x);
    xv.y = apply_act<Act>(xv.y);
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(
            weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv = apply_act<Act>(
        __half2float(*reinterpret_cast<const __half*>(x_row + K - 1)));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv,
                      __half2float(*reinterpret_cast<const __half*>(
                          weight_t + static_cast<int64_t>(n) * K + K - 1)),
                      acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) =
            __float2half_rn(sum);
      }
    }
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wag_rank_in_f16_kernel(
    int M, int K, int Rw, int Ra, int Rg, int Rmax,
    const dtype* __restrict__ xw, const dtype* __restrict__ xa,
    const dtype* __restrict__ xg, const dtype* __restrict__ w1_t,
    const dtype* __restrict__ a1_t, const dtype* __restrict__ g1_t,
    dtype* __restrict__ w1, dtype* __restrict__ a1, dtype* __restrict__ g1) {
  const int r = blockIdx.x;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* x = xw;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    x = xa;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    x = xg;
    wt = g1_t;
    y = g1;
  }
  if (m >= M || r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv =
        __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    const float2 wv =
        __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
    acc = fmaf(xv.x, wv.x, acc);
    acc = fmaf(xv.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * R + r) =
        __float2half_rn(acc);
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_in_f16_kernel(
    int M, int K, int Rw, int Ra, int Rg, int Rv, int Rmax,
    const dtype* __restrict__ xw, const dtype* __restrict__ xa,
    const dtype* __restrict__ xg, const dtype* __restrict__ xv,
    const dtype* __restrict__ w1_t, const dtype* __restrict__ a1_t,
    const dtype* __restrict__ g1_t, const dtype* __restrict__ v1_t,
    dtype* __restrict__ w1, dtype* __restrict__ a1, dtype* __restrict__ g1,
    dtype* __restrict__ v1) {
  const int r = blockIdx.x;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* x = xw;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    x = xa;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    x = xg;
    wt = g1_t;
    y = g1;
  } else if (group == 3) {
    R = Rv;
    x = xv;
    wt = v1_t;
    y = v1;
  }
  if (m >= M || r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv2 =
        __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    const float2 wv =
        __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
    acc = fmaf(xv2.x, wv.x, acc);
    acc = fmaf(xv2.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * R + r) =
        __float2half_rn(acc);
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_wag_rank_out_f16_kernel(
    int M, int C, int Kw, int Ka, int Kg, const dtype* __restrict__ w1,
    const dtype* __restrict__ a1, const dtype* __restrict__ g1,
    const dtype* __restrict__ w2_t, const dtype* __restrict__ a2_t,
    const dtype* __restrict__ g2_t, dtype* __restrict__ w,
    dtype* __restrict__ a, dtype* __restrict__ g) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int K = Kw;
  const dtype* x = w1;
  const dtype* wt = w2_t;
  dtype* y = w;
  if (group == 1) {
    K = Ka;
    x = a1;
    wt = a2_t;
    y = a;
  } else if (group == 2) {
    K = Kg;
    x = g1;
    wt = g2_t;
    y = g;
  }
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    if (group == 0) {
      xv = tanhf(xv);
    } else if (group == 2) {
      xv = 1.0f / (1.0f + expf(-xv));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < C) {
        acc[j] = fmaf(xv,
                      __half2float(*reinterpret_cast<const __half*>(
                          wt + static_cast<int64_t>(n) * K + k)),
                      acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int u = 0; u < Threads / 32; ++u) {
        sum += partial[u][j];
      }
      const int n = n0 + j;
      if (n < C) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * C + n) =
            __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_out_f16_kernel(
    int M, int C, int Kw, int Ka, int Kg, int Kv, const dtype* __restrict__ w1,
    const dtype* __restrict__ a1, const dtype* __restrict__ g1,
    const dtype* __restrict__ v1, const dtype* __restrict__ w2_t,
    const dtype* __restrict__ a2_t, const dtype* __restrict__ g2_t,
    const dtype* __restrict__ v2_t, const dtype* __restrict__ v,
    const dtype* __restrict__ v_first, const dtype* __restrict__ v0,
    dtype* __restrict__ w, dtype* __restrict__ a, dtype* __restrict__ g,
    dtype* __restrict__ v_out) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int K = Kw;
  const dtype* x = w1;
  const dtype* wt = w2_t;
  dtype* y = w;
  if (group == 1) {
    K = Ka;
    x = a1;
    wt = a2_t;
    y = a;
  } else if (group == 2) {
    K = Kg;
    x = g1;
    wt = g2_t;
    y = g;
  } else if (group == 3) {
    K = Kv;
    x = v1;
    wt = v2_t;
    y = v_out;
  }
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    if (group == 0) {
      xv = tanhf(xv);
    } else if (group == 2) {
      xv = 1.0f / (1.0f + expf(-xv));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < C) {
        acc[j] = fmaf(xv,
                      __half2float(*reinterpret_cast<const __half*>(
                          wt + static_cast<int64_t>(n) * K + k)),
                      acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int u = 0; u < Threads / 32; ++u) {
        sum += partial[u][j];
      }
      const int n = n0 + j;
      if (n < C) {
        if (group == 3) {
          const int64_t idx = static_cast<int64_t>(m) * C + n;
          const float vv =
              __half2float(*reinterpret_cast<const __half*>(v + idx));
          const float vf =
              __half2float(*reinterpret_cast<const __half*>(v_first + idx));
          const float gate =
              1.0f /
              (1.0f +
               expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) +
                      sum)));
          *reinterpret_cast<__half*>(y + idx) =
              __float2half_rn(vv + (vf - vv) * gate);
        } else {
          *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * C + n) =
              __float2half_rn(sum);
        }
      }
    }
  }
}

template <int Threads, int OutTile>
__global__
__launch_bounds__(Threads, 2) void linear_t_vres_f16_ntile_scalar_kernel(
    int M, int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t, const dtype* __restrict__ v,
    const dtype* __restrict__ v_first, const dtype* __restrict__ v0,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv,
                      __half2float(*reinterpret_cast<const __half*>(
                          weight_t + static_cast<int64_t>(n) * K + k)),
                      acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        const int64_t idx = static_cast<int64_t>(m) * N + n;
        const float vv =
            __half2float(*reinterpret_cast<const __half*>(v + idx));
        const float vf =
            __half2float(*reinterpret_cast<const __half*>(v_first + idx));
        const float gate =
            1.0f /
            (1.0f +
             expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) +
                    sum)));
        *reinterpret_cast<__half*>(y + idx) =
            __float2half_rn(vv + (vf - vv) * gate);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_vres_f16_ntile_kernel(
    int M, int K, int N, const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t, const dtype* __restrict__ v,
    const dtype* __restrict__ v_first, const dtype* __restrict__ v0,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv =
        __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(
            weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv =
        __half2float(*reinterpret_cast<const __half*>(x_row + K - 1));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv,
                      __half2float(*reinterpret_cast<const __half*>(
                          weight_t + static_cast<int64_t>(n) * K + K - 1)),
                      acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        const int64_t idx = static_cast<int64_t>(m) * N + n;
        const float vv =
            __half2float(*reinterpret_cast<const __half*>(v + idx));
        const float vf =
            __half2float(*reinterpret_cast<const __half*>(v_first + idx));
        const float gate =
            1.0f /
            (1.0f +
             expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) +
                    sum)));
        *reinterpret_cast<__half*>(y + idx) =
            __float2half_rn(vv + (vf - vv) * gate);
      }
    }
  }
}

__global__ void layer_norm_f16_kernel(int C, const dtype* __restrict__ x,
                                      const dtype* __restrict__ weight,
                                      const dtype* __restrict__ bias,
                                      dtype* __restrict__ y, int64_t rows,
                                      float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * C;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c));
    sum += v;
  }
  sum = block_sum_t<LN_THREADS>(sum);
  const float inv_c = 1.0f / static_cast<float>(C);
  const float mean = sum * inv_c;
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<LN_THREADS>(sum_var);
  const float var = sum_var * inv_c;
  const float rstd = rsqrtf(var + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c));
    const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
    const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
    *reinterpret_cast<__half*>(y + base + c) =
        __float2half_rn((v - mean) * rstd * w + b);
  }
}

__global__ void add_layer_norm_f16_kernel(
    int C, const dtype* __restrict__ x, const dtype* __restrict__ residual,
    const dtype* __restrict__ weight, const dtype* __restrict__ bias,
    dtype* __restrict__ x_out, dtype* __restrict__ y, int64_t rows, float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * C;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    sum += v;
  }
  sum = block_sum_t<LN_THREADS>(sum);
  const float inv_c = 1.0f / static_cast<float>(C);
  const float mean = sum * inv_c;
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<LN_THREADS>(sum_var);
  const float rstd = rsqrtf(sum_var * inv_c + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
    const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
    *reinterpret_cast<__half*>(x_out + base + c) = __float2half_rn(v);
    *reinterpret_cast<__half*>(y + base + c) =
        __float2half_rn((v - mean) * rstd * w + b);
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void layer_norm_f16_small_kernel(
    const dtype* __restrict__ x, const dtype* __restrict__ weight,
    const dtype* __restrict__ bias, dtype* __restrict__ y, int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v =
          __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      sum += v.x + v.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v =
          __half2float(*reinterpret_cast<const __half*>(x + base + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v =
          __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float dx = v.x - mean;
      const float dy = v.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v =
          __half2float(*reinterpret_cast<const __half*>(x + base + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd =
      rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v =
          __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 w =
          __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 b =
          __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(y + base)[idx] = __floats2half2_rn(
          (v.x - mean) * rstd * w.x + b.x, (v.y - mean) * rstd * w.y + b.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v =
          __half2float(*reinterpret_cast<const __half*>(x + base + c));
      const float w =
          __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(y + base + c) =
          __float2half_rn((v - mean) * rstd * w + b);
    }
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_f16_small_kernel(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    const dtype* __restrict__ weight, const dtype* __restrict__ bias,
    dtype* __restrict__ x_out, dtype* __restrict__ y, int64_t rows, float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv =
          __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(
          reinterpret_cast<const __half2*>(residual + base)[idx]);
      sum += xv.x + rv.x + xv.y + rv.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v =
          __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
          __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv =
          __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(
          reinterpret_cast<const __half2*>(residual + base)[idx]);
      const float dx = xv.x + rv.x - mean;
      const float dy = xv.y + rv.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v =
          __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
          __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd =
      rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv =
          __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(
          reinterpret_cast<const __half2*>(residual + base)[idx]);
      const float sx = xv.x + rv.x;
      const float sy = xv.y + rv.y;
      const float2 w =
          __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 b =
          __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(x_out + base)[idx] = __floats2half2_rn(sx, sy);
      reinterpret_cast<__half2*>(y + base)[idx] = __floats2half2_rn(
          (sx - mean) * rstd * w.x + b.x, (sy - mean) * rstd * w.y + b.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v =
          __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
          __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      const float w =
          __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(x_out + base + c) = __float2half_rn(v);
      *reinterpret_cast<__half*>(y + base + c) =
          __float2half_rn((v - mean) * rstd * w + b);
    }
  }
}

template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_kernel(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state, const dtype* __restrict__ weight,
    const dtype* __restrict__ bias, const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out, dtype* __restrict__ mixed, int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  const int64_t base2 = base >> 1;
  constexpr int pairs = LN_SMALL_C >> 1;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    sum_var += d0 * d0 + d1 * d1;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd =
      rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w =
        __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(
        reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x,
                                         (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] = __floats2half2_rn(
        yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_scalar_stats_kernel(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state, const dtype* __restrict__ weight,
    const dtype* __restrict__ bias, const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out, dtype* __restrict__ mixed, int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  const int64_t base2 = base >> 1;
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd =
      rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w =
        __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(
        reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x,
                                         (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] = __floats2half2_rn(
        yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_kernel(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state, const dtype* __restrict__ weight,
    const dtype* __restrict__ bias, const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w, const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v, const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g, dtype* __restrict__ x_out,
    dtype* __restrict__ out_r, dtype* __restrict__ out_w,
    dtype* __restrict__ out_k, dtype* __restrict__ out_v,
    dtype* __restrict__ out_a, dtype* __restrict__ out_g, int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base2 = row * (LN_SMALL_C >> 1);
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    sum_var += d0 * d0 + d1 * d1;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd =
      rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w =
        __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(
        reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x,
                                         (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_scalar_stats_kernel(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state, const dtype* __restrict__ weight,
    const dtype* __restrict__ bias, const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w, const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v, const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g, dtype* __restrict__ x_out,
    dtype* __restrict__ out_r, dtype* __restrict__ out_w,
    dtype* __restrict__ out_k, dtype* __restrict__ out_v,
    dtype* __restrict__ out_a, dtype* __restrict__ out_g, int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  const int64_t base2 = row * (LN_SMALL_C >> 1);
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
    const int c = threadIdx.x + k * Threads;
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd =
      rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w =
        __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(
        reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x,
                                         (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__
__launch_bounds__(Threads, 1) void add_last_layer_norm_f16_small_kernel(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    const dtype* __restrict__ weight, const dtype* __restrict__ bias,
    dtype* __restrict__ y, int64_t B, int64_t T, float eps) {
  const int64_t bidx = blockIdx.x;
  if (bidx >= B) {
    return;
  }
  const int64_t src = (bidx * T + (T - 1)) * LN_SMALL_C;
  const int64_t dst = bidx * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv =
          __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv =
          __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      sum += xv.x + rv.x + xv.y + rv.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v =
          __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
          __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv =
          __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv =
          __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      const float dx = xv.x + rv.x - mean;
      const float dy = xv.y + rv.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v =
          __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
          __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd =
      rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv =
          __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv =
          __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      const float sx = xv.x + rv.x;
      const float sy = xv.y + rv.y;
      const float2 w =
          __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 bb =
          __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(y + dst)[idx] = __floats2half2_rn(
          (sx - mean) * rstd * w.x + bb.x, (sy - mean) * rstd * w.y + bb.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v =
          __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
          __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      const float w =
          __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float bb = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(y + dst + c) =
          __float2half_rn((v - mean) * rstd * w + bb);
    }
  }
}

template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_last_layer_norm_f16_generic_kernel(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    const dtype* __restrict__ weight, const dtype* __restrict__ bias,
    dtype* __restrict__ y, int64_t B, int64_t T, int C, float eps) {
  const int64_t bidx = blockIdx.x;
  if (bidx >= B) {
    return;
  }
  const int64_t src = (bidx * T + (T - 1)) * static_cast<int64_t>(C);
  const int64_t dst = bidx * static_cast<int64_t>(C);
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    sum += __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + src + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum / static_cast<float>(C);
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + src + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var / static_cast<float>(C) + eps);
  const int pairs = C >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x + src)[p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual + src)[p]);
    const float sx = xv.x + rv.x;
    const float sy = xv.y + rv.y;
    const float2 w =
        __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 bb = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    reinterpret_cast<__half2*>(y + dst)[p] = __floats2half2_rn(
        (sx - mean) * rstd * w.x + bb.x, (sy - mean) * rstd * w.y + bb.y);
  }
}

template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_generic_kernel(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state, const dtype* __restrict__ weight,
    const dtype* __restrict__ bias, const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out, dtype* __restrict__ mixed, int64_t rows, int C,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * static_cast<int64_t>(C);
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum / static_cast<float>(C);
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var / static_cast<float>(C) + eps);
  const int pairs = C >> 1;
  const int64_t base2 = base >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w =
        __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(
        reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x,
                                         (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] = __floats2half2_rn(
        yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__
__launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_generic_kernel(
    const dtype* __restrict__ x, const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state, const dtype* __restrict__ weight,
    const dtype* __restrict__ bias, const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w, const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v, const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g, dtype* __restrict__ x_out,
    dtype* __restrict__ out_r, dtype* __restrict__ out_w,
    dtype* __restrict__ out_k, dtype* __restrict__ out_v,
    dtype* __restrict__ out_a, dtype* __restrict__ out_g, int64_t rows, int C,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * static_cast<int64_t>(C);
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    sum += __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
           __half2float(*reinterpret_cast<const __half*>(residual + base + c));
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum / static_cast<float>(C);
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += Threads) {
    const float v =
        __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
        __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var / static_cast<float>(C) + eps);
  const int pairs = C >> 1;
  const int64_t base2 = base >> 1;
  for (int p = threadIdx.x; p < pairs; p += Threads) {
    const float2 xv =
        __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv =
        __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w =
        __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(
        reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x,
                                         (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] =
        __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

}  // namespace

at::Tensor add_f16_cuda(at::Tensor x, at::Tensor y) {
  TORCH_CHECK((x.numel() % 2) == 0, "add_f16 requires even numel");
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t n_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  add_f16_kernel<<<static_cast<int>(ceil_div(n_pairs, threads)), threads, 0,
                   stream>>>(x.data_ptr<dtype>(), y.data_ptr<dtype>(),
                             out.data_ptr<dtype>(), n_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void advance_i32_cuda(at::Tensor x, int64_t amount) {
  TORCH_CHECK(amount >= INT_MIN && amount <= INT_MAX,
              "advance_i32 amount out of int range");
  constexpr int threads = 256;
  const int64_t n = x.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  advance_i32_kernel<<<static_cast<int>(ceil_div(n, threads)), threads, 0,
                       stream>>>(x.data_ptr<int>(), static_cast<int>(amount),
                                 n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor layer_norm_f16_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias,
                               double eps) {
  auto y = at::empty_like(x);
  const int64_t c64 = x.size(-1);
  TORCH_CHECK(c64 <= INT_MAX, "C too large");
  const int C = static_cast<int>(c64);
  const int64_t rows = x.numel() / C;
  TORCH_CHECK(rows <= INT_MAX, "rows too large");
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    if (rows >= 1024) {
      layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true>
          <<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
              x.data_ptr<dtype>(), weight.data_ptr<dtype>(),
              bias.data_ptr<dtype>(), y.data_ptr<dtype>(), rows,
              static_cast<float>(eps));
    } else if (rows >= 512) {
      layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false>
          <<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
              x.data_ptr<dtype>(), weight.data_ptr<dtype>(),
              bias.data_ptr<dtype>(), y.data_ptr<dtype>(), rows,
              static_cast<float>(eps));
    } else {
      layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false>
          <<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
              x.data_ptr<dtype>(), weight.data_ptr<dtype>(),
              bias.data_ptr<dtype>(), y.data_ptr<dtype>(), rows,
              static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  layer_norm_f16_kernel<<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
      C, x.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
      y.data_ptr<dtype>(), rows, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor layer_norm_f16_small_cuda(at::Tensor x, at::Tensor weight,
                                     at::Tensor bias, double eps) {
  auto y = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false>
      <<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          y.data_ptr<dtype>(), rows, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor layer_norm_f16_small512_cuda(at::Tensor x, at::Tensor weight,
                                        at::Tensor bias, double eps) {
  auto y = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false>
      <<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          y.data_ptr<dtype>(), rows, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor emb_ln0_bf16_to_f16_cuda(at::Tensor emb, at::Tensor weight,
                                    at::Tensor bias, double eps) {
  auto out = at::empty(emb.sizes(), emb.options().dtype(at::kHalf));
  const int64_t v64 = emb.size(0);
  const int64_t c64 = emb.size(1);
  TORCH_CHECK(v64 <= INT_MAX && c64 <= INT_MAX, "emb shape too large");
  const int V = static_cast<int>(v64);
  const int C = static_cast<int>(c64);
  auto stream = at::cuda::getCurrentCUDAStream();
  emb_ln0_bf16_to_f16_kernel<<<V, 256, 0, stream>>>(
      V, C, reinterpret_cast<const uint16_t*>(emb.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint16_t*>(weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const uint16_t*>(bias.data_ptr<at::BFloat16>()),
      out.data_ptr<dtype>(), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<at::Tensor> add_layer_norm_f16_cuda(at::Tensor x,
                                                at::Tensor residual,
                                                at::Tensor weight,
                                                at::Tensor bias, double eps) {
  auto x_out = at::empty_like(x);
  auto y = at::empty_like(x);
  const int64_t c64 = x.size(-1);
  TORCH_CHECK(c64 <= INT_MAX, "C too large");
  const int C = static_cast<int>(c64);
  const int64_t rows = x.numel() / C;
  TORCH_CHECK(rows <= INT_MAX, "rows too large");
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    if (rows >= 1024) {
      add_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true>
          <<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
              x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
              weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
              x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows,
              static_cast<float>(eps));
    } else if (rows >= 512) {
      add_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false>
          <<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
              x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
              weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
              x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows,
              static_cast<float>(eps));
    } else {
      add_layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false>
          <<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
              x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
              weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
              x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows,
              static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {x_out, y};
  }
  add_layer_norm_f16_kernel<<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
      C, x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
      weight.data_ptr<dtype>(), bias.data_ptr<dtype>(), x_out.data_ptr<dtype>(),
      y.data_ptr<dtype>(), rows, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, y};
}

at::Tensor add_last_layer_norm_f16_cuda(at::Tensor x, at::Tensor residual,
                                        at::Tensor weight, at::Tensor bias,
                                        double eps) {
  const int64_t B = x.size(0);
  const int64_t T = x.size(1);
  const int64_t C = x.size(2);
  TORCH_CHECK((C % 2) == 0, "add_last_layer_norm_f16 requires even C");
  auto y = at::empty({B, C}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C != LN_SMALL_C) {
    add_last_layer_norm_f16_generic_kernel<LN_THREADS>
        <<<static_cast<int>(B), LN_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            y.data_ptr<dtype>(), B, T, static_cast<int>(C),
            static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  if (B >= 1024) {
    add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true>
        <<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
  } else if (B >= 512) {
    add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false>
        <<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
  } else {
    add_last_layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false>
        <<<static_cast<int>(B), LN_SMALL_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
            y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_cuda(
    at::Tensor x, at::Tensor residual, at::Tensor shift_state,
    at::Tensor weight, at::Tensor bias, at::Tensor x_k, double eps) {
  auto x_out = at::empty_like(x);
  auto mixed = at::empty_like(x);
  const int64_t C = x.size(-1);
  TORCH_CHECK((C % 2) == 0, "add_layer_norm_cmix_mix_f16 requires even C");
  const int64_t rows = x.numel() / C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    add_layer_norm_cmix_mix_f16_scalar_stats_kernel<LN_SMALL_THREADS>
        <<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows,
            static_cast<float>(eps));
  } else {
    add_layer_norm_cmix_mix_f16_generic_kernel<LN_THREADS>
        <<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows,
            static_cast<int>(C), static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, mixed};
}

std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_scalar_stats_cuda(
    at::Tensor x, at::Tensor residual, at::Tensor shift_state,
    at::Tensor weight, at::Tensor bias, at::Tensor x_k, double eps) {
  auto x_out = at::empty_like(x);
  auto mixed = at::empty_like(x);
  const int64_t C = x.size(-1);
  TORCH_CHECK((C % 2) == 0, "add_layer_norm_cmix_mix_f16 requires even C");
  const int64_t rows = x.numel() / C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    add_layer_norm_cmix_mix_f16_scalar_stats_kernel<LN_SMALL_THREADS>
        <<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows,
            static_cast<float>(eps));
  } else {
    add_layer_norm_cmix_mix_f16_generic_kernel<LN_THREADS>
        <<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows,
            static_cast<int>(C), static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, mixed};
}

std::vector<at::Tensor> add_layer_norm_tmix_mix6_f16_cuda(
    at::Tensor x, at::Tensor residual, at::Tensor shift_state,
    at::Tensor weight, at::Tensor bias, at::Tensor x_r, at::Tensor x_w,
    at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g,
    double eps) {
  auto x_out = at::empty_like(x);
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t C = x.size(-1);
  TORCH_CHECK((C % 2) == 0, "add_layer_norm_tmix_mix6_f16 requires even C");
  const int64_t rows = x.numel() / C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    add_layer_norm_tmix_mix6_f16_scalar_stats_kernel<LN_SMALL_THREADS>
        <<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_r.data_ptr<dtype>(),
            x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(),
            x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), out_r.data_ptr<dtype>(),
            out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
            out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(),
            out_g.data_ptr<dtype>(), rows, static_cast<float>(eps));
  } else {
    add_layer_norm_tmix_mix6_f16_generic_kernel<LN_THREADS>
        <<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_r.data_ptr<dtype>(),
            x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(),
            x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), out_r.data_ptr<dtype>(),
            out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
            out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(),
            out_g.data_ptr<dtype>(), rows, static_cast<int>(C),
            static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> add_layer_norm_tmix_mix6_f16_cfg_cuda(
    at::Tensor x, at::Tensor residual, at::Tensor shift_state,
    at::Tensor weight, at::Tensor bias, at::Tensor x_r, at::Tensor x_w,
    at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g, double eps,
    int threads) {
  auto x_out = at::empty_like(x);
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (threads == 256) {
    add_layer_norm_tmix_mix6_f16_kernel<256>
        <<<static_cast<int>(rows), 256, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_r.data_ptr<dtype>(),
            x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(),
            x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), out_r.data_ptr<dtype>(),
            out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
            out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(),
            out_g.data_ptr<dtype>(), rows, static_cast<float>(eps));
  } else if (threads == 512) {
    add_layer_norm_tmix_mix6_f16_kernel<512>
        <<<static_cast<int>(rows), 512, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_r.data_ptr<dtype>(),
            x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(),
            x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), out_r.data_ptr<dtype>(),
            out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
            out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(),
            out_g.data_ptr<dtype>(), rows, static_cast<float>(eps));
  } else {
    add_layer_norm_tmix_mix6_f16_kernel<1024>
        <<<static_cast<int>(rows), 1024, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_r.data_ptr<dtype>(),
            x_w.data_ptr<dtype>(), x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(),
            x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), out_r.data_ptr<dtype>(),
            out_w.data_ptr<dtype>(), out_k.data_ptr<dtype>(),
            out_v.data_ptr<dtype>(), out_a.data_ptr<dtype>(),
            out_g.data_ptr<dtype>(), rows, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> add_layer_norm_tmix_mix6_f16_scalar_stats_cuda(
    at::Tensor x, at::Tensor residual, at::Tensor shift_state,
    at::Tensor weight, at::Tensor bias, at::Tensor x_r, at::Tensor x_w,
    at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g,
    double eps) {
  auto x_out = at::empty_like(x);
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  add_layer_norm_tmix_mix6_f16_scalar_stats_kernel<LN_SMALL_THREADS>
      <<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
          shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
          bias.data_ptr<dtype>(), x_r.data_ptr<dtype>(), x_w.data_ptr<dtype>(),
          x_k.data_ptr<dtype>(), x_v.data_ptr<dtype>(), x_a.data_ptr<dtype>(),
          x_g.data_ptr<dtype>(), x_out.data_ptr<dtype>(),
          out_r.data_ptr<dtype>(), out_w.data_ptr<dtype>(),
          out_k.data_ptr<dtype>(), out_v.data_ptr<dtype>(),
          out_a.data_ptr<dtype>(), out_g.data_ptr<dtype>(), rows,
          static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_cfg_cuda(
    at::Tensor x, at::Tensor residual, at::Tensor shift_state,
    at::Tensor weight, at::Tensor bias, at::Tensor x_k, double eps,
    int threads) {
  auto x_out = at::empty_like(x);
  auto mixed = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (threads == 256) {
    add_layer_norm_cmix_mix_f16_kernel<256>
        <<<static_cast<int>(rows), 256, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows,
            static_cast<float>(eps));
  } else if (threads == 512) {
    add_layer_norm_cmix_mix_f16_kernel<512>
        <<<static_cast<int>(rows), 512, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows,
            static_cast<float>(eps));
  } else {
    add_layer_norm_cmix_mix_f16_kernel<1024>
        <<<static_cast<int>(rows), 1024, 0, stream>>>(
            x.data_ptr<dtype>(), residual.data_ptr<dtype>(),
            shift_state.data_ptr<dtype>(), weight.data_ptr<dtype>(),
            bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
            x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows,
            static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, mixed};
}

at::Tensor linear_f16_cuda(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16 K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16 M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  // Row-major y[M,N] = x[M,K] @ weight[K,N] is column-major
  // y^T[N,M] = weight^T[N,K] @ x^T[K,M].
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                   weight.data_ptr<dtype>(), CUDA_R_16F, n, x.data_ptr<dtype>(),
                   CUDA_R_16F, k, &beta, y.data_ptr<dtype>(), CUDA_R_16F, n,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "linear_f16 cublasGemmEx");
  return y;
}

at::Tensor linear_f16_orig_cuda(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX,
              "linear_f16_orig K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_orig M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  // weight_orig is row-major [N,K], i.e. column-major [K,N].
  // Row-major y[M,N] = x[M,K] @ weight_orig[N,K]^T becomes
  // column-major y^T[N,M] = opT(weight_orig_col[K,N]) @ x_col[K,M].
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                            weight_orig.data_ptr<dtype>(), CUDA_R_16F, k,
                            x.data_ptr<dtype>(), CUDA_R_16F, k, &beta,
                            y.data_ptr<dtype>(), CUDA_R_16F, n,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
               "linear_f16_orig cublasGemmEx");
  return y;
}

template <int RowTile, int OutTile>
at::Tensor linear_orig_rows_f16_cuda_impl(at::Tensor x,
                                          at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX,
              "linear_orig_rows_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_orig_rows_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_orig_rows_f16_kernel<128, RowTile, OutTile>
      <<<dim3(ceil_div(N, OutTile), ceil_div(M, RowTile), 1), 128, 0, stream>>>(
          M, K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(),
          y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

template <int Threads, int RowTile, int OutTile>
at::Tensor linear_orig_rows_cfg_f16_cuda_impl(at::Tensor x,
                                              at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX,
              "linear_orig_rows_cfg_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_orig_rows_cfg_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_orig_rows_f16_kernel<Threads, RowTile, OutTile>
      <<<dim3(ceil_div(N, OutTile), ceil_div(M, RowTile), 1), Threads, 0,
         stream>>>(M, K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(),
                   y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_orig_rows_f16_cuda(at::Tensor x, at::Tensor weight_orig,
                                     int64_t row_tile, int64_t out_tile) {
  if (row_tile == 1 && out_tile == 2)
    return linear_orig_rows_f16_cuda_impl<1, 2>(x, weight_orig);
  if (row_tile == 1 && out_tile == 4)
    return linear_orig_rows_f16_cuda_impl<1, 4>(x, weight_orig);
  if (row_tile == 1 && out_tile == 8)
    return linear_orig_rows_f16_cuda_impl<1, 8>(x, weight_orig);
  if (row_tile == 1 && out_tile == 16)
    return linear_orig_rows_f16_cuda_impl<1, 16>(x, weight_orig);
  if (row_tile == 2 && out_tile == 2)
    return linear_orig_rows_f16_cuda_impl<2, 2>(x, weight_orig);
  if (row_tile == 2 && out_tile == 4)
    return linear_orig_rows_f16_cuda_impl<2, 4>(x, weight_orig);
  if (row_tile == 2 && out_tile == 8)
    return linear_orig_rows_f16_cuda_impl<2, 8>(x, weight_orig);
  if (row_tile == 3 && out_tile == 2)
    return linear_orig_rows_f16_cuda_impl<3, 2>(x, weight_orig);
  if (row_tile == 3 && out_tile == 4)
    return linear_orig_rows_f16_cuda_impl<3, 4>(x, weight_orig);
  if (row_tile == 3 && out_tile == 8)
    return linear_orig_rows_f16_cuda_impl<3, 8>(x, weight_orig);
  if (row_tile == 4 && out_tile == 2)
    return linear_orig_rows_f16_cuda_impl<4, 2>(x, weight_orig);
  if (row_tile == 4 && out_tile == 4)
    return linear_orig_rows_f16_cuda_impl<4, 4>(x, weight_orig);
  if (row_tile == 4 && out_tile == 8)
    return linear_orig_rows_f16_cuda_impl<4, 8>(x, weight_orig);
  if (row_tile == 8 && out_tile == 2)
    return linear_orig_rows_f16_cuda_impl<8, 2>(x, weight_orig);
  if (row_tile == 8 && out_tile == 4)
    return linear_orig_rows_f16_cuda_impl<8, 4>(x, weight_orig);
  if (row_tile == 16 && out_tile == 1)
    return linear_orig_rows_f16_cuda_impl<16, 1>(x, weight_orig);
  if (row_tile == 16 && out_tile == 2)
    return linear_orig_rows_f16_cuda_impl<16, 2>(x, weight_orig);
  if (row_tile == 16 && out_tile == 4)
    return linear_orig_rows_f16_cuda_impl<16, 4>(x, weight_orig);
  TORCH_CHECK(false, "unsupported linear_orig_rows_f16 row_tile/out_tile");
}

at::Tensor linear_orig_rows_cfg_f16_cuda(at::Tensor x, at::Tensor weight_orig,
                                         int64_t threads, int64_t row_tile,
                                         int64_t out_tile) {
  if (threads == 64 && row_tile == 1 && out_tile == 4)
    return linear_orig_rows_cfg_f16_cuda_impl<64, 1, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 1 && out_tile == 8)
    return linear_orig_rows_cfg_f16_cuda_impl<64, 1, 8>(x, weight_orig);
  if (threads == 128 && row_tile == 1 && out_tile == 8)
    return linear_orig_rows_cfg_f16_cuda_impl<128, 1, 8>(x, weight_orig);
  if (threads == 256 && row_tile == 1 && out_tile == 1)
    return linear_orig_rows_cfg_f16_cuda_impl<256, 1, 1>(x, weight_orig);
  if (threads == 32 && row_tile == 4 && out_tile == 4)
    return linear_orig_rows_cfg_f16_cuda_impl<32, 4, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 4 && out_tile == 4)
    return linear_orig_rows_cfg_f16_cuda_impl<64, 4, 4>(x, weight_orig);
  if (threads == 96 && row_tile == 4 && out_tile == 4)
    return linear_orig_rows_cfg_f16_cuda_impl<96, 4, 4>(x, weight_orig);
  if (threads == 32 && row_tile == 4 && out_tile == 8)
    return linear_orig_rows_cfg_f16_cuda_impl<32, 4, 8>(x, weight_orig);
  if (threads == 64 && row_tile == 4 && out_tile == 8)
    return linear_orig_rows_cfg_f16_cuda_impl<64, 4, 8>(x, weight_orig);
  if (threads == 32 && row_tile == 8 && out_tile == 4)
    return linear_orig_rows_cfg_f16_cuda_impl<32, 8, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 8 && out_tile == 4)
    return linear_orig_rows_cfg_f16_cuda_impl<64, 8, 4>(x, weight_orig);
  if (threads == 32 && row_tile == 2 && out_tile == 4)
    return linear_orig_rows_cfg_f16_cuda_impl<32, 2, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 2 && out_tile == 2)
    return linear_orig_rows_cfg_f16_cuda_impl<64, 2, 2>(x, weight_orig);
  if (threads == 64 && row_tile == 2 && out_tile == 4)
    return linear_orig_rows_cfg_f16_cuda_impl<64, 2, 4>(x, weight_orig);
  if (threads == 32 && row_tile == 3 && out_tile == 4)
    return linear_orig_rows_cfg_f16_cuda_impl<32, 3, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 3 && out_tile == 4)
    return linear_orig_rows_cfg_f16_cuda_impl<64, 3, 4>(x, weight_orig);
  if (threads == 96 && row_tile == 3 && out_tile == 4)
    return linear_orig_rows_cfg_f16_cuda_impl<96, 3, 4>(x, weight_orig);
  if (threads == 32 && row_tile == 3 && out_tile == 8)
    return linear_orig_rows_cfg_f16_cuda_impl<32, 3, 8>(x, weight_orig);
  if (threads == 64 && row_tile == 3 && out_tile == 8)
    return linear_orig_rows_cfg_f16_cuda_impl<64, 3, 8>(x, weight_orig);
  TORCH_CHECK(false,
              "unsupported linear_orig_rows_cfg_f16 threads/row_tile/out_tile");
}

template <int Threads, int OutTile, bool Use4>
at::Tensor linear_orig_row1_exact_f16_cuda_impl(at::Tensor x,
                                                at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX,
              "linear_orig_row1_exact_f16 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0,
              "linear_orig_row1_exact_f16 requires N divisible by out_tile");
  TORCH_CHECK((k64 % (Use4 ? 4 : 2)) == 0,
              "linear_orig_row1_exact_f16 unsupported K alignment");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 1, "linear_orig_row1_exact_f16 requires one row");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if constexpr (Use4) {
    linear_orig_row1_exact4_f16_kernel<Threads, OutTile>
        <<<N / OutTile, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(),
            y.data_ptr<dtype>());
  } else {
    linear_orig_row1_exact_f16_kernel<Threads, OutTile>
        <<<N / OutTile, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(),
            y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

template <int Threads, int OutTile, bool Use4>
at::Tensor linear_orig_row2_exact_f16_cuda_impl(at::Tensor x,
                                                at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX,
              "linear_orig_row2_exact_f16 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0,
              "linear_orig_row2_exact_f16 requires N divisible by out_tile");
  TORCH_CHECK((k64 % (Use4 ? 4 : 2)) == 0,
              "linear_orig_row2_exact_f16 unsupported K alignment");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 2, "linear_orig_row2_exact_f16 requires two rows");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if constexpr (Use4) {
    linear_orig_row2_exact4_f16_kernel<Threads, OutTile>
        <<<N / OutTile, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(),
            y.data_ptr<dtype>());
  } else {
    linear_orig_row2_exact_f16_kernel<Threads, OutTile>
        <<<N / OutTile, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(),
            y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_orig_rows_exact_f16_cuda(at::Tensor x, at::Tensor weight_orig,
                                           int64_t threads, int64_t out_tile,
                                           bool use4) {
  const int64_t rows = x.numel() / x.size(-1);
  if (rows == 1) {
    if (!use4 && threads == 128 && out_tile == 2)
      return linear_orig_row1_exact_f16_cuda_impl<128, 2, false>(x,
                                                                 weight_orig);
    if (use4 && threads == 128 && out_tile == 2)
      return linear_orig_row1_exact_f16_cuda_impl<128, 2, true>(x, weight_orig);
  }
  if (rows == 2) {
    if (use4 && threads == 64 && out_tile == 2)
      return linear_orig_row2_exact_f16_cuda_impl<64, 2, true>(x, weight_orig);
    if (use4 && threads == 256 && out_tile == 1)
      return linear_orig_row2_exact_f16_cuda_impl<256, 1, true>(x, weight_orig);
    if (!use4 && threads == 128 && out_tile == 2)
      return linear_orig_row2_exact_f16_cuda_impl<128, 2, false>(x,
                                                                 weight_orig);
  }
  TORCH_CHECK(
      false,
      "unsupported linear_orig_rows_exact_f16 rows/threads/out_tile/use4");
}

at::Tensor linear_orig_wmma16_f16_cuda(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX,
              "linear_orig_wmma16_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_orig_wmma16_f16 M too large");
  const int M = static_cast<int>(m64);
  TORCH_CHECK((K % 16) == 0 && (N % 16) == 0,
              "linear_orig_wmma16_f16 requires K/N multiple of 16");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_orig_wmma16_f16_kernel<<<dim3(N / 16, ceil_div(M, 16), 1), 32, 0,
                                  stream>>>(M, K, N, x.data_ptr<dtype>(),
                                            weight_orig.data_ptr<dtype>(),
                                            y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_f16_orig_lt_cfg_cuda(at::Tensor x, at::Tensor weight_orig,
                                       int64_t workspace_mb,
                                       int64_t algo_index);

at::Tensor linear_f16_orig_lt_cuda(at::Tensor x, at::Tensor weight_orig) {
  return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
}

at::Tensor linear_f16_orig_lt_cfg_cuda(at::Tensor x, at::Tensor weight_orig,
                                       int64_t workspace_mb,
                                       int64_t algo_index) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX,
              "linear_f16_orig_lt_cfg K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_orig_lt_cfg M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  const size_t workspace_size = static_cast<size_t>(workspace_mb) << 20;
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)},
                          x.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }

  static cublasLtHandle_t lt_handle = nullptr;
  if (lt_handle == nullptr) {
    check_cublaslt(cublasLtCreate(&lt_handle), "cublasLtCreate");
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  check_cublaslt(
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
      "linear_f16_orig_lt desc");
  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  check_cublaslt(
      cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &transa, sizeof(transa)),
      "linear_f16_orig_lt transa");
  check_cublaslt(
      cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &transb, sizeof(transb)),
      "linear_f16_orig_lt transb");
  check_cublaslt(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, k, n, k),
                 "linear_f16_orig_lt a layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, k, m, k),
                 "linear_f16_orig_lt b layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, n, m, n),
                 "linear_f16_orig_lt c layout");
  check_cublaslt(cublasLtMatmulPreferenceCreate(&pref),
                 "linear_f16_orig_lt preference");
  check_cublaslt(cublasLtMatmulPreferenceSetAttribute(
                     pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                     &workspace_size, sizeof(workspace_size)),
                 "linear_f16_orig_lt workspace");

  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(64);
  int returned = 0;
  check_cublaslt(
      cublasLtMatmulAlgoGetHeuristic(
          lt_handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref,
          static_cast<int>(heuristics.size()), heuristics.data(), &returned),
      "linear_f16_orig_lt heuristic");
  TORCH_CHECK(returned > 0, "linear_f16_orig_lt found no algorithm");
  const int selected_algo =
      algo_index < returned ? static_cast<int>(algo_index) : 0;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublaslt(
      cublasLtMatmul(lt_handle, op_desc, &alpha, weight_orig.data_ptr<dtype>(),
                     a_desc, x.data_ptr<dtype>(), b_desc, &beta,
                     y.data_ptr<dtype>(), c_desc, y.data_ptr<dtype>(), c_desc,
                     &heuristics[selected_algo].algo, workspace_ptr,
                     workspace_size, at::cuda::getCurrentCUDAStream()),
      "linear_f16_orig_lt matmul");
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(op_desc);
  return y;
}

at::Tensor linear_f16_lt_cuda(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_lt K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_lt M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  static cublasLtHandle_t lt_handle = nullptr;
  if (lt_handle == nullptr) {
    check_cublaslt(cublasLtCreate(&lt_handle), "cublasLtCreate");
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  check_cublaslt(
      cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
      "cublasLtMatmulDescCreate");
  const cublasOperation_t trans = CUBLAS_OP_N;
  check_cublaslt(
      cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &trans, sizeof(trans)),
      "cublasLt set transa");
  check_cublaslt(
      cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &trans, sizeof(trans)),
      "cublasLt set transb");
  check_cublaslt(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, n, k, n),
                 "cublasLt a layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, k, m, k),
                 "cublasLt b layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, n, m, n),
                 "cublasLt c layout");
  check_cublaslt(cublasLtMatmulPreferenceCreate(&pref), "cublasLt preference");
  const size_t workspace_size = 0;
  check_cublaslt(cublasLtMatmulPreferenceSetAttribute(
                     pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                     &workspace_size, sizeof(workspace_size)),
                 "cublasLt set workspace");

  cublasLtMatmulHeuristicResult_t heuristic = {};
  int returned = 0;
  check_cublaslt(
      cublasLtMatmulAlgoGetHeuristic(lt_handle, op_desc, a_desc, b_desc, c_desc,
                                     c_desc, pref, 1, &heuristic, &returned),
      "cublasLt heuristic");
  TORCH_CHECK(returned > 0, "cublasLt found no algorithm");
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublaslt(
      cublasLtMatmul(lt_handle, op_desc, &alpha, weight.data_ptr<dtype>(),
                     a_desc, x.data_ptr<dtype>(), b_desc, &beta,
                     y.data_ptr<dtype>(), c_desc, y.data_ptr<dtype>(), c_desc,
                     &heuristic.algo, nullptr, 0,
                     at::cuda::getCurrentCUDAStream()),
      "cublasLtMatmul");
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(op_desc);
  return y;
}

template <int ChunkK, int Warps, bool WarpReduce = false>
at::Tensor linear_f16_m1_splitk_cuda_impl(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX,
              "linear_f16_m1_splitk K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  TORCH_CHECK(x.numel() == k64, "linear_f16_m1_splitk requires M=1");
  TORCH_CHECK((N % 64) == 0, "linear_f16_m1_splitk requires N multiple of 64");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (K == 0 || N == 0) {
    return y;
  }
  const int chunks = static_cast<int>(ceil_div(K, ChunkK));
  auto partial = at::empty({chunks, n64}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_f16_m1_splitk_partial_kernel<ChunkK, Warps>
      <<<dim3(ceil_div(N, Warps * 64), chunks, 1), Warps * 32, 0, stream>>>(
          K, N, x.data_ptr<dtype>(), weight.data_ptr<dtype>(),
          partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if constexpr (WarpReduce) {
    linear_f16_m1_splitk_reduce_warp_kernel<<<
        static_cast<int>(ceil_div(N / 2, 4)), 128, 0, stream>>>(
        chunks, N, partial.data_ptr<float>(), y.data_ptr<dtype>());
  } else {
    linear_f16_m1_splitk_reduce_kernel<<<static_cast<int>(ceil_div(N / 2, 128)),
                                         128, 0, stream>>>(
        chunks, N, partial.data_ptr<float>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_f16_m1_splitk_cuda(at::Tensor x, at::Tensor weight) {
  const int64_t K = x.size(-1);
  const int64_t N = weight.size(1);
  if (K == 4096 && N == 4096) {
    return linear_f16_m1_splitk_cuda_impl<160, 1, true>(x, weight);
  }
  if (N >= 65536) {
    return linear_f16_m1_splitk_cuda_impl<768, 2>(x, weight);
  }
  if (K == 4096 && N == 16384) {
    return linear_f16_m1_splitk_cuda_impl<512, 2>(x, weight);
  }
  if (K >= 8192) {
    return linear_f16_m1_splitk_cuda_impl<512, 2>(x, weight);
  }
  return linear_f16_m1_splitk_cuda_impl<256, 4>(x, weight);
}

at::Tensor linear_f16_m1_splitk_cfg_cuda(at::Tensor x, at::Tensor weight,
                                         int64_t chunk_k) {
  switch (chunk_k) {
    case 64:
      return linear_f16_m1_splitk_cuda_impl<64, 4>(x, weight);
    case 96:
      return linear_f16_m1_splitk_cuda_impl<96, 4>(x, weight);
    case 112:
      return linear_f16_m1_splitk_cuda_impl<112, 4>(x, weight);
    case 128:
      return linear_f16_m1_splitk_cuda_impl<128, 4>(x, weight);
    case 144:
      return linear_f16_m1_splitk_cuda_impl<144, 4>(x, weight);
    case 152:
      return linear_f16_m1_splitk_cuda_impl<152, 4>(x, weight);
    case 160:
      return linear_f16_m1_splitk_cuda_impl<160, 4>(x, weight);
    case 168:
      return linear_f16_m1_splitk_cuda_impl<168, 4>(x, weight);
    case 176:
      return linear_f16_m1_splitk_cuda_impl<176, 4>(x, weight);
    case 184:
      return linear_f16_m1_splitk_cuda_impl<184, 4>(x, weight);
    case 192:
      return linear_f16_m1_splitk_cuda_impl<192, 4>(x, weight);
    case 208:
      return linear_f16_m1_splitk_cuda_impl<208, 4>(x, weight);
    case 224:
      return linear_f16_m1_splitk_cuda_impl<224, 4>(x, weight);
    case 256:
      return linear_f16_m1_splitk_cuda_impl<256, 4>(x, weight);
    case 384:
      return linear_f16_m1_splitk_cuda_impl<384, 4>(x, weight);
    case 512:
      return linear_f16_m1_splitk_cuda_impl<512, 4>(x, weight);
    case 640:
      return linear_f16_m1_splitk_cuda_impl<640, 4>(x, weight);
    case 768:
      return linear_f16_m1_splitk_cuda_impl<768, 4>(x, weight);
    case 896:
      return linear_f16_m1_splitk_cuda_impl<896, 4>(x, weight);
    case 1024:
      return linear_f16_m1_splitk_cuda_impl<1024, 4>(x, weight);
    case 2048:
      return linear_f16_m1_splitk_cuda_impl<2048, 4>(x, weight);
    case 4096:
      return linear_f16_m1_splitk_cuda_impl<4096, 4>(x, weight);
    default:
      TORCH_CHECK(false, "unsupported chunk_k");
  }
}

at::Tensor linear_f16_m1_splitk_tile_cuda(at::Tensor x, at::Tensor weight,
                                          int64_t chunk_k, int64_t tile_cols) {
  if (tile_cols == 64) {
    switch (chunk_k) {
      case 64:
        return linear_f16_m1_splitk_cuda_impl<64, 1>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 1>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 1>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 1>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 1>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 1>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 1>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 1>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 1>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 1>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 1>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 1>(x, weight);
      case 224:
        return linear_f16_m1_splitk_cuda_impl<224, 1>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 1>(x, weight);
      case 384:
        return linear_f16_m1_splitk_cuda_impl<384, 1>(x, weight);
      case 512:
        return linear_f16_m1_splitk_cuda_impl<512, 1>(x, weight);
      case 640:
        return linear_f16_m1_splitk_cuda_impl<640, 1>(x, weight);
      case 768:
        return linear_f16_m1_splitk_cuda_impl<768, 1>(x, weight);
      case 896:
        return linear_f16_m1_splitk_cuda_impl<896, 1>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported chunk_k");
    }
  }
  if (tile_cols == 128) {
    switch (chunk_k) {
      case 64:
        return linear_f16_m1_splitk_cuda_impl<64, 2>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 2>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 2>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 2>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 2>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 2>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 2>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 2>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 2>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 2>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 2>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 2>(x, weight);
      case 224:
        return linear_f16_m1_splitk_cuda_impl<224, 2>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 2>(x, weight);
      case 384:
        return linear_f16_m1_splitk_cuda_impl<384, 2>(x, weight);
      case 512:
        return linear_f16_m1_splitk_cuda_impl<512, 2>(x, weight);
      case 640:
        return linear_f16_m1_splitk_cuda_impl<640, 2>(x, weight);
      case 768:
        return linear_f16_m1_splitk_cuda_impl<768, 2>(x, weight);
      case 896:
        return linear_f16_m1_splitk_cuda_impl<896, 2>(x, weight);
      case 1024:
        return linear_f16_m1_splitk_cuda_impl<1024, 2>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported chunk_k");
    }
  }
  TORCH_CHECK(tile_cols == 256, "unsupported tile_cols");
  return linear_f16_m1_splitk_cfg_cuda(x, weight, chunk_k);
}

at::Tensor linear_f16_m1_splitk_warpred_tile_cuda(at::Tensor x,
                                                  at::Tensor weight,
                                                  int64_t chunk_k,
                                                  int64_t tile_cols) {
  if (tile_cols == 64) {
    switch (chunk_k) {
      case 64:
        return linear_f16_m1_splitk_cuda_impl<64, 1, true>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 1, true>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 1, true>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 1, true>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 1, true>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 1, true>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 1, true>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 1, true>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 1, true>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 1, true>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 1, true>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 1, true>(x, weight);
      case 224:
        return linear_f16_m1_splitk_cuda_impl<224, 1, true>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 1, true>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported warpred chunk_k");
    }
  }
  if (tile_cols == 128) {
    switch (chunk_k) {
      case 64:
        return linear_f16_m1_splitk_cuda_impl<64, 2, true>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 2, true>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 2, true>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 2, true>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 2, true>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 2, true>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 2, true>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 2, true>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 2, true>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 2, true>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 2, true>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 2, true>(x, weight);
      case 224:
        return linear_f16_m1_splitk_cuda_impl<224, 2, true>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 2, true>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported warpred chunk_k");
    }
  }
  TORCH_CHECK(tile_cols == 256, "unsupported warpred tile_cols");
  switch (chunk_k) {
    case 64:
      return linear_f16_m1_splitk_cuda_impl<64, 4, true>(x, weight);
    case 96:
      return linear_f16_m1_splitk_cuda_impl<96, 4, true>(x, weight);
    case 112:
      return linear_f16_m1_splitk_cuda_impl<112, 4, true>(x, weight);
    case 128:
      return linear_f16_m1_splitk_cuda_impl<128, 4, true>(x, weight);
    case 144:
      return linear_f16_m1_splitk_cuda_impl<144, 4, true>(x, weight);
    case 152:
      return linear_f16_m1_splitk_cuda_impl<152, 4, true>(x, weight);
    case 160:
      return linear_f16_m1_splitk_cuda_impl<160, 4, true>(x, weight);
    case 168:
      return linear_f16_m1_splitk_cuda_impl<168, 4, true>(x, weight);
    case 176:
      return linear_f16_m1_splitk_cuda_impl<176, 4, true>(x, weight);
    case 184:
      return linear_f16_m1_splitk_cuda_impl<184, 4, true>(x, weight);
    case 192:
      return linear_f16_m1_splitk_cuda_impl<192, 4, true>(x, weight);
    case 208:
      return linear_f16_m1_splitk_cuda_impl<208, 4, true>(x, weight);
    case 224:
      return linear_f16_m1_splitk_cuda_impl<224, 4, true>(x, weight);
    case 256:
      return linear_f16_m1_splitk_cuda_impl<256, 4, true>(x, weight);
    default:
      TORCH_CHECK(false, "unsupported warpred chunk_k");
  }
}

template <int ChunkK, int Warps>
at::Tensor linear_f16_rows_splitk_cuda_impl(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX,
              "linear_f16_rows_splitk K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_rows_splitk M too large");
  const int M = static_cast<int>(m64);
  TORCH_CHECK((N % 64) == 0,
              "linear_f16_rows_splitk requires N multiple of 64");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || K == 0 || N == 0) {
    return y;
  }
  const int chunks = static_cast<int>(ceil_div(K, ChunkK));
  auto partial = at::empty({m64, chunks, n64}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_f16_rows_splitk_partial_kernel<ChunkK, Warps>
      <<<dim3(ceil_div(N, Warps * 64), chunks, M), Warps * 32, 0, stream>>>(
          K, N, chunks, x.data_ptr<dtype>(), weight.data_ptr<dtype>(),
          partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_f16_rows_splitk_reduce_kernel<<<
      dim3(static_cast<int>(ceil_div(N / 2, 128)), M, 1), 128, 0, stream>>>(
      chunks, N, partial.data_ptr<float>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_f16_rows_splitk_cuda(at::Tensor x, at::Tensor weight,
                                       int64_t chunk_k) {
  switch (chunk_k) {
    case 128:
      return linear_f16_rows_splitk_cuda_impl<128, 2>(x, weight);
    case 256:
      return linear_f16_rows_splitk_cuda_impl<256, 2>(x, weight);
    case 512:
      return linear_f16_rows_splitk_cuda_impl<512, 2>(x, weight);
    case 1024:
      return linear_f16_rows_splitk_cuda_impl<1024, 2>(x, weight);
    default:
      TORCH_CHECK(false, "unsupported chunk_k");
  }
}

at::Tensor linear_t_f16_cuda(at::Tensor x, at::Tensor weight_t) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_t_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_t_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (K <= 512 && N >= 1024 && M <= 4) {
    if (M == 1) {
      linear_t_f16_ntile_scalar_kernel<128, 2>
          <<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
              M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(),
              y.data_ptr<dtype>());
    } else {
      linear_t_f16_ntile_kernel<128, 4>
          <<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
              M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(),
              y.data_ptr<dtype>());
    }
  } else if (K >= 1024) {
    linear_t_f16_kernel<256><<<dim3(N, M, 1), 256, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(),
        y.data_ptr<dtype>());
  } else {
    linear_t_f16_kernel<128><<<dim3(N, M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(),
        y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

template <int Act>
at::Tensor linear_t_act_f16_cuda_impl(at::Tensor x, at::Tensor weight_t) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX,
              "linear_t_act_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_t_act_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(K <= 512 && N >= 1024 && M <= 4,
              "linear_t_act_f16 currently supports only small-rank rank-out");
  if (M == 1) {
    linear_t_act_f16_ntile_scalar_kernel<128, 2, Act>
        <<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
            M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(),
            y.data_ptr<dtype>());
  } else {
    linear_t_act_f16_ntile_kernel<128, 4, Act>
        <<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
            M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(),
            y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_t_act_f16_cuda(at::Tensor x, at::Tensor weight_t,
                                 int64_t act) {
  if (act == 1) {
    return linear_t_act_f16_cuda_impl<1>(x, weight_t);
  }
  return linear_t_act_f16_cuda_impl<2>(x, weight_t);
}

std::vector<at::Tensor> linear_wag_rank_in_f16_cuda(
    at::Tensor xw, at::Tensor xa, at::Tensor xg, at::Tensor w1_t,
    at::Tensor a1_t, at::Tensor g1_t) {
  const int64_t k64 = xw.size(-1);
  const int64_t rw64 = w1_t.size(0);
  const int64_t ra64 = a1_t.size(0);
  const int64_t rg64 = g1_t.size(0);
  const int64_t m64 = xw.numel() / k64;
  TORCH_CHECK(k64 <= INT_MAX && rw64 <= INT_MAX && ra64 <= INT_MAX &&
                  rg64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wag_rank_in_f16 shape too large");
  const int K = static_cast<int>(k64);
  const int Rw = static_cast<int>(rw64);
  const int Ra = static_cast<int>(ra64);
  const int Rg = static_cast<int>(rg64);
  const int Rmax = std::max(Rw, std::max(Ra, Rg));
  const int M = static_cast<int>(m64);
  TORCH_CHECK(K >= 1024 && Rmax <= 512 && M <= 8,
              "linear_wag_rank_in_f16 supports only K>=1024,R<=512,M<=8");
  std::vector<int64_t> w_sizes(xw.sizes().begin(), xw.sizes().end());
  std::vector<int64_t> a_sizes = w_sizes;
  std::vector<int64_t> g_sizes = w_sizes;
  w_sizes.back() = rw64;
  a_sizes.back() = ra64;
  g_sizes.back() = rg64;
  auto w1 = at::empty(w_sizes, xw.options());
  auto a1 = at::empty(a_sizes, xw.options());
  auto g1 = at::empty(g_sizes, xw.options());
  if (M == 0 || K == 0 || Rmax == 0) {
    return {w1, a1, g1};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_wag_rank_in_f16_kernel<256><<<dim3(Rmax, M, 3), 256, 0, stream>>>(
      M, K, Rw, Ra, Rg, Rmax, xw.data_ptr<dtype>(), xa.data_ptr<dtype>(),
      xg.data_ptr<dtype>(), w1_t.data_ptr<dtype>(), a1_t.data_ptr<dtype>(),
      g1_t.data_ptr<dtype>(), w1.data_ptr<dtype>(), a1.data_ptr<dtype>(),
      g1.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w1, a1, g1};
}

std::vector<at::Tensor> linear_wagv_rank_in_f16_cuda(
    at::Tensor xw, at::Tensor xa, at::Tensor xg, at::Tensor xv, at::Tensor w1_t,
    at::Tensor a1_t, at::Tensor g1_t, at::Tensor v1_t) {
  const int64_t k64 = xw.size(-1);
  const int64_t rw64 = w1_t.size(0);
  const int64_t ra64 = a1_t.size(0);
  const int64_t rg64 = g1_t.size(0);
  const int64_t rv64 = v1_t.size(0);
  const int64_t m64 = xw.numel() / k64;
  TORCH_CHECK(k64 <= INT_MAX && rw64 <= INT_MAX && ra64 <= INT_MAX &&
                  rg64 <= INT_MAX && rv64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wagv_rank_in_f16 shape too large");
  const int K = static_cast<int>(k64);
  const int Rw = static_cast<int>(rw64);
  const int Ra = static_cast<int>(ra64);
  const int Rg = static_cast<int>(rg64);
  const int Rv = static_cast<int>(rv64);
  const int Rmax = std::max(std::max(Rw, Ra), std::max(Rg, Rv));
  const int M = static_cast<int>(m64);
  TORCH_CHECK(K >= 1024 && Rmax <= 512 && M <= 8,
              "linear_wagv_rank_in_f16 supports only K>=1024,R<=512,M<=8");
  std::vector<int64_t> w_sizes(xw.sizes().begin(), xw.sizes().end());
  std::vector<int64_t> a_sizes = w_sizes;
  std::vector<int64_t> g_sizes = w_sizes;
  std::vector<int64_t> v_sizes = w_sizes;
  w_sizes.back() = rw64;
  a_sizes.back() = ra64;
  g_sizes.back() = rg64;
  v_sizes.back() = rv64;
  auto w1 = at::empty(w_sizes, xw.options());
  auto a1 = at::empty(a_sizes, xw.options());
  auto g1 = at::empty(g_sizes, xw.options());
  auto v1 = at::empty(v_sizes, xw.options());
  if (M == 0 || K == 0 || Rmax == 0) {
    return {w1, a1, g1, v1};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_wagv_rank_in_f16_kernel<256><<<dim3(Rmax, M, 4), 256, 0, stream>>>(
      M, K, Rw, Ra, Rg, Rv, Rmax, xw.data_ptr<dtype>(), xa.data_ptr<dtype>(),
      xg.data_ptr<dtype>(), xv.data_ptr<dtype>(), w1_t.data_ptr<dtype>(),
      a1_t.data_ptr<dtype>(), g1_t.data_ptr<dtype>(), v1_t.data_ptr<dtype>(),
      w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(),
      v1.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w1, a1, g1, v1};
}

std::vector<at::Tensor> linear_wag_rank_out_f16_cuda(
    at::Tensor w1, at::Tensor a1, at::Tensor g1, at::Tensor w2_t,
    at::Tensor a2_t, at::Tensor g2_t) {
  const int64_t kw64 = w1.size(-1);
  const int64_t ka64 = a1.size(-1);
  const int64_t kg64 = g1.size(-1);
  const int64_t c64 = w2_t.size(0);
  const int64_t m64 = w1.numel() / kw64;
  TORCH_CHECK(kw64 <= INT_MAX && ka64 <= INT_MAX && kg64 <= INT_MAX &&
                  c64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wag_rank_out_f16 shape too large");
  const int Kw = static_cast<int>(kw64);
  const int Ka = static_cast<int>(ka64);
  const int Kg = static_cast<int>(kg64);
  const int C = static_cast<int>(c64);
  const int M = static_cast<int>(m64);
  TORCH_CHECK(Kw <= 512 && Ka <= 512 && Kg <= 512 && C >= 1024 && M <= 4,
              "linear_wag_rank_out_f16 supports only small-rank M<=4");
  std::vector<int64_t> out_sizes(w1.sizes().begin(), w1.sizes().end());
  out_sizes.back() = c64;
  auto w = at::empty(out_sizes, w1.options());
  auto a = at::empty(out_sizes, w1.options());
  auto g = at::empty(out_sizes, w1.options());
  if (M == 0 || C == 0 || Kw == 0 || Ka == 0 || Kg == 0) {
    return {w, a, g};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M == 1) {
    linear_wag_rank_out_f16_kernel<128, 4>
        <<<dim3(ceil_div(C, 4), M, 3), 128, 0, stream>>>(
            M, C, Kw, Ka, Kg, w1.data_ptr<dtype>(), a1.data_ptr<dtype>(),
            g1.data_ptr<dtype>(), w2_t.data_ptr<dtype>(),
            a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(), w.data_ptr<dtype>(),
            a.data_ptr<dtype>(), g.data_ptr<dtype>());
  } else {
    linear_wag_rank_out_f16_kernel<128, 4>
        <<<dim3(ceil_div(C, 4), M, 3), 128, 0, stream>>>(
            M, C, Kw, Ka, Kg, w1.data_ptr<dtype>(), a1.data_ptr<dtype>(),
            g1.data_ptr<dtype>(), w2_t.data_ptr<dtype>(),
            a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(), w.data_ptr<dtype>(),
            a.data_ptr<dtype>(), g.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, a, g};
}

std::vector<at::Tensor> linear_wagv_rank_out_f16_cuda(
    at::Tensor w1, at::Tensor a1, at::Tensor g1, at::Tensor v1, at::Tensor w2_t,
    at::Tensor a2_t, at::Tensor g2_t, at::Tensor v2_t, at::Tensor v,
    at::Tensor v_first, at::Tensor v0) {
  const int64_t kw64 = w1.size(-1);
  const int64_t ka64 = a1.size(-1);
  const int64_t kg64 = g1.size(-1);
  const int64_t kv64 = v1.size(-1);
  const int64_t c64 = w2_t.size(0);
  const int64_t m64 = w1.numel() / kw64;
  TORCH_CHECK(kw64 <= INT_MAX && ka64 <= INT_MAX && kg64 <= INT_MAX &&
                  kv64 <= INT_MAX && c64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wagv_rank_out_f16 shape too large");
  const int Kw = static_cast<int>(kw64);
  const int Ka = static_cast<int>(ka64);
  const int Kg = static_cast<int>(kg64);
  const int Kv = static_cast<int>(kv64);
  const int C = static_cast<int>(c64);
  const int M = static_cast<int>(m64);
  TORCH_CHECK(
      Kw <= 512 && Ka <= 512 && Kg <= 512 && Kv <= 512 && C >= 1024 && M <= 4,
      "linear_wagv_rank_out_f16 supports only small-rank M<=4");
  std::vector<int64_t> out_sizes(w1.sizes().begin(), w1.sizes().end());
  out_sizes.back() = c64;
  auto w = at::empty(out_sizes, w1.options());
  auto a = at::empty(out_sizes, w1.options());
  auto g = at::empty(out_sizes, w1.options());
  auto v_out = at::empty(out_sizes, w1.options());
  if (M == 0 || C == 0 || Kw == 0 || Ka == 0 || Kg == 0 || Kv == 0) {
    return {w, a, g, v_out};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M == 1) {
    linear_wagv_rank_out_f16_kernel<128, 4>
        <<<dim3(ceil_div(C, 4), M, 4), 128, 0, stream>>>(
            M, C, Kw, Ka, Kg, Kv, w1.data_ptr<dtype>(), a1.data_ptr<dtype>(),
            g1.data_ptr<dtype>(), v1.data_ptr<dtype>(), w2_t.data_ptr<dtype>(),
            a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(),
            v2_t.data_ptr<dtype>(), v.data_ptr<dtype>(),
            v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(),
            w.data_ptr<dtype>(), a.data_ptr<dtype>(), g.data_ptr<dtype>(),
            v_out.data_ptr<dtype>());
  } else {
    linear_wagv_rank_out_f16_kernel<128, 4>
        <<<dim3(ceil_div(C, 4), M, 4), 128, 0, stream>>>(
            M, C, Kw, Ka, Kg, Kv, w1.data_ptr<dtype>(), a1.data_ptr<dtype>(),
            g1.data_ptr<dtype>(), v1.data_ptr<dtype>(), w2_t.data_ptr<dtype>(),
            a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(),
            v2_t.data_ptr<dtype>(), v.data_ptr<dtype>(),
            v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(),
            w.data_ptr<dtype>(), a.data_ptr<dtype>(), g.data_ptr<dtype>(),
            v_out.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, a, g, v_out};
}

at::Tensor linear_t_vres_f16_cuda(at::Tensor x, at::Tensor weight_t,
                                  at::Tensor v, at::Tensor v_first,
                                  at::Tensor v0) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX,
              "linear_t_vres_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_t_vres_f16 M too large");
  const int M = static_cast<int>(m64);
  auto y = at::empty_like(v);
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(K <= 512 && N >= 1024 && M <= 4,
              "linear_t_vres_f16 currently supports only small-rank rank-out");
  if (M == 1) {
    linear_t_vres_f16_ntile_scalar_kernel<128, 2>
        <<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
            M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(),
            v.data_ptr<dtype>(), v_first.data_ptr<dtype>(),
            v0.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_t_vres_f16_ntile_kernel<128, 4>
        <<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
            M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(),
            v.data_ptr<dtype>(), v_first.data_ptr<dtype>(),
            v0.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
