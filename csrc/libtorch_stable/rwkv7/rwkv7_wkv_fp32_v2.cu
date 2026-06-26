// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from BlinkDL/Albatross faster3a_2605/cuda at commit
// 5e941fb1eeb7f735a562fb5bbb30fad19adc825b. Source:
// https://github.com/BlinkDL/Albatross/tree/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/faster3a_2605/cuda
// Upstream license: Apache-2.0
// (https://github.com/BlinkDL/Albatross/blob/5e941fb1eeb7f735a562fb5bbb30fad19adc825b/LICENSE).

#include <assert.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace {

constexpr int N = 64;
constexpr int WARP_THREADS = 32;
constexpr int BLOCK_THREADS = 32;
constexpr float W_SCALE_LOG2_E = -0.8750387749145276f;
constexpr float NLOG2_E = -1.4426950408889634f;

#ifdef _IO_FP16_
using io_t = __half;
__device__ __forceinline__ float io_to_float(io_t x) { return __half2float(x); }
__device__ __forceinline__ io_t float_to_io(float x) {
  return __float2half_rn(x);
}
#else
using io_t = float;
__device__ __forceinline__ float io_to_float(float x) { return x; }
__device__ __forceinline__ float float_to_io(float x) { return x; }
#endif

__device__ __forceinline__ float w_eff(float w) {
  return exp2f(W_SCALE_LOG2_E / (1.0f + exp2f(NLOG2_E * w)));
}

__device__ __forceinline__ float load_io(const io_t* ptr, int64_t idx) {
  return io_to_float(__ldg(ptr + idx));
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__device__ __forceinline__ float warp_sum_broadcast(float x) {
  return __shfl_sync(0xffffffffu, warp_sum(x), 0);
}

__device__ __forceinline__ float block_sum_broadcast(float x) {
  __shared__ float partial[BLOCK_THREADS / WARP_THREADS];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  x = warp_sum(x);
  if (lane == 0) {
    partial[warp] = x;
  }
  __syncthreads();
  x = (threadIdx.x < (BLOCK_THREADS / WARP_THREADS)) ? partial[lane] : 0.0f;
  if (warp == 0) {
    x = warp_sum(x);
  }
  if (threadIdx.x == 0) {
    partial[0] = x;
  }
  __syncthreads();
  return partial[0];
}

template <int HeadSize>
__launch_bounds__(HeadSize, 2) __global__
    void wkv_fp32_v2_kernel(int T, int C, int H, float* __restrict__ state_ptr,
                            const io_t* __restrict__ r_ptr,
                            const io_t* __restrict__ w_ptr,
                            const io_t* __restrict__ k_ptr,
                            const io_t* __restrict__ v_ptr,
                            const io_t* __restrict__ a_ptr,
                            const io_t* __restrict__ b_ptr,
                            io_t* __restrict__ y_ptr) {
  const int bh = blockIdx.x;
  const int b_id = bh / H;
  const int h = bh - b_id * H;
  const int i = threadIdx.x;
  const int c_base = h * HeadSize;
  const int64_t bt_base = static_cast<int64_t>(b_id) * T * C + c_base;
  float* state_base =
      state_ptr + (static_cast<int64_t>(b_id) * H * HeadSize * HeadSize +
                   h * HeadSize * HeadSize + i * HeadSize);

  float state[HeadSize];
#pragma unroll
  for (int j = 0; j < HeadSize; ++j) {
    state[j] = state_base[j];
  }

  __shared__ float r[HeadSize];
  __shared__ float w[HeadSize];
  __shared__ float k[HeadSize];
  __shared__ float a[HeadSize];
  __shared__ float b[HeadSize];

  for (int t = 0; t < T; ++t) {
    const int64_t idx = bt_base + static_cast<int64_t>(t) * C + i;
    __syncthreads();
    r[i] = load_io(r_ptr, idx);
    w[i] = w_eff(load_io(w_ptr, idx));
    k[i] = load_io(k_ptr, idx);
    a[i] = load_io(a_ptr, idx);
    b[i] = load_io(b_ptr, idx);
    __syncthreads();

    float sa = 0.0f;
#pragma unroll
    for (int j = 0; j < HeadSize; ++j) {
      sa += state[j] * a[j];
    }

    const float vi = load_io(v_ptr, idx);
    float y = 0.0f;
#pragma unroll
    for (int j = 0; j < HeadSize; ++j) {
      float s = state[j];
      s = s * w[j] + sa * b[j] + k[j] * vi;
      y += s * r[j];
      state[j] = s;
    }
    y_ptr[idx] = float_to_io(y);
  }

#pragma unroll
  for (int j = 0; j < HeadSize; ++j) {
    state_base[j] = state[j];
  }
}

__global__ __launch_bounds__(
    WARP_THREADS,
    4) void wkv_fp32_v2_small_warp_kernel(int T, int C, int H,
                                          float* __restrict__ state_ptr,
                                          const io_t* __restrict__ r_ptr,
                                          const io_t* __restrict__ w_ptr,
                                          const io_t* __restrict__ k_ptr,
                                          const io_t* __restrict__ v_ptr,
                                          const io_t* __restrict__ a_ptr,
                                          const io_t* __restrict__ b_ptr,
                                          io_t* __restrict__ y_ptr) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int b_id = blockIdx.z;
  const int lane = threadIdx.x;
  const int c_base = h * N;
  const int state_base = ((b_id * H + h) * N + row) * N;

  for (int t = 0; t < T; ++t) {
    const int token = (b_id * T + t) * C + c_base;
    float sa = 0.0f;
    for (int j = lane; j < N; j += WARP_THREADS) {
      sa += state_ptr[state_base + j] * load_io(a_ptr, token + j);
    }
    sa = warp_sum_broadcast(sa);

    float yy = 0.0f;
    const float vv = load_io(v_ptr, token + row);
    for (int j = lane; j < N; j += WARP_THREADS) {
      const int idx = token + j;
      const float s = state_ptr[state_base + j] * w_eff(load_io(w_ptr, idx)) +
                      vv * load_io(k_ptr, idx) + sa * load_io(b_ptr, idx);
      state_ptr[state_base + j] = s;
      yy += s * load_io(r_ptr, idx);
    }
    yy = warp_sum(yy);
    if (lane == 0) {
      y_ptr[token + row] = float_to_io(yy);
    }
  }
}

__global__ __launch_bounds__(
    BLOCK_THREADS,
    4) void wkv_fp32_v2_short_block_kernel(int T, int C, int H,
                                           float* __restrict__ state_ptr,
                                           const io_t* __restrict__ r_ptr,
                                           const io_t* __restrict__ w_ptr,
                                           const io_t* __restrict__ k_ptr,
                                           const io_t* __restrict__ v_ptr,
                                           const io_t* __restrict__ a_ptr,
                                           const io_t* __restrict__ b_ptr,
                                           io_t* __restrict__ y_ptr) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int b_id = blockIdx.z;
  const int tid = threadIdx.x;
  const int c_base = h * N;
  const int state_base = ((b_id * H + h) * N + row) * N;

  for (int t = 0; t < T; ++t) {
    const int token = (b_id * T + t) * C + c_base;
    float sa = 0.0f;
    for (int j = tid; j < N; j += BLOCK_THREADS) {
      sa += state_ptr[state_base + j] * load_io(a_ptr, token + j);
    }
    sa = block_sum_broadcast(sa);

    float yy = 0.0f;
    const float vv = load_io(v_ptr, token + row);
    for (int j = tid; j < N; j += BLOCK_THREADS) {
      const int idx = token + j;
      const float s = state_ptr[state_base + j] * w_eff(load_io(w_ptr, idx)) +
                      vv * load_io(k_ptr, idx) + sa * load_io(b_ptr, idx);
      state_ptr[state_base + j] = s;
      yy += s * load_io(r_ptr, idx);
    }
    yy = block_sum_broadcast(yy);
    if (tid == 0) {
      y_ptr[token + row] = float_to_io(yy);
    }
    __syncthreads();
  }
}

bool use_small_auto(int B, int T) {
#ifdef _IO_FP16_
  return (T == 1 && B <= 96) || (T == 2 && B <= 21) || (T == 3 && B <= 3) ||
         (T == 4 && (B == 1 || B == 3)) || (B == 1 && T >= 5 && T <= 11);
#else
  return (T == 1) || (T == 2 && B <= 96) || (T == 3 && (B <= 4 || B == 6)) ||
         (T == 4 && (B == 1 || B == 3)) || (B == 1 && T >= 5 && T <= 9);
#endif
}

}  // namespace

void wkv_fp32_v2_cuda(int B, int T, int C, int H, int mode, at::Tensor state,
                      at::Tensor r, at::Tensor w, at::Tensor k, at::Tensor v,
                      at::Tensor a, at::Tensor b, at::Tensor y) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool use_small = (mode == 2) || (mode == 0 && use_small_auto(B, T));
  if (mode == 3) {
    wkv_fp32_v2_short_block_kernel<<<dim3(N, H, B), dim3(BLOCK_THREADS), 0,
                                     stream>>>(
        T, C, H, state.data_ptr<float>(), reinterpret_cast<io_t*>(r.data_ptr()),
        reinterpret_cast<io_t*>(w.data_ptr()),
        reinterpret_cast<io_t*>(k.data_ptr()),
        reinterpret_cast<io_t*>(v.data_ptr()),
        reinterpret_cast<io_t*>(a.data_ptr()),
        reinterpret_cast<io_t*>(b.data_ptr()),
        reinterpret_cast<io_t*>(y.data_ptr()));
  } else if (use_small) {
    wkv_fp32_v2_small_warp_kernel<<<dim3(N, H, B), dim3(WARP_THREADS), 0,
                                    stream>>>(
        T, C, H, state.data_ptr<float>(), reinterpret_cast<io_t*>(r.data_ptr()),
        reinterpret_cast<io_t*>(w.data_ptr()),
        reinterpret_cast<io_t*>(k.data_ptr()),
        reinterpret_cast<io_t*>(v.data_ptr()),
        reinterpret_cast<io_t*>(a.data_ptr()),
        reinterpret_cast<io_t*>(b.data_ptr()),
        reinterpret_cast<io_t*>(y.data_ptr()));
  } else {
    wkv_fp32_v2_kernel<N><<<dim3(B * H), dim3(N), 0, stream>>>(
        T, C, H, state.data_ptr<float>(), reinterpret_cast<io_t*>(r.data_ptr()),
        reinterpret_cast<io_t*>(w.data_ptr()),
        reinterpret_cast<io_t*>(k.data_ptr()),
        reinterpret_cast<io_t*>(v.data_ptr()),
        reinterpret_cast<io_t*>(a.data_ptr()),
        reinterpret_cast<io_t*>(b.data_ptr()),
        reinterpret_cast<io_t*>(y.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
