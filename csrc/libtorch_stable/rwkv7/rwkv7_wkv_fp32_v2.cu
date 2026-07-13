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

template <int HeadSize>
__launch_bounds__(HeadSize, 2) __global__ void wkv_fp32_v2_kernel(
    int T, int C, int H, float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr, const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ k_ptr, const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr, const io_t* __restrict__ b_ptr,
    io_t* __restrict__ y_ptr, const int* __restrict__ slot_indices) {
  const int bh = blockIdx.x;
  const int b_id = bh / H;
  const int h = bh - b_id * H;
  const int i = threadIdx.x;
  const int state_b = slot_indices == nullptr ? b_id : slot_indices[b_id];
  const int c_base = h * HeadSize;
  const int64_t bt_base = static_cast<int64_t>(b_id) * T * C + c_base;
  float* state_base =
      state_ptr + (static_cast<int64_t>(state_b) * H * HeadSize * HeadSize +
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

__global__
__launch_bounds__(WARP_THREADS, 4) void wkv_fp32_v2_small_warp_kernel(
    int T, int C, int H, float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr, const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ k_ptr, const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr, const io_t* __restrict__ b_ptr,
    io_t* __restrict__ y_ptr, const int* __restrict__ slot_indices) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int b_id = blockIdx.z;
  const int lane = threadIdx.x;
  const int state_b = slot_indices == nullptr ? b_id : slot_indices[b_id];
  const int c_base = h * N;
  const int state_base = ((state_b * H + h) * N + row) * N;

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

template <int HeadSize>
__launch_bounds__(HeadSize, 2) __global__ void wkv_fp32_v2_varlen_kernel(
    int C, int H, const int* __restrict__ query_start_loc,
    const int* __restrict__ slot_indices, float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr, const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ k_ptr, const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr, const io_t* __restrict__ b_ptr,
    io_t* __restrict__ y_ptr) {
  const int bh = blockIdx.x;
  const int b_id = bh / H;
  const int h = bh - b_id * H;
  const int i = threadIdx.x;
  const int state_b = slot_indices[b_id];
  const int c_base = h * HeadSize;
  const int token_base = query_start_loc[b_id];
  const int my_t = query_start_loc[b_id + 1] - token_base;
  float* state_base =
      state_ptr + (static_cast<int64_t>(state_b) * H * HeadSize * HeadSize +
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

  for (int t = 0; t < my_t; ++t) {
    const int64_t idx = static_cast<int64_t>(token_base + t) * C + c_base + i;
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

__global__
__launch_bounds__(WARP_THREADS, 4) void wkv_fp32_v2_small_warp_varlen_kernel(
    int C, int H, const int* __restrict__ query_start_loc,
    const int* __restrict__ slot_indices, float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr, const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ k_ptr, const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr, const io_t* __restrict__ b_ptr,
    io_t* __restrict__ y_ptr) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int b_id = blockIdx.z;
  const int lane = threadIdx.x;
  const int state_b = slot_indices[b_id];
  const int c_base = h * N;
  const int state_base = ((state_b * H + h) * N + row) * N;
  const int token_base = query_start_loc[b_id];
  const int my_t = query_start_loc[b_id + 1] - token_base;

  for (int t = 0; t < my_t; ++t) {
    const int token = (token_base + t) * C + c_base;
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

void wkv_fp32_v2_cuda(int B, int T, int C, int H, at::Tensor state,
                      at::Tensor r, at::Tensor w, at::Tensor k, at::Tensor v,
                      at::Tensor a, at::Tensor b, at::Tensor y,
                      at::Tensor slot_indices) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int* slot_ptr = slot_indices.defined() && slot_indices.numel() > 0
                            ? slot_indices.data_ptr<int>()
                            : nullptr;
  if (use_small_auto(B, T)) {
    wkv_fp32_v2_small_warp_kernel<<<dim3(N, H, B), dim3(WARP_THREADS), 0,
                                    stream>>>(
        T, C, H, state.data_ptr<float>(), reinterpret_cast<io_t*>(r.data_ptr()),
        reinterpret_cast<io_t*>(w.data_ptr()),
        reinterpret_cast<io_t*>(k.data_ptr()),
        reinterpret_cast<io_t*>(v.data_ptr()),
        reinterpret_cast<io_t*>(a.data_ptr()),
        reinterpret_cast<io_t*>(b.data_ptr()),
        reinterpret_cast<io_t*>(y.data_ptr()), slot_ptr);
  } else {
    wkv_fp32_v2_kernel<N><<<dim3(B * H), dim3(N), 0, stream>>>(
        T, C, H, state.data_ptr<float>(), reinterpret_cast<io_t*>(r.data_ptr()),
        reinterpret_cast<io_t*>(w.data_ptr()),
        reinterpret_cast<io_t*>(k.data_ptr()),
        reinterpret_cast<io_t*>(v.data_ptr()),
        reinterpret_cast<io_t*>(a.data_ptr()),
        reinterpret_cast<io_t*>(b.data_ptr()),
        reinterpret_cast<io_t*>(y.data_ptr()), slot_ptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void wkv_fp32_v2_cuda_varlen(int B, int max_t, int C, int H,
                             at::Tensor query_start_loc,
                             at::Tensor slot_indices, at::Tensor state,
                             at::Tensor r, at::Tensor w, at::Tensor k,
                             at::Tensor v, at::Tensor a, at::Tensor b,
                             at::Tensor y) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int* query_start_loc_ptr = query_start_loc.data_ptr<int>();
  const int* slot_ptr = slot_indices.data_ptr<int>();
  if (use_small_auto(B, max_t)) {
    wkv_fp32_v2_small_warp_varlen_kernel<<<dim3(N, H, B), dim3(WARP_THREADS), 0,
                                           stream>>>(
        C, H, query_start_loc_ptr, slot_ptr, state.data_ptr<float>(),
        reinterpret_cast<io_t*>(r.data_ptr()),
        reinterpret_cast<io_t*>(w.data_ptr()),
        reinterpret_cast<io_t*>(k.data_ptr()),
        reinterpret_cast<io_t*>(v.data_ptr()),
        reinterpret_cast<io_t*>(a.data_ptr()),
        reinterpret_cast<io_t*>(b.data_ptr()),
        reinterpret_cast<io_t*>(y.data_ptr()));
  } else {
    wkv_fp32_v2_varlen_kernel<N><<<dim3(B * H), dim3(N), 0, stream>>>(
        C, H, query_start_loc_ptr, slot_ptr, state.data_ptr<float>(),
        reinterpret_cast<io_t*>(r.data_ptr()),
        reinterpret_cast<io_t*>(w.data_ptr()),
        reinterpret_cast<io_t*>(k.data_ptr()),
        reinterpret_cast<io_t*>(v.data_ptr()),
        reinterpret_cast<io_t*>(a.data_ptr()),
        reinterpret_cast<io_t*>(b.data_ptr()),
        reinterpret_cast<io_t*>(y.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
