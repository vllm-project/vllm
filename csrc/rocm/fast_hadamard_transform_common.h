/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#define HIP_ENABLE_WARP_SYNC_BUILTINS

#include <ATen/ATen.h>
#include <ATen/ScalarType.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_warp_functions.h>

#define FULL_MASK 0xffffffff

const unsigned long long AllThreads = ~0;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct uint8 {
  uint4 u;
  uint4 v;
};

template <int BYTES>
struct BytesToType {};

template <>
struct BytesToType<32> {
  using Type = uint8;
  static_assert(sizeof(Type) == 32);
};

template <>
struct BytesToType<16> {
  using Type = uint4;
  static_assert(sizeof(Type) == 16);
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
  static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// https://stackoverflow.com/questions/35311711/whats-the-right-way-to-compute-integral-base-2-logarithms-at-compile-time
constexpr int cilog2(int val) { return val > 0 ? 1 + cilog2(val >> 1) : -1; }

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kLogN, int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread(
    float x[kNChunks][1 << kLogN]) {
  constexpr int N = 1 << kLogN;
#pragma unroll
  for (int i = 0; i < kLogN; ++i) {
    const int stride = 1 << i;
#pragma unroll
    for (int j = 0; j < N / 2; ++j) {
      const int lo = j & (stride - 1);
      const int idx = (j - lo) * 2 + lo;
#pragma unroll
      for (int c = 0; c < kNChunks; ++c) {
        const float a = x[c][idx];
        const float b = x[c][idx + stride];
        x[c][idx] = a + b;
        x[c][idx + stride] = a - b;
      }
    }
  }
}

template <int N, int kLogWarpSize, int kStepStart, int kNChunks, int kNItems>
__device__ __forceinline__ void hadamard_mult_warp(float x[kNChunks][kNItems]) {
  // constexpr int N = 1 << kLogWarpSize;
  int lane_id = threadIdx.x % N;
#pragma unroll
  for (int step = kStepStart; step < kLogWarpSize; ++step) {
    const int lane_mask = 1 << step;
    const float sign = (lane_id & lane_mask) ? -1.f : 1.f;
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
      for (int i = 0; i < kNItems; ++i) {
        int srcLane = lane_id ^ lane_mask;
        float x_val_other = __shfl(
            x[c][i], srcLane);  // source lane = current_lane XOR lane_mask
        // float x_val_other = __shfl_xor_sync(FULL_MASK, x[c][i], lane_mask);
        x[c][i] = sign * x[c][i] + x_val_other;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNChunks, int kNElts, typename input_t>
inline __device__ void load_input(input_t* x, float x_vals[kNChunks][kNElts],
                                  int dim) {
  using vec_t = typename BytesToType<sizeof(input_t) * kNElts>::Type;
  input_t x_vals_load[kNChunks][kNElts] = {0};
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    if ((c * blockDim.x + threadIdx.x) * kNElts < dim) {
      reinterpret_cast<vec_t*>(x_vals_load)[c] =
          reinterpret_cast<const vec_t*>(x)[c * blockDim.x + threadIdx.x];
    }
  }
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      x_vals[c][i] = float(x_vals_load[c][i]);
    }
  }
}

template <int kNChunks, int kNElts, typename output_t>
inline __device__ void store_output(output_t* out,
                                    float out_vals[kNChunks][kNElts], int dim,
                                    float scale = 1.f) {
  using vec_t = typename BytesToType<sizeof(output_t) * kNElts>::Type;
  output_t out_vals_store[kNChunks][kNElts];
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
#pragma unroll
    for (int i = 0; i < kNElts; ++i) {
      out_vals_store[c][i] = out_vals[c][i] * scale;
    }
  }
#pragma unroll
  for (int c = 0; c < kNChunks; ++c) {
    if ((c * blockDim.x + threadIdx.x) * kNElts < dim) {
      reinterpret_cast<vec_t*>(out)[c * blockDim.x + threadIdx.x] =
          reinterpret_cast<const vec_t*>(out_vals_store)[c];
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////

// Pre=true means the exchange before the hadamard_mult_warp, Pre=false means
// after.
template <int kNChunks, int kChunksPerExchange, int kNElts, int kWarpSize,
          int kNWarps, bool Pre, typename vec_t>
inline __device__ void exchange_smem_pre(float x_vals[kNChunks][kNElts],
                                         vec_t* smem) {
  constexpr int kNThreads = kWarpSize * kNWarps;
  constexpr int kNExchangePerVec = kNElts / (sizeof(vec_t) / sizeof(float));
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int row_t = threadIdx.x % kNWarps;
  const int col_t = threadIdx.x / kNWarps;
// We use the XOR swizzle trick (new_col = col ^ row) to avoid / reduce smem
// bank conflicts.
#pragma unroll
  for (int c0 = 0; c0 < kNChunks / kChunksPerExchange; ++c0) {
    __syncthreads();
#pragma unroll
    for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
#pragma unroll
      for (int r = 0; r < kNExchangePerVec; ++r) {
        smem[(c1 * kNExchangePerVec + r) * kNThreads +
             (Pre ? warp_id * kWarpSize + lane_id ^ warp_id
                  : row_t * kWarpSize + col_t ^ row_t)] =
            reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r];
      }
    }
    __syncthreads();
#pragma unroll
    for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
#pragma unroll
      for (int r = 0; r < kNExchangePerVec; ++r) {
        reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r] =
            smem[(c1 * kNExchangePerVec + r) * kNThreads +
                 (Pre ? row_t * kWarpSize + col_t ^ row_t
                      : warp_id * kWarpSize + lane_id ^ warp_id)];
      }
    }
  }
}