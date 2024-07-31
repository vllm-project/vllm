/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/reduce_kernel_utils.cuh
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cuda_compat.h"

namespace vllm {

namespace detail {

template <typename T>
__inline__ __device__ T _max(T a, T b) {
  return max(a, b);
}

template <typename T>
__inline__ __device__ T _min(T a, T b) {
  return min(a, b);
}

template <typename T>
__inline__ __device__ T _sum(T a, T b) {
  return a + b;
}

}  // namespace detail

template <typename T>
__device__ __host__ static constexpr T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
using ReduceFnType = T (*)(T, T);

// Helper function to return the next largest power of 2
static constexpr int _nextPow2(unsigned int num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename T, int numLanes = WARP_SIZE>
__inline__ __device__ T warpReduce(T val, ReduceFnType<T> fn) {
  static_assert(numLanes > 0 && (numLanes & (numLanes - 1)) == 0,
                "numLanes is not a positive power of 2!");
  static_assert(numLanes <= WARP_SIZE);
#pragma unroll
  for (int mask = numLanes >> 1; mask > 0; mask >>= 1) {
    auto const other_idx = threadIdx.x ^ mask;
    auto const other_val = VLLM_SHFL_XOR_SYNC(val, mask);
    val = other_idx < blockDim.x ? fn(val, other_val) : val;
  }

  return val;
}

// Make sure you call __syncthreads() between different blockReduce calls, as
// they are allowed to use the same shared memory.
template <typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduce(T val, ReduceFnType<T> fn, T init = T{}) {
  static_assert(maxBlockSize <= 1024);
  if constexpr (maxBlockSize > WARP_SIZE) {
    val = warpReduce<T>(val, fn);
    // Calculates max number of lanes that need to participate in the last
    // warpReduce
    constexpr int maxActiveLanes =
        ceil_div<unsigned int>(maxBlockSize, WARP_SIZE);

    // shared memory can be reused between function calls, make static
    // explicitly.
    static __shared__ T shared[maxActiveLanes];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0) shared[wid] = val;

    __syncthreads();

    auto const num_sh_lanes = ceil_div<unsigned int>(blockDim.x, WARP_SIZE);
    val = threadIdx.x < num_sh_lanes ? shared[lane] : init;
    if (wid == 0) {
      val = warpReduce<T>(val, fn);
    }
  } else {
    // A single warpReduce is equal to blockReduce
    val = warpReduce<T, _nextPow2(maxBlockSize)>(val, fn);
  }
  return val;
}

template <typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduceMax(T val) {
  auto const min_val = std::numeric_limits<T>::lowest();
  return blockReduce<T, maxBlockSize>(val, detail::_max<T>, min_val);
}

template <typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduceMin(T val) {
  auto const max_val = std::numeric_limits<T>::max();
  return blockReduce<T, maxBlockSize>(val, detail::_min<T>, max_val);
}

template <typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduceSum(T val) {
  return blockReduce<T, maxBlockSize>(val, detail::_sum<T>);
}

}  // namespace vllm
