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
__inline__ __device__ T _sum(T a, T b) {
  return a + b;
}

}  // namespace detail

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
  for (int mask = numLanes >> 1; mask > 0; mask >>= 1)
    val = fn(val, VLLM_SHFL_XOR_SYNC(val, mask));

  return val;
}

template <typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduce(T val, ReduceFnType<T> fn) {
  static_assert(maxBlockSize <= 1024);
  if constexpr (maxBlockSize > WARP_SIZE) {
    val = warpReduce<T>(val, fn);
    // Calculates max number of lanes that need to participate in the last
    // warpReduce
    constexpr int maxActiveLanes = (maxBlockSize + WARP_SIZE - 1) / WARP_SIZE;
    static __shared__ T shared[maxActiveLanes];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    if (lane == 0) shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / float(WARP_SIZE)) ? shared[lane]
                                                        : (T)(0.0f);
    val = warpReduce<T, _nextPow2(maxActiveLanes)>(val, fn);
  } else {
    // A single warpReduce is equal to blockReduce
    val = warpReduce<T, _nextPow2(maxBlockSize)>(val, fn);
  }
  return val;
}

template <typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduceMax(T val) {
  return blockReduce<T, maxBlockSize>(val, detail::_max<T>);
}

template <typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduceSum(T val) {
  return blockReduce<T, maxBlockSize>(val, detail::_sum<T>);
}

}  // namespace vllm
