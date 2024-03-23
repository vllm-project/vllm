/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/reduce_kernel_utils.cuh
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

/* On ROCm, warpSize is a reserved keyword implemented as a macro.
   On CUDA, warpSize is a reserved keyword but its value is read
   from a special memory region at run time.
   Thus, we have to define warpSize at compile time for CUDA.
 */
#ifndef USE_ROCM
// On CUDA, limit our macro's scope as much as possible
#pragma push_macro("warpSize")
#undef warpSize
#define warpSize 32
#endif

namespace vllm {
template<typename T, int numLanes = warpSize>
__inline__ __device__ T warpReduceSum(T val) {
  static_assert(numLanes > 0 && (numLanes & (numLanes - 1)) == 0,
                "numLanes is not a positive power of 2!");
  static_assert(numLanes <= warpSize);
  #pragma unroll
  for (int mask = numLanes >> 1; mask > 0; mask >>= 1)
    val += VLLM_SHFL_XOR_SYNC(val, mask);
  return val;
}

// Helper function to return the next largest power of 2
constexpr int _nextPow2(int num) {
  if (num <= 1) return num;
  return 1 << (8 * sizeof(num) - __builtin_clz(num - 1));
}

/* Calculate the sum of all elements in a block */
template<typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduceSum(T val) {
  static_assert(maxBlockSize <= 1024);
  if constexpr (maxBlockSize > warpSize) {
    val = warpReduceSum<T>(val);
    // Calculates max number of lanes that need to participate in the last warpReduce
    constexpr int maxActiveLanes = (maxBlockSize + warpSize - 1) / warpSize;
    static __shared__ T shared[maxActiveLanes];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0)
      shared[wid] = val;

    __syncthreads();

    // Only (a subset of) the first warp needs to participate in the last warpReduce
    if (threadIdx.x < (blockDim.x / (float) warpSize)) {
      val = shared[lane];
      val = warpReduceSum<T, _nextPow2(maxActiveLanes)>(val);
    }
  } else {
    // A single warpReduce is equal to blockReduce
    val = warpReduceSum<T, _nextPow2(maxBlockSize)>(val);
  }
  return val;
}

} // namespace vllm
#ifndef USE_ROCM
#pragma pop_macro("warpSize")
#endif
