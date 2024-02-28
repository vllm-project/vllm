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
  #pragma unroll
  for (int mask = numLanes >> 1; mask > 0; mask >>= 1)
    val += VLLM_SHFL_XOR_SYNC(val, mask);
  return val;
}

/* Calculate the sum of all elements in a block */
template<typename T, int maxBlockSize = 1024>
__inline__ __device__ T blockReduceSum(T val) {
  val = warpReduceSum<T>(val);
  // If the block fits into a single warp, we are already done
  if constexpr (maxBlockSize > warpSize) {
    constexpr int maxActiveLanes = (maxBlockSize + warpSize - 1) / warpSize;
    static __shared__ T shared[maxActiveLanes];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0)
      shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < (blockDim.x / (float) warpSize)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T, maxActiveLanes>(val);
  }
  return val;
}

} // namespace vllm
#ifndef USE_ROCM
#pragma pop_macro("warpSize")
#endif
