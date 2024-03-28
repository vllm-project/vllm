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

namespace vllm {

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = WARP_SIZE/2; mask > 0; mask >>= 1)
    val += VLLM_SHFL_XOR_SYNC(val, mask);
  return val;
}

__inline__ __device__ constexpr int _calculateLaneMask(int warp_size) {
  return warp_size - 1;
}

__inline__ __device__ constexpr int _calculateWidShift(int warp_size) {
  return 5 + (warp_size >> 6);
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[WARP_SIZE];
  constexpr auto LANE_MASK = _calculateLaneMask(WARP_SIZE);
  constexpr auto WID_SHIFT = _calculateWidShift(WARP_SIZE);
  int lane = threadIdx.x & LANE_MASK;
  int wid = threadIdx.x >> WID_SHIFT;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / (WARP_SIZE * 1.0f))) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

} // namespace vllm
