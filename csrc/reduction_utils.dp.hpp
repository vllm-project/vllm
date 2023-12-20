#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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

namespace vllm {

template<typename T>
__inline__ T warpReduceSum(T val,const sycl::nd_item<3> &item_ct1) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += dpct::experimental::permute_sub_group_by_xor(0xffffffff, item_ct1.get_sub_group(), val, mask, 32); 
    //val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ T blockReduceSum(T val, const sycl::nd_item<3> &item_ct1, T *shared) {

  int lane = item_ct1.get_local_id(2) & 0x1f;
  int wid = item_ct1.get_local_id(2) >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  
  item_ct1.barrier();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (item_ct1.get_local_id(2) < (item_ct1.get_local_range(2) / 32.f))
            ? shared[lane]
            : (T)(0.0f);
  val = warpReduceSum<T>(val, item_ct1);
  return val;
}

} // namespace vllm
