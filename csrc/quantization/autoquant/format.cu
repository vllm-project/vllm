/*
 * Adapted from https://github.com/InternLM/lmdeploy
 * Copyright (c) OpenMMLab. All rights reserved.
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

#include <iostream>
#include "common.h"

namespace vllm {
namespace autoquant {

__device__ void atomic_assign_u4(uint32_t* address, uint32_t index, uint32_t value)
{
    uint32_t old = *address;
    uint32_t assumed;
    do {
        assumed      = old;
        uint32_t tmp = (assumed & ~(0xfu << (index * 4u))) | (value << (index * 4u));
        old          = atomicCAS(address, assumed, tmp);
    } while (assumed != old);
}

__device__ uint32_t read_u4(const uint32_t* address, uint32_t index)
{
    return (*address >> (index * 4u)) & 0xfu;
}

template<int... Ds>
__global__ void permute_u4(uint* dst, const uint* src, Array<int, sizeof...(Ds)> dims)
{
    constexpr int N = sizeof...(Ds);

    size_t count = 1;
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        count *= dims[i];
    }

    constexpr int order[] = {Ds...};

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {

        int indices[N]{};

        PRAGMA_UNROLL
        for (int j = N - 1, ii = i; j >= 0; --j) {
            indices[j] = ii % dims[j];
            ii /= dims[j];
        }

        auto data = read_u4(src + i / 8, i % 8);

        int index = 0;

        PRAGMA_UNROLL
        for (int j = N - 1, stride = 1; j >= 0; --j) {
            index += indices[order[j]] * stride;
            stride *= dims[order[j]];
        }

        atomic_assign_u4(dst + index / 8, index % 8, data);
    }
}

void reformat_s4_k8_m(uint32_t* dst, const uint32_t* src, int m, int k, cudaStream_t st)
{
    // permutation for [k/8, m] layout
    Array<int, 10> shape{k / 32, 2, 2, m / 32, 2, 2, 8, 2, 2, 2};
    //        |warp|  lane  | 2x2 |  a0-7  |
    permute_u4<0, 3, 6, 8, 9, 1, 4, 7, 2, 5><<<512, 512, 0, st>>>(dst, src, shape);
}

void reformat_s4_k_m8(uint32_t* dst, const uint32_t* src, int m, int k, cudaStream_t st)
{
    // permutation for [k, m/8] layout
    Array<int, 10> shape{k / 32, 2, 2, 4, 2, m / 32, 2, 2, 2, 4};
    //        |warp|  lane  | 2x2 |  a0-7  |
    permute_u4<0, 5, 9, 8, 3, 1, 6, 4, 2, 7><<<512, 512, 0, st>>>(dst, src, shape);
}

__global__ void dequantize_s4_offset_64(uint4* dst, const uint32_t* src, size_t count)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        dst[i] = dequantize_s4_to_fp16x2_v2(src[i]);
    }
}

__global__ void dequantize_s4_offset_64_bf16(uint4* dst, const uint32_t* src, size_t count)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        dst[i] = dequantize_s4_to_bf16x2_v2(src[i]);
    }
}

__global__ void merge_Q(half2* Q, const half* scales, const half* zeros, int count)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        Q[i] = __halves2half2(zeros[i], scales[i]);
    }
}

__global__ void merge_Q(__nv_bfloat162* Q, const __nv_bfloat16* scales, const __nv_bfloat16* zeros, int count)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        Q[i] = 	halves2bfloat162(zeros[i], scales[i]);
    }
}

void convert_s4_k_m8(uint32_t*       A_dst,
                     half2*          Q_dst,
                     half*           workspace,
                     const uint32_t* A_src,
                     const half*     scales,
                     const uint32_t* qzeros,
                     int             m,
                     int             k,
                     int             group_size,
                     cudaStream_t    st)
{
    dequantize_s4_offset_64<<<256, 256, 0, st>>>((uint4*)workspace, qzeros, k / group_size * m / 8);
    merge_Q<<<256, 256, 0, st>>>(Q_dst, scales, workspace, k / group_size * m);
    reformat_s4_k_m8(A_dst, A_src, m, k, st);
}
void convert_s4_k_m8(uint32_t*            A_dst,
                     __nv_bfloat162*      Q_dst,
                     __nv_bfloat16*       workspace,
                     const uint32_t*      A_src,
                     const __nv_bfloat16* scales,
                     const uint32_t*      qzeros,
                     int                  m,
                     int                  k,
                     int                  group_size,
                     cudaStream_t         st)
{
    dequantize_s4_offset_64_bf16<<<256, 256, 0, st>>>((uint4*)workspace, qzeros, k / group_size * m / 8);
    merge_Q<<<256, 256, 0, st>>>(Q_dst, scales, workspace, k / group_size * m);
    reformat_s4_k_m8(A_dst, A_src, m, k, st);
}

void transpose_qk_s4_k_m8_hf(uint32_t* dst, const uint32_t* src, int m, int k, int size_per_head, cudaStream_t st)
{
    Array<int, 7> shape{k, m / size_per_head, 2, size_per_head / 2 / 8, 2, 2, 2};
    //      dequant   transpose    quant
    // 0123456 -> 0123564 -> 0135642 -> 0135264
    permute_u4<0, 1, 3, 5, 2, 6, 4><<<512, 512, 0, st>>>(dst, src, shape);
}

// [2, k, m/8] -> [k, m/8, 2]
void fuse_w1_w3_s4_k_m8(uint32_t* dst, const uint32_t* src, int m, int k, cudaStream_t st)
{
    Array<int, 6> shape{2, k, m / 8, 2, 2, 2};
    //     dequant   transpose   quant
    // 012345 -> 012453 -> 124530 -> 124053
    permute_u4<1, 2, 4, 0, 5, 3><<<512, 512, 0, st>>>(dst, src, shape);
}

__global__ void dequantize_s4_kernel(uint4* dst, const uint* src, size_t count)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < count; i += blockDim.x * gridDim.x) {
        dst[i] = dequantize_s4_to_fp16x2(src[i]);
    }
}

void dequantize_s4(uint4* dst, const uint32_t* src, size_t count, cudaStream_t st)
{
    dequantize_s4_kernel<<<512, 512>>>(dst, src, count);
}

}  // namespace autoquant
}  // namespace vllm
