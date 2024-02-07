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

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace vllm {
namespace autoquant {

void reformat_s4_k8_m(uint32_t* dst, const uint32_t* src, int m, int k, cudaStream_t st = {});

void reformat_s4_k_m8(uint32_t* dst, const uint32_t* src, int m, int k, cudaStream_t st = {});

void convert_s4_k_m8(uint32_t*       A_dst,
                     half2*          Q_dst,
                     half*           workspace,
                     const uint32_t* A_src,
                     const half*     scales,
                     const uint32_t* qzeros,
                     int             m,
                     int             k,
                     int             group_size,
                     cudaStream_t    st = {});

void convert_s4_k_m8(uint32_t*            A_dst,
                     __nv_bfloat162*      Q_dst,
                     __nv_bfloat16*       workspace,
                     const uint32_t*      A_src,
                     const __nv_bfloat16* scales,
                     const uint32_t*      qzeros,
                     int                  m,
                     int                  k,
                     int                  group_size,
                     cudaStream_t         st = {});

void transpose_qk_s4_k_m8_hf(uint32_t* dst, const uint32_t* src, int m, int k, int size_per_head, cudaStream_t st = {});

void fuse_w1_w3_s4_k_m8(uint32_t* dst, const uint32_t* src, int m, int k, cudaStream_t st = {});

void dequantize_s4(uint4* dst, const uint32_t* src, size_t count, cudaStream_t st = {});

}  // namespace autoquant
}  // namespace vllm
