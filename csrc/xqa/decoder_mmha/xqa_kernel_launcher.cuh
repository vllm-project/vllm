/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <torch/all.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "decoder_xqa_impl_common.h"


bool const isPerfsim = []() {
    auto const v = std::getenv("XQA_IS_PERFSIM");
    if (!v) {
        return false;
    }
    return bool(std::stoi(v));
}();

template <typename T>
class ManagedMemBuf
{
public:
    ManagedMemBuf(size_t nbElems): mSize {nbElems} {
        if (nbElems != 0) {
            void* p;
            checkCuda(cudaMallocManaged(&p, sizeof(T) * nbElems));
            mData.reset(reinterpret_cast<T*>(p));
        }
    }
    T* get() const {return mData.get();}
    size_t size() const {return mSize;}
    void prefetch(int dstDevice, cudaStream_t stream = nullptr) const {
        if (!isPerfsim) {
            checkCuda(cudaMemPrefetchAsync(get(), sizeof(T) * size(), dstDevice, stream));
        }
    }
    T& operator[](size_t i) const {
        return mData[i];
    };
private:
    struct CudaDeleter
    {
        void operator()(void *p) const {
            cudaFree(p);
        }
    };
    std::unique_ptr<T[], CudaDeleter> mData;
    size_t mSize;
};



void xqa_paged_attention(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_value_cache,
    int64_t num_heads, int64_t num_kv_heads, int64_t rotary_embedding_dim, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const std::string kv_cache_dtype, double k_scale, double v_scale);
