/*
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
#include "decoder_xqa_impl_precompiled.h"
#include "xqa_params.h"
#include "decoder_xqa_impl_common.h"

class DecoderXQARunner
{
public:
    DecoderXQARunner(
        const XQADataType data_type, int num_heads, int num_kv_heads, int head_size, bool multi_block_mode);
    ~DecoderXQARunner();

    /**
     * \param[in] xqaParams the xqaParams to be tested against.
     */
    bool shouldUse(XQAParams const& xqaParams);

    size_t getWorkspaceSize(int max_num_tokens);

    void prepare(XQAParams const& xqa_params)
    {
        this->prepareForRun(xqa_params);
    }

    void dispatch(XQAParams const& xqa_params, KVCacheListParams const& kv_cache_buffer, cudaStream_t const& stream)
    {
        // sync_check_cuda_error(); //TODO
        this->run(xqa_params, kv_cache_buffer, stream);
    }

    class Resource;
    static Resource* getResourceGlobal();

private:
    void prepareForRun(XQAParams const& xqa_params);

    void run(XQAParams const& xqa_params, KVCacheListParams const& kv_cache_buffer, cudaStream_t const& stream);

    static constexpr int kMaxBeamWidth = 4;

    XQADataType mDataType;
    int mNumHeads;
    int mNumKVHeads;
    int mHeadSize;
    bool mMultiBlockMode;
    int mMultiProcessorCount;

    // std::unique_ptr<DecoderXQAImpl> mJITImpl, 
    std::unique_ptr<DecoderXQAImpl> mPrecompiledImpl;
    DecoderXQAImpl* getImplFromXQAParams(XQAParams const& params);

    friend DecoderXQAImplPrecompiled;
};
