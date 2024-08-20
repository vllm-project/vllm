/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <vector>

namespace tensorrt_llm::runtime
{

class SpeculativeDecodingModule
{
public:
    explicit SpeculativeDecodingModule(SizeType32 maxAcceptedTokens, SizeType32 maxDraftTokens) noexcept
        : mMaxAcceptedTokens(maxAcceptedTokens)
        , mMaxDraftTokens(maxDraftTokens)
    {
        computeNumPackedMasks();
    }

    explicit SpeculativeDecodingModule() noexcept
        : SpeculativeDecodingModule(0, 0)
    {
    }

    virtual ~SpeculativeDecodingModule() = default;

    SpeculativeDecodingModule(SpeculativeDecodingModule const& o) = default;
    SpeculativeDecodingModule& operator=(SpeculativeDecodingModule const& o) = default;

    [[nodiscard]] SizeType32 getMaxAcceptedDraftTokensPerStep() const noexcept
    {
        return mMaxAcceptedTokens;
    }

    [[nodiscard]] SizeType32 getMaxNewTokensPerStep() const noexcept
    {
        return getMaxAcceptedDraftTokensPerStep() + 1;
    }

    [[nodiscard]] SizeType32 getMaxDraftTokens() const noexcept
    {
        return mMaxDraftTokens;
    }

    [[nodiscard]] SizeType32 getNumPackedMasks() const noexcept
    {
        return mMaxNumPackedMasks;
    }

    void setMaxDraftTokens(SizeType32 maxDraftTokens) noexcept
    {
        mMaxDraftTokens = maxDraftTokens;
        computeNumPackedMasks();
    }

    void setMaxAcceptedDraftTokensPerStep(SizeType32 maxAcceptedTokens) noexcept
    {
        mMaxAcceptedTokens = maxAcceptedTokens;
    }

private:
    void computeNumPackedMasks() noexcept
    {
        mMaxNumPackedMasks = tensorrt_llm::common::divUp(mMaxDraftTokens, 32);
    }

private:
    SizeType32 mMaxAcceptedTokens;
    SizeType32 mMaxDraftTokens;
    SizeType32 mMaxNumPackedMasks;
};
} // namespace tensorrt_llm::runtime
