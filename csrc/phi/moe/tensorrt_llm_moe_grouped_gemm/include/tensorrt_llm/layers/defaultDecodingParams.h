/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/common.h"

#include <limits>
#include <string>
#include <unordered_map>

namespace tensorrt_llm
{
namespace layers
{

class DefaultDecodingParams
{
public:
    [[nodiscard]] __host__ __device__ static constexpr float getTemperature()
    {
        return 1.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getRepetitionPenalty()
    {
        return 1.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getPresencePenalty()
    {
        return 0.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getFrequencyPenalty()
    {
        return 0.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr runtime::SizeType32 getMinLength()
    {
        return 1;
    }

    [[nodiscard]] __host__ __device__ static constexpr uint64_t getSeed()
    {
        return 0;
    }

    [[nodiscard]] __host__ __device__ static constexpr runtime::SizeType32 getTopK()
    {
        return 0;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getTopP()
    {
        return 0.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getTopPDecay()
    {
        return 1.0f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getTopPMin()
    {
        return 1.0e-6f;
    }

    [[nodiscard]] __host__ __device__ static constexpr runtime::TokenIdType getTopPResetId()
    {
        return -1;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getBeamSearchDiversity()
    {
        return 0.f;
    }

    [[nodiscard]] __host__ __device__ static constexpr float getLengthPenalty()
    {
        return 0.f;
    }

    [[nodiscard]] __host__ __device__ static constexpr runtime::SizeType32 getEarlyStopping()
    {
        return 1;
    }

    [[nodiscard]] __host__ __device__ static constexpr bool getNormalizeLogProbs()
    {
        return false;
    }

    [[nodiscard]] static std::vector<runtime::SizeType32> getTopKMedusaHeads()
    {
        return {};
    }

    [[nodiscard]] __host__ __device__ static constexpr runtime::SizeType32 getNoRepeatNgramSize()
    {
        return 1 << 30;
    }
};
} // namespace layers
} // namespace tensorrt_llm
