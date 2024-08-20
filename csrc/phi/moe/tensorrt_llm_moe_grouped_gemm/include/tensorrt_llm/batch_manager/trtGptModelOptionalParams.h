/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/batch_manager/peftCacheManagerConfig.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"

#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager
{

class TrtGptModelOptionalParams
{
    using KvCacheConfig = kv_cache_manager::KvCacheConfig;

public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    explicit TrtGptModelOptionalParams(KvCacheConfig const& kvCacheConfig = KvCacheConfig{},
        bool enableTrtOverlap = false, std::optional<std::vector<SizeType32>> const& deviceIds = std::nullopt,
        bool normalizeLogProbs = true, bool enableChunkedContext = false,
        PeftCacheManagerConfig const& peftCacheManagerConfig = PeftCacheManagerConfig{},
        executor::DecodingConfig const& decodingConfig = executor::DecodingConfig{}, float gpuWeightsPercent = 1)
        : kvCacheConfig{kvCacheConfig}
        , enableTrtOverlap{enableTrtOverlap}
        , deviceIds(deviceIds)
        , normalizeLogProbs{normalizeLogProbs}
        , enableChunkedContext{enableChunkedContext}
        , peftCacheManagerConfig(peftCacheManagerConfig)
        , decodingConfig(decodingConfig)
        , gpuWeightsPercent(gpuWeightsPercent)
    {
    }

    explicit TrtGptModelOptionalParams(executor::ExecutorConfig const& executorConfig)
        : TrtGptModelOptionalParams(KvCacheConfig(executorConfig.getKvCacheConfig()), false,
            executorConfig.getParallelConfig().value_or(executor::ParallelConfig()).getDeviceIds(),
            executorConfig.getNormalizeLogProbs(), executorConfig.getEnableChunkedContext(),
            PeftCacheManagerConfig(executorConfig.getPeftCacheConfig().value_or(executor::PeftCacheConfig())),
            executorConfig.getDecodingConfig().value_or(executor::DecodingConfig{}),
            executorConfig.getGpuWeightsPercent())
    {
    }

    bool operator==(TrtGptModelOptionalParams const& other) const
    {
        return kvCacheConfig == other.kvCacheConfig && enableTrtOverlap == other.enableTrtOverlap
            && deviceIds == other.deviceIds && normalizeLogProbs == other.normalizeLogProbs
            && enableChunkedContext == other.enableChunkedContext && decodingConfig == other.decodingConfig;
    }

    friend std::ostream& operator<<(std::ostream& os, TrtGptModelOptionalParams const& self);

    KvCacheConfig kvCacheConfig;

    bool enableTrtOverlap;
    std::optional<std::vector<SizeType32>> deviceIds;
    bool normalizeLogProbs;
    bool enableChunkedContext;
    PeftCacheManagerConfig peftCacheManagerConfig;
    executor::DecodingConfig decodingConfig;
    // Percentage of weights on the gpu at runtime
    float gpuWeightsPercent;
};

} // namespace tensorrt_llm::batch_manager
