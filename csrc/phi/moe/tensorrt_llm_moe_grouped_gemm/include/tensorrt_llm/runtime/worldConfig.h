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

#include <NvInferRuntime.h>
#include <optional>
#include <vector>

namespace tensorrt_llm::runtime
{
class WorldConfig
{
public:
#if ENABLE_MULTI_DEVICE
    static SizeType32 constexpr kDefaultGpusPerNode = 8;
#else
    static SizeType32 constexpr kDefaultGpusPerNode = 1;
#endif

    explicit WorldConfig(SizeType32 tensorParallelism = 1, SizeType32 pipelineParallelism = 1, SizeType32 rank = 0,
        SizeType32 gpusPerNode = kDefaultGpusPerNode,
        std::optional<std::vector<SizeType32>> const& deviceIds = std::nullopt);

    [[nodiscard]] SizeType32 constexpr getSize() const noexcept
    {
        return mTensorParallelism * mPipelineParallelism;
    }

    [[nodiscard]] SizeType32 constexpr getTensorParallelism() const noexcept
    {
        return mTensorParallelism;
    }

    [[nodiscard]] bool constexpr isTensorParallel() const noexcept
    {
        return mTensorParallelism > 1;
    }

    [[nodiscard]] SizeType32 constexpr getPipelineParallelism() const noexcept
    {
        return mPipelineParallelism;
    }

    [[nodiscard]] bool constexpr isPipelineParallel() const noexcept
    {
        return mPipelineParallelism > 1;
    }

    [[nodiscard]] SizeType32 constexpr getRank() const noexcept
    {
        return mRank;
    }

    [[nodiscard]] SizeType32 constexpr getGpusPerNode() const noexcept
    {
        return mGpusPerNode;
    }

    [[nodiscard]] SizeType32 getGpusPerGroup() const noexcept
    {
        return static_cast<SizeType32>(mDeviceIds.size());
    }

    [[nodiscard]] SizeType32 getDevice() const noexcept
    {
        return mDeviceIds[mRank % getGpusPerGroup()];
    }

    [[nodiscard]] SizeType32 getDeviceOf(SizeType32 rank) const noexcept
    {
        return mDeviceIds[rank % getGpusPerGroup()];
    }

    [[nodiscard]] SizeType32 constexpr getPipelineParallelRank() const noexcept
    {
        return mRank / mTensorParallelism;
    }

    [[nodiscard]] SizeType32 constexpr getTensorParallelRank() const noexcept
    {
        return mRank % mTensorParallelism;
    }

    [[nodiscard]] SizeType32 constexpr getLocalRank() const noexcept
    {
        return mRank % mGpusPerNode;
    }

    [[nodiscard]] SizeType32 constexpr getNodeRank() const noexcept
    {
        return mRank / mGpusPerNode;
    }

    [[nodiscard]] SizeType32 constexpr getNodeRankOf(SizeType32 rank) const noexcept
    {
        return rank / mGpusPerNode;
    }

    [[nodiscard]] bool constexpr isFirstPipelineParallelRank() const noexcept
    {
        return getPipelineParallelRank() == 0;
    }

    //! \brief Is my rank the last rank in its pipeline?
    [[nodiscard]] bool constexpr isLastPipelineParallelRank() const noexcept
    {
        return getPipelineParallelRank() == getPipelineParallelism() - 1;
    }

    [[nodiscard]] SizeType32 constexpr getLastRank() const noexcept
    {
        return getSize() - 1;
    }

    [[nodiscard]] std::vector<SizeType32> getPipelineParallelGroup() const;
    [[nodiscard]] std::vector<SizeType32> getTensorParallelGroup() const;

    static WorldConfig mpi(SizeType32 gpusPerNode = kDefaultGpusPerNode,
        std::optional<SizeType32> tensorParallelism = std::nullopt,
        std::optional<SizeType32> pipelineParallelism = std::nullopt,
        std::optional<std::vector<SizeType32>> const& deviceIds = std::nullopt);

    [[nodiscard]] bool validMpiConfig() const;

private:
    SizeType32 mTensorParallelism;
    SizeType32 mPipelineParallelism;
    SizeType32 mRank;
    SizeType32 mGpusPerNode;
    std::vector<SizeType32> mDeviceIds;
};

} // namespace tensorrt_llm::runtime
