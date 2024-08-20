
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

#include "common.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime
{

class IpcMemory
{
public:
    using BufferPtr = IBuffer::SharedPtr;

    // MAX_ALL_REDUCE_BLOCKS for block_barrier, 1 for multi_gpu_barrier
    size_t static constexpr FLAGS_SIZE = (kernels::MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t);

    IpcMemory(std::size_t bufferSize, BufferManager const& manager, WorldConfig const& worldConfig);
    ~IpcMemory();

    IpcMemory(IpcMemory const&) = delete;
    IpcMemory& operator=(IpcMemory const&) = delete;

    IpcMemory(IpcMemory&&) = default;
    IpcMemory& operator=(IpcMemory&&) = default;

    [[nodiscard]] std::vector<void*> const& getCommPtrs() const
    {
        return mCommPtrs;
    }

private:
    void allocateIpcMemory(std::size_t bufferSize, BufferManager const& manager, WorldConfig const& worldConfig);
    void destroyIpcMemory();

    SizeType32 mTpRank;
    std::vector<void*> mCommPtrs;
    BufferPtr mBuffer;
    bool mOpenIpc;
};

class AllReduceBuffers
{
public:
    using TensorPtr = ITensor::SharedPtr;

    AllReduceBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxSequenceLength,
        SizeType32 hiddenSize, BufferManager const& manager, WorldConfig const& worldConfig);

    TensorPtr mAllReduceCommPtrs;
    std::vector<runtime::IpcMemory> mIpcMemoryHandles;
};

} // namespace tensorrt_llm::runtime
