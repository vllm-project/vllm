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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <utility>

namespace tensorrt_llm::runtime
{

template <typename TTensor>
class GenericPromptTuningParams
{
public:
    using TensorPtr = TTensor;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    explicit GenericPromptTuningParams(
        TensorPtr embeddingTable = TensorPtr(), TensorPtr tasks = TensorPtr(), TensorPtr vocabSize = TensorPtr())
        : embeddingTable{std::move(embeddingTable)}
        , tasks{std::move(tasks)}
        , vocabSize{std::move(vocabSize)} {};

    // The prompt embedding table
    TensorPtr embeddingTable; // [numTasks * taskVocabSize, hidden_dim], on gpu
    // In GenerationInput, tasks expected shape is [batchSize]
    // For context requests with non-packed inputs, expected shape is [batchSize, 1]
    // For generation requests with non-packed inputs, expected shape is [batchSize*beamWidth] for generation requests.
    // For packed inputs, expected shape is [packedLength] (note that ifb currently doesn't support non-packed
    // inputs)
    TensorPtr tasks;
    TensorPtr vocabSize; // [1], on gpu

    std::vector<bool>
        promptTuningEnabled; // [batchSize] vector of bool that indicates which requests in a batch have ptuning enabled
};

class PromptTuningParams : public GenericPromptTuningParams<ITensor::SharedPtr>
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using SizeType32 = GenericPromptTuningParams::SizeType32;

    explicit PromptTuningParams(
        TensorPtr embeddingTable = nullptr, TensorPtr tasks = nullptr, TensorPtr vocabSize = nullptr)
        : GenericPromptTuningParams(std::move(embeddingTable), std::move(tasks), std::move(vocabSize))
    {
    }

    // Fill the tasks tensor for the batch using the provided tasksHost
    // Function assumes that the first numContextRequests requests in the batch are context requests
    void fillTasksTensor(TensorPtr tasksHost, const SizeType32 batchSize, const SizeType32 numContextRequests,
        std::vector<SizeType32> const& reqBeamWidths, std::vector<SizeType32> const& reqPromptLengths,
        BufferManager const& manager, bool packedInput);
};

} // namespace tensorrt_llm::runtime
