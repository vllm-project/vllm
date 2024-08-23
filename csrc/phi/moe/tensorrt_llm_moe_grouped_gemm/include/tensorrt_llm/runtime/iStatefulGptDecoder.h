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

#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/samplingConfig.h"

#include <memory>
#include <utility>

#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{

namespace decoder
{

class Input
{
public:
    using TensorPtr = std::shared_ptr<ITensor const>;

    explicit Input(TensorPtr logits)
        : logits{std::move(logits)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->logits), "Invalid logits tensor");
    }

    // mandatory parameters
    TensorPtr logits; // [batchSize, maxBeamWidth, vocabSizePadded], on gpu

    // parameters for beam search
    TensorPtr cacheIndirection; // [batchSize, maxBeamWidth, maxSeqLen] - the k/v cache index for beam search, on gpu
};

class Output
{
public:
    using TensorPtr = std::shared_ptr<ITensor>;

    Output() = default;

    // parameters for beam search
    TensorPtr cacheIndirection; // [batchSize, maxBeamWidth, maxSeqLen], mandatory in beam search, on gpu
    TensorPtr sequenceLengths;  // [batchSize, maxBeamWidth], mandatory, on gpu
};
} // namespace decoder

//! GPT decoder class with support for in-flight batching
class IStatefulGptDecoder
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = std::shared_ptr<ITensor>;

    //! Setup the decoder before calling `forward()`, also calls reshapeBuffers
    virtual void setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
        SizeType32 maxTokensPerStep, bool fusedDecoder, nvinfer1::DataType dtype, ModelConfig const& modelConfig)
        = 0;

    //! @brief Initialize the decoder with new batch of inputs.
    virtual void newBatch(
        GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig)
        = 0;

    //! @brief Run one step for all requests without blocking the host thread.
    virtual void forwardAsync(decoder::Output& output, decoder::Input const& input) = 0;

    //! @brief Wait for the last call to `forwardAsync` to complete.
    virtual void forwardSync() = 0;

    //! @brief Run one step for all requests.
    virtual void forward(decoder::Output& output, decoder::Input const& input)
    {
        forwardAsync(output, input);
        return forwardSync();
    }

    //! @brief Gather final beam search results for all requests.
    virtual void finalize() const = 0;

    //! @returns [batchSize, beamWidth, maxSequenceLength], all token ids, on gpu
    [[nodiscard]] virtual TensorPtr getOutputIds() const = 0;

    //! @returns [batchSize, maxBeamWidth], cumulative log probabilities (per beam), on gpu
    [[nodiscard]] virtual TensorPtr getCumLogProbs() const = 0;

    //! @returns [batchSize, maxBeamWidth, maxSequenceLength], log probabilities (per beam), on gpu
    [[nodiscard]] virtual TensorPtr getLogProbs() const = 0;

    //! @brief Get tokens generated in one step of last forward pass
    //! @param iter The iteration within [0; maxTokensPerStep) for which to get the tokens
    //! @returns [batchSize, beamWidth], tokens generated in `iter` (per beam), on gpu
    [[nodiscard]] virtual TensorPtr getNewTokens(SizeType32 iter = 0) const = 0;

    //! @brief Get maxTokensPerStep tokens generated in the last forward pass
    //! @returns [maxTokensPerStep, batchSize, maxBeamWidth], tokens generated in last forward pass, on gpu
    [[nodiscard]] virtual TensorPtr getAllNewTokens() const = 0;

    //! @returns [1], number of finished sequences, in pinned host memory
    [[nodiscard]] virtual TensorPtr getNbFinished() const = 0;

    virtual ~IStatefulGptDecoder() = default;

protected:
    IStatefulGptDecoder() = default;
};

} // namespace tensorrt_llm::runtime
