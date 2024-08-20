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

#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iStatefulGptDecoder.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

#include <memory>
#include <utility>
#include <vector>

namespace tensorrt_llm::runtime
{

namespace decoder_batch
{
class Request
{
public:
    using ConstTensorPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;
    using BufferPtr = IBuffer::SharedPtr;

    explicit Request(ConstTensorPtr ids, SizeType32 inputLen, std::optional<SizeType32> maxNewTokens = std::nullopt,
        std::optional<SizeType32> endId = std::nullopt)
        : ids{std::move(ids)}
        , inputLen(inputLen)
        , maxNewTokens{maxNewTokens}
        , endId{endId}
        , generatedTokensPerEngineStep(1)
    {
    }

    // mandatory parameters
    ConstTensorPtr ids;  // [inputSeqLen], the input sequence of token ids, on gpu
    SizeType32 inputLen; // the input length without draft tokens

    // optional parameters
    std::optional<SizeType32> maxNewTokens; // maximum number of tokens to generate for this request
    std::optional<SizeType32> endId;        // end token id
    BufferPtr draftTokens;   // [generatedTokensPerStep - 1], on gpu, draft tokens from speculative decoding
    std::optional<TensorPtr>
        draftLogits;         // [generatedTokensPerStep - 1, vocabSize], on gpu, draft tokens from speculative decoding
    TensorPtr embeddingBias; // [vocabSizePadded], on gpu
    TensorPtr badWordsList;  // [2, badWordsLength], on gpu
    TensorPtr stopWordsList; // [2, stopWordsLength], on gpu

    SizeType32 generatedTokensPerEngineStep;
    TensorPtr medusaPaths;   // [maxDraftTokens + 1, maxAcceptedDraftTokensPerStep + 1], on gpu
    TensorPtr medusaTreeIds; // [maxDraftTokens + 1], on gpu
};

class Input
{
public:
    using TensorConstPtr = ITensor::SharedConstPtr;
    using TensorPtr = ITensor::SharedPtr;

    explicit Input(std::vector<TensorConstPtr> const& logits, std::vector<bool> const& active)
        : logits{logits}
        , active{active}
    {
        TLLM_CHECK_WITH_INFO(
            this->active.size() == logits.size(), "'active' vector size does not match logits vector size");
    }

    explicit Input(std::vector<TensorConstPtr> const& logits)
        : Input{logits, std::vector<bool>(logits.size(), true)}
    {
    }

    explicit Input(std::vector<TensorPtr> const& logits, std::vector<bool> const& active)
        : Input{
            utils::transformVector(logits, [](auto& x) { return std::const_pointer_cast<ITensor const>(x); }), active}
    {
    }

    explicit Input(std::vector<TensorPtr> const& logits)
        : Input{logits, std::vector<bool>(logits.size(), true)}
    {
    }

    // mandatory parameters
    std::vector<TensorConstPtr>
        logits; // batchSize * [1, beamWidth, vocabSizePadded] or [generatedTokensPerStep, 1, vocabSizePadded], on gpu

    // control activity of decoder slots in batch
    std::vector<bool> active; // [batchSize]

    // parameters for beam search
    TensorConstPtr cacheIndirection; // [batchSize, maxBeamWidth, maxSeqLen] - indices into KV cache of different rays
                                     // within one beam for beam search, on gpu
    std::vector<std::vector<TensorConstPtr>>
        predictedDraftLogits; // [maxBatchSize][maxAcceptedDraftTokensPerStep][maxDraftTokens + 1, vocabSizePadded]
};

using Output = decoder::Output;

// TODO: is this a bad name to mix up with token concept in LLM? Would 'Event' be better? And should move to common.h
class Token
{
public:
    explicit Token(CudaEvent&& event, std::vector<bool> const& active)
        : event(std::move(event))
        , active(active)
    {
    }

    CudaEvent event;
    std::vector<bool> active;
};
} // namespace decoder_batch

//! GPT decoder class with support for in-flight batching
class IGptDecoderBatch : public virtual IStatefulGptDecoder
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = std::shared_ptr<ITensor>;
    using TokenPtr = std::unique_ptr<decoder_batch::Token const>;

    //! @brief Run one step for all requests without blocking the host process and return the token for synchronization.
    virtual TokenPtr forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) = 0;

    //! @brief Call decoder forwardSync and wait for the call to `forwardAsync` associated with a token to complete.
    virtual void forwardSync(
        decoder_batch::Token const& token, decoder_batch::Output& output, decoder_batch::Input const& input)
        = 0;

    //! @brief Wait for the call to `forwardAsync` associated with a token to complete.
    virtual void forwardSync(decoder_batch::Token const& token) = 0;

    //! @brief Run one step for all requests and wait for completion on the host.
    virtual void forward(decoder_batch::Output& output, decoder_batch::Input const& input)
    {
        forwardSync(*forwardAsync(output, input));
    }

    //! @param batchIdx index of the batch
    //! @returns [maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token
    //! ids without padding for request `batchIdx`, on gpu
    [[nodiscard]] virtual TensorPtr getOutputIds(SizeType32 batchIdx) const = 0;

    //! @brief Gather final beam search results for request `batchIdx`.
    //! Result will only be available after event returned
    [[nodiscard]] virtual CudaEvent finalize(SizeType32 batchIdx) const = 0;

    //! @returns [batchSize (actual)], marks finished requests (per batch)
    [[nodiscard]] virtual std::vector<bool> getFinished() const = 0;

    //! @returns [batchSize, beamWidth], cumulative log probabilities (per beam), on gpu
    [[nodiscard]] virtual TensorPtr getCumLogProbs() const = 0;

    //! @returns [beamWidth], cumulative log probabilities (per beam) for request batchIdx, on gpu
    [[nodiscard]] virtual TensorPtr getCumLogProbs(SizeType32 batchIdx) const = 0;

    //! @returns [batchSize, beamWidth, maxSeqLen], log probabilities (per beam), on gpu
    [[nodiscard]] virtual TensorPtr getLogProbs() const = 0;

    //! @returns [beamWidth, maxSeqLen], cumulative log probabilities (per beam) for request batchIdx, on gpu
    [[nodiscard]] virtual TensorPtr getLogProbs(SizeType32 batchIdx) const = 0;

    [[nodiscard]] virtual TensorPtr getParentIds() const = 0;

    [[nodiscard]] virtual std::vector<SizeType32> getNbSteps() const = 0;

    //! @brief Initialize batched decoder at seqSlots with a new `requests`.
    virtual void newRequests(std::vector<SizeType32> const& seqSlots,
        std::vector<decoder_batch::Request> const& requests, std::vector<SamplingConfig> const& samplingConfigs)
        = 0;

    //! @returns [batchSize, maxTokensPerStep-1], predicted draft tokens for next step, on gpu
    virtual TensorPtr getNextDraftTokens() const = 0;

    //! @returns [batchSize + 1], exclusive sum of accepted draft token lengths, on gpu
    virtual TensorPtr getSpecDecodingAcceptedLengthsCumSum() const = 0;

    //! @returns [batchSize, maxAcceptedDraftTokensPerStep], accepted paths packed into continuous tensor, on gpu
    virtual TensorPtr getSpecDecodingAcceptedPackedPaths() const = 0;

protected:
    IGptDecoderBatch() = default;
};

} // namespace tensorrt_llm::runtime
