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
#include <optional>
#include <utility>

namespace tensorrt_llm::runtime
{
class DecodingOutput
{
public:
    using TensorPtr = ITensor::SharedPtr;

    // BS: batch_size, BM: beam_width, MSL: max_seq_length
    // All TensorPtr without special comments are on gpu

    class BeamHypotheses
    {
    public:
        // Keep same as cpp/tensorrt_llm/kernels/beamSearchKernels.h
        TensorPtr outputIdsCBA;       // [BS, BM*2, MSL]
        TensorPtr logProbsCBA;        // [BS, BM*2, MSL]
        TensorPtr sequenceLengthsCBA; // [BS, BM*2]
        TensorPtr cumLogProbsCBA;     // [BS, BM*2]
        TensorPtr normedScoresCBA;    // [BS, BM*2]
        TensorPtr numBeamsCBA;        // [BS]
        TensorPtr minNormedScoresCBA; // [BS]
        TensorPtr batchDones;         // [BS]

        void empty(BufferManager& manager);

        void reshape(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxSequenceLength);

        void release();

        void init(BufferManager& manager, TokenIdType endId);

        BeamHypotheses slice(SizeType32 batchIndex, SizeType32 size) const;
    };

    static float constexpr kNegativeInfinity = -1e20f;

    explicit DecodingOutput(TensorPtr ids)
        : ids{std::move(ids)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
    }

    // mandatory parameters
    TensorPtr ids;                       // [BS, BM, MSL], contains previously generated token ids for all
                                         // steps before DecodingInput.step
    TensorPtr newTokensSteps;            // [maxTokensPerStep, BS, BM] new tokens at each generated token of
                                         // maxTokensPerStep
    TensorPtr newTokens;                 // [BS, BM] usually a view of newTokensSteps for the current token
    std::vector<TensorPtr> newTokensVec; // vector of size maxTokensPerStep with tensor [BS, BM].
                                         // Vector of views on newTokensSteps for each token

    // optional parameters
    TensorPtr finished; // [BS, BM], set to true by decoding if any of the stop conditions are met or if
                        // DecodingInput.finished is true. In beam search and to determine whether to stop according to
                        // DecodingInput.sequenceLimitLength
    TensorPtr finishedSum; // [BS], the sum of finished sequences per request, in pinned memory

    // mandatory parameters for beam search
    TensorPtr logProbs;         // [BS, BM, MSL], must be float*
    TensorPtr cumLogProbs;      // [BS, BM], optional for sampling
    TensorPtr parentIds;        // [BS, BM, MSL]
    TensorPtr lengths;          // [BS, BM], total sequence lengths including padding
    TensorPtr cacheIndirection; // [BS, BM, MSL], k/v indirection for next generation step

    BeamHypotheses beamHypotheses;

    // Speculative decoding
    class SpeculativeDecodingOutputs
    {
    public:
        TensorPtr nextDraftTokens;       // [maxBatchSize, maxDraftTokens]
        TensorPtr acceptedTokensLen;     // [maxBatchSize]
        TensorPtr acceptedLengthsCumSum; // [maxBatchSize + 1]
        TensorPtr pathsOffsets;          // [maxBatchSize, maxAcceptedDraftTokensPerStep]
    };

    std::optional<SpeculativeDecodingOutputs> speculativeDecodingOutputs;
};

} // namespace tensorrt_llm::runtime
