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
#include "tensorrt_llm/runtime/iTensor.h"

#include <functional>
#include <utility>

namespace tensorrt_llm::runtime
{

//! @details
//! ***Mandatory outputs***
//!
//!  * `ids`, is a tensor that contains the output token IDs. Its shape is
//!    `[batchSize, beamWidth, maxSeqLength]` where `maxSeqLength` is the sum of
//!    `maxInputLength` and `maxNewTokens`. After generation, it contains, for each
//!    sequence, a copy of the input tokens followed by the output tokens. When a
//!    sequence is shorter than `maxSeqLength`, padding tokens are added at the end
//!    of the sequence.
//!
//! _Note that the shape of that tensor is different in this version of
//! TensorRT-LLM from its shape in previous versions where it was `[maxSeqLength,
//! batchSize, beamWidth]`_.
//!
//! ***Optional outputs***
//!
//!  * `logProbs`, is a tensor of floating-point values on the GPU to store the
//!    log-prob of the generated tokens. Its shape is `[maxNewTokens, batchSize,
//!    beamWidth]`. Its shape will likely change in a future release to match the
//!    shape of the output `ids` tensor.
//!  * `contextLogits`, is a tensor of values on the GPU (same datatype as the
//!    computation type) to store the logits for the context. Its shape is
//!    `[batchSize, maxSequenceLength, vocabSizePadded]`. If use `remove_input_padding`, its shape is `[packedSize,
//!    vocabSizePadded]`. This buffer will only be filled in if the TensorRT engine was built with the
//!    `gather_context_logits` or `gather_all_token_logits` parameter enabled.
//!
//!    After inference is complete, you can get the context logits in `GenerationOutput.contextLogits`, these are
//!    variables on the GPU. For specific acquisition methods, please refer to the example of
//!    [gptSessionBenchmark.cpp](https://github.com/NVIDIA/TensorRT-LLM/blob/main/benchmarks/cpp/gptSessionBenchmark.cpp).
//!
//!    It is important to point out
//!    that enabling the computation may have an impact on performance (the language modeling head (LM head) has to
//!    perform a matrix multiplication on all the context tokens instead of a just the last one).
//!  * `generationLogits`, is a tensor of values on the GPU (same datatype as the
//!    computation type) to store the logits for the generation. Its shape is
//!    `[batchSize, beamWidth, maxOutputLen, vocabSizePadded]`. This buffer will only be
//!    filled in if the TensorRT engine was built with the `gather_generation_logits` or
//!    `gather_all_token_logits` parameter enabled.
//!
//!    Generation logits can also be obtained through `GenerationOutput.generationLogits` after inference is completed.
//!  * `onTokenGenerated`, is a callback function invoked in the generation loop to
//!    pass newly generated tokens to the caller while the loop continues to
//!    execute. An implementation of that callback must accept the output `ids`
//!    tensor, the generation `step` and a boolean flag that indicates if the
//!    generation is complete.
template <typename TTensor>
class GenericGenerationOutput
{
public:
    using TensorPtr = TTensor;
    using Callback = std::function<void(TensorPtr const& ids, SizeType32 step, bool finished)>;

    explicit GenericGenerationOutput(TensorPtr ids, TensorPtr lengths)
        : ids{std::move(ids)}
        , lengths{std::move(lengths)}
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->lengths), "Invalid lengths tensor");
    }

    // mandatory parameters
    TensorPtr ids;     // [batchSize, beamWidth, maxInputLength + maxNewTokens]
    TensorPtr lengths; // [batchSize, beamWidth]

    // optional parameters
    TensorPtr cumLogProbs;      // [batchSize, beamWidth], must be float*, on gpu
    TensorPtr logProbs;         // [batchSize, beamWidth, maxInputLength + maxNewTokens], must be float*, on gpu
    TensorPtr contextLogits;    // [batch_size, max_input_length, vocab_size_padded], if packed, the shape will be
                                // [packed_size, vocab_size_padded]
    TensorPtr generationLogits; // [batch_size, beam_width, max_output_length, vocab_size_padded]

    // callbacks
    Callback onTokenGenerated;
};

class GenerationOutput : public GenericGenerationOutput<ITensor::SharedPtr>
{
public:
    using Base = GenericGenerationOutput<ITensor::SharedPtr>;
    using TensorPtr = Base::TensorPtr;

    explicit GenerationOutput(TensorPtr ids, TensorPtr lengths)
        : GenericGenerationOutput(std::move(ids), std::move(lengths))
    {
    }
};

} // namespace tensorrt_llm::runtime
