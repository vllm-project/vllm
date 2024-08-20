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
#include "tensorrt_llm/runtime/promptTuningParams.h"

#include <optional>
#include <utility>

namespace tensorrt_llm::runtime
{

//! @details
//! ***Mandatory inputs***
//!
//!  * `endId`, is the token ID that marks the end of the input sequence (aka `EOS`
//!    or end-of-sequence). It's `50,256` for the GPT2 model which has a vocabulary
//!    of `50,257` tokens, for example,
//!  * `padId`, is the token ID that is used for padding (i.e. fills in the slots
//!    that are at an index greater-or-equal to the input length for padded
//!    sequences). It can be set to the same value as `endId`,
//!  * `ids`, is the tensor of input IDs. That tensor must be allocated on the GPU.
//!    When the input tensor is padded, the shape of `ids` is `[batchSize,
//!    maxInputLength]`, where `batchSize` and `maxInputLength` must respect the
//!    maximum sizes in `sessionConfig` passed to the `GptSession` constructor.
//!    When the input is packed, the shape of `ids` is `[numTokens]`, where
//!    `numTokens` is the sum of the lengths of the different sequences in the batch,
//!  * `lengths`, is the tensor of input sequence lengths. That tensor must be
//!    allocated on the GPU and contain `batchSize` values,
//!  * `packed`, indicates if the `ids` tensor is packed or padded. In this
//!    release, that flag must match the value passed to the constructor through
//!    the instance of the `ModelConfig` class. In a future release, the session
//!    may be made more flexible and automatically pad or pack the input,
//!
//! ***Optional inputs***
//!
//!  * `embeddingBiasOpt`, is a tensor of floating-point values on the GPU that
//!    contains the bias to add to the logits during sampling (after the projection
//!    from hidden states to logits as the last step of the model). This tensor
//!    must have `vocabSize` elements (as defined in the `modelConfig` argument
//!    passed to the constructor),
//!  * `badWordsList`, is a tensor of integers on the GPU that encodes the list of
//!    words that have to be banned from generated sequences. Its shape is `[2,
//!    badWordsLength]`, as explained below, or `[batchSize, 2, badWordsLength]`
//!    when there is a different list for each sequence in the batch,
//!  * `stopWordsList`, is a tensor of integers on the GPU that encodes the list of
//!    words that trigger the end of the generation for a sequence. Its shape is
//!    `[2, stopWordsLength]`, as explained below, or `[batchSize, 2,
//!    stopWordsLength]` when there is a different list for each sequence in the
//!    batch,
//!  * `maxNewTokens`, is the maximum number of tokens to generate.
//!
//! The `badWordsList` and `stopWordsList` tensors have the same shape `[2,
//! length]`. Let's consider an example with three words to describe the
//! representation of those lists.  The first word contains tokens `[5, 7, 3]`, the
//! second one contains `[9, 2]` and the third one is composed of tokens `[6, 2, 4,
//! 1]`. In total, there are 9 tokens. That's the length. The shape of the tensor
//! is `[2, 9]`.  The first row of the tensor must contain the 9 token IDs and the
//! second row must store the
//! [inclusive prefix-sum](https://en.wikipedia.org/wiki/Prefix_sum)
//! of the word lengths as shown on the following diagram:
//!
//! ```
//!    0           3       5              9
//!    |           |       |              |
//!    V           V       V              V
//! [  5,  7,  3,  9,  2,  6,  2,  4,  1]
//! [  3,  5,  9, -1, -1, -1, -1, -1, -1]
//! ```
//!
//! In case all the words are made of a single token, the inner-most dimension of
//! the tensor must be increased by 1 (i.e. the length for 4 words, each made of a
//! single token, must be 5 instead of 4 -- the shape is `[2, 5]`).
template <typename TTensor, typename PromptTuningParams>
class GenericGenerationInput
{
public:
    using TensorPtr = TTensor;

    explicit GenericGenerationInput(
        SizeType32 const endId, SizeType32 const padId, TensorPtr ids, TensorPtr lengths, bool packed = false)
        : endId{endId}
        , padId{padId}
        , ids{std::move(ids)}
        , lengths{std::move(lengths)}
        , packed{packed}
        , maxNewTokens(std::nullopt)
    {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->lengths), "Invalid lengths tensor");
    }

    // mandatory parameters
    SizeType32 endId;
    SizeType32 padId;
    TensorPtr ids;     // [packedLength] or [batchSize, maxInputLength], on gpu
    TensorPtr lengths; // [batchSize], on gpu
    bool packed;       // indicates if ids are packed or padded to maxInputLength

    // optional parameters
    TensorPtr embeddingBias;                // [vocabSizePadded], on gpu
    TensorPtr badWordsList;                 // [2, badWordsLength] or [batchSize, 2, badWordsLength], on gpu
    TensorPtr stopWordsList;                // [batchSize, 2, stopWordsLength], on gpu
    std::optional<SizeType32> maxNewTokens; // max number of tokens to generate

    // Ptuning parameters
    PromptTuningParams promptTuningParams; // See promptTuningParams.h for expected shapes
};

class GenerationInput : public GenericGenerationInput<ITensor::SharedPtr, PromptTuningParams>
{
public:
    using Base = GenericGenerationInput<ITensor::SharedPtr, PromptTuningParams>;
    using TensorPtr = Base::TensorPtr;

    explicit GenerationInput(
        SizeType32 const endId, SizeType32 const padId, TensorPtr ids, TensorPtr lengths, bool packed = false)
        : GenericGenerationInput(endId, padId, std::move(ids), std::move(lengths), packed)
    {
    }
};

} // namespace tensorrt_llm::runtime
