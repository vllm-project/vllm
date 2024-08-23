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

#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/lookaheadModule.h"
#include "tensorrt_llm/runtime/loraModule.h"
#include "tensorrt_llm/runtime/medusaModule.h"
#include "tensorrt_llm/runtime/speculativeDecodingMode.h"
#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{

class ModelConfig
{
public:
    enum class ModelVariant : std::int32_t
    {
        kGpt = 0,
        kGlm = 1,            // https://github.com/THUDM/GLM and https://github.com/THUDM/ChatGLM-6B
        kMamba = 2,          // https://github.com/state-spaces/mamba
        kRecurrentGemma = 3, // https://github.com/google-deepmind/recurrentgemma
        kEncDec = 4,
    };

    struct RnnConfig
    {
        SizeType32 stateSize = 0;
        SizeType32 convKernel = 0;
        SizeType32 rnnHiddenSize = 0;
    };

    enum class LayerType : std::int32_t
    {
        kATTENTION,
        kRECURRENT,
    };

    explicit ModelConfig(SizeType32 vocabSize, SizeType32 nbAttentionLayers, SizeType32 nbRnnLayers, SizeType32 nbHeads,
        SizeType32 hiddenSize, nvinfer1::DataType dtype)
        : mVocabSize(vocabSize)
        , mNbAttentionLayers(nbAttentionLayers)
        , mNbRnnLayers(nbRnnLayers)
        , mNbHeads(nbHeads)
        , mNbKvHeads(nbHeads)
        , mHiddenSize(hiddenSize)
        , mSizePerHead(mHiddenSize / mNbHeads)
        , mDataType(dtype)
        , mUseGptAttentionPlugin(false)
        , mUseMambaConv1dPlugin(false)
        , mInputPacked{false}
        , mPagedKvCache{false}
        , mPagedState{false}
        , mTokensPerBlock{64}
        , mQuantMode{common::QuantMode::none()}
        , mMaxBatchSize(0)
        , mMaxBeamWidth(0)
        , mMaxInputLen(0)
        , mMaxSequenceLen(0)
        , mMaxNumTokens(std::nullopt)
        , mComputeContextLogits(false)
        , mComputeGenerationLogits(false)
        , mModelVariant(ModelVariant::kGpt)
        , mUseCustomAllReduce(false)
        , mMaxPromptEmbeddingTableSize(0)
        , mMaxDraftLen(0)
        , mPagedContextFMHA(false)
        , mUseXQA{false}
        , mUseLoraPlugin(false)
        , mMlpHiddenSize(0)
        , mUseCrossAttention(false)
        , mUsePositionEmbedding(false)
        , mUseTokenTypeEmbedding(false)
        , mSpeculativeDecodingMode(SpeculativeDecodingMode::None())
    {
    }

    [[nodiscard]] SizeType32 constexpr getVocabSize() const noexcept
    {
        return mVocabSize;
    }

    [[nodiscard]] SizeType32 constexpr getVocabSizePadded(SizeType32 worldSize) const noexcept
    {
        return (mVocabSize + worldSize - 1) / worldSize * worldSize;
    }

    [[nodiscard]] SizeType32 constexpr getNbAttentionLayers(SizeType32 pipelineParallelism = 1) const
    {
        TLLM_CHECK(mNbAttentionLayers % pipelineParallelism == 0);
        return mNbAttentionLayers / pipelineParallelism;
    }

    [[nodiscard]] SizeType32 constexpr getNbRnnLayers(SizeType32 pipelineParallelism = 1) const
    {
        TLLM_CHECK(mNbRnnLayers % pipelineParallelism == 0);
        return mNbRnnLayers / pipelineParallelism;
    }

    [[nodiscard]] SizeType32 constexpr getNbHeads() const noexcept
    {
        return mNbHeads;
    }

    [[nodiscard]] SizeType32 constexpr getNbKvHeads() const noexcept
    {
        return mNbKvHeads;
    }

    void constexpr setNbKvHeads(SizeType32 nbKvHeads) noexcept
    {
        mNbKvHeads = nbKvHeads;
    }

    [[nodiscard]] SizeType32 constexpr getHiddenSize() const noexcept
    {
        return mHiddenSize;
    }

    [[nodiscard]] SizeType32 constexpr getEncoderHiddenSize() const noexcept
    {
        return mEncoderHiddenSize;
    }

    void constexpr setEncoderHiddenSize(SizeType32 encoderHiddenSize) noexcept
    {
        mEncoderHiddenSize = encoderHiddenSize;
    }

    [[nodiscard]] SizeType32 constexpr getSizePerHead() const noexcept
    {
        return mSizePerHead;
    }

    void constexpr setSizePerHead(SizeType32 sizePerHead) noexcept
    {
        mSizePerHead = sizePerHead;
    }

    [[nodiscard]] nvinfer1::DataType constexpr getDataType() const noexcept
    {
        return mDataType;
    }

    [[nodiscard]] bool constexpr useGptAttentionPlugin() const noexcept
    {
        return mUseGptAttentionPlugin;
    }

    void constexpr useGptAttentionPlugin(bool useGptAttentionPlugin) noexcept
    {
        mUseGptAttentionPlugin = useGptAttentionPlugin;
    }

    [[nodiscard]] bool constexpr useMambaConv1dPlugin() const noexcept
    {
        return mUseMambaConv1dPlugin;
    }

    void constexpr useMambaConv1dPlugin(bool useMambaConv1dPlugin) noexcept
    {
        mUseMambaConv1dPlugin = useMambaConv1dPlugin;
    }

    [[nodiscard]] bool constexpr usePackedInput() const noexcept
    {
        return mInputPacked;
    }

    void constexpr usePackedInput(bool inputPacked) noexcept
    {
        mInputPacked = inputPacked;
    }

    [[nodiscard]] bool constexpr usePagedKvCache() const noexcept
    {
        return mPagedKvCache;
    }

    void constexpr usePagedKvCache(bool pagedKvCache) noexcept
    {
        mPagedKvCache = pagedKvCache;
    }

    [[nodiscard]] bool constexpr usePagedState() const noexcept
    {
        return mPagedState;
    }

    void constexpr usePagedState(bool pagedState) noexcept
    {
        mPagedState = pagedState;
    }

    [[nodiscard]] SizeType32 constexpr getTokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    void constexpr setTokensPerBlock(SizeType32 TokensPerBlock) noexcept
    {
        mTokensPerBlock = TokensPerBlock;
    }

    [[nodiscard]] common::QuantMode constexpr getQuantMode() const noexcept
    {
        return mQuantMode;
    }

    void constexpr setQuantMode(common::QuantMode QuantMode) noexcept
    {
        mQuantMode = QuantMode;
    }

    [[nodiscard]] bool constexpr supportsInflightBatching() const noexcept
    {
        return (isTransformerBased() && mUseGptAttentionPlugin && mInputPacked && mPagedKvCache)
            || (isRnnBased() && mUseMambaConv1dPlugin && mInputPacked && mPagedState);
    }

    [[nodiscard]] SizeType32 constexpr getMaxBatchSize() const noexcept
    {
        return mMaxBatchSize;
    }

    void constexpr setMaxBatchSize(SizeType32 maxBatchSize) noexcept
    {
        mMaxBatchSize = maxBatchSize;
    }

    [[nodiscard]] SizeType32 constexpr getMaxBeamWidth() const noexcept
    {
        return mMaxBeamWidth;
    }

    void constexpr setMaxBeamWidth(SizeType32 maxBeamWidth) noexcept
    {
        mMaxBeamWidth = maxBeamWidth;
    }

    [[nodiscard]] SizeType32 constexpr getMaxInputLen() const noexcept
    {
        return mMaxInputLen;
    }

    void constexpr setMaxInputLen(SizeType32 maxInputLen) noexcept
    {
        mMaxInputLen = maxInputLen;
    }

    [[nodiscard]] SizeType32 constexpr getMaxSequenceLen() const noexcept
    {
        return mMaxSequenceLen;
    }

    void constexpr setMaxSequenceLen(SizeType32 maxSequenceLen) noexcept
    {
        mMaxSequenceLen = maxSequenceLen;
    }

    [[nodiscard]] std::optional<SizeType32> constexpr getMaxNumTokens() const noexcept
    {
        return mMaxNumTokens;
    }

    void constexpr setMaxNumTokens(std::optional<SizeType32> maxNumTokens) noexcept
    {
        mMaxNumTokens = maxNumTokens;
    }

    [[nodiscard]] SizeType32 constexpr getMaxEncoderLen() const noexcept
    {
        return mMaxEncoderLen;
    }

    void constexpr setMaxEncoderLen(SizeType32 maxEncoderLen) noexcept
    {
        mMaxEncoderLen = maxEncoderLen;
    }

    [[nodiscard]] bool constexpr usePromptTuning() const noexcept
    {
        return mMaxPromptEmbeddingTableSize > 0;
    }

    [[nodiscard]] SizeType32 constexpr getMaxPromptEmbeddingTableSize() const noexcept
    {
        return mMaxPromptEmbeddingTableSize;
    }

    void constexpr setMaxPromptEmbeddingTableSize(SizeType32 maxPromptEmbeddingTableSize) noexcept
    {
        mMaxPromptEmbeddingTableSize = maxPromptEmbeddingTableSize;
    }

    [[nodiscard]] bool constexpr computeContextLogits() const noexcept
    {
        return mComputeContextLogits;
    }

    void constexpr computeContextLogits(bool computeContextLogits) noexcept
    {
        mComputeContextLogits = computeContextLogits;
    }

    [[nodiscard]] bool constexpr computeGenerationLogits() const noexcept
    {
        return mComputeGenerationLogits;
    }

    void constexpr computeGenerationLogits(bool computeGenerationLogits) noexcept
    {
        mComputeGenerationLogits = computeGenerationLogits;
    }

    [[nodiscard]] ModelVariant getModelVariant() const
    {
        return mModelVariant;
    }

    void setModelVariant(ModelVariant modelVariant)
    {
        mModelVariant = modelVariant;
    }

    [[nodiscard]] bool constexpr useCustomAllReduce() const noexcept
    {
        return mUseCustomAllReduce;
    }

    void constexpr useCustomAllReduce(bool customAllReduce) noexcept
    {
        mUseCustomAllReduce = customAllReduce;
    }

    void constexpr setMaxDraftLen(SizeType32 maxDraftLen) noexcept
    {
        mMaxDraftLen = maxDraftLen;
    }

    [[nodiscard]] SizeType32 getMaxDraftLen() const
    {
        return mMaxDraftLen;
    }

    [[nodiscard]] SizeType32 constexpr getMaxTokensPerStep() const noexcept
    {
        return mMaxDraftLen + 1;
    }

    void constexpr setPagedContextFMHA(bool pagedContextFMHA) noexcept
    {
        mPagedContextFMHA = pagedContextFMHA;
    }

    [[nodiscard]] bool constexpr getPagedContextFMHA() const noexcept
    {
        return mPagedContextFMHA;
    }

    void constexpr useXQA(bool useXQA) noexcept
    {
        mUseXQA = useXQA;
    }

    [[nodiscard]] bool constexpr useXQA() const noexcept
    {
        return mUseXQA;
    }

    [[nodiscard]] bool constexpr useLoraPlugin() const noexcept
    {
        return mUseLoraPlugin;
    }

    void constexpr useLoraPlugin(bool useLoraPlugin) noexcept
    {
        mUseLoraPlugin = useLoraPlugin;
    }

    [[nodiscard]] std::vector<LoraModule> const& getLoraModules() const noexcept
    {
        return mLoraModules;
    }

    void setLoraModules(std::vector<LoraModule> const& loraModules) noexcept
    {
        mLoraModules = loraModules;
    }

    [[nodiscard]] SizeType32 constexpr getMlpHiddenSize() const noexcept
    {
        return mMlpHiddenSize;
    }

    void constexpr setMlpHiddenSize(SizeType32 mlpHiddenSize) noexcept
    {
        mMlpHiddenSize = mlpHiddenSize;
    }

    [[nodiscard]] bool constexpr useCrossAttention() const noexcept
    {
        return mUseCrossAttention;
    }

    void constexpr setUseCrossAttention(bool useCrossAttention) noexcept
    {
        mUseCrossAttention = useCrossAttention;
    }

    [[nodiscard]] bool constexpr usePositionEmbedding() const noexcept
    {
        return mUsePositionEmbedding;
    }

    void constexpr setUsePositionEmbedding(bool usePositionEmbedding) noexcept
    {
        mUsePositionEmbedding = usePositionEmbedding;
    }

    [[nodiscard]] bool constexpr useTokenTypeEmbedding() const noexcept
    {
        return mUseTokenTypeEmbedding;
    }

    void constexpr setUseTokenTypeEmbedding(bool useTokenTypeEmbedding) noexcept
    {
        mUseTokenTypeEmbedding = useTokenTypeEmbedding;
    }

    [[nodiscard]] SizeType32 constexpr getMaxLoraRank() const noexcept
    {
        return mMaxLoraRank;
    }

    void constexpr setMaxLoraRank(SizeType32 maxLoraRank) noexcept
    {
        mMaxLoraRank = maxLoraRank;
    }

    void setSpeculativeDecodingMode(SpeculativeDecodingMode mode) noexcept
    {
        mSpeculativeDecodingMode = mode;
    }

    [[nodiscard]] bool hasSpeculativeDecodingModule() const noexcept
    {
        return mSpeculativeDecodingModule != nullptr;
    }

    [[nodiscard]] SpeculativeDecodingModule const& getSpeculativeDecodingModule() const noexcept
    {
        TLLM_CHECK_WITH_INFO(mSpeculativeDecodingModule, "Speculative decoding module is not set");
        return *mSpeculativeDecodingModule;
    }

    [[nodiscard]] std::shared_ptr<SpeculativeDecodingModule const> getSpeculativeDecodingModulePtr() const noexcept
    {
        TLLM_CHECK_WITH_INFO(mSpeculativeDecodingModule, "Speculative decoding module is not set");
        return mSpeculativeDecodingModule;
    }

    [[nodiscard]] std::shared_ptr<SpeculativeDecodingModule> getSpeculativeDecodingModulePtr() noexcept
    {
        TLLM_CHECK_WITH_INFO(mSpeculativeDecodingModule, "Speculative decoding module is not set");
        return mSpeculativeDecodingModule;
    }

    void setSpeculativeDecodingModule(
        std::shared_ptr<SpeculativeDecodingModule> const& speculativeDecodingModule) noexcept
    {
        mSpeculativeDecodingModule = speculativeDecodingModule;
    }

    [[nodiscard]] nvinfer1::DataType getKvDataType() const noexcept
    {
        if (getQuantMode().hasFp8KvCache())
        {
            return nvinfer1::DataType::kFP8;
        }
        else if (getQuantMode().hasInt8KvCache())
        {
            return nvinfer1::DataType::kINT8;
        }
        else
        {
            return getDataType();
        }
    }

    [[nodiscard]] bool constexpr isTransformerBased() const noexcept
    {
        return mModelVariant == ModelVariant::kGpt || mModelVariant == ModelVariant::kGlm
            || mModelVariant == ModelVariant::kRecurrentGemma;
    }

    [[nodiscard]] bool hasRnnConfig() const noexcept
    {
        return mRnnConfig.has_value();
    }

    [[nodiscard]] std::optional<RnnConfig> getRnnConfig() const noexcept
    {
        return mRnnConfig;
    }

    void setRnnConfig(RnnConfig const& rnnConfig) noexcept
    {
        mRnnConfig = rnnConfig;
    }

    [[nodiscard]] bool constexpr isRnnBased() const noexcept
    {
        return mModelVariant == ModelVariant::kMamba || mModelVariant == ModelVariant::kRecurrentGemma;
    }

    [[nodiscard]] std::vector<LayerType> const& getLayerTypes() const noexcept
    {
        return mLayerTypes;
    }

    void setLayerTypes(std::vector<LayerType> const& layerTypes) noexcept
    {
        mLayerTypes = layerTypes;
    }

    [[nodiscard]] SpeculativeDecodingMode getSpeculativeDecodingMode() const noexcept
    {
        return mSpeculativeDecodingMode;
    }

private:
    SizeType32 mVocabSize;
    SizeType32 mNbAttentionLayers;
    SizeType32 mNbRnnLayers;
    SizeType32 mNbHeads;
    SizeType32 mNbKvHeads;
    SizeType32 mHiddenSize;
    SizeType32 mSizePerHead;
    nvinfer1::DataType mDataType;
    bool mUseGptAttentionPlugin;
    bool mUseMambaConv1dPlugin;
    bool mInputPacked;
    bool mPagedKvCache;
    bool mPagedState;
    SizeType32 mTokensPerBlock;
    common::QuantMode mQuantMode;
    SizeType32 mMaxBatchSize;
    SizeType32 mMaxBeamWidth;
    SizeType32 mMaxInputLen;
    SizeType32 mMaxSequenceLen;
    std::optional<SizeType32> mMaxNumTokens;

    bool mComputeContextLogits;
    bool mComputeGenerationLogits;
    ModelVariant mModelVariant;
    bool mUseCustomAllReduce;

    SizeType32 mMaxPromptEmbeddingTableSize;
    SizeType32 mMaxDraftLen;

    bool mPagedContextFMHA;
    bool mUseXQA;

    bool mUseLoraPlugin;
    std::vector<LoraModule> mLoraModules;
    SizeType32 mMlpHiddenSize;
    SizeType32 mMaxLoraRank;

    std::optional<RnnConfig> mRnnConfig;

    // Configs related to encoder / enc-dec models
    SizeType32 mMaxEncoderLen{};
    SizeType32 mEncoderHiddenSize{};
    bool mUseCrossAttention;
    bool mUsePositionEmbedding;
    bool mUseTokenTypeEmbedding;

    std::vector<LayerType> mLayerTypes;
    // Speculative decoding members
    std::shared_ptr<SpeculativeDecodingModule> mSpeculativeDecodingModule;
    SpeculativeDecodingMode mSpeculativeDecodingMode;
};

} // namespace tensorrt_llm::runtime
