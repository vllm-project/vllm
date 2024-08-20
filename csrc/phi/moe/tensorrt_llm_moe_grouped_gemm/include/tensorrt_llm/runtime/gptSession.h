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

/*****************************************************************************
 *
 * GptSession is going to be deprecated soon.
 * Please do not add new functionality in this file!
 *
 *****************************************************************************/

#pragma once

#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/generationInput.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/samplingConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <NvInferRuntime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tensorrt_llm::batch_manager
{
class TrtGptModelV1;
}

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class KVCacheManager;
}

namespace tensorrt_llm::runtime
{

namespace utils
{
std::vector<uint8_t> loadEngine(std::string const& enginePath);
}

class AllReduceBuffers;
class IStatefulGptDecoder;
class NcclCommunicator;
class RuntimeBuffers;
class TllmRuntime;

class [[deprecated("Use the executor API instead.")]] GptSession
{
    using KvCacheManager = batch_manager::kv_cache_manager::KVCacheManager;
    using KvCacheConfig = batch_manager::kv_cache_manager::KvCacheConfig;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TokenGeneratedCallback = std::function<void(SizeType32 step, bool finished)>;

public:
    using LoggerPtr = std::shared_ptr<nvinfer1::ILogger>;

    //! @brief   Configuration for session execution and buffer sizes.
    //!          `generate` may be called with batch size and beam width smaller than the configured parameters.
    //! @details `maxBatchSize` will be divided by the number of micro batches to initialize each batch buffer.
    class Config
    {
    public:
        Config(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxSequenceLength,
            float gpuWeightsPercent = 1.0)
            : maxBatchSize{maxBatchSize}
            , maxBeamWidth{maxBeamWidth}
            , maxSequenceLength{maxSequenceLength}
            , gpuWeightsPercent{gpuWeightsPercent}
        {
        }

        // The maximum number of sequences in a batch
        SizeType32 maxBatchSize;
        // The maximum width of the beams in beam-search
        SizeType32 maxBeamWidth;
        // The length of the longest input sequence
        SizeType32 maxSequenceLength;
        // Percentage of weights on the gpu at runtime
        float gpuWeightsPercent;
        // Whether the session will use a different decoder per request.
        // It must be set to `true` when running in-flight batching
        bool decoderPerRequest{false};
        // Whether the session will use CUDA graphs for the engine   execution in generation phase
        bool cudaGraphMode{false};
        KvCacheConfig kvCacheConfig{};
        // The micro batch size to be used in context phase.
        // Batches entered in `GptSession::generation` will be split into smaller micro batches of this size
        std::optional<SizeType32> ctxMicroBatchSize = std::nullopt;
        // The micro batch size to be used in generation phase.
        // Batches entered in `GptSession::generation` will be split into smaller micro batches of this size.
        std::optional<SizeType32> genMicroBatchSize = std::nullopt;
        std::optional<executor::DecodingMode> decodingMode = std::nullopt;
        bool normalizeLogProbs = true;
    };

    //! @brief Optional profiler class to profile the generation phase of an inference request
    class GenerationProfiler
    {
    public:
        // Use a constexpr variable to resolve the ambiguous match for overloaded CudaEvent constructor
        static constexpr unsigned int flags{cudaEventDefault};

        GenerationProfiler()
            : start(flags)
            , end(flags)
        {
        }

        CudaEvent const& getStart() const
        {
            return start;
        }

        CudaEvent const& getEnd() const
        {
            return end;
        }

        float getElapsedTimeMs()
        {
            start.synchronize();
            end.synchronize();

            float result;
            TLLM_CUDA_CHECK(::cudaEventElapsedTime(&result, start.get(), end.get()));

            return result;
        }

    private:
        CudaEvent start;
        CudaEvent end;
    };

    //! @param sessionConfig Configuration of the session,
    //! @param modelConfig   Description of the model,
    //! @param worldConfig   Description of the environment,
    //! @param engineBuffer  The compiled TensorRT engine (const void*),
    //! @param engineSize    The size in bytes of the TensorRT engine (size_t),
    //! @param logger        The optional logger.
    GptSession(Config const& sessionConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        void const* engineBuffer, std::size_t engineSize, LoggerPtr logger = nullptr);

    GptSession(Config const& sessionConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::vector<uint8_t> const& engineBuffer, LoggerPtr logger = nullptr)
        : GptSession(
            sessionConfig, modelConfig, worldConfig, engineBuffer.data(), engineBuffer.size(), std::move(logger))
    {
    }

    GptSession(Config const& sessionConfig, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        std::string const& engineFile, LoggerPtr logger = nullptr)
        : GptSession(sessionConfig, modelConfig, worldConfig, utils::loadEngine(engineFile), std::move(logger))
    {
    }

    [[nodiscard]] nvinfer1::ILogger& getLogger() const;

    [[nodiscard]] BufferManager const& getBufferManager() const;

    [[nodiscard]] ModelConfig const& getModelConfig() const
    {
        return mModelConfig;
    }

    [[nodiscard]] WorldConfig const& getWorldConfig() const
    {
        return mWorldConfig;
    }

    [[nodiscard]] int getDevice() const noexcept
    {
        return mDevice;
    }

    [[nodiscard]] bool getNormalizeLogProbs() const noexcept
    {
        return mNormalizeLogProbs;
    }

    [[nodiscard]] nvinfer1::IEngineInspector& getEngineInspector() const;

    [[nodiscard]] nvinfer1::DataType getLogitDataType() const;

    //! @brief This function performs the generation loop.
    //! @details Given input tensors to read from, output tensors to populate, that member function
    //!          can be produced or each sequence has reached completion (due to the production
    //!          will run the generation loop until it reaches the maximum number of tokens that
    //!          of "end-of-sequence" or a word in the list of "stop words"). The pseudo-code of
    //!          that function looks like (member function names were changed to keep the
    //!          presentation simple):
    //!
    //!    ```cpp
    //!    // Have all the sequences in the batch reached completion?
    //!    bool allFinished = false;
    //!
    //!    // Until all sequences are finished or the number of steps reaches the limit...
    //!    for (int step = 0; !allFinished && step < maxNewTokens; ++step) {
    //!
    //!    // Trigger the computation of the logits...
    //!    computeLogits(...);
    //!
    //!    // Run the sampling to produce a token (for each active sequence) from the logits.
    //!    allFinished = generateTokensFromLogits(...);
    //!
    //!    // Callback to stream the output tokens while the generation loop continues.
    //!    onTokenGenerated(...);
    //!    }
    //!    ```
    void generate(GenerationOutput& outputs, GenerationInput const& inputs, SamplingConfig const& samplingConfig,
        std::shared_ptr<GenerationProfiler> const generationProfiler = nullptr);

    //! @brief Set LayerProfiler to collect performance per layer.
    void setLayerProfiler();

    //! @brief Print profile information per layer.
    [[nodiscard]] std::string getLayerProfileInfo() const;

private:
    [[nodiscard]] bool useCudaGraphs()
    {
        return !mCudaGraphInstances.empty();
    }

    void generateBatched(std::vector<GenerationOutput>& microBatchesOutputs,
        std::vector<GenerationInput> const& microBatchesInputs, SamplingConfig const& samplingConfig,
        TokenGeneratedCallback const& onTokenGenerated, std::shared_ptr<GenerationProfiler> const generationProfiler);

    void setup(Config const& sessionConfig);

    void createContexts();
    void createBuffers(SizeType32 numMicroBatches);
    void createDecoders(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxAttentionWindow,
        SizeType32 sinkTokenLength, SizeType32 maxSequenceLength, nvinfer1::DataType logitsType, bool decoderPerRequest,
        SizeType32 numMicroBatches, executor::DecodingMode const& decodingMode);
    void createKvCacheManager(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxAttentionWindow,
        SizeType32 sinkTokenLength, SizeType32 maxSequenceLength, KvCacheConfig const& config);
    void createCustomAllReduceWorkspace(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxSequenceLength);

    void executeContextStep(std::vector<GenerationInput> const& generationBatchesInputs,
        std::vector<SizeType32> const& generationBatchesOffsets, KvCacheManager const* kvCacheManager);
    SizeType32 executeGenerationStep(SizeType32 step, std::vector<GenerationInput> const& microBatchesInputs,
        std::vector<GenerationOutput>& microBatchesOutputs, std::vector<SizeType32> const& microBatchOffsets,
        KvCacheManager* kvCacheManager, std::vector<bool>& microBatchesFinished);

    //! @brief Execute decoder on last PP rank, receive decoder output on other PP ranks.
    void decoderStepAsync(SizeType32 decoderStep, SizeType32 microBatchId);

    //! @brief Synchronize with the decoder and return the `shouldStop` flag.
    bool shouldStopSync(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 microBatchId);

    //! @brief Collect final output ids and log probs on last PP rank and send them to first PP rank.
    //! @details Receives are asynchronous on host, so synchronization is required before access.
    void finalize(SizeType32 microBatchId);

    void kvCacheAddSequences(SizeType32 beamWidth, SizeType32 microBatchId, SizeType32 firstBatchIdx);

    //! @brief Populate outputIds and return reference to newTokens tensor
    ITensor::SharedPtr initDecoder(ITensor& outputIds, GenerationInput const& inputs, GenerationOutput const& outputs,
        SamplingConfig const& samplingConfig, SizeType32 microBatchId) const;

    TokenGeneratedCallback createOnTokenGeneratedCallback(GenerationOutput& outputs);

    class CudaGraphExecutor
    {
    public:
        CudaGraphExecutor() = default;

        ~CudaGraphExecutor()
        {
            try
            {
                clear();
            }
            catch (std::exception& e)
            {
                TLLM_LOG_EXCEPTION(e);
            }
        }

        bool hasInstance()
        {
            return mInstance != nullptr;
        }

        void clear();
        void prepareNextGraph(TllmRuntime const& runtime, SizeType32 nextContextId);
        void launch(CudaStream const& stream);

    private:
        void create(cudaGraph_t const& graph);
        bool update(cudaGraph_t const& graph);
        void uploadToStream(CudaStream const& stream);

        cudaGraphExec_t mInstance;
    };

    class MicroBatchConfig
    {
    public:
        MicroBatchConfig()
            : numCtxBatches{1}
            , numGenBatches{1}
            , ctxBatchSize{0}
            , genBatchSize{0}
        {
        }

        explicit MicroBatchConfig(SizeType32 maxBatchSize, SizeType32 pipelineParallelism,
            std::optional<SizeType32> genMicroBatchSize, std::optional<SizeType32> ctxMicroBatchSize);

        constexpr SizeType32 numCtxPerGen() const
        {
            return numCtxBatches / numGenBatches;
        }

        //! @details flip-flop between 2 graph instances for each generation batch.
        constexpr SizeType32 getGenGraphId(SizeType32 flipFlopId, SizeType32 generationBatchId) const
        {
            return flipFlopId * numGenBatches + generationBatchId;
        }

        SizeType32 numCtxBatches;
        SizeType32 numGenBatches;
        SizeType32 ctxBatchSize;
        SizeType32 genBatchSize;
    };

    friend class batch_manager::TrtGptModelV1;

private:
    ModelConfig const mModelConfig;
    WorldConfig const mWorldConfig;
    int mDevice{-1};
    std::shared_ptr<NcclCommunicator> mPipelineComm;
    std::shared_ptr<CudaStream> mCommStream;
    CudaEvent mCommEvent{};

    std::shared_ptr<AllReduceBuffers> mAllReduceBuffers;

    SizeType32 mDecoderMaxSequenceLength{};
    SizeType32 mDecoderMaxAttentionWindow{};
    SizeType32 mDecoderSinkTokenLength{};

    LoggerPtr mLogger;
    std::shared_ptr<TllmRuntime> mRuntime;
    std::shared_ptr<KvCacheManager> mKvCacheManager;

    MicroBatchConfig mMicroBatchConfig;
    // for each micro batch
    std::vector<std::shared_ptr<IStatefulGptDecoder>> mDecoders;
    std::vector<std::shared_ptr<RuntimeBuffers>> mBuffers;
    std::vector<CudaEvent> mReceivedEvents;

    bool mCudaGraphMode{false};
    // ping-pong instances
    std::vector<CudaGraphExecutor> mCudaGraphInstances;

    bool mNormalizeLogProbs = true;
};

} // namespace tensorrt_llm::runtime
