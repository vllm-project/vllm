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

#include "tensorrt_llm/batch_manager/BatchManager.h"
#include "tensorrt_llm/batch_manager/callbacks.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/trtGptModelOptionalParams.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <atomic>
#include <filesystem>
#include <optional>

namespace nvinfer1
{
class ILogger;
}

namespace tensorrt_llm::batch_manager
{

class InferenceRequest;
class TrtGptModel;

/* Responsible for shepherding requests through to completion
   using TRT Backend. */
class GptManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TokenIdType = tensorrt_llm::runtime::TokenIdType;
    using RequestList = std::list<std::shared_ptr<LlmRequest>>;
    using TensorPtr = runtime::ITensor::SharedPtr;

    GptManager(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType, SizeType32 maxBeamWidth,
        executor::SchedulerConfig const& schedulerConfig, GetInferenceRequestsCallback getInferenceRequestsCb,
        SendResponseCallback sendResponseCb, PollStopSignalCallback pollStopSignalCb = nullptr,
        ReturnBatchManagerStatsCallback returnBatchManagerStatsCb = nullptr,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams(),
        std::optional<uint64_t> terminateReqId = std::nullopt, bool excludeInputInOutput = false);

    /* Wraps the user-provided callback for requests.
       Adds requests to request table.
       Invoked every generation loop iteration. */
    BatchManagerErrorCode_t fetchNewRequests();

    /* Returns completed requests.
       Deletes entry from activeRequests */
    BatchManagerErrorCode_t returnCompletedRequests();

    BatchManagerErrorCode_t pollStopSignals();

    BatchManagerErrorCode_t returnBatchManagerStats();

    BatchManagerErrorCode_t waitUntilTerminate();

    BatchManagerErrorCode_t shutdown();

    SizeType32 getNumActiveRequests();

    virtual ~GptManager();

    void setLayerProfiler();

    [[nodiscard]] std::string getLayerProfileInfo() const;

protected:
    /* Synchronizes the decoder */
    virtual BatchManagerErrorCode_t forwardSync();

    /* Invokes one step of backend
       Updates state of all requests */
    virtual BatchManagerErrorCode_t forwardAsync(
        RequestList& activeRequests, std::unordered_set<uint64_t>& activeRequestsIds);

private:
    [[nodiscard]] SizeType32 getMaxInputLen() const;
    [[nodiscard]] SizeType32 getMaxSequenceLen() const;
    [[nodiscard]] SizeType32 getMaxNumSequences() const;
    [[nodiscard]] SizeType32 getMaxDraftLen() const;

    void validateLlmRequest(
        LlmRequest& newReq, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig) const;
    static std::shared_ptr<LlmRequest> fillLlmRequest(std::shared_ptr<InferenceRequest> newReq);
    static std::shared_ptr<std::vector<TokenIdType>> getReqInputTokens(std::shared_ptr<InferenceRequest> newReq);
    static SizeType32 getMaxNewTokens(std::shared_ptr<InferenceRequest> newReq);

    GetInferenceRequestsCallback mGetInferenceRequestsCb;
    SendResponseCallback mSendResponseCb;
    PollStopSignalCallback mPollStopSignalCb;
    ReturnBatchManagerStatsCallback mReturnBatchManagerStatsCb;

    std::shared_ptr<TrtGptModel> mTrtGptModel;
    std::optional<uint64_t> mTerminateReqId;

    // Iteration counter - incremented every iteration of the generation loop
    int64_t mIterationCounter;
    // List of live requests
    RequestList mActiveRequests;
    // IDs of live requests
    std::unordered_set<uint64_t> mActiveRequestsIds;
    // Boolean that controls if prompt should be included in output tokens for non-streaming
    bool mExcludeInputInOutput;

    std::atomic<bool> shutdown_requested_;
    void decoupled_execution_loop();
    std::shared_ptr<std::thread> worker_thread_;
    std::shared_ptr<nvinfer1::ILogger> mLogger{};
};

} // namespace tensorrt_llm::batch_manager
