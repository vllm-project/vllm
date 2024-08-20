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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/executor/types.h"
#include <istream>
#include <ostream>

namespace tensorrt_llm::executor
{

class Serialization
{
public:
    // SamplingConfig
    [[nodiscard]] static SamplingConfig deserializeSamplingConfig(std::istream& is);
    static void serialize(SamplingConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(SamplingConfig const& config);

    // OutputConfig
    [[nodiscard]] static OutputConfig deserializeOutputConfig(std::istream& is);
    static void serialize(OutputConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(OutputConfig const& config);

    // ExternalDraftTokensConfig
    [[nodiscard]] static ExternalDraftTokensConfig deserializeExternalDraftTokensConfig(std::istream& is);
    static void serialize(ExternalDraftTokensConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(ExternalDraftTokensConfig const& config);

    // PromptTuningConfig
    [[nodiscard]] static PromptTuningConfig deserializePromptTuningConfig(std::istream& is);
    static void serialize(PromptTuningConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(PromptTuningConfig const& config);

    // LoraConfig
    [[nodiscard]] static LoraConfig deserializeLoraConfig(std::istream& is);
    static void serialize(LoraConfig const& config, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(LoraConfig const& config);

    // Request
    [[nodiscard]] static Request deserializeRequest(std::istream& is);
    static void serialize(Request const& request, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(Request const& request);

    // Tensor
    [[nodiscard]] static Tensor deserializeTensor(std::istream& is);
    static void serialize(Tensor const& tensor, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(Tensor const& tensor);

    // Result
    [[nodiscard]] static Result deserializeResult(std::istream& is);
    static void serialize(Result const& result, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(Result const& result);

    // Response
    [[nodiscard]] static Response deserializeResponse(std::istream& is);
    static void serialize(Response const& response, std::ostream& os);
    [[nodiscard]] static size_t serializedSize(Response const& response);

    // Vector of responses
    static std::vector<Response> deserializeResponses(std::vector<char>& buffer);
    static std::vector<char> serialize(std::vector<Response> const& responses);

    // KvCacheConfig
    static KvCacheConfig deserializeKvCacheConfig(std::istream& is);
    static void serialize(KvCacheConfig const& kvCacheConfig, std::ostream& os);
    static size_t serializedSize(KvCacheConfig const& kvCacheConfig);

    // SchedulerConfig
    static SchedulerConfig deserializeSchedulerConfig(std::istream& is);
    static void serialize(SchedulerConfig const& schedulerConfig, std::ostream& os);
    static size_t serializedSize(SchedulerConfig const& schedulerConfig);

    // ParallelConfig
    static ParallelConfig deserializeParallelConfig(std::istream& is);
    static void serialize(ParallelConfig const& parallelConfig, std::ostream& os);
    static size_t serializedSize(ParallelConfig const& parallelConfig);

    // PeftCacheConfig
    static PeftCacheConfig deserializePeftCacheConfig(std::istream& is);
    static void serialize(PeftCacheConfig const& peftCacheConfig, std::ostream& os);
    static size_t serializedSize(PeftCacheConfig const& peftCacheConfig);

    // OrchestratorConfig
    static OrchestratorConfig deserializeOrchestratorConfig(std::istream& is);
    static void serialize(OrchestratorConfig const& orchestratorConfig, std::ostream& os);
    static size_t serializedSize(OrchestratorConfig const& orchestratorConfig);

    // DecodingMode
    static DecodingMode deserializeDecodingMode(std::istream& is);
    static void serialize(DecodingMode const& decodingMode, std::ostream& os);
    static size_t serializedSize(DecodingMode const& decodingMode);

    // LookaheadDecodingConfig
    static LookaheadDecodingConfig deserializeLookaheadDecodingConfig(std::istream& is);
    static void serialize(LookaheadDecodingConfig const& lookaheadDecodingConfig, std::ostream& os);
    static size_t serializedSize(LookaheadDecodingConfig const& lookaheadDecodingConfig);

    // DecodingConfig
    static DecodingConfig deserializeDecodingConfig(std::istream& is);
    static void serialize(DecodingConfig const& decodingConfig, std::ostream& os);
    static size_t serializedSize(DecodingConfig const& decodingConfig);

    // ExecutorConfig
    static ExecutorConfig deserializeExecutorConfig(std::istream& is);
    static void serialize(ExecutorConfig const& executorConfig, std::ostream& os);
    static size_t serializedSize(ExecutorConfig const& executorConfig);

    // KvCacheStats
    static KvCacheStats deserializeKvCacheStats(std::istream& is);
    static void serialize(KvCacheStats const& kvCacheStats, std::ostream& os);
    static size_t serializedSize(KvCacheStats const& kvCacheStats);

    // StaticBatchingStats
    static StaticBatchingStats deserializeStaticBatchingStats(std::istream& is);
    static void serialize(StaticBatchingStats const& staticBatchingStats, std::ostream& os);
    static size_t serializedSize(StaticBatchingStats const& staticBatchingStats);

    // InflightBatchingStats
    static InflightBatchingStats deserializeInflightBatchingStats(std::istream& is);
    static void serialize(InflightBatchingStats const& inflightBatchingStats, std::ostream& os);
    static size_t serializedSize(InflightBatchingStats const& inflightBatchingStats);

    // IterationStats
    static IterationStats deserializeIterationStats(std::vector<char>& buffer);
    static IterationStats deserializeIterationStats(std::istream& is);
    static void serialize(IterationStats const& iterStats, std::ostream& os);
    static std::vector<char> serialize(IterationStats const& iterStats);
    static size_t serializedSize(IterationStats const& iterStats);

    // String
    static std::string deserializeString(std::istream& is);

    // ModelType
    static ModelType deserializeModelType(std::istream& is);
};

} // namespace tensorrt_llm::executor
