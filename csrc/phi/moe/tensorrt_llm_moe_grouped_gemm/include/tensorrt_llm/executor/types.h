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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace tensorrt_llm::runtime
{
class CudaStream;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::executor
{

class Request;
class Tensor;

using TensorPtr = std::shared_ptr<Tensor>;
using SizeType32 = std::int32_t;
using FloatType = float;
using TokenIdType = std::int32_t;
using VecTokens = std::vector<TokenIdType>;
using BeamTokens = std::vector<VecTokens>;
using IdType = std::uint64_t;
using IterationType = std::uint64_t;
using RandomSeedType = std::uint64_t;
using VecLogProbs = std::vector<FloatType>;
using StreamPtr = std::shared_ptr<tensorrt_llm::runtime::CudaStream>;
using LogitsPostProcessor = std::function<void(IdType, Tensor&, BeamTokens const&, StreamPtr&)>;
using LogitsPostProcessorMap = std::unordered_map<std::string, LogitsPostProcessor>;
using MedusaChoices = std::vector<std::vector<SizeType32>>;

enum class DataType
{
    kBOOL,
    kUINT8,
    kINT8,
    kINT32,
    kINT64,
    kBF16,
    kFP8,
    kFP16,
    kFP32,
    kUNKNOWN
};

//! \brief For converting a C++ data type to a `TrtLmmDataType`.
template <typename T, bool = false>
struct TypeTraits
{
};

template <>
struct TypeTraits<float>
{
    static constexpr auto value = DataType::kFP32;
};

template <>
struct TypeTraits<half>
{
    static constexpr auto value = DataType::kFP16;
};

template <>
struct TypeTraits<std::int8_t>
{
    static constexpr auto value = DataType::kINT8;
};

template <>
struct TypeTraits<std::int32_t>
{
    static constexpr auto value = DataType::kINT32;
};

template <>
struct TypeTraits<std::int64_t>
{
    static constexpr auto value = DataType::kINT64;
};

template <>
struct TypeTraits<bool>
{
    static constexpr auto value = DataType::kBOOL;
};

template <>
struct TypeTraits<std::uint8_t>
{
    static constexpr auto value = DataType::kUINT8;
};

#ifdef ENABLE_BF16
template <>
struct TypeTraits<__nv_bfloat16>
{
    static constexpr auto value = DataType::kBF16;
};
#endif

#ifdef ENABLE_FP8
template <>
struct TypeTraits<__nv_fp8_e4m3>
{
    static constexpr auto value = DataType::kFP8;
};
#endif

template <typename T>
struct TypeTraits<T*>
{
    // Pointers are stored as int64_t.
    static constexpr auto value = DataType::kINT64;
};

enum class MemoryType
{
    kCPU,
    kCPU_PINNED,
    kGPU,
    kUVM,
    kUNKNOWN
};

enum class ModelType
{
    kDECODER_ONLY = 0,
    kENCODER_ONLY = 1,
    kENCODER_DECODER = 2,
};

/// @brief The batching type
enum class BatchingType
{
    /// @brief STATIC refers to the traditional batching scheme with a batch of requests running in lockstep until the
    /// full generation for all of them is complete. Requests in a batch are all padded up to the maximum input and
    /// output sequence length of any member of the batch.
    kSTATIC = 0,

    /// @brief INFLIGHT refers to a scheme where newly arrived requests are dynamically incorporated into the batch
    /// under execution, and requests are returned as soon as the end condition is met without any padding.
    kINFLIGHT = 1,
};

/// @brief The policy used to select the subset of available requests in each iteration of the executor generation loop
enum class CapacitySchedulerPolicy
{
    /// @brief MAX_UTILIZATION packs as many requests as the underlying TRT engine can support in any iteration of the
    /// InflightBatching generation loop. While this is expected to maximize GPU throughput, it might require that some
    /// requests be paused and restarted depending on peak KV cache memory availability.
    kMAX_UTILIZATION = 0,

    /// @brief GUARANTEED_NO_EVICT uses KV cache more conservatively guaranteeing that a request, once started, will run
    /// to completion without eviction.
    kGUARANTEED_NO_EVICT = 1,
};

std::ostream& operator<<(std::ostream& os, CapacitySchedulerPolicy policy);

enum class ContextChunkingPolicy
{
    /// @brief Sequential chunking, complete the unfinished context phase first.
    kFIRST_COME_FIRST_SERVED = 0,

    /// @brief Iterate through each context request in sequence and attempt to increase its chunk
    /// count until the constraint is exceeded.
    kEQUAL_PROGRESS = 1,
};

std::ostream& operator<<(std::ostream& os, ContextChunkingPolicy policy);

enum class CommunicationType
{
    kMPI = 0
};

enum class CommunicationMode
{
    kLEADER, // With the leader mode, only the leader can enqueue requests. The requests will be
             // broadcasted to the workers. All participants can get response via awaitResponses. The leader is the
             // first participant in the provided participant IDS, or 0 if participant ID is not provided
    kORCHESTRATOR, // With the orchestrator mode, only the orchestrator can enqueue requests and await responses. The
                   // requests will be broadcasted to the workers. The orchestrator will spawn new processes for the
                   // execution of the model
};

/// @brief Struct that holds the stats of a KV cache manager
struct KvCacheStats
{
    /// @brief Max number of blocks
    SizeType32 maxNumBlocks;
    /// @brief Number of free blocks
    SizeType32 freeNumBlocks;
    /// @brief Number of used blocks
    SizeType32 usedNumBlocks;
    /// @brief Number of tokens per block
    SizeType32 tokensPerBlock;
};

/// @brief Struct that holds the stats of static batching models for a single iteration
struct StaticBatchingStats
{
    /// @brief Number of scheduled requests
    SizeType32 numScheduledRequests;
    /// @brief Number of requests in context stage
    SizeType32 numContextRequests;
    /// @brief Total number of context tokens in the iteration
    SizeType32 numCtxTokens;
    /// @brief Total number of tokens to generate in the iteration
    SizeType32 numGenTokens;
    /// @brief Total number of unused generation token slots
    SizeType32 emptyGenSlots;
};

/// @brief Struct that holds the stats of inflight batching models for a single iteration
struct InflightBatchingStats
{
    /// @brief Number of scheduled requests
    SizeType32 numScheduledRequests;
    /// @brief Number of requests in context stage
    SizeType32 numContextRequests;
    /// @brief Number of requests in generation stage
    SizeType32 numGenRequests;
    /// @brief Number of paused requests
    SizeType32 numPausedRequests;
    /// @brief Total number of context tokens in the iteration
    SizeType32 numCtxTokens;
    /// @brief Index of mirco batch
    SizeType32 microBatchId;
    /// @brief Average number of tokens decoded per request per iteration
    float avgNumDecodedTokensPerIter;
};

/// @brief Struct that holds the stats of a single iteration
struct IterationStats
{
    /// @brief Ending time of this iteration
    std::string timestamp;
    /// @brief Iteration id
    IterationType iter;
    /// @brief Number of active requests
    SizeType32 numActiveRequests;
    /// @brief Number of max active requests
    SizeType32 maxNumActiveRequests;
    /// @brief GPU memory usage in bytes
    size_t gpuMemUsage;
    /// @brief CPU memory usage in bytes
    size_t cpuMemUsage;
    /// @brief Pinned memory usage in bytes
    size_t pinnedMemUsage;
    /// @brief Stats specific to KV caches
    std::optional<KvCacheStats> kvCacheStats;
    /// @brief Stats specific to cross KV caches
    std::optional<KvCacheStats> crossKvCacheStats;
    /// @brief Stats specific to static batching
    std::optional<StaticBatchingStats> staticBatchingStats;
    /// @brief Stats specific to inflight batching
    std::optional<InflightBatchingStats> inflightBatchingStats;
};

/// @brief Enum class that represents the state of a request
enum class RequestStage
{
    /// @brief Request that have been received but not yet included in the active requests (due to constraints such as
    /// maximum batch size for example).
    kQUEUED,
    /// @brief Active request in encoder phase
    kENCODER_IN_PROGRESS,
    /// @brief Active request in context phase
    kCONTEXT_IN_PROGRESS,
    /// @brief Active request in generation phase
    kGENERATION_IN_PROGRESS,
    /// @brief Active request for which generation has completed
    kGENERATION_COMPLETE,
};

/// @brief Struct that holds the stats of a single request
struct RequestStats
{
    /// @brief The request id
    IdType id;
    /// @brief The current stage the request is in
    RequestStage stage;
    /// @brief If using chunked context, the current context prefill position
    SizeType32 contextPrefillPosition;
    /// @brief The number of generated tokens so far
    SizeType32 numGeneratedTokens;
    /// @brief The average number of decoded tokens per iteration. It is >= 1 for speculative decoding.
    float avgNumDecodedTokensPerIter;
    /// @brief Whether the request is scheduled for the current iteration
    bool scheduled;
    /// @brief Whether the request is being paused at the current iteration due to lack of resources (KV cache blocks
    /// exhaustion for example)
    bool paused;
};

/// @brief Struct that holds the stats of all requests in an iteration
struct RequestStatsPerIteration
{
    /// @brief The iteration id for these stats
    IterationType iter;
    /// @brief The stats of all active requests for this iteration
    std::vector<RequestStats> requestStats;
};

/// @brief mode of the decoder
class DecodingMode
{
public:
    /// @brief No mode specified. Config will be determined from the beam width of the first request at runtime
    /// TopKTopP if beamWidth == 1, BeamSearch otherwise
    static auto constexpr Auto()
    {
        return DecodingMode{kAuto};
    }

    static auto constexpr TopK()
    {
        return DecodingMode{kTopK | kUsePenalties | kUseBanTokens | kStandardStopCriteria};
    }

    static auto constexpr TopP()
    {
        return DecodingMode{kTopP | kUsePenalties | kUseBanTokens | kStandardStopCriteria};
    }

    static auto constexpr TopKTopP()
    {
        return DecodingMode{kTopKTopP | kUsePenalties | kUseBanTokens | kStandardStopCriteria};
    }

    static auto constexpr BeamSearch()
    {
        return DecodingMode{kBeamSearch | kUsePenalties | kUseBanTokens | kStandardStopCriteria};
    }

    static auto constexpr Medusa()
    {
        return DecodingMode{kMedusa | kUseMinLength | kUseMaxLengthStop};
    }

    static auto constexpr Lookahead()
    {
        return DecodingMode{kLookahead | kUseMinLength | kUseMaxLengthStop};
    }

    static auto constexpr ExplicitDraftTokens()
    {
        return DecodingMode{kExplicitDraftTokens | kUseMaxLengthStop | kUseExplicitEosStop};
    }

    auto constexpr useTemperature(bool useTemp)
    {
        mState = setBitTo(kUseTemperature, useTemp);
        return *this;
    }

    auto constexpr useOccurrencePenalties(bool usePenalty)
    {
        mState = setBitTo(kUseOccurrencePenalties, usePenalty);
        return *this;
    }

    auto constexpr usePresencePenalty(bool usePenalty)
    {
        mState = setBitTo(kUsePresencePenalties, usePenalty);
        return *this;
    }

    auto constexpr useRepetitionPenalty(bool usePenalty)
    {
        mState = setBitTo(kUseRepetitionPenalties, usePenalty);
        return *this;
    }

    auto constexpr useFrequencyPenalty(bool usePenalty)
    {
        mState = setBitTo(kUseFrequencyPenalties, usePenalty);
        return *this;
    }

    auto constexpr useMinLength(bool useMinLen)
    {
        mState = setBitTo(kUseMinLength, useMinLen);
        return *this;
    }

    auto constexpr useBanTokens(bool banTokens)
    {
        mState = setBitTo(kUseBanTokens, banTokens);
        return *this;
    }

    auto constexpr useBanWords(bool banWords)
    {
        mState = setBitTo(kUseBanWords, banWords);
        return *this;
    }

    auto constexpr useNoRepeatNgramSize(bool noRepeatNgramSize)
    {
        mState = setBitTo(kUseNoRepeatNgramSize, noRepeatNgramSize);
        return *this;
    }

    auto constexpr useStopWords(bool stopWords)
    {
        mState = setBitTo(kUseStopWords, stopWords);
        return *this;
    }

    auto constexpr useMaxLengthStop(bool maxLengthStop)
    {
        mState = setBitTo(kUseMaxLengthStop, maxLengthStop);
        return *this;
    }

    auto constexpr useExplicitEosStop(bool explicitEosStop)
    {
        mState = setBitTo(kUseExplicitEosStop, explicitEosStop);
        return *this;
    }

    bool constexpr isAuto() const
    {
        return anyBitSet(kAuto);
    }

    bool constexpr isTopK() const
    {
        return anyBitSet(kTopK);
    }

    bool constexpr isTopP() const
    {
        return anyBitSet(kTopP);
    }

    bool constexpr isTopKorTopP() const
    {
        return anyBitSet(kTopKTopP);
    }

    bool constexpr isTopKandTopP() const
    {
        return allBitSet(kTopKTopP);
    }

    bool constexpr isBeamSearch() const
    {
        return anyBitSet(kBeamSearch);
    }

    bool constexpr isMedusa() const
    {
        return anyBitSet(kMedusa);
    }

    bool constexpr isLookahead() const
    {
        return anyBitSet(kLookahead);
    }

    bool constexpr isExplicitDraftTokens() const
    {
        return anyBitSet(kExplicitDraftTokens);
    }

    bool constexpr isUseTemperature() const
    {
        return anyBitSet(kUseTemperature);
    }

    bool constexpr isUsePresencePenalty() const
    {
        return anyBitSet(kUsePresencePenalties);
    }

    bool constexpr isUseFrequencyPenalty() const
    {
        return anyBitSet(kUseFrequencyPenalties);
    }

    bool constexpr isUseRepetitionPenalty() const
    {
        return anyBitSet(kUseRepetitionPenalties);
    }

    bool constexpr isUseMinLength() const
    {
        return anyBitSet(kUseMinLength);
    }

    bool constexpr isUseOccurrencePenalty() const
    {
        return anyBitSet(kUseOccurrencePenalties);
    }

    bool constexpr isUsePenalty() const
    {
        return anyBitSet(kUsePenalties);
    }

    bool constexpr isUseBanWords() const
    {
        return anyBitSet(kUseBanWords);
    }

    bool constexpr isUseNoRepeatNgramSize() const
    {
        return anyBitSet(kUseNoRepeatNgramSize);
    }

    bool constexpr isUseBanTokens() const
    {
        return anyBitSet(kUseBanTokens);
    }

    bool constexpr isUseStopWords() const
    {
        return anyBitSet(kUseStopWords);
    }

    bool constexpr isUseMaxLengthStop() const
    {
        return anyBitSet(kUseMaxLengthStop);
    }

    bool constexpr isUseExplicitEosStop() const
    {
        return anyBitSet(kUseExplicitEosStop);
    }

    bool constexpr isUseStopCriteria() const
    {
        return anyBitSet(kStandardStopCriteria | kUseExplicitEosStop);
    }

    using UnderlyingType = uint32_t;

    bool operator==(DecodingMode const& other) const
    {
        return mState == other.mState;
    }

    constexpr DecodingMode(UnderlyingType state)
        : mState(state)
    {
    }

    constexpr UnderlyingType getState() const
    {
        return mState;
    }

private:
    // No mode specified. Config will be determined from the beam width of the first request at runtime
    // TopKTopP if beamWidth == 1, BeamSearch otherwise
    static UnderlyingType constexpr kUseRepetitionPenalties{1u << 0};
    static UnderlyingType constexpr kUseFrequencyPenalties{1u << 1};
    static UnderlyingType constexpr kUsePresencePenalties{1u << 2};
    static UnderlyingType constexpr kUseTemperature{1u << 3};
    static UnderlyingType constexpr kUseMinLength{1u << 4};
    static UnderlyingType constexpr kUseBanWords{1u << 5};
    static UnderlyingType constexpr kUseStopWords{1u << 6};
    static UnderlyingType constexpr kUseMaxLengthStop{1u << 7};
    static UnderlyingType constexpr kUseExplicitEosStop{1u << 8};
    static UnderlyingType constexpr kUseNoRepeatNgramSize{1u << 9};
    static UnderlyingType constexpr kStandardStopCriteria{kUseStopWords | kUseMaxLengthStop};
    static UnderlyingType constexpr kUseOccurrencePenalties{
        kUseRepetitionPenalties | kUseFrequencyPenalties | kUsePresencePenalties};
    static UnderlyingType constexpr kUsePenalties{kUseOccurrencePenalties | kUseTemperature | kUseMinLength};
    static UnderlyingType constexpr kUseBanTokens{kUseNoRepeatNgramSize | kUseBanWords};
    static SizeType32 constexpr kNumFlags{10};
    static UnderlyingType constexpr kAuto{1u << (kNumFlags + 0)};
    static UnderlyingType constexpr kTopK{1u << (kNumFlags + 1)};
    static UnderlyingType constexpr kTopP{1u << (kNumFlags + 2)};
    static UnderlyingType constexpr kBeamSearch{1u << (kNumFlags + 3)};
    static UnderlyingType constexpr kMedusa{1u << (kNumFlags + 4)};
    static UnderlyingType constexpr kLookahead{1u << (kNumFlags + 5)};
    static UnderlyingType constexpr kExplicitDraftTokens{1u << (kNumFlags + 6)};
    static UnderlyingType constexpr kTopKTopP{kTopK | kTopP};

    bool constexpr anyBitSet(UnderlyingType bits) const
    {
        return (mState & bits) != 0;
    }

    bool constexpr allBitSet(UnderlyingType bits) const
    {
        return (mState & bits) == bits;
    }

    UnderlyingType constexpr setBitTo(UnderlyingType state, bool x)
    {
        return (mState & (~state)) | (state * static_cast<UnderlyingType>(x));
    }

    UnderlyingType mState{};
};

static_assert(DecodingMode::Auto().isAuto());
static_assert(!DecodingMode::Auto().isUseBanWords());
static_assert(!DecodingMode::Auto().isUseOccurrencePenalty());
static_assert(!DecodingMode::Auto().isUseStopCriteria());
static_assert(!DecodingMode::Auto().isTopK());
static_assert(!DecodingMode::Auto().isTopP());
static_assert(!DecodingMode::Auto().isBeamSearch());
static_assert(!DecodingMode::Auto().isMedusa());
static_assert(!DecodingMode::Auto().isLookahead());
static_assert(!DecodingMode::Auto().isExplicitDraftTokens());

static_assert(DecodingMode::TopK().isTopK());
static_assert(DecodingMode::TopK().isTopKorTopP());
static_assert(DecodingMode::TopK().isUseBanWords());
static_assert(DecodingMode::TopK().isUseOccurrencePenalty());
static_assert(DecodingMode::TopK().isUseStopCriteria());
static_assert(!DecodingMode::TopK().useRepetitionPenalty(false).isUseRepetitionPenalty());
static_assert(DecodingMode::TopK().useRepetitionPenalty(false).isUseOccurrencePenalty());
static_assert(!DecodingMode::TopK()
                   .useRepetitionPenalty(false)
                   .usePresencePenalty(false)
                   .useFrequencyPenalty(false)
                   .isUseOccurrencePenalty());
static_assert(!DecodingMode::TopK().isTopKandTopP());
static_assert(!DecodingMode::TopK().isTopP());
static_assert(!DecodingMode::TopK().isBeamSearch());
static_assert(!DecodingMode::TopK().isMedusa());
static_assert(!DecodingMode::TopK().isLookahead());
static_assert(!DecodingMode::TopK().isAuto());
static_assert(!DecodingMode::TopK().isExplicitDraftTokens());

static_assert(DecodingMode::TopP().isTopP());
static_assert(DecodingMode::TopP().isTopKorTopP());
static_assert(DecodingMode::TopP().isUseBanWords());
static_assert(DecodingMode::TopP().isUseOccurrencePenalty());
static_assert(DecodingMode::TopP().isUseStopCriteria());
static_assert(!DecodingMode::TopP().isTopKandTopP());
static_assert(!DecodingMode::TopP().isTopK());
static_assert(!DecodingMode::TopP().isBeamSearch());
static_assert(!DecodingMode::TopP().isMedusa());
static_assert(!DecodingMode::TopP().isLookahead());
static_assert(!DecodingMode::TopP().isAuto());
static_assert(!DecodingMode::TopP().isExplicitDraftTokens());

static_assert(DecodingMode::TopKTopP().isTopK());
static_assert(DecodingMode::TopKTopP().isTopP());
static_assert(DecodingMode::TopKTopP().isTopKorTopP());
static_assert(DecodingMode::TopKTopP().isTopKandTopP());
static_assert(DecodingMode::TopKTopP().isUseBanWords());
static_assert(DecodingMode::TopKTopP().isUseOccurrencePenalty());
static_assert(DecodingMode::TopKTopP().isUseStopCriteria());
static_assert(!DecodingMode::TopKTopP().isBeamSearch());
static_assert(!DecodingMode::TopKTopP().isMedusa());
static_assert(!DecodingMode::TopKTopP().isLookahead());
static_assert(!DecodingMode::TopKTopP().isAuto());
static_assert(!DecodingMode::TopKTopP().isExplicitDraftTokens());

static_assert(DecodingMode::BeamSearch().isBeamSearch());
static_assert(DecodingMode::BeamSearch().isUseStopCriteria());
static_assert(!DecodingMode::BeamSearch().isTopKorTopP());
static_assert(!DecodingMode::BeamSearch().isMedusa());
static_assert(!DecodingMode::BeamSearch().isLookahead());
static_assert(!DecodingMode::BeamSearch().isAuto());
static_assert(!DecodingMode::BeamSearch().isExplicitDraftTokens());

static_assert(!DecodingMode::Medusa().isTopK());
static_assert(!DecodingMode::Medusa().isTopKorTopP());
static_assert(!DecodingMode::Medusa().isTopKandTopP());
static_assert(!DecodingMode::Medusa().isTopP());
static_assert(!DecodingMode::Medusa().isBeamSearch());
static_assert(!DecodingMode::Medusa().isLookahead());
static_assert(!DecodingMode::Medusa().isAuto());
static_assert(!DecodingMode::Medusa().isUseBanWords());
static_assert(!DecodingMode::Medusa().isUseOccurrencePenalty());
static_assert(!DecodingMode::Medusa().isExplicitDraftTokens());
static_assert(DecodingMode::Medusa().isUseStopCriteria());
static_assert(!DecodingMode::Medusa().isUseStopWords());
static_assert(!DecodingMode::Medusa().isUseExplicitEosStop());
static_assert(DecodingMode::Medusa().isUsePenalty());
static_assert(DecodingMode::Medusa().isUseMinLength());
static_assert(DecodingMode::Medusa().isMedusa());

static_assert(!DecodingMode::Lookahead().isAuto());
static_assert(!DecodingMode::Lookahead().isTopK());
static_assert(!DecodingMode::Lookahead().isTopKorTopP());
static_assert(!DecodingMode::Lookahead().isTopKandTopP());
static_assert(!DecodingMode::Lookahead().isTopP());
static_assert(!DecodingMode::Lookahead().isBeamSearch());
static_assert(!DecodingMode::Lookahead().isMedusa());
static_assert(!DecodingMode::Lookahead().isExplicitDraftTokens());
static_assert(DecodingMode::Lookahead().isUseStopCriteria());
static_assert(!DecodingMode::Lookahead().isUseStopWords());
static_assert(!DecodingMode::Lookahead().isUseExplicitEosStop());
static_assert(DecodingMode::Lookahead().isLookahead());

static_assert(!DecodingMode::ExplicitDraftTokens().isAuto());
static_assert(!DecodingMode::ExplicitDraftTokens().isTopK());
static_assert(!DecodingMode::ExplicitDraftTokens().isTopKorTopP());
static_assert(!DecodingMode::ExplicitDraftTokens().isTopKandTopP());
static_assert(!DecodingMode::ExplicitDraftTokens().isTopP());
static_assert(!DecodingMode::ExplicitDraftTokens().isBeamSearch());
static_assert(!DecodingMode::ExplicitDraftTokens().isMedusa());
static_assert(!DecodingMode::ExplicitDraftTokens().isLookahead());
static_assert(!DecodingMode::ExplicitDraftTokens().isUsePenalty());
static_assert(DecodingMode::ExplicitDraftTokens().isUseStopCriteria());
static_assert(DecodingMode::ExplicitDraftTokens().isUseMaxLengthStop());
static_assert(DecodingMode::ExplicitDraftTokens().isUseExplicitEosStop());
static_assert(!DecodingMode::ExplicitDraftTokens().isUseStopWords());
static_assert(!DecodingMode::ExplicitDraftTokens().isUseBanWords());
static_assert(DecodingMode::ExplicitDraftTokens().isExplicitDraftTokens());
} // namespace tensorrt_llm::executor
