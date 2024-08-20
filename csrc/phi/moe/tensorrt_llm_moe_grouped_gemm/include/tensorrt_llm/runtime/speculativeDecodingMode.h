/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

namespace tensorrt_llm
{
namespace runtime
{

class SpeculativeDecodingMode
{
    // [WARNING] KEEP BELOW DEFINITION IN SYNC WITH tensorrt_llm/models/modeling_utils.py
public:
    static auto constexpr None()
    {
        return SpeculativeDecodingMode{kNone};
    }

    static auto constexpr DraftTokensExternal()
    {
        return SpeculativeDecodingMode{kDraftTokensExternal};
    }

    static auto constexpr Medusa()
    {
        return SpeculativeDecodingMode{kMedusa};
    }

    static auto constexpr LookaheadDecoding()
    {
        return SpeculativeDecodingMode{kLookaheadDecoding};
    }

    bool constexpr isNone() const
    {
        return anyBitSet(kNone);
    }

    bool constexpr isDraftTokensExternal() const
    {
        return anyBitSet(kDraftTokensExternal);
    }

    bool constexpr isMedusa() const
    {
        return anyBitSet(kMedusa);
    }

    bool constexpr isLookaheadDecoding() const
    {
        return anyBitSet(kLookaheadDecoding);
    }

    bool constexpr requiresAttentionMask() const
    {
        return anyBitSet(kLookaheadDecoding | kMedusa);
    }

    bool constexpr predictsDraftTokens() const
    {
        return anyBitSet(kLookaheadDecoding | kMedusa);
    }

    bool constexpr needsKVCacheRewind() const
    {
        return anyBitSet(kLookaheadDecoding | kMedusa);
    }

    bool constexpr hasDraftLogits() const
    {
        return anyBitSet(kMedusa);
    }

    using UnderlyingType = uint8_t;

    bool operator==(SpeculativeDecodingMode const& other) const
    {
        return mState == other.mState;
    }

    constexpr SpeculativeDecodingMode(UnderlyingType state)
        : mState(state)
    {
    }

private:
    // No speculative decoding is used.
    static UnderlyingType constexpr kNone{1u << 0};
    static UnderlyingType constexpr kDraftTokensExternal{1u << 1};
    static UnderlyingType constexpr kMedusa{1u << 2};
    static UnderlyingType constexpr kLookaheadDecoding{1u << 3};

    bool constexpr anyBitSet(UnderlyingType bits) const
    {
        return (mState & bits) != 0;
    }

    bool constexpr allBitSet(UnderlyingType bits) const
    {
        return (mState & bits) == bits;
    }

    UnderlyingType mState{kNone};
};

static_assert(SpeculativeDecodingMode::None().isNone());
static_assert(!SpeculativeDecodingMode::None().isDraftTokensExternal());
static_assert(!SpeculativeDecodingMode::None().isMedusa());
static_assert(!SpeculativeDecodingMode::None().isLookaheadDecoding());

static_assert(SpeculativeDecodingMode::DraftTokensExternal().isDraftTokensExternal());
static_assert(!SpeculativeDecodingMode::DraftTokensExternal().isNone());
static_assert(!SpeculativeDecodingMode::DraftTokensExternal().isMedusa());
static_assert(!SpeculativeDecodingMode::DraftTokensExternal().isLookaheadDecoding());

static_assert(SpeculativeDecodingMode::Medusa().isMedusa());
static_assert(!SpeculativeDecodingMode::Medusa().isNone());
static_assert(!SpeculativeDecodingMode::Medusa().isDraftTokensExternal());
static_assert(!SpeculativeDecodingMode::Medusa().isLookaheadDecoding());

static_assert(SpeculativeDecodingMode::LookaheadDecoding().isLookaheadDecoding());
static_assert(!SpeculativeDecodingMode::LookaheadDecoding().isNone());
static_assert(!SpeculativeDecodingMode::LookaheadDecoding().isDraftTokensExternal());
static_assert(!SpeculativeDecodingMode::LookaheadDecoding().isMedusa());

} // namespace runtime
} // namespace tensorrt_llm
