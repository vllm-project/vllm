/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <vector>

namespace tensorrt_llm::runtime
{

class LoraModule
{
public:
    using TensorPtr = ITensor::SharedPtr;

    enum class ModuleType : SizeType32
    {
        kINVALID = -1,
        kATTN_QKV = 0,
        kATTN_Q = 1,
        kATTN_K = 2,
        kATTN_V = 3,
        kATTN_DENSE = 4,
        kMLP_H_TO_4H = 5,
        kMLP_4H_TO_H = 6,
        kMLP_GATE = 7,
        kCROSS_ATTN_QKV = 8,
        kCROSS_ATTN_Q = 9,
        kCROSS_ATTN_K = 10,
        kCROSS_ATTN_V = 11,
        kCROSS_ATTN_DENSE = 12,
        kMOE_H_TO_4H = 13,
        kMOE_4H_TO_H = 14,
        kMOE_GATE = 15,
        kMOE_ROUTER = 16,
    };

    explicit constexpr LoraModule(ModuleType const& t, SizeType32 inDim, SizeType32 outDim, bool inDimFirst,
        bool outDimFirst, SizeType32 inTpSplitDim, SizeType32 outTpSplitDim) noexcept
        : mType(t)
        , mInDim(inDim)
        , mOutDim(outDim)
        , mInDimFirst(inDimFirst)
        , mOutDimFirst(outDimFirst)
        , mInTpSplitDim(inTpSplitDim)
        , mOutTpSplitDim(outTpSplitDim)
    {
    }

    explicit constexpr LoraModule() noexcept
        : LoraModule(ModuleType::kATTN_QKV, 0, 0, false, true, -1, -1)
    {
    }

    explicit constexpr LoraModule(LoraModule const& o) = default;
    constexpr LoraModule& operator=(LoraModule const& o) = default;

    [[nodiscard]] SizeType32 constexpr flattenedInOutSize(SizeType32 adapterSize) const noexcept
    {
        return adapterSize * (mInDim + mOutDim);
    }

    [[nodiscard]] SizeType32 constexpr inSize(SizeType32 adapterSize) const noexcept
    {
        return adapterSize * mInDim;
    }

    [[nodiscard]] SizeType32 constexpr outSize(SizeType32 adapterSize) const noexcept
    {
        return adapterSize * mOutDim;
    }

    [[nodiscard]] SizeType32 constexpr localInSize(SizeType32 adapterSize, SizeType32 tpSize) const noexcept
    {
        return localInAdapterSize(adapterSize, tpSize) * localInDim(tpSize);
    }

    [[nodiscard]] SizeType32 constexpr localOutSize(SizeType32 adapterSize, SizeType32 tpSize) const noexcept
    {
        return localOutAdapterSize(adapterSize, tpSize) * localOutDim(tpSize);
    }

    [[nodiscard]] SizeType32 constexpr localInDim(SizeType32 tpSize) const noexcept
    {
        if (inTpSplitDim() == 1)
        {
            return inDim() / tpSize;
        }
        return inDim();
    }

    [[nodiscard]] SizeType32 constexpr localOutDim(SizeType32 tpSize) const noexcept
    {
        if (outTpSplitDim() == 0)
        {
            return outDim() / tpSize;
        }
        return outDim();
    }

    [[nodiscard]] SizeType32 constexpr localInAdapterSize(SizeType32 adapterSize, SizeType32 tpSize) const noexcept
    {
        if (inTpSplitDim() == 0)
        {
            return adapterSize / tpSize;
        }
        return adapterSize;
    }

    [[nodiscard]] SizeType32 constexpr localOutAdapterSize(SizeType32 adapterSize, SizeType32 tpSize) const noexcept
    {
        if (outTpSplitDim() == 1)
        {
            return adapterSize / tpSize;
        }
        return adapterSize;
    }

    [[nodiscard]] SizeType32 constexpr localInOutSize(SizeType32 adapterSize, SizeType32 tpSize) const noexcept
    {
        return localInSize(adapterSize, tpSize) + localOutSize(adapterSize, tpSize);
    }

    [[nodiscard]] SizeType32 constexpr value() const noexcept
    {
        return static_cast<SizeType32>(mType);
    }

    [[nodiscard]] std::string_view constexpr name() const noexcept
    {
        return toModuleName(mType);
    }

    [[nodiscard]] SizeType32 constexpr inDim() const noexcept
    {
        return mInDim;
    }

    [[nodiscard]] SizeType32 constexpr outDim() const noexcept
    {
        return mOutDim;
    }

    [[nodiscard]] bool constexpr inDimFirst() const noexcept
    {
        return mInDimFirst;
    }

    [[nodiscard]] bool constexpr outDimFirst() const noexcept
    {
        return mOutDimFirst;
    }

    [[nodiscard]] SizeType32 constexpr inTpSplitDim() const noexcept
    {
        return mInTpSplitDim;
    }

    [[nodiscard]] SizeType32 constexpr outTpSplitDim() const noexcept
    {
        return mOutTpSplitDim;
    }

    static std::vector<LoraModule> createLoraModules(std::vector<std::string> const& loraModuleNames,
        SizeType32 hiddenSize, SizeType32 mlpHiddenSize, SizeType32 numAttentionHeads, SizeType32 numKvAttentionHeads,
        SizeType32 attentionHeadSize, SizeType32 tpSize);

    static ModuleType constexpr toModuleType(std::string_view const& name)
    {
        if (name == "attn_qkv")
            return ModuleType::kATTN_QKV;
        else if (name == "attn_q")
            return ModuleType::kATTN_Q;
        else if (name == "attn_k")
            return ModuleType::kATTN_K;
        else if (name == "attn_v")
            return ModuleType::kATTN_V;
        else if (name == "attn_dense")
            return ModuleType::kATTN_DENSE;
        else if (name == "mlp_h_to_4h")
            return ModuleType::kMLP_H_TO_4H;
        else if (name == "mlp_4h_to_h")
            return ModuleType::kMLP_4H_TO_H;
        else if (name == "mlp_gate")
            return ModuleType::kMLP_GATE;
        else if (name == "cross_attn_qkv")
            return ModuleType::kCROSS_ATTN_QKV;
        else if (name == "cross_attn_q")
            return ModuleType::kCROSS_ATTN_Q;
        else if (name == "cross_attn_k")
            return ModuleType::kCROSS_ATTN_K;
        else if (name == "cross_attn_v")
            return ModuleType::kCROSS_ATTN_V;
        else if (name == "cross_attn_dense")
            return ModuleType::kCROSS_ATTN_DENSE;
        else if (name == "moe_h_to_4h")
            return ModuleType::kMOE_H_TO_4H;
        else if (name == "moe_4h_to_h")
            return ModuleType::kMOE_4H_TO_H;
        else if (name == "moe_gate")
            return ModuleType::kMOE_GATE;
        else if (name == "moe_router")
            return ModuleType::kMOE_ROUTER;
        else
            return ModuleType::kINVALID;
    }

    static std::string_view constexpr toModuleName(ModuleType t) noexcept
    {
        switch (t)
        {
        case ModuleType::kATTN_QKV: return "attn_qkv";
        case ModuleType::kATTN_Q: return "attn_q";
        case ModuleType::kATTN_K: return "attn_k";
        case ModuleType::kATTN_V: return "attn_v";
        case ModuleType::kATTN_DENSE: return "attn_dense";
        case ModuleType::kMLP_H_TO_4H: return "mlp_h_to_4h";
        case ModuleType::kMLP_4H_TO_H: return "mlp_4h_to_h";
        case ModuleType::kMLP_GATE: return "mlp_gate";
        case ModuleType::kCROSS_ATTN_QKV: return "cross_attn_qkv";
        case ModuleType::kCROSS_ATTN_Q: return "cross_attn_q";
        case ModuleType::kCROSS_ATTN_K: return "cross_attn_k";
        case ModuleType::kCROSS_ATTN_V: return "cross_attn_v";
        case ModuleType::kCROSS_ATTN_DENSE: return "cross_attn_dense";
        case ModuleType::kMOE_H_TO_4H: return "moe_h_to_4h";
        case ModuleType::kMOE_4H_TO_H: return "moe_4h_to_h";
        case ModuleType::kMOE_GATE: return "moe_gate";
        case ModuleType::kMOE_ROUTER: return "moe_router";
        case ModuleType::kINVALID: return "INVALID";
        }
        return "INVALID";
    }

    static std::string_view constexpr toModuleName(SizeType32 id)
    {
        auto t = LoraModule::ModuleType(id);
        return toModuleName(t);
    }

private:
    ModuleType mType;
    SizeType32 mInDim;
    SizeType32 mOutDim;
    bool mInDimFirst;
    bool mOutDimFirst;
    SizeType32 mInTpSplitDim;
    SizeType32 mOutTpSplitDim;
};

inline std::ostream& operator<<(std::ostream& output, LoraModule const& module)
{
    return output << "LoraModule(id=" << module.value() << ", "
                  << "name=" << module.name() << ", "
                  << "inDim=" << module.inDim() << ", "
                  << "outDim=" << module.outDim() << ", "
                  << "inTpSplitDim=" << module.inTpSplitDim() << ", "
                  << "outTpSplitDim=" << module.outTpSplitDim() << ")";
}
} // namespace tensorrt_llm::runtime
