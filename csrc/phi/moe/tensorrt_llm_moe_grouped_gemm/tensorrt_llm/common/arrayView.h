/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include <cstdint>

namespace tensorrt_llm::common
{

//!
//! \brief A very rudimentary implementation of std::span.
//!
template <typename T>
class ArrayView
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = value_type&;
    using const_reference = value_type const&;
    using pointer = T*;
    using const_pointer = T const*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    ArrayView(T* data, size_type size)
        : mData{data}
        , mSize{size}
    {
    }

    [[nodiscard]] iterator begin()
    {
        return mData;
    }

    [[nodiscard]] iterator end()
    {
        return mData + mSize;
    }

    [[nodiscard]] const_iterator begin() const
    {
        return mData;
    }

    [[nodiscard]] const_iterator end() const
    {
        return mData + mSize;
    }

    [[nodiscard]] const_iterator cbegin() const
    {
        return mData;
    }

    [[nodiscard]] const_iterator cend() const
    {
        return mData + mSize;
    }

    [[nodiscard]] size_type size() const
    {
        return mSize;
    }

    [[nodiscard]] reference operator[](size_type index)
    {
#ifdef INDEX_RANGE_CHECK
        TLLM_CHECK_WITH_INFO(index < mSize, "Index %lu is out of bounds [0, %lu)", index, mSize);
#endif
        return mData[index];
    }

    [[nodiscard]] const_reference operator[](size_type index) const
    {
#ifdef INDEX_RANGE_CHECK
        TLLM_CHECK_WITH_INFO(index < mSize, "Index %lu is out of bounds [0, %lu)", index, mSize);
#endif
        return mData[index];
    }

private:
    T* mData;
    size_type mSize;
};

} // namespace tensorrt_llm::common
