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

#include "tensorrt_llm/runtime/iTensor.h"

#include <string>

namespace tensorrt_llm::batch_manager
{
template <typename TTensor>
class GenericNamedTensor
{
public:
    using TensorPtr = TTensor;

    TensorPtr tensor;
    std::string name;

    GenericNamedTensor() = default;
    ~GenericNamedTensor() = default;

    GenericNamedTensor(TensorPtr _tensor, std::string _name)
        : tensor{std::move(_tensor)}
        , name{std::move(_name)}
    {
    }

    explicit GenericNamedTensor(std::string _name)
        : tensor{}
        , name{std::move(_name)}
    {
    }

    TensorPtr operator()()
    {
        return tensor;
    }

    TensorPtr const& operator()() const
    {
        return tensor;
    }
};

class NamedTensor : public GenericNamedTensor<tensorrt_llm::runtime::ITensor::SharedPtr>
{
public:
    using Base = GenericNamedTensor<tensorrt_llm::runtime::ITensor::SharedPtr>;
    using TensorPtr = Base::TensorPtr;

    NamedTensor(
        nvinfer1::DataType _type, std::vector<int64_t> const& _shape, std::string _name, void const* _data = nullptr);

    NamedTensor(TensorPtr _tensor, std::string _name)
        : Base(std::move(_tensor), std::move(_name)){};

    explicit NamedTensor(std::string _name)
        : Base(std::move(_name)){};

    [[nodiscard]] std::vector<int64_t> serialize() const;

    void serialize(int64_t* out, const size_t totalSize) const;

    [[nodiscard]] size_t serializedSize() const;

    static NamedTensor deserialize(int64_t const* packed);
};
} // namespace tensorrt_llm::batch_manager
