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

#include "tensorrt_llm/common/stringUtils.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#define NEW_TLLM_EXCEPTION(...)                                                                                        \
    tensorrt_llm::common::TllmException(__FILE__, __LINE__, tensorrt_llm::common::fmtstr(__VA_ARGS__))

namespace tensorrt_llm::common
{

class TllmException : public std::runtime_error
{
public:
    static auto constexpr MAX_FRAMES = 128;

    explicit TllmException(char const* file, std::size_t line, std::string const& msg);

    ~TllmException() noexcept override;

    [[nodiscard]] std::string getTrace() const;

    static std::string demangle(char const* name);

private:
    std::array<void*, MAX_FRAMES> mCallstack{};
    int mNbFrames;
};

} // namespace tensorrt_llm::common
