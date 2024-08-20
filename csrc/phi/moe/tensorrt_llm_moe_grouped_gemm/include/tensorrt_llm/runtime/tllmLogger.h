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

#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{

class TllmLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;

    Severity getLevel();

    void setLevel(Severity level);
};

} // namespace tensorrt_llm::runtime
