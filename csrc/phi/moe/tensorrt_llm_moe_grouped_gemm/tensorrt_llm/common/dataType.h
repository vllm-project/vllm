/*
 * Copyright (c) 1993-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/logger.h"
#include <NvInferRuntime.h>

namespace tensorrt_llm::common
{

constexpr static size_t getDTypeSize(nvinfer1::DataType type)
{
    switch (type)
    {
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kINT32: [[fallthrough]];
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kBF16: [[fallthrough]];
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL: [[fallthrough]];
    case nvinfer1::DataType::kUINT8: [[fallthrough]];
    case nvinfer1::DataType::kINT8: [[fallthrough]];
    case nvinfer1::DataType::kFP8: return 1;
    case nvinfer1::DataType::kINT4: TLLM_THROW("Cannot determine size of INT4 data type");
    default: return 0;
    }
    return 0;
}

} // namespace tensorrt_llm::common
