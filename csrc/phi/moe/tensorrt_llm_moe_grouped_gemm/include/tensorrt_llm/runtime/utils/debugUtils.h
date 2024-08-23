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

#include "tensorrt_llm/runtime/bufferManager.h"

namespace tensorrt_llm::runtime::utils
{

template <typename T>
bool tensorHasNan(ITensor const& tensor, BufferManager const& manager, std::string const& infoStr);

bool tensorHasNan(
    size_t M, size_t K, nvinfer1::DataType type, void const* data, cudaStream_t stream, std::string const& infoStr);

} // namespace tensorrt_llm::runtime::utils
