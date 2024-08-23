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
#include <mutex>

namespace tensorrt_llm::plugins::api
{

auto constexpr kDefaultNamespace = "tensorrt_llm";

class LoggerManager
{
public:
    //! Set the logger finder.
    void setLoggerFinder(nvinfer1::ILoggerFinder* finder);

    //! Get the logger.
    [[maybe_unused]] nvinfer1::ILogger* logger();

    static LoggerManager& getInstance() noexcept;

    static nvinfer1::ILogger* defaultLogger() noexcept;

private:
    LoggerManager() = default;

    nvinfer1::ILoggerFinder* mLoggerFinder{nullptr};
    std::mutex mMutex;
};
} // namespace tensorrt_llm::plugins::api

extern "C"
{
    // This function is used for explicitly registering the TRT-LLM plugins and the default logger.
    bool initTrtLlmPlugins(void* logger = tensorrt_llm::plugins::api::LoggerManager::defaultLogger(),
        char const* libNamespace = tensorrt_llm::plugins::api::kDefaultNamespace);

    // The functions below are used by TensorRT to when loading a shared plugin library with automatic registering.
    // see https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#generating-plugin-library
    [[maybe_unused]] void setLoggerFinder([[maybe_unused]] nvinfer1::ILoggerFinder* finder);
    [[maybe_unused]] nvinfer1::IPluginCreator* const* getPluginCreators(int32_t& nbCreators);
    [[maybe_unused]] nvinfer1::IPluginCreatorInterface* const* getCreators(std::int32_t& nbCreators);
}
