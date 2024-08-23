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

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/stringUtils.h"

namespace tensorrt_llm::common
{

class Logger
{

// On Windows, the file wingdi.h is included which has
// #define ERROR 0
// This breaks everywhere ERROR is used in the Level enum
#ifdef _WIN32
#undef ERROR
#endif // _WIN32

public:
    enum Level
    {
        TRACE = 0,
        DEBUG = 10,
        INFO = 20,
        WARNING = 30,
        ERROR = 40
    };

    static Logger* getLogger();

    Logger(Logger const&) = delete;
    void operator=(Logger const&) = delete;

#if defined(_MSC_VER)
    template <typename... Args>
    void log(Level level, char const* format, Args const&... args);

    template <typename... Args>
    void log(Level level, int rank, char const* format, Args const&... args);
#else
    template <typename... Args>
    void log(Level level, char const* format, Args const&... args) __attribute__((format(printf, 3, 0)));

    template <typename... Args>
    void log(Level level, int rank, char const* format, Args const&... args) __attribute__((format(printf, 4, 0)));
#endif

    template <typename... Args>
    void log(Level level, std::string const& format, Args const&... args)
    {
        return log(level, format.c_str(), args...);
    }

    template <typename... Args>
    void log(const Level level, int const rank, std::string const& format, Args const&... args)
    {
        return log(level, rank, format.c_str(), args...);
    }

    void log(std::exception const& ex, Level level = Level::ERROR);

    Level getLevel()
    {
        return level_;
    }

    void setLevel(const Level level)
    {
        level_ = level;
        log(INFO, "Set logger level by %s", getLevelName(level));
    }

private:
    static auto constexpr kPREFIX = "[TensorRT-LLM]";

#ifndef NDEBUG
    const Level DEFAULT_LOG_LEVEL = DEBUG;
#else
    const Level DEFAULT_LOG_LEVEL = INFO;
#endif
    Level level_ = DEFAULT_LOG_LEVEL;

    Logger(); // NOLINT(modernize-use-equals-delete)

    static inline char const* getLevelName(const Level level)
    {
        switch (level)
        {
        case TRACE: return "TRACE";
        case DEBUG: return "DEBUG";
        case INFO: return "INFO";
        case WARNING: return "WARNING";
        case ERROR: return "ERROR";
        }

        TLLM_THROW("Unknown log level: %d", level);
    }

    static inline std::string getPrefix(const Level level)
    {
        return fmtstr("%s[%s] ", kPREFIX, getLevelName(level));
    }

    static inline std::string getPrefix(const Level level, int const rank)
    {
        return fmtstr("%s[%s][%d] ", kPREFIX, getLevelName(level), rank);
    }
};

template <typename... Args>
void Logger::log(Logger::Level level, char const* format, Args const&... args)
{
    if (level_ <= level)
    {
        auto const fmt = getPrefix(level) + format;
        auto& out = level_ < WARNING ? std::cout : std::cerr;
        if constexpr (sizeof...(args) > 0)
        {
            out << fmtstr(fmt.c_str(), args...);
        }
        else
        {
            out << fmt;
        }
        out << std::endl;
    }
}

template <typename... Args>
void Logger::log(const Logger::Level level, int const rank, char const* format, Args const&... args)
{
    if (level_ <= level)
    {
        auto const fmt = getPrefix(level, rank) + format;
        auto& out = level_ < WARNING ? std::cout : std::cerr;
        if constexpr (sizeof...(args) > 0)
        {
            out << fmtstr(fmt.c_str(), args...);
        }
        else
        {
            out << fmt;
        }
        out << std::endl;
    }
}

#define TLLM_LOG(level, ...) tensorrt_llm::common::Logger::getLogger()->log(level, __VA_ARGS__)
#define TLLM_LOG_TRACE(...) TLLM_LOG(tensorrt_llm::common::Logger::TRACE, __VA_ARGS__)
#define TLLM_LOG_DEBUG(...) TLLM_LOG(tensorrt_llm::common::Logger::DEBUG, __VA_ARGS__)
#define TLLM_LOG_INFO(...) TLLM_LOG(tensorrt_llm::common::Logger::INFO, __VA_ARGS__)
#define TLLM_LOG_WARNING(...) TLLM_LOG(tensorrt_llm::common::Logger::WARNING, __VA_ARGS__)
#define TLLM_LOG_ERROR(...) TLLM_LOG(tensorrt_llm::common::Logger::ERROR, __VA_ARGS__)
#define TLLM_LOG_EXCEPTION(ex, ...) tensorrt_llm::common::Logger::getLogger()->log(ex, ##__VA_ARGS__)
} // namespace tensorrt_llm::common
