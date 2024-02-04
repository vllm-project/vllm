/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cinttypes>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cerrno>
#include <cstdarg>
#include <cstring>

namespace tensorrt_llm::common
{

/* **************************** debug tools ********************************* */
static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}

static const char* _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

static std::string vformat(char const* fmt, va_list args)
{
    va_list args0;
    va_copy(args0, args);
    auto const size = std::vsnprintf(nullptr, 0, fmt, args0);
    if (size <= 0)
        return "";

    std::string stringBuf(size, char{});
    auto const size2 = std::vsnprintf(&stringBuf[0], size + 1, fmt, args);
    return stringBuf;
}

static std::string fmtstr(char const* format, ...)
{
    va_list args;
    va_start(args, format);
    std::string result = vformat(format, args);
    va_end(args);
    return result;
};

// FIXME(woosuk)
template <typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        throw std::runtime_error(
            fmtstr("[ERROR] CUDA runtime error in %s: %s %s:%d\n", func, _cudaGetErrorEnum(result), file, line));
    }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) check((val), #val, file, line)

inline int getSMVersion()
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    check_cuda_error(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

} // namespace tensorrt_llm::common
