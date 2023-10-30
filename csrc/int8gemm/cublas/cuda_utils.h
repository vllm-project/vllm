/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


enum CublasDataType {
  FLOAT_DATATYPE = 0,
  HALF_DATATYPE = 1,
  BFLOAT16_DATATYPE = 2,
  INT8_DATATYPE = 3,
  FP8_DATATYPE = 4
};

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorString(error);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";

  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";

  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") +
                             (_cudaGetErrorEnum(result)) + " " + file + ":" +
                             std::to_string(line) + " \n");
  }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) check((val), #val, file, line)

inline void syncAndCheck(const char *const file, int const line) {
  // When FT_DEBUG_LEVEL=DEBUG, must check error
  static char *level_name = std::getenv("FT_DEBUG_LEVEL");
  if (level_name != nullptr) {
    static std::string level = std::string(level_name);
    if (level == "DEBUG") {
      cudaDeviceSynchronize();
      cudaError_t result = cudaGetLastError();
      if (result) {
        throw std::runtime_error(
            std::string("[FT][ERROR] CUDA runtime error: ") +
            (_cudaGetErrorEnum(result)) + " " + file + ":" +
            std::to_string(line) + " \n");
      }
      // FT_LOG_DEBUG(fmtstr("run syncAndCheck at %s:%d", file, line));
    }
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  cudaError_t result = cudaGetLastError();
  if (result) {
    throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") +
                             (_cudaGetErrorEnum(result)) + " " + file + ":" +
                             std::to_string(line) + " \n");
  }
#endif
}

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__)


[[noreturn]] inline void throwRuntimeError(const char *const file,
                                           int const line,
                                           std::string const &info = "") {
  throw std::runtime_error(std::string("[FT][ERROR] ") + info +
                           " Assertion fail: " + file + ":" +
                           std::to_string(line) + " \n");
}

inline void myAssert(bool result, const char *const file, int const line,
                     std::string const &info = "") {
  if (!result) {
    throwRuntimeError(file, line, info);
  }
}

#define FT_CHECK(val) myAssert(val, __FILE__, __LINE__)
#define FT_CHECK_WITH_INFO(val, info)                                          \
  do {                                                                         \
    bool is_valid_val = (val);                                                 \
    if (!is_valid_val) {                                                       \
      fastertransformer::myAssert(is_valid_val, __FILE__, __LINE__, (info));   \
    }                                                                          \
  } while (0)

#define FT_THROW(info) throwRuntimeError(__FILE__, __LINE__, info)

cudaError_t getSetDevice(int i_device, int *o_device = NULL);

inline int getDevice() {
  int current_dev_id = 0;
  check_cuda_error(cudaGetDevice(&current_dev_id));
  return current_dev_id;
}

inline int getDeviceCount() {
  int count = 0;
  check_cuda_error(cudaGetDeviceCount(&count));
  return count;
}