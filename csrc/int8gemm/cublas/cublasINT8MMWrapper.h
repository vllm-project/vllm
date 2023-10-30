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

#include "allocator.h"
#include "cublasAlgoMap.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <string>

#pragma once

class cublasINT8MMWrapper{
protected:
  cublasHandle_t cublas_handle_;
  cublasLtHandle_t cublaslt_handle_;
  cudaStream_t stream_;
  cublasAlgoMap *cublas_algo_map_;
  std::mutex *mu_;
  IAllocator *allocator_ = nullptr;
  
private:
  bool use_ORDER_COL32_2R_4R4_;

public:
  cublasINT8MMWrapper(cublasLtHandle_t cublaslt_handle_, cudaStream_t stream,
                      cublasAlgoMap *map, std::mutex *mu,
                      bool use_ORDER_COL32_2R_4R4);

  cublasINT8MMWrapper(cublasHandle_t cublas_handle,
                      cublasLtHandle_t cublaslt_handle, cudaStream_t stream,
                      cublasAlgoMap *map, std::mutex *mu,
                      bool use_ORDER_COL32_2R_4R4);

  ~cublasINT8MMWrapper();

  cublasINT8MMWrapper(const cublasINT8MMWrapper &wrapper);

  void Gemm(int *res, int batchCount, int m, int n, int k, int64_t stridea,
            int64_t strideb, int64_t stridec, const int8_t *ATransform,
            const int8_t *kernel);

  void Gemm_(int *res, int batchCount, int m, int n, int k, int64_t stridea,
             int64_t strideb, int64_t stridec, const int8_t *ATransform,
             const int8_t *kernel);

  void Gemm(int8_t *res, int batchCount, int m, int n, int k, int64_t stridea,
            int64_t strideb, int64_t stridec, const float alpha,
            const int8_t *ATransform, const int8_t *kernel);

  void Gemm_(int8_t *res, int batchCount, int m, int n, int k, int64_t stridea,
             int64_t strideb, int64_t stridec, const float alpha,
             const int8_t *ATransform, const int8_t *kernel);

  bool getUseOrderCol322R4R4();
};