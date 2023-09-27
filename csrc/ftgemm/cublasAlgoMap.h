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

#include "cuda_utils.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>

#pragma once

#define GEMM_NUM 6
#define GEMM_CONFIG "gemm_config.in"
#define IGEMM_CONFIG "igemm_config.in"
#define SPGEMM_CONFIG "spgemm_config.in"
#define SPIGEMM_CONFIG "spigemm_config.in"

typedef struct {
  int algoId, customOption, tile, splitK_val;
  int swizzle, reductionScheme, workspaceSize;
  // only used in cublasLt >= 11.0
  int stages;
#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
  uint16_t inner_shapeId, cluster_shapeId;
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
  uint16_t mma_shapeId, cga_shapeId, sche_mode;
#endif
  float exec_time;
} cublasLtMatmulAlgo_info;

/* Structure to store information about different run trials */
typedef struct {
  cublasLtMatmulAlgo_t algo;
  cublasStatus_t status;
  float time;
  size_t workspaceSize; // actual memory workspace needed
  cublasMath_t mathMode;
  cublasLtReductionScheme_t reductionScheme;
  int customOption;
  float wavesCount;
} customMatmulPerf_t;

struct cublasAlgoConfig_t {
  int batch_count;
  int m;
  int n;
  int k;
  CublasDataType data_type;
  bool operator==(cublasAlgoConfig_t const &config) const {
    return (batch_count == config.batch_count) && (m == config.m) &&
           (n == config.n) && (k == config.k) &&
           (data_type == config.data_type);
  }
};

class cublasAlgoConfig_hasher {
public:
  std::size_t operator()(cublasAlgoConfig_t const &config) const {
    return config.batch_count * 98317ull ^ config.m * 49157ull ^
           config.n * 24593ull ^ config.k * 196613ull ^
           static_cast<int>(config.data_type) * 6151ull;
  }
};

class cublasAlgoMap {
private:
  std::unordered_map<cublasAlgoConfig_t, cublasLtMatmulAlgo_info,
                     cublasAlgoConfig_hasher>
      algo_map_;
  std::string config_filename_;
  std::string sp_config_filename_;
  std::map<std::string, int> sp_algo_map_;

public:
  cublasAlgoMap(){};
  explicit cublasAlgoMap(const std::string filename,
                         const std::string sp_config_filename = "");
  cublasAlgoMap(const cublasAlgoMap &map);
  ~cublasAlgoMap();
  void loadGemmConfig();
  void loadSpGemmConfig();
  int getSpAlgo(const int batch_count, const int m, const int n, const int k);
  bool isUseSparse(const int batch_count, const int m, const int n,
                   const int k);

  bool isExist(const int batch_count, const int m, const int n, const int k,
               const CublasDataType data_type);

  cublasLtMatmulAlgo_info getAlgo(const int batch_count, const int m,
                                  const int n, const int k,
                                  const CublasDataType data_type);
};
