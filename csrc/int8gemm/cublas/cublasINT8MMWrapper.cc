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

#include "cublasINT8MMWrapper.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

cublasINT8MMWrapper::cublasINT8MMWrapper(cublasLtHandle_t cublaslt_handle,
                                         cudaStream_t stream,
                                         cublasAlgoMap *cublas_algo_map,
                                         std::mutex *mu,
                                         bool use_ORDER_COL32_2R_4R4)
    : cublas_handle_(nullptr), cublaslt_handle_(cublaslt_handle),
      stream_(stream), cublas_algo_map_(cublas_algo_map), mu_(mu),
      allocator_(nullptr), use_ORDER_COL32_2R_4R4_(use_ORDER_COL32_2R_4R4) {}

cublasINT8MMWrapper::cublasINT8MMWrapper(cublasHandle_t cublas_handle,
                                         cublasLtHandle_t cublaslt_handle,
                                         cudaStream_t stream,
                                         cublasAlgoMap *cublas_algo_map,
                                         std::mutex *mu,
                                         bool use_ORDER_COL32_2R_4R4)
    : cublas_handle_(cublas_handle), cublaslt_handle_(cublaslt_handle),
      stream_(stream), cublas_algo_map_(cublas_algo_map), mu_(mu),
      allocator_(nullptr), use_ORDER_COL32_2R_4R4_(use_ORDER_COL32_2R_4R4) {}


cublasINT8MMWrapper::~cublasINT8MMWrapper() { mu_ = nullptr; }

cublasINT8MMWrapper::cublasINT8MMWrapper(const cublasINT8MMWrapper &wrapper)
    : cublas_handle_(nullptr), cublaslt_handle_(wrapper.cublaslt_handle_),
      stream_(wrapper.stream_), cublas_algo_map_(wrapper.cublas_algo_map_), mu_(wrapper.mu_),
      allocator_(wrapper.allocator_), use_ORDER_COL32_2R_4R4_(wrapper.use_ORDER_COL32_2R_4R4_) {
}

// for int8 cublasLtMM with algo
// ATransform should be m*n, CUBLASLT_ORDER_COL32
// kernel should be n*k, CUBLASLT_ORDER_COL4_4R2_8C or
// CUBLASLT_ORDER_COL32_2R_4R4 res is m*n, CUBLASLT_ORDER_COL32
void cublasINT8MMWrapper::Gemm(int *res, int batchCount, int m, int n, int k,
                               int64_t stridea, int64_t strideb,
                               int64_t stridec, const int8_t *ATransform,
                               const int8_t *kernel) {
  mu_->lock();
  cublasOperation_t opTranspose = CUBLAS_OP_T;
#if (CUDART_VERSION >= 11000)
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t computeType = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t AtransformDesc = NULL;
  cublasLtMatrixLayout_t BtransformDesc = NULL;
  cublasLtMatrixLayout_t CtransformDesc = NULL;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;

  cublasLtOrder_t order_matrixB;
#if (CUDART_VERSION >= 11000)
  if (use_ORDER_COL32_2R_4R4_) {
    order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
  } else {
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
  }
#else
  order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#endif

  int ldaTransform = 32 * m;
  int ldbTransform;
  if (use_ORDER_COL32_2R_4R4_) {
    ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
  } else {
    ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;
  }
  int ldcTransform = 32 * m;

  // create matmulDesc
#if (CUDART_VERSION >= 11000)
  cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32I);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &opTranspose, sizeof(cublasOperation_t));
  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));
  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
  cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_matrixB, sizeof(order_matrixB));
  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, m, n, ldcTransform);
  cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));
  if (batchCount > 1) {
    cublasLtMatrixLayoutSetAttribute(AtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
        sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
        sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        CtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
        sizeof(stridec));
  }

  int alphaI = 1;
  int betaI = 0;

  // get algo
  cublasLtMatmulAlgo_t algo;
  int findAlgo = 0;
  if (cublas_algo_map_->isExist(batchCount, m, n, k, INT8_DATATYPE)) {
    // printf("find algo %s\n", markStr.c_str());
    findAlgo = 1;

    cublasLtMatmulAlgo_info tmp_info =
        cublas_algo_map_->getAlgo(batchCount, m, n, k, INT8_DATATYPE);

    cublasLtMatmulAlgoInit(cublaslt_handle_, computeType, CUDA_R_32I, CUDA_R_8I,
                           CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, tmp_info.algoId,
                           &algo);
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(tmp_info.customOption),
        sizeof(tmp_info.customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                         &(tmp_info.tile),
                                         sizeof(tmp_info.tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &(tmp_info.splitK_val),
                                         sizeof(tmp_info.splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(tmp_info.swizzle),
        sizeof(tmp_info.swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(tmp_info.reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &(tmp_info.stages),
                                         sizeof(tmp_info.stages));
#endif
  } else {
    findAlgo = 1;
    int algoId;
    if (use_ORDER_COL32_2R_4R4_) {
      algoId = 7;
    } else {
      algoId = 6;
    }
    int swizzle = 0;
    int customOption = 0;
    int tile = 20;
    int splitK_val = 0;
    int reductionScheme = 0;
    cublasLtMatmulAlgoInit(cublaslt_handle_, computeType, CUDA_R_32I, CUDA_R_8I,
                           CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                         CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                         &(customOption), sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                         &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &(splitK_val), sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                         CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                         &(reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
    int stages;
    if (use_ORDER_COL32_2R_4R4_) {
      stages = 15;
    } else {
      stages = 13;
    }
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &(stages), sizeof(stages));
#endif
  }

  cublasLtMatmul(cublaslt_handle_, matmulDesc, &alphaI, ATransform,
                 AtransformDesc, kernel, BtransformDesc, &betaI, res,
                 CtransformDesc, res, CtransformDesc,
                 (findAlgo == 1 ? (&algo) : NULL), NULL, 0, stream_);

  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixLayoutDestroy(AtransformDesc);
  cublasLtMatrixLayoutDestroy(BtransformDesc);
  cublasLtMatrixLayoutDestroy(CtransformDesc);
  sync_check_cuda_error();
  mu_->unlock();
}

// Atransform: mxk CUDA_R_8I
// kernel: nxk CUDA_R_8I
// res: mxn CUDA_R_32I
// alpha: CUDA_R_32I should be 1
// beta: CUDA_R_32I should be 0
// computeType: CUBLAS_COMPUTE_32I
void cublasINT8MMWrapper::Gemm_(int *res, int batchCount, int m, int n, int k,
                                int64_t stridea, int64_t strideb,
                                int64_t stridec, const int8_t *ATransform,
                                const int8_t *kernel) {
  mu_->lock();
  cublasOperation_t opTranspose = CUBLAS_OP_T;
#if (CUDART_VERSION >= 11000)
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t computeType = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t AtransformDesc = NULL;
  cublasLtMatrixLayout_t BtransformDesc = NULL;
  cublasLtMatrixLayout_t CtransformDesc = NULL;

  // create matmulDesc
#if (CUDART_VERSION >= 11000)
  cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32I);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                 &opTranspose, sizeof(cublasOperation_t));

  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, k, n, k);

  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, k, m, k);

  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, n, m, n);

  if (batchCount > 1) {
    cublasLtMatrixLayoutSetAttribute(AtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
        sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
        sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        CtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
        sizeof(stridec));
  }

  int alphaI = 1;
  int betaI = 0;

  // get algo
  cublasLtMatmulAlgo_t algo;
  int findAlgo = 0;
  if (cublas_algo_map_->isExist(batchCount, m, n, k, INT8_DATATYPE)) {
    // printf("find algo %s\n", markStr.c_str());
    findAlgo = 1;

    cublasLtMatmulAlgo_info tmp_info =
        cublas_algo_map_->getAlgo(batchCount, m, n, k, INT8_DATATYPE);

    cublasLtMatmulAlgoInit(cublaslt_handle_, computeType, CUDA_R_32I, CUDA_R_8I,
                           CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, tmp_info.algoId,
                           &algo);
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(tmp_info.customOption),
        sizeof(tmp_info.customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                         &(tmp_info.tile),
                                         sizeof(tmp_info.tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &(tmp_info.splitK_val),
                                         sizeof(tmp_info.splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(tmp_info.swizzle),
        sizeof(tmp_info.swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(tmp_info.reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &(tmp_info.stages),
                                         sizeof(tmp_info.stages));
#endif
  } else {
    findAlgo = 1;
    int algoId;
    algoId = 21;
    int swizzle = 0;
    int customOption = 0;
    int tile = 20;
    int splitK_val = 0;
    int reductionScheme = 0;
    cublasLtMatmulAlgoInit(cublaslt_handle_, computeType, CUDA_R_32I, CUDA_R_8I,
                           CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                         CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                         &(customOption), sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                         &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &(splitK_val), sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                         CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                         &(reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
    int stages;
    stages = 17;
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &(stages), sizeof(stages));
#endif
  }

  cublasLtMatmul(cublaslt_handle_, matmulDesc, &alphaI, kernel, AtransformDesc,
                 ATransform, BtransformDesc, &betaI, res, CtransformDesc, res,
                 CtransformDesc, (findAlgo == 1 ? (&algo) : NULL), NULL, 0,
                 stream_);

  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixLayoutDestroy(AtransformDesc);
  cublasLtMatrixLayoutDestroy(BtransformDesc);
  cublasLtMatrixLayoutDestroy(CtransformDesc);
  sync_check_cuda_error();
  mu_->unlock();
}

// for int8 IO cublasLtMM with algo
// ATransform should be m*k CUBLASLT_ORDER_COL32
// kernel should be n*k CUBLASLT_ORDER_COL4_4R2_8C
// res is m*n CUBLASLT_ORDER_COL32
void cublasINT8MMWrapper::Gemm(int8_t *res, int batchCount, int m, int n, int k,
                               int64_t stridea, int64_t strideb,
                               int64_t stridec, const float alpha,
                               const int8_t *ATransform, const int8_t *kernel) {
  mu_->lock();
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  // int8 gemm does not support CUBLAS_POINTER_MODE_DEVICE
  // cublasLtPointerMode_t pointerMode =
  // CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
  cudaDataType_t scaleType = CUDA_R_32F;
#if (CUDART_VERSION >= 11000)
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t computeType = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t AtransformDesc = NULL;
  cublasLtMatrixLayout_t BtransformDesc = NULL;
  cublasLtMatrixLayout_t CtransformDesc = NULL;
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;

  cublasLtOrder_t order_matrixB;
#if (CUDART_VERSION >= 11000)
  if (use_ORDER_COL32_2R_4R4_) {
    order_matrixB = CUBLASLT_ORDER_COL32_2R_4R4;
  } else {
    order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
  }
#else
  order_matrixB = CUBLASLT_ORDER_COL4_4R2_8C;
#endif

  int ldaTransform = 32 * m;

  int ldbTransform;
  if (use_ORDER_COL32_2R_4R4_) {
    ldbTransform = 32 * ((n + 32 - 1) / 32) * 32;
  } else {
    ldbTransform = 32 * ((n + 8 - 1) / 8) * 8;
  }

  int ldcTransform = 32 * m;

  // create matmulDesc
#if (CUDART_VERSION >= 11000)
  cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &opTranspose, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                 &scaleType, sizeof(scaleType));
  // cublasLtMatmulDescSetAttribute(matmulDesc,
  // CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode,
  // sizeof(cublasLtPointerMode_t));
  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, m, k, ldaTransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));
  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, n, k, ldbTransform);
  cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_matrixB, sizeof(order_matrixB));
  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_8I, m, n, ldcTransform);
  cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &order_COL32, sizeof(order_COL32));
  if (batchCount > 1) {
    cublasLtMatrixLayoutSetAttribute(AtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
        sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
        sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        CtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
        sizeof(stridec));
  }

  // get algo
  cublasLtMatmulAlgo_t algo;
  int findAlgo = 0;
  if (cublas_algo_map_->isExist(batchCount, m, n, k, INT8_DATATYPE)) {
    findAlgo = 1;

    cublasLtMatmulAlgo_info tmp_info =
        cublas_algo_map_->getAlgo(batchCount, m, n, k, INT8_DATATYPE);

    cublasLtMatmulAlgoInit(cublaslt_handle_, computeType, CUDA_R_32F, CUDA_R_8I,
                           CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, tmp_info.algoId,
                           &algo);
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(tmp_info.customOption),
        sizeof(tmp_info.customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                         &(tmp_info.tile),
                                         sizeof(tmp_info.tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &(tmp_info.splitK_val),
                                         sizeof(tmp_info.splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(tmp_info.swizzle),
        sizeof(tmp_info.swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(tmp_info.reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &(tmp_info.stages),
                                         sizeof(tmp_info.stages));
#endif
  } else {
    findAlgo = 1;
    int algoId;
    if (use_ORDER_COL32_2R_4R4_) {
      algoId = 7;
    } else {
      algoId = 6;
    }
    int swizzle = 0;
    int customOption = 0;
    int tile = 20;
    int splitK_val = 0;
    int reductionScheme = 0;
    cublasLtMatmulAlgoInit(cublaslt_handle_, computeType, CUDA_R_32F, CUDA_R_8I,
                           CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                         CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                         &(customOption), sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                         &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &(splitK_val), sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                         CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                         &(reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
    int stages;
    if (use_ORDER_COL32_2R_4R4_) {
      stages = 15;
    } else {
      stages = 13;
    }
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &(stages), sizeof(stages));
#endif
  }

  float beta = 0.0f;
  cublasLtMatmul(cublaslt_handle_, matmulDesc, &alpha, kernel, AtransformDesc,
                 ATransform, BtransformDesc, &beta, res, CtransformDesc, res,
                 CtransformDesc, (findAlgo == 1 ? (&algo) : NULL), NULL, 0,
                 stream_);

  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixLayoutDestroy(AtransformDesc);
  cublasLtMatrixLayoutDestroy(BtransformDesc);
  cublasLtMatrixLayoutDestroy(CtransformDesc);
  sync_check_cuda_error();
  mu_->unlock();
}

// Atransform: mxk CUDA_R_8I
// kernel: nxk CUDA_R_8I
// res: mxn CUDA_R_8I
// alpha: CUDA_R_32F
// beta: CUDA_R_32F
// computeType: CUBLAS_COMPUTE_32I
void cublasINT8MMWrapper::Gemm_(int8_t *res, int batchCount, int m, int n,
                                int k, int64_t stridea, int64_t strideb,
                                int64_t stridec, const float alpha,
                                const int8_t *ATransform,
                                const int8_t *kernel) {
  mu_->lock();
  cublasOperation_t opTranspose = CUBLAS_OP_T;
  // int8 gemm does not support CUBLAS_POINTER_MODE_DEVICE
  // cublasLtPointerMode_t pointerMode =
  // CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
  cudaDataType_t scaleType = CUDA_R_32F;
#if (CUDART_VERSION >= 11000)
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
#else
  cudaDataType_t computeType = CUDA_R_32I;
#endif
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatrixLayout_t AtransformDesc = NULL;
  cublasLtMatrixLayout_t BtransformDesc = NULL;
  cublasLtMatrixLayout_t CtransformDesc = NULL;

  // create matmulDesc
#if (CUDART_VERSION >= 11000)
  cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);
#else
  cublasLtMatmulDescCreate(&matmulDesc, computeType);
#endif
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                 &opTranspose, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                                 &scaleType, sizeof(scaleType));
  // cublasLtMatmulDescSetAttribute(matmulDesc,
  // CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointerMode,
  // sizeof(cublasLtPointerMode_t));
  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, k, n, k);

  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, k, m, k);

  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_8I, n, m, n);

  if (batchCount > 1) {
    cublasLtMatrixLayoutSetAttribute(AtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        AtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridea,
        sizeof(stridea));
    cublasLtMatrixLayoutSetAttribute(BtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        BtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideb,
        sizeof(strideb));
    cublasLtMatrixLayoutSetAttribute(CtransformDesc,
                                     CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batchCount, sizeof(batchCount));
    cublasLtMatrixLayoutSetAttribute(
        CtransformDesc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stridec,
        sizeof(stridec));
  }

  // get algo
  cublasLtMatmulAlgo_t algo;
  int findAlgo = 0;
  if (cublas_algo_map_->isExist(batchCount, n, m, k, INT8_DATATYPE)) {
    findAlgo = 1;
    cublasLtMatmulAlgo_info tmp_info =
        cublas_algo_map_->getAlgo(batchCount, n, m, k, INT8_DATATYPE);

    cublasLtMatmulAlgoInit(cublaslt_handle_, computeType, CUDA_R_32F, CUDA_R_8I,
                           CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, tmp_info.algoId,
                           &algo);
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(tmp_info.customOption),
        sizeof(tmp_info.customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                         &(tmp_info.tile),
                                         sizeof(tmp_info.tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &(tmp_info.splitK_val),
                                         sizeof(tmp_info.splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(tmp_info.swizzle),
        sizeof(tmp_info.swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        &(tmp_info.reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &(tmp_info.stages),
                                         sizeof(tmp_info.stages));
#endif
  } else {
    findAlgo = 1;
    int algoId;
    algoId = 21;
    int swizzle = 0;
    int customOption = 0;
    int tile = 20;
    int splitK_val = 0;
    int reductionScheme = 0;
    cublasLtMatmulAlgoInit(cublaslt_handle_, computeType, CUDA_R_32F, CUDA_R_8I,
                           CUDA_R_8I, CUDA_R_8I, CUDA_R_8I, algoId, &algo);
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                         CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                         &(customOption), sizeof(customOption));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                         &(tile), sizeof(tile));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                         &(splitK_val), sizeof(splitK_val));
    cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(swizzle), sizeof(swizzle));
    cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                         CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                         &(reductionScheme), sizeof(int));
#if (CUDART_VERSION >= 11000)
    int stages;
    stages = 17;
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                         &(stages), sizeof(stages));
#endif
  }

  float beta = 0.0f;
  cublasLtMatmul(cublaslt_handle_, matmulDesc, &alpha, kernel, AtransformDesc,
                 ATransform, BtransformDesc, &beta, res, CtransformDesc, res,
                 CtransformDesc, (findAlgo == 1 ? (&algo) : NULL), NULL, 0,
                 stream_);

  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixLayoutDestroy(AtransformDesc);
  cublasLtMatrixLayoutDestroy(BtransformDesc);
  cublasLtMatrixLayoutDestroy(CtransformDesc);
  sync_check_cuda_error();
  mu_->unlock();
}

bool cublasINT8MMWrapper::getUseOrderCol322R4R4() {
  return use_ORDER_COL32_2R_4R4_;
}
