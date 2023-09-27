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

#include "cublasMMWrapper.h"
#include "cuda_utils.h"
#include <algorithm>

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

cublasMMWrapper::cublasMMWrapper(cublasHandle_t cublas_handle,
                                 cublasLtHandle_t cublaslt_handle,
                                 cudaStream_t stream,
                                 cublasAlgoMap *cublas_algo_map, std::mutex *mu,
                                 IAllocator *allocator)
    : cublas_handle_(cublas_handle), cublaslt_handle_(cublaslt_handle),
      stream_(stream), cublas_algo_map_(cublas_algo_map), mu_(mu),
      allocator_(allocator) {
  // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  if (allocator_ != nullptr) {
    cublas_workspace_ =
        allocator_->reMalloc(cublas_workspace_, CUBLAS_WORKSPACE_SIZE, false);
  }
}

#ifdef SPARSITY_ENABLED
cublasMMWrapper::cublasMMWrapper(cublasHandle_t cublas_handle,
                                 cublasLtHandle_t cublaslt_handle,
                                 cusparseLtHandle_t cusparselt_handle,
                                 cudaStream_t stream,
                                 cublasAlgoMap *cublas_algo_map, std::mutex *mu,
                                 IAllocator *allocator)
    : cublas_handle_(cublas_handle), cublaslt_handle_(cublaslt_handle),
      cusparselt_handle_(cusparselt_handle), stream_(stream),
      cublas_algo_map_(cublas_algo_map), mu_(mu), allocator_(allocator) {
  // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  if (allocator_ != nullptr) {
    cublas_workspace_ =
        allocator_->reMalloc(cublas_workspace_, CUBLAS_WORKSPACE_SIZE, false);
  }
}
#endif

cublasMMWrapper::~cublasMMWrapper() {
  // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  mu_ = nullptr;
  if (allocator_ != nullptr) {
    allocator_->free((void **)(&cublas_workspace_));
    allocator_ = nullptr;
  }
}

cublasMMWrapper::cublasMMWrapper(const cublasMMWrapper &wrapper)
    : cublas_handle_(wrapper.cublas_handle_),
      cublaslt_handle_(wrapper.cublaslt_handle_),
#ifdef SPARSITY_ENABLED
      cusparselt_handle_(wrapper.cusparselt_handle_),
#endif
      stream_(wrapper.stream_), cublas_algo_map_(wrapper.cublas_algo_map_),
      mu_(wrapper.mu_), allocator_(wrapper.allocator_) {
  // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  if (allocator_ != nullptr) {
    cublas_workspace_ =
        allocator_->reMalloc(cublas_workspace_, CUBLAS_WORKSPACE_SIZE, false);
  }
}

void cublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb,
                           const int m, const int n, const int k,
                           const void *alpha, const void *A,
                           cudaDataType_t Atype, int lda, const void *B,
                           cudaDataType_t Btype, int ldb, const void *beta,
                           void *C, cudaDataType_t Ctype, int ldc,
                           cudaDataType_t computeType, cublasGemmAlgo_t algo) {
  // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  mu_->lock();
  check_cuda_error(cublasGemmEx(cublas_handle_, transa, transb, m, n, k, alpha,
                                A, Atype, lda, B, Btype, ldb, beta, C, Ctype,
                                ldc, computeType, algo));
  sync_check_cuda_error();
  mu_->unlock();
}

void cublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb,
                           const int m, const int n, const int k, const void *A,
                           const int lda, const void *B, const int ldb, void *C,
                           const int ldc) {
  // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  Gemm(transa, transb, m, n, k, A, lda, B, ldb, C, ldc, 1.0f, 0.0f);
}

void cublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb,
                           const int m, const int n, const int k, const void *A,
                           const int lda, const void *B, const int ldb, void *C,
                           const int ldc, float f_alpha, float f_beta) {
  // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  half h_alpha = (half)(f_alpha);
  half h_beta = (half)(f_beta);

  mu_->lock();
  // TODO: default cublas libs
  int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
  bool using_cublasLt = (Atype_ == CUDA_R_16F) ? true : false;
  int batch_count = 1;
  // fp32 use cublas as default
  // fp16 use cublasLt as default
  const void *alpha = is_fp16_computeType ? reinterpret_cast<void *>(&h_alpha)
                                          : reinterpret_cast<void *>(&f_alpha);
  const void *beta = is_fp16_computeType ? reinterpret_cast<void *>(&h_beta)
                                         : reinterpret_cast<void *>(&f_beta);

  int findAlgo = cublas_algo_map_->isExist(batch_count, m, n, k,
                                           getCublasDataType(Atype_));

  cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(
      batch_count, m, n, k, getCublasDataType(Atype_));
  if (findAlgo) {
    if (info.stages != -1) {
      using_cublasLt = true;
    } else {
      using_cublasLt = false;
    }
  }

  if (using_cublasLt) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cudaDataType_t scaleType;
#if (CUDART_VERSION >= 11000)
    cublasComputeType_t computeType;
#else
    cudaDataType_t computeType;
#endif

    if (is_fp16_computeType) {
#if (CUDART_VERSION >= 11000)
      computeType = CUBLAS_COMPUTE_16F;
#else
      computeType = CUDA_R_16F;
#endif
      scaleType = CUDA_R_16F;
    } else {
#if (CUDART_VERSION >= 11000)
      computeType = CUBLAS_COMPUTE_32F;
#else
      computeType = CUDA_R_32F;
#endif
      scaleType = CUDA_R_32F;
    }

    // --------------------------------------
    // Create descriptors for the original matrices
    cublasLtMatrixLayoutCreate(&Adesc, Atype_, transa == CUBLAS_OP_N ? m : k,
                               transa == CUBLAS_OP_N ? k : m, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, Btype_, transb == CUBLAS_OP_N ? k : n,
                               transb == CUBLAS_OP_N ? n : k, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, Ctype_, m, n, ldc);
#if (CUDART_VERSION >= 11000)
    cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
#else
    cublasLtMatmulDescCreate(&operationDesc, computeType);
#endif

    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &transa, sizeof(cublasOperation_t));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &transb, sizeof(cublasOperation_t));

    cublasLtMatmulAlgo_t algo;
    void *workSpace = cublas_workspace_;
    int workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;
    if (findAlgo) {
      if (info.workspaceSize > workspaceSize) {
        findAlgo = 0;
      } else {
        cublasLtMatmulAlgoInit(cublaslt_handle_, computeType, scaleType, Atype_,
                               Btype_, Ctype_, Ctype_, info.algoId, &algo);
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &(info.customOption),
            sizeof(info.customOption));
        cublasLtMatmulAlgoConfigSetAttribute(&algo,
                                             CUBLASLT_ALGO_CONFIG_TILE_ID,
                                             &(info.tile), sizeof(info.tile));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &(info.splitK_val),
            sizeof(info.splitK_val));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &(info.swizzle),
            sizeof(info.swizzle));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
            &(info.reductionScheme), sizeof(info.reductionScheme));

#if (CUDART_VERSION >= 11000)
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &(info.stages),
            sizeof(info.stages));
#endif

#if (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH >= 3)
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &(info.inner_shapeId),
            sizeof(info.inner_shapeId));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
            &(info.cluster_shapeId), sizeof(info.cluster_shapeId));
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_MMA_SHAPE_ID, &(info.mma_shapeId),
            sizeof(info.mma_shapeId));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_CGA_SHAPE_ID, &(info.cga_shapeId),
            sizeof(info.cga_shapeId));
        cublasLtMatmulAlgoConfigSetAttribute(
            &algo, CUBLASLT_ALGO_CONFIG_SCHEDULING_MODE, &(info.sche_mode),
            sizeof(info.sche_mode));
#endif
      }
    }

    cublasLtMatmul(cublaslt_handle_, operationDesc, alpha, A, Adesc, B, Bdesc,
                   beta, C, Cdesc, C, Cdesc, (findAlgo == 1 ? (&algo) : NULL),
                   workSpace, workspaceSize, stream_);

    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    sync_check_cuda_error();
  } else {
    int cublasAlgo = info.algoId;
    check_cuda_error(cublasGemmEx(cublas_handle_, transa, transb, m, n, k,
                                  alpha, A, Atype_, lda, B, Btype_, ldb, beta,
                                  C, Ctype_, ldc, computeType_,
                                  static_cast<cublasGemmAlgo_t>(cublasAlgo)));
    sync_check_cuda_error();
  }
  mu_->unlock();
}

void cublasMMWrapper::setFP32GemmConfig() {
  Atype_ = CUDA_R_32F;
  Btype_ = CUDA_R_32F;
  Ctype_ = CUDA_R_32F;
  computeType_ = CUDA_R_32F;
}

void cublasMMWrapper::setFP16GemmConfig() {
  Atype_ = CUDA_R_16F;
  Btype_ = CUDA_R_16F;
  Ctype_ = CUDA_R_16F;
  computeType_ = CUDA_R_32F;
}

#ifdef ENABLE_BF16
void cublasMMWrapper::setBF16GemmConfig() {
  Atype_ = CUDA_R_16BF;
  Btype_ = CUDA_R_16BF;
  Ctype_ = CUDA_R_16BF;
  computeType_ = CUDA_R_32F;
}
#endif

void cublasMMWrapper::setGemmConfig(cudaDataType_t aType, cudaDataType_t bType,
                                    cudaDataType_t cType,
                                    cudaDataType_t computeType) {
  Atype_ = aType;
  Btype_ = bType;
  Ctype_ = cType;
  computeType_ = computeType;
}

CublasDataType cublasMMWrapper::getCublasDataType(cudaDataType_t data_type) {
  if (data_type == CUDA_R_16F) {
    return HALF_DATATYPE;
  } else if (data_type == CUDA_R_32F) {
    return FLOAT_DATATYPE;
  }
#ifdef ENABLE_BF16
  else if (data_type == CUDA_R_16BF) {
    return BFLOAT16_DATATYPE;
  }
#endif
  return FLOAT_DATATYPE;
}

#if (CUDART_VERSION >= 11000)
// input, weight, output are row-major
// only works for cublas 11.x
void cublasMMWrapper::Gemm(cublasOperation_t transa, cublasOperation_t transb,
                           const int m, const int n, const int k, const void *A,
                           const int lda, const void *B, const int ldb,
                           const void *bias, void *C, const int ldc) {
  // FT_LOG_DEBUG(__PRETTY_FUNCTION__);
  cudaDataType_t Atype, Btype, Ctype;
  cublasComputeType_t computeType;
  cudaDataType_t scaleType;
  float alpha_float = 1.0f;
  float beta_float = 0.0f;
  half alpha_half = half(1.0f);
  half beta_half = half(0.0f);
  void *alpha, *beta;

  // int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
  if (Atype_ == CUDA_R_32F) {
    computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    Atype = CUDA_R_32F;
    Btype = CUDA_R_32F;
    Ctype = CUDA_R_32F;
    scaleType = CUDA_R_32F;
    alpha = &alpha_float;
    beta = &beta_float;
  } else if (Atype_ == CUDA_R_16BF) {
    computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    Atype = CUDA_R_16BF;
    Btype = CUDA_R_16BF;
    Ctype = CUDA_R_16BF;
    scaleType = CUDA_R_32F;
    alpha = &alpha_float;
    beta = &beta_float;
  } else {
    computeType = CUBLAS_COMPUTE_16F;
    Atype = CUDA_R_16F;
    Btype = CUDA_R_16F;
    Ctype = CUDA_R_16F;
    scaleType = CUDA_R_16F;
    alpha = &alpha_half;
    beta = &beta_half;
  }

  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
  cublasLtMatrixLayoutCreate(&Adesc, Atype, (transa == CUBLAS_OP_N) ? m : k,
                             (transa == CUBLAS_OP_N) ? k : m, lda);
  cublasLtMatrixLayoutCreate(&Bdesc, Btype, (transb == CUBLAS_OP_N) ? k : n,
                             (transb == CUBLAS_OP_N) ? n : k, ldb);
  cublasLtMatrixLayoutCreate(&Cdesc, Ctype, m, n, ldc);

  cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType);
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                 &transa, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                 &transb, sizeof(cublasOperation_t));
  cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                 &epi, sizeof(cublasLtEpilogue_t));
  cublasLtMatmulDescSetAttribute(operationDesc,
                                 CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                 sizeof(const void *));
  check_cuda_error(cublasLtMatmul(cublaslt_handle_, operationDesc, alpha, A,
                                  Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc,
                                  NULL, NULL, 0, stream_));
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatmulDescDestroy(operationDesc);
}
#endif
void cublasMMWrapper::setStream(cudaStream_t stream) { stream_ = stream; }

void cublasMMWrapper::stridedBatchedGemm(
    cublasOperation_t transa, cublasOperation_t transb, const int m,
    const int n, const int k, const void *A, const int lda,
    const int64_t strideA, const void *B, const int ldb, const int64_t strideB,
    void *C, const int ldc, const int64_t strideC, const int batch_count,
    const float f_alpha, const float f_beta) {
  half h_alpha = (half)f_alpha;
  half h_beta = (half)f_beta;

  mu_->lock();
  int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
  const void *alpha = is_fp16_computeType
                          ? reinterpret_cast<void *>(&h_alpha)
                          : reinterpret_cast<const void *>(&f_alpha);
  const void *beta = is_fp16_computeType
                         ? reinterpret_cast<void *>(&h_beta)
                         : reinterpret_cast<const void *>(&f_beta);
  cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(
      batch_count, m, n, k, getCublasDataType(Atype_));

  check_cuda_error(cublasGemmStridedBatchedEx(
      cublas_handle_, transa, transb, m, n, k, alpha, A, Atype_, lda, strideA,
      B, Btype_, ldb, strideB, beta, C, Ctype_, ldc, strideC, batch_count,
      computeType_, static_cast<cublasGemmAlgo_t>(info.algoId)));

  mu_->unlock();
}

void cublasMMWrapper::stridedBatchedGemm(
    cublasOperation_t transa, cublasOperation_t transb, const int m,
    const int n, const int k, const float f_alpha, const void *A,
    cudaDataType_t AType, const int lda, const int64_t strideA, const void *B,
    cudaDataType_t BType, const int ldb, const int64_t strideB,
    const float f_beta, void *C, cudaDataType_t CType, const int ldc,
    const int64_t strideC, const int batch_count, cudaDataType_t computeType) {
  half h_alpha = (half)f_alpha;
  half h_beta = (half)f_beta;

  mu_->lock();
  int is_fp16_computeType = computeType == CUDA_R_16F ? 1 : 0;
  const void *alpha = is_fp16_computeType
                          ? reinterpret_cast<void *>(&h_alpha)
                          : reinterpret_cast<const void *>(&f_alpha);
  const void *beta = is_fp16_computeType
                         ? reinterpret_cast<void *>(&h_beta)
                         : reinterpret_cast<const void *>(&f_beta);
  cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(
      batch_count, m, n, k, getCublasDataType(Atype_));

  check_cuda_error(cublasGemmStridedBatchedEx(
      cublas_handle_, transa, transb, m, n, k, alpha, A, AType, lda, strideA, B,
      BType, ldb, strideB, beta, C, CType, ldc, strideC, batch_count,
      computeType, static_cast<cublasGemmAlgo_t>(info.algoId)));

  mu_->unlock();
}

void cublasMMWrapper::batchedGemm(cublasOperation_t transa,
                                  cublasOperation_t transb, const int m,
                                  const int n, const int k,
                                  const void *const *A, const int lda,
                                  const void *const *B, const int ldb,
                                  void *const *C, const int ldc,
                                  const int batch_count) {
  float f_alpha = static_cast<float>(1.0f);
  float f_beta = static_cast<float>(0.0f);

  half h_alpha = (half)1.0f;
  half h_beta = (half)0.0f;

  mu_->lock();
  int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
  const void *alpha = is_fp16_computeType ? reinterpret_cast<void *>(&h_alpha)
                                          : reinterpret_cast<void *>(&f_alpha);
  const void *beta = is_fp16_computeType ? reinterpret_cast<void *>(&h_beta)
                                         : reinterpret_cast<void *>(&f_beta);
  cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(
      batch_count, m, n, k, getCublasDataType(Atype_));

  check_cuda_error(cublasGemmBatchedEx(
      cublas_handle_, transa, transb, m, n, k, alpha, A, Atype_, lda, B, Btype_,
      ldb, beta, C, Ctype_, ldc, batch_count, computeType_,
      static_cast<cublasGemmAlgo_t>(info.algoId)));
  mu_->unlock();
}

bool cublasMMWrapper::isFuseBatchGemm(const int batch_count, const int m,
                                      const int k, const int n) {
  CublasDataType data_type = getCublasDataType(Atype_);

  if (cublas_algo_map_->isExist(batch_count, m, k, n, data_type) == false ||
      cublas_algo_map_->isExist(1, m, k, n, data_type) == false) {
    return false;
  } else {
    return cublas_algo_map_->getAlgo(batch_count, m, k, n, data_type)
               .exec_time <
           3 * cublas_algo_map_->getAlgo(1, m, k, n, data_type).exec_time;
  }
}

#ifdef SPARSITY_ENABLED
void cublasMMWrapper::SpGemm(cublasOperation_t transa, cublasOperation_t transb,
                             const int m, const int n, const int k,
                             const void *A, const void *B, void *C) {
  if (Atype_ != CUDA_R_16F || Btype_ != CUDA_R_16F || Ctype_ != CUDA_R_16F) {
    throw std::runtime_error(
        "\n[FT][ERROR] sparse GEMM only supports FP16 data type now.");
  }
  static bool not_printed_fp32_accumulation_warning = true;
  if (computeType_ != CUDA_R_16F && not_printed_fp32_accumulation_warning) {
    printf("[FT][WARNING] cublasMMWrapper sets to FP32 compute type, "
           "but sparse gemm will use FP16 compute type since cusparselt "
           "supports FP16 accumulation only.\n");
    not_printed_fp32_accumulation_warning = false;
  }
  cusparseOrder_t order = CUSPARSE_ORDER_COL;
  cusparseOperation_t opA = (transa == CUBLAS_OP_N)
                                ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                : CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t opB = (transb == CUBLAS_OP_N)
                                ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                : CUSPARSE_OPERATION_TRANSPOSE;
  cusparseComputeType compute_type = CUSPARSE_COMPUTE_16F;
  cusparseLtMatmulDescriptor_t matmul;
  cusparseLtMatmulAlgSelection_t alg_sel;
  cusparseLtMatmulPlan_t plan;

  bool is_rowmajor = (order == CUSPARSE_ORDER_ROW);
  bool isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
  bool isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
  auto num_A_rows = (isA_transposed) ? k : m;
  auto num_A_cols = (isA_transposed) ? m : k;
  auto num_B_rows = (isB_transposed) ? n : k;
  auto num_B_cols = (isB_transposed) ? k : n;
  auto num_C_rows = m;
  auto num_C_cols = n;
  unsigned alignment = 16;
  auto lda = (is_rowmajor) ? num_A_cols : num_A_rows;
  auto ldb = (is_rowmajor) ? num_B_cols : num_B_rows;
  auto ldc = (is_rowmajor) ? num_C_cols : num_C_rows;
  float _alpha(1.0f);
  float _beta(0.0f);

  char mark[256];
  sprintf(mark, "%d_%d_%d_%d", 1, m, n, k);
  if (sp_mat_A_desc_map_.find(mark) != sp_mat_A_desc_map_.end()) {
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
        &cusparselt_handle_, &matmul, opA, opB, &sp_mat_A_desc_map_[mark],
        &sp_mat_B_desc_map_[mark], &sp_mat_C_desc_map_[mark],
        &sp_mat_C_desc_map_[mark], compute_type))
  } else {
    // initializing MatDesc takes a lot of time
    cusparseLtMatDescriptor_t matA, matB, matC;
    sp_mat_A_desc_map_[mark] = matA;
    sp_mat_B_desc_map_[mark] = matB;
    sp_mat_C_desc_map_[mark] = matC;
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
        &cusparselt_handle_, &sp_mat_A_desc_map_[mark], num_A_rows, num_A_cols,
        lda, alignment, Atype_, order, CUSPARSELT_SPARSITY_50_PERCENT))
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        &cusparselt_handle_, &sp_mat_B_desc_map_[mark], num_B_rows, num_B_cols,
        ldb, alignment, Btype_, order))
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
        &cusparselt_handle_, &sp_mat_C_desc_map_[mark], num_C_rows, num_C_cols,
        ldc, alignment, Ctype_, order))
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
        &cusparselt_handle_, &matmul, opA, opB, &sp_mat_A_desc_map_[mark],
        &sp_mat_B_desc_map_[mark], &sp_mat_C_desc_map_[mark],
        &sp_mat_C_desc_map_[mark], compute_type))
  }
  mu_->lock();
  CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
      &cusparselt_handle_, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT))
  int alg = cublas_algo_map_->getSpAlgo(1, num_A_rows, num_B_cols, num_A_cols);
  CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
      &cusparselt_handle_, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg,
      sizeof(alg)))
  size_t workspace_size;
  CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&cusparselt_handle_, &alg_sel,
                                              &workspace_size))
  CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&cusparselt_handle_, &plan, &matmul,
                                          &alg_sel, workspace_size))

  void *d_workspace = nullptr;
  int num_streams = 1;
  cudaStream_t streams[1] = {stream_};
  CHECK_CUSPARSE(cusparseLtMatmul(&cusparselt_handle_, &plan, &_alpha, A, B,
                                  &_beta, C, C, d_workspace, streams,
                                  num_streams))
  CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan))
  sync_check_cuda_error();
  mu_->unlock();
}

size_t cublasMMWrapper::getSparseMatrixSize(int m, int k) {
  // Get a compressed matrix size of shape (m, k) used in cusparselt.
  auto Atype_ = CUDA_R_16F;
  cusparseOrder_t order = CUSPARSE_ORDER_COL;
  unsigned alignment = 16;
  int num_A_rows = m;
  int num_A_cols = k;
  int lda = num_A_rows;

  cusparseLtMatDescriptor_t matA;
  CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
      &cusparselt_handle_, &matA, num_A_rows, num_A_cols, lda, alignment,
      Atype_, order, CUSPARSELT_SPARSITY_50_PERCENT));
  size_t compressed_size = 0;
  CHECK_CUSPARSE(cusparseLtSpMMACompressedSize2(&cusparselt_handle_, &matA,
                                                &compressed_size));
  return compressed_size;
}

void cublasMMWrapper::compressMatrix(const void *input, void *output,
                                     const int m, const int k) {
  cusparseOrder_t order = CUSPARSE_ORDER_COL;
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseLtMatDescriptor_t matA;
  unsigned alignment = 16;
  CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
      &cusparselt_handle_, &matA, m, k, m, alignment, CUDA_R_16F, order,
      CUSPARSELT_SPARSITY_50_PERCENT))
  CHECK_CUSPARSE(cusparseLtSpMMACompress2(&cusparselt_handle_, &matA, true, opA,
                                          input, output, stream_))
  sync_check_cuda_error();
}

bool cublasMMWrapper::isUseSparse(const int batch_count, const int m,
                                  const int n, const int k) {
  return cublas_algo_map_->isUseSparse(batch_count, m, n, k);
}
#endif

std::pair<bool, cublasLtMatmulAlgo_t> cublasMMWrapper::findBestAlgo(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
    const void *alpha, const void *A, cublasLtMatrixLayout_t Adesc,
    const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta,
    const void *C, cublasLtMatrixLayout_t Cdesc, void *D,
    cublasLtMatrixLayout_t Ddesc, cudaStream_t stream) {
#if (CUBLAS_VERSION) < 11601
  FT_CHECK_WITH_INFO(false, "CUBLAS version too low.");
  return {false, cublasLtMatmulAlgo_t{}};
#else
  size_t returnSize;
  int32_t pointer_mode;
  cublasLtMatmulDescGetAttribute(computeDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                 &pointer_mode, sizeof(pointer_mode),
                                 &returnSize);

  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(200);
  cublasLtMatmulPreference_t preference;
  check_cuda_error(cublasLtMatmulPreferenceCreate(&preference));
  check_cuda_error(cublasLtMatmulPreferenceInit(preference));
  uint64_t workspace_size = CUBLAS_WORKSPACE_SIZE;
  check_cuda_error(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size,
      sizeof(workspace_size)));
#if (CUBLAS_VERSION) <= 12000
  uint32_t pointer_mode_mask = 0;
  check_cuda_error(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK, &pointer_mode_mask,
      sizeof(pointer_mode_mask)));
#endif

  int return_count = 0;
  auto ret = cublasLtMatmulAlgoGetHeuristic(
      lightHandle, computeDesc, Adesc, Bdesc, Cdesc, Ddesc, preference,
      heuristics.size(), heuristics.data(), &return_count);
  heuristics.resize(return_count);

  std::map<int, std::vector<float>> algo_results;
  for (const auto &heuristic : heuristics) {
    cublasLtMatmulAlgo_t algo = heuristic.algo;
    int32_t algo_id;
    cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &returnSize);

    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    float my_alpha = 1.0f;
    float my_beta = 0.0f;

    for (int i = 0; i < 11; i++) {
      float duration_ms;
      cudaEventRecord(start_event, stream);
      check_cuda_error(cublasLtMatmul(
          lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc,
          D, Ddesc, &algo, cublas_workspace_, CUBLAS_WORKSPACE_SIZE, stream));
      cudaEventRecord(stop_event, stream);
      cudaEventSynchronize(stop_event);
      cudaEventElapsedTime(&duration_ms, start_event, stop_event);

      algo_results[algo_id].push_back(duration_ms);
    }
    std::sort(algo_results[algo_id].begin(), algo_results[algo_id].end());
  }

  cublasLtMatmulHeuristicResult_t result;
  float best_time = INFINITY;
  for (const auto &heuristic : heuristics) {
    cublasLtMatmulAlgo_t algo = heuristic.algo;
    int32_t algo_id;
    cublasLtMatmulAlgoConfigGetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &returnSize);
    const auto &results = algo_results[algo_id];

    if (results.size() > 0 && results[5] < best_time) {
      best_time = results[5];
      result = heuristic;
    }
  }

  return {best_time != INFINITY, result.algo};
#endif
}

cublasMMWrapper::MatrixLayout
cublasMMWrapper::createMatrixLayout(cublasLtMatrixLayout_t Mdesc) {
  size_t returnSize;
  MatrixLayout m_layout;

  cublasLtMatrixLayoutGetAttribute(Mdesc, CUBLASLT_MATRIX_LAYOUT_TYPE,
                                   &std::get<0>(m_layout),
                                   sizeof(std::get<0>(m_layout)), &returnSize);
  cublasLtMatrixLayoutGetAttribute(Mdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                   &std::get<1>(m_layout),
                                   sizeof(std::get<1>(m_layout)), &returnSize);
  cublasLtMatrixLayoutGetAttribute(Mdesc, CUBLASLT_MATRIX_LAYOUT_ROWS,
                                   &std::get<2>(m_layout),
                                   sizeof(std::get<2>(m_layout)), &returnSize);
  cublasLtMatrixLayoutGetAttribute(Mdesc, CUBLASLT_MATRIX_LAYOUT_COLS,
                                   &std::get<3>(m_layout),
                                   sizeof(std::get<3>(m_layout)), &returnSize);

  return m_layout;
}

cublasStatus_t cublasMMWrapper::cublasLtMatmulWrapper(
    cublasLtHandle_t lightHandle, cublasLtMatmulDesc_t computeDesc,
    const void *alpha, const void *A, cublasLtMatrixLayout_t Adesc,
    const void *B, cublasLtMatrixLayout_t Bdesc, const void *beta,
    const void *C, cublasLtMatrixLayout_t Cdesc, void *D,
    cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t *algo,
    void *workspace, size_t workspaceSizeInBytes, cudaStream_t stream) {
  cache_idx_t cache_idx{computeDesc,
                        {createMatrixLayout(Adesc), createMatrixLayout(Bdesc),
                         createMatrixLayout(Cdesc), createMatrixLayout(Ddesc)}};

  cublasLtMatmulAlgo_t algo_value;
  bool found_algo = false;
  if (algo == nullptr) {
    if (algo_cache.find(cache_idx) == algo_cache.end()) {
      auto result = findBestAlgo(lightHandle, computeDesc, alpha, A, Adesc, B,
                                 Bdesc, beta, C, Cdesc, D, Ddesc, stream);
      if (result.first) {
        algo_cache[cache_idx] = result.second;
        algo_value = result.second;
        found_algo = true;
      }
    } else {
      algo_value = algo_cache[cache_idx];
      found_algo = true;
    }
  }

  return cublasLtMatmul(lightHandle, computeDesc, alpha, A, Adesc, B, Bdesc,
                        beta, C, Cdesc, D, Ddesc,
                        found_algo ? &algo_value : algo, workspace,
                        workspaceSizeInBytes, stream);
}

void cublasMMWrapper::_Int8Gemm(const int m, const int n, const int k,
                                const int8_t *A, const int lda, const int8_t *B,
                                const int ldb, void *C, const int ldc,
                                const void *alpha, const int mode,
                                const bool per_column_scaling) {
/* mode:
 *  - 0: int8 * int8 -> int32 -> int8
 *  - 1: int8 * int8 -> int32 -> int32
 */
#if (CUBLAS_VERSION) < 11601
  FT_CHECK_WITH_INFO(false, "CUBLAS version too low.");
#else

  mu_->lock();
  const auto op_a = CUBLAS_OP_T;
  const auto op_b = CUBLAS_OP_N;
  const auto dataType = CUDA_R_8I;
  const auto resultType = mode == 0 ? CUDA_R_8I : CUDA_R_32I;
  const auto computeType = CUBLAS_COMPUTE_32I;
  const auto scaleType = mode == 0 ? CUDA_R_32F : CUDA_R_32I;
  const int batch_count = 1;
  const void *beta;

  int findAlgo = cublas_algo_map_->isExist(batch_count, m, n, k,
                                           getCublasDataType(dataType));

  cublasLtMatmulAlgo_info info = cublas_algo_map_->getAlgo(
      batch_count, m, n, k, getCublasDataType(dataType));

  cublasLtMatmulDesc_t operationDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;

  // --------------------------------------
  // Create descriptors for the original matrices
  check_cuda_error(cublasLtMatrixLayoutCreate(&Adesc, dataType, k, m, lda));
  check_cuda_error(cublasLtMatrixLayoutCreate(&Bdesc, dataType, k, n, ldb));
  check_cuda_error(cublasLtMatrixLayoutCreate(&Cdesc, resultType, m, n, ldc));

  check_cuda_error(
      cublasLtMatmulDescCreate(&operationDesc, computeType, scaleType));

  auto pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  if (mode == 0) {
    pointer_mode = per_column_scaling
                       ? CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST
                       : CUBLASLT_POINTER_MODE_DEVICE;
  }
  check_cuda_error(
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &op_a, sizeof(cublasOperation_t)));
  check_cuda_error(
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &op_b, sizeof(cublasOperation_t)));
  check_cuda_error(
      cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSC,
                                     &op_b, sizeof(cublasOperation_t)));
  check_cuda_error(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode,
      sizeof(pointer_mode)));

  const int32_t int_one = 1;
  const int32_t int_zero = 0;
  const float float_zero = 0;
  if (mode == 0) {
    beta = per_column_scaling ? &float_zero : NULL;
  } else {
    alpha = &int_one;
    beta = &int_zero;
  }

  cublasLtMatmulAlgo_t algo;
  void *workSpace = cublas_workspace_;
  int workspaceSize = cublas_workspace_ == NULL ? 0 : CUBLAS_WORKSPACE_SIZE;

  sync_check_cuda_error();
  auto ret = cublasLtMatmulWrapper(cublaslt_handle_, operationDesc, alpha, A,
                                   Adesc, B, Bdesc, beta, C, Cdesc, C, Cdesc,
                                   NULL, workSpace, workspaceSize, stream_);
  check_cuda_error(ret);
  sync_check_cuda_error();

  cublasLtMatmulDescDestroy(operationDesc);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  sync_check_cuda_error();
  mu_->unlock();
#endif
}

void cublasMMWrapper::Int8Gemm(const int m, const int n, const int k,
                               const int8_t *A, const int lda, const int8_t *B,
                               const int ldb, int8_t *C, const int ldc,
                               const float *alpha,
                               const bool per_column_scaling) {
  return _Int8Gemm(m, n, k, A, lda, B, ldb, C, ldc, alpha, 0,
                   per_column_scaling);
}

void cublasMMWrapper::Int8Gemm(const int m, const int n, const int k,
                               const int8_t *A, const int lda, const int8_t *B,
                               const int ldb, int32_t *C, const int ldc) {
  return _Int8Gemm(m, n, k, A, lda, B, ldb, C, ldc, (float *)nullptr, 1, false);
}
