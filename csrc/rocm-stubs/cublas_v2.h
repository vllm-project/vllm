#pragma once
#include <rocblas/rocblas.h>
#define cublasHandle_t rocblas_handle
#define cublasCreate rocblas_create_handle
#define cublasDestroy rocblas_destroy_handle
#define cublasSetStream rocblas_set_stream
#define cublasGetStream rocblas_get_stream
#define cublasSgemm rocblas_sgemm
#define cublasDgemm rocblas_dgemm
#define cublasHgemm rocblas_hgemm
#define cublasGemmEx rocblas_gemm_ex
