/*
 * Copyright 1993-2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <cstdio>             // printf
#include <cstdlib>            // std::rand
#include <vector>             // std::vector
#include <torch/extension.h>
#include <iostream>


#define INT8_OUTPUT_TYPE int32_t //at::Half //int8_t
#define INT8_OUTPUT_TYPE_CUDA CUDA_R_8I //CUDA_R_32I
#define INT8_OUTPUT_TYPE_TORCH torch::kInt32 //torch::kInt32


#define MAX(a, b) ((abs(a) > abs(b) ? (a) : (b)))
#define MIN(a, b) ((abs(a) < abs(b) ? (a) : (b)))


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


#define CHECK_CUDA_TORCH(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return torch::ones(1);                                                   \
    }                                                                          \
}


#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


#define CHECK_CUSPARSE_TORCH(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return torch::ones(1);                                                   \
    }                                                                          \
}

constexpr int EXIT_UNSUPPORTED = 2;

cusparseLtHandle_t handle;

float alpha = 1.0;
float beta  = 0.0;


typedef struct {
   at::Half data;
   int index;
} indexed_half;


int init_cusparse_lt_cuda()
{
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6) &&
        !(major_cc == 8 && minor_cc == 9)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6, 8.9 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }
    CHECK_CUSPARSE( cusparseLtInit(&handle) )

    return EXIT_SUCCESS;
}


typedef struct cusparseLtMatmulArgs_t {
    cusparseLtMatmulPlan_t*         plan;
    cusparseLtMatmulDescriptor_t*   matmul;
    cusparseLtMatmulAlgSelection_t* alg_sel;
    cudaStream_t*                   streams;
    int                             num_streams;
    cudaStream_t                    stream;
    size_t                          workspace_size;
//     void*                           d_workspace;
    void                            *dCompressed;
    int                             m;
    int                             n;
//     torch::Tensor                   grad;

    cusparseLtMatmulArgs_t()
    {
        plan = new cusparseLtMatmulPlan_t;
        matmul = new cusparseLtMatmulDescriptor_t;
        alg_sel = new cusparseLtMatmulAlgSelection_t;
        streams = nullptr;
        num_streams = 0;
        stream = nullptr;
        m = 0;
        n = 0;
        dCompressed = nullptr;
    }

    ~cusparseLtMatmulArgs_t()
    {
        cusparseLtMatmulPlanDestroy(plan);
//         cudaFree(d_workspace);
    }
} cusparseLtMatmulArgs ;


std::vector<cusparseLtMatmulArgs*> matmul_args;


template <class T, class V>
int setup_prune_matmul( const int                       m,
                        const int                       n,
                        const int                       k,
                        T                               *dSparse,
                        T                               *dDense,
                        int                             *index,
                        const bool                      transpose_A=false,
                        const bool                      transpose_B=false,
                        const bool                      sparseA=true,
                        const bool                      transposable_mask=false,
                        const bool                      is_sparse_pruned=false,
                        const bool                      check_sparsity=false,
                        cudaDataType_t                  input_type=CUDA_R_16F,
                        cudaDataType_t                  output_type=CUDA_R_16F,
                        cusparseComputeType             compute_type=CUSPARSE_COMPUTE_16F)
{
    matmul_args.push_back(new cusparseLtMatmulArgs_t);
    *index = matmul_args.size() - 1;

    auto args = matmul_args.back();
    args->m = m;
    args->n = n;

    // Host problem definition, row-major order
    // bigger sizes may require dynamic allocations
    auto          order        = CUSPARSE_ORDER_ROW;
    auto          opA          = transpose_A ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB          = transpose_B ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;

    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows     = (isA_transposed) ? k : m;
    auto     num_A_cols     = (isA_transposed) ? m : k;
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 16;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
    auto     C_size         = C_height * ldc * sizeof(V);


    cusparseLtMatDescriptor_t*      matA;
    cusparseLtMatDescriptor_t*      matB;
    cusparseLtMatDescriptor_t*      matC;
    matA = new cusparseLtMatDescriptor_t;
    matB = new cusparseLtMatDescriptor_t;
    matC = new cusparseLtMatDescriptor_t;

    V *dC, *dD;
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    dD = dC;

    int *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )

    // matrix descriptor initialization
    if(sparseA)
    {
        CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                                &handle, matA, num_A_rows,
                                                num_A_cols, lda, alignment,
                                                input_type, order,
                                                CUSPARSELT_SPARSITY_50_PERCENT) )

        CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                                &handle, matB, num_B_rows,
                                                num_B_cols, ldb, alignment,
                                                input_type, order) )
    }
    else
    {
        CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                                &handle, matB, num_B_rows,
                                                num_B_cols, ldb, alignment,
                                                input_type, order,
                                                CUSPARSELT_SPARSITY_50_PERCENT) )

        CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                                &handle, matA, num_A_rows,
                                                num_A_cols, lda, alignment,
                                                input_type, order) )
    }
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            output_type, order) )

    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, args->matmul, opA, opB,
                                            matA, matB, matC, matC,
                                            compute_type) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, args->alg_sel, args->matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )

    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, args->plan, args->matmul, args->alg_sel))

    //--------------------------------------------------------------------------
    // Prune the A matrix (in-place) and check the correctness
    if (!is_sparse_pruned){
        cusparseLtPruneAlg_t prune_alg = transposable_mask ? CUSPARSELT_PRUNE_SPMMA_TILE : CUSPARSELT_PRUNE_SPMMA_STRIP;
        CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, args->matmul, dSparse, dSparse,
                                             prune_alg, args->stream) )
    }
    if (check_sparsity)
    {
        CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, args->matmul, dSparse, d_valid, args->stream) )
        int is_valid;
        CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost, args->stream) )
        CHECK_CUDA( cudaStreamSynchronize(args->stream) )
        if (is_valid != 0) {
            std::printf("!!!! The matrix does not conform to the SpMMA sparsity pattern. "
                        "cusparseLtMatmul does not provide correct results\n");
            return EXIT_FAILURE;
        }
    }
    

//     int    *d_valid;
//     CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )
//     CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck2(    &handle,
//                                                     sparseA ? matA : matB,
//                                                     sparseA,
//                                                     sparseA ? opA : opB,
//                                                     dSparse,
//                                                     d_valid,
//                                                     args->stream) )

//     int is_valid;
//     CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
//                                 cudaMemcpyDeviceToHost, args->stream) )
//     CHECK_CUDA( cudaStreamSynchronize(args->stream) )
//     if (is_valid != 0) {
//         std::printf("!!!! The matrix has been pruned in a wrong way. "
//                     "cusparseLtMatmul will not provide correct results\n");
//         return EXIT_FAILURE;
//     }
    CHECK_CUDA( cudaFree(d_valid) )

    //--------------------------------------------------------------------------
    // Compress the A matrix
    size_t compressed_size, compressed_buffer_size;
    void*  dCompressedBuffer;
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle,
                                                  args->plan,
                                                  &compressed_size,
                                                  &compressed_buffer_size) )

    CHECK_CUDA( cudaMalloc((void**) &args->dCompressed, compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dCompressedBuffer,
                           compressed_buffer_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle,
                                            args->plan,
                                            dSparse,
                                            (T *) args->dCompressed,
                                            dCompressedBuffer,
                                            args->stream) )
    CHECK_CUDA( cudaFree(dCompressedBuffer) )

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    if(sparseA)
    {
//         printf("%f, %f, %f, %f, %f, %f\n", alpha, beta, *dDense,0.,0.,0.);// , dDense[0], beta, dC[0], dD[0]);
        CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, args->plan, &alpha,
                                            (T*) args->dCompressed, dDense, &beta,
                                            dC, dD, nullptr,
                                            args->streams, args->num_streams) )
    } else {
        CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, args->plan, &alpha,
                                            dDense, (T*) args->dCompressed, &beta,
                                            dC, dD, nullptr,
                                            args->streams, args->num_streams) )
    }
//     // otherwise, it is possible to set it directly:
//     int alg = 0;
//     CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
//                                            &handle, args->alg_sel,
//                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
//                                            &alg, sizeof(alg)))


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, args->plan, args->matmul, args->alg_sel))

    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, args->plan,
                                                 &args->workspace_size))

//     printf("workspace_size: %lu (MB)\n", args->workspace_size / 1024 / 1024);
    CHECK_CUDA( cudaFree(dC) )
    cusparseLtMatDescriptorDestroy(matA);
    cusparseLtMatDescriptorDestroy(matB);
    cusparseLtMatDescriptorDestroy(matC);

    return EXIT_SUCCESS;
}

int destroy_cusparse_matmul_cuda(int index){
    if (index > matmul_args.size() - 1)
        throw std::runtime_error("Index out of range of matmul_args");

    auto args = matmul_args[index];
    cusparseLtMatmulPlanDestroy(args->plan);
    CHECK_CUDA(cudaFree(args->streams));
    CHECK_CUDA(cudaFree(args->dCompressed));
    matmul_args.erase(matmul_args.begin() + index);

    return EXIT_SUCCESS;
}

torch::Tensor setup_spmatmul_cuda(torch::Tensor A,
                                torch::Tensor B,
                                const bool transpose_A=false,
                                const bool transpose_B=false,
                                const bool sparseA=true,
                                const bool transposable_mask=false,
                                const bool is_sparse_pruned=false,
                                const bool check_sparsity=false) {
   auto index = torch::zeros({1}, torch::kInt32);
   int result;
   int m, k, n;
   if(transpose_A && transpose_B)
   {
        m = A.size(1);
        k = A.size(0);
        n = B.size(0);
   } else if(transpose_A)
   {
        m = A.size(1);
        k = A.size(0);
        n = B.size(1);
   } else if(transpose_B)
   {
        m = A.size(0);
        k = A.size(1);
        n = B.size(0);
   } else {
        m = A.size(0);
        k = A.size(1);
        n = B.size(1);
   }
   switch (A.type().scalarType()) {
        case torch::ScalarType::Half:
        {
            auto sparse_mat = sparseA ? A.data_ptr<at::Half>() : B.data_ptr<at::Half>();
            auto dense_mat = sparseA ? B.data_ptr<at::Half>() : A.data_ptr<at::Half>();
            at::Half *dCompressed;
            result = setup_prune_matmul<at::Half, at::Half>(     m,
                                             n,
                                             k,
                                             sparse_mat,
                                             dense_mat,
                                             index.data_ptr<int>(),
                                             transpose_A,
                                             transpose_B,
                                             sparseA,
                                             transposable_mask,
                                             is_sparse_pruned,
                                             check_sparsity,
                                             CUDA_R_16F,
                                             CUDA_R_16F,
                                             CUSPARSE_COMPUTE_16F);
            break;
        }
        case torch::ScalarType::Char:
        {
            auto sparse_mat = sparseA ? A.data_ptr<int8_t>() : B.data_ptr<int8_t>();
            auto dense_mat = sparseA ? B.data_ptr<int8_t>() : A.data_ptr<int8_t>();
            int8_t *dCompressed;
            result = setup_prune_matmul<int8_t, INT8_OUTPUT_TYPE>(     m,
                                             n,
                                             k,
                                             sparse_mat,
                                             dense_mat,
                                             index.data_ptr<int>(),
                                             transpose_A,
                                             transpose_B,
                                             sparseA,
                                             transposable_mask,
                                             is_sparse_pruned,
                                             check_sparsity,
                                             CUDA_R_8I,
                                             INT8_OUTPUT_TYPE_CUDA,
                                             CUSPARSE_COMPUTE_32I);
            break;}
        default:
        {
            std::cout << A.type().scalarType() << std::endl;
            throw std::runtime_error("Unsupported data type");
        }
   }
   if(result == EXIT_SUCCESS) {
     return index;
   } else {
     return -torch::ones({1}, torch::kInt32);
   }
}


template <class T, class V>
torch::Tensor matmul(   T* dDense,
                        int index,
                        bool sparseA,
                        int m,
                        torch::TensorOptions options=torch::TensorOptions()
                    )
{
    auto args = matmul_args[index];

    torch::Tensor C = torch::zeros({m, args->n}, options);
    auto dC = C.data_ptr<V>();
    auto dD = dC;
    auto dA = sparseA ? (T*) args->dCompressed : dDense;
    auto dB = sparseA ? dDense : (T*) args->dCompressed;
    void *d_workspace;
    CHECK_CUDA_TORCH( cudaMalloc((void**) &d_workspace, args->workspace_size) )
    // Perform the matrix multiplication
    CHECK_CUSPARSE_TORCH( cusparseLtMatmul(&handle, args->plan, &alpha, dA, dB,
                                     &beta, dC, dD, d_workspace, args->streams,
                                     args->num_streams) )
    CHECK_CUDA_TORCH( cudaFree(d_workspace) )
    return C;
}


torch::Tensor spmatmul_cuda(torch::Tensor   Dense,
                            int             index,
                            bool            sparseA)
{
    switch (Dense.type().scalarType()) {
        case torch::ScalarType::Half: {
            auto options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
            return matmul<at::Half, at::Half>(Dense.data_ptr<at::Half>(), index, sparseA, Dense.size(0), options);
        }
        case torch::ScalarType::Char: {
            auto options = torch::TensorOptions().dtype(INT8_OUTPUT_TYPE_TORCH).device(torch::kCUDA);
            return matmul<int8_t, INT8_OUTPUT_TYPE>(Dense.data_ptr<int8_t>(), index, sparseA, Dense.size(0), options);
        }
        default:
        {
            throw std::runtime_error("Unsupported data type");
        }
    }
}


void save_grad_cuda(torch::Tensor grad, int index)
{
    auto args = matmul_args[index];
//    args->grad = grad.clone().detach();
}


__global__ void prune_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size) {
    const int column = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        reinterpret_cast<float4*>(&output[index])[0] = reinterpret_cast<const float4*>(&input[index])[0];
        if(abs(output[index]) > abs(output[index + 1])){
            output[index + 1] = 0.;
            mask[index + 1] = true;
        } else {
            output[index] = 0.;
            mask[index] = true;
        }
        if(abs(output[index + 2]) > abs(output[index + 3])){
            output[index + 3] = 0.;
            mask[index + 3] = true;
        } else {
            output[index + 2] = 0.;
            mask[index + 2] = true;
        }
  }
}


__global__ void prune_kernel(
        const at::Half* __restrict__ input,
        at::Half* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size) {
    const int column = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        reinterpret_cast<float4*>(&output[index])[0] = reinterpret_cast<const float4*>(&input[index])[0];
        at::Half min1, min2;
        int min_idx1, min_idx2;
        min1 = output[index];
        min_idx1 = index;
        if(MIN(min1, output[index + 1]) == output[index + 1]){
            min1 = output[index + 1];
            min_idx1 = index + 1;
        }
        if(MIN(min1, output[index + 2]) == output[index + 2]){
            min1 = output[index + 2];
            min_idx1 = index + 2;
        }
        if(MIN(min1, output[index + 3]) == output[index + 3]){
            min1 = output[index + 3];
            min_idx1 = index + 3;
        }
        min2 = min_idx1 == index ? output[index + 1] : output[index];
        min_idx2 = min_idx1 == index ? index + 1 : index;
        if((MIN(min2, output[index + 1]) == output[index + 1]) && min_idx1 != index + 1){
            min2 = output[index + 1];
            min_idx2 = index + 1;
        }
        if((MIN(min2, output[index + 2]) == output[index + 2]) && min_idx1 != index + 2){
            min2 = output[index + 2];
            min_idx2 = index + 2;
        }
        if((MIN(min2, output[index + 3]) == output[index + 3]) && min_idx1 != index + 3){
            min2 = output[index + 3];
            min_idx2 = index + 3;
        }
        output[min_idx1] = 0.; mask[min_idx1] = true;
        output[min_idx2] = 0.; mask[min_idx2] = true;

        min1 = output[index + 4];
        min_idx1 = index + 4;
        if(MIN(min1, output[index + 5]) == output[index + 5]){
            min1 = output[index + 5];
            min_idx1 = index + 5;
        }
        if(MIN(min1, output[index + 6]) == output[index + 6]){
            min1 = output[index + 6];
            min_idx1 = index + 6;
        }
        if(MIN(min1, output[index + 7]) == output[index + 7]){
            min1 = output[index + 7];
            min_idx1 = index + 7;
        }
        min2 = min_idx1 == index + 4 ? output[index + 5] : output[index + 4];
        min_idx2 = min_idx1 == index + 4 ? index + 5 : index + 4;
        if((MIN(min2, output[index + 5]) == output[index + 5]) && min_idx1 != index + 5){
            min2 = output[index + 5];
            min_idx2 = index + 5;
        }
        if((MIN(min2, output[index + 6]) == output[index + 6]) && min_idx1 != index + 6){
            min2 = output[index + 6];
            min_idx2 = index + 6;
        }
        if((MIN(min2, output[index + 7]) == output[index + 7]) && min_idx1 != index + 7){
            min2 = output[index + 7];
            min_idx2 = index + 7;
        }

        output[min_idx1] = 0.; mask[min_idx1] = true;
        output[min_idx2] = 0.; mask[min_idx2] = true;
  }
}


template <class T>
__device__ void find_kth_smallest(
                                    int *smallest_idx,
                                    const T* __restrict__ input,
                                    const int k,
                                    const int M, int index) {
    int min_idx = 0;
    T min = 6.0e4;

    for(int i = 0; i < M; i++)
    {
        bool ignore = false;
        for(int j = 0; j < k; j++)
        {
            if(smallest_idx[j] == i)
            {
                ignore = true;
            }
        }
        if(ignore)
        {
            continue;
        }
        if(MIN(min, input[i]) == input[i]){
            min = input[i];
            min_idx = i;
        }
    }
    smallest_idx[k] = min_idx;
}


__global__ void prune_kernel(
        const at::Half* __restrict__ input,
        at::Half* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size,
        const int N,
        const int M) {

    const int column = M * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        for(int i = 0; i < M / 8; i++)
        {
            reinterpret_cast<float4*>(&output[index + 8 * i])[0] = reinterpret_cast<const float4*>(&input[index + 8 * i])[0];
        }

        int min_idx_list[16];
        for(int k = 0; k < (M - N); k++)
        {
            find_kth_smallest<at::Half>(min_idx_list, &input[index], k, M, index);
        }

        for(int i = 0; i < (M - N); i++)
        {
            output[min_idx_list[i] + index] = 0.; mask[min_idx_list[i] + index] = true;
        }
  }
}


__global__ void prune_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size,
        const int N,
        const int M) {

    const int column = M * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        for(int i = 0; i < M / 4; i++)
        {
            reinterpret_cast<float4*>(&output[index + 4 * i])[0] = reinterpret_cast<const float4*>(&input[index + 4 * i])[0];
        }

        int *min_idx_list;
        min_idx_list = (int*)malloc((M - N) * sizeof(int));
        for(int k = 0; k < (M - N); k++)
        {
            find_kth_smallest<float>(min_idx_list, &input[index], k, M, index);
        }

        for(int i = 0; i < (M - N); i++)
        {
            output[min_idx_list[i] + index] = 0.; mask[min_idx_list[i] + index] = true;
        }
  }
}


template <int N, int M>
__global__ void prune_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size) {

    const int column = M * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        for(int i = 0; i < M / 4; i++)
        {
            reinterpret_cast<float4*>(&output[index + 4 * i])[0] = reinterpret_cast<const float4*>(&input[index + 4 * i])[0];
        }

        int min_idx_list[M - N];
        for(int k = 0; k < (M - N); k++)
        {
            find_kth_smallest<float>(min_idx_list, &input[index], k, M, index);
        }

        for(int i = 0; i < (M - N); i++)
        {
            output[min_idx_list[i] + index] = 0.; mask[min_idx_list[i] + index] = true;
        }
  }
}


template <int N, int M>
__global__ void prune_kernel(
        const at::Half* __restrict__ input,
        at::Half* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size) {

    const int column = M * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        for(int i = 0; i < M / 8; i++)
        {
            reinterpret_cast<float4*>(&output[index + 8 * i])[0] = reinterpret_cast<const float4*>(&input[index + 8 * i])[0];
        }

        int min_idx_list[M - N];
        for(int k = 0; k < (M - N); k++)
        {
            find_kth_smallest<at::Half>(min_idx_list, &input[index], k, M, index);
        }

        for(int i = 0; i < (M - N); i++)
        {
            output[min_idx_list[i] + index] = 0.; mask[min_idx_list[i] + index] = true;
        }
  }
}


std::vector<torch::Tensor> prune_cuda(
    torch::Tensor input, const int N, const int M) {

    auto output = torch::zeros_like(input);
    auto options = torch::TensorOptions().dtype(torch::kBool);
    auto mask = torch::zeros_like(input, options);

    const auto batch_size = input.size(0);
    const auto row_size = input.size(1);

    const int threads = 1024;

    if(N == 1 && M == 2) {
        switch (input.type().scalarType()) {
            case torch::ScalarType::Float: {
                const dim3 blocks(((row_size / 4) + threads - 1) / threads, batch_size);
                prune_kernel<<<blocks, threads>>>(
                        input.data<float>(),
                        output.data<float>(),
                        mask.data<bool>(),
                        row_size);
                break;
            }
            case torch::ScalarType::Half: {
                throw std::runtime_error("Half precision not supported for N=1, M=2");
            }
        }
    }
    else if(N == 2 && M == 4)
    {
            switch (input.type().scalarType()) {
                case torch::ScalarType::Float: {
                    throw std::runtime_error("Full precision not supported for N=2, M=4");
                    break;
                }
                case torch::ScalarType::Half: {
                    const dim3 blocks(((row_size / 8) + threads - 1) / threads, batch_size);
                    prune_kernel<<<blocks, threads>>>(
                            input.data<at::Half>(),
                            output.data<at::Half>(),
                            mask.data<bool>(),
                            row_size);
                }
            }
    }
    else if((N == 2 && M == 8))
    {
        switch (input.type().scalarType()){
            case torch::ScalarType::Float: {
            const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
            prune_kernel<2, 8><<<blocks, threads>>>(
                    input.data<float>(),
                    output.data<float>(),
                    mask.data<bool>(),
                    row_size);
            break;
            }
            case torch::ScalarType::Half: {
                const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
                prune_kernel<2, 8><<<blocks, threads>>>(
                        input.data<at::Half>(),
                        output.data<at::Half>(),
                        mask.data<bool>(),
                        row_size);
            }
        }
    }
    else if((N == 2 && M == 16))
    {
        switch (input.type().scalarType()){
            case torch::ScalarType::Float: {
            const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
            prune_kernel<2, 16><<<blocks, threads>>>(
                    input.data<float>(),
                    output.data<float>(),
                    mask.data<bool>(),
                    row_size);
            break;
            }
            case torch::ScalarType::Half: {
                const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
                prune_kernel<2, 16><<<blocks, threads>>>(
                        input.data<at::Half>(),
                        output.data<at::Half>(),
                        mask.data<bool>(),
                        row_size);
            }
        }
    }
    else
    {
        if(M < 8 || M % 8 != 0)
        {
            throw std::runtime_error("M must be a multiple of 8");
        }
        switch (input.type().scalarType()) {
            case torch::ScalarType::Float:
            {
                const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
                prune_kernel<<<blocks, threads>>>(
                    input.data<float>(),
                    output.data<float>(),
                    mask.data<bool>(),
                    row_size,
                    N,
                    M);
                 break;
            }
            case torch::ScalarType::Half:
            {
                const dim3 blocks(((row_size / M) + threads - 1) / threads, batch_size);
                prune_kernel<<<blocks, threads>>>(
                    input.data<at::Half>(),
                    output.data<at::Half>(),
                    mask.data<bool>(),
                    row_size,
                    N,
                    M);
            }
        }
    }
  return {output, mask};
}


__global__ void prune_and_compress_kernel(
        const at::Half* __restrict__ input,
        at::Half* __restrict__ output,
        bool* __restrict__ mask,
        size_t row_size) {
    const int input_column = 16 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int output_column = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int input_row = blockIdx.y * row_size;
    const int output_row = blockIdx.y * (row_size / 2);
    const int input_index = input_row + input_column;
    const int output_index = output_row + output_column;
    if (input_column < row_size) {
        bool local_mask[16];
        reinterpret_cast<float4*>(local_mask)[0] = reinterpret_cast<const float4*>(&mask[input_index])[0];

        int local_index = 0;
        #pragma unroll (2)
        for(int i = 0; i < 2; i++)
        {
            at::Half local_data[8];
            reinterpret_cast<float4*>(local_data)[0] = reinterpret_cast<const float4*>(&input[input_index + 8 * i])[0];
            #pragma unroll (8)
            for(int j = 0; j < 8; j++)
            {
                if(local_mask[8 * i + j])
                {
                    output[local_index + output_index] = local_data[j];
                    local_index++;
                }
            }
        }
    }
}


torch::Tensor prune_and_compress_cuda(torch::Tensor dense, torch::Tensor mask)
{
    auto row_size = dense.size(1);
    auto batch_size = dense.size(0);
    if(row_size % 16 != 0)
    {
        throw std::runtime_error("Pruning dimension should be a multiple of 128.");
    }
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
    torch::Tensor result = torch::zeros({dense.size(0), dense.size(1) / 2}, options);
    const int threads = 1024;
    switch (dense.type().scalarType()) {
        case torch::ScalarType::Float:
        {
            throw std::runtime_error("Full precision not supported for prune_and_compress");
        }
        case torch::ScalarType::Half:
        {
            const dim3 blocks(((row_size / 16) + threads - 1) / threads, batch_size);
            prune_and_compress_kernel<<<blocks, threads>>>(
                dense.data<at::Half>(),
                result.data<at::Half>(),
                mask.data<bool>(),
                row_size);
        }
    }
    return result;
}


__global__ void sparse_add_kernel(
        const at::Half* __restrict__ mat1,
        const at::Half* __restrict__ mat2,
        const at::Half alpha,
        const at::Half beta,
        at::Half* __restrict__ output,
        size_t row_size) {
    const int column = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        at::Half mat1_local[8], mat2_local[8];
        reinterpret_cast<float4 *>(&mat1_local)[0] = reinterpret_cast<const float4 *>(&mat1[index])[0];
        reinterpret_cast<float4 *>(&mat2_local)[0] = reinterpret_cast<const float4 *>(&mat2[index])[0];
        #pragma unroll (8)
        for(int i = 0; i < 8; i++)
        {
            output[index + i] = alpha * mat1_local[i] + beta * mat2_local[i];
        }
    }

}


torch::Tensor sparse_add_cuda(torch::Tensor dense, torch::Tensor sparse_index, torch::Tensor alpha, torch::Tensor beta)
{
    int row_size = dense.size(1);
    int batch_size = dense.size(0);
    if(row_size % 8 != 0)
    {
        throw std::runtime_error("Pruning dimension should be a multiple of 8.");
    }
    int index = sparse_index.item<int>();
    auto args = matmul_args[index];
    torch::Tensor result = torch::zeros_like(dense);
    const int threads = 1024;
    switch (dense.type().scalarType()) {
        case torch::ScalarType::Float:
        {
            throw std::runtime_error("Full precision not supported for prune_and_compress");
        }
        case torch::ScalarType::Half:
        {
            const dim3 blocks(((row_size / 8) + threads - 1) / threads, batch_size);
            sparse_add_kernel<<<blocks, threads>>>(
                dense.data<at::Half>(),
                (at::Half*) args->dCompressed,
                alpha.item<float>(),
                beta.item<float>(),
                result.data<at::Half>(),
                row_size);
        }
    }
    return result;
}


__global__ void update_sparse_matrix_kernel(
        const at::Half* __restrict__ new_data,
        at::Half* __restrict__ output,
        size_t row_size) {
    const int column = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int index = blockIdx.y * row_size + column;
    if (column < row_size) {
        reinterpret_cast<float4 *>(&output[index])[0] = reinterpret_cast<const float4 *>(&new_data[index])[0];
    }
}


void update_sparse_matrix_cuda(torch::Tensor new_data, torch::Tensor sparse_idx)
{
    auto args = matmul_args[sparse_idx.item<int>()];
    const int threads = 1024;
    switch (new_data.type().scalarType()) {
        case torch::ScalarType::Float:
        {
            throw std::runtime_error("Full precision not supported for prune_and_compress");
        }
        case torch::ScalarType::Half:
        {
            cudaMemcpy(args->dCompressed, new_data.data<at::Half>(), new_data.size(0) * new_data.size(1) * sizeof(at::Half), cudaMemcpyDeviceToDevice);
        }
    }
}


// sparse = prune_and_compress(dense, mask)
// result = add_sparse_dense(sparse_idx, dense, alpha, beta)
// update_sparse(data, sparse_idx, sparse_transpose_idx)
