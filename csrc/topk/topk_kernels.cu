#include "ATen/core/TensorBody.h"
#include "kernel_utils.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <cstdio>
#include <cub/cub.cuh>
#include <stdexcept>
#include <stdio.h>
#include <torch/extension.h>

template<typename T, int BLOCK_SIZE, int BLOCKS_PER_SEQ>
__global__ void topk_stage1(const T* __restrict src,
                            T*         tmp_log_probs,
                            int*       topk_tmp_id_buf,
                            T*         topk_tmp_val_buf,
                            const int  max_top_k,
                            const int* top_ks,
                            const int  vocab_size)
{
    typedef cub::BlockReduce<TopK<T>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage  temp_storage;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int batch_id = bid / BLOCKS_PER_SEQ;  // row id for log_probs

    const int block_lane = bid % BLOCKS_PER_SEQ;  // block id for a beam
    const int k          = (top_ks != nullptr) ? top_ks[batch_id] : vocab_size;
    // const int k          = max_top_k;             // batch_id = batch index

    const int tmp_log_buf_index  = batch_id * vocab_size;
    const int tmp_topk_buf_index = batch_id * BLOCKS_PER_SEQ * max_top_k + block_lane * k;

    TopK<T>    partial;
    const bool IS_FP16   = std::is_same<T, half>::value;
    const T    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    for (int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size; elem_id += BLOCK_SIZE * BLOCKS_PER_SEQ) {
        int index            = elem_id + tmp_log_buf_index;
        tmp_log_probs[index] = src[index];
    }

    for (int ite = 0; ite < k; ite++) {
        partial.init();
#pragma unroll
        for (int elem_id = tid + block_lane * BLOCK_SIZE; elem_id < vocab_size;
             elem_id += BLOCK_SIZE * BLOCKS_PER_SEQ) {
            int index = elem_id + tmp_log_buf_index;
            partial.insert(tmp_log_probs[index], index);
        }

        TopK<T> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T>);

        if (tid == 0) {
            const int index         = tmp_topk_buf_index + ite;
            topk_tmp_id_buf[index]  = total.p;
            topk_tmp_val_buf[index] = total.u;
            tmp_log_probs[total.p]  = -MAX_T_VAL;
        }
        __syncthreads();
    }
}

template<typename T, int BLOCK_SIZE, int BLOCKS_PER_SEQ>
__global__ void topk_stage2_sampling(const int* __restrict topk_tmp_id_buf,
                                     T*           topk_tmp_val_buf,
                                     const T*     src,
                                     T*           dst,
                                     const int    max_top_k,
                                     const int*   top_ks,
                                     const float* top_ps,
                                     const int    vocab_size)
{
    const bool IS_FP16   = std::is_same<T, half>::value;
    const T    MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    const int  tid       = threadIdx.x;
    const int  batch_id  = blockIdx.x;

    const int   k              = (top_ks != nullptr) ? top_ks[batch_id] : vocab_size;
    const float prob_threshold = (top_ps != nullptr) ? top_ps[batch_id] : 1.0;
    const int   size           = k * BLOCKS_PER_SEQ;
    const int   stride         = max_top_k * BLOCKS_PER_SEQ;

    typedef cub::BlockReduce<TopK<float>, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage      temp_storage;
    __shared__ float                                  s_sum;
    T*                                                s_val = topk_tmp_val_buf + batch_id * stride;

    if (tid == 0) {
        s_sum = 0.0f;
    }
    TopK<float> partial;
    for (int ite = 0; ite < k; ite++) {
        partial.init();
#pragma unroll
        for (int i = tid; i < size; i += BLOCK_SIZE) {
            partial.insert((float)s_val[i], i);
        }
        TopK<float> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<float>);
        if (tid == 0) {
            s_val[total.p]                                    = -MAX_T_VAL;
            dst[topk_tmp_id_buf[batch_id * stride + total.p]] = src[topk_tmp_id_buf[batch_id * stride + total.p]];
            s_sum += total.u;
            if (s_sum >= prob_threshold) {
                break;
            }
        }
        __syncthreads();
    }
}

#define CASE_K(K_MIN, K_MAX, BLOCK_SIZE, BLOCK_PER_SEQ)                                                                \
    case K_MIN ... K_MAX:                                                                                              \
        topk_stage1<T, BLOCK_SIZE, BLOCK_PER_SEQ><<<batch_size * BLOCK_PER_SEQ, BLOCK_SIZE, 0, stream>>>(              \
            softmax_src, temp_log_probs, topk_tmp_id_buf, topk_tmp_val_buf, max_top_k, top_ks, vocab_size);            \
        topk_stage2_sampling<T, BLOCK_SIZE, BLOCK_PER_SEQ><<<batch_size, BLOCK_SIZE, 0, stream>>>(                     \
            topk_tmp_id_buf, topk_tmp_val_buf, src, dst, max_top_k, top_ks, top_ps, vocab_size);                       \
        break;

template<typename T>
void invokeBatchTopKSampling(void*        workspace,
                             size_t&      workspace_size,
                             const T*     src,
                             const T*     softmax_src,
                             T*           dst,
                             const int    max_top_k,
                             const int*   top_ks,
                             const float* top_ps,
                             const int    vocab_size,
                             const int    batch_size,
                             cudaStream_t stream)
{
    const int max_block_per_seq       = 8;
    int       temp_log_probs_buf_size = batch_size * vocab_size;                     // type float
    int       topk_tmp_ids_buf_size   = batch_size * max_top_k * max_block_per_seq;  // type int
    int       topk_tmp_val_buf_size   = batch_size * max_top_k * max_block_per_seq;  // type float
    // prevent memory misaligned address
    temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
    topk_tmp_ids_buf_size   = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
    topk_tmp_val_buf_size   = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

    if (workspace == nullptr) {
        workspace_size = sizeof(T) * temp_log_probs_buf_size + sizeof(int) * topk_tmp_ids_buf_size
                         + sizeof(T) * topk_tmp_val_buf_size;
        return;
    }

    T*   temp_log_probs   = (T*)workspace;
    int* topk_tmp_id_buf  = (int*)(temp_log_probs + temp_log_probs_buf_size);
    T*   topk_tmp_val_buf = (T*)(topk_tmp_id_buf + topk_tmp_ids_buf_size);

    switch (max_top_k) {
        CASE_K(1, 16, 128, 8);
        CASE_K(17, 32, 256, 8);
        CASE_K(33, 64, 256, 8);
        CASE_K(65, 1024, 256, 8);
        default:
            break;
    }
    // cudaDeviceSynchronize();
}

template<typename T>
void top_k_cuda(const T*     src,
                const T*     softmax_src,
                T*           dst,
                const int    max_top_k,
                const int*   top_ks,
                const float* top_ps,
                int          batch_size,
                int          vocab_size,
                cudaStream_t stream)
{

    size_t workspace_size = 0;
    void*  workspace      = nullptr;
    invokeBatchTopKSampling<T>(nullptr,
                               workspace_size,
                               nullptr,
                               nullptr,
                               nullptr,
                               max_top_k,
                               nullptr,
                               nullptr,
                               vocab_size,
                               batch_size,
                               stream);

    cudaMalloc(&workspace, workspace_size);

    invokeBatchTopKSampling<T>(
        workspace, workspace_size, src, softmax_src, dst, max_top_k, top_ks, top_ps, vocab_size, batch_size, stream);

    cudaFree(workspace);
}
void top_k(const torch::Tensor src,
           const torch::Tensor softmax_src,
           torch::Tensor       dst,
           bool                top_k,
           int                 max_top_k,
           torch::Tensor       top_ks,
           bool                top_p,
           torch::Tensor       top_ps)
{
    float*       src_ptr         = reinterpret_cast<float*>(src.data_ptr());
    float*       softmax_src_ptr = reinterpret_cast<float*>(softmax_src.data_ptr());
    float*       dst_ptr         = reinterpret_cast<float*>(dst.data_ptr());
    int*         top_ks_ptr      = nullptr;
    float*       top_ps_ptr      = nullptr;
    auto         shape           = src.sizes();
    auto         shape_len       = shape.size();
    unsigned int batch_size      = src.sizes()[0];
    unsigned int vocab_size      = shape[shape_len - 1];
    if (top_k) {
        top_ks_ptr = reinterpret_cast<int*>(top_ks.data_ptr());
    }
    else {
        max_top_k = 1024;
    }
    if (top_p) {
        top_ps_ptr = reinterpret_cast<float*>(top_ps.data_ptr());
    }
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    top_k_cuda(src_ptr, softmax_src_ptr, dst_ptr, max_top_k, top_ks_ptr, top_ps_ptr, batch_size, vocab_size, stream);
}
