#include <c10/cuda/CUDAGuard.h>
#include <stdio.h>
#include <torch/extension.h>
#include <cstdio>
#include <cub/cub.cuh>
#include <stdexcept>
#include <vector>
#include "ATen/core/TensorBody.h"
#include "kernel_utils.cuh"

__global__ void curandInitialize(curandState_t *state, const int size,
                                 const unsigned long long random_seed) {
  if (threadIdx.x + blockIdx.x * blockDim.x < size) {
    curand_init(random_seed, 0, 0,
                &state[blockIdx.x * blockDim.x + threadIdx.x]);
  }
}

void invokeCurandInitialize(curandState_t *state, const size_t batch_size,
                            const unsigned long long random_seed,
                            cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((int)(ceil(batch_size * 1.0 / 256)));
  curandInitialize<<<grid, block, 0, stream>>>(state, batch_size, random_seed);
}

__global__ void curandBatchInitialize(curandState_t *states, const int size,
                                      const unsigned long long *random_seeds) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    curand_init(random_seeds[idx], 0, 0, &states[idx]);
  }
}

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_SEQ>
__global__ void topk_stage1(const T *__restrict log_probs, T *tmp_log_probs,
                            int *topk_tmp_id_buf, T *topk_tmp_val_buf,
                            const int *top_ks, const int max_top_k,
                            const int vocab_size) {
  typedef cub::BlockReduce<TopK<T>, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  // if (tid == 0) printf("stage1\n");

  const int batch_id = bid / BLOCKS_PER_SEQ; // row id for log_probs

  const int block_lane = bid % BLOCKS_PER_SEQ; // block id for a beam
  const int k = max_top_k;                     // batch_id = batch index

  const int tmp_log_buf_index = batch_id * vocab_size;
  const int tmp_topk_buf_index =
      batch_id * BLOCKS_PER_SEQ * max_top_k + block_lane * k;

  TopK<T> partial;
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size;
       elem_id += BLOCK_SIZE_ * BLOCKS_PER_SEQ) {
    int index = elem_id + tmp_log_buf_index;
    tmp_log_probs[index] = log_probs[index];
  }

  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int elem_id = tid + block_lane * BLOCK_SIZE_; elem_id < vocab_size;
         elem_id += BLOCK_SIZE_ * BLOCKS_PER_SEQ) {
      int index = elem_id + tmp_log_buf_index;
      partial.insert(tmp_log_probs[index], index);
    }

    TopK<T> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T>);

    if (tid == 0) {
      const int index = tmp_topk_buf_index + ite;
      topk_tmp_id_buf[index] = total.p;
      topk_tmp_val_buf[index] = total.u;
      tmp_log_probs[total.p] = -MAX_T_VAL;
    }
    __syncthreads();
  }
}

template <typename T, int BLOCK_SIZE_, int BLOCKS_PER_SEQ>
__global__ void topk_stage2_sampling(const int *__restrict topk_tmp_id_buf,
                                     T *topk_tmp_val_buf, T *ids,
                                     const int max_top_k, const int *top_ks,
                                     const int vocab_size,
                                     curandState_t *curandstate,
                                     bool temp_top_k, T *temp_top_k_res) {
  const bool IS_FP16 = std::is_same<T, half>::value;
  const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

  const int tid = threadIdx.x;
  const int batch_id = blockIdx.x;

  const int k = top_ks[batch_id];

  const int size = k * BLOCKS_PER_SEQ;
  const int stride = max_top_k * BLOCKS_PER_SEQ;

  typedef cub::BlockReduce<TopK<float>, BLOCK_SIZE_> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  extern __shared__ char array[];
  __shared__ float rand_num;
  __shared__ float s_sum;
  __shared__ float s_max;
  T *s_val = topk_tmp_val_buf + batch_id * stride;
  int *s_id = reinterpret_cast<int *>(array);
  if (tid == 0) {
    s_sum = 0.0f;
  }
  TopK<float> partial;
  int res_index = batch_id * max_top_k;
  float *s_val2 = reinterpret_cast<float *>(s_id + k);
  for (int ite = 0; ite < k; ite++) {
    partial.init();
#pragma unroll
    for (int i = tid; i < size; i += BLOCK_SIZE_) {
      partial.insert((float)s_val[i], i);
    }
    TopK<float> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<float>);

    if (tid == 0) {
      if (ite == 0) {
        s_max = total.u;
      }
      s_id[ite] = total.p;
      s_val[total.p] = -MAX_T_VAL;
      s_val2[ite] = total.u;
      if (temp_top_k) {
        temp_top_k_res[res_index + ite] = total.u;
      }
      s_sum += total.u;
    }
    __syncthreads();
  }

  if (tid == 0) {
    rand_num = (float)curand_uniform(curandstate + blockIdx.x) * s_sum;
    for (int i = 0; i < k; i++) {
      float exp_logit = s_val2[i];
      rand_num = rand_num - exp_logit;
      if (rand_num <= 0.0f || i == k - 1) {
        ids[batch_id] =
            topk_tmp_id_buf[batch_id * stride + s_id[i]] % vocab_size;
        break;
      }
    }
  }
}

#define CASE_K(K_MIN, K_MAX, BLOCK_SIZE_1_, BLOCK_SIZE_2_, BLOCK_PER_SEQ) \
  case K_MIN ... K_MAX:                                                   \
    topk_stage1<T, BLOCK_SIZE_1_, BLOCK_PER_SEQ>                          \
        <<<batch_size * BLOCK_PER_SEQ, BLOCK_SIZE_1_, 0, stream>>>(       \
            log_probs, temp_log_probs, topk_tmp_id_buf, topk_tmp_val_buf, \
            top_ks, max_top_k, vocab_size);                               \
    topk_stage2_sampling<T, BLOCK_SIZE_2_, BLOCK_PER_SEQ>                 \
        <<<batch_size, BLOCK_SIZE_2_,                                     \
           K_MAX * sizeof(int) + K_MAX * sizeof(float), stream>>>(        \
            topk_tmp_id_buf, topk_tmp_val_buf, ids, max_top_k, top_ks,    \
            vocab_size, curandstate, temp_top_k, temp_top_k_res);         \
    break;

template <typename T>
void invokeBatchTopKSampling(void *workspace, size_t &workspace_size,
                             const T *log_probs, T *ids, const int *top_ks,
                             const int max_top_k, const int vocab_size,
                             const int batch_size, curandState_t *curandstate,
                             cudaStream_t stream, bool temp_top_k) {
  const int max_block_per_seq = 8;
  int temp_log_probs_buf_size = batch_size * vocab_size; // type float
  int topk_tmp_ids_buf_size =
      batch_size * max_top_k * max_block_per_seq; // type int
  int topk_tmp_val_buf_size =
      batch_size * max_top_k * max_block_per_seq; // type float
  T *temp_top_k_res = nullptr;
  // prevent memory misaligned address
  temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
  topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
  topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;

  if (workspace == nullptr) {
    workspace_size = sizeof(T) * temp_log_probs_buf_size +
                     sizeof(int) * topk_tmp_ids_buf_size +
                     sizeof(T) * topk_tmp_val_buf_size;
    if (temp_top_k) {
      workspace_size += sizeof(T) * max_top_k * batch_size;
    }
    return;
  }

  T *temp_log_probs = (T *)workspace;
  int *topk_tmp_id_buf = (int *)(temp_log_probs + temp_log_probs_buf_size);
  T *topk_tmp_val_buf = (T *)(topk_tmp_id_buf + topk_tmp_ids_buf_size);

  if (temp_top_k) {
    temp_top_k_res = (T *)(topk_tmp_val_buf + topk_tmp_val_buf_size);
  }
  switch (max_top_k) {
    CASE_K(1, 16, 128, 128, 8);
    CASE_K(17, 32, 256, 128, 8);
    CASE_K(33, 64, 256, 256, 8);
    CASE_K(65, 1024, 256, 256, 8);
    default:
      break;
  }
  cudaDeviceSynchronize();
}

template <typename T>
void top_k_cuda(const T *src, T *dst, const int *top_ks, int batch_size,
                int vocab_size, cudaStream_t stream, bool temp_top_k,
                T *temp_top_k_res) {
  curandState_t *curand_states = nullptr;
  cudaMalloc(&curand_states, sizeof(curandState_t) * batch_size);
  unsigned long long seed = 0;
  invokeCurandInitialize(curand_states, batch_size, seed, stream);

  int max_top_k = -1;
  for (int i = 0; i < batch_size; i++) {
    if (top_ks[i] > max_top_k) {
      max_top_k = top_ks[i];
    }
  }
  int *top_ks_cuda = nullptr;
  cudaMalloc(&top_ks_cuda, sizeof(int) * batch_size);
  cudaMemcpy(top_ks_cuda, top_ks, sizeof(int) * batch_size,
             cudaMemcpyHostToDevice);

  size_t workspace_size = 0;
  void *workspace = nullptr;
  invokeBatchTopKSampling<T>(nullptr, // workspace
                             workspace_size, nullptr,
                             nullptr, // ids
                             nullptr, max_top_k, vocab_size, batch_size,
                             curand_states, stream, true);
  cudaMalloc(&workspace, workspace_size);
  invokeBatchTopKSampling<T>(workspace, workspace_size, src, dst, top_ks_cuda,
                             max_top_k, vocab_size, batch_size, curand_states,
                             stream, temp_top_k);
  if (temp_top_k) {
    int temp_log_probs_buf_size = batch_size * vocab_size;
    int topk_tmp_ids_buf_size = batch_size * max_top_k * 8;
    int topk_tmp_val_buf_size = batch_size * max_top_k * 8;
    temp_log_probs_buf_size = (int)(ceil(temp_log_probs_buf_size / 4.)) * 4;
    topk_tmp_ids_buf_size = (int)(ceil(topk_tmp_ids_buf_size / 4.)) * 4;
    topk_tmp_val_buf_size = (int)(ceil(topk_tmp_val_buf_size / 4.)) * 4;
    T *base_ptr = (T *)workspace;
    T *temp_top_k_res_ptr =
        (T *)(base_ptr + temp_log_probs_buf_size + topk_tmp_ids_buf_size +
              topk_tmp_val_buf_size);

    T *res_nums = new T[batch_size * max_top_k];

    cudaMemcpy(res_nums, temp_top_k_res_ptr,
               sizeof(float) * batch_size * max_top_k, cudaMemcpyDeviceToHost);

    int cnt = 0;
    for (int i = 0; i < batch_size * max_top_k; i++) {
      cnt++;
      printf("%f ", res_nums[i]);
      if (cnt % max_top_k == 0) printf("\n");
    }
    delete[] res_nums;
  }
  cudaFree(top_ks_cuda);
  cudaFree(curand_states);
  cudaFree(workspace);
}
void top_k(const torch::Tensor src, torch::Tensor dst,
           const std::vector<int> &top_ks, const std::vector<float> &top_ps) {
  float *src_ptr = reinterpret_cast<float *>(src.data_ptr());
  float *dst_ptr = reinterpret_cast<float *>(dst.data_ptr());
  auto shape = src.sizes();
  auto shape_len = shape.size();
  assert(shape_len == 2);
  unsigned int batch_size = src.sizes()[0];
  unsigned int len = shape[shape_len - 1];
  float *temp_top_k_res = new float[batch_size * 3];

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  top_k_cuda(src_ptr, dst_ptr, top_ks.data(), batch_size, len, stream, true,
             temp_top_k_res);
  delete[] temp_top_k_res;
}
