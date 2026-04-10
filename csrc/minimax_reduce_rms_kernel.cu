
/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <torch/cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "cuda_utils.h"
#include "core/registration.h"
#include "minimax_reduce_rms_kernel.h"

#include <algorithm>

#define FINAL_MASK 0xffffffff
#define MINIMAX_REDUCE_RMS_WARP_SIZE 32

namespace vllm {
namespace tensorrt_llm {

template <int NRanks>
struct LamportComm {
  __device__ __forceinline__ LamportComm(void** workspace, int rank) {
    counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
    flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
    clear_ptr = &reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[0];
    flag_value = *flag_ptr;
    auto comm_size = reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[1];
    clear_size = *clear_ptr;
    int data_offset = flag_value % 3;
    int clear_offset = (flag_value + 2) % 3;
    for (int r = 0; r < NRanks; ++r) {
      data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) +
                     data_offset * comm_size;
    }
    clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) +
                clear_offset * comm_size;
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(counter_ptr, 1);
    }
  }

  __device__ __forceinline__ void update(int64_t new_clear_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x) {
      }
      *flag_ptr = (flag_value + 1) % 3;
      *clear_ptr = new_clear_size;
      *counter_ptr = 0;
    }
  }

  int* counter_ptr;
  int* flag_ptr;
  int64_t* clear_ptr;
  uint8_t* data_bufs[NRanks];
  uint8_t* clear_buf;
  int64_t clear_size;
  int flag_value;
};

__device__ __forceinline__ bool is_neg_zero(float v) {
  return *reinterpret_cast<uint32_t*>(&v) == 0x80000000;
}

__device__ __forceinline__ bool is_neg_zero(float4 v) {
  return is_neg_zero(v.x) || is_neg_zero(v.y) || is_neg_zero(v.z) ||
         is_neg_zero(v.w);
}

__device__ __forceinline__ float4 get_neg_zero() {
  float4 vec;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    reinterpret_cast<uint32_t*>(&vec)[i] = 0x80000000;
  }
  return vec;
}

template <int Dim>
__device__ __forceinline__ float rms_rsqrt(float& v, float eps) {
  constexpr float kInvDim = 1.0F / static_cast<float>(Dim);
  v = rsqrtf((v * kInvDim) + eps);
  return v;
}

template <int Dim>
__device__ __forceinline__ float4 rms_rsqrt(float4& v, float eps) {
  constexpr float kInvDim = 1.0F / static_cast<float>(Dim);
  v.x = rsqrtf((v.x * kInvDim) + eps);
  v.y = rsqrtf((v.y * kInvDim) + eps);
  v.z = rsqrtf((v.z * kInvDim) + eps);
  v.w = rsqrtf((v.w * kInvDim) + eps);
  return v;
}
__device__ __forceinline__ float4 ld_global_volatile(float4* addr) {
  float4 val;
  asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
               : "l"(addr));
  return val;
}

__device__ __forceinline__ float ld_global_volatile(float* addr) {
  float val;
  asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(val) : "l"(addr));
  return val;
}

// Used by the scalar (non-float4) kernel only
template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

// for float4 version
template <uint32_t kNumThreads, typename T, int ArraySize = 4>
__device__ __forceinline__ void local_warp_reduce_sum_array(
    T* value_ptr, uint32_t active_mask = 0xffffffffu) {
  static_assert(kNumThreads >= 1 &&
                kNumThreads <= MINIMAX_REDUCE_RMS_WARP_SIZE);
#pragma unroll
  for (int i = 0; i < ArraySize; ++i) {
#pragma unroll
    for (int mask = kNumThreads / 2; mask > 0; mask >>= 1) {
      value_ptr[i] += __shfl_xor_sync(active_mask, value_ptr[i], mask,
                                      MINIMAX_REDUCE_RMS_WARP_SIZE);
    }
  }
}

constexpr int next_pow2(int val) {
  int result = 1;
  while (result < val) {
    result <<= 1;
  }
  return result;
}

// ---------------------------------------------------------------------------

template <typename DType>
class IndexHelper {
 public:
  __device__ __forceinline__ IndexHelper(MiniMaxReduceRMSParams const& params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    token_id = grid.cluster_rank();
    access_id_in_token = cluster.thread_rank();
    token_stride = grid.num_clusters();
#else
    token_id = blockIdx.x;
    access_id_in_token = threadIdx.x;
    token_stride = gridDim.x;
#endif
    access_id = token_id * params.hidden_dim / kElemsPerAccess<DType> +
                access_id_in_token;
    access_stride = token_stride * params.hidden_dim / kElemsPerAccess<DType>;
    tot_access = params.size_q / kElemsPerAccess<DType>;
  }

  int token_id;
  int access_id_in_token;
  int token_stride;
  int access_id;
  int access_stride;
  int tot_access;
};

/**
* this kernel is used to for minimax attention module
* input tensor [total_tokens, hidden_dim / tp_size], fp32
* rms weight [hidden_dim / tp_size], bf16
step 1: reduce from single rank to get the variance sum (reduce(input^2,
dim=-1)) step 2: reduce from all ranks to get the variance sum
(all_reduce(variance_sum)) step 3: calculate the rms norm (input *
rsqrt(variance + eps)) in this case, max hidden_dim is 6144 (float data), for
each token, we only need 6144 / 4 / tp_size = (1536 / tp_size) threads so we can
assume cluster size is 1 (tp_size >= 2)
 */
template <typename DType, int NRanks>
__global__ void __launch_bounds__(1024)
    minimax_reduce_rms_kernel_lamport(MiniMaxReduceRMSParams params) {
  IndexHelper<DType> index_helper(params);
  int token_id = index_helper.token_id;
  int access_id_in_token = index_helper.access_id_in_token;
  int token_stride = index_helper.token_stride;
  int access_id = index_helper.access_id;
  int access_stride = index_helper.access_stride;
  int tot_access = index_helper.tot_access;
  int tot_tokens = params.size_q / params.hidden_dim;
  float4 clear_vec = get_neg_zero();

  LamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / kElemsPerAccess<DType>;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  for (int idx = access_id; idx < tot_access;
       idx += access_stride, token_id += token_stride) {
    alignas(16) DType vals[kElemsPerAccess<DType>];
    float sum_variance = 0.F;
    *reinterpret_cast<float4*>(vals) =
        reinterpret_cast<float4*>(params.allreduce_in)[idx];
#pragma unroll
    for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
      sum_variance += static_cast<float>(vals[i]) * static_cast<float>(vals[i]);
    }
    blockReduceSumV2<float, 1>(&sum_variance);
    if (is_neg_zero(sum_variance)) {
      sum_variance = 0.F;
    }
    if (threadIdx.x == 0) {
      for (int r = 0; r < NRanks; ++r) {
        reinterpret_cast<float*>(
            comm.data_bufs[r])[(params.rank * tot_tokens) + token_id] =
            (sum_variance);
      }
    }

    bool done = false;
    float vars_all_ranks[NRanks];
    while (!done) {
      done = true;
#pragma unroll
      for (int r = 0; r < NRanks; ++r) {
        vars_all_ranks[r] = ld_global_volatile(&reinterpret_cast<float*>(
            comm.data_bufs[params.rank])[(r * tot_tokens) + token_id]);
        done &= !is_neg_zero(vars_all_ranks[r]);
      }
    }
    sum_variance = 0.F;
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      sum_variance += vars_all_ranks[r];
    }

    DType norm_weight[kElemsPerAccess<DType>];
    *reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(norm_weight) =
        reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(
            params.rms_gamma)[access_id_in_token];

#pragma unroll
    for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
      vals[i] = static_cast<DType>(
          static_cast<float>(vals[i]) *
          rsqrtf(
              (sum_variance / static_cast<float>(params.hidden_dim) / NRanks) +
              params.rms_eps) *
          static_cast<float>(norm_weight[i]));
    }

    reinterpret_cast<float4*>(params.rms_norm_out)[idx] =
        *reinterpret_cast<float4*>(vals);
  }
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }
  comm.update(params.size_q * NRanks);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

/**
 * Float4 variant: process 4 rows at once, allreduce variance sums as float4 for
 * better memory coalescing. sum_variance is always float; applies to all DTypes
 * (half, bf16, float). When tot_tokens % 4 != 0, the last group pads rows with
 * zeros; padded rows are not written to rms_norm_out. IsQK: when true, process
 * Q+K in one loop with doubled comm buffer; when false, single-matrix (Q only).
 */
template <typename DType, int NRanks, int OriginQDim, int OriginKDim>
__global__ void __launch_bounds__(1024)
    minimax_reduce_qk_rms_kernel_lamport_float4(MiniMaxReduceRMSParams params) {
  // Compile-time per-rank dimensions
  constexpr int RankQDim = OriginQDim / NRanks;
  constexpr int RankKDim = OriginKDim / NRanks;
  // Threads needed to cover one row of Q / K with float4 accesses
  constexpr int ThreadsPerRowQ = RankQDim / kElemsPerAccess<DType>;
  constexpr int ThreadsPerRowK = RankKDim / kElemsPerAccess<DType>;
  // Number of warps dedicated to Q / K
  constexpr int NumWarpQ = (ThreadsPerRowQ + MINIMAX_REDUCE_RMS_WARP_SIZE - 1) /
                           MINIMAX_REDUCE_RMS_WARP_SIZE;
  constexpr int NumWarpK = (ThreadsPerRowK + MINIMAX_REDUCE_RMS_WARP_SIZE - 1) /
                           MINIMAX_REDUCE_RMS_WARP_SIZE;

  int tot_tokens = params.size_q / RankQDim;
  int tot_groups = (tot_tokens + 3) / 4;  // ceiling; last group may be partial

  // Memory strides for strided qkv tensors (elements -> float4-access units)
  int access_stride_q = (params.stride_q > 0 ? params.stride_q : RankQDim) /
                        kElemsPerAccess<DType>;
  int access_stride_k = (params.stride_k > 0 ? params.stride_k : RankKDim) /
                        kElemsPerAccess<DType>;
  // Output strides: default to contiguous (hidden_dim / hidden_dim_k)
  int access_stride_q_out =
      (params.stride_q_out > 0 ? params.stride_q_out : params.hidden_dim) /
      kElemsPerAccess<DType>;
  int access_stride_k_out =
      (params.stride_k_out > 0 ? params.stride_k_out : params.hidden_dim_k) /
      kElemsPerAccess<DType>;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  cg::grid_group grid = cg::this_grid();
  int group_id = grid.cluster_rank();
  int access_id_in_token = cluster.thread_rank();
  int group_stride = grid.num_clusters();
#else
  int group_id = blockIdx.x;
  int access_id_in_token = threadIdx.x;
  int group_stride = gridDim.x;
#endif

  bool is_q = (access_id_in_token < NumWarpQ * MINIMAX_REDUCE_RMS_WARP_SIZE);
  int k_thread_idx =
      access_id_in_token - (NumWarpQ * MINIMAX_REDUCE_RMS_WARP_SIZE);
  bool is_valid_q = (access_id_in_token < ThreadsPerRowQ);
  bool is_valid_k = (k_thread_idx >= 0 && k_thread_idx < ThreadsPerRowK);
  float4 clear_vec = get_neg_zero();

  // Shared memory for two-level block reduction and scale broadcast
  __shared__ float block_reduce_sum[4][MINIMAX_REDUCE_RMS_WARP_SIZE + 1];
  __shared__ float global_scale_q[4];
  __shared__ float global_scale_k[4];

  LamportComm<NRanks> comm(params.workspace, params.rank);

  DType norm_weight[kElemsPerAccess<DType>]{};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  if (is_q) {
    if (is_valid_q) {
      *reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(
          norm_weight) =
          reinterpret_cast<typename ElemsPerAccess<DType>::vec_type const*>(
              params.rms_gamma)[access_id_in_token];
    }
  } else {
    if (is_valid_k) {
      *reinterpret_cast<typename ElemsPerAccess<DType>::vec_type*>(
          norm_weight) =
          reinterpret_cast<typename ElemsPerAccess<DType>::vec_type const*>(
              params.rms_gamma_k)[k_thread_idx];
    }
  }

  // Main loop: process one group of 4 tokens per iteration.
  for (int g = group_id; g < tot_groups; g += group_stride) {
    alignas(16) DType vals[4][kElemsPerAccess<DType>]{};
    float warp_sum_variance[4]{0.F, 0.F, 0.F, 0.F};

    if (is_q) {
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        int token_r = g * 4 + row;
        if (token_r >= tot_tokens || !is_valid_q) {
          continue;
        }
        int idx_r = token_r * access_stride_q + access_id_in_token;
        *reinterpret_cast<float4*>(&vals[row][0]) =
            reinterpret_cast<float4 const*>(params.allreduce_in)[idx_r];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          float x = static_cast<float>(vals[row][i]);
          warp_sum_variance[row] += x * x;
        }
      }
    } else {
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        int token_r = g * 4 + row;
        if (token_r >= tot_tokens || !is_valid_k) {
          continue;
        }
        int idx_r = token_r * access_stride_k + k_thread_idx;
        *reinterpret_cast<float4*>(&vals[row][0]) =
            reinterpret_cast<float4 const*>(params.allreduce_in_k)[idx_r];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          float x = static_cast<float>(vals[row][i]);
          warp_sum_variance[row] += x * x;
        }
      }
    }

    local_warp_reduce_sum_array<MINIMAX_REDUCE_RMS_WARP_SIZE, float, 4>(
        warp_sum_variance);
    // Warp lane 0 writes its warp's partial sum to shared memory
    int lane = threadIdx.x & (MINIMAX_REDUCE_RMS_WARP_SIZE - 1);
    if (lane == 0) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        block_reduce_sum[t][threadIdx.x / MINIMAX_REDUCE_RMS_WARP_SIZE] =
            warp_sum_variance[t];
      }
    }
    __syncthreads();

    int tid = threadIdx.x;

    if (tid < MINIMAX_REDUCE_RMS_WARP_SIZE) {
      constexpr int kNumWarpQPow2 =
          (next_pow2(NumWarpQ) > NRanks) ? next_pow2(NumWarpQ) : NRanks;
      float local_sum[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        local_sum[t] = (tid < NumWarpQ) ? block_reduce_sum[t][tid] : 0.F;
      }
      // After this, all kNumWarpQPow2 lanes (including tid 0..NRanks-1) have
      // the total Q sum-of-squares for all 4 tokens.
      local_warp_reduce_sum_array<kNumWarpQPow2, float, 4>(local_sum);

      if (tid < NRanks) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          if (is_neg_zero(local_sum[t])) {
            local_sum[t] = 0.F;
          }
        }
        // Parallel push: thread tid writes this rank's Q sum to rank tid's buf
        reinterpret_cast<float4*>(
            comm.data_bufs[tid])[(params.rank * tot_groups * 2) + (2 * g)] =
            *reinterpret_cast<float4*>(local_sum);

        // Parallel pull: thread tid reads rank tid's contribution from
        // this rank's (params.rank's) buffer
        bool done = false;
        float4 var_all_ranks;
        while (!done) {
          done = true;
          var_all_ranks = ld_global_volatile(&reinterpret_cast<float4*>(
              comm.data_bufs[params.rank])[(tid * tot_groups * 2) + (2 * g)]);
          done &= !is_neg_zero(var_all_ranks);
        }

        // Warp-level allreduce: each of the NRanks threads holds one rank's
        // partial sum; after this all NRanks threads have the global total.
        constexpr uint32_t kQActiveMask = (1u << NRanks) - 1u;
        local_warp_reduce_sum_array<NRanks, float, 4>(
            reinterpret_cast<float*>(&var_all_ranks), kQActiveMask);

        // Thread 0 computes rsqrt with compile-time Dim and writes to smem
        if (tid == 0) {
          *reinterpret_cast<float4*>(global_scale_q) =
              rms_rsqrt<OriginQDim>(var_all_ranks, params.rms_eps);
        }
      }
    } else if (tid >= MINIMAX_REDUCE_RMS_WARP_SIZE * NumWarpQ &&
               tid < MINIMAX_REDUCE_RMS_WARP_SIZE * (NumWarpQ + 1)) {
      // --- K leader warp ---
      constexpr int kNumWarpKPow2 =
          (next_pow2(NumWarpK) > NRanks) ? next_pow2(NumWarpK) : NRanks;
      float local_sum[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        local_sum[t] = (k_thread_idx < NumWarpK)
                           ? block_reduce_sum[t][NumWarpQ + k_thread_idx]
                           : 0.F;
      }
      local_warp_reduce_sum_array<kNumWarpKPow2, float, 4>(local_sum);

      if (k_thread_idx < NRanks) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          if (is_neg_zero(local_sum[t])) {
            local_sum[t] = 0.F;
          }
        }
        reinterpret_cast<float4*>(
            comm.data_bufs[k_thread_idx])[(params.rank * tot_groups * 2) +
                                          (2 * g + 1)] =
            *reinterpret_cast<float4*>(local_sum);

        bool done = false;
        float4 var_all_ranks;
        while (!done) {
          done = true;
          var_all_ranks = ld_global_volatile(&reinterpret_cast<float4*>(
              comm.data_bufs[params.rank])[(k_thread_idx * tot_groups * 2) +
                                           (2 * g + 1)]);
          done &= !is_neg_zero(var_all_ranks);
        }

        constexpr uint32_t kKActiveMask = (1u << NRanks) - 1u;
        local_warp_reduce_sum_array<NRanks, float, 4>(
            reinterpret_cast<float*>(&var_all_ranks), kKActiveMask);

        if (k_thread_idx == 0) {
          *reinterpret_cast<float4*>(global_scale_k) =
              rms_rsqrt<OriginKDim>(var_all_ranks, params.rms_eps);
        }
      }
    }
    __syncthreads();

    if (is_q) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        warp_sum_variance[t] = global_scale_q[t];
      }
#pragma unroll
      for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          vals[r][i] = static_cast<DType>(static_cast<float>(vals[r][i]) *
                                          warp_sum_variance[r] *
                                          static_cast<float>(norm_weight[i]));
        }
        int token_r = g * 4 + r;
        if (token_r >= tot_tokens || !is_valid_q) {
          continue;
        }
        int idx_out = token_r * access_stride_q_out + access_id_in_token;
        reinterpret_cast<float4*>(params.rms_norm_out)[idx_out] =
            *reinterpret_cast<float4*>(&vals[r][0]);
      }
    } else {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        warp_sum_variance[t] = global_scale_k[t];
      }
#pragma unroll
      for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          vals[r][i] = static_cast<DType>(static_cast<float>(vals[r][i]) *
                                          warp_sum_variance[r] *
                                          static_cast<float>(norm_weight[i]));
        }
        int token_r = g * 4 + r;
        if (token_r >= tot_tokens || !is_valid_k) {
          continue;
        }
        int idx_out = token_r * access_stride_k_out + k_thread_idx;
        reinterpret_cast<float4*>(params.rms_norm_out_k)[idx_out] =
            *reinterpret_cast<float4*>(&vals[r][0]);
      }
    }
  }  // end group loop
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  int clear_access = static_cast<int>(comm.clear_size / kElemsPerAccess<DType>);
  int clear_stride = group_stride * blockDim.x;
  for (int idx = group_id * blockDim.x + threadIdx.x; idx < clear_access;
       idx += clear_stride) {
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }

  comm.update(static_cast<int64_t>(2) * tot_groups * kElemsPerAccess<DType> *
              NRanks);
}

int get_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    sm_count = device_prop.multiProcessorCount;
  }
  return sm_count;
}

inline int getSMVersion(bool queryRealSmArch = false) {
  int device{-1};
  CUDA_CHECK(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_major,
                                    cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_minor,
                                    cudaDevAttrComputeCapabilityMinor, device));
  int sm = sm_major * 10 + sm_minor;
  if (sm == 121 && !queryRealSmArch) {
    return 120;
  }
  return sm;
}

template <typename KernelFunc>
int get_max_active_blocks(KernelFunc kernel, int block_size,
                          int dynamic_smem = 0) {
  int max_active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active, kernel, block_size, dynamic_smem));
  return std::max(max_active, 1);
}

template <typename DType, int NRanks>
void minimax_reduce_rms_kernel_launcher(MiniMaxReduceRMSParams const& params) {
  static int SM = getSMVersion();
  int token_num = params.size_q / params.hidden_dim;
  int sm_count = get_sm_count();
  int cluster_size = 1;
  int cluster_num = token_num;
  int threads_per_token = params.hidden_dim / kElemsPerAccess<DType>;
  int block_size = threads_per_token;

  int max_blocks_per_sm = get_max_active_blocks(
      minimax_reduce_rms_kernel_lamport<DType, NRanks>, block_size);
  int max_grid = max_blocks_per_sm * sm_count;

  int grid_size =
      (std::min(max_grid, cluster_num * cluster_size) / cluster_size) *
      cluster_size;

  cudaLaunchConfig_t cfg;
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;

  cudaLaunchAttribute attribute[2];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;

  CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, minimax_reduce_rms_kernel_lamport<DType, NRanks>, params));
}

template <typename DType, int NRanks, int OriginQDim, int OriginKDim>
void minimax_reduce_rms_kernel_launcher_float4(
    MiniMaxReduceRMSParams const& params) {
  TORCH_CHECK(params.size_q % params.hidden_dim == 0);
  TORCH_CHECK(params.hidden_dim % kElemsPerAccess<DType> == 0);
  if (params.stride_q > 0) {
    TORCH_CHECK(params.stride_q % kElemsPerAccess<DType> == 0);
  }
  TORCH_CHECK(params.allreduce_in_k != nullptr,
              "float4 QK kernel requires K input");
  TORCH_CHECK(params.hidden_dim >= params.hidden_dim_k);
  TORCH_CHECK(params.size_k % params.hidden_dim_k == 0);
  TORCH_CHECK(params.hidden_dim_k % kElemsPerAccess<DType> == 0);
  TORCH_CHECK(params.size_q / params.hidden_dim ==
              params.size_k / params.hidden_dim_k);
  if (params.stride_k > 0) {
    TORCH_CHECK(params.stride_k % kElemsPerAccess<DType> == 0);
  }

  int token_num = params.size_q / params.hidden_dim;
  int tot_groups = (token_num + 3) / 4;
  if (tot_groups == 0) {
    return;
  }

  static int SM = getSMVersion();
  int sm_count = get_sm_count();
  int cluster_size = 1;
  int cluster_num = tot_groups;

  int access_per_row_q = params.hidden_dim / kElemsPerAccess<DType>;
  int access_per_row_k = params.hidden_dim_k / kElemsPerAccess<DType>;

  // Round each section up to a warp boundary
  auto divUp = [](int a, int b) { return (a + b - 1) / b * b; };
  int block_size = divUp(access_per_row_q, MINIMAX_REDUCE_RMS_WARP_SIZE) +
                   divUp(access_per_row_k, MINIMAX_REDUCE_RMS_WARP_SIZE);

  auto kfn =
      minimax_reduce_qk_rms_kernel_lamport_float4<DType, NRanks, OriginQDim,
                                                  OriginKDim>;

  int max_blocks_per_sm = get_max_active_blocks(kfn, block_size);
  int max_grid = max_blocks_per_sm * sm_count;
  int grid_size =
      (std::min(max_grid, cluster_num * cluster_size) / cluster_size) *
      cluster_size;

  cudaLaunchConfig_t cfg;
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;

  cudaLaunchAttribute attribute[2];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;

  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kfn, params));
}

template <int NRanks>
void dispatch_dtype(MiniMaxReduceRMSParams const& params) {
  // Use the optimized QK float4 kernel when:
  //  - K input is present, AND
  //  - the full (NRanks * per-rank) dimensions match the MiniMax M2 shape.
  // Otherwise fall back to the scalar kernel.
  bool use_float4 = (params.allreduce_in_k != nullptr) &&
                    (params.hidden_dim * params.nranks == 6144) &&
                    (params.hidden_dim_k * params.nranks == 1024);

  if (params.dtype == at::ScalarType::Half) {
    if (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<half, NRanks, 6144, 1024>(
          params);
    } else {
      minimax_reduce_rms_kernel_launcher<half, NRanks>(params);
    }
  } else if (params.dtype == at::ScalarType::BFloat16) {
    if (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<__nv_bfloat16, NRanks, 6144,
                                                1024>(params);
    } else {
      minimax_reduce_rms_kernel_launcher<__nv_bfloat16, NRanks>(params);
    }
  } else if (params.dtype == at::ScalarType::Float) {
    if (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<float, NRanks, 6144, 1024>(
          params);
    } else {
      minimax_reduce_rms_kernel_launcher<float, NRanks>(params);
    }
  } else {
    TORCH_CHECK(false, "Unsupported data type for minimax_reduce_rms_op");
  }
}

void minimax_reduce_rms_op(MiniMaxReduceRMSParams const& params) {
  if (params.nranks == 2) {
    dispatch_dtype<2>(params);
  } else if (params.nranks == 4) {
    dispatch_dtype<4>(params);
  } else if (params.nranks == 8) {
    dispatch_dtype<8>(params);
  } else if (params.nranks == 16) {
    dispatch_dtype<16>(params);
  } else {
    TORCH_CHECK(false, "minimax_reduce_rms_op: unsupported ranks number!");
  }
}
}  // namespace tensorrt_llm
}  // namespace vllm

torch::Tensor minimax_allreduce_rms(torch::Tensor const& input,
                                    torch::Tensor const& norm_weight,
                                    torch::Tensor workspace, int64_t const rank,
                                    int64_t const nranks, double const eps) {
  auto allreduce_params = vllm::tensorrt_llm::MiniMaxReduceRMSParams();

  allreduce_params.nranks = static_cast<int>(nranks);
  allreduce_params.rank = static_cast<int>(rank);
  allreduce_params.dtype = input.scalar_type();
  allreduce_params.size_q = static_cast<int>(input.numel());
  allreduce_params.hidden_dim = static_cast<int>(input.size(-1));
  allreduce_params.stride_q = allreduce_params.hidden_dim;
  allreduce_params.workspace =
      reinterpret_cast<void**>(workspace.mutable_data_ptr());
  allreduce_params.allreduce_in = input.data_ptr();
  allreduce_params.rms_gamma = norm_weight.data_ptr();
  allreduce_params.rms_eps = static_cast<float>(eps);
  allreduce_params.stream = at::cuda::getCurrentCUDAStream(input.get_device());

  torch::Tensor rms_norm_out = torch::empty_like(input);
  allreduce_params.rms_norm_out = rms_norm_out.mutable_data_ptr();

  vllm::tensorrt_llm::minimax_reduce_rms_op(allreduce_params);

  return rms_norm_out;
}

std::tuple<torch::Tensor, torch::Tensor> minimax_allreduce_rms_qk(
    torch::Tensor qkv, torch::Tensor const& norm_weight_q,
    torch::Tensor const& norm_weight_k, torch::Tensor workspace,
    int64_t const q_size, int64_t const kv_size, int64_t const rank,
    int64_t const nranks, double const eps) {
  TORCH_CHECK(qkv.dim() == 2, "minimax_allreduce_rms_qk: qkv must be 2D");
  TORCH_CHECK(qkv.is_contiguous(),
              "minimax_allreduce_rms_qk: qkv must be contiguous");
  int64_t qkv_dim = qkv.size(-1);
  TORCH_CHECK(qkv_dim == q_size + 2 * kv_size,
              "minimax_allreduce_rms_qk: qkv last dim must equal "
              "q_size + 2 * kv_size");
  TORCH_CHECK(rank < nranks,
              "minimax_allreduce_rms_qk: rank must be less than nranks");

  int64_t num_tokens = qkv.size(0);
  int elem_bytes = qkv.element_size();

  torch::Tensor q_out = torch::empty({num_tokens, q_size}, qkv.options());
  torch::Tensor k_out = torch::empty({num_tokens, kv_size}, qkv.options());

  auto params = vllm::tensorrt_llm::MiniMaxReduceRMSParams();
  params.nranks = static_cast<int>(nranks);
  params.rank = static_cast<int>(rank);
  params.dtype = qkv.scalar_type();
  params.size_q = static_cast<int>(num_tokens * q_size);
  params.hidden_dim = static_cast<int>(q_size);
  params.size_k = static_cast<int>(num_tokens * kv_size);
  params.hidden_dim_k = static_cast<int>(kv_size);
  params.stride_q = static_cast<int>(qkv_dim);
  params.stride_k = static_cast<int>(qkv_dim);
  params.stride_q_out = 0;  // q_out is contiguous; kernel uses hidden_dim
  params.stride_k_out = 0;  // k_out is contiguous; kernel uses hidden_dim_k
  params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());

  uint8_t* base = static_cast<uint8_t*>(qkv.data_ptr());
  params.allreduce_in = base;
  params.allreduce_in_k = base + q_size * elem_bytes;
  params.rms_gamma = norm_weight_q.data_ptr();
  params.rms_gamma_k = norm_weight_k.data_ptr();
  params.rms_eps = static_cast<float>(eps);
  params.stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  params.rms_norm_out = q_out.mutable_data_ptr();
  params.rms_norm_out_k = k_out.mutable_data_ptr();

  vllm::tensorrt_llm::minimax_reduce_rms_op(params);
  return {q_out, k_out};
}
