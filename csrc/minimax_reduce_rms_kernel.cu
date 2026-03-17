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

template <typename T, int NUM>
__device__ __forceinline__ void blockReduceSumRange(T* val, int rangeStart,
                                                    int rangeEnd) {
  constexpr int kWarpSize = 32;
  constexpr unsigned kFullMask = 0xffffffffu;
  static __shared__ T shared[NUM][33];

  int const activeThreadCount = max(rangeEnd - rangeStart, 0);
  bool const isActive = threadIdx.x >= rangeStart && threadIdx.x < rangeEnd;
  int const lane = threadIdx.x & (kWarpSize - 1);
  unsigned const activeMask = __ballot_sync(kFullMask, isActive);

  if (isActive) {
#pragma unroll
    for (int i = 0; i < NUM; ++i) {
      T sum = val[i];
#pragma unroll
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(activeMask, sum, offset, kWarpSize);
      }
      val[i] = sum;
    }
  }

  if (isActive && lane == 0) {
    int const localWarpId = (threadIdx.x - rangeStart) >> 5;
#pragma unroll
    for (int i = 0; i < NUM; ++i) {
      shared[i][localWarpId] = val[i];
    }
  }

  __syncthreads();

  int const shiftedTid = threadIdx.x - rangeStart;
  int const warpCount = (activeThreadCount + kWarpSize - 1) / kWarpSize;
  bool const inLeaderWarp = shiftedTid >= 0 && shiftedTid < kWarpSize;
  bool const leaderLaneIsValid = inLeaderWarp && shiftedTid < warpCount;
  unsigned const leaderMask = __ballot_sync(kFullMask, leaderLaneIsValid);

  if (inLeaderWarp) {
#pragma unroll
    for (int i = 0; i < NUM; ++i) {
      T sum = leaderLaneIsValid ? shared[i][shiftedTid] : static_cast<T>(0);
#pragma unroll
      for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(leaderMask, sum, offset, kWarpSize);
      }
      if (threadIdx.x == rangeStart) {
        val[i] = sum;
      }
    }
  }
}

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
  // FusedOp<Pattern, DType> fused_op(params, access_id, access_id_in_token);

  LamportComm<NRanks> comm(params.workspace, params.rank);
  int clear_access = comm.clear_size / kElemsPerAccess<DType>;
  for (int idx = access_id; idx < tot_access;
       idx += access_stride, token_id += token_stride) {
    alignas(16) DType vals[kElemsPerAccess<DType>];
    // we use float to load and store variance sum
    float sum_variance = 0.F;
    *reinterpret_cast<float4*>(vals) =
        reinterpret_cast<float4*>(params.allreduce_in)[idx];
#pragma unroll
    for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
      sum_variance += static_cast<float>(vals[i]) * static_cast<float>(vals[i]);
    }
    // step 1: reduce from single rank to get the variance sum
    blockReduceSumV2<float, 1>(&sum_variance);
    if (is_neg_zero(sum_variance)) {
      sum_variance = 0.F;
    }
    // step 2: reduce from all ranks to get the variance sum
    // be careful, we only use float to load and store variance sum
    // but we use float4 to load input tensor
    // Push data to other ranks
    // we only need the first thread to push data to other ranks
    if (threadIdx.x == 0) {
      for (int r = 0; r < NRanks; ++r) {
        // temp data buffer [nranks, total_tokens, 1]
        reinterpret_cast<float*>(
            comm.data_bufs[r])[(params.rank * tot_tokens) + token_id] =
            (sum_variance);
      }
    }

    // Load data from other ranks
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

    // step 3: calculate the rms norm (input * rsqrt(variance + eps))

    // load norm weight
    // TODO: correct the access_id_in_token
    __nv_bfloat16 norm_weight[kElemsPerAccess<DType>];
    *reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(
        norm_weight) =
        reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(
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

    // step 4: store the rms norm
    reinterpret_cast<float4*>(params.rms_norm_out)[idx] =
        *reinterpret_cast<float4*>(vals);
  }
  for (int idx = access_id; idx < clear_access; idx += access_stride) {
    // Clear comm buffer that previous kernel used
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }
  comm.update(params.size_q * NRanks);
}

/**
 * Float4 variant: process 4 rows at once, allreduce variance sums as float4 for
 * better memory coalescing. sum_variance is always float; applies to all DTypes
 * (half, bf16, float). When tot_tokens % 4 != 0, the last group pads rows with
 * zeros; padded rows are not written to rms_norm_out. IsQK: when true, process
 * Q+K in one loop with doubled comm buffer; when false, single-matrix (Q only).
 */
template <typename DType, int NRanks, bool IsQK>
__global__ void __launch_bounds__(1024)
    minimax_reduce_rms_kernel_lamport_float4(MiniMaxReduceRMSParams params) {
  int tot_tokens = params.size_q / params.hidden_dim;
  int tot_groups =
      (tot_tokens + 3) / 4;  // ceiling: last group may have 1-3 valid rows
  int access_per_row_q = params.hidden_dim / kElemsPerAccess<DType>;
  int access_per_row_k =
      IsQK ? (params.hidden_dim_k / kElemsPerAccess<DType>) : 0;
  int q_warps = (access_per_row_q + MINIMAX_REDUCE_RMS_WARP_SIZE - 1) /
                MINIMAX_REDUCE_RMS_WARP_SIZE;
  int k_warps = IsQK ? ((access_per_row_k + MINIMAX_REDUCE_RMS_WARP_SIZE - 1) /
                        MINIMAX_REDUCE_RMS_WARP_SIZE)
                     : 0;
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
  bool is_q = (access_id_in_token < q_warps * MINIMAX_REDUCE_RMS_WARP_SIZE);
  int k_thread_idx =
      IsQK ? (access_id_in_token - q_warps * MINIMAX_REDUCE_RMS_WARP_SIZE) : 0;
  bool is_valid_token = is_q ? (access_id_in_token < access_per_row_q)
                             : (k_thread_idx < access_per_row_k);
  float4 clear_vec = get_neg_zero();

  LamportComm<NRanks> comm(params.workspace, params.rank);

  for (int g = group_id; g < tot_groups; g += group_stride) {
    alignas(16) DType vals[4][kElemsPerAccess<DType>]{};
    float sum_variance[4] = {0.F, 0.F, 0.F, 0.F};
    float sum_variance_k[4] = {0.F, 0.F, 0.F, 0.F};

    if (is_q) {
// Q branch: each thread only covers 128bit
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        int token_r = g * 4 + r;
        if (token_r >= tot_tokens || (!is_valid_token)) {
          continue;
        }
        int idx_r = token_r * access_per_row_q + access_id_in_token;
        *reinterpret_cast<float4*>(&vals[r][0]) =
            reinterpret_cast<float4 const*>(params.allreduce_in)[idx_r];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          float x = static_cast<float>(vals[r][i]);
          sum_variance[r] += x * x;
        }
      }
    } else if constexpr (IsQK)  // k branch
    {
// K branch: k_thread_idx = threadIdx.x - q_warps, each thread covers 32 K
// columns
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        int token_r = g * 4 + r;
        if (token_r >= tot_tokens || k_thread_idx >= access_per_row_k) {
          continue;
        }

        int idx_r = token_r * access_per_row_k + k_thread_idx;
        *reinterpret_cast<float4*>(&vals[r][0]) =
            reinterpret_cast<float4 const*>(params.allreduce_in_k)[idx_r];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
          float x = static_cast<float>(vals[r][i]);
          sum_variance_k[r] += x * x;
        }
      }
    }

    // Local reduce: only Q segment contributes to sum_variance, only K segment
    // to sum_variance_k here we use all threads to reduce sum_variance and
    // sum_variance_k
    // TODO: we can do local reduce only within q threads and k threads
    // respectively
    blockReduceSumV2<float, 4>(sum_variance);
    if constexpr (IsQK) {
      int const kStartThread = q_warps * MINIMAX_REDUCE_RMS_WARP_SIZE;
      int const kEndThread = (q_warps + k_warps) * MINIMAX_REDUCE_RMS_WARP_SIZE;
      blockReduceSumRange<float, 4>(sum_variance_k, kStartThread, kEndThread);
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      if (is_neg_zero(sum_variance[r])) {
        sum_variance[r] = 0.F;
      }
      if constexpr (IsQK) {
        if (is_neg_zero(sum_variance_k[r])) {
          sum_variance_k[r] = 0.F;
        }
      }
    }

    // Allreduce: write float4(s) to comm (thread 0 has both after broadcast)
    if (threadIdx.x == 0 ||
        threadIdx.x == q_warps * MINIMAX_REDUCE_RMS_WARP_SIZE) {
      if (is_q) {
        float4 sum4;
        sum4.x = sum_variance[0];
        sum4.y = sum_variance[1];
        sum4.z = sum_variance[2];
        sum4.w = sum_variance[3];
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
          if constexpr (IsQK) {
            reinterpret_cast<float4*>(
                comm.data_bufs[r])[(params.rank * 2 * tot_groups) + 2 * g] =
                sum4;
          } else {
            reinterpret_cast<float4*>(
                comm.data_bufs[r])[(params.rank * tot_groups) + g] = sum4;
          }
        }
      } else if constexpr (IsQK) {
        float4 sum4;
        sum4.x = sum_variance_k[0];
        sum4.y = sum_variance_k[1];
        sum4.z = sum_variance_k[2];
        sum4.w = sum_variance_k[3];
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
          reinterpret_cast<float4*>(
              comm.data_bufs[r])[(params.rank * 2 * tot_groups) + 2 * g + 1] =
              sum4;
        }
      }
    }

    // Read Q from buffer, sum, then RMS and store Q
    bool done = false;
    float4 vars_all_ranks[NRanks];
    if (is_q) {
      while (!done) {
        done = true;
#pragma unroll
        for (int r = 0; r < NRanks; ++r) {
          if constexpr (IsQK) {
            vars_all_ranks[r] = ld_global_volatile(&reinterpret_cast<float4*>(
                comm.data_bufs[params.rank])[(r * 2 * tot_groups) + 2 * g]);
          } else {
            vars_all_ranks[r] = ld_global_volatile(&reinterpret_cast<float4*>(
                comm.data_bufs[params.rank])[(r * tot_groups) + g]);
          }
          done &= !is_neg_zero(vars_all_ranks[r]);
        }
      }
    } else if constexpr (IsQK) {
      while (!done) {
        done = true;
        for (int r = 0; r < NRanks; ++r) {
          vars_all_ranks[r] = ld_global_volatile(&reinterpret_cast<float4*>(
              comm.data_bufs[params.rank])[(r * 2 * tot_groups) + 2 * g + 1]);
          done &= !is_neg_zero(vars_all_ranks[r]);
        }
      }
    }

    sum_variance[0] = 0.F;
    sum_variance[1] = 0.F;
    sum_variance[2] = 0.F;
    sum_variance[3] = 0.F;
#pragma unroll
    for (int r = 0; r < NRanks; ++r) {
      sum_variance[0] += vars_all_ranks[r].x;
      sum_variance[1] += vars_all_ranks[r].y;
      sum_variance[2] += vars_all_ranks[r].z;
      sum_variance[3] += vars_all_ranks[r].w;
    }

    // RMS norm and store 4 rows of Q (Q branch only, reload and store per
    // column)
    if (is_q) {
      if (access_id_in_token < access_per_row_q) {
        __nv_bfloat16 norm_weight[kElemsPerAccess<DType>];
        *reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(
            norm_weight) =
            reinterpret_cast<
                typename ElemsPerAccess<DType>::norm_weight_type const*>(
                params.rms_gamma)[access_id_in_token];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          int token_r = g * 4 + r;
          if (token_r >= tot_tokens) {
            continue;
          }
          float scale =
              rsqrtf((sum_variance[r] / static_cast<float>(params.hidden_dim) /
                      NRanks) +
                     params.rms_eps);

#pragma unroll
          for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
            vals[r][i] =
                static_cast<DType>(static_cast<float>(vals[r][i]) * scale *
                                   static_cast<float>(norm_weight[i]));
          }
          int idx_out = token_r * access_per_row_q + access_id_in_token;
          reinterpret_cast<float4*>(params.rms_norm_out)[idx_out] =
              *reinterpret_cast<float4*>(&vals[r][0]);
        }
      }
    } else if constexpr (IsQK) {
      if (k_thread_idx < access_per_row_k) {
        __nv_bfloat16 norm_weight_k[kElemsPerAccess<DType>];
        *reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(
            norm_weight_k) =
            reinterpret_cast<
                typename ElemsPerAccess<DType>::norm_weight_type const*>(
                params.rms_gamma_k)[k_thread_idx];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          int token_r = g * 4 + r;
          if (token_r >= tot_tokens) {
            continue;
          }
          float scale_k =
              rsqrtf((sum_variance[r] /
                      static_cast<float>(params.hidden_dim_k) / NRanks) +
                     params.rms_eps);
#pragma unroll
          for (int i = 0; i < kElemsPerAccess<DType>; ++i) {
            vals[r][i] =
                static_cast<DType>(static_cast<float>(vals[r][i]) * scale_k *
                                   static_cast<float>(norm_weight_k[i]));
          }
          int idx_out = token_r * access_per_row_k + k_thread_idx;
          reinterpret_cast<float4*>(params.rms_norm_out_k)[idx_out] =
              *reinterpret_cast<float4*>(&vals[r][0]);
        }
      }
    }
  }

  // Clear comm buffer
  int clear_access = static_cast<int>(comm.clear_size / kElemsPerAccess<DType>);
  int clear_stride = group_stride * blockDim.x;

  for (int idx = group_id * blockDim.x + threadIdx.x; idx < clear_access;
       idx += clear_stride) {
    reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
  }

  comm.update(IsQK ? (2 * tot_groups * 8 * NRanks) : (tot_groups * 8 * NRanks));
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

template <typename DType, int NRanks>
void minimax_reduce_rms_kernel_launcher(MiniMaxReduceRMSParams const& params) {
  static int SM = getSMVersion();
  int token_num = params.size_q / params.hidden_dim;
  // for current problem size, we only need one cluster
  int sm_count = get_sm_count();
  int cluster_size = 1;
  int cluster_num = token_num;
  int threads_per_token = params.hidden_dim / kElemsPerAccess<DType>;
  int block_size = threads_per_token;
  int grid_size =
      (std::min(sm_count, cluster_num * cluster_size) / cluster_size) *
      cluster_size;

  cudaLaunchConfig_t cfg;
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;

  cudaLaunchAttribute attribute[2];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;

  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;

  CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, minimax_reduce_rms_kernel_lamport<DType, NRanks>, params));
}

template <typename DType, int NRanks>
void minimax_reduce_rms_kernel_launcher_float4(
    MiniMaxReduceRMSParams const& params) {
  TORCH_CHECK(params.size_q % params.hidden_dim == 0);
  TORCH_CHECK(params.hidden_dim % kElemsPerAccess<DType> == 0);
  if (params.allreduce_in_k != nullptr) {
    TORCH_CHECK(params.hidden_dim >= params.hidden_dim_k);
    TORCH_CHECK(params.size_k % params.hidden_dim_k == 0);
    TORCH_CHECK(params.hidden_dim_k % kElemsPerAccess<DType> == 0);
    TORCH_CHECK(params.size_q / params.hidden_dim ==
                params.size_k / params.hidden_dim_k);
  }
  int token_num = params.size_q / params.hidden_dim;
  int tot_groups = (token_num + 3) / 4;  // ceiling
  if (tot_groups == 0) {
    return;
  }
  static int SM = getSMVersion();
  int sm_count = get_sm_count();
  int cluster_size = 1;
  int cluster_num = tot_groups;
  int access_per_row_q = params.hidden_dim / kElemsPerAccess<DType>;
  int access_per_row_k = (params.allreduce_in_k != nullptr)
                             ? (params.hidden_dim_k / kElemsPerAccess<DType>)
                             : 0;
  auto divUp = [](int a, int b) {
    return (a + b - 1) / b * b;
  };  // round up to the nearest multiple of b
  int block_size = divUp(access_per_row_q, MINIMAX_REDUCE_RMS_WARP_SIZE) +
                   ((params.allreduce_in_k != nullptr)
                        ? divUp(access_per_row_k, MINIMAX_REDUCE_RMS_WARP_SIZE)
                        : 0);
  int grid_size =
      (std::min(sm_count, cluster_num * cluster_size) / cluster_size) *
      cluster_size;

  cudaLaunchConfig_t cfg;
  cfg.gridDim = grid_size;
  cfg.blockDim = block_size;
  cfg.dynamicSmemBytes = 0;
  cfg.stream = params.stream;

  cudaLaunchAttribute attribute[2];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[1].id = cudaLaunchAttributeClusterDimension;
  attribute[1].val.clusterDim.x = cluster_size;
  attribute[1].val.clusterDim.y = 1;
  attribute[1].val.clusterDim.z = 1;
  cfg.attrs = attribute;
  cfg.numAttrs = SM >= 90 ? 2 : 0;
  bool is_qk = (params.allreduce_in_k != nullptr);

  if (is_qk) {
    CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, minimax_reduce_rms_kernel_lamport_float4<DType, NRanks, true>,
        params));
  } else {
    CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, minimax_reduce_rms_kernel_lamport_float4<DType, NRanks, false>,
        params));
  }
}

template <int NRanks>
void dispatch_dtype(MiniMaxReduceRMSParams const& params) {
  bool use_float4 = true;

  if (params.dtype == at::ScalarType::Half) {
    if (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<half, NRanks>(params);
    } else {
      minimax_reduce_rms_kernel_launcher<half, NRanks>(params);
    }
  } else if (params.dtype == at::ScalarType::BFloat16) {
    if (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<__nv_bfloat16, NRanks>(params);
    } else {
      minimax_reduce_rms_kernel_launcher<__nv_bfloat16, NRanks>(params);
    }
  } else if (params.dtype == at::ScalarType::Float) {
    if (use_float4) {
      minimax_reduce_rms_kernel_launcher_float4<float, NRanks>(params);
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

std::vector<torch::Tensor> minimax_allreduce_rms_qk(
    torch::Tensor const& q, torch::Tensor const& k,
    torch::Tensor const& norm_weight_q, torch::Tensor const& norm_weight_k,
    torch::Tensor workspace, int64_t const rank, int64_t const nranks,
    double const eps) {
  TORCH_CHECK(q.scalar_type() == k.scalar_type(),
              "minimax_allreduce_rms_qk: q and k must have same dtype");
  TORCH_CHECK(q.dim() == 2 && k.dim() == 2,
              "minimax_allreduce_rms_qk: q and k must be 2D");
  TORCH_CHECK(q.size(0) == k.size(0),
              "minimax_allreduce_rms_qk: q and k must have same num_token");
  TORCH_CHECK(rank < nranks,
              "minimax_allreduce_rms_qk: rank must be less than nranks");
  int64_t head_dim_q = q.size(-1);
  int64_t head_dim_k = k.size(-1);
  auto params = vllm::tensorrt_llm::MiniMaxReduceRMSParams();
  params.nranks = static_cast<int>(nranks);
  params.rank = static_cast<int>(rank);
  params.dtype = q.scalar_type();
  params.size_q = static_cast<int>(q.numel());
  params.hidden_dim = static_cast<int>(q.size(-1));
  params.size_k = static_cast<int>(k.numel());
  params.hidden_dim_k = static_cast<int>(k.size(-1));
  params.workspace = reinterpret_cast<void**>(workspace.mutable_data_ptr());
  params.allreduce_in = q.data_ptr();
  params.rms_gamma = norm_weight_q.data_ptr();
  params.allreduce_in_k = k.data_ptr();
  params.rms_gamma_k = norm_weight_k.data_ptr();
  params.rms_eps = static_cast<float>(eps);
  params.stream = at::cuda::getCurrentCUDAStream(q.get_device());

  torch::Tensor rms_norm_out_q = torch::empty_like(q);
  torch::Tensor rms_norm_out_k = torch::empty_like(k);
  params.rms_norm_out = rms_norm_out_q.mutable_data_ptr();
  params.rms_norm_out_k = rms_norm_out_k.mutable_data_ptr();

  vllm::tensorrt_llm::minimax_reduce_rms_op(params);

  return {rms_norm_out_q, rms_norm_out_k};
}
