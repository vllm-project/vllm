// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Direct physical-block ZoomKV retrieval kernels. These kernels consume the
// request's block-table slice and index the global summary pools in-place,
// avoiding the advanced-index + permute + contiguous materialization chain.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace {

constexpr int SCORE_THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int CHUNK_SIZE = 16;
constexpr int HEAD_DIM_128 = 128;
constexpr int WARPS_PER_BLOCK = 8;

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_reduce_sum(float value) {
  __shared__ float warp_sums[BLOCK_THREADS / WARP_SIZE];
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();
  value = threadIdx.x < (BLOCK_THREADS / WARP_SIZE) ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffff, value, offset);
    }
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ bool warp_reduce_or(bool value) {
  return __any_sync(0xffffffff, value);
}

// BF16 D=128: one warp scores one parent; vectorized 4-elem/thread loads.
__global__ void quest_parent_score_physical_bf16_d128_kernel(
    const __nv_bfloat16* __restrict__ q,
    const int32_t* __restrict__ physical_ids,
    const int32_t* __restrict__ actual_num_chunks,
    const __nv_bfloat16* __restrict__ global_min,
    const __nv_bfloat16* __restrict__ global_max,
    const bool* __restrict__ global_valid,
    const __nv_bfloat16* __restrict__ global_parent_min,
    const __nv_bfloat16* __restrict__ global_parent_max,
    const bool* __restrict__ global_parent_valid,
    const int32_t* __restrict__ global_parent_first_child,
    float* __restrict__ scores,
    int batch, int kv_heads, int num_blocks, int n_chunks, int factor,
    int n_parent, int64_t pid_stride_b, int64_t pid_stride_n,
    int64_t summary_stride_p, int64_t summary_stride_h,
    int64_t score_stride_b, int64_t score_stride_h) {
  constexpr int D = HEAD_DIM_128;
  constexpr int ELEMS = D / WARP_SIZE;  // 4
  const int parent = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x / WARP_SIZE);
  const int bh = blockIdx.y;
  const int b = bh / kv_heads;
  const int h = bh - b * kv_heads;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  if (b >= batch || parent >= n_parent) return;
  const int actual_chunks =
      actual_num_chunks ? min(actual_num_chunks[b], n_chunks) : n_chunks;
  if (parent >= (actual_chunks + factor - 1) / factor) return;

  const int child_begin = parent * factor;
  int32_t my_phys = -1;
  bool my_valid = false;
  if (lane < factor) {
    const int child = child_begin + lane;
    if (child < actual_chunks) {
      my_phys = physical_ids[static_cast<int64_t>(b) * pid_stride_b +
                             static_cast<int64_t>(child) * pid_stride_n];
      my_valid = my_phys >= 0 && my_phys < num_blocks && global_valid[my_phys];
    }
  }

  const int32_t first_phys = __shfl_sync(0xffffffff, my_phys, 0);
  const int32_t anchor_phys = __shfl_sync(0xffffffff, my_phys, factor - 1);
  bool all_children_valid = true;
#pragma unroll 1
  for (int child_off = 0; child_off < WARP_SIZE; ++child_off) {
    if (child_off >= factor) break;
    const bool valid =
        __shfl_sync(0xffffffff, my_valid ? 1 : 0, child_off) != 0;
    all_children_valid = all_children_valid && valid;
  }
  const bool use_precomputed =
      global_parent_min != nullptr && global_parent_max != nullptr &&
      global_parent_valid != nullptr &&
      global_parent_first_child != nullptr && child_begin + factor - 1 < actual_chunks &&
      anchor_phys >= 0 && anchor_phys < num_blocks &&
      global_parent_valid[anchor_phys] &&
      global_parent_first_child[anchor_phys] == first_phys &&
      all_children_valid;

  const int64_t q_base =
      (static_cast<int64_t>(b) * kv_heads + h) * D;
  float acc = 0.0f;
  bool any_valid = false;
  const int d_base = lane * ELEMS;
#pragma unroll
  for (int i = 0; i < ELEMS; ++i) {
    const int d = d_base + i;
    float parent_min = INFINITY;
    float parent_max = -INFINITY;
    bool dim_valid = false;
    if (use_precomputed) {
      const int64_t base = static_cast<int64_t>(anchor_phys) * summary_stride_p +
                           static_cast<int64_t>(h) * summary_stride_h;
      parent_min = __bfloat162float(global_parent_min[base + d]);
      parent_max = __bfloat162float(global_parent_max[base + d]);
      dim_valid = true;
    } else {
#pragma unroll 1
      for (int child_off = 0; child_off < WARP_SIZE; ++child_off) {
        if (child_off >= factor) break;
        const int32_t physical = __shfl_sync(0xffffffff, my_phys, child_off);
        const bool valid =
            __shfl_sync(0xffffffff, my_valid ? 1 : 0, child_off) != 0;
        if (!valid) continue;
        const int64_t base = static_cast<int64_t>(physical) * summary_stride_p +
                             static_cast<int64_t>(h) * summary_stride_h;
        parent_min = fminf(parent_min, __bfloat162float(global_min[base + d]));
        parent_max = fmaxf(parent_max, __bfloat162float(global_max[base + d]));
        dim_valid = true;
      }
    }
    any_valid = any_valid || dim_valid;
    if (dim_valid) {
      const float qv = __bfloat162float(q[q_base + d]);
      acc += fmaxf(qv * parent_min, qv * parent_max);
    }
  }
  acc = warp_reduce_sum(acc);
  const bool parent_valid = warp_reduce_or(any_valid);
  if (lane == 0) {
    scores[static_cast<int64_t>(b) * score_stride_b +
           static_cast<int64_t>(h) * score_stride_h + parent] =
        parent_valid ? acc : -INFINITY;
  }
}

// BF16 D=128: one warp scores one child candidate; multi-warp blocks.
__global__ void quest_sub_score_physical_bf16_d128_kernel(
    const __nv_bfloat16* __restrict__ q,
    const int32_t* __restrict__ physical_ids,
    const int32_t* __restrict__ actual_num_chunks,
    const __nv_bfloat16* __restrict__ global_min,
    const __nv_bfloat16* __restrict__ global_max,
    const bool* __restrict__ global_valid,
    const int64_t* __restrict__ large_ids, float* __restrict__ scores,
    int batch, int kv_heads, int num_blocks, int n_chunks, int n_selected,
    int factor, int scores_n, int64_t pid_stride_b, int64_t pid_stride_n,
    int64_t summary_stride_p, int64_t summary_stride_h) {
  constexpr int D = HEAD_DIM_128;
  constexpr int ELEMS = D / WARP_SIZE;
  const int pos = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x / WARP_SIZE);
  const int bh = blockIdx.y;
  const int b = bh / kv_heads;
  const int h = bh - b * kv_heads;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  if (b >= batch || pos >= n_selected * factor) return;
  const int actual_chunks =
      actual_num_chunks ? min(actual_num_chunks[b], n_chunks) : n_chunks;

  const int selected_parent = pos / factor;
  const int child_off = pos - selected_parent * factor;
  const int64_t parent =
      large_ids[(static_cast<int64_t>(b) * kv_heads + h) * n_selected +
                selected_parent];
  const int64_t child64 = parent * factor + child_off;
  bool valid = child64 >= 0 && child64 < actual_chunks;
  int32_t physical = -1;
  if (valid) {
    physical = physical_ids[static_cast<int64_t>(b) * pid_stride_b +
                            child64 * pid_stride_n];
    valid = physical >= 0 && physical < num_blocks && global_valid[physical];
  }

  float acc = 0.0f;
  if (valid) {
    const int64_t summary_base =
        static_cast<int64_t>(physical) * summary_stride_p +
        static_cast<int64_t>(h) * summary_stride_h;
    const int64_t q_base =
        (static_cast<int64_t>(b) * kv_heads + h) * D;
    const int d_base = lane * ELEMS;
#pragma unroll
    for (int i = 0; i < ELEMS; ++i) {
      const int d = d_base + i;
      const float qv = __bfloat162float(q[q_base + d]);
      const float mn = __bfloat162float(global_min[summary_base + d]);
      const float mx = __bfloat162float(global_max[summary_base + d]);
      acc += fmaxf(qv * mn, qv * mx);
    }
  }
  acc = warp_reduce_sum(acc);
  if (lane == 0) {
    scores[(static_cast<int64_t>(b) * kv_heads + h) * scores_n + pos] =
        valid ? acc : -INFINITY;
  }
}

__global__ void density_score_physical_bf16_d128_kernel(
    const int64_t* __restrict__ chunk_ids,
    const int32_t* __restrict__ physical_ids,
    const int32_t* __restrict__ actual_num_chunks,
    const __nv_bfloat16* __restrict__ global_centroid,
    const bool* __restrict__ global_valid, const __nv_bfloat16* __restrict__ q,
    float* __restrict__ scores, int batch, int kv_heads, int num_blocks,
    int n_chunks, int nk, int64_t pid_stride_b, int64_t pid_stride_n,
    int64_t centroid_stride_p, int64_t centroid_stride_h) {
  constexpr int D = HEAD_DIM_128;
  constexpr int ELEMS = D / WARP_SIZE;
  const int slot = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x / WARP_SIZE);
  const int bh = blockIdx.y;
  const int b = bh / kv_heads;
  const int h = bh - b * kv_heads;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  if (b >= batch || slot >= nk) return;
  const int actual_chunks =
      actual_num_chunks ? min(actual_num_chunks[b], n_chunks) : n_chunks;

  const int64_t chunk =
      chunk_ids[(static_cast<int64_t>(b) * kv_heads + h) * nk + slot];
  bool valid = chunk >= 0 && chunk < actual_chunks;
  int32_t physical = -1;
  if (valid) {
    physical = physical_ids[static_cast<int64_t>(b) * pid_stride_b +
                            chunk * pid_stride_n];
    valid = physical >= 0 && physical < num_blocks && global_valid[physical];
  }

  float acc = 0.0f;
  if (valid) {
    const int64_t cent_base =
        static_cast<int64_t>(physical) * centroid_stride_p +
        static_cast<int64_t>(h) * centroid_stride_h;
    const int64_t q_base =
        (static_cast<int64_t>(b) * kv_heads + h) * D;
    const int d_base = lane * ELEMS;
#pragma unroll
    for (int i = 0; i < ELEMS; ++i) {
      const int d = d_base + i;
      acc += __bfloat162float(global_centroid[cent_base + d]) *
             __bfloat162float(q[q_base + d]);
    }
  }
  acc = warp_reduce_sum(acc);
  if (lane == 0) {
    scores[(static_cast<int64_t>(b) * kv_heads + h) * nk + slot] =
        valid ? acc : -INFINITY;
  }
}

template <typename scalar_t>
__global__ void quest_chunk_score_physical_kernel(
    const scalar_t* __restrict__ q, const int32_t* __restrict__ physical_ids,
    const int32_t* __restrict__ actual_num_chunks,
    const scalar_t* __restrict__ global_min,
    const scalar_t* __restrict__ global_max,
    const bool* __restrict__ global_valid, float* __restrict__ scores,
    int batch, int kv_heads, int head_dim, int num_blocks, int n_chunks,
    int64_t pid_stride_b, int64_t pid_stride_n, int64_t summary_stride_p,
    int64_t summary_stride_h, int64_t score_stride_b,
    int64_t score_stride_h) {
  const int chunk = blockIdx.x;
  const int bh = blockIdx.y;
  const int b = bh / kv_heads;
  const int h = bh - b * kv_heads;
  const int tid = threadIdx.x;
  if (b >= batch || chunk >= n_chunks) return;
  const int actual_chunks =
      actual_num_chunks ? min(actual_num_chunks[b], n_chunks) : n_chunks;
  if (chunk >= actual_chunks) {
    if (tid == 0) {
      scores[static_cast<int64_t>(b) * score_stride_b +
             static_cast<int64_t>(h) * score_stride_h + chunk] = -INFINITY;
    }
    return;
  }

  const int32_t physical =
      physical_ids[static_cast<int64_t>(b) * pid_stride_b +
                   static_cast<int64_t>(chunk) * pid_stride_n];
  const bool valid =
      physical >= 0 && physical < num_blocks && global_valid[physical];
  float acc = 0.0f;
  if (valid) {
    const int64_t summary_base =
        static_cast<int64_t>(physical) * summary_stride_p +
        static_cast<int64_t>(h) * summary_stride_h;
    const int64_t q_base =
        (static_cast<int64_t>(b) * kv_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += SCORE_THREADS) {
      const float qv = static_cast<float>(q[q_base + d]);
      const float mn = static_cast<float>(global_min[summary_base + d]);
      const float mx = static_cast<float>(global_max[summary_base + d]);
      acc += fmaxf(qv * mn, qv * mx);
    }
  }
  acc = block_reduce_sum<SCORE_THREADS>(acc);
  if (tid == 0) {
    scores[static_cast<int64_t>(b) * score_stride_b +
           static_cast<int64_t>(h) * score_stride_h + chunk] =
        valid ? acc : -INFINITY;
  }
}

template <typename scalar_t>
__global__ void quest_parent_score_physical_kernel(
    const scalar_t* __restrict__ q, const int32_t* __restrict__ physical_ids,
    const int32_t* __restrict__ actual_num_chunks,
    const scalar_t* __restrict__ global_min,
    const scalar_t* __restrict__ global_max,
    const bool* __restrict__ global_valid,
    const scalar_t* __restrict__ global_parent_min,
    const scalar_t* __restrict__ global_parent_max,
    const bool* __restrict__ global_parent_valid,
    const int32_t* __restrict__ global_parent_first_child,
    float* __restrict__ scores,
    int batch, int kv_heads, int head_dim, int num_blocks, int n_chunks,
    int factor, int n_parent, int64_t pid_stride_b, int64_t pid_stride_n,
    int64_t summary_stride_p, int64_t summary_stride_h,
    int64_t score_stride_b, int64_t score_stride_h) {
  const int parent = blockIdx.x;
  const int bh = blockIdx.y;
  const int b = bh / kv_heads;
  const int h = bh - b * kv_heads;
  const int tid = threadIdx.x;
  if (b >= batch || parent >= n_parent) return;
  const int actual_chunks =
      actual_num_chunks ? min(actual_num_chunks[b], n_chunks) : n_chunks;
  if (parent >= (actual_chunks + factor - 1) / factor) return;

  const int child_begin = parent * factor;
  int32_t first_phys = -1;
  int32_t anchor_phys = -1;
  bool all_children_valid = true;
  for (int child_off = 0; child_off < factor; ++child_off) {
    const int child = child_begin + child_off;
    if (child >= actual_chunks) {
      all_children_valid = false;
      break;
    }
    const int32_t physical =
        physical_ids[static_cast<int64_t>(b) * pid_stride_b +
                     static_cast<int64_t>(child) * pid_stride_n];
    const bool valid =
        physical >= 0 && physical < num_blocks && global_valid[physical];
    all_children_valid = all_children_valid && valid;
    if (child_off == 0) first_phys = physical;
    if (child_off == factor - 1) anchor_phys = physical;
  }
  const bool use_precomputed =
      global_parent_min != nullptr && global_parent_max != nullptr &&
      global_parent_valid != nullptr &&
      global_parent_first_child != nullptr && child_begin + factor - 1 < actual_chunks &&
      anchor_phys >= 0 && anchor_phys < num_blocks &&
      global_parent_valid[anchor_phys] &&
      global_parent_first_child[anchor_phys] == first_phys &&
      all_children_valid;

  bool any_valid = false;
  float acc = 0.0f;
  const int64_t q_base =
      (static_cast<int64_t>(b) * kv_heads + h) * head_dim;
  for (int d = tid; d < head_dim; d += SCORE_THREADS) {
    float parent_min = INFINITY;
    float parent_max = -INFINITY;
    bool dim_valid = false;
    if (use_precomputed) {
      const int64_t base = static_cast<int64_t>(anchor_phys) * summary_stride_p +
                           static_cast<int64_t>(h) * summary_stride_h;
      parent_min = static_cast<float>(global_parent_min[base + d]);
      parent_max = static_cast<float>(global_parent_max[base + d]);
      dim_valid = true;
    } else {
      for (int child_off = 0; child_off < factor; ++child_off) {
        const int child = child_begin + child_off;
        if (child >= actual_chunks) break;
        const int32_t physical =
            physical_ids[static_cast<int64_t>(b) * pid_stride_b +
                         static_cast<int64_t>(child) * pid_stride_n];
        const bool valid =
            physical >= 0 && physical < num_blocks && global_valid[physical];
        if (!valid) continue;
        const int64_t base =
            static_cast<int64_t>(physical) * summary_stride_p +
            static_cast<int64_t>(h) * summary_stride_h;
        parent_min = fminf(parent_min, static_cast<float>(global_min[base + d]));
        parent_max = fmaxf(parent_max, static_cast<float>(global_max[base + d]));
        dim_valid = true;
      }
    }
    any_valid = any_valid || dim_valid;
    if (dim_valid) {
      const float qv = static_cast<float>(q[q_base + d]);
      acc += fmaxf(qv * parent_min, qv * parent_max);
    }
  }
  acc = block_reduce_sum<SCORE_THREADS>(acc);
  const bool parent_valid = __syncthreads_or(any_valid);
  if (tid == 0) {
    scores[static_cast<int64_t>(b) * score_stride_b +
           static_cast<int64_t>(h) * score_stride_h + parent] =
        parent_valid ? acc : -INFINITY;
  }
}

template <typename scalar_t>
__global__ void quest_sub_score_physical_kernel(
    const scalar_t* __restrict__ q, const int32_t* __restrict__ physical_ids,
    const int32_t* __restrict__ actual_num_chunks,
    const scalar_t* __restrict__ global_min,
    const scalar_t* __restrict__ global_max,
    const bool* __restrict__ global_valid,
    const int64_t* __restrict__ large_ids, float* __restrict__ scores,
    int batch, int kv_heads, int head_dim, int num_blocks, int n_chunks,
    int n_selected, int factor, int scores_n, int64_t pid_stride_b,
    int64_t pid_stride_n, int64_t summary_stride_p,
    int64_t summary_stride_h) {
  const int pos = blockIdx.x;
  const int bh = blockIdx.y;
  const int b = bh / kv_heads;
  const int h = bh - b * kv_heads;
  const int tid = threadIdx.x;
  if (b >= batch || pos >= n_selected * factor) return;
  const int actual_chunks =
      actual_num_chunks ? min(actual_num_chunks[b], n_chunks) : n_chunks;

  const int selected_parent = pos / factor;
  const int child_off = pos - selected_parent * factor;
  const int64_t parent =
      large_ids[(static_cast<int64_t>(b) * kv_heads + h) * n_selected +
                selected_parent];
  const int64_t child64 = parent * factor + child_off;
  bool valid = child64 >= 0 && child64 < actual_chunks;
  int32_t physical = -1;
  if (valid) {
    physical =
        physical_ids[static_cast<int64_t>(b) * pid_stride_b +
                     child64 * pid_stride_n];
    valid = physical >= 0 && physical < num_blocks && global_valid[physical];
  }

  float acc = 0.0f;
  if (valid) {
    const int64_t summary_base =
        static_cast<int64_t>(physical) * summary_stride_p +
        static_cast<int64_t>(h) * summary_stride_h;
    const int64_t q_base =
        (static_cast<int64_t>(b) * kv_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += SCORE_THREADS) {
      const float qv = static_cast<float>(q[q_base + d]);
      const float mn = static_cast<float>(global_min[summary_base + d]);
      const float mx = static_cast<float>(global_max[summary_base + d]);
      acc += fmaxf(qv * mn, qv * mx);
    }
  }
  acc = block_reduce_sum<SCORE_THREADS>(acc);
  if (tid == 0) {
    scores[(static_cast<int64_t>(b) * kv_heads + h) * scores_n + pos] =
        valid ? acc : -INFINITY;
  }
}

template <typename scalar_t>
__global__ void density_score_physical_kernel(
    const int64_t* __restrict__ chunk_ids,
    const int32_t* __restrict__ physical_ids,
    const int32_t* __restrict__ actual_num_chunks,
    const scalar_t* __restrict__ global_centroid,
    const bool* __restrict__ global_valid, const scalar_t* __restrict__ q,
    float* __restrict__ scores, int batch, int kv_heads, int head_dim,
    int num_blocks, int n_chunks, int nk, int64_t pid_stride_b,
    int64_t pid_stride_n, int64_t centroid_stride_p,
    int64_t centroid_stride_h) {
  const int slot = blockIdx.x;
  const int bh = blockIdx.y;
  const int b = bh / kv_heads;
  const int h = bh - b * kv_heads;
  const int tid = threadIdx.x;
  if (b >= batch || slot >= nk) return;
  const int actual_chunks =
      actual_num_chunks ? min(actual_num_chunks[b], n_chunks) : n_chunks;

  const int64_t chunk =
      chunk_ids[(static_cast<int64_t>(b) * kv_heads + h) * nk + slot];
  bool valid = chunk >= 0 && chunk < actual_chunks;
  int32_t physical = -1;
  if (valid) {
    physical = physical_ids[static_cast<int64_t>(b) * pid_stride_b +
                            chunk * pid_stride_n];
    valid = physical >= 0 && physical < num_blocks && global_valid[physical];
  }

  float acc = 0.0f;
  if (valid) {
    const int64_t cent_base =
        static_cast<int64_t>(physical) * centroid_stride_p +
        static_cast<int64_t>(h) * centroid_stride_h;
    const int64_t q_base =
        (static_cast<int64_t>(b) * kv_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += SCORE_THREADS) {
      acc += static_cast<float>(global_centroid[cent_base + d]) *
             static_cast<float>(q[q_base + d]);
    }
  }
  acc = block_reduce_sum<SCORE_THREADS>(acc);
  if (tid == 0) {
    scores[(static_cast<int64_t>(b) * kv_heads + h) * nk + slot] =
        valid ? acc : -INFINITY;
  }
}

struct PhysicalKiviParams {
  const int64_t* chunk_ids;
  const bool* dense_mask;
  const int32_t* physical_ids;
  const int32_t* actual_num_chunks;
  const int32_t* packed;
  const __nv_bfloat16* chunk_min;
  const __nv_bfloat16* chunk_max;
  const bool* global_valid;
  const __nv_bfloat16* q;
  float* out_scores;
  int64_t* out_indices;
  int batch;
  int kv_heads;
  int num_blocks;
  int n_chunks;
  int nk;
  int head_dim;
  int n_pack;
  int dense_topk;
  int sparse_topk;
  int output_slots;
  int token_offset;
  int64_t pid_stride_b;
  int64_t pid_stride_n;
  int64_t packed_stride_p;
  int64_t packed_stride_h;
  int64_t packed_stride_pack;
  int64_t minmax_stride_p;
  int64_t minmax_stride_h;
};

template <int HEAD_DIM>
__global__ void kivi_physical_kernel(PhysicalKiviParams p) {
  constexpr int CHUNKS_PER_WARP = WARP_SIZE / CHUNK_SIZE;
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;
  const int sub = lane / CHUNK_SIZE;
  const int lane_in_chunk = lane % CHUNK_SIZE;
  const int chunks_per_block = (blockDim.x / WARP_SIZE) * CHUNKS_PER_WARP;
  const int slot = blockIdx.z * chunks_per_block + warp * CHUNKS_PER_WARP + sub;
  const bool valid_slot = b < p.batch && h < p.kv_heads && slot < p.nk;

  int64_t logical_chunk = -1;
  int32_t physical = -1;
  bool dense = false;
  bool valid = false;
  if (valid_slot) {
    const int64_t bh_slot =
        (static_cast<int64_t>(b) * p.kv_heads + h) * p.nk + slot;
    logical_chunk = p.chunk_ids[bh_slot];
    dense = p.dense_mask[bh_slot];
    const int actual_chunks =
        p.actual_num_chunks
            ? min(p.actual_num_chunks[b], p.n_chunks)
            : p.n_chunks;
    if (logical_chunk >= 0 && logical_chunk < actual_chunks) {
      physical =
          p.physical_ids[static_cast<int64_t>(b) * p.pid_stride_b +
                         logical_chunk * p.pid_stride_n];
      valid = physical >= 0 && physical < p.num_blocks &&
              p.global_valid[physical];
    }
  }

  int token_idx =
      p.token_offset + static_cast<int>(logical_chunk) * CHUNK_SIZE +
      lane_in_chunk;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  const int local_chunk = warp * CHUNKS_PER_WARP + sub;
  __nv_bfloat16* sm_q = smem + local_chunk * 3 * HEAD_DIM;
  __nv_bfloat16* sm_min = sm_q + HEAD_DIM;
  __nv_bfloat16* sm_scale = sm_min + HEAD_DIM;
  const int64_t mm_base =
      static_cast<int64_t>(physical) * p.minmax_stride_p +
      static_cast<int64_t>(h) * p.minmax_stride_h;
  const int64_t q_base =
      (static_cast<int64_t>(b) * p.kv_heads + h) * HEAD_DIM;
  if (valid) {
    for (int pack_idx = lane_in_chunk; pack_idx < HEAD_DIM / 8;
         pack_idx += CHUNK_SIZE) {
      const int d_base = pack_idx * 8;
      const uint4 qv = *reinterpret_cast<const uint4*>(p.q + q_base + d_base);
      const uint4 mnv =
          *reinterpret_cast<const uint4*>(p.chunk_min + mm_base + d_base);
      const uint4 mxv =
          *reinterpret_cast<const uint4*>(p.chunk_max + mm_base + d_base);
      __nv_bfloat16 mn[8], mx[8], scale[8];
      *reinterpret_cast<uint4*>(mn) = mnv;
      *reinterpret_cast<uint4*>(mx) = mxv;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        scale[j] =
            __hdiv(__hsub(mx[j], mn[j]), __float2bfloat16(15.0f));
      }
      *reinterpret_cast<uint4*>(sm_q + d_base) = qv;
      *reinterpret_cast<uint4*>(sm_min + d_base) = mnv;
      *reinterpret_cast<uint4*>(sm_scale + d_base) =
          *reinterpret_cast<uint4*>(scale);
    }
  }
  __syncwarp();

  float score = -INFINITY;
  if (valid) {
    float acc = 0.0f;
#pragma unroll
    for (int pack_idx = 0; pack_idx < HEAD_DIM / 8; ++pack_idx) {
      const int32_t codes =
          p.packed[static_cast<int64_t>(physical) * p.packed_stride_p +
                   static_cast<int64_t>(h) * p.packed_stride_h +
                   static_cast<int64_t>(pack_idx) * p.packed_stride_pack +
                   lane_in_chunk];
      const int d_base = pack_idx * 8;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int code = (static_cast<uint32_t>(codes) >> (j * 4)) & 0xF;
        const __nv_bfloat16 kval =
            __hadd(__hmul(__int2bfloat16_rn(code), sm_scale[d_base + j]),
                   sm_min[d_base + j]);
        acc += __bfloat162float(__hmul(kval, sm_q[d_base + j]));
      }
    }
    score = acc;
  }

#pragma unroll
  for (int k = 2; k <= CHUNK_SIZE; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j >= 1; j >>= 1) {
      const float other_score = __shfl_xor_sync(0xffffffff, score, j);
      const int other_idx = __shfl_xor_sync(0xffffffff, token_idx, j);
      const bool desc = ((lane_in_chunk & k) == 0);
      const bool lower = ((lane_in_chunk & j) == 0);
      const bool want_larger = desc == lower;
      const bool swap =
          want_larger ? other_score > score : other_score < score;
      if (swap) {
        score = other_score;
        token_idx = other_idx;
      }
    }
  }

  if (valid_slot) {
    const int out_base =
        (static_cast<int64_t>(b) * p.kv_heads + h) * p.nk *
            p.output_slots +
        static_cast<int64_t>(slot) * p.output_slots + lane_in_chunk;
    const int keep = dense ? p.dense_topk : p.sparse_topk;
    if (lane_in_chunk < p.output_slots) {
      if (valid && lane_in_chunk < keep) {
        p.out_scores[out_base] = score;
        p.out_indices[out_base] = token_idx;
      } else {
        p.out_scores[out_base] = -INFINITY;
        p.out_indices[out_base] = -1;
      }
    }
  }
}

void check_direct_common(const at::Tensor& q, const at::Tensor& physical_ids,
                         const at::Tensor& global_min,
                         const at::Tensor& global_max,
                         const at::Tensor& global_valid) {
  TORCH_CHECK(q.is_cuda() && physical_ids.is_cuda() && global_min.is_cuda() &&
                  global_max.is_cuda() && global_valid.is_cuda(),
              "direct physical retrieval tensors must be CUDA");
  TORCH_CHECK(physical_ids.scalar_type() == at::ScalarType::Int,
              "physical_ids must be int32");
  TORCH_CHECK(global_valid.scalar_type() == at::ScalarType::Bool,
              "global_valid must be bool");
  TORCH_CHECK(q.dim() == 3 && physical_ids.dim() == 2,
              "q/physical_ids must be [B,H,D]/[B,N]");
  TORCH_CHECK(global_min.dim() == 3 && global_max.dim() == 3,
              "global min/max must be [num_blocks,H,D]");
  TORCH_CHECK(q.scalar_type() == global_min.scalar_type() &&
                  q.scalar_type() == global_max.scalar_type(),
              "q and global summaries must share dtype");
  TORCH_CHECK(q.size(1) == global_min.size(1) &&
                  q.size(2) == global_min.size(2),
              "q and global summary head geometry mismatch");
  TORCH_CHECK(global_max.sizes() == global_min.sizes(),
              "global min/max shape mismatch");
}

const int32_t* check_actual_num_chunks(
    const std::optional<at::Tensor>& actual_num_chunks,
    const at::Tensor& q) {
  if (!actual_num_chunks.has_value()) return nullptr;
  const auto& actual = *actual_num_chunks;
  TORCH_CHECK(actual.is_cuda() && actual.device() == q.device(),
              "actual_num_chunks must be CUDA on the same device as q");
  TORCH_CHECK(actual.scalar_type() == at::ScalarType::Int &&
                  actual.dim() == 1 && actual.size(0) == q.size(0) &&
                  actual.is_contiguous(),
              "actual_num_chunks must be contiguous int32 [B]");
  return actual.data_ptr<int32_t>();
}

void check_launch(const char* name) {
  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, name, " kernel failed: ",
              cudaGetErrorString(error));
}

}  // namespace

void quest_chunk_score_physical_cuda(
    at::Tensor q, at::Tensor physical_ids, at::Tensor global_min,
    at::Tensor global_max, at::Tensor global_valid, at::Tensor scores,
    int64_t n_chunks, std::optional<at::Tensor> actual_num_chunks) {
  check_direct_common(q, physical_ids, global_min, global_max, global_valid);
  TORCH_CHECK(scores.scalar_type() == at::ScalarType::Float &&
                  scores.dim() == 3,
              "scores must be float32 [B,H,N]");
  const int batch = q.size(0), heads = q.size(1), dim = q.size(2);
  const int32_t* actual_ptr =
      check_actual_num_chunks(actual_num_chunks, q);
  c10::cuda::CUDAGuard guard(q.device());
  const dim3 grid(n_chunks, batch * heads);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
      "quest_chunk_score_physical", [&] {
        quest_chunk_score_physical_kernel<scalar_t>
            <<<grid, SCORE_THREADS, 0, stream>>>(
                q.data_ptr<scalar_t>(), physical_ids.data_ptr<int32_t>(),
                actual_ptr,
                global_min.data_ptr<scalar_t>(), global_max.data_ptr<scalar_t>(),
                global_valid.data_ptr<bool>(), scores.data_ptr<float>(), batch,
                heads, dim, global_min.size(0), n_chunks,
                physical_ids.stride(0), physical_ids.stride(1),
                global_min.stride(0), global_min.stride(1), scores.stride(0),
                scores.stride(1));
      });
  check_launch("quest_chunk_score_physical");
}

void quest_parent_score_physical_cuda(
    at::Tensor q, at::Tensor physical_ids, at::Tensor global_min,
    at::Tensor global_max, at::Tensor global_valid, at::Tensor scores,
    int64_t n_chunks, int64_t factor,
    std::optional<at::Tensor> actual_num_chunks,
    std::optional<at::Tensor> global_parent_min,
    std::optional<at::Tensor> global_parent_max,
    std::optional<at::Tensor> global_parent_valid,
    std::optional<at::Tensor> global_parent_first_child) {
  check_direct_common(q, physical_ids, global_min, global_max, global_valid);
  const int batch = q.size(0), heads = q.size(1), dim = q.size(2);
  const int n_parent = n_chunks / factor;
  const int32_t* actual_ptr =
      check_actual_num_chunks(actual_num_chunks, q);
  const bool have_parent_pools =
      global_parent_min.has_value() && global_parent_max.has_value() &&
      global_parent_valid.has_value() &&
      global_parent_first_child.has_value();
  if (have_parent_pools) {
    TORCH_CHECK(global_parent_min->sizes() == global_min.sizes() &&
                    global_parent_max->sizes() == global_max.sizes(),
                "parent/global min/max shape mismatch");
    TORCH_CHECK(global_parent_valid->scalar_type() == at::ScalarType::Bool &&
                    global_parent_valid->dim() == 1 &&
                    global_parent_valid->size(0) == global_min.size(0),
                "global_parent_valid must be bool [num_blocks]");
    TORCH_CHECK(global_parent_first_child->scalar_type() ==
                        at::ScalarType::Int &&
                    global_parent_first_child->dim() == 1 &&
                    global_parent_first_child->size(0) == global_min.size(0),
                "global_parent_first_child must be int32 [num_blocks]");
  }
  c10::cuda::CUDAGuard guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (q.scalar_type() == at::ScalarType::BFloat16 && dim == HEAD_DIM_128 &&
      factor <= WARP_SIZE) {
    const dim3 grid((n_parent + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
                    batch * heads);
    const int threads = WARPS_PER_BLOCK * WARP_SIZE;
    quest_parent_score_physical_bf16_d128_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        physical_ids.data_ptr<int32_t>(),
        actual_ptr,
        reinterpret_cast<const __nv_bfloat16*>(global_min.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(global_max.data_ptr()),
        global_valid.data_ptr<bool>(),
        have_parent_pools
            ? reinterpret_cast<const __nv_bfloat16*>(global_parent_min->data_ptr())
            : nullptr,
        have_parent_pools
            ? reinterpret_cast<const __nv_bfloat16*>(global_parent_max->data_ptr())
            : nullptr,
        have_parent_pools ? global_parent_valid->data_ptr<bool>() : nullptr,
        have_parent_pools ? global_parent_first_child->data_ptr<int32_t>()
                          : nullptr,
        scores.data_ptr<float>(), batch, heads,
        global_min.size(0), n_chunks, factor, n_parent, physical_ids.stride(0),
        physical_ids.stride(1), global_min.stride(0), global_min.stride(1),
        scores.stride(0), scores.stride(1));
  } else {
    const dim3 grid(n_parent, batch * heads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
        "quest_parent_score_physical", [&] {
          quest_parent_score_physical_kernel<scalar_t>
              <<<grid, SCORE_THREADS, 0, stream>>>(
                  q.data_ptr<scalar_t>(), physical_ids.data_ptr<int32_t>(),
                  actual_ptr,
                  global_min.data_ptr<scalar_t>(),
                  global_max.data_ptr<scalar_t>(),
                  global_valid.data_ptr<bool>(),
                  have_parent_pools ? global_parent_min->data_ptr<scalar_t>()
                                    : nullptr,
                  have_parent_pools ? global_parent_max->data_ptr<scalar_t>()
                                    : nullptr,
                  have_parent_pools ? global_parent_valid->data_ptr<bool>()
                                    : nullptr,
                  have_parent_pools
                      ? global_parent_first_child->data_ptr<int32_t>()
                      : nullptr,
                  scores.data_ptr<float>(), batch, heads, dim,
                  global_min.size(0), n_chunks, factor, n_parent,
                  physical_ids.stride(0), physical_ids.stride(1),
                  global_min.stride(0), global_min.stride(1), scores.stride(0),
                  scores.stride(1));
        });
  }
  check_launch("quest_parent_score_physical");
}

void quest_sub_score_physical_cuda(
    at::Tensor q, at::Tensor physical_ids, at::Tensor global_min,
    at::Tensor global_max, at::Tensor global_valid, at::Tensor large_ids,
    at::Tensor scores, int64_t n_selected, int64_t factor,
    int64_t n_chunks, std::optional<at::Tensor> actual_num_chunks) {
  check_direct_common(q, physical_ids, global_min, global_max, global_valid);
  const int batch = q.size(0), heads = q.size(1), dim = q.size(2);
  const int32_t* actual_ptr =
      check_actual_num_chunks(actual_num_chunks, q);
  c10::cuda::CUDAGuard guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int n_pos = static_cast<int>(n_selected * factor);
  if (q.scalar_type() == at::ScalarType::BFloat16 && dim == HEAD_DIM_128) {
    const dim3 grid((n_pos + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
                    batch * heads);
    const int threads = WARPS_PER_BLOCK * WARP_SIZE;
    quest_sub_score_physical_bf16_d128_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        physical_ids.data_ptr<int32_t>(),
        actual_ptr,
        reinterpret_cast<const __nv_bfloat16*>(global_min.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(global_max.data_ptr()),
        global_valid.data_ptr<bool>(), large_ids.data_ptr<int64_t>(),
        scores.data_ptr<float>(), batch, heads, global_min.size(0), n_chunks,
        n_selected, factor, scores.size(2), physical_ids.stride(0),
        physical_ids.stride(1), global_min.stride(0), global_min.stride(1));
  } else {
    const dim3 grid(n_pos, batch * heads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
        "quest_sub_score_physical", [&] {
          quest_sub_score_physical_kernel<scalar_t>
              <<<grid, SCORE_THREADS, 0, stream>>>(
                  q.data_ptr<scalar_t>(), physical_ids.data_ptr<int32_t>(),
                  actual_ptr,
                  global_min.data_ptr<scalar_t>(),
                  global_max.data_ptr<scalar_t>(),
                  global_valid.data_ptr<bool>(), large_ids.data_ptr<int64_t>(),
                  scores.data_ptr<float>(), batch, heads, dim,
                  global_min.size(0), n_chunks, n_selected, factor,
                  scores.size(2), physical_ids.stride(0),
                  physical_ids.stride(1), global_min.stride(0),
                  global_min.stride(1));
        });
  }
  check_launch("quest_sub_score_physical");
}

void density_score_physical_cuda(
    at::Tensor chunk_ids, at::Tensor physical_ids, at::Tensor global_centroid,
    at::Tensor global_valid, at::Tensor q, at::Tensor scores,
    int64_t n_chunks, std::optional<at::Tensor> actual_num_chunks) {
  TORCH_CHECK(chunk_ids.scalar_type() == at::ScalarType::Long &&
                  physical_ids.scalar_type() == at::ScalarType::Int,
              "chunk_ids/physical_ids must be int64/int32");
  TORCH_CHECK(q.scalar_type() == global_centroid.scalar_type(),
              "q/centroid dtype mismatch");
  const int batch = q.size(0), heads = q.size(1), dim = q.size(2);
  const int32_t* actual_ptr =
      check_actual_num_chunks(actual_num_chunks, q);
  const int nk = chunk_ids.size(2);
  c10::cuda::CUDAGuard guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (q.scalar_type() == at::ScalarType::BFloat16 && dim == HEAD_DIM_128) {
    const dim3 grid((nk + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
                    batch * heads);
    const int threads = WARPS_PER_BLOCK * WARP_SIZE;
    density_score_physical_bf16_d128_kernel<<<grid, threads, 0, stream>>>(
        chunk_ids.data_ptr<int64_t>(), physical_ids.data_ptr<int32_t>(),
        actual_ptr,
        reinterpret_cast<const __nv_bfloat16*>(global_centroid.data_ptr()),
        global_valid.data_ptr<bool>(),
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        scores.data_ptr<float>(), batch, heads, global_centroid.size(0),
        n_chunks, nk, physical_ids.stride(0), physical_ids.stride(1),
        global_centroid.stride(0), global_centroid.stride(1));
  } else {
    const dim3 grid(nk, batch * heads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
        "density_score_physical", [&] {
          density_score_physical_kernel<scalar_t>
              <<<grid, SCORE_THREADS, 0, stream>>>(
                  chunk_ids.data_ptr<int64_t>(),
                  physical_ids.data_ptr<int32_t>(),
                  actual_ptr,
                  global_centroid.data_ptr<scalar_t>(),
                  global_valid.data_ptr<bool>(), q.data_ptr<scalar_t>(),
                  scores.data_ptr<float>(), batch, heads, dim,
                  global_centroid.size(0), n_chunks, nk,
                  physical_ids.stride(0), physical_ids.stride(1),
                  global_centroid.stride(0), global_centroid.stride(1));
        });
  }
  check_launch("density_score_physical");
}

void kivi_physical_cuda(
    at::Tensor chunk_ids, at::Tensor dense_mask, at::Tensor physical_ids,
    at::Tensor global_packed, at::Tensor global_min, at::Tensor global_max,
    at::Tensor global_valid, at::Tensor q, int64_t dense_topk,
    int64_t sparse_topk, int64_t token_offset, at::Tensor out_scores,
    at::Tensor out_indices, std::optional<at::Tensor> actual_num_chunks) {
  TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16,
              "direct KIVI currently requires bfloat16");
  TORCH_CHECK(global_packed.scalar_type() == at::ScalarType::Int,
              "global packed summary must be int32");
  const int batch = q.size(0), heads = q.size(1), dim = q.size(2);
  const int32_t* actual_ptr =
      check_actual_num_chunks(actual_num_chunks, q);
  const int nk = chunk_ids.size(2);
  const int output_slots =
      static_cast<int>(dense_topk > sparse_topk ? dense_topk : sparse_topk);
  TORCH_CHECK(dense_topk >= 1 && dense_topk <= CHUNK_SIZE &&
                  sparse_topk >= 1 && sparse_topk <= CHUNK_SIZE,
              "direct KIVI top-k values must be in [1, ", CHUNK_SIZE, "]");
  TORCH_CHECK(out_scores.size(2) >= nk * output_slots &&
                  out_indices.size(2) >= nk * output_slots,
              "direct KIVI output buffers need at least ", nk * output_slots,
              " slots");
  PhysicalKiviParams p{
      chunk_ids.data_ptr<int64_t>(),
      dense_mask.data_ptr<bool>(),
      physical_ids.data_ptr<int32_t>(),
      actual_ptr,
      global_packed.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(global_min.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(global_max.data_ptr()),
      global_valid.data_ptr<bool>(),
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      out_scores.data_ptr<float>(),
      out_indices.data_ptr<int64_t>(),
      batch,
      heads,
      static_cast<int>(global_packed.size(0)),
      static_cast<int>(physical_ids.size(1)),
      nk,
      dim,
      static_cast<int>(global_packed.size(2)),
      static_cast<int>(dense_topk),
      static_cast<int>(sparse_topk),
      output_slots,
      static_cast<int>(token_offset),
      physical_ids.stride(0),
      physical_ids.stride(1),
      global_packed.stride(0),
      global_packed.stride(1),
      global_packed.stride(2),
      global_min.stride(0),
      global_min.stride(1)};
  constexpr int threads = 128;
  const int chunks_per_block = (threads / WARP_SIZE) * (WARP_SIZE / CHUNK_SIZE);
  const dim3 grid(batch, heads, (nk + chunks_per_block - 1) / chunks_per_block);
  const int smem_bytes =
      chunks_per_block * 3 * dim * sizeof(__nv_bfloat16);
  c10::cuda::CUDAGuard guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (dim == 128) {
    kivi_physical_kernel<128><<<grid, threads, smem_bytes, stream>>>(p);
  } else if (dim == 256) {
    kivi_physical_kernel<256><<<grid, threads, smem_bytes, stream>>>(p);
  } else {
    TORCH_CHECK(false, "direct KIVI head_dim must be 128 or 256");
  }
  check_launch("kivi_physical");
}
