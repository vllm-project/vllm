// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int HEAD_DIM_FAST = 128;
constexpr int GENERIC_BLOCK_THREADS = 256;

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}

template <int BLOCK_THREADS>
__device__ __forceinline__ float block_reduce_sum(float value) {
  __shared__ float warp_sums[BLOCK_THREADS / WARP_SIZE];
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;
  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < (BLOCK_THREADS / WARP_SIZE) ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  return value;
}

__device__ __forceinline__ float score_bf16x4(const uint2 q_u2,
                                              const uint2 min_u2,
                                              const uint2 max_u2) {
  const __nv_bfloat162 q_lo = *reinterpret_cast<const __nv_bfloat162*>(&q_u2.x);
  const __nv_bfloat162 q_hi = *reinterpret_cast<const __nv_bfloat162*>(&q_u2.y);
  const __nv_bfloat162 mn_lo =
      *reinterpret_cast<const __nv_bfloat162*>(&min_u2.x);
  const __nv_bfloat162 mn_hi =
      *reinterpret_cast<const __nv_bfloat162*>(&min_u2.y);
  const __nv_bfloat162 mx_lo =
      *reinterpret_cast<const __nv_bfloat162*>(&max_u2.x);
  const __nv_bfloat162 mx_hi =
      *reinterpret_cast<const __nv_bfloat162*>(&max_u2.y);

  const float q0 = __bfloat162float(__low2bfloat16(q_lo));
  const float q1 = __bfloat162float(__high2bfloat16(q_lo));
  const float q2 = __bfloat162float(__low2bfloat16(q_hi));
  const float q3 = __bfloat162float(__high2bfloat16(q_hi));
  const float mn0 = __bfloat162float(__low2bfloat16(mn_lo));
  const float mn1 = __bfloat162float(__high2bfloat16(mn_lo));
  const float mn2 = __bfloat162float(__low2bfloat16(mn_hi));
  const float mn3 = __bfloat162float(__high2bfloat16(mn_hi));
  const float mx0 = __bfloat162float(__low2bfloat16(mx_lo));
  const float mx1 = __bfloat162float(__high2bfloat16(mx_lo));
  const float mx2 = __bfloat162float(__low2bfloat16(mx_hi));
  const float mx3 = __bfloat162float(__high2bfloat16(mx_hi));

  float sum = fmaxf(q0 * mn0, q0 * mx0);
  sum += fmaxf(q1 * mn1, q1 * mx1);
  sum += fmaxf(q2 * mn2, q2 * mx2);
  sum += fmaxf(q3 * mn3, q3 * mx3);
  return sum;
}

__global__ void quest_chunk_score_kernel_bf16_d128(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ chunk_min,
    const __nv_bfloat16* __restrict__ chunk_max, const bool* __restrict__ valid,
    float* __restrict__ scores, int chunk_n_max, int valid_n, int scores_n) {
  const int chunk_idx = blockIdx.x;
  const int bh_idx = blockIdx.y;
  const int lane = threadIdx.x;
  const int out_idx = bh_idx * scores_n + chunk_idx;

  if (valid != nullptr) {
    const bool in_range = chunk_idx < valid_n;
    const bool is_valid = in_range && valid[bh_idx * valid_n + chunk_idx];
    if (!is_valid) {
      if (lane == 0) {
        scores[out_idx] = -INFINITY;
      }
      return;
    }
  }

  const int d_base = lane * 4;
  const __nv_bfloat16* q_ptr = q + bh_idx * HEAD_DIM_FAST;
  const __nv_bfloat16* min_ptr =
      chunk_min + (bh_idx * chunk_n_max + chunk_idx) * HEAD_DIM_FAST;
  const __nv_bfloat16* max_ptr =
      chunk_max + (bh_idx * chunk_n_max + chunk_idx) * HEAD_DIM_FAST;

  const uint2 q_u2 = *reinterpret_cast<const uint2*>(q_ptr + d_base);
  const uint2 min_u2 = *reinterpret_cast<const uint2*>(min_ptr + d_base);
  const uint2 max_u2 = *reinterpret_cast<const uint2*>(max_ptr + d_base);
  float sum = warp_reduce_sum(score_bf16x4(q_u2, min_u2, max_u2));

  if (lane == 0) {
    scores[out_idx] = sum;
  }
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void quest_chunk_score_kernel_generic(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ chunk_min,
    const scalar_t* __restrict__ chunk_max, const bool* __restrict__ valid,
    float* __restrict__ scores, int chunk_n_max, int valid_n, int scores_n,
    int head_dim) {
  const int chunk_idx = blockIdx.x;
  const int bh_idx = blockIdx.y;
  const int tid = threadIdx.x;
  const int out_idx = bh_idx * scores_n + chunk_idx;

  if (valid != nullptr) {
    const bool in_range = chunk_idx < valid_n;
    const bool is_valid = in_range && valid[bh_idx * valid_n + chunk_idx];
    if (!is_valid) {
      if (tid == 0) {
        scores[out_idx] = -INFINITY;
      }
      return;
    }
  }

  const scalar_t* q_ptr = q + static_cast<int64_t>(bh_idx) * head_dim;
  const scalar_t* min_ptr =
      chunk_min +
      (static_cast<int64_t>(bh_idx) * chunk_n_max + chunk_idx) * head_dim;
  const scalar_t* max_ptr =
      chunk_max +
      (static_cast<int64_t>(bh_idx) * chunk_n_max + chunk_idx) * head_dim;

  float sum = 0.0f;
  for (int d = tid; d < head_dim; d += BLOCK_THREADS) {
    const float qv = static_cast<float>(q_ptr[d]);
    const float mn = static_cast<float>(min_ptr[d]);
    const float mx = static_cast<float>(max_ptr[d]);
    sum += fmaxf(qv * mn, qv * mx);
  }
  sum = block_reduce_sum<BLOCK_THREADS>(sum);
  if (tid == 0) {
    scores[out_idx] = sum;
  }
}

__global__ void quest_sub_chunk_score_kernel_bf16_d128(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ chunk_min,
    const __nv_bfloat16* __restrict__ chunk_max,
    const int64_t* __restrict__ large_ids, float* __restrict__ scores,
    int chunk_n_max, int n_selected, int factor, int scores_n) {
  const int sel_idx = blockIdx.x;
  const int bh_idx = blockIdx.y;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;

  __shared__ __nv_bfloat16 q_smem[HEAD_DIM_FAST];
  __shared__ int64_t large_id_smem;

  if (warp == 0) {
    const int d_base = lane * 4;
    *reinterpret_cast<uint2*>(q_smem + d_base) =
        *reinterpret_cast<const uint2*>(
            q + static_cast<int64_t>(bh_idx) * HEAD_DIM_FAST + d_base);
    if (lane == 0) {
      large_id_smem = large_ids[bh_idx * n_selected + sel_idx];
    }
  }
  __syncthreads();

  const int sub_offset = warp;
  const int chunk_idx = static_cast<int>(large_id_smem) * factor + sub_offset;
  const int out_idx = bh_idx * scores_n + sel_idx * factor + sub_offset;

  if (chunk_idx < 0 || chunk_idx >= chunk_n_max) {
    if (lane == 0) {
      scores[out_idx] = -INFINITY;
    }
    return;
  }

  const int d_base = lane * 4;
  const uint2 q_u2 = *reinterpret_cast<const uint2*>(q_smem + d_base);
  const __nv_bfloat16* min_ptr =
      chunk_min + (bh_idx * chunk_n_max + chunk_idx) * HEAD_DIM_FAST;
  const __nv_bfloat16* max_ptr =
      chunk_max + (bh_idx * chunk_n_max + chunk_idx) * HEAD_DIM_FAST;
  const uint2 min_u2 = *reinterpret_cast<const uint2*>(min_ptr + d_base);
  const uint2 max_u2 = *reinterpret_cast<const uint2*>(max_ptr + d_base);
  float sum = warp_reduce_sum(score_bf16x4(q_u2, min_u2, max_u2));

  if (lane == 0) {
    scores[out_idx] = sum;
  }
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void quest_sub_chunk_score_kernel_generic(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ chunk_min,
    const scalar_t* __restrict__ chunk_max,
    const int64_t* __restrict__ large_ids, float* __restrict__ scores,
    int chunk_n_max, int n_selected, int factor, int scores_n, int head_dim) {
  const int pos = blockIdx.x;
  const int bh_idx = blockIdx.y;
  const int tid = threadIdx.x;
  const int large_pos = pos / factor;
  const int sub_offset = pos - large_pos * factor;
  const int64_t large_id = large_ids[bh_idx * n_selected + large_pos];
  const int chunk_idx = static_cast<int>(large_id) * factor + sub_offset;
  const int out_idx = bh_idx * scores_n + pos;

  if (chunk_idx < 0 || chunk_idx >= chunk_n_max) {
    if (tid == 0) {
      scores[out_idx] = -INFINITY;
    }
    return;
  }

  const scalar_t* q_ptr = q + static_cast<int64_t>(bh_idx) * head_dim;
  const scalar_t* min_ptr =
      chunk_min +
      (static_cast<int64_t>(bh_idx) * chunk_n_max + chunk_idx) * head_dim;
  const scalar_t* max_ptr =
      chunk_max +
      (static_cast<int64_t>(bh_idx) * chunk_n_max + chunk_idx) * head_dim;

  float sum = 0.0f;
  for (int d = tid; d < head_dim; d += BLOCK_THREADS) {
    const float qv = static_cast<float>(q_ptr[d]);
    const float mn = static_cast<float>(min_ptr[d]);
    const float mx = static_cast<float>(max_ptr[d]);
    sum += fmaxf(qv * mn, qv * mx);
  }
  sum = block_reduce_sum<BLOCK_THREADS>(sum);
  if (tid == 0) {
    scores[out_idx] = sum;
  }
}

__global__ void quest_map_back_kernel(const int64_t* __restrict__ large_ids,
                                      const int64_t* __restrict__ sub_topk_pos,
                                      int64_t* __restrict__ chunk_idx,
                                      int nk_large, int nk_small, int factor,
                                      int n_chunks) {
  const int bh_idx = blockIdx.y;
  const int local_k = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_k >= nk_small) {
    return;
  }

  const int row_off = bh_idx * nk_small;
  const int64_t sub_pos = sub_topk_pos[row_off + local_k];
  const int64_t sub_total = static_cast<int64_t>(nk_large) * factor;
  if (sub_pos < 0 || sub_pos >= sub_total) {
    chunk_idx[row_off + local_k] = -1;
    return;
  }

  int large_sel = static_cast<int>(sub_pos / factor);
  if (large_sel < 0) {
    large_sel = 0;
  } else if (large_sel >= nk_large) {
    large_sel = nk_large - 1;
  }
  const int sub_off =
      static_cast<int>(sub_pos - static_cast<int64_t>(large_sel) * factor);
  const int64_t large_id = large_ids[bh_idx * nk_large + large_sel];
  if (large_id < 0) {
    chunk_idx[row_off + local_k] = -1;
    return;
  }

  int64_t out_chunk = large_id * factor + sub_off;
  if (out_chunk < 0) {
    out_chunk = 0;
  } else if (out_chunk >= n_chunks) {
    out_chunk = n_chunks - 1;
  }
  chunk_idx[row_off + local_k] = out_chunk;
}

void check_quest_score_tensors(const at::Tensor& q, const at::Tensor& chunk_min,
                               const at::Tensor& chunk_max,
                               const at::Tensor& scores, int64_t n_chunks) {
  TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
  TORCH_CHECK(chunk_min.is_cuda(), "chunk_min must be on CUDA");
  TORCH_CHECK(chunk_max.is_cuda(), "chunk_max must be on CUDA");
  TORCH_CHECK(scores.is_cuda(), "scores must be on CUDA");
  TORCH_CHECK(q.device() == chunk_min.device() &&
                  q.device() == chunk_max.device() &&
                  q.device() == scores.device(),
              "Quest tensors must be on the same CUDA device");
  TORCH_CHECK(q.dim() == 3, "q must be 3D [bs, kv_heads, head_dim]");
  TORCH_CHECK(chunk_min.dim() == 4 && chunk_max.dim() == 4,
              "chunk_min/max must be 4D [bs, kv_heads, n_chunks, head_dim]");
  TORCH_CHECK(scores.dim() == 3,
              "scores must be 3D [bs, kv_heads, score_slots]");
  TORCH_CHECK(q.scalar_type() == chunk_min.scalar_type() &&
                  q.scalar_type() == chunk_max.scalar_type(),
              "q and chunk_min/max dtype must match");
  TORCH_CHECK(scores.scalar_type() == at::ScalarType::Float,
              "scores must be float32");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(chunk_min.is_contiguous(), "chunk_min must be contiguous");
  TORCH_CHECK(chunk_max.is_contiguous(), "chunk_max must be contiguous");
  TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
  TORCH_CHECK(q.size(0) == chunk_min.size(0) &&
                  q.size(0) == chunk_max.size(0) && q.size(0) == scores.size(0),
              "Quest tensors must share batch size");
  TORCH_CHECK(q.size(1) == chunk_min.size(1) &&
                  q.size(1) == chunk_max.size(1) && q.size(1) == scores.size(1),
              "Quest tensors must share kv_heads");
  TORCH_CHECK(q.size(2) == chunk_min.size(3) && q.size(2) == chunk_max.size(3),
              "Quest tensors must share head_dim");
  TORCH_CHECK(n_chunks >= 0, "n_chunks must be non-negative");
  TORCH_CHECK(n_chunks <= chunk_min.size(2) && n_chunks <= chunk_max.size(2),
              "n_chunks must be <= chunk_min/max chunk dimension");
  TORCH_CHECK(scores.size(2) >= n_chunks,
              "scores has fewer slots than n_chunks");
}

}  // namespace

void quest_chunk_score_cuda(at::Tensor q, at::Tensor chunk_min,
                            at::Tensor chunk_max, at::Tensor scores,
                            int64_t n_chunks, std::optional<at::Tensor> valid) {
  check_quest_score_tensors(q, chunk_min, chunk_max, scores, n_chunks);

  const int64_t bs = q.size(0);
  const int64_t kv_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t chunk_n_max = chunk_min.size(2);
  const int64_t scores_n = scores.size(2);
  const int64_t bh = bs * kv_heads;
  if (bh == 0 || n_chunks == 0) {
    return;
  }
  TORCH_CHECK(bh <= std::numeric_limits<int>::max() &&
                  n_chunks <= std::numeric_limits<int>::max() &&
                  chunk_n_max <= std::numeric_limits<int>::max() &&
                  scores_n <= std::numeric_limits<int>::max() &&
                  head_dim <= std::numeric_limits<int>::max(),
              "quest_chunk_score dimensions exceed int32 range");

  const bool* valid_ptr = nullptr;
  int64_t valid_n = 0;
  if (valid.has_value()) {
    const at::Tensor& v = valid.value();
    TORCH_CHECK(v.is_cuda(), "chunk_valid must be on CUDA");
    TORCH_CHECK(v.device() == q.device(),
                "chunk_valid must be on the same CUDA device as q");
    TORCH_CHECK(v.scalar_type() == at::ScalarType::Bool,
                "chunk_valid must be bool");
    TORCH_CHECK(v.dim() == 3, "chunk_valid must be 3D [bs, kv_heads, n]");
    TORCH_CHECK(v.size(0) == bs && v.size(1) == kv_heads,
                "chunk_valid must share [bs, kv_heads] with q");
    TORCH_CHECK(v.size(2) >= n_chunks,
                "chunk_valid has fewer slots than n_chunks");
    TORCH_CHECK(v.is_contiguous(), "chunk_valid must be contiguous");
    valid_ptr = v.data_ptr<bool>();
    valid_n = v.size(2);
  }

  c10::cuda::CUDAGuard device_guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(static_cast<unsigned>(n_chunks), static_cast<unsigned>(bh));

  if (q.scalar_type() == at::ScalarType::BFloat16 &&
      head_dim == HEAD_DIM_FAST) {
    quest_chunk_score_kernel_bf16_d128<<<grid, WARP_SIZE, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(chunk_min.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(chunk_max.data_ptr()), valid_ptr,
        scores.data_ptr<float>(), static_cast<int>(chunk_n_max),
        static_cast<int>(valid_n), static_cast<int>(scores_n));
  } else {
    const dim3 block(GENERIC_BLOCK_THREADS);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
        "zoomkv_quest_chunk_score", [&] {
          quest_chunk_score_kernel_generic<scalar_t, GENERIC_BLOCK_THREADS>
              <<<grid, block, 0, stream>>>(
                  q.data_ptr<scalar_t>(), chunk_min.data_ptr<scalar_t>(),
                  chunk_max.data_ptr<scalar_t>(), valid_ptr,
                  scores.data_ptr<float>(), static_cast<int>(chunk_n_max),
                  static_cast<int>(valid_n), static_cast<int>(scores_n),
                  static_cast<int>(head_dim));
        });
  }

  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "quest_chunk_score kernel failed: ", cudaGetErrorString(error));
}

void quest_sub_chunk_score_cuda(at::Tensor q, at::Tensor chunk_min,
                                at::Tensor chunk_max, at::Tensor large_ids,
                                at::Tensor scores, int64_t n_selected,
                                int64_t factor) {
  check_quest_score_tensors(q, chunk_min, chunk_max, scores, 0);
  TORCH_CHECK(large_ids.is_cuda(), "large_ids must be on CUDA");
  TORCH_CHECK(large_ids.device() == q.device(),
              "large_ids must be on the same CUDA device as q");
  TORCH_CHECK(large_ids.scalar_type() == at::ScalarType::Long,
              "large_ids must be int64");
  TORCH_CHECK(large_ids.dim() == 3,
              "large_ids must be 3D [bs, kv_heads, n_selected]");
  TORCH_CHECK(large_ids.size(0) == q.size(0) && large_ids.size(1) == q.size(1),
              "large_ids must share [bs, kv_heads] with q");
  TORCH_CHECK(large_ids.size(2) >= n_selected,
              "large_ids has fewer slots than n_selected");
  TORCH_CHECK(large_ids.is_contiguous(), "large_ids must be contiguous");
  TORCH_CHECK(n_selected > 0, "n_selected must be > 0");
  TORCH_CHECK(factor > 0, "factor must be > 0");
  const int64_t needed_scores = n_selected * factor;
  TORCH_CHECK(scores.size(2) >= needed_scores,
              "scores has fewer slots than n_selected * factor");

  const int64_t bs = q.size(0);
  const int64_t kv_heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t chunk_n_max = chunk_min.size(2);
  const int64_t scores_n = scores.size(2);
  const int64_t bh = bs * kv_heads;
  TORCH_CHECK(bh <= std::numeric_limits<int>::max() &&
                  n_selected <= std::numeric_limits<int>::max() &&
                  factor <= std::numeric_limits<int>::max() &&
                  needed_scores <= std::numeric_limits<int>::max() &&
                  chunk_n_max <= std::numeric_limits<int>::max() &&
                  scores_n <= std::numeric_limits<int>::max() &&
                  head_dim <= std::numeric_limits<int>::max(),
              "quest_sub_chunk_score dimensions exceed int32 range");

  c10::cuda::CUDAGuard device_guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (q.scalar_type() == at::ScalarType::BFloat16 &&
      head_dim == HEAD_DIM_FAST && factor <= 32) {
    const dim3 grid(static_cast<unsigned>(n_selected),
                    static_cast<unsigned>(bh));
    const dim3 block(WARP_SIZE, static_cast<unsigned>(factor));
    quest_sub_chunk_score_kernel_bf16_d128<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(chunk_min.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(chunk_max.data_ptr()),
        large_ids.data_ptr<int64_t>(), scores.data_ptr<float>(),
        static_cast<int>(chunk_n_max), static_cast<int>(n_selected),
        static_cast<int>(factor), static_cast<int>(scores_n));
  } else {
    const dim3 grid(static_cast<unsigned>(needed_scores),
                    static_cast<unsigned>(bh));
    const dim3 block(GENERIC_BLOCK_THREADS);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
        "zoomkv_quest_sub_chunk_score", [&] {
          quest_sub_chunk_score_kernel_generic<scalar_t, GENERIC_BLOCK_THREADS>
              <<<grid, block, 0, stream>>>(
                  q.data_ptr<scalar_t>(), chunk_min.data_ptr<scalar_t>(),
                  chunk_max.data_ptr<scalar_t>(), large_ids.data_ptr<int64_t>(),
                  scores.data_ptr<float>(), static_cast<int>(chunk_n_max),
                  static_cast<int>(n_selected), static_cast<int>(factor),
                  static_cast<int>(scores_n), static_cast<int>(head_dim));
        });
  }

  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "quest_sub_chunk_score kernel failed: ",
              cudaGetErrorString(error));
}

void quest_map_back_cuda(at::Tensor large_ids, at::Tensor sub_topk_pos,
                         at::Tensor chunk_idx, int64_t factor,
                         int64_t n_chunks) {
  TORCH_CHECK(large_ids.is_cuda(), "large_ids must be on CUDA");
  TORCH_CHECK(sub_topk_pos.is_cuda(), "sub_topk_pos must be on CUDA");
  TORCH_CHECK(chunk_idx.is_cuda(), "chunk_idx must be on CUDA");
  TORCH_CHECK(large_ids.device() == sub_topk_pos.device() &&
                  large_ids.device() == chunk_idx.device(),
              "Quest map tensors must be on the same CUDA device");
  TORCH_CHECK(large_ids.is_contiguous(), "large_ids must be contiguous");
  TORCH_CHECK(sub_topk_pos.is_contiguous(), "sub_topk_pos must be contiguous");
  TORCH_CHECK(chunk_idx.is_contiguous(), "chunk_idx must be contiguous");
  TORCH_CHECK(large_ids.scalar_type() == at::ScalarType::Long,
              "large_ids must be int64");
  TORCH_CHECK(sub_topk_pos.scalar_type() == at::ScalarType::Long,
              "sub_topk_pos must be int64");
  TORCH_CHECK(chunk_idx.scalar_type() == at::ScalarType::Long,
              "chunk_idx must be int64");
  TORCH_CHECK(large_ids.dim() == 3,
              "large_ids must be 3D [bs, kv_heads, nk_large]");
  TORCH_CHECK(sub_topk_pos.dim() == 3,
              "sub_topk_pos must be 3D [bs, kv_heads, nk_small]");
  TORCH_CHECK(chunk_idx.sizes() == sub_topk_pos.sizes(),
              "chunk_idx shape must match sub_topk_pos");
  TORCH_CHECK(large_ids.size(0) == sub_topk_pos.size(0) &&
                  large_ids.size(1) == sub_topk_pos.size(1),
              "large_ids and sub_topk_pos must share [bs, kv_heads]");
  TORCH_CHECK(factor > 0, "factor must be > 0");
  TORCH_CHECK(n_chunks > 0, "n_chunks must be > 0");

  const int64_t nk_large = large_ids.size(2);
  const int64_t nk_small = sub_topk_pos.size(2);
  const int64_t bh = large_ids.size(0) * large_ids.size(1);
  if (bh == 0 || nk_small == 0) {
    return;
  }
  TORCH_CHECK(nk_large > 0, "nk_large must be > 0");
  TORCH_CHECK(bh <= std::numeric_limits<int>::max() &&
                  nk_large <= std::numeric_limits<int>::max() &&
                  nk_small <= std::numeric_limits<int>::max() &&
                  factor <= std::numeric_limits<int>::max() &&
                  n_chunks <= std::numeric_limits<int>::max(),
              "quest_map_back dimensions exceed int32 range");

  c10::cuda::CUDAGuard device_guard(large_ids.device());
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  constexpr int BLOCK = 128;
  const dim3 grid(static_cast<unsigned>((nk_small + BLOCK - 1) / BLOCK),
                  static_cast<unsigned>(bh));
  const dim3 block(BLOCK);
  quest_map_back_kernel<<<grid, block, 0, stream>>>(
      large_ids.data_ptr<int64_t>(), sub_topk_pos.data_ptr<int64_t>(),
      chunk_idx.data_ptr<int64_t>(), static_cast<int>(nk_large),
      static_cast<int>(nk_small), static_cast<int>(factor),
      static_cast<int>(n_chunks));

  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess,
              "quest_map_back kernel failed: ", cudaGetErrorString(error));
}

#ifndef ZOOMKV_UNIFIED_EXTENSION
namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quest_chunk_score", &quest_chunk_score_cuda,
        "Quest chunk upper-bound score (CUDA)", py::arg("q"),
        py::arg("chunk_min"), py::arg("chunk_max"), py::arg("scores"),
        py::arg("n_chunks"), py::arg("chunk_valid") = std::nullopt);
  m.def("quest_sub_chunk_score", &quest_sub_chunk_score_cuda,
        "Quest sub-chunk upper-bound score (CUDA)");
  m.def("quest_map_back", &quest_map_back_cuda,
        "Map hierarchical Quest sub-chunk positions back to chunk ids (CUDA)");
}
#endif
