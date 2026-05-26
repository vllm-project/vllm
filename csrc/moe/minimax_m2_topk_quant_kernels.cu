#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cuda_fp8.h>

#include <cmath>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"

namespace vllm {
namespace moe {

namespace {

constexpr int kMiniMaxExperts = 256;
constexpr int kMiniMaxTopK = 8;
constexpr int kMiniMaxBlockK = 128;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / WARP_SIZE;
constexpr float kFp8E4M3Min = -448.0f;
constexpr float kFp8E4M3Max = 448.0f;
constexpr float kQuantEps = 1.0e-10f;

__device__ __forceinline__ float sigmoidf_stable(float x) {
  float y = 1.0f / (1.0f + expf(-x));
  return y == y ? y : 0.0f;
}

__device__ __forceinline__ float warp_reduce_max(float val, uint32_t mask) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(mask, val, offset));
  }
  return val;
}

__device__ __forceinline__ void warp_argmax(float& score, int& id,
                                            float& weight, uint32_t mask) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    float other_score = __shfl_down_sync(mask, score, offset);
    int other_id = __shfl_down_sync(mask, id, offset);
    float other_weight = __shfl_down_sync(mask, weight, offset);
    if (other_score > score ||
        (other_score == score && other_id >= 0 &&
         (id < 0 || other_id < id))) {
      score = other_score;
      id = other_id;
      weight = other_weight;
    }
  }
}

__device__ __forceinline__ float block_reduce_max(float val,
                                                  float* __restrict__ smem) {
  const int lane_id = threadIdx.x & (WARP_SIZE - 1);
  const int warp_id = threadIdx.x / WARP_SIZE;
  const uint32_t full_mask = 0xffffffffu;

  val = warp_reduce_max(val, full_mask);
  if (lane_id == 0) {
    smem[warp_id] = val;
  }
  __syncthreads();

  if (warp_id == 0) {
    val = lane_id < kWarps ? smem[lane_id] : 0.0f;
    val = warp_reduce_max(val, full_mask);
    if (lane_id == 0) {
      smem[0] = val;
    }
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ void block_argmax(float& score, int& id,
                                             float& weight,
                                             float* __restrict__ smem_scores,
                                             int* __restrict__ smem_ids,
                                             float* __restrict__ smem_weights) {
  const int lane_id = threadIdx.x & (WARP_SIZE - 1);
  const int warp_id = threadIdx.x / WARP_SIZE;
  const uint32_t full_mask = 0xffffffffu;

  warp_argmax(score, id, weight, full_mask);
  if (lane_id == 0) {
    smem_scores[warp_id] = score;
    smem_ids[warp_id] = id;
    smem_weights[warp_id] = weight;
  }
  __syncthreads();

  if (warp_id == 0) {
    score = lane_id < kWarps ? smem_scores[lane_id] : -INFINITY;
    id = lane_id < kWarps ? smem_ids[lane_id] : -1;
    weight = lane_id < kWarps ? smem_weights[lane_id] : 0.0f;
    warp_argmax(score, id, weight, full_mask);
    if (lane_id == 0) {
      smem_scores[0] = score;
      smem_ids[0] = id;
      smem_weights[0] = weight;
    }
  }
  __syncthreads();
  score = smem_scores[0];
  id = smem_ids[0];
  weight = smem_weights[0];
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads)
void minimax_m2_topk_sigmoid_quant_kernel(
    const scalar_t* __restrict__ hidden_states,
    const float* __restrict__ router_logits,
    const float* __restrict__ e_score_correction_bias,
    float* __restrict__ topk_weights, int32_t* __restrict__ topk_ids,
    __nv_fp8_e4m3* __restrict__ a1q, float* __restrict__ a1q_scale,
    int64_t hidden_stride_m, int64_t logits_stride_m, int64_t a1q_stride_m,
    int64_t topk_weights_stride_m, int64_t topk_ids_stride_m,
    int64_t a1q_scale_stride_m, int hidden_size) {
  const int token_id = blockIdx.x;
  const int group_id = blockIdx.y;
  const int tid = threadIdx.x;

  __shared__ float smem_scores[kWarps];
  __shared__ float smem_weights[kWarps];
  __shared__ int smem_ids[kWarps];

  const int hidden_offset = group_id * kMiniMaxBlockK + tid;
  float hidden_val = 0.0f;
  if (hidden_offset < hidden_size) {
    hidden_val = static_cast<float>(
        hidden_states[token_id * hidden_stride_m + hidden_offset]);
  }

  const float absmax =
      block_reduce_max(fmaxf(fabsf(hidden_val), kQuantEps), smem_scores);
  const float scale = absmax / kFp8E4M3Max;

  if (tid == 0) {
    a1q_scale[token_id * a1q_scale_stride_m + group_id] = scale;
  }
  if (hidden_offset < hidden_size) {
    const float q = fminf(fmaxf(hidden_val / scale, kFp8E4M3Min),
                          kFp8E4M3Max);
    a1q[token_id * a1q_stride_m + hidden_offset] = __nv_fp8_e4m3(q);
  }

  if (group_id != 0) {
    return;
  }

  float scores_per_thread[2];
  float weights_per_thread[2];
  int expert_ids_per_thread[2];

#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int expert_id = tid + i * kThreads;
    const float weight = sigmoidf_stable(
        router_logits[token_id * logits_stride_m + expert_id]);
    weights_per_thread[i] = weight;
    scores_per_thread[i] = weight + e_score_correction_bias[expert_id];
    expert_ids_per_thread[i] = expert_id;
  }

  float selected_weights[kMiniMaxTopK];
  float selected_sum = 0.0f;

#pragma unroll
  for (int k_idx = 0; k_idx < kMiniMaxTopK; ++k_idx) {
    float best_score = -INFINITY;
    float best_weight = 0.0f;
    int best_id = -1;

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int expert_id = expert_ids_per_thread[i];
      const float score = scores_per_thread[i];
      if (score > best_score ||
          (score == best_score && expert_id >= 0 &&
           (best_id < 0 || expert_id < best_id))) {
        best_score = score;
        best_weight = weights_per_thread[i];
        best_id = expert_id;
      }
    }

    block_argmax(best_score, best_id, best_weight, smem_scores, smem_ids,
                 smem_weights);
    const int selected_id = best_id;
    const float selected_weight = best_weight;

    if (tid == 0) {
      selected_weights[k_idx] = selected_weight;
      selected_sum += selected_weight;
      topk_ids[token_id * topk_ids_stride_m + k_idx] = selected_id;
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if (expert_ids_per_thread[i] == selected_id) {
        scores_per_thread[i] = -INFINITY;
      }
    }
  }

  if (tid == 0) {
    const float inv_sum = selected_sum > 0.0f ? 1.0f / selected_sum : 1.0f;
#pragma unroll
    for (int k_idx = 0; k_idx < kMiniMaxTopK; ++k_idx) {
      topk_weights[token_id * topk_weights_stride_m + k_idx] =
          selected_weights[k_idx] * inv_sum;
    }
  }
}

}  // namespace

}  // namespace moe
}  // namespace vllm

void minimax_m2_topk_sigmoid_quant(
    const torch::Tensor& hidden_states, const torch::Tensor& router_logits,
    const torch::Tensor& e_score_correction_bias, torch::Tensor& topk_weights,
    torch::Tensor& topk_ids, torch::Tensor& a1q, torch::Tensor& a1q_scale,
    int64_t top_k, int64_t block_k) {
  TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
  TORCH_CHECK(router_logits.is_cuda(), "router_logits must be a CUDA tensor");
  TORCH_CHECK(e_score_correction_bias.is_cuda(),
              "e_score_correction_bias must be a CUDA tensor");
  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2D");
  TORCH_CHECK(router_logits.dim() == 2, "router_logits must be 2D");
  TORCH_CHECK(hidden_states.size(0) == router_logits.size(0),
              "hidden_states and router_logits must have the same M");
  TORCH_CHECK(router_logits.size(1) == vllm::moe::kMiniMaxExperts,
              "CUDA MiniMax topk+quant only supports 256 experts");
  TORCH_CHECK(top_k == vllm::moe::kMiniMaxTopK,
              "CUDA MiniMax topk+quant only supports top_k=8");
  TORCH_CHECK(block_k == vllm::moe::kMiniMaxBlockK,
              "CUDA MiniMax topk+quant only supports block_k=128");
  TORCH_CHECK(hidden_states.stride(1) == 1,
              "hidden_states last dimension must be contiguous");
  TORCH_CHECK(router_logits.stride(1) == 1,
              "router_logits last dimension must be contiguous");
  TORCH_CHECK(hidden_states.scalar_type() == at::ScalarType::BFloat16,
              "hidden_states must be bfloat16");
  TORCH_CHECK(router_logits.scalar_type() == at::ScalarType::Float,
              "router_logits must be float32");
  TORCH_CHECK(e_score_correction_bias.scalar_type() == at::ScalarType::Float,
              "e_score_correction_bias must be float32");
  TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float,
              "topk_weights must be float32");
  TORCH_CHECK(topk_ids.scalar_type() == at::ScalarType::Int,
              "topk_ids must be int32");
  TORCH_CHECK(a1q.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "a1q must be float8_e4m3fn");
  TORCH_CHECK(a1q_scale.scalar_type() == at::ScalarType::Float,
              "a1q_scale must be float32");

  const int64_t num_tokens = hidden_states.size(0);
  const int64_t hidden_size = hidden_states.size(1);
  const int64_t num_groups =
      (hidden_size + vllm::moe::kMiniMaxBlockK - 1) /
      vllm::moe::kMiniMaxBlockK;

  TORCH_CHECK(e_score_correction_bias.numel() == vllm::moe::kMiniMaxExperts,
              "e_score_correction_bias must have 256 elements");
  TORCH_CHECK(topk_weights.sizes() == torch::IntArrayRef({num_tokens, top_k}),
              "topk_weights has an unexpected shape");
  TORCH_CHECK(topk_ids.sizes() == torch::IntArrayRef({num_tokens, top_k}),
              "topk_ids has an unexpected shape");
  TORCH_CHECK(a1q.sizes() == hidden_states.sizes(),
              "a1q must have the same shape as hidden_states");
  TORCH_CHECK(a1q_scale.sizes() ==
                  torch::IntArrayRef({num_tokens, num_groups}),
              "a1q_scale has an unexpected shape");

  if (num_tokens == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(hidden_states));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned int>(num_tokens),
                  static_cast<unsigned int>(num_groups));

  VLLM_DISPATCH_HALF_TYPES(
      hidden_states.scalar_type(), "minimax_m2_topk_sigmoid_quant", ([&] {
        vllm::moe::minimax_m2_topk_sigmoid_quant_kernel<scalar_t>
            <<<grid, vllm::moe::kThreads, 0, stream>>>(
                static_cast<const scalar_t*>(hidden_states.data_ptr()),
                static_cast<const float*>(router_logits.data_ptr()),
                static_cast<const float*>(e_score_correction_bias.data_ptr()),
                static_cast<float*>(topk_weights.data_ptr()),
                static_cast<int32_t*>(topk_ids.data_ptr()),
                static_cast<__nv_fp8_e4m3*>(a1q.data_ptr()),
                static_cast<float*>(a1q_scale.data_ptr()),
                hidden_states.stride(0), router_logits.stride(0),
                a1q.stride(0), topk_weights.stride(0), topk_ids.stride(0),
                a1q_scale.stride(0), static_cast<int>(hidden_size));
      }));
}
