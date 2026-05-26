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
constexpr int kThreads = 256;
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
    int64_t a1q_scale_stride_m, int hidden_size, int num_groups) {
  const int token_id = blockIdx.x;
  const int lane_id = threadIdx.x & (WARP_SIZE - 1);
  const int warp_id = threadIdx.x / WARP_SIZE;
  const uint32_t full_mask = 0xffffffffu;

  if (warp_id == 0) {
    float scores_per_lane[kMiniMaxTopK];
    float weights_per_lane[kMiniMaxTopK];
    int expert_ids_per_lane[kMiniMaxTopK];

#pragma unroll
    for (int i = 0; i < kMiniMaxTopK; ++i) {
      const int expert_id = lane_id + i * WARP_SIZE;
      float weight = sigmoidf_stable(
          router_logits[token_id * logits_stride_m + expert_id]);
      weights_per_lane[i] = weight;
      scores_per_lane[i] =
          weight + e_score_correction_bias[expert_id];
      expert_ids_per_lane[i] = expert_id;
    }

    float selected_weights[kMiniMaxTopK];
    float selected_sum = 0.0f;

#pragma unroll
    for (int k_idx = 0; k_idx < kMiniMaxTopK; ++k_idx) {
      float best_score = -INFINITY;
      float best_weight = 0.0f;
      int best_id = -1;

#pragma unroll
      for (int i = 0; i < kMiniMaxTopK; ++i) {
        const int expert_id = expert_ids_per_lane[i];
        const float score = scores_per_lane[i];
        if (score > best_score ||
            (score == best_score && expert_id >= 0 &&
             (best_id < 0 || expert_id < best_id))) {
          best_score = score;
          best_weight = weights_per_lane[i];
          best_id = expert_id;
        }
      }

      warp_argmax(best_score, best_id, best_weight, full_mask);
      const int selected_id = __shfl_sync(full_mask, best_id, 0);
      const float selected_weight = __shfl_sync(full_mask, best_weight, 0);

      if (lane_id == 0) {
        selected_weights[k_idx] = selected_weight;
        selected_sum += selected_weight;
        topk_ids[token_id * topk_ids_stride_m + k_idx] = selected_id;
      }

#pragma unroll
      for (int i = 0; i < kMiniMaxTopK; ++i) {
        if (expert_ids_per_lane[i] == selected_id) {
          scores_per_lane[i] = -INFINITY;
        }
      }
    }

    if (lane_id == 0) {
      const float inv_sum = selected_sum > 0.0f ? 1.0f / selected_sum : 1.0f;
#pragma unroll
      for (int k_idx = 0; k_idx < kMiniMaxTopK; ++k_idx) {
        topk_weights[token_id * topk_weights_stride_m + k_idx] =
            selected_weights[k_idx] * inv_sum;
      }
    }
  }

  for (int group_id = warp_id; group_id < num_groups; group_id += 8) {
    const int hidden_base = group_id * kMiniMaxBlockK + lane_id * 4;
    float vals[4];
    float local_absmax = kQuantEps;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int hidden_offset = hidden_base + i;
      float v = 0.0f;
      if (hidden_offset < hidden_size) {
        v = static_cast<float>(
            hidden_states[token_id * hidden_stride_m + hidden_offset]);
      }
      vals[i] = v;
      local_absmax = fmaxf(local_absmax, fabsf(v));
    }

    const float absmax = __shfl_sync(full_mask,
                                     warp_reduce_max(local_absmax, full_mask),
                                     0);
    const float scale = absmax / kFp8E4M3Max;

    if (lane_id == 0) {
      a1q_scale[token_id * a1q_scale_stride_m + group_id] = scale;
    }


#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int hidden_offset = hidden_base + i;
      if (hidden_offset < hidden_size) {
        float q = fminf(fmaxf(vals[i] / scale, kFp8E4M3Min),
                        kFp8E4M3Max);
        a1q[token_id * a1q_stride_m + hidden_offset] = __nv_fp8_e4m3(q);
      }
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

  VLLM_DISPATCH_HALF_TYPES(
      hidden_states.scalar_type(), "minimax_m2_topk_sigmoid_quant", ([&] {
        vllm::moe::minimax_m2_topk_sigmoid_quant_kernel<scalar_t>
            <<<num_tokens, vllm::moe::kThreads, 0, stream>>>(
                static_cast<const scalar_t*>(hidden_states.data_ptr()),
                static_cast<const float*>(router_logits.data_ptr()),
                static_cast<const float*>(e_score_correction_bias.data_ptr()),
                static_cast<float*>(topk_weights.data_ptr()),
                static_cast<int32_t*>(topk_ids.data_ptr()),
                static_cast<__nv_fp8_e4m3*>(a1q.data_ptr()),
                static_cast<float*>(a1q_scale.data_ptr()),
                hidden_states.stride(0), router_logits.stride(0),
                a1q.stride(0), topk_weights.stride(0), topk_ids.stride(0),
                a1q_scale.stride(0), static_cast<int>(hidden_size),
                static_cast<int>(num_groups));
      }));
}
