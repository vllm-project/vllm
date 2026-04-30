// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Gemma4 MoE Routing Kernel
//
// Warp-cooperative routing for Gemma4 MoE: softmax -> top-K -> renormalize
// -> per_expert_scale. Each warp independently processes one token.
//
// Gemma4 routing differs from standard MoE routing:
//   1. softmax over ALL E experts (not just selected ones)
//   2. top-K selection
//   3. renormalize: divide selected weights by their sum
//   4. multiply by per_expert_scale[expert_id]
//
// This CUDA implementation matches gemma4_routing_function_torch() from
// vllm/model_executor/models/gemma4.py.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARP_SIZE = 32;
constexpr int MAX_TOP_K = 8;

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

// ---------------------------------------------------------------------------
// Warp-cooperative top-K selection with softmax + renormalize + scale
//
// Each warp (32 threads) cooperatively selects top-K from E experts.
// For E=128, each thread handles 4 elements. K rounds of warp-wide argmax
// find the top-K indices, then weights are renormalized and scaled.
// ---------------------------------------------------------------------------
__device__ void select_topk_warp(const float* __restrict__ logits,
                                 float* __restrict__ out_weights,
                                 int* __restrict__ out_ids,
                                 const float* __restrict__ per_expert_scale,
                                 int E, int K, int lane) {
  const int elems_per_thread = (E + WARP_SIZE - 1) / WARP_SIZE;

  // Step 1: Find global max for numerically stable softmax
  float local_max = -INFINITY;
  for (int i = 0; i < elems_per_thread; i++) {
    int idx = lane + i * WARP_SIZE;
    if (idx < E) {
      local_max = fmaxf(local_max, logits[idx]);
    }
  }
  float global_max = warp_reduce_max(local_max);
  global_max = __shfl_sync(0xFFFFFFFF, global_max, 0);

  // Step 2: Compute exp(logit - max) and softmax denominator
  float local_vals[4];
  int local_ids[4];
  float local_exps[4];
  int local_count = 0;
  float local_exp_sum = 0.0f;

  for (int i = 0; i < elems_per_thread; i++) {
    int idx = lane + i * WARP_SIZE;
    if (idx < E) {
      local_vals[local_count] = logits[idx];
      local_ids[local_count] = idx;
      local_exps[local_count] = expf(local_vals[local_count] - global_max);
      local_exp_sum += local_exps[local_count];
      local_count++;
    }
  }

  float total_exp_sum = warp_reduce_sum(local_exp_sum);
  total_exp_sum = __shfl_sync(0xFFFFFFFF, total_exp_sum, 0);
  float inv_sum = 1.0f / total_exp_sum;

  // Step 3: Compute softmax probabilities
  float local_probs[4];
  for (int i = 0; i < local_count; i++) {
    local_probs[i] = local_exps[i] * inv_sum;
  }

  // Step 4: K rounds of warp-wide argmax to find top-K
  bool used[4] = {false, false, false, false};

  for (int k = 0; k < K; k++) {
    float my_max = -1.0f;
    int my_idx = -1;
    int my_local = -1;
    for (int i = 0; i < local_count; i++) {
      if (!used[i] && local_probs[i] > my_max) {
        my_max = local_probs[i];
        my_idx = local_ids[i];
        my_local = i;
      }
    }

    // Warp-wide argmax reduction
    float best_val = my_max;
    int best_lane = lane;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xFFFFFFFF, best_val, offset);
      int other_lane = __shfl_down_sync(0xFFFFFFFF, best_lane, offset);
      if (other_val > best_val) {
        best_val = other_val;
        best_lane = other_lane;
      }
    }
    best_val = __shfl_sync(0xFFFFFFFF, best_val, 0);
    best_lane = __shfl_sync(0xFFFFFFFF, best_lane, 0);
    int winner_idx = __shfl_sync(0xFFFFFFFF, my_idx, best_lane);

    if (lane == 0) {
      out_ids[k] = winner_idx;
      out_weights[k] = best_val;
    }

    // Mark winner as used
    if (lane == best_lane && my_local >= 0) {
      used[my_local] = true;
    }
  }

  // Step 5: Renormalize and apply per_expert_scale
  if (lane == 0) {
    float renorm_sum = 0.0f;
    for (int k = 0; k < K; k++) {
      renorm_sum += out_weights[k];
    }
    float inv_renorm = (renorm_sum > 0.0f) ? (1.0f / renorm_sum) : 1.0f;
    for (int k = 0; k < K; k++) {
      out_weights[k] =
          out_weights[k] * inv_renorm * per_expert_scale[out_ids[k]];
    }
  }
}

// ---------------------------------------------------------------------------
// Standalone routing kernel (no cooperative launch needed)
// ---------------------------------------------------------------------------
__global__ void gemma4_routing_kernel(
    const float* __restrict__ router_logits,     // [T, E]
    const float* __restrict__ per_expert_scale,  // [E]
    float* __restrict__ topk_weights_out,        // [T, K]
    int* __restrict__ topk_ids_out,              // [T, K]
    int T, int E, int K) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane = tid % WARP_SIZE;
  const int warps_per_block = blockDim.x / WARP_SIZE;

  for (int base_t = blockIdx.x * warps_per_block; base_t < T;
       base_t += gridDim.x * warps_per_block) {
    int token_id = base_t + warp_id;

    if (token_id < T) {
      const float* my_logits = router_logits + token_id * E;

      float local_topk_weights[MAX_TOP_K];
      int local_topk_ids[MAX_TOP_K];

      select_topk_warp(my_logits, local_topk_weights, local_topk_ids,
                       per_expert_scale, E, K, lane);

      if (lane == 0) {
        for (int k = 0; k < K; k++) {
          topk_weights_out[token_id * K + k] = local_topk_weights[k];
          topk_ids_out[token_id * K + k] = local_topk_ids[k];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Python binding
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> gemma4_routing(
    torch::Tensor router_logits,     // [T, E] fp32
    torch::Tensor per_expert_scale,  // [E] fp32
    int top_k) {
  TORCH_CHECK(router_logits.is_cuda(), "router_logits must be CUDA");
  TORCH_CHECK(per_expert_scale.is_cuda(), "per_expert_scale must be CUDA");
  TORCH_CHECK(router_logits.dtype() == torch::kFloat32,
              "router_logits must be fp32");
  TORCH_CHECK(per_expert_scale.dtype() == torch::kFloat32,
              "per_expert_scale must be fp32");

  const int T = router_logits.size(0);
  const int E = router_logits.size(1);

  auto topk_weights = torch::empty({T, top_k}, router_logits.options());
  auto topk_ids = torch::empty(
      {T, top_k}, torch::dtype(torch::kInt32).device(router_logits.device()));

  // 4 warps per block (128 threads)
  const int warps_per_block = 4;
  const int threads = warps_per_block * WARP_SIZE;
  const int blocks = (T + warps_per_block - 1) / warps_per_block;

  gemma4_routing_kernel<<<blocks, threads>>>(
      router_logits.data_ptr<float>(), per_expert_scale.data_ptr<float>(),
      topk_weights.data_ptr<float>(), topk_ids.data_ptr<int>(), T, E, top_k);

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "gemma4_routing_kernel failed: ", cudaGetErrorString(err));

  return {topk_weights, topk_ids};
}

// When built via JIT (torch.utils.cpp_extension.load), TORCH_EXTENSION_NAME
// is defined and we need the pybind11 module. When built via CMake as part
// of _moe_C, the ops are registered in torch_bindings.cpp instead.
#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("routing", &gemma4_routing,
        "Gemma4 MoE routing: softmax -> topK -> renormalize -> scale",
        py::arg("router_logits"), py::arg("per_expert_scale"),
        py::arg("top_k"));
}
#endif
