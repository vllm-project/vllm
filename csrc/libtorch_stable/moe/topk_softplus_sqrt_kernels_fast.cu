/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/pull/15402
 * cpp/tensorrt_llm/kernels/customMoeRoutingKernels.cu (gate_forward)
 * Copyright (c) 2026, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "../../cuda_compat.h"
#include "libtorch_stable/torch_utils.h"

// CUDA-only: depends on cub-based reduce_topk and a 32-lane warp layout.
#ifndef USE_ROCM

  #include <cooperative_groups.h>
  #include <cooperative_groups/reduce.h>

  #include "moeTopKFuncs.cuh"  // vllm::moe::reduce_topk::reduceTopK

namespace cg = cooperative_groups;

namespace vllm {
namespace moe {

// CUDA kernel for gate forward.
// Input: pre-computed scores from linear(x, weight) done outside the kernel.
// Template parameters:
//   nExperts: number of experts
//   topK: number of top experts to select
//   hash: true for hash mode, false for topk mode
// One warp per row (batch element).
template <int nExperts, int topK, bool hash>
__global__ void topk_softplus_sqrt_fast_kernel(
    float const* __restrict__ scores_in,  // [batch_size, nExperts]
    float const* __restrict__ bias,       // [nExperts] (only when hash=false)
    int const* __restrict__ input_ids,    // [batch_size] (only when hash=true)
    int const* __restrict__ tid2eid,      // [vocab_size, topK] (only hash=true)
    float* __restrict__ out_weights,      // [batch_size, topK]
    int* __restrict__ out_indices,        // [batch_size, topK]
    int batch_size, float route_scale) {
  // Compile-time constants
  constexpr int kExpertsPerThread = nExperts / WARP_SIZE;
  constexpr int kWarpsPerBlock = 4;  // Adjust based on occupancy needs

  // Shared memory for original scores (one array per warp in the block)
  __shared__ float smem_scores[kWarpsPerBlock][nExperts];

  // One warp per batch element
  int const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int const local_warp_id = (threadIdx.x / WARP_SIZE) % kWarpsPerBlock;
  int const lane_id = threadIdx.x % WARP_SIZE;

  if (global_warp_id >= batch_size) return;

  auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

  // Pointer to this warp's shared memory and input scores
  float* my_smem = smem_scores[local_warp_id];
  float const* scores_row = scores_in + global_warp_id * nExperts;

  // Load scores, apply score function (softplus + sqrt), store to shared mem.
#pragma unroll
  for (int e = 0; e < kExpertsPerThread; ++e) {
    int expert_id = lane_id + e * WARP_SIZE;
    float s = scores_row[expert_id];
    float sp = log1pf(expf(s));
    float score = sqrtf(sp);
    my_smem[expert_id] = score;  // Store original score to shared memory
  }
  __syncwarp();  // Ensure all scores are written before reading

  // Output: each of the first K lanes holds one value.
  float my_topk_value = 0.0f;
  int my_topk_index = 0;

  if constexpr (hash) {
    // Hash mode: directly read from shared memory
    int token_id = input_ids[global_warp_id];
    int const* expert_ids = tid2eid + token_id * topK;

    if (lane_id < topK) {
      int expert_id = expert_ids[lane_id];
      my_topk_index = expert_id;
      my_topk_value = my_smem[expert_id];  // Direct lookup from shared memory
    }
  } else {
    // Topk mode: load from shared memory, add bias in registers for topk.
    float scores[kExpertsPerThread];
    int indices[kExpertsPerThread];

#pragma unroll
    for (int e = 0; e < kExpertsPerThread; ++e) {
      int expert_id = lane_id + e * WARP_SIZE;
      indices[e] = expert_id;
      scores[e] = my_smem[expert_id] + bias[expert_id];  // biased for selection
    }

    // Use reduceTopK to find the top-k experts (result broadcast to all lanes).
    float topk_values[topK];
    int32_t topk_indices[topK];
    constexpr float minValue = -1e30f;
    reduce_topk::reduceTopK<topK, float, kExpertsPerThread>(
        warp, topk_values, topk_indices, scores, indices, minValue, topK);

    // Gather original weights (without bias) from shared memory.
    if (lane_id < topK) {
      int expert_id = topk_indices[lane_id];
      my_topk_index = expert_id;
      my_topk_value = my_smem[expert_id];  // original score (no bias)
    }
  }

  // Reduce to get the sum (first K lanes have values, others have 0).
  float weight_sum = cg::reduce(warp, my_topk_value, cg::plus<float>{});

  // Normalize weights and write output (first K lanes).
  if (lane_id < topK) {
    out_weights[global_warp_id * topK + lane_id] =
        (my_topk_value / weight_sum) * route_scale;
    out_indices[global_warp_id * topK + lane_id] = my_topk_index;
  }
}

// C++ launcher (output tensors passed as parameters). All tensors are float32.
template <int nExperts, int topK, bool hash>
void launch_topk_softplus_sqrt_fast_kernel(float* scores_in, float* bias, int* input_ids,
                                int* tid2eid, float* out_weights,
                                int* out_indices, int batch_size,
                                float route_scale, cudaStream_t stream) {
  constexpr int warps_per_block = 4;
  constexpr int threads_per_block = warps_per_block * WARP_SIZE;
  int const blocks = (batch_size + warps_per_block - 1) / warps_per_block;

  topk_softplus_sqrt_fast_kernel<nExperts, topK, hash>
      <<<blocks, threads_per_block, 0, stream>>>(scores_in, bias, input_ids,
                                                 tid2eid, out_weights,
                                                 out_indices, batch_size,
                                                 route_scale);
}

// Dispatch over (n_experts, is_hash). topK and n_experts match the source:
// DeepSeek-V4 uses top-k 6, and n_experts is 256 or 384.
template <int topK>
void topk_softplus_sqrt_fast_dispatch_experts(float* scores, float* bias, int* input_ids,
                                   int* tid2eid, float* weights, int* indices,
                                   int batch_size, int n_experts,
                                   float route_scale, bool is_hash,
                                   cudaStream_t stream) {
  switch (n_experts) {
    case 256:
      if (is_hash) {
        launch_topk_softplus_sqrt_fast_kernel<256, topK, true>(
            scores, nullptr, input_ids, tid2eid, weights, indices, batch_size,
            route_scale, stream);
      } else {
        launch_topk_softplus_sqrt_fast_kernel<256, topK, false>(
            scores, bias, nullptr, nullptr, weights, indices, batch_size,
            route_scale, stream);
      }
      break;
    case 384:
      if (is_hash) {
        launch_topk_softplus_sqrt_fast_kernel<384, topK, true>(
            scores, nullptr, input_ids, tid2eid, weights, indices, batch_size,
            route_scale, stream);
      } else {
        launch_topk_softplus_sqrt_fast_kernel<384, topK, false>(
            scores, bias, nullptr, nullptr, weights, indices, batch_size,
            route_scale, stream);
      }
      break;
    default:
      STD_TORCH_CHECK(false, "topk_softplus_sqrt_fast only supports n_experts 256 or 384, "
                             "got ",
                      n_experts);
  }
}

}  // namespace moe
}  // namespace vllm

// Stable-ABI entry. Signature mirrors `topk_softplus_sqrt` so it is a drop-in
// replacement. gating_output/topk_weights are float32, topk_indices is int32.
// topk is taken from topk_weights.size(-1) (must be 6), hash mode is inferred
// from tid2eid, and route_scale = routed_scaling_factor. token_expert_indices
// is accepted for ABI parity but unused (this kernel does not permute).
// Selection: sqrt(softplus(score)) (+ bias for topk selection), top-k, then
// renormalize by the sum of the selected (unbiased) weights and scale.
void topk_softplus_sqrt_fast(
    torch::stable::Tensor& topk_weights,          // [num_tokens, topk] fp32
    torch::stable::Tensor& topk_indices,          // [num_tokens, topk] int32
    torch::stable::Tensor& token_expert_indices,  // unused (ABI parity)
    torch::stable::Tensor& gating_output,         // [num_tokens, n_experts] fp32
    bool renormalize, double routed_scaling_factor,
    const std::optional<torch::stable::Tensor>& correction_bias,
    const std::optional<torch::stable::Tensor>& input_ids,
    const std::optional<torch::stable::Tensor>& tid2eid) {
  const int n_experts = gating_output.size(-1);
  const int batch_size = gating_output.numel() / n_experts;
  const int topk = topk_weights.size(-1);
  const bool is_hash = tid2eid.has_value();

  STD_TORCH_CHECK(
      renormalize,
      "topk_softplus_sqrt_fast always renormalizes; renormalize must be true");
  STD_TORCH_CHECK(topk == 6, "topk_softplus_sqrt_fast only supports topk 6, got ",
                  topk);
  STD_TORCH_CHECK(
      gating_output.scalar_type() == torch::headeronly::ScalarType::Float,
      "gating_output must be float32");
  STD_TORCH_CHECK(
      topk_weights.scalar_type() == torch::headeronly::ScalarType::Float,
      "topk_weights must be float32");
  STD_TORCH_CHECK(
      topk_indices.scalar_type() == torch::headeronly::ScalarType::Int,
      "topk_indices must be int32");

  float* bias_ptr = nullptr;
  int* input_ids_ptr = nullptr;
  int* tid2eid_ptr = nullptr;
  if (is_hash) {
    STD_TORCH_CHECK(input_ids.has_value(), "hash mode requires input_ids");
    input_ids_ptr = input_ids.value().mutable_data_ptr<int>();
    tid2eid_ptr = tid2eid.value().mutable_data_ptr<int>();
  } else {
    STD_TORCH_CHECK(correction_bias.has_value(), "topk mode requires bias");
    bias_ptr = correction_bias.value().mutable_data_ptr<float>();
  }

  const torch::stable::accelerator::DeviceGuard guard(
      gating_output.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(gating_output.get_device_index());

  float* scores = gating_output.mutable_data_ptr<float>();
  float* weights = topk_weights.mutable_data_ptr<float>();
  int* indices = topk_indices.mutable_data_ptr<int>();

  vllm::moe::topk_softplus_sqrt_fast_dispatch_experts<6>(
      scores, bias_ptr, input_ids_ptr, tid2eid_ptr, weights, indices,
      batch_size, n_experts, static_cast<float>(routed_scaling_factor), is_hash,
      stream);
}

#endif  // USE_ROCM
