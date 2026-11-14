/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
 * Copyright (c) 2024, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
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

// Argument marshalling and dtype dispatch of the fused softplus-sqrt topk. The
// kernels, the launchers and the per-(gating dtype, topk_indices dtype)
// instantiations live in topk_softplus_sqrt_kernels.cuh and in the
// topk_softplus_sqrt_<dtype>_<indtype>.cu units that instantiate them.
#include "topk_softplus_sqrt_kernels.cuh"

template <typename ComputeType>
void dispatch_topk_softplus_sqrt_launch(
    const ComputeType* gating_output, torch::stable::Tensor& topk_weights,
    torch::stable::Tensor& topk_indices,
    torch::stable::Tensor& token_expert_indices, int num_tokens,
    int num_experts, int topk, bool renormalize, double routed_scaling_factor,
    const std::optional<torch::stable::Tensor>& correction_bias,
    const std::optional<torch::stable::Tensor>& input_ids,
    const std::optional<torch::stable::Tensor>& tid2eid, cudaStream_t stream,
    const std::optional<torch::stable::Tensor>& is_padding,
    const std::optional<torch::stable::Tensor>& bias_vl,
    int64_t image_sentinel_lo) {
  const float* bias_ptr = nullptr;
  if (correction_bias.has_value()) {
    bias_ptr = correction_bias.value().const_data_ptr<float>();
  }

  const float* bias_vl_ptr = nullptr;
  if (bias_vl.has_value()) {
    const torch::stable::Tensor& bias_vl_tensor = bias_vl.value();
    STD_TORCH_CHECK(input_ids.has_value(),
                    "input_ids is required when bias_vl is set");
    STD_TORCH_CHECK(
        bias_vl_tensor.scalar_type() == torch::headeronly::ScalarType::Float,
        "bias_vl tensor must be float32");
    STD_TORCH_CHECK(bias_vl_tensor.dim() == 1, "bias_vl tensor must be 1D");
    STD_TORCH_CHECK(bias_vl_tensor.size(0) == num_experts,
                    "bias_vl size mismatch, expected: ", num_experts);
    STD_TORCH_CHECK(bias_vl_tensor.is_contiguous(),
                    "bias_vl tensor must be contiguous");
    bias_vl_ptr = bias_vl_tensor.const_data_ptr<float>();
  }

  auto launch = [&](auto* topk_indices_ptr) {
    using OutIndType =
        typename std::remove_pointer<decltype(topk_indices_ptr)>::type;

    const bool* is_padding_ptr = nullptr;
    if (is_padding.has_value()) {
      const torch::stable::Tensor& is_padding_tensor = is_padding.value();
      STD_TORCH_CHECK(is_padding_tensor.scalar_type() ==
                          torch::headeronly::ScalarType::Bool,
                      "is_padding tensor must be bool");
      STD_TORCH_CHECK(is_padding_tensor.dim() == 1,
                      "is_padding tensor must be 1D");
      STD_TORCH_CHECK(is_padding_tensor.size(0) == num_tokens,
                      "is_padding size mismatch, expected: ", num_tokens);
      STD_TORCH_CHECK(is_padding_tensor.is_contiguous(),
                      "is_padding tensor must be contiguous");
      is_padding_ptr = is_padding_tensor.const_data_ptr<bool>();
    }

    if (tid2eid.has_value()) {
      STD_TORCH_CHECK(input_ids.has_value(),
                      "input_ids is required for hash MoE");
      STD_TORCH_CHECK(
          input_ids.value().scalar_type() == tid2eid.value().scalar_type(),
          "input_ids and tid2eid must have the same dtype");
      if (tid2eid.value().scalar_type() ==
          torch::headeronly::ScalarType::Long) {
        vllm::moe::topkGatingSoftplusSqrtKernelLauncher<OutIndType, ComputeType,
                                                        int64_t>(
            gating_output, topk_weights.mutable_data_ptr<float>(),
            topk_indices_ptr, token_expert_indices.mutable_data_ptr<int>(),
            num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
            bias_ptr, true, input_ids.value().const_data_ptr<int64_t>(),
            tid2eid.value().const_data_ptr<int64_t>(), stream, is_padding_ptr,
            bias_vl_ptr, image_sentinel_lo);
      } else {
        STD_TORCH_CHECK(tid2eid.value().scalar_type() ==
                        torch::headeronly::ScalarType::Int);
        vllm::moe::topkGatingSoftplusSqrtKernelLauncher<OutIndType, ComputeType,
                                                        int>(
            gating_output, topk_weights.mutable_data_ptr<float>(),
            topk_indices_ptr, token_expert_indices.mutable_data_ptr<int>(),
            num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
            bias_ptr, true, input_ids.value().const_data_ptr<int>(),
            tid2eid.value().const_data_ptr<int>(), stream, is_padding_ptr,
            bias_vl_ptr, image_sentinel_lo);
      }
    } else if (bias_vl_ptr != nullptr) {
      // Non-hash rows with bias_vl: image tokens (five sentinel ids starting
      // at image_sentinel_lo) select experts by score + bias_vl. input_ids
      // drives the per-row branch, so
      // dispatch the hash-index type on its dtype.
      if (input_ids.value().scalar_type() ==
          torch::headeronly::ScalarType::Long) {
        vllm::moe::topkGatingSoftplusSqrtKernelLauncher<OutIndType, ComputeType,
                                                        int64_t>(
            gating_output, topk_weights.mutable_data_ptr<float>(),
            topk_indices_ptr, token_expert_indices.mutable_data_ptr<int>(),
            num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
            bias_ptr, false, input_ids.value().const_data_ptr<int64_t>(),
            static_cast<const int64_t*>(nullptr), stream, is_padding_ptr,
            bias_vl_ptr, image_sentinel_lo);
      } else {
        STD_TORCH_CHECK(input_ids.value().scalar_type() ==
                        torch::headeronly::ScalarType::Int);
        vllm::moe::topkGatingSoftplusSqrtKernelLauncher<OutIndType, ComputeType,
                                                        int>(
            gating_output, topk_weights.mutable_data_ptr<float>(),
            topk_indices_ptr, token_expert_indices.mutable_data_ptr<int>(),
            num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
            bias_ptr, false, input_ids.value().const_data_ptr<int>(),
            static_cast<const int*>(nullptr), stream, is_padding_ptr,
            bias_vl_ptr, image_sentinel_lo);
      }
    } else {
      vllm::moe::topkGatingSoftplusSqrtKernelLauncher<OutIndType, ComputeType>(
          gating_output, topk_weights.mutable_data_ptr<float>(),
          topk_indices_ptr, token_expert_indices.mutable_data_ptr<int>(),
          num_tokens, num_experts, topk, renormalize, routed_scaling_factor,
          bias_ptr, false, static_cast<const OutIndType*>(nullptr),
          static_cast<const OutIndType*>(nullptr), stream, is_padding_ptr,
          nullptr, 0);
    }
  };

  if (topk_indices.scalar_type() == torch::headeronly::ScalarType::Int) {
    launch(topk_indices.mutable_data_ptr<int>());
  } else if (topk_indices.scalar_type() ==
             torch::headeronly::ScalarType::UInt32) {
    launch(topk_indices.mutable_data_ptr<uint32_t>());
  } else {
    STD_TORCH_CHECK(topk_indices.scalar_type() ==
                    torch::headeronly::ScalarType::Long);
    launch(topk_indices.mutable_data_ptr<int64_t>());
  }
}

void topk_softplus_sqrt(
    torch::stable::Tensor& topk_weights,          // [num_tokens, topk]
    torch::stable::Tensor& topk_indices,          // [num_tokens, topk]
    torch::stable::Tensor& token_expert_indices,  // [num_tokens, topk]
    torch::stable::Tensor& gating_output,         // [num_tokens, num_experts]
    bool renormalize, double routed_scaling_factor,
    const std::optional<torch::stable::Tensor>& correction_bias,
    const std::optional<torch::stable::Tensor>& input_ids,
    const std::optional<torch::stable::Tensor>& tid2eid,
    const std::optional<torch::stable::Tensor>& is_padding,
    const std::optional<torch::stable::Tensor>& bias_vl,
    int64_t image_sentinel_lo) {
  const int num_experts = gating_output.size(-1);
  const auto num_tokens = gating_output.numel() / num_experts;
  const int topk = topk_weights.size(-1);
  const torch::stable::accelerator::DeviceGuard guard(
      gating_output.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(gating_output.get_device_index());

  if (gating_output.scalar_type() == torch::headeronly::ScalarType::Float) {
    dispatch_topk_softplus_sqrt_launch<float>(
        gating_output.const_data_ptr<float>(), topk_weights, topk_indices,
        token_expert_indices, num_tokens, num_experts, topk, renormalize,
        routed_scaling_factor, correction_bias, input_ids, tid2eid, stream,
        is_padding, bias_vl, image_sentinel_lo);
  } else if (gating_output.scalar_type() ==
             torch::headeronly::ScalarType::Half) {
    dispatch_topk_softplus_sqrt_launch<__half>(
        reinterpret_cast<const __half*>(gating_output.const_data_ptr()),
        topk_weights, topk_indices, token_expert_indices, num_tokens,
        num_experts, topk, renormalize, routed_scaling_factor, correction_bias,
        input_ids, tid2eid, stream, is_padding, bias_vl, image_sentinel_lo);
  } else if (gating_output.scalar_type() ==
             torch::headeronly::ScalarType::BFloat16) {
    dispatch_topk_softplus_sqrt_launch<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(gating_output.const_data_ptr()),
        topk_weights, topk_indices, token_expert_indices, num_tokens,
        num_experts, topk, renormalize, routed_scaling_factor, correction_bias,
        input_ids, tid2eid, stream, is_padding, bias_vl, image_sentinel_lo);
  } else {
    STD_TORCH_CHECK(false, "Unsupported gating_output data type: ",
                    gating_output.scalar_type());
  }
}
