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
// instantiations live in topk_softplus_sqrt_kernels.cuh and in the generated
// topk_softplus_sqrt_launch_*.cu units, see generate_topk_softplus_sqrt.py.
#include "topk_softplus_sqrt_kernels.cuh"

// Prototypes of the generated entry points. Regenerate them - together with
// the dispatch below - with csrc/moe/generate_topk_softplus_sqrt.py.
#include "topk_softplus_sqrt_launch_entries.inc"

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

  vllm::moe::TopkSoftplusSqrtParams params;
  params.topk_weights = topk_weights.mutable_data_ptr<float>();
  params.token_expert_indices = token_expert_indices.mutable_data_ptr<int>();
  params.num_tokens = num_tokens;
  params.num_experts = num_experts;
  params.topk = topk;
  params.renormalize = renormalize;
  params.routed_scaling_factor = routed_scaling_factor;
  params.stream = stream;
  params.image_sentinel_lo = image_sentinel_lo;

  if (correction_bias.has_value()) {
    params.correction_bias = correction_bias.value().const_data_ptr<float>();
  }

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
    params.bias_vl = bias_vl_tensor.const_data_ptr<float>();
  }

  if (is_padding.has_value()) {
    const torch::stable::Tensor& is_padding_tensor = is_padding.value();
    STD_TORCH_CHECK(is_padding_tensor.scalar_type() ==
                        torch::headeronly::ScalarType::Bool,
                    "is_padding tensor must be bool");
    STD_TORCH_CHECK(is_padding_tensor.dim() == 1, "is_padding tensor must be 1D");
    STD_TORCH_CHECK(is_padding_tensor.size(0) == num_tokens,
                    "is_padding size mismatch, expected: ", num_tokens);
    STD_TORCH_CHECK(is_padding_tensor.is_contiguous(),
                    "is_padding tensor must be contiguous");
    params.is_padding = is_padding_tensor.const_data_ptr<bool>();
  }

  if (tid2eid.has_value()) {
    STD_TORCH_CHECK(input_ids.has_value(),
                    "input_ids is required for hash MoE");
    STD_TORCH_CHECK(
        input_ids.value().scalar_type() == tid2eid.value().scalar_type(),
        "input_ids and tid2eid must have the same dtype");
    const auto hash_scalar_type = tid2eid.value().scalar_type();
    STD_TORCH_CHECK(hash_scalar_type == torch::headeronly::ScalarType::Long ||
                        hash_scalar_type == torch::headeronly::ScalarType::Int,
                    "Unsupported input_ids/tid2eid data type: ",
                    hash_scalar_type);
    params.hash_ind_type_is_int64 =
        hash_scalar_type == torch::headeronly::ScalarType::Long;
    params.input_ids = input_ids.value().const_data_ptr();
    params.tid2eid = tid2eid.value().const_data_ptr();
  } else if (params.bias_vl != nullptr) {
    // Non-hash rows with bias_vl: image tokens (five sentinel ids starting at
    // image_sentinel_lo) select experts by score + bias_vl. input_ids drives
    // the per-row branch, so its dtype drives the hash-index instantiation.
    const auto hash_scalar_type = input_ids.value().scalar_type();
    STD_TORCH_CHECK(hash_scalar_type == torch::headeronly::ScalarType::Long ||
                        hash_scalar_type == torch::headeronly::ScalarType::Int,
                    "Unsupported input_ids data type: ", hash_scalar_type);
    params.hash_ind_type_is_int64 =
        hash_scalar_type == torch::headeronly::ScalarType::Long;
    params.input_ids = input_ids.value().const_data_ptr();
  }

  // clang-format off
  #include "topk_softplus_sqrt_dispatch.inc"
  // clang-format on
}
