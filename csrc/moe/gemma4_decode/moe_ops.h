// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Declarations for Gemma4 MoE decode-optimized CUDA kernels.

#pragma once

#include <torch/all.h>

// Gemma4 MoE expert GEMV forward pass for decode (small batch).
// Runs gate_up GEMV + GELU activation + down GEMV for each expert assignment.
//
// Args:
//   hidden_states: [T, H] bf16 input activations
//   w13: [E, 2*N, H] bf16 packed gate+up expert weights
//   w2: [E, H, N] bf16 down-projection expert weights
//   topk_ids: [T, K] int32 selected expert indices per token
//   topk_weights: [T, K] fp32 routing weights per token
//   intermediate_size: N (per TP shard)
//
// Returns: [T, H] bf16 output
torch::Tensor gemma4_moe_decode_forward(torch::Tensor hidden_states,
                                        torch::Tensor w13, torch::Tensor w2,
                                        torch::Tensor topk_ids,
                                        torch::Tensor topk_weights,
                                        int intermediate_size);

// Gemma4 routing: softmax -> top-K -> renormalize -> per_expert_scale.
//
// Args:
//   router_logits: [T, E] fp32 router output logits
//   per_expert_scale: [E] fp32 Gemma4-specific expert scaling factors
//   top_k: number of experts to select per token
//
// Returns: (topk_weights [T, K] fp32, topk_ids [T, K] int32)
std::tuple<torch::Tensor, torch::Tensor> gemma4_routing(
    torch::Tensor router_logits, torch::Tensor per_expert_scale, int top_k);
