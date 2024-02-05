#pragma once

#include <torch/extension.h>

void topk_softmax(
  torch::Tensor& topk_weights,
  torch::Tensor& topk_indices,
  torch::Tensor& token_expert_indices,
  torch::Tensor& gating_output);

void expand_and_permute(
  torch::Tensor& permuted_tokens,
  torch::Tensor& cum_num_tokens_per_expert,
  torch::Tensor& reverse_permutation_map,
  torch::Tensor& input_tokens,
  torch::Tensor& topk_indices,
  torch::Tensor& token_expert_indices);

void moe_mlp(
  torch::Tensor& moe_output,
  torch::Tensor& input_tokens,
  torch::Tensor& cum_num_tokens_per_expert,
  torch::Tensor& fc1_expert_weights,
  const c10::optional<torch::Tensor>& fc1_expert_biases,
  int fc1_activation_type,
  torch::Tensor& fc2_expert_weights);

void unpermute_and_reduce(
  torch::Tensor& output_tokens,
  torch::Tensor& experts_output,
  torch::Tensor& topk_weights,
  torch::Tensor& topk_indices,
  torch::Tensor& reverse_permutation_map,
  bool renormalize);
