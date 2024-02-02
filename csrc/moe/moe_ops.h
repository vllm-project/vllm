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
