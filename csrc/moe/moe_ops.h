#pragma once

#include <torch/extension.h>

void topk_softmax(
  torch::Tensor& topk_weights,
  torch::Tensor& topk_indices,
  torch::Tensor& token_expert_indices,
  torch::Tensor& gating_output);
