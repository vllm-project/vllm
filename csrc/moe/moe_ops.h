#pragma once

#include <torch/extension.h>

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output);
