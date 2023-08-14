#pragma once
#include <torch/extension.h>

void rotary_embedding_neox(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  torch::Tensor& cos_sin_cache);
