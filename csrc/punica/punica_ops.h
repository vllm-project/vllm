#pragma once

#include <torch/extension.h>

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

void dispatch_bgmv(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                   torch::Tensor indicies, int64_t layer_idx, double scale);

void dispatch_bgmv_low_level(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                             torch::Tensor indicies, int64_t layer_idx,
                             double scale, int64_t h_in, int64_t h_out,
                             int64_t y_offset);
