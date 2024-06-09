#pragma once

#include <torch/all.h>

void dispatch_bgmv(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                   torch::Tensor indicies, int64_t layer_idx, double scale);

void dispatch_bgmv_low_level(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                             torch::Tensor indicies, int64_t layer_idx,
                             double scale, int64_t h_in, int64_t h_out,
                             int64_t y_offset);
