#pragma once

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#ifndef USE_ROCM
torch::stable::Tensor permute_cols(torch::stable::Tensor A,
                                   torch::stable::Tensor perm);
#endif
