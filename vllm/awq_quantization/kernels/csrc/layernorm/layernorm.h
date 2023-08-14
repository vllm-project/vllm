#include <torch/extension.h>

void layernorm_forward_cuda(torch::Tensor _input, torch::Tensor _gamma, torch::Tensor _out, float eps);
