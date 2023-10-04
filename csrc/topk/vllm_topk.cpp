
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cstdio>
#include <vector>
#include "ATen/core/TensorBody.h"

void top_k(const torch::Tensor src, torch::Tensor dst,
           const std::vector<int> &top_ks, const std::vector<float> &top_ps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("top_k", &top_k, "Apply vllm top_k");
}
