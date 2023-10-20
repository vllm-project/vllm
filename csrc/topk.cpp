
#include "ATen/core/TensorBody.h"
#include <c10/cuda/CUDAGuard.h>
#include <cstdio>
#include <torch/extension.h>
#include <vector>

void top_k(const torch::Tensor src,
           const torch::Tensor softmax_src,
           torch::Tensor       dst,
           bool                top_k,
           int                 max_top_k,
           torch::Tensor       top_ks,
           bool                top_p,
           torch::Tensor       top_ps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("top_k", &top_k, "Apply vllm top_k");
}
