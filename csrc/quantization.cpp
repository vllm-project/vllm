#include <torch/extension.h>

torch::Tensor gemm_forward_cuda(torch::Tensor _in_feats, torch::Tensor _kernel,
    torch::Tensor _scaling_factors, torch::Tensor _zeros, int split_k_iters);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "gemm_forward_cuda",
    &gemm_forward_cuda,
    "quantized gemm");
}
