#include <torch/extension.h>

torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);

void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("awq_gemm", &awq_gemm, "Quantized GEMM for AWQ");
  m.def("squeezellm_gemm", &squeezellm_gemm, "Quantized GEMM for SqueezeLLM");
}
