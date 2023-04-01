#include <torch/extension.h>

void silu_and_mul(
  torch::Tensor& out,
  torch::Tensor& input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "silu_and_mul",
    &silu_and_mul,
    "Activation function used in SwiGLU.");
}
