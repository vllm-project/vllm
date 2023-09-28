#include <torch/extension.h>

void invoke_dequant_add_residual(
    torch::Tensor &out,      // [num_tokens, hidden_size]
    torch::Tensor &input,    // [num_tokens, hidden_size]
    torch::Tensor &residual, // [num_tokens, hidden_size]
    float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("invoke_dequant_add_residual", &invoke_dequant_add_residual,
        "Add the dequanted result and residual.");
}
