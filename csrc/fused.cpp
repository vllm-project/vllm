#include <torch/extension.h>

void invoke_dequant_add_residual(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    float scale);

void invoke_dequant_add_residual(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &scale);   // [num_tokens]

void invoke_dequant(torch::Tensor &out,   // [..., hidden_size]
                    torch::Tensor &input, // [..., hidden_size]
                    float scale);

void invoke_dequant(torch::Tensor &out,   // [..., hidden_size]
                    torch::Tensor &input, // [..., hidden_size]
                    torch::Tensor &scale);

void invoke_quant(torch::Tensor &out,   // [..., hidden_size]
                  torch::Tensor &input, // [..., hidden_size]
                  float scale);

void invoke_quant(torch::Tensor &out,   // [..., hidden_size]
                  torch::Tensor &input, // [..., hidden_size]
                  torch::Tensor &scale);  // [num_tokens]

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("invoke_dequant_add_residual",
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          float>(&invoke_dequant_add_residual),
        "Add the dequanted result and residual.");
  m.def("invoke_dequant_add_residual",
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &>(&invoke_dequant_add_residual),
        "Add the dequanted result and residual.");
  m.def("invoke_dequant", py::overload_cast<torch::Tensor &, torch::Tensor &, float>(&invoke_dequant), "Dequant.");
  m.def("invoke_dequant", py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &>(&invoke_dequant), "Dequant.");
  m.def(
      "invoke_quant",
      py::overload_cast<torch::Tensor &, torch::Tensor &, float>(&invoke_quant),
      "Quant.");
  m.def("invoke_quant", py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &>(
      &invoke_quant),
      "Quant.");
}
