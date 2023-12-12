#include <torch/extension.h>

void rms_norm(torch::Tensor &out,    // [num_tokens, hidden_size]
              torch::Tensor &input,  // [num_tokens, hidden_size]
              torch::Tensor &weight, // [hidden_size]
              float epsilon, bool use_quant);

void invoke_dequant_add_residual_rms_norm_quant(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &gamma,    // [hidden_size]
    float scale, float epsilon);

void invoke_dequant_add_residual_rms_norm_quant(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &gamma,    // [hidden_size]
    torch::Tensor &scale,    // [num_tokens]
    float epsilon,
    float weight_dequant_scale);

void invoke_add_residual_rms_norm_quant(
    torch::Tensor &out,      // [..., hidden_size]
    torch::Tensor &input,    // [..., hidden_size]
    torch::Tensor &residual, // [..., hidden_size]
    torch::Tensor &gamma,    // [hidden_size]
    float epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &rms_norm, py::arg("out"), py::arg("input"),
        py::arg("weight"), py::arg("epsilon"), py::arg("use_quant") = false,
        "Apply Root Mean Square (RMS) Normalization to the input tensor.");

  m.def("invoke_dequant_add_residual_rms_norm_quant",
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &, float, float>(
            &invoke_dequant_add_residual_rms_norm_quant),
        "Add the dequanted result and residual, then use RMS norm and quant output.");
  m.def("invoke_dequant_add_residual_rms_norm_quant",
        py::overload_cast<torch::Tensor &, torch::Tensor &, torch::Tensor &,
                          torch::Tensor &, torch::Tensor &, float, float>(
            &invoke_dequant_add_residual_rms_norm_quant),
        "Add the dequanted result and residual, then use RMS norm and quant output.");
  m.def("invoke_add_residual_rms_norm_quant",
        &invoke_add_residual_rms_norm_quant,
        "Add the result and residual, then use RMS norm and quant output.");
}
