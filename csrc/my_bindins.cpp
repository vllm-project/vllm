#include <torch/extension.h>
#include <pybind11/pybind11.h>

torch::Tensor ggml_moe_a8(torch::Tensor X, torch::Tensor W,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids, int64_t type, int64_t row,
                          int64_t top_k, int64_t tokens);

torch::Tensor ggml_moe_kenel(torch::Tensor X, torch::Tensor W,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids, int64_t type,
                             int64_t row, int64_t top_k, int64_t tokens) {
  return ggml_moe_a8(X, W, sorted_token_ids, expert_ids, type, row, top_k,
                     tokens);
}

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X,
                                  int64_t type, int64_t row);

torch::Tensor ggml_mul_mat_vec_a8_k(torch::Tensor W, torch::Tensor X,
                                  int64_t type, int64_t row)
{
    return ggml_mul_mat_vec_a8(W, X, type, row);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggmp_moe_a8", &ggml_moe_kenel, "GGML moe kernel");
  m.def("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8_k, "matvedc");
}
