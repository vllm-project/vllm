#include <torch/extension.h>

void rotary_embedding(
    torch::Tensor &positions, // [batch_size, seq_len] or [num_tokens]
    torch::Tensor &query,     // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    torch::Tensor &key,       // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    int head_size,
    torch::Tensor &cos_sin_cache, // [max_position, rot_dim]
    bool is_neox,
    torch::Tensor &query_out, // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    torch::Tensor &key_out, // [batch_size, seq_len, num_heads * head_size] or [num_tokens, num_heads * head_size]
    bool use_dequant = false, const float query_scale = 1.0f,
    const float key_scale = 1.0f);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rotary_embedding", &rotary_embedding, py::arg("positions"),
        py::arg("query"), py::arg("key"), py::arg("head_size"),
        py::arg("cos_sin_cache"), py::arg("is_neox"),
        py::arg("query_out") = torch::empty({}),
        py::arg("key_out") = torch::empty({}), py::arg("use_dequant") = false,
        py::arg("query_scale") = 1.0f, py::arg("key_scale") = 1.0f,
        "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");
}
