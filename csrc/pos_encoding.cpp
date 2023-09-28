#include <torch/extension.h>

void rotary_embedding(torch::Tensor &positions, torch::Tensor &query,
                      torch::Tensor &key, int head_size,
                      torch::Tensor &cos_sin_cache, bool is_neox);
void invoke_dequant_rotary_embedding(
    torch::Tensor &positions, // [num_tokens]
    torch::Tensor &query,     // [num_tokens, num_heads * head_size]
    torch::Tensor &query_out, // [num_tokens, num_heads * head_size]
    torch::Tensor &key,       // [num_tokens, num_kv_heads * head_size]
    torch::Tensor &key_out,   // [num_tokens, num_kv_heads * head_size]
    int head_size,
    torch::Tensor &cos_sin_cache, // [max_position, rot_dim]
    const float query_scale, const float key_scale, bool is_neox);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rotary_embedding", &rotary_embedding,
        "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");
  m.def("invoke_dequant_rotary_embedding", &invoke_dequant_rotary_embedding,
        "Dequant the input and apply rotary embedding.");
}
