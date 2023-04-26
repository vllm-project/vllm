#include <torch/extension.h>

void rotary_embedding_neox(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  torch::Tensor& cos_sin_cache);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "rotary_embedding_neox",
    &rotary_embedding_neox,
    "Apply GPT-NeoX style rotary embedding to query and key");
}
