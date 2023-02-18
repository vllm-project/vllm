#include <torch/extension.h>

void copy_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);

void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "copy_cache_blocks",
    &copy_blocks,
    "Copy the cache blocks from src to dst");
  m.def(
    "reshape_and_cache",
    &reshape_and_cache,
    "Reshape the key and value tensors and cache them");
}
