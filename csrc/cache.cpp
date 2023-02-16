#include <torch/extension.h>

void copy_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);

void copy_cache_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping) {
  copy_blocks(src, dst, block_mapping);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "copy_cache_blocks",
    &copy_cache_blocks,
    "Copy the cache blocks from src to dst");
}
