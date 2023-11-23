#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping);

void reshape_and_cache(
    torch::Tensor& key,   
    torch::Tensor& value, 
    torch::Tensor& key_cache, 
    torch::Tensor& value_cache, 
    torch::Tensor& slot_mapping, 
    bool use_quant = false, const float k_scale = 1.0f, const float k_zp = 0.0f,
    const float v_scale = 1.0f, const float v_zp = 0.0f);

void gather_cached_kv(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "swap_blocks",
    &swap_blocks,
    "Swap in (out) the cache blocks from src to dst");
  m.def(
    "copy_blocks",
    &copy_blocks,
    "Copy the cache blocks from src to dst");
  m.def("reshape_and_cache", &reshape_and_cache, py::arg("key"),
        py::arg("value"), py::arg("key_cache"), py::arg("value_cache"),
        py::arg("slot_mapping"), py::arg("use_quant") = false,
        py::arg("k_scale") = 1.0f, py::arg("k_zp") = 0.0f,
        py::arg("v_scale") = 1.0f, py::arg("v_zp") = 0.0f,
        "Reshape the key and value tensors and cache them");
  m.def(
    "gather_cached_kv",
    &gather_cached_kv,
    "Gather key and value from the cache into contiguous QKV tensors");
}
