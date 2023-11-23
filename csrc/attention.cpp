#include <torch/extension.h>
#include <c10/util/Optional.h>

void paged_attention_v1(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& head_mapping,    // [num_heads]
  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  bool enable_quant = false,
  const float k_scale = 1.0f,
  const float k_zp = 0.0f,
  const float v_scale = 1.0f,
  const float v_zp = 0.0f);

void paged_attention_v2(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& exp_sums,        // [num_seqs, num_heads, max_num_partitions]
  torch::Tensor& max_logits,      // [num_seqs, num_heads, max_num_partitions]
  torch::Tensor& tmp_out,         // [num_seqs, num_heads, max_num_partitions, head_size]
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& head_mapping,    // [num_heads]
  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
   bool enable_quant = false,
  const float k_scale = 1.0f,
  const float k_zp = 0.0f,
  const float v_scale = 1.0f,
  const float v_zp = 0.0f);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "paged_attention_v1",
    &paged_attention_v1,
    py::arg("out"), py::arg("query"), py::arg("key_cache"),
    py::arg("value_cache"), py::arg("head_mapping"), py::arg("scale"),
    py::arg("block_tables"), py::arg("context_lens"), py::arg("block_size"),
    py::arg("max_context_len"), py::arg("alibi_slopes"),
    py::arg("enable_quant") = false, py::arg("k_scale") = 1.0f,
    py::arg("k_zp") = 0.0f, py::arg("v_scale") = 1.0f,
    py::arg("v_zp") = 0.0f,
    "Compute the attention between an input query and the cached keys/values using PagedAttention.");
  m.def(
    "paged_attention_v2",
    &paged_attention_v2,
    py::arg("out"), py::arg("exp_sums"), py::arg("max_logits"), py::arg("tmp_out"), py::arg("query"), py::arg("key_cache"),
    py::arg("value_cache"), py::arg("head_mapping"), py::arg("scale"),
    py::arg("block_tables"), py::arg("context_lens"), py::arg("block_size"),
    py::arg("max_context_len"), py::arg("alibi_slopes"),
    py::arg("enable_quant") = false, py::arg("k_scale") = 1.0f,
    py::arg("k_zp") = 0.0f, py::arg("v_scale") = 1.0f,
    py::arg("v_zp") = 0.0f,
    "PagedAttention V2.");
}
