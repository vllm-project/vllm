#include <torch/extension.h>
#include "metal_kernels.h"

// PyTorch bindings for Metal kernels
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_v1",
          &vllm::metal::paged_attention_v1,
          "Paged attention V1 (Metal implementation)",
          py::arg("out"),
          py::arg("query"),
          py::arg("key_cache"),
          py::arg("value_cache"),
          py::arg("block_tables"),
          py::arg("seq_lens"),
          py::arg("num_kv_heads"),
          py::arg("scale"),
          py::arg("block_size"),
          py::arg("max_seq_len"),
          py::arg("alibi_slopes") = py::none(),
          py::arg("kv_cache_scales") = py::none());

    m.def("paged_attention_v2",
          &vllm::metal::paged_attention_v2,
          "Paged attention V2 with partitioning (Metal implementation)",
          py::arg("out"),
          py::arg("exp_sums"),
          py::arg("max_logits"),
          py::arg("tmp_out"),
          py::arg("query"),
          py::arg("key_cache"),
          py::arg("value_cache"),
          py::arg("block_tables"),
          py::arg("seq_lens"),
          py::arg("num_kv_heads"),
          py::arg("scale"),
          py::arg("block_size"),
          py::arg("max_seq_len"),
          py::arg("alibi_slopes") = py::none(),
          py::arg("kv_cache_scales") = py::none());

    m.def("reshape_and_cache",
          &vllm::metal::reshape_and_cache,
          "Reshape and cache K/V tensors (Metal implementation)",
          py::arg("key"),
          py::arg("value"),
          py::arg("key_cache"),
          py::arg("value_cache"),
          py::arg("slot_mapping"));

    m.def("copy_blocks",
          &vllm::metal::copy_blocks,
          "Copy cache blocks (Metal implementation)",
          py::arg("key_cache"),
          py::arg("value_cache"),
          py::arg("src_to_dst"));

    m.def("swap_blocks",
          &vllm::metal::swap_blocks,
          "Swap cache blocks (Metal implementation)",
          py::arg("src_cache"),
          py::arg("dst_cache"),
          py::arg("src_to_dst"));
}
