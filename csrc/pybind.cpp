#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "int8gemm/cublas/int8_gemm.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // vLLM custom ops
  pybind11::module ops = m.def_submodule("ops", "vLLM custom operators");

  // Attention ops
  ops.def(
    "paged_attention_v1",
    &paged_attention_v1,
    py::arg("out"),
    py::arg("query"),
    py::arg("key_cache"),
    py::arg("value_cache"),
    py::arg("head_mapping"),
    py::arg("scale"),
    py::arg("block_tables"),
    py::arg("context_lens"),
    py::arg("block_size"),
    py::arg("max_context_len"),
    py::arg("alibi_slopes"),
    py::arg("enable_quant") = false,
    py::arg("k_scale") = 1.0f,
    py::arg("k_zp") = 0.0f,
    py::arg("v_scale") = 1.0f,
    py::arg("v_zp") = 0.0f,
    "Compute the attention between an input query and the cached keys/values using PagedAttention.");
  ops.def(
    "paged_attention_v2",
    &paged_attention_v2,
    py::arg("out"),
    py::arg("exp_sums"),
    py::arg("max_logits"),
    py::arg("tmp_out"),
    py::arg("query"),
    py::arg("key_cache"),
    py::arg("value_cache"),
    py::arg("head_mapping"),
    py::arg("scale"),
    py::arg("block_tables"),
    py::arg("context_lens"),
    py::arg("block_size"),
    py::arg("max_context_len"),
    py::arg("alibi_slopes"),
    py::arg("enable_quant") = false,
    py::arg("k_scale") = 1.0f,
    py::arg("k_zp") = 0.0f,
    py::arg("v_scale") = 1.0f,
    py::arg("v_zp") = 0.0f,
    "PagedAttention V2.");

  // Activation ops
  ops.def(
    "silu_and_mul",
    &silu_and_mul,
    "Activation function used in SwiGLU.");
  ops.def(
    "gelu_new",
    &gelu_new,
    "GELU implementation used in GPT-2.");
  ops.def(
    "gelu_fast",
    &gelu_fast,
    "Approximate GELU implementation.");
  ops.def(
    "dequant_silu_and_mul_quant",
    py::overload_cast<
    torch::Tensor&,
    torch::Tensor&,
    float,
    float,
    float>(&dequant_silu_and_mul_quant),
    "Dequant input, apply silu act and quant output.");
  ops.def(
    "dequant_silu_and_mul_quant",
    py::overload_cast<
    torch::Tensor&,
    torch::Tensor&,
    float,
    float,
    torch::Tensor&,
    torch::Tensor&>(&dequant_silu_and_mul_quant),
    "Dequant input, apply silu act and quant output.");

  // Layernorm
  ops.def(
    "rms_norm",
    &rms_norm,
    py::arg("out"),
    py::arg("input"),
    py::arg("weight"),
    py::arg("epsilon"),
    py::arg("use_quant") = false,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");
  ops.def(
    "dequant_add_residual_rms_norm_quant",
    py::overload_cast<
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    float,
    float>(&dequant_add_residual_rms_norm_quant),
    "Add the dequanted result and residual, then use RMS norm and quant output.");
  ops.def(
    "dequant_add_residual_rms_norm_quant",
    py::overload_cast<
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    float>(&dequant_add_residual_rms_norm_quant),
    "Add the dequanted result and residual, then use RMS norm and quant output.");
  ops.def(
    "fused_add_rms_norm",
    &fused_add_rms_norm,
    py::arg("out"),
    py::arg("input"),
    py::arg("residual"),
    py::arg("weight"),
    py::arg("epsilon"),
    py::arg("use_quant") = false,
    "In-place fused Add and RMS Normalization");

  // Fused ops
  ops.def(
    "dequant_add_residual",
    py::overload_cast<
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    float>(&dequant_add_residual),
    "Add the dequanted result and residual.");
  ops.def(
    "dequant_add_residual",
    py::overload_cast<
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&>(&dequant_add_residual),
    "Add the per-token dequanted result and residual.");
  ops.def(
    "dequant",
    py::overload_cast<
    torch::Tensor&,
    torch::Tensor&,
    float>(&dequant),
    "Dequant.");
  ops.def(
    "dequant",
    py::overload_cast<
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&>(&dequant),
    "Per-token dequant.");
  ops.def(
    "quant",
    py::overload_cast<
    torch::Tensor&,
    torch::Tensor&,
    float>(&quant),
    "Quant.");
  ops.def(
    "quant",
    py::overload_cast<
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&>(
    &quant),
    "Per-token quant.");

  // Rotary embedding
  ops.def(
    "rotary_embedding",
    &rotary_embedding,
    py::arg("positions"),
    py::arg("query"),
    py::arg("key"),
    py::arg("head_size"),
    py::arg("cos_sin_cache"),
    py::arg("is_neox"),
    py::arg("query_out") = torch::empty({}),
    py::arg("key_out") = torch::empty({}),
    py::arg("use_dequant") = false,
    py::arg("query_scale") = 1.0f,
    py::arg("key_scale") = 1.0f,
    "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");

  // Quantization ops
  ops.def("awq_gemm", &awq_gemm, "Quantized GEMM for AWQ");
  ops.def("squeezellm_gemm", &squeezellm_gemm, "Quantized GEMM for SqueezeLLM");
  pybind11::class_<I8CUGEMM>(ops, "I8CUGEMM")
    .def(pybind11::init<>())
    .def("linear_a8_w8_o32", &I8CUGEMM::linear_a8_w8_o32)
    .def("linear_a8_w8_o8", &I8CUGEMM::linear_a8_w8_o8)
    .def("linear_a8_w8_o8_", &I8CUGEMM::linear_a8_w8_o8_)
    .def("linear_a8_w8_o32_", &I8CUGEMM::linear_a8_w8_o32_);

  // Cache ops
  pybind11::module cache_ops = m.def_submodule("cache_ops", "vLLM cache ops");
  cache_ops.def(
    "swap_blocks",
    &swap_blocks,
    "Swap in (out) the cache blocks from src to dst");
  cache_ops.def(
    "copy_blocks",
    &copy_blocks,
    "Copy the cache blocks from src to dst");
  cache_ops.def(
    "reshape_and_cache",
    &reshape_and_cache,
    py::arg("key"),
    py::arg("value"),
    py::arg("key_cache"),
    py::arg("value_cache"),
    py::arg("slot_mapping"),
    py::arg("use_quant") = false,
    py::arg("k_scale") = 1.0f,
    py::arg("k_zp") = 0.0f,
    py::arg("v_scale") = 1.0f,
    py::arg("v_zp") = 0.0f,
    "Reshape the key and value tensors and cache them");
  cache_ops.def(
    "gather_cached_kv",
    &gather_cached_kv,
    "Gather key and value from the cache into contiguous QKV tensors");

  // Cuda utils
  pybind11::module cuda_utils = m.def_submodule("cuda_utils", "vLLM cuda utils");
  cuda_utils.def(
    "get_device_attribute",
    &get_device_attribute,
    "Gets the specified device attribute.");
}
