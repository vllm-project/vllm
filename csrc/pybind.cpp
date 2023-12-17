#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // vLLM custom ops
  pybind11::module ops = m.def_submodule("ops", "vLLM custom operators");

  // Attention ops
  ops.def(
    "paged_attention_v1",
    &paged_attention_v1,
    "Compute the attention between an input query and the cached keys/values using PagedAttention.");
  ops.def(
    "paged_attention_v2",
    &paged_attention_v2,
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

  // Layernorm
  ops.def(
    "rms_norm",
    &rms_norm,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");

  ops.def(
    "fused_add_rms_norm",
    &fused_add_rms_norm,
    "In-place fused Add and RMS Normalization");

  // Rotary embedding
  ops.def(
    "rotary_embedding",
    &rotary_embedding,
    "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");

#ifndef USE_ROCM
  // Quantization ops
  ops.def("awq_gemm", &awq_gemm, "Quantized GEMM for AWQ");
#endif
  ops.def("gptq_gemm", &gptq_gemm, "Quantized GEMM for GPTQ");
  ops.def("gptq_shuffle", &gptq_shuffle, "Post processing for GPTQ");
  ops.def("squeezellm_gemm", &squeezellm_gemm, "Quantized GEMM for SqueezeLLM");

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
