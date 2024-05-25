#include "cache.h"
#include "ops.h"
#include <torch/extension.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops

  // Attention ops
  // Compute the attention between an input query and the cached keys/values
  //using PagedAttention.
  ops.impl("paged_attention_v1", torch::kCPU, &paged_attention_v1);
  // PagedAttention V2.
  ops.impl("paged_attention_v2", torch::kCPU, &paged_attention_v2);

  // Activation ops
  // Activation function used in SwiGLU.
  ops.impl("silu_and_mul", torch::kCPU, &silu_and_mul);
  // Activation function used in GeGLU with `none` approximation.
  ops.impl("gelu_and_mul", torch::kCPU, &gelu_and_mul);
  // Activation function used in GeGLU with `tanh` approximation.
  ops.impl("gelu_tanh_and_mul", torch::kCPU, &gelu_tanh_and_mul);
  // GELU implementation used in GPT-2.
  ops.impl("gelu_new", torch::kCPU, &gelu_new);
  // Approximate GELU implementation.
  ops.impl("gelu_fast", torch::kCPU, &gelu_fast);

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.impl("rms_norm", torch::kCPU, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.impl("fused_add_rms_norm", torch::kCPU, &fused_add_rms_norm);

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.impl("rotary_embedding", torch::kCPU, &rotary_embedding);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  // Cache ops
  pybind11::module cache_ops = m.impl_submodule("cache_ops", "vLLM cache ops");
  cache_ops.def("swap_blocks", &swap_blocks,
                "Swap in (out) the cache blocks from src to dst");
  cache_ops.def("copy_blocks", &copy_blocks,
                "Copy the cache blocks from src to dst");
  cache_ops.def("reshape_and_cache", &reshape_and_cache,
                "Reshape the key and value tensors and cache them");
}
