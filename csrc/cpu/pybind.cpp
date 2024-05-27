#include "cache.h"
#include "ops.h"
#include <torch/extension.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops

  // Attention ops
  // Compute the attention between an input query and the cached keys/values
  // using PagedAttention.
  ops.def("paged_attention_v1", &paged_attention_v1);
  ops.impl("paged_attention_v1", torch::kCPU, &paged_attention_v1);

  // PagedAttention V2.
  ops.def("paged_attention_v2", &paged_attention_v2);
  ops.impl("paged_attention_v2", torch::kCPU, &paged_attention_v2);

  // Activation ops

  // Activation function used in SwiGLU.
  ops.def("silu_and_mul", &silu_and_mul);
  ops.impl("silu_and_mul", torch::kCPU, &silu_and_mul);

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul", &gelu_and_mul);
  ops.impl("gelu_and_mul", torch::kCPU, &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul", &gelu_tanh_and_mul);
  ops.impl("gelu_tanh_and_mul", torch::kCPU, &gelu_tanh_and_mul);

  // GELU implementation used in GPT-2.
  ops.def("gelu_new", &gelu_new);
  ops.impl("gelu_new", torch::kCPU, &gelu_new);

  // Approximate GELU implementation.
  ops.def("gelu_fast", &gelu_fast);
  ops.impl("gelu_fast", torch::kCPU, &gelu_fast);

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def("rms_norm", &rms_norm);
  ops.impl("rms_norm", torch::kCPU, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.def("fused_add_rms_norm", &fused_add_rms_norm);
  ops.impl("fused_add_rms_norm", torch::kCPU, &fused_add_rms_norm);

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def("rotary_embedding", &rotary_embedding);
  ops.impl("rotary_embedding", torch::kCPU, &rotary_embedding);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  cache_ops.def("swap_blocks", &swap_blocks);
  cache_ops.impl("swap_blocks", torch::kCPU, &swap_blocks);

  // Copy the cache blocks from src to dst.
  cache_ops.def("copy_blocks", &copy_blocks);
  cache_ops.impl("copy_blocks", torch::kCPU, &copy_blocks);

  // Reshape the key and value tensors and cache them.
  cache_ops.def("reshape_and_cache", &reshape_and_cache);
  cache_ops.impl("reshape_and_cache", torch::kCPU, &reshape_and_cache);
}
