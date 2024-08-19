#include "cache.h"
#include "ops.h"
#include "core/registration.h"

#include <torch/library.h>

std::string init_cpu_threads_env(const std::string& cpu_ids);

void int8_scaled_mm(torch::Tensor& c, const torch::Tensor& a,
                    const torch::Tensor& b, const torch::Tensor& a_scales,
                    const torch::Tensor& b_scales,
                    const c10::optional<torch::Tensor>& bias);

TORCH_LIBRARY_IMPL_EXPAND(_C, CPU, ops) {
  // vLLM custom ops

  // Attention ops
  // Compute the attention between an input query and the cached keys/values
  // using PagedAttention.
  ops.impl("paged_attention_v1", &paged_attention_v1);

  // PagedAttention V2.
  ops.impl("paged_attention_v2", &paged_attention_v2);

  // Activation ops

  // Activation function used in SwiGLU.
  ops.impl("silu_and_mul", &silu_and_mul);

  // Activation function used in GeGLU with `none` approximation.
  ops.impl("gelu_and_mul", &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  ops.impl("gelu_tanh_and_mul", &gelu_tanh_and_mul);

  // GELU implementation used in GPT-2.
  ops.impl("gelu_new", &gelu_new);

  // Approximate GELU implementation.
  ops.impl("gelu_fast", &gelu_fast);

  // Quick GELU implementation.
  ops.impl("gelu_quick", &gelu_quick);

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.impl("rms_norm", &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.impl("fused_add_rms_norm", &fused_add_rms_norm);

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.impl("rotary_embedding", &rotary_embedding);
}

TORCH_LIBRARY_IMPL_EXPAND(CONCAT(_C, _cache_ops), CPU,
                          cache_ops) {
  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  cache_ops.impl("swap_blocks", &swap_blocks);

  // Copy the cache blocks from src to dst.
  cache_ops.impl("copy_blocks", &copy_blocks);

  // Reshape the key and value tensors and cache them.
  cache_ops.impl("reshape_and_cache", &reshape_and_cache);
}

TORCH_LIBRARY_EXPAND(CONCAT(_C, _utils), utils) {
  // CPU utils
  utils.def("init_cpu_threads_env(str cpu_ids) -> str", &init_cpu_threads_env);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
