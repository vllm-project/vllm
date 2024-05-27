#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include <torch/extension.h>

using vllm::def;

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops

  // Attention ops
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  // ops.def("paged_attention_v1", &paged_attention_v1);
  def(ops, "paged_attention_v1", &paged_attention_v1, {0});
  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

  // PagedAttention V2.
  def(ops, "paged_attention_v2", &paged_attention_v2, {0});
  ops.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);

  // Activation ops
  // Activation function used in SwiGLU.
  def(ops, "silu_and_mul", &silu_and_mul, {0});
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  // Activation function used in GeGLU with `none` approximation.
  def(ops, "gelu_and_mul", &gelu_and_mul, {0});
  ops.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  def(ops, "gelu_tanh_and_mul", &gelu_tanh_and_mul, {0});
  ops.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  // GELU implementation used in GPT-2.
  def(ops, "gelu_new", &gelu_new, {0});
  ops.impl("gelu_new", torch::kCUDA, &gelu_new);

  // Approximate GELU implementation.
  def(ops, "gelu_fast", &gelu_fast, {0});
  ops.impl("gelu_fast", torch::kCUDA, &gelu_fast);

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  def(ops, "rms_norm", &rms_norm, {0});
  // ops.def("rms_norm",  &rms_norm);
  //  ops.def(torch::schema("rms_norm(Tensor out, Tensor input, Tensor weight,
  //  float epsilon) -> ()"), c10::AliasAnalysisKind::CONSERVATIVE);
  ops.impl("rms_norm", torch::kCUDA, &rms_norm);

  // In-place fused Add and RMS Normalization.
  def(ops, "fused_add_rms_norm", &fused_add_rms_norm, {0, 1});
  ops.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  def(ops, "rotary_embedding", &rotary_embedding, {1, 2});
  ops.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key
  // (supports multiple loras).
  def(ops, "batched_rotary_embedding", &batched_rotary_embedding, {1, 2});  // ?
  ops.impl("batched_rotary_embedding", torch::kCUDA, &batched_rotary_embedding);

  // Quantization ops
#ifndef USE_ROCM
  // Quantized GEMM for AQLM.
  ops.def("aqlm_gemm", &aqlm_gemm);
  ops.impl("aqlm_gemm", torch::kCUDA, &aqlm_gemm);

  // Decompression method for AQLM.
  ops.def("aqlm_dequant", &aqlm_dequant);
  ops.impl("aqlm_dequant", torch::kCUDA, &aqlm_dequant);

  // Quantized GEMM for AWQ.
  ops.def("awq_gemm", &awq_gemm);
  ops.impl("awq_gemm", torch::kCUDA, &awq_gemm);

  // Marlin (Dense) Optimized Quantized GEMM for GPTQ.
  ops.def("marlin_gemm", &marlin_gemm);
  ops.impl("marlin_gemm", torch::kCUDA, &marlin_gemm);

  // Marlin_24 (Sparse) Optimized Quantized GEMM for GPTQ.
  ops.def("gptq_marlin_24_gemm", &gptq_marlin_24_gemm);
  ops.impl("gptq_marlin_24_gemm", torch::kCUDA, &gptq_marlin_24_gemm);

  // gptq_marlin Optimized Quantized GEMM for GPTQ.
  ops.def("gptq_marlin_gemm", &gptq_marlin_gemm);
  ops.impl("gptq_marlin_gemm", torch::kCUDA, &gptq_marlin_gemm);

  // gptq_marlin repack from GPTQ.
  ops.def("gptq_marlin_repack", &gptq_marlin_repack);
  ops.impl("gptq_marlin_repack", torch::kCUDA, &gptq_marlin_repack);

  // Dequantization for AWQ.
  ops.def("awq_dequantize", &awq_dequantize);
  ops.impl("awq_dequantize", torch::kCUDA, &awq_dequantize);

  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column
  // quantization.
  def(ops, "cutlass_scaled_mm_dq", &cutlass_scaled_mm_dq, {0});
  ops.impl("cutlass_scaled_mm_dq", torch::kCUDA, &cutlass_scaled_mm_dq);
#endif

  // Quantized GEMM for GPTQ.
  ops.def("gptq_gemm", &gptq_gemm);
  ops.impl("gptq_gemm", torch::kCUDA, &gptq_gemm);

  // Post processing for GPTQ.
  def(ops, "gptq_shuffle", &gptq_shuffle, {0});
  ops.impl("gptq_shuffle", torch::kCUDA, &gptq_shuffle);

  // Quantized GEMM for SqueezeLLM.
  def(ops, "squeezellm_gemm", &squeezellm_gemm, {2});
  ops.impl("squeezellm_gemm", torch::kCUDA, &squeezellm_gemm);

  // Compute FP8 quantized tensor for given scaling factor.
  def(ops, "static_scaled_fp8_quant", &static_scaled_fp8_quant, {0});
  ops.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // Compute FP8 quantized tensor and scaling factor.
  def(ops, "dynamic_scaled_fp8_quant", &dynamic_scaled_fp8_quant, {0});
  ops.impl("dynamic_scaled_fp8_quant", torch::kCUDA, &dynamic_scaled_fp8_quant);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  def(ops, "moe_align_block_size", &moe_align_block_size, {3, 4, 5});
  ops.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  // Compute int8 quantized tensor for given scaling factor.
  def(ops, "static_scaled_int8_quant", &static_scaled_int8_quant, {0});
  ops.impl("static_scaled_int8_quant", torch::kCUDA, &static_scaled_int8_quant);

  // Compute int8 quantized tensor and scaling factor
  ops.def("dynamic_scaled_int8_quant", &dynamic_scaled_int8_quant,
          "Compute int8 quantized tensor and scaling factor");
  ops.impl("dynamic_scaled_int8_quant", torch::kCUDA, &dynamic_scaled_int8_quant);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  def(cache_ops, "swap_blocks", &swap_blocks, {0, 1});
  cache_ops.impl("swap_blocks", torch::kCUDA, &swap_blocks);

  // Copy the cache blocks from src to dst.
  def(cache_ops, "copy_blocks", &copy_blocks, {0, 1});
  cache_ops.impl("copy_blocks", torch::kCUDA, &copy_blocks);

  // Reshape the key and value tensors and cache them.
  def(cache_ops, "reshape_and_cache", &reshape_and_cache, {2, 3});  // 4?
  cache_ops.impl("reshape_and_cache", torch::kCUDA, &reshape_and_cache);

  // Reshape the key and value tensors and cache them.
  def(cache_ops, "reshape_and_cache_flash", &reshape_and_cache_flash,
      {2, 3});  // 4?
  cache_ops.impl("reshape_and_cache_flash", torch::kCUDA,
                 &reshape_and_cache_flash);

  // Convert the key and value cache to fp8 data type.
  def(cache_ops, "convert_fp8", &convert_fp8, {0});
  cache_ops.impl("convert_fp8", torch::kCUDA, &convert_fp8);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cuda_utils), cuda_utils) {
  // Cuda utils

  // Gets the specified device attribute.
  cuda_utils.def("get_device_attribute", &get_device_attribute);
  cuda_utils.impl("get_device_attribute", torch::kCUDA, &get_device_attribute);

  // Gets the maximum shared memory per block device attribute.
  cuda_utils.def("get_max_shared_memory_per_block_device_attribute",
                 &get_max_shared_memory_per_block_device_attribute);
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute",
                  torch::kCUDA,
                  &get_max_shared_memory_per_block_device_attribute);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifndef USE_ROCM
  // Custom all-reduce kernels
  pybind11::module custom_ar = m.def_submodule("custom_ar", "custom allreduce");
  custom_ar.def("init_custom_ar", &init_custom_ar, "init_custom_ar");
  custom_ar.def("should_custom_ar", &should_custom_ar, "should_custom_ar");
  custom_ar.def("all_reduce_reg", &all_reduce_reg, "all_reduce_reg");
  custom_ar.def("all_reduce_unreg", &all_reduce_unreg, "all_reduce_unreg");
  custom_ar.def("dispose", &dispose, "dispose");
  custom_ar.def("meta_size", &meta_size, "meta_size");
  custom_ar.def("register_buffer", &register_buffer, "register_buffer");
  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta,
                "get_graph_buffer_ipc_meta");
  custom_ar.def("register_graph_buffers", &register_graph_buffers,
                "register_graph_buffers");
#endif
}
