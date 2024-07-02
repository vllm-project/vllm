#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "registration.h"

#include <torch/library.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops

  // Attention ops
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  ops.def(
      "paged_attention_v1("
      "    Tensor! out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, float kv_scale, int tp_rank,"
      "    int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

  // PagedAttention V2.
  ops.def(
      "paged_attention_v2("
      "    Tensor! out, Tensor exp_sums, Tensor max_logits,"
      "    Tensor tmp_out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, float kv_scale, int tp_rank,"
      "    int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);

  // Activation ops
  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_new", torch::kCUDA, &gelu_new);

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_fast", torch::kCUDA, &gelu_fast);

  // Quick GELU implementation.
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_quick", torch::kCUDA, &gelu_quick);

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm(Tensor! out, Tensor input, Tensor weight, float epsilon) -> "
      "()");
  ops.impl("rms_norm", torch::kCUDA, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  ops.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor! key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key
  // (supports multiple loras).
  ops.def(
      "batched_rotary_embedding(Tensor positions, Tensor! query,"
      "                         Tensor! key, int head_size,"
      "                         Tensor cos_sin_cache, bool is_neox,"
      "                         int rot_dim,"
      "                         Tensor cos_sin_cache_offsets) -> ()");
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

  // Dequantization for AWQ.
  ops.def("awq_dequantize", &awq_dequantize);
  ops.impl("awq_dequantize", torch::kCUDA, &awq_dequantize);

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

  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column
  // quantization.
  ops.def(
      "cutlass_scaled_mm(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_mm", torch::kCUDA, &cutlass_scaled_mm);

  // Check if cutlass scaled_mm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_scaled_mm_supports_fp8", &cutlass_scaled_mm_supports_fp8);
  ops.impl("cutlass_scaled_mm_supports_fp8", torch::kCUDA,
           &cutlass_scaled_mm_supports_fp8);
#endif

  // Quantized GEMM for GPTQ.
  ops.def("gptq_gemm", &gptq_gemm);
  ops.impl("gptq_gemm", torch::kCUDA, &gptq_gemm);

  // Post processing for GPTQ.
  ops.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");
  ops.impl("gptq_shuffle", torch::kCUDA, &gptq_shuffle);

  // Quantized GEMM for SqueezeLLM.
  ops.def(
      "squeezellm_gemm(Tensor vec, Tensor mat, Tensor! mul, Tensor "
      "lookup_table) -> ()");
  ops.impl("squeezellm_gemm", torch::kCUDA, &squeezellm_gemm);

  // Compute FP8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_fp8_quant(Tensor! out, Tensor input, Tensor scale) -> ()");
  ops.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // Compute FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! out, Tensor input, Tensor! scale) -> "
      "()");
  ops.impl("dynamic_scaled_fp8_quant", torch::kCUDA, &dynamic_scaled_fp8_quant);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  ops.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  ops.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  // Compute int8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_int8_quant(Tensor! out, Tensor input, Tensor scale) -> "
      "()");
  ops.impl("static_scaled_int8_quant", torch::kCUDA, &static_scaled_int8_quant);

  // Compute int8 quantized tensor and scaling factor
  ops.def(
      "dynamic_scaled_int8_quant(Tensor! out, Tensor input, Tensor! scale) -> "
      "()");
  ops.impl("dynamic_scaled_int8_quant", torch::kCUDA,
           &dynamic_scaled_int8_quant);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  cache_ops.def(
      "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
  cache_ops.impl("swap_blocks", torch::kCUDA, &swap_blocks);

  // Copy the cache blocks from src to dst.
  cache_ops.def(
      "copy_blocks(Tensor[]! key_caches, Tensor[]! value_caches, Tensor "
      "block_mapping) -> ()");
  cache_ops.impl("copy_blocks", torch::kCUDA, &copy_blocks);

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  float kv_scale) -> ()");
  cache_ops.impl("reshape_and_cache", torch::kCUDA, &reshape_and_cache);

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype) -> ()");
  cache_ops.impl("reshape_and_cache_flash", torch::kCUDA,
                 &reshape_and_cache_flash);

  // Convert the key and value cache to fp8 data type.
  cache_ops.def(
      "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, str "
      "kv_cache_dtype) -> ()");
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

#ifndef USE_ROCM
TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  // Custom all-reduce kernels
  custom_ar.def("init_custom_ar", &init_custom_ar);
  custom_ar.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);

  custom_ar.def("should_custom_ar", &should_custom_ar);
  custom_ar.impl("should_custom_ar", torch::kCUDA, &should_custom_ar);

  custom_ar.def("all_reduce_reg(int fa, Tensor inp, Tensor! out) -> ()");
  custom_ar.impl("all_reduce_reg", torch::kCUDA, &all_reduce_reg);

  custom_ar.def(
      "all_reduce_unreg(int fa, Tensor inp, Tensor reg_buffer, Tensor! out) -> "
      "()");
  custom_ar.impl("all_reduce_unreg", torch::kCUDA, &all_reduce_unreg);

  custom_ar.def("dispose", &dispose);
  custom_ar.impl("dispose", torch::kCPU, &dispose);

  custom_ar.def("meta_size", &meta_size);
  custom_ar.impl("meta_size", torch::kCPU, &meta_size);

  custom_ar.def("register_buffer", &register_buffer);
  custom_ar.impl("register_buffer", torch::kCUDA, &register_buffer);

  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  custom_ar.impl("get_graph_buffer_ipc_meta", torch::kCPU,
                 &get_graph_buffer_ipc_meta);

  custom_ar.def("register_graph_buffers", &register_graph_buffers);
  custom_ar.impl("register_graph_buffers", torch::kCPU,
                 &register_graph_buffers);
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
