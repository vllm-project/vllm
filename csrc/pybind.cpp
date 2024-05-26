#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include <torch/extension.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM custom ops

  // Attention ops
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  ops.def("paged_attention_v1(Tensor out, Tensor query, Tensor key_cache, "
          "Tensor value_cache, int num_kv_heads, float scale, Tensor "
          "block_tables, Tensor seq_lens, int block_size, int max_seq_len, "
          "Tensor? alibi_slopes, str kv_cache_dtype, float kv_scale, int tp_rank,"
          "int blocksparse_local_blocks, int blocksparse_vert_stride, "
          "int blocksparse_block_size, int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

  // PagedAttention V2.
  ops.def("paged_attention_v2(Tensor out, Tensor exp_sums, Tensor max_logits,"
          "Tensor tmp_out, Tensor query, Tensor key_cache, Tensor value_cache,"
          "int num_kv_heads, float scale, Tensor block_tables, Tensor seq_lens,"
          "int block_size, int max_seq_len, Tensor? alibi_slopes, "
          "str kv_cache_dtype, float kv_scale, int tp_rank, "
          "int blocksparse_local_blocks, int blocksparse_vert_stride,"
          "int blocksparse_block_size, int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);

  // Activation ops
  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor out, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor out, Tensor input) -> ()");
  ops.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor out, Tensor input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor out, Tensor input) -> ()");
  ops.impl("gelu_new", torch::kCUDA, &gelu_new);

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor out, Tensor input) -> ()");
  ops.impl("gelu_fast", torch::kCUDA, &gelu_fast);

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def("rms_norm(Tensor out, Tensor input, Tensor weight, float epsilon) -> ()");
  //ops.def(torch::schema("rms_norm(Tensor out, Tensor input, Tensor weight, float epsilon) -> ()"), c10::AliasAnalysisKind::CONSERVATIVE);
  ops.impl("rms_norm", torch::kCUDA, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.def("fused_add_rms_norm(Tensor input, Tensor residual, Tensor weight, float epsilon) -> ()");
  ops.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def("rotary_embedding(Tensor positions, Tensor query, Tensor key, int head_size, Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key
  // (supports multiple loras).
  ops.def("batched_rotary_embedding(Tensor positions, Tensor query, Tensor "
          "key, int head_size, Tensor cos_sin_cache, bool is_neox, int "
          "rot_dim, Tensor cos_sin_cache_offsets) -> ()");
  ops.impl("batched_rotary_embedding", torch::kCUDA, &batched_rotary_embedding);

  // Quantization ops
#ifndef USE_ROCM
  // Quantized GEMM for AQLM.
  ops.def("aqlm_gemm(Tensor input, Tensor codes, Tensor codebooks, Tensor scales, Tensor codebook_partition_sizes, Tensor? bias) -> Tensor");
  ops.impl("aqlm_gemm", torch::kCUDA, &aqlm_gemm);

  // Decompression method for AQLM.
  ops.def("aqlm_dequant(Tensor codes, Tensor codebooks, Tensor codebook_partition_sizes) -> Tensor");
  ops.impl("aqlm_dequant", torch::kCUDA, &aqlm_dequant);

  // Quantized GEMM for AWQ.
  ops.def("awq_gemm(Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, Tensor _zeros, int split_k_iters) -> Tensor");
  ops.impl("awq_gemm", torch::kCUDA, &awq_gemm);

  // Marlin (Dense) Optimized Quantized GEMM for GPTQ.
  ops.def("marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, Tensor workspace, int size_m, int size_n, int size_k) -> Tensor");
  ops.impl("marlin_gemm", torch::kCUDA, &marlin_gemm);

  // Marlin_24 (Sparse) Optimized Quantized GEMM for GPTQ.
  ops.def("gptq_marlin_24_gemm(Tensor a, Tensor b_q_weight, Tensor b_meta, Tensor b_scales, Tensor workspace, int num_bits, int size_m, int size_n, int size_k) -> Tensor");
  ops.impl("gptq_marlin_24_gemm", torch::kCUDA, &gptq_marlin_24_gemm);

  // gptq_marlin Optimized Quantized GEMM for GPTQ.
  ops.def("gptq_marlin_gemm(Tensor a, Tensor b_q_weight, Tensor b_scales, Tensor g_idx, Tensor perm, Tensor workspace, int num_bits, int size_m, int size_n, int size_k, bool is_k_full) -> Tensor");
  ops.impl("gptq_marlin_gemm", torch::kCUDA, &gptq_marlin_gemm);

  // gptq_marlin repack from GPTQ.
  ops.def("gptq_marlin_repack(Tensor b_q_weight, Tensor perm, int size_k, int size_n, int num_bits) -> Tensor");
  ops.impl("gptq_marlin_repack", torch::kCUDA, &gptq_marlin_repack);

  // Dequantization for AWQ.
  ops.def("awq_dequantize(Tensor _kernel, Tensor _scaling_factors, Tensor _zeros, int split_k_iters, int thx, int thy) -> Tensor");
  ops.impl("awq_dequantize", torch::kCUDA, &awq_dequantize);

  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column
  // quantization.
  ops.def("cutlass_scaled_mm_dq(Tensor out, Tensor a, Tensor b, Tensor a_scales, Tensor b_scales) -> ()");
  ops.impl("cutlass_scaled_mm_dq", torch::kCUDA, &cutlass_scaled_mm_dq);
#endif

  // Quantized GEMM for GPTQ.
  ops.def("gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, Tensor b_gptq_scales, Tensor b_g_idx, bool use_exllama, int bit) -> Tensor");
  ops.impl("gptq_gemm", torch::kCUDA, &gptq_gemm);

  // Post processing for GPTQ.
  ops.def("gptq_shuffle(Tensor q_weight, Tensor q_perm, int bit) -> ()");
  ops.impl("gptq_shuffle", torch::kCUDA, &gptq_shuffle);

  // Quantized GEMM for SqueezeLLM.
  ops.def("squeezellm_gemm(Tensor vec, Tensor mat, Tensor mul, Tensor lookup_table) -> ()");
  ops.impl("squeezellm_gemm", torch::kCUDA, &squeezellm_gemm);

  // Compute FP8 quantized tensor for given scaling factor.
  ops.def("static_scaled_fp8_quant(Tensor out, Tensor input, Tensor scale) -> ()");
  ops.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // Compute FP8 quantized tensor and scaling factor.
  ops.def("dynamic_scaled_fp8_quant(Tensor out, Tensor input, Tensor scale) -> ()");
  ops.impl("dynamic_scaled_fp8_quant", torch::kCUDA, &dynamic_scaled_fp8_quant);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  ops.def("moe_align_block_size(Tensor topk_ids, int num_experts, int block_size,"
          "Tensor sorted_token_ids, Tensor experts_ids, Tensor num_tokens_post_pad) -> ()");
  ops.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  // Compute int8 quantized tensor for given scaling factor.
  ops.def("static_scaled_int8_quant(Tensor out, Tensor input, float scale) -> ()");
  ops.impl("static_scaled_int8_quant", torch::kCUDA, &static_scaled_int8_quant);

  // Compute int8 quantized tensor and scaling factor
  ops.def("dynamic_scaled_int8_quant", &dynamic_scaled_int8_quant,
          "Compute int8 quantized tensor and scaling factor");
  ops.impl("dynamic_scaled_int8_quant", torch::kCUDA, &dynamic_scaled_int8_quant);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  // Cache ops
  pybind11::module cache_ops = m.def_submodule("cache_ops", "vLLM cache ops");
  cache_ops.def("swap_blocks", &swap_blocks,
                "Swap in (out) the cache blocks from src to dst");
  cache_ops.def("copy_blocks", &copy_blocks,
                "Copy the cache blocks from src to dst");
  cache_ops.def("reshape_and_cache", &reshape_and_cache,
                "Reshape the key and value tensors and cache them");
  cache_ops.def("reshape_and_cache_flash", &reshape_and_cache_flash,
                "Reshape the key and value tensors and cache them");
  cache_ops.def("convert_fp8", &convert_fp8,
                "Convert the key and value cache to fp8 data type");

  // Cuda utils
  pybind11::module cuda_utils =
      m.def_submodule("cuda_utils", "vLLM cuda utils");
  cuda_utils.def("get_device_attribute", &get_device_attribute,
                 "Gets the specified device attribute.");

  cuda_utils.def("get_max_shared_memory_per_block_device_attribute",
                 &get_max_shared_memory_per_block_device_attribute,
                 "Gets the maximum shared memory per block device attribute.");

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
