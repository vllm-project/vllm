#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
// #include "quantization/gptq_marlin/gptq_marlin.cuh"  //??
#include <torch/extension.h>

torch::Tensor gptq_marlin_repack_meta(torch::Tensor& b_q_weight,
                                      torch::Tensor& perm, int64_t size_k,
                                      int64_t size_n, int64_t num_bits);

// See
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

// where should these live?  near the implementations of the kernels?
namespace vllm::meta {

torch::Tensor aqlm_gemm(const torch::Tensor& input, const torch::Tensor& codes,
                        const torch::Tensor& codebooks,
                        const torch::Tensor& scales,
                        const torch::Tensor& codebook_partition_sizes,
                        const std::optional<torch::Tensor>& bias) {
  auto input_sizes = input.sizes();

  auto out_features = codes.size(0) * codebooks.size(2);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty(
      {flat_input.size(0), out_features},
      torch::TensorOptions().dtype(input.dtype()).device(input.device()));

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(-1);
  return flat_output.reshape(output_sizes);
}

torch::Tensor aqlm_dequant(const torch::Tensor& codes,
                           const torch::Tensor& codebooks,
                           const torch::Tensor& codebook_partition_sizes) {
  auto in_features = codes.size(1) * 8;
  auto out_features = codes.size(0);
  return torch::empty({out_features, in_features},
                      torch::TensorOptions()
                          .dtype(codebooks.dtype())
                          .device(codebooks.device()));
}

torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters) {
  int num_in_feats = _in_feats.size(0);
  auto options = torch::TensorOptions()
                     .dtype(_in_feats.dtype())
                     .device(_in_feats.device());
#if 0
  at::Tensor _out_feats =
      torch::empty({num_in_feats, _kernel.size(1) * 8}, options);
  return _out_feats.sum(0);
#else
  return torch::empty({_kernel.size(1) * 8}, options);
#endif
}

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy) {
  int in_c = _kernel.size(0);
  int qout_c = _kernel.size(1);
  int out_c = qout_c * 8;

  auto options = torch::TensorOptions()
                     .dtype(_scaling_factors.dtype())
                     .device(_scaling_factors.device());

  return torch::empty({in_c, out_c}, options);
}

torch::Tensor marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                          torch::Tensor& b_scales, torch::Tensor& workspace,
                          int64_t size_m, int64_t size_n, int64_t size_k) {
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  return torch::empty({size_m, size_n}, options);
}

torch::Tensor gptq_marlin_24_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                                  torch::Tensor& b_meta,
                                  torch::Tensor& b_scales,
                                  torch::Tensor& workspace, int64_t num_bits,
                                  int64_t size_m, int64_t size_n,
                                  int64_t size_k) {
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  return torch::empty({size_m, size_n}, options);
}

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& g_idx,
                               torch::Tensor& perm, torch::Tensor& workspace,
                               int64_t num_bits, int64_t size_m, int64_t size_n,
                               int64_t size_k, bool is_k_full) {
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  return torch::empty({size_m, size_n}, options);
}

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit) {
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  return torch::empty({a.size(0), b_q_weight.size(1)}, options);
}

}  // namespace vllm::meta

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
  // def(ops, "paged_attention_v2", &paged_attention_v2, {0});
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

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm(Tensor! out, Tensor input, Tensor weight, float epsilon) -> "
      "()");
  //  ops.def(torch::schema("rms_norm(Tensor out, Tensor input, Tensor weight,
  //  float epsilon) -> ()"), c10::AliasAnalysisKind::CONSERVATIVE);
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
  ops.impl("aqlm_gemm", torch::kMeta, &vllm::meta::aqlm_gemm);

  // Decompression method for AQLM.
  ops.def("aqlm_dequant", &aqlm_dequant);
  ops.impl("aqlm_dequant", torch::kCUDA, &aqlm_dequant);
  ops.impl("aqlm_dequant", torch::kMeta, &vllm::meta::aqlm_dequant);

  // Quantized GEMM for AWQ.
  ops.def("awq_gemm", &awq_gemm);
  ops.impl("awq_gemm", torch::kCUDA, &awq_gemm);
  ops.impl("awq_gemm", torch::kMeta, &vllm::meta::awq_gemm);

  // Dequantization for AWQ.
  ops.def("awq_dequantize", &awq_dequantize);
  ops.impl("awq_dequantize", torch::kCUDA, &awq_dequantize);
  ops.impl("awq_dequantize", torch::kMeta, &vllm::meta::awq_dequantize);

  // Marlin (Dense) Optimized Quantized GEMM for GPTQ.
  ops.def("marlin_gemm", &marlin_gemm);
  ops.impl("marlin_gemm", torch::kCUDA, &marlin_gemm);
  ops.impl("marlin_gemm", torch::kMeta, &vllm::meta::marlin_gemm);

  // Marlin_24 (Sparse) Optimized Quantized GEMM for GPTQ.
  ops.def("gptq_marlin_24_gemm", &gptq_marlin_24_gemm);
  ops.impl("gptq_marlin_24_gemm", torch::kCUDA, &gptq_marlin_24_gemm);
  ops.impl("gptq_marlin_24_gemm", torch::kMeta,
           &vllm::meta::gptq_marlin_24_gemm);

  // gptq_marlin Optimized Quantized GEMM for GPTQ.
  ops.def("gptq_marlin_gemm", &gptq_marlin_gemm);
  ops.impl("gptq_marlin_gemm", torch::kCUDA, &gptq_marlin_gemm);
  ops.impl("gptq_marlin_gemm", torch::kMeta, &vllm::meta::gptq_marlin_gemm);

  // gptq_marlin repack from GPTQ.
  ops.def("gptq_marlin_repack", &gptq_marlin_repack);
  ops.impl("gptq_marlin_repack", torch::kCUDA, &gptq_marlin_repack);
  ops.impl("gptq_marlin_repack", torch::kMeta, &gptq_marlin_repack_meta);

  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column
  // quantization.
  ops.def(
      "cutlass_scaled_mm_dq(Tensor! out, Tensor a,"
      "                     Tensor b, Tensor a_scales,"
      "                     Tensor b_scales) -> ()");
  ops.impl("cutlass_scaled_mm_dq", torch::kCUDA, &cutlass_scaled_mm_dq);
#endif

  // Quantized GEMM for GPTQ.
  ops.def("gptq_gemm", &gptq_gemm);
  ops.impl("gptq_gemm", torch::kCUDA, &gptq_gemm);
  ops.impl("gptq_gemm", torch::kMeta, &vllm::meta::gptq_gemm);

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
      "dynamic_scaled_fp8_quant(Tensor! out, Tensor input, Tensor scale) -> "
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
      "static_scaled_int8_quant(Tensor! out, Tensor input, float scale) -> ()");
  ops.impl("static_scaled_int8_quant", torch::kCUDA, &static_scaled_int8_quant);

  // Compute int8 quantized tensor and scaling factor
  ops.def("dynamic_scaled_int8_quant", &dynamic_scaled_int8_quant,
          "Compute int8 quantized tensor and scaling factor");
  ops.impl("dynamic_scaled_int8_quant", torch::kCUDA, &dynamic_scaled_int8_quant);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  cache_ops.def(
      "swap_blocks(Tensor! src, Tensor! dst, Tensor block_mapping) -> ()");
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
