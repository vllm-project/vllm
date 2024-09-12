#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"

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

TORCH_LIBRARY_FRAGMENT_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // Check if cutlass scaled_mm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_scaled_mm_supports_fp8", &cutlass_scaled_mm_supports_fp8);
  ops.def("machete_supported_schedules", &machete::supported_schedules);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
  ops.impl("paged_attention_v1", &paged_attention_v1);
  ops.impl("paged_attention_v2", &paged_attention_v2);
  ops.impl("silu_and_mul", &silu_and_mul);
  ops.impl("gelu_and_mul", &gelu_and_mul);
  ops.impl("gelu_tanh_and_mul", &gelu_tanh_and_mul);
  ops.impl("gelu_new", &gelu_new);
  ops.impl("gelu_fast", &gelu_fast);
  ops.impl("gelu_quick", &gelu_quick);
  ops.impl("advance_step_flashattn", &advance_step_flashattn);
  ops.impl("advance_step_flashinfer", &advance_step_flashinfer);
  ops.impl("rms_norm", &rms_norm);
  ops.impl("fused_add_rms_norm", &fused_add_rms_norm);
  ops.impl("rotary_embedding", &rotary_embedding);
  ops.impl("batched_rotary_embedding", &batched_rotary_embedding);
#ifndef USE_ROCM
  ops.impl("aqlm_gemm", &aqlm_gemm);
  ops.impl("aqlm_dequant", &aqlm_dequant);
  ops.impl("awq_gemm", &awq_gemm);
  ops.impl("awq_dequantize", &awq_dequantize);
  ops.impl("marlin_gemm", &marlin_gemm);
  ops.impl("gptq_marlin_24_gemm", &gptq_marlin_24_gemm);
  ops.impl("machete_gemm", &machete::gemm);
  ops.impl("machete_prepack_B", &machete::prepack_B);
  ops.impl("gptq_marlin_gemm", &gptq_marlin_gemm);
  ops.impl("gptq_marlin_repack", &gptq_marlin_repack);
  ops.impl("awq_marlin_repack", &awq_marlin_repack);
  ops.impl("ggml_dequantize", &ggml_dequantize);
  ops.impl("ggml_mul_mat_vec_a8", &ggml_mul_mat_vec_a8);
  ops.impl("ggml_mul_mat_a8", &ggml_mul_mat_a8);
  ops.impl("fp8_marlin_gemm", &fp8_marlin_gemm);
  ops.impl("marlin_qqq_gemm", &marlin_qqq_gemm);
  ops.impl("cutlass_scaled_mm", &cutlass_scaled_mm);
  ops.impl("cutlass_scaled_mm_azp", &cutlass_scaled_mm_azp);
  ops.impl("selective_scan_fwd", &selective_scan_fwd);
  ops.impl("causal_conv1d_update", &causal_conv1d_update);
  ops.impl("causal_conv1d_fwd", &causal_conv1d_fwd);
#endif
  ops.impl("gptq_gemm", &gptq_gemm);
  ops.impl("gptq_shuffle", &gptq_shuffle);
  ops.impl("static_scaled_fp8_quant", &static_scaled_fp8_quant);
  ops.impl("dynamic_scaled_fp8_quant", &dynamic_scaled_fp8_quant);
  ops.impl("dynamic_per_token_scaled_fp8_quant",
           &dynamic_per_token_scaled_fp8_quant);
  ops.impl("moe_align_block_size", &moe_align_block_size);
  ops.impl("static_scaled_int8_quant", &static_scaled_int8_quant);
  ops.impl("dynamic_scaled_int8_quant", &dynamic_scaled_int8_quant);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, Meta, ops) {
  ops.impl("gptq_marlin_repack", &gptq_marlin_repack_meta);
  ops.impl("awq_marlin_repack", &awq_marlin_repack_meta);
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cuda_utils), cuda_utils) {
  // Cuda utils

  // Gets the specified device attribute.
  cuda_utils.def("get_device_attribute(int attribute, int device_id) -> int");
  cuda_utils.impl("get_device_attribute", &get_device_attribute);

  // Gets the maximum shared memory per block device attribute.
  cuda_utils.def(
      "get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute",
                  &get_max_shared_memory_per_block_device_attribute);
}

#ifndef USE_ROCM
TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  // Custom all-reduce kernels
  custom_ar.def(
      "init_custom_ar(Tensor meta, Tensor rank_data, "
      "str[] handles, int[] offsets, int rank, "
      "bool full_nvlink) -> int");
  custom_ar.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);

  custom_ar.def(
      "should_custom_ar(Tensor inp, int max_size, int world_size, "
      "bool full_nvlink) -> bool");
  custom_ar.impl("should_custom_ar", torch::kCUDA, &should_custom_ar);

  custom_ar.def("all_reduce_reg(int fa, Tensor inp, Tensor! out) -> ()");
  custom_ar.impl("all_reduce_reg", torch::kCUDA, &all_reduce_reg);

  custom_ar.def(
      "all_reduce_unreg(int fa, Tensor inp, Tensor reg_buffer, Tensor! out) -> "
      "()");
  custom_ar.impl("all_reduce_unreg", torch::kCUDA, &all_reduce_unreg);

  custom_ar.def("dispose", &dispose);
  custom_ar.def("meta_size", &meta_size);

  custom_ar.def(
      "register_buffer(int fa, Tensor t, str[] handles, "
      "int[] offsets) -> ()");
  custom_ar.impl("register_buffer", torch::kCUDA, &register_buffer);

  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  custom_ar.def("register_graph_buffers", &register_graph_buffers);
}
#endif

TORCH_LIBRARY_IMPL_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), CUDA,
                          cache_ops) {
  cache_ops.impl("swap_blocks", &swap_blocks);
  cache_ops.impl("copy_blocks", &copy_blocks);
  cache_ops.impl("reshape_and_cache", &reshape_and_cache);
  cache_ops.impl("reshape_and_cache_flash", &reshape_and_cache_flash);
  cache_ops.impl("convert_fp8", &convert_fp8);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
