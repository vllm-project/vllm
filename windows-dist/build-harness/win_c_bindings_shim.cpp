// _C bindings for ROCm Windows — registers ops from compiled kernel files.
// Mirrors csrc/libtorch_stable/torch_bindings.cpp but only for sources we build.
#include "ops.h"
#include "cuda_utils.h"
#include "core/registration.h"
#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
  ops.def("rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) -> ()");
  ops.def("fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, float epsilon) -> ()");
  ops.def(
      "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");
  ops.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, Tensor scale, float epsilon) -> ()");
  ops.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");
  ops.def(
      "rms_norm_per_block_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual, int group_size, "
      "bool is_scale_transposed) -> ()");
  ops.def(
      "silu_and_mul_per_block_quant("
      "Tensor! out, Tensor input, Tensor! scales, int group_size, "
      "Tensor? scale_ub=None, bool is_scale_transposed=False) -> ()");
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox, int "
      "rope_dim_offset=0, bool inverse=False) -> ()");
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");
  ops.def(
      "silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit, "
      "float alpha=1.0, float beta=0.0) -> ()");
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  ops.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");
  ops.def(
      "static_scaled_int8_quant(Tensor! result, Tensor input, Tensor scale,"
      "Tensor? azp) -> ()");
  ops.def(
      "dynamic_scaled_int8_quant(Tensor! result, Tensor input, Tensor! scale, "
      "Tensor!? azp) -> ()");
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale, "
      "int[]? group_shape=None) -> ()");
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) -> ()");
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> ()");
  ops.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, "
      "Tensor b_gptq_scales, Tensor b_g_idx, bool use_exllama, bool "
      "use_v2_format, int bit) -> Tensor");
  ops.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");
  ops.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, int group_size, float eps, float fp8_min, float fp8_max, "
      "bool scale_ue8m0, bool dummy_is_scale_transposed, "
      "bool dummy_is_tma_aligned) -> ()");
  ops.def(
      "per_token_group_fp8_quant_packed(Tensor input, Tensor! output_q, "
      "Tensor! output_s_packed, int group_size, float eps, float fp8_min, "
      "float fp8_max) -> ()");
  ops.def(
      "per_token_group_quant_int8(Tensor input, Tensor! output_q, "
      "Tensor! output_s, int group_size, float eps, "
      "float int8_min, float int8_max) -> ()");

  // Sampler
  ops.def("apply_repetition_penalties_(Tensor! logits, Tensor prompt_mask, "
          "Tensor output_mask, Tensor repetition_penalties) -> ()");
  ops.def("top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, "
          "Tensor! indices, int numRows, int stride0, int stride1, int topK) -> ()");
  ops.def("top_k_per_row_decode(Tensor logits, int next_n, Tensor seq_lens, "
          "Tensor! indices, int numRows, int stride0, int stride1, int topK) -> ()");
  ops.def("persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
          "Tensor workspace, int k, int max_seq_len) -> ()");

  // Weak ref tensor
  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("permute_cols", TORCH_BOX(&permute_cols));
  ops.impl("rms_norm", TORCH_BOX(&rms_norm));
  ops.impl("fused_add_rms_norm", TORCH_BOX(&fused_add_rms_norm));
  ops.impl("rms_norm_static_fp8_quant", TORCH_BOX(&rms_norm_static_fp8_quant));
  ops.impl("fused_add_rms_norm_static_fp8_quant", TORCH_BOX(&fused_add_rms_norm_static_fp8_quant));
  ops.impl("rms_norm_dynamic_per_token_quant", TORCH_BOX(&rms_norm_dynamic_per_token_quant));
  ops.impl("rms_norm_per_block_quant", TORCH_BOX(&rms_norm_per_block_quant));
  ops.impl("silu_and_mul_per_block_quant", TORCH_BOX(&silu_and_mul_per_block_quant));
  ops.impl("rotary_embedding", TORCH_BOX(&rotary_embedding));
  ops.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
  ops.impl("mul_and_silu", TORCH_BOX(&mul_and_silu));
  ops.impl("gelu_and_mul", TORCH_BOX(&gelu_and_mul));
  ops.impl("gelu_tanh_and_mul", TORCH_BOX(&gelu_tanh_and_mul));
  ops.impl("fatrelu_and_mul", TORCH_BOX(&fatrelu_and_mul));
  ops.impl("swigluoai_and_mul", TORCH_BOX(&swigluoai_and_mul));
  ops.impl("gelu_new", TORCH_BOX(&gelu_new));
  ops.impl("gelu_fast", TORCH_BOX(&gelu_fast));
  ops.impl("gelu_quick", TORCH_BOX(&gelu_quick));
  ops.impl("silu_and_mul_with_clamp", TORCH_BOX(&silu_and_mul_clamp));
  ops.impl("static_scaled_int8_quant", TORCH_BOX(&static_scaled_int8_quant));
  ops.impl("dynamic_scaled_int8_quant", TORCH_BOX(&dynamic_scaled_int8_quant));
  ops.impl("static_scaled_fp8_quant", TORCH_BOX(&static_scaled_fp8_quant));
  ops.impl("dynamic_scaled_fp8_quant", TORCH_BOX(&dynamic_scaled_fp8_quant));
  ops.impl("dynamic_per_token_scaled_fp8_quant", TORCH_BOX(&dynamic_per_token_scaled_fp8_quant));
  ops.impl("gptq_gemm", TORCH_BOX(&gptq_gemm));
  ops.impl("gptq_shuffle", TORCH_BOX(&gptq_shuffle));
  ops.impl("per_token_group_fp8_quant", TORCH_BOX(&per_token_group_quant_fp8));
  ops.impl("per_token_group_fp8_quant_packed", TORCH_BOX(&per_token_group_quant_8bit_packed));
  ops.impl("per_token_group_quant_int8", TORCH_BOX(&per_token_group_quant_int8));
  ops.impl("apply_repetition_penalties_", TORCH_BOX(&apply_repetition_penalties_));
  ops.impl("top_k_per_row_prefill", TORCH_BOX(&top_k_per_row_prefill));
  ops.impl("top_k_per_row_decode", TORCH_BOX(&top_k_per_row_decode));
  ops.impl("persistent_topk", TORCH_BOX(&persistent_topk));
  ops.impl("weak_ref_tensor", TORCH_BOX(&weak_ref_tensor));
}

STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, ops) {
}

// Cache ops
STABLE_TORCH_LIBRARY_FRAGMENT(_C_cache_ops, ops) {
  ops.def("swap_blocks(Tensor src, Tensor! dst, int block_size_in_bytes, Tensor block_mapping) -> ()");
  ops.def("reshape_and_cache(Tensor key, Tensor value, Tensor! key_cache, Tensor! value_cache, Tensor slot_mapping, str kv_cache_dtype, Tensor k_scale, Tensor v_scale) -> ()");
  ops.def("reshape_and_cache_flash(Tensor key, Tensor value, Tensor! key_cache, Tensor! value_cache, Tensor slot_mapping, str kv_cache_dtype, Tensor k_scale, Tensor v_scale) -> ()");
  ops.def("convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, str kv_cache_dtype) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_cache_ops, CUDA, ops) {
  ops.impl("swap_blocks", TORCH_BOX(&swap_blocks));
  ops.impl("reshape_and_cache", TORCH_BOX(&reshape_and_cache));
  ops.impl("reshape_and_cache_flash", TORCH_BOX(&reshape_and_cache_flash));
  ops.impl("convert_fp8", TORCH_BOX(&convert_fp8));
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C_custom_ar, custom_ar) {
  custom_ar.def("init_custom_ar(int[] ipc_tensors, Tensor rank_data, int rank, bool fully_connected) -> int");
  custom_ar.def("all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, int reg_buffer_sz_bytes) -> ()");
  custom_ar.def("dispose(int fa) -> ()");
  custom_ar.def("meta_size() -> int");
  custom_ar.def("register_buffer(int fa, int[] ipc_tensors) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CUDA, custom_ar) {
  custom_ar.impl("init_custom_ar", TORCH_BOX(&init_custom_ar));
  custom_ar.impl("all_reduce", TORCH_BOX(&all_reduce));
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CompositeExplicitAutograd, custom_ar) {
  custom_ar.impl("dispose", TORCH_BOX(&dispose));
  custom_ar.impl("meta_size", TORCH_BOX(&meta_size));
  custom_ar.impl("register_buffer", TORCH_BOX(&register_buffer));
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C_cuda_utils, cuda_utils) {
  cuda_utils.def("get_device_attribute(int attribute, int device_id) -> int");
  cuda_utils.def("get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(_C_cuda_utils, CompositeExplicitAutograd, cuda_utils) {
  cuda_utils.impl("get_device_attribute", TORCH_BOX(&get_device_attribute));
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute", TORCH_BOX(&get_max_shared_memory_per_block_device_attribute));
}

REGISTER_EXTENSION(_C)
