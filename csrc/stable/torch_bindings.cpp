#include "ops.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

// Register ops using STABLE_TORCH_LIBRARY for stable ABI compatibility.
// Note: We register under namespace "_C" so ops are accessible as
// torch.ops._C.<op_name> for compatibility with existing code.
STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  // Activation ops - gated activations (input: [..., 2*d] -> output: [..., d])
  m.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  m.def("mul_and_silu(Tensor! out, Tensor input) -> ()");
  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  m.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");
  m.def(
      "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float "
      "limit=7.0) -> ()");

  // Activation ops - element-wise (input: [..., d] -> output: [..., d])
  m.def("gelu_new(Tensor! out, Tensor input) -> ()");
  m.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  m.def("gelu_quick(Tensor! out, Tensor input) -> ()");

  // Utility ops
  m.def("get_cuda_view_from_cpu_tensor(Tensor! cpu_tensor) -> Tensor");
#ifndef USE_ROCM
  m.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
#endif

  // Layernorm
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  m.def(
      "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> "
      "()");

  // In-place fused Add and RMS Normalization.
  m.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");

  // Layernorm-quant
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  m.def(
      "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");

  // In-place fused Add and RMS Normalization.
  m.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, Tensor scale, float epsilon) -> ()");

  // Fused Layernorm + Quant kernels
  m.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");

  // Fused Layernorm + Block quant kernels
  m.def(
      "rms_norm_per_block_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual, int group_size, "
      "bool is_scale_transposed) -> ()");

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  m.def(
      "rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, "
      "int head_size, Tensor cos_sin_cache, bool is_neox) -> ()");

  // Function for fused QK Norm and RoPE
  m.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, int num_heads_k, "
      "int num_heads_v, int head_dim, float eps, Tensor q_weight, "
      "Tensor k_weight, Tensor cos_sin_cache, bool is_neox, "
      "Tensor position_ids) -> ()");

  // Apply repetition penalties to logits in-place
  m.def(
      "apply_repetition_penalties_(Tensor! logits, Tensor prompt_mask, "
      "Tensor output_mask, Tensor repetition_penalties) -> ()");

  // Optimized top-k per row operation
  m.def(
      "top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, "
      "Tensor! indices, int numRows, int stride0, int stride1, int topK) -> "
      "()");
  m.def(
      "top_k_per_row_decode(Tensor logits, int next_n, Tensor seqLens, "
      "Tensor! indices, int numRows, int stride0, int stride1, int topK) -> "
      "()");

  // Quantized activation
  m.def("silu_and_mul_quant(Tensor! result, Tensor input, Tensor scale) -> ()");

  m.def(
      "persistent_masked_m_silu_mul_quant(Tensor input, Tensor counts, "
      "Tensor! y_q, Tensor! y_s, bool use_ue8m0) -> ()");

  // Compute FP8 quantized tensor for given scaling factor.
  // Supports per-tensor, per-channel, per-token, and arbitrary 2D group
  // scaling. Optional group_m/group_n specify the group shape explicitly;
  // required for 1D scales to disambiguate per-channel vs per-token.
  m.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale, "
      "int[]? group_shape=None) -> ()");
  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  m.def(
      "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) "
      "-> ()");
  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  m.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> ()");

  // Compute int8 quantized tensor for given scaling factor.
  m.def(
      "static_scaled_int8_quant(Tensor! result, Tensor input, Tensor scale, "
      "Tensor? azp) -> ()");
  // Compute int8 quantized tensor and scaling factor.
  m.def(
      "dynamic_scaled_int8_quant(Tensor! result, Tensor input, Tensor! scale, "
      "Tensor!? azp) -> ()");

#ifndef USE_ROCM
  // Compute per-token-group FP8 quantized tensor and scaling factor.
  m.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, "
      "Tensor! output_s, int group_size, float eps, float fp8_min, "
      "float fp8_max, bool scale_ue8m0) -> ()");
  // Compute per-token-group 8-bit quantized tensor and UE8M0-packed,
  // TMA-aligned scales for DeepGEMM.
  m.def(
      "per_token_group_fp8_quant_packed(Tensor input, Tensor! output_q, "
      "Tensor! output_s_packed, int group_size, float eps, float min_8bit, "
      "float max_8bit) -> ()");
  // Compute per-token-group INT8 quantized tensor and scaling factor.
  m.def(
      "per_token_group_quant_int8(Tensor input, Tensor! output_q, "
      "Tensor! output_s, int group_size, float eps, float int8_min, "
      "float int8_max) -> ()");
#endif

  // Attention operations
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  m.def(
      "paged_attention_v1(Tensor! out, Tensor query, Tensor key_cache, "
      "Tensor value_cache, int num_kv_heads, float scale, "
      "Tensor block_tables, Tensor seq_lens, int block_size, int max_seq_len, "
      "Tensor? alibi_slopes, str kv_cache_dtype, Tensor k_scale, "
      "Tensor v_scale, int tp_rank, int blocksparse_local_blocks, "
      "int blocksparse_vert_stride, int blocksparse_block_size, "
      "int blocksparse_head_sliding_step) -> ()");
  // PagedAttention V2.
  m.def(
      "paged_attention_v2(Tensor! out, Tensor! exp_sums, Tensor! max_logits, "
      "Tensor! tmp_out, Tensor query, Tensor key_cache, Tensor value_cache, "
      "int num_kv_heads, float scale, Tensor block_tables, Tensor seq_lens, "
      "int block_size, int max_seq_len, Tensor? alibi_slopes, "
      "str kv_cache_dtype, Tensor k_scale, Tensor v_scale, int tp_rank, "
      "int blocksparse_local_blocks, int blocksparse_vert_stride, "
      "int blocksparse_block_size, int blocksparse_head_sliding_step) -> ()");
  // Merge attn states
  // Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
  // can be used to combine partial attention results (in the split-KV case)
  m.def(
      "merge_attn_states(Tensor! output, Tensor? output_lse, "
      "Tensor prefix_output, Tensor prefix_lse, "
      "Tensor suffix_output, Tensor suffix_lse) -> ()");
#ifndef USE_ROCM
  m.def(
      "convert_vertical_slash_indexes(Tensor! block_count, "
      "Tensor! block_offset, Tensor! column_count, Tensor! column_index, "
      "Tensor q_seqlens, Tensor kv_seqlens, Tensor vertical_indexes, "
      "Tensor slash_indexes, int context_size, int block_size_M, "
      "int block_size_N, bool causal) -> ()");
  m.def(
      "convert_vertical_slash_indexes_mergehead(Tensor! block_count, "
      "Tensor! block_offset, Tensor! column_count, Tensor! column_index, "
      "Tensor q_seqlens, Tensor kv_seqlens, Tensor vertical_indexes, "
      "Tensor slash_indexes, Tensor vertical_indices_count, "
      "Tensor slash_indices_count, int context_size, int block_size_M, "
      "int block_size_N, bool causal) -> ()");
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  // Gated activations
  m.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
  m.impl("mul_and_silu", TORCH_BOX(&mul_and_silu));
  m.impl("gelu_and_mul", TORCH_BOX(&gelu_and_mul));
  m.impl("gelu_tanh_and_mul", TORCH_BOX(&gelu_tanh_and_mul));
  m.impl("fatrelu_and_mul", TORCH_BOX(&fatrelu_and_mul));
  m.impl("swigluoai_and_mul", TORCH_BOX(&swigluoai_and_mul));

  // Element-wise activations
  m.impl("gelu_new", TORCH_BOX(&gelu_new));
  m.impl("gelu_fast", TORCH_BOX(&gelu_fast));
  m.impl("gelu_quick", TORCH_BOX(&gelu_quick));

  // Layernorm ops
  m.impl("rms_norm", TORCH_BOX(&rms_norm));
  m.impl("fused_add_rms_norm", TORCH_BOX(&fused_add_rms_norm));

  // Layernorm + quantization ops
  m.impl("rms_norm_static_fp8_quant", TORCH_BOX(&rms_norm_static_fp8_quant));
  m.impl("fused_add_rms_norm_static_fp8_quant",
         TORCH_BOX(&fused_add_rms_norm_static_fp8_quant));

  // Fused layernorm + dynamic quantization ops
  m.impl("rms_norm_dynamic_per_token_quant",
         TORCH_BOX(&rms_norm_dynamic_per_token_quant));
  m.impl("rms_norm_per_block_quant", TORCH_BOX(&rms_norm_per_block_quant));

  // Positional encoding
  m.impl("rotary_embedding", TORCH_BOX(&rotary_embedding));
  m.impl("fused_qk_norm_rope", TORCH_BOX(&fused_qk_norm_rope));

  // Sampler
  m.impl("apply_repetition_penalties_",
         TORCH_BOX(&apply_repetition_penalties_));
  m.impl("top_k_per_row_prefill", TORCH_BOX(&top_k_per_row_prefill));
  m.impl("top_k_per_row_decode", TORCH_BOX(&top_k_per_row_decode));

#ifndef USE_ROCM
  // Utility ops
  m.impl("permute_cols", TORCH_BOX(&permute_cols));
#endif

  // Quantized activation
  m.impl("silu_and_mul_quant", TORCH_BOX(&silu_and_mul_quant));

  m.impl("persistent_masked_m_silu_mul_quant",
         TORCH_BOX(&persistent_masked_m_silu_mul_quant));

  // FP8 quantization
  m.impl("static_scaled_fp8_quant", TORCH_BOX(&static_scaled_fp8_quant));
  m.impl("dynamic_scaled_fp8_quant", TORCH_BOX(&dynamic_scaled_fp8_quant));
  m.impl("dynamic_per_token_scaled_fp8_quant",
         TORCH_BOX(&dynamic_per_token_scaled_fp8_quant));

  // INT8 quantization
  m.impl("static_scaled_int8_quant", TORCH_BOX(&static_scaled_int8_quant));
  m.impl("dynamic_scaled_int8_quant", TORCH_BOX(&dynamic_scaled_int8_quant));

#ifndef USE_ROCM
  // Per-token group quantization
  m.impl("per_token_group_fp8_quant", TORCH_BOX(&per_token_group_quant_fp8));
  m.impl("per_token_group_fp8_quant_packed",
         TORCH_BOX(&per_token_group_quant_8bit_packed));
  m.impl("per_token_group_quant_int8", TORCH_BOX(&per_token_group_quant_int8));
#endif

  // Attention operations
  m.impl("paged_attention_v1", TORCH_BOX(&paged_attention_v1));
  m.impl("paged_attention_v2", TORCH_BOX(&paged_attention_v2));
  m.impl("merge_attn_states", TORCH_BOX(&merge_attn_states));
#ifndef USE_ROCM
  m.impl("convert_vertical_slash_indexes",
         TORCH_BOX(&convert_vertical_slash_indexes));
  m.impl("convert_vertical_slash_indexes_mergehead",
         TORCH_BOX(&convert_vertical_slash_indexes_mergehead));
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_C, CPU, m) {
  m.impl("get_cuda_view_from_cpu_tensor",
         TORCH_BOX(&get_cuda_view_from_cpu_tensor));
}

REGISTER_EXTENSION(_C_stable_libtorch)
