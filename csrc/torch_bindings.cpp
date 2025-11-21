#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"

#include <torch/library.h>
#include <torch/version.h>

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
  //

  ops.def(
      "persistent_masked_m_silu_mul_quant(Tensor input, Tensor counts, Tensor! "
      "y_q, Tensor! y_s,"
      "bool use_ue8m0) -> ()");
  ops.impl("persistent_masked_m_silu_mul_quant", torch::kCUDA,
           &persistent_masked_m_silu_mul_quant);

  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.impl("weak_ref_tensor", torch::kCUDA, &weak_ref_tensor);

  ops.def("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");
  ops.impl("get_cuda_view_from_cpu_tensor", torch::kCPU,
           &get_cuda_view_from_cpu_tensor);

  // Attention ops
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  ops.def(
      "paged_attention_v1("
      "    Tensor! out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

  // PagedAttention V2.
  ops.def(
      "paged_attention_v2("
      "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"
      "    Tensor! tmp_out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);

#ifndef USE_ROCM
  // Merge attn states
  // Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
  // can be used to combine partial attention results (in the split-KV case)
  ops.def(
      "merge_attn_states("
      "    Tensor! output,"
      "    Tensor!? output_lse,"
      "    Tensor prefix_output,"
      "    Tensor prefix_lse,"
      "    Tensor suffix_output,"
      "    Tensor suffix_lse) -> ()");
  ops.impl("merge_attn_states", torch::kCUDA, &merge_attn_states);

  ops.def(
      "convert_vertical_slash_indexes("
      "   Tensor! block_count, Tensor! block_offset, "
      "   Tensor! column_count, Tensor! column_index, "
      "   Tensor q_seqlens, Tensor q_seqlens, "
      "   Tensor vertical_indexes, Tensor slash_indexes, "
      "   int context_size, int block_size_M, int block_size_N, "
      "   bool causal) -> ()");
  ops.impl("convert_vertical_slash_indexes", torch::kCUDA,
           &convert_vertical_slash_indexes);

  ops.def(
      "convert_vertical_slash_indexes_mergehead("
      "   Tensor! block_count, Tensor! block_offset, "
      "   Tensor! column_count, Tensor! column_index, "
      "   Tensor q_seqlens, Tensor q_seqlens, "
      "   Tensor vertical_indexes, Tensor slash_indexes, "
      "   Tensor vertical_indices_count, Tensor slash_indices_count, "
      "   int context_size, int block_size_M, int block_size_N, "
      "   bool causal) -> ()");
  ops.impl("convert_vertical_slash_indexes_mergehead", torch::kCUDA,
           &convert_vertical_slash_indexes_mergehead);
#endif

  // Activation ops
  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  ops.def(
      "silu_and_mul_quant(Tensor! result, Tensor input, Tensor scale) -> ()");
  ops.impl("silu_and_mul_quant", torch::kCUDA, &silu_and_mul_quant);

#ifndef USE_ROCM
  ops.def(
      "silu_and_mul_nvfp4_quant(Tensor! result, Tensor! result_block_scale, "
      "Tensor input, Tensor input_global_scale) -> ()");
  ops.impl("silu_and_mul_nvfp4_quant", torch::kCUDA, &silu_and_mul_nvfp4_quant);
#endif

  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");
  ops.impl("mul_and_silu", torch::kCUDA, &mul_and_silu);

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  // FATReLU implementation.
  ops.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");
  ops.impl("fatrelu_and_mul", torch::kCUDA, &fatrelu_and_mul);

  ops.def(
      "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float "
      "limit=7.0) "
      "-> ()");
  ops.impl("swigluoai_and_mul", torch::kCUDA, &swigluoai_and_mul);

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
      "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> "
      "()");
  ops.impl("rms_norm", torch::kCUDA, &rms_norm);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  ops.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);

  // Function for fused QK Norm and RoPE
  ops.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, "
      "int num_heads_k, int num_heads_v, int head_dim, float eps, "
      "Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, "
      "bool is_neox, Tensor position_ids) -> ()");
  ops.impl("fused_qk_norm_rope", torch::kCUDA, &fused_qk_norm_rope);

  // Apply repetition penalties to logits in-place
  ops.def(
      "apply_repetition_penalties_(Tensor! logits, Tensor prompt_mask, "
      "Tensor output_mask, Tensor repetition_penalties) -> ()");
  ops.impl("apply_repetition_penalties_", torch::kCUDA,
           &apply_repetition_penalties_);

  // Optimized top-k per row operation
  ops.def(
      "top_k_per_row(Tensor logits, Tensor rowStarts, Tensor rowEnds, "
      "Tensor! indices, int numRows, int stride0, "
      "int stride1) -> ()");
  ops.impl("top_k_per_row", torch::kCUDA, &top_k_per_row);

  ops.def(
      "top_k_per_row_decode(Tensor logits, int next_n, "
      "Tensor seq_lens, Tensor! indices, int numRows, "
      "int stride0, int stride1) -> ()");
  ops.impl("top_k_per_row_decode", torch::kCUDA, &top_k_per_row_decode);

  // Layernorm-quant
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor scale, float epsilon) -> "
      "()");
  ops.impl("rms_norm_static_fp8_quant", torch::kCUDA,
           &rms_norm_static_fp8_quant);

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");
  ops.impl("fused_add_rms_norm_static_fp8_quant", torch::kCUDA,
           &fused_add_rms_norm_static_fp8_quant);

  // Fused Layernorm + Quant kernels
  ops.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");
  ops.impl("rms_norm_dynamic_per_token_quant", torch::kCUDA,
           &rms_norm_dynamic_per_token_quant);

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  // Quantization ops
#ifndef USE_ROCM
  // Quantized GEMM for AWQ.
  ops.def(
      "awq_gemm(Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters) -> Tensor");
  ops.impl("awq_gemm", torch::kCUDA, &awq_gemm);

  // Dequantization for AWQ.
  ops.def(
      "awq_dequantize(Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters, int thx, int thy) -> Tensor");
  ops.impl("awq_dequantize", torch::kCUDA, &awq_dequantize);

  // Note about marlin kernel 'workspace' arguments:
  // Technically these should be mutable since they are modified by the kernel.
  // But since they are set back to zero once the kernel is finished we can
  // hand wave and say that they have no net effect.
  //
  // The reason to mark 'workspace' as immutable is so that they don't interfere
  // with using ScalarType arguments in the ops. If they are marked as mutable,
  // pytorch throws an assert in
  // 'torch._higher_order_ops._register_effectful_op' that prevents these
  // kernels from being torch.compile'd.
  // See the following document for more info on custom types and ops that use
  // custom types:
  // https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA

  // Marlin_24 (Sparse) Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_24_gemm(Tensor a, Tensor b_q_weight, Tensor b_meta, "
      "Tensor b_scales, Tensor workspace, "
      "int b_q_type, "
      "SymInt size_m, SymInt size_n, SymInt size_k) -> Tensor");
  //  conditionally compiled so impl in source file

  // Machete (Dense) Optimized Mixed Precision GEMM for Hopper.
  ops.def(
      "machete_supported_schedules("
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? maybe_group_scales_type,"
      "   ScalarType? maybe_group_zeros_type,"
      "   ScalarType? maybe_channel_scales_type,"
      "   ScalarType? maybe_token_scales_type,"
      "   ScalarType? maybe_out_type"
      ") -> str[]");
  ops.def(
      "machete_mm("
      "   Tensor A,"
      "   Tensor B,"
      "   int b_type,"
      "   ScalarType? out_type,"
      "   Tensor? group_scales,"
      "   Tensor? group_zeros,"
      "   int?    group_size,"
      "   Tensor? channel_scales,"
      "   Tensor? token_scales,"
      "   str?    schedule"
      ") -> Tensor");
  ops.def(
      "machete_prepack_B("
      "   Tensor B,"
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? group_scales_type"
      ") -> Tensor");
  // conditionally compiled so impl registration is in source file

  ops.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
  ops.impl("permute_cols", torch::kCUDA, &permute_cols);

  // gptq_marlin Optimized Quantized GEMM for GPTQ.
  ops.def(
      "gptq_marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none,"
      "Tensor b_scales, Tensor? global_scale, Tensor? b_zeros_or_none, Tensor? "
      "g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_q_type, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  // conditionally compiled so impl registration is in source file

  // gptq_marlin repack from GPTQ.
  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // awq_marlin repack from AWQ.
  ops.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // CUTLASS w4a8 GEMM
  ops.def(
      "cutlass_w4a8_mm("
      "   Tensor A,"
      "   Tensor B,"
      "   Tensor group_scales,"
      "   int    group_size,"
      "   Tensor channel_scales,"
      "   Tensor token_scales,"
      "   ScalarType? out_type,"
      "   str?   maybe_schedule"
      ") -> Tensor");
  // pack scales
  ops.def("cutlass_pack_scale_fp8(Tensor scales) -> Tensor");
  // encode and reorder weight matrix
  ops.def("cutlass_encode_and_reorder_int4b(Tensor B) -> Tensor");
  // conditionally compiled so impl registration is in source file

#endif

  // Dequantization for GGML.
  ops.def(
      "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
      "dtype) -> Tensor");
  ops.impl("ggml_dequantize", torch::kCUDA, &ggml_dequantize);

  // mmvq kernel for GGML.
  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) "
      "-> Tensor");
  ops.impl("ggml_mul_mat_vec_a8", torch::kCUDA, &ggml_mul_mat_vec_a8);

  // mmq kernel for GGML.
  ops.def(
      "ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.impl("ggml_mul_mat_a8", torch::kCUDA, &ggml_mul_mat_a8);

  // moe kernel for GGML.
  ops.def(
      "ggml_moe_a8(Tensor X, Tensor W, "
      "Tensor sorted_token_ids, Tensor expert_ids, Tensor "
      "num_tokens_post_padded, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");
  ops.impl("ggml_moe_a8", torch::kCUDA, &ggml_moe_a8);

  ops.def(
      "ggml_moe_a8_vec(Tensor X, Tensor W, "
      "Tensor topk_ids, int top_k, "
      "int type, SymInt row, SymInt tokens) -> Tensor");
  ops.impl("ggml_moe_a8_vec", torch::kCUDA, &ggml_moe_a8_vec);

  ops.def("ggml_moe_get_block_size", &ggml_moe_get_block_size);

#ifndef USE_ROCM
  // CUTLASS nvfp4 block scaled GEMM
  ops.def(
      "cutlass_scaled_fp4_mm(Tensor! out, Tensor a, Tensor b,"
      "                      Tensor block_scale_a, Tensor block_scale_b,"
      "                      Tensor alpha) -> ()");
  ops.impl("cutlass_scaled_fp4_mm", torch::kCUDA, &cutlass_scaled_fp4_mm);

  // cutlass blockwise scaledgroup GEMM
  ops.def(
      "cutlass_blockwise_scaled_grouped_mm(Tensor! output, Tensor a, Tensor b, "
      "Tensor scales_a, Tensor scales_b, "
      "Tensor problem_sizes, Tensor expert_offsets) -> ()");
  // conditionally compiled so impl registration is in source file

  // cutlass nvfp4 block scaled group GEMM
  ops.def(
      "cutlass_fp4_group_mm(Tensor! out, Tensor a, Tensor b,"
      " Tensor a_blockscale, Tensor b_blockscales, Tensor alphas,"
      " Tensor problem_sizes, Tensor expert_offsets, Tensor sf_offsets) -> ()");
  // conditionally compiled so impl registration is in source file

  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column
  // quantization, as well as bias
  ops.def(
      "cutlass_scaled_mm(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_mm", torch::kCUDA, &cutlass_scaled_mm);

  // CUTLASS w8a8 GEMM, supporting asymmetric per-tensor or per-row/column
  // quantization.
  ops.def(
      "cutlass_scaled_mm_azp(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor azp_adj,"
      "                  Tensor? azp, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_mm_azp", torch::kCUDA, &cutlass_scaled_mm_azp);

  // Check if cutlass scaled_mm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool");
  ops.impl("cutlass_scaled_mm_supports_fp8", &cutlass_scaled_mm_supports_fp8);

  // Check if cutlass grouped gemm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_group_gemm_supported(int cuda_device_capability) -> bool");
  ops.impl("cutlass_group_gemm_supported", &cutlass_group_gemm_supported);

  // CUTLASS w8a8 grouped GEMM
  ops.def(
      "cutlass_moe_mm(Tensor! out_tensors, Tensor a_tensors, Tensor b_tensors, "
      "               Tensor a_scales, Tensor b_scales, Tensor expert_offsets, "
      "               Tensor problem_sizes, Tensor a_strides, "
      "               Tensor b_strides, Tensor c_strides, bool per_act_token, "
      "               bool per_out_ch) -> ()");
  ops.impl("cutlass_moe_mm", torch::kCUDA, &cutlass_moe_mm);

  // A function that computes data required to run fused MoE with w8a8 grouped
  // GEMM. It takes topk_ids as an input, and computes expert_offsets
  // (token start indices of each expert). In addition to this, it computes
  // problem sizes for each expert's multiplication used by the two mms called
  // from fused MoE operation, and arrays with permutations required to shuffle
  // and de-shuffle the input/output of the fused operation.
  ops.def(
      "get_cutlass_moe_mm_data(Tensor topk_ids, Tensor! expert_offsets, "
      "                        Tensor! problem_sizes1, Tensor! problem_sizes2, "
      "                        Tensor! input_permutation, "
      "                        Tensor! output_permutation, int num_experts, "
      "                        int n, int k, Tensor? blockscale_offsets) -> "
      "()");
  ops.impl("get_cutlass_moe_mm_data", torch::kCUDA, &get_cutlass_moe_mm_data);

  // A function that computes problem sizes for each expert's multiplication
  // used by the two mms called from fused MoE operation. It takes topk_ids as
  // an input, and computes problem_sizes1 and problem_sizes2 only.
  ops.def(
      "get_cutlass_moe_mm_problem_sizes(Tensor topk_ids, "
      "                                 Tensor! problem_sizes1, "
      "                                 Tensor! problem_sizes2, "
      "                                 int num_experts, int n, int k, "
      "                                 Tensor? blockscale_offsets) -> ()");
  ops.impl("get_cutlass_moe_mm_problem_sizes", torch::kCUDA,
           &get_cutlass_moe_mm_problem_sizes);

  // A function that computes data required to run fused MoE with w8a8 grouped
  // GEMM and PPLX. It takes expert_num_tokens and non_zero_expert_idxs
  // as an input, and computes expert_offsets (token start indices of each
  // expert). In addition to this, it computes problem sizes for each expert's
  // multiplication used by the two mms called from fused MoE operation.
  ops.def(
      "get_cutlass_pplx_moe_mm_data(Tensor! expert_offsets, "
      "                             Tensor! problem_sizes1, "
      "                             Tensor! problem_sizes2, "
      "                             Tensor expert_num_tokens, "
      "                             int num_local_experts, int padded_m, "
      "                             int n, int k) -> ()");
  ops.impl("get_cutlass_pplx_moe_mm_data", torch::kCUDA,
           &get_cutlass_pplx_moe_mm_data);

  // Check if cutlass scaled_mm supports block quantization (used by DeepSeekV3)
  ops.def(
      "cutlass_scaled_mm_supports_block_fp8(int cuda_device_capability) -> "
      "bool");
  ops.impl("cutlass_scaled_mm_supports_block_fp8",
           &cutlass_scaled_mm_supports_block_fp8);

  // Check if cutlass sparse scaled_mm is supported for CUDA devices of the
  // given capability
  ops.def(
      "cutlass_sparse_scaled_mm_supported(int cuda_device_capability) -> bool");
  ops.impl("cutlass_sparse_scaled_mm_supported",
           &cutlass_sparse_scaled_mm_supported);

  // CUTLASS sparse GEMM, supporting symmetric per-tensor or per-row/column
  // quantization, as well as bias
  ops.def(
      "cutlass_scaled_sparse_mm(Tensor! out, Tensor a,"
      "                         Tensor bt_nzs,"
      "                         Tensor bt_meta, Tensor a_scales,"
      "                         Tensor b_scales, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_sparse_mm", torch::kCUDA, &cutlass_scaled_sparse_mm);

  // CUTLASS sparse matrix compressor
  ops.def("cutlass_sparse_compress(Tensor a) -> Tensor[]");
  ops.impl("cutlass_sparse_compress", &cutlass_sparse_compress);

  // SM100 CUTLASS MLA decode
  ops.def(
      "sm100_cutlass_mla_decode(Tensor! out, Tensor! lse, Tensor q_nope,"
      "                         Tensor q_pe, Tensor kv_c_and_k_pe_cache,"
      "                         Tensor seq_lens, Tensor page_table,"
      "                         Tensor workspace, float scale,"
      "                         int num_kv_splits) -> ()");
  // conditionally compiled so impl in source file

  // SM100 CUTLASS MLA workspace
  ops.def(
      "sm100_cutlass_mla_get_workspace_size(int max_seq_len, int num_batches,"
      "                                     int sm_count, int num_kv_splits) "
      "-> int");
  // conditionally compiled so impl in source file

  // Compute NVFP4 block quantized tensor.
  ops.def(
      "scaled_fp4_quant(Tensor! output, Tensor input,"
      "                 Tensor! output_scale, Tensor input_scale) -> ()");
  ops.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant);

  // Compute NVFP4 experts quantization.
  ops.def(
      "scaled_fp4_experts_quant(Tensor! output, Tensor! output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts) -> ()");
  ops.impl("scaled_fp4_experts_quant", torch::kCUDA, &scaled_fp4_experts_quant);

  // Check if cutlass_scaled_mm_fp4 is supported for CUDA devices
  // of the given capability
  ops.def("cutlass_scaled_mm_supports_fp4(int cuda_device_capability) -> bool");
  ops.impl("cutlass_scaled_mm_supports_fp4", &cutlass_scaled_mm_supports_fp4);
#endif

  // Quantized GEMM for GPTQ.
  // Note: even though the C++ inferred schema is correct for this op, it seems
  // to prevent the meta function registry.
  ops.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, "
      "Tensor b_gptq_scales, Tensor b_g_idx, bool use_exllama, bool "
      "use_v2_format, int bit) "
      "-> Tensor");
  ops.impl("gptq_gemm", torch::kCUDA, &gptq_gemm);

  // Post processing for GPTQ.
  ops.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");
  ops.impl("gptq_shuffle", torch::kCUDA, &gptq_shuffle);

  // Compute FP8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale) -> "
      "()");
  ops.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) "
      "-> "
      "()");
  ops.impl("dynamic_scaled_fp8_quant", torch::kCUDA, &dynamic_scaled_fp8_quant);

  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");
  ops.impl("dynamic_per_token_scaled_fp8_quant", torch::kCUDA,
           &dynamic_per_token_scaled_fp8_quant);

  // Compute int8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_int8_quant(Tensor! result, Tensor input, Tensor scale,"
      "Tensor? azp) -> ()");
  ops.impl("static_scaled_int8_quant", torch::kCUDA, &static_scaled_int8_quant);

  // Compute int8 quantized tensor and scaling factor
  ops.def(
      "dynamic_scaled_int8_quant(Tensor! result, Tensor input, Tensor! scale, "
      "Tensor!? azp) -> ()");
  ops.impl("dynamic_scaled_int8_quant", torch::kCUDA,
           &dynamic_scaled_int8_quant);

  // Mamba selective scan kernel
  ops.def(
      "selective_scan_fwd(Tensor! u, Tensor! delta,"
      "Tensor! A, Tensor! B, Tensor! C,"
      "Tensor? D_, Tensor!? z_, Tensor? delta_bias_,"
      "bool delta_softplus,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "Tensor! ssm_states,"
      "int pad_slot_id,"
      "int block_size,"
      "Tensor? block_idx_first_scheduled_token,"
      "Tensor? block_idx_last_scheduled_token,"
      "Tensor? initial_state_idx) -> ()");
  ops.impl("selective_scan_fwd", torch::kCUDA, &selective_scan_fwd);

  // Hadamard transforms
  ops.def("hadacore_transform(Tensor! x, bool inplace) -> Tensor");

#ifndef USE_ROCM
  // Compute per-token-group FP8 quantized tensor and scaling factor.
  ops.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, "
      "int group_size, float eps, float fp8_min, float fp8_max, bool "
      "scale_ue8m0) -> ()");
  ops.impl("per_token_group_fp8_quant", torch::kCUDA,
           &per_token_group_quant_fp8);

  // Compute per-token-group INT8 quantized tensor and scaling factor.
  ops.def(
      "per_token_group_quant_int8(Tensor input, Tensor! output_q, Tensor! "
      "output_s, int group_size, float eps, float int8_min, float int8_max) -> "
      "()");
  ops.impl("per_token_group_quant_int8", torch::kCUDA,
           &per_token_group_quant_int8);

  // reorder weight for AllSpark Ampere W8A16 Fused Gemm kernel
  ops.def(
      "rearrange_kn_weight_as_n32k16_order(Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_zeros, "
      "bool has_zp, Tensor! b_qweight_reorder, Tensor! b_scales_reorder, "
      "Tensor!? b_zeros_reorder, "
      "int K, int N, int N_32align) -> ()");
  //  conditionally compiled so impl in source file

  // AllSpark quantization ops
  ops.def(
      "allspark_w8a16_gemm(Tensor a, Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_qzeros, "
      "SymInt n, SymInt group_size, SymInt sm_count, SymInt sm_version, SymInt "
      "CUBLAS_M_THRESHOLD, bool has_zp, bool n32k16_reorder) -> Tensor");
  //  conditionally compiled so impl in source file
#endif
}

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
  // Cache ops
  // Swap in (out) the cache blocks from src to dst.
  cache_ops.def(
      "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
  cache_ops.impl("swap_blocks", torch::kCUDA, &swap_blocks);

  // Copy the cache blocks from src to dst.
  cache_ops.def(
      "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, "
      "Tensor block_mapping) -> ()");
  cache_ops.impl("copy_blocks", torch::kCUDA, &copy_blocks);

  cache_ops.def(
      "copy_blocks_mla(Tensor(a!)[] kv_caches, Tensor block_mapping) -> ()");
  cache_ops.impl("copy_blocks_mla", torch::kCUDA, &copy_blocks_mla);

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale) -> ()");
  cache_ops.impl("reshape_and_cache", torch::kCUDA, &reshape_and_cache);

  // Reshape the key and value tensors and cache them.
  cache_ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");
  cache_ops.impl("reshape_and_cache_flash", torch::kCUDA,
                 &reshape_and_cache_flash);

  // Concat kv_c and k_pe and cache them.
  cache_ops.def(
      "concat_and_cache_mla(Tensor kv_c, Tensor k_pe,"
      "                     Tensor! kv_cache,"
      "                     Tensor slot_mapping,"
      "                     str kv_cache_dtype,"
      "                     Tensor scale) -> ()");
  cache_ops.impl("concat_and_cache_mla", torch::kCUDA, &concat_and_cache_mla);

  // Convert the key and value cache to fp8 data type.
  cache_ops.def(
      "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
      "str kv_cache_dtype) -> ()");
  cache_ops.impl("convert_fp8", torch::kCUDA, &convert_fp8);

  // Gather cache blocks from src_cache to dst, dequantizing from
  // src_cache's dtype to dst's dtype if necessary.
  cache_ops.def(
      "gather_and_maybe_dequant_cache(Tensor src_cache, Tensor! dst, "
      "                               Tensor block_table, Tensor cu_seq_lens, "
      "                               int batch_size, "
      "                               str kv_cache_dtype, "
      "                               Tensor scale, Tensor? seq_starts) -> ()");
  cache_ops.impl("gather_and_maybe_dequant_cache", torch::kCUDA,
                 &gather_and_maybe_dequant_cache);

  cache_ops.def(
      "cp_gather_cache(Tensor src_cache, Tensor! dst, Tensor block_table, "
      "Tensor cu_seq_lens, int batch_size, Tensor? seq_starts) -> ()");
  cache_ops.impl("cp_gather_cache", torch::kCUDA, &cp_gather_cache);

  cache_ops.def(
      "indexer_k_quant_and_cache(Tensor k, Tensor! kv_cache, Tensor "
      "slot_mapping, "
      "int quant_block_size, str kv_cache_dtype) -> ()");
  cache_ops.impl("indexer_k_quant_and_cache", torch::kCUDA,
                 &indexer_k_quant_and_cache);

  cache_ops.def(
      "cp_gather_indexer_k_quant_cache(Tensor kv_cache, Tensor! dst_k, Tensor! "
      "dst_scale, Tensor block_table, Tensor cu_seq_lens) -> ()");
  cache_ops.impl("cp_gather_indexer_k_quant_cache", torch::kCUDA,
                 &cp_gather_indexer_k_quant_cache);
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

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  // Custom all-reduce kernels
  custom_ar.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool fully_connected) -> int");
  custom_ar.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
  custom_ar.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ar.impl("all_reduce", torch::kCUDA, &all_reduce);

  custom_ar.def("dispose", &dispose);
  custom_ar.def("meta_size", &meta_size);

  custom_ar.def("register_buffer", &register_buffer);
  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  custom_ar.def("register_graph_buffers", &register_graph_buffers);

  custom_ar.def("allocate_shared_buffer_and_handle",
                &allocate_shared_buffer_and_handle);
  custom_ar.def("open_mem_handle(Tensor mem_handle) -> int", &open_mem_handle);
  custom_ar.impl("open_mem_handle", torch::kCPU, &open_mem_handle);

  custom_ar.def("free_shared_buffer", &free_shared_buffer);
#ifdef USE_ROCM
  // Quick Reduce all-reduce kernels
  custom_ar.def(
      "qr_all_reduce(int fa, Tensor inp, Tensor out, int quant_level, bool "
      "cast_bf2half) -> ()");
  custom_ar.impl("qr_all_reduce", torch::kCUDA, &qr_all_reduce);

  custom_ar.def("init_custom_qr", &init_custom_qr);
  custom_ar.def("qr_destroy", &qr_destroy);

  custom_ar.def("qr_get_handle", &qr_get_handle);

  custom_ar.def("qr_open_handles(int _fa, Tensor[](b!) handles) -> ()");
  custom_ar.impl("qr_open_handles", torch::kCPU, &qr_open_handles);

  // Max input size in bytes
  custom_ar.def("qr_max_size", &qr_max_size);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
