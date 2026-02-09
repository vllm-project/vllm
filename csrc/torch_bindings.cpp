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

  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.impl("weak_ref_tensor", torch::kCUDA, &weak_ref_tensor);

#ifndef USE_ROCM
  ops.def(
      "silu_and_mul_nvfp4_quant(Tensor! result, Tensor! result_block_scale, "
      "Tensor input, Tensor input_global_scale) -> ()");
  ops.impl("silu_and_mul_nvfp4_quant", torch::kCUDA, &silu_and_mul_nvfp4_quant);
#endif

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

  // Marlin Optimized Quantized GEMM (supports GPTQ, AWQ, FP8, NVFP4, MXFP4).
  ops.def(
      "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none,Tensor b_scales, "
      "Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, "
      "Tensor? "
      "g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_type_id, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  // conditionally compiled so impl registration is in source file

  // gptq_marlin repack from GPTQ.
  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // awq_marlin repack from AWQ.
  ops.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  // conditionally compiled so impl registrations are in source file

  // preprocess W-int4A-fp8 weight for marlin kernel
  ops.def(
      "marlin_int4_fp8_preprocess(Tensor qweight, "
      "Tensor? qzeros_or_none, bool inplace) -> Tensor");
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

  // CUTLASS w4a8 grouped GEMM
  ops.def(
      "cutlass_w4a8_moe_mm("
      "   Tensor! out_tensors,"
      "   Tensor a_tensors,"
      "   Tensor b_tensors,"
      "   Tensor a_scales,"
      "   Tensor b_scales,"
      "   Tensor b_group_scales,"
      "   int b_group_size,"
      "   Tensor expert_offsets,"
      "   Tensor problem_sizes,"
      "   Tensor a_strides,"
      "   Tensor b_strides,"
      "   Tensor c_strides,"
      "   Tensor group_scale_strides,"
      "   str? maybe_schedule"
      ") -> ()");
  ops.def(
      "cutlass_encode_and_reorder_int4b_grouped(Tensor b_tensors) -> (Tensor, "
      "Tensor)");
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

  // compute per-expert problem sizes from expert_first_token_offset
  // produced by vLLM's moe_permute kernel
  ops.def(
      "get_cutlass_moe_mm_problem_sizes_from_expert_offsets("
      "    Tensor expert_first_token_offset, "
      "    Tensor! problem_sizes1, "
      "    Tensor! problem_sizes2, "
      "    int n, int k, bool swap_ab) -> ()");
  ops.impl("get_cutlass_moe_mm_problem_sizes_from_expert_offsets", torch::kCUDA,
           &get_cutlass_moe_mm_problem_sizes_from_expert_offsets);

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
      "                 Tensor! output_scale, Tensor input_scale, bool "
      "is_sf_swizzled_layout) -> ()");
  ops.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant);

  // Compute NVFP4 experts quantization.
  ops.def(
      "scaled_fp4_experts_quant(Tensor! output, Tensor! output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts) -> ()");
  ops.impl("scaled_fp4_experts_quant", torch::kCUDA, &scaled_fp4_experts_quant);

  // Fused SiLU+Mul+NVFP4 experts quantization.
  ops.def(
      "silu_and_mul_scaled_fp4_experts_quant(Tensor! output, Tensor! "
      "output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts) -> ()");
  ops.impl("silu_and_mul_scaled_fp4_experts_quant", torch::kCUDA,
           &silu_and_mul_scaled_fp4_experts_quant);

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
      "swap_blocks(Tensor src, Tensor! dst,"
      "            int block_size_in_bytes, Tensor block_mapping) -> ()");
  cache_ops.impl("swap_blocks", torch::kCUDA, &swap_blocks);

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

  // Rotate Q and K, then write to kv cache for MLA
  cache_ops.def(
      "concat_and_cache_mla_rope_fused("
      "                     Tensor positions,"
      "                     Tensor! q_pe,"
      "                     Tensor! k_pe,"
      "                     Tensor kv_c,"
      "                     Tensor cos_sin_cache,"
      "                     bool is_neox,"
      "                     Tensor slot_mapping,"
      "                     Tensor! kv_cache,"
      "                     str kv_cache_dtype,"
      "                     Tensor kv_cache_scale) -> ()");
  cache_ops.impl("concat_and_cache_mla_rope_fused", torch::kCUDA,
                 &concat_and_cache_mla_rope_fused);

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
      "                               Tensor token_to_seq, "
      "                               int num_tokens, "
      "                               str kv_cache_dtype, "
      "                               Tensor scale, Tensor? seq_starts) -> ()");
  cache_ops.impl("gather_and_maybe_dequant_cache", torch::kCUDA,
                 &gather_and_maybe_dequant_cache);

  cache_ops.def(
      "cp_gather_cache(Tensor src_cache, Tensor! dst, Tensor block_table, "
      "Tensor cu_seq_lens, int batch_size, Tensor? seq_starts) -> ()");
  cache_ops.impl("cp_gather_cache", torch::kCUDA, &cp_gather_cache);

  cache_ops.def(
      "cp_gather_and_upconvert_fp8_kv_cache(Tensor src_cache, Tensor! dst, "
      "Tensor block_table, Tensor seq_lens, Tensor workspace_starts, int "
      "batch_size) -> ()");
  cache_ops.impl("cp_gather_and_upconvert_fp8_kv_cache", torch::kCUDA,
                 &cp_gather_and_upconvert_fp8_kv_cache);

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
