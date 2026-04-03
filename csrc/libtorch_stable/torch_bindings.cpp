#include "ops.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

// Register ops with STABLE_TORCH_LIBRARY for libtorch stable ABI compatibility.
// Note: We register under namespace "_C" so ops are accessible as
// torch.ops._C.<op_name> for compatibility with existing code.
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
#ifndef USE_ROCM
  ops.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
#endif

#ifndef USE_ROCM
  // Compute per-token-group FP8 quantized tensor and scaling factor.
  // The dummy arguments are here so we can correctly fuse with RMSNorm.
  ops.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, "
      "int group_size, float eps, float fp8_min, float fp8_max, bool "
      "scale_ue8m0, bool dummy_is_scale_transposed, bool dummy_is_tma_aligned "
      ") -> ()");
  // Compute per-token-group 8-bit quantized tensor and UE8M0-packed,
  // TMA-aligned scales for DeepGEMM.
  ops.def(
      "per_token_group_fp8_quant_packed(Tensor input, Tensor! output_q, "
      "Tensor! output_s_packed, int group_size, float eps, float fp8_min, "
      "float fp8_max) -> ()");
  // Compute per-token-group INT8 quantized tensor and scaling factor.
  ops.def(
      "per_token_group_quant_int8(Tensor input, Tensor! output_q, Tensor! "
      "output_s, int group_size, float eps, float int8_min, float int8_max) -> "
      "()");

  // CUTLASS w8a8 GEMM, supporting symmetric per-tensor or per-row/column
  // quantization, as well as bias
  ops.def(
      "cutlass_scaled_mm(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor? bias) -> ()");

  // CUTLASS w8a8 GEMM, supporting asymmetric per-tensor or per-row/column
  // quantization.
  ops.def(
      "cutlass_scaled_mm_azp(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor azp_adj,"
      "                  Tensor? azp, Tensor? bias) -> ()");

  // Check if cutlass scaled_mm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool");

  // Check if cutlass grouped gemm is supported for CUDA devices of the given
  // capability
  ops.def("cutlass_group_gemm_supported(int cuda_device_capability) -> bool");

  // CUTLASS w8a8 grouped GEMM
  ops.def(
      "cutlass_moe_mm(Tensor! out_tensors, Tensor a_tensors, Tensor b_tensors, "
      "               Tensor a_scales, Tensor b_scales, Tensor expert_offsets, "
      "               Tensor problem_sizes, Tensor a_strides, "
      "               Tensor b_strides, Tensor c_strides, bool per_act_token, "
      "               bool per_out_ch) -> ()");

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
      "                        int n, int k, Tensor? blockscale_offsets, "
      "                        bool is_gated) -> ()");

  // compute per-expert problem sizes from expert_first_token_offset
  // produced by vLLM's moe_permute kernel
  ops.def(
      "get_cutlass_moe_mm_problem_sizes_from_expert_offsets("
      "    Tensor expert_first_token_offset, "
      "    Tensor! problem_sizes1, "
      "    Tensor! problem_sizes2, "
      "    int n, int k, bool swap_ab) -> ()");

  // A function that computes data required to run fused MoE with w8a8 grouped
  // GEMM in batched expert format. It takes expert_num_tokens
  // as an input, and computes expert_offsets (token start indices of each
  // expert). In addition to this, it computes problem sizes for each expert's
  // multiplication used by the two mms called from fused MoE operation.
  ops.def(
      "get_cutlass_batched_moe_mm_data(Tensor! expert_offsets, "
      "                             Tensor! problem_sizes1, "
      "                             Tensor! problem_sizes2, "
      "                             Tensor expert_num_tokens, "
      "                             int num_local_experts, int padded_m, "
      "                             int n, int k) -> ()");

  // Check if cutlass scaled_mm supports block quantization (used by DeepSeekV3)
  ops.def(
      "cutlass_scaled_mm_supports_block_fp8(int cuda_device_capability) -> "
      "bool");

  // CUTLASS nvfp4 block scaled GEMM
  ops.def(
      "cutlass_scaled_fp4_mm(Tensor! out, Tensor a, Tensor b,"
      "                      Tensor block_scale_a, Tensor block_scale_b,"
      "                      Tensor alpha) -> ()");

  // cutlass nvfp4 block scaled group GEMM
  ops.def(
      "cutlass_fp4_group_mm(Tensor! out, Tensor a, Tensor b,"
      " Tensor a_blockscale, Tensor b_blockscales, Tensor alphas,"
      " Tensor problem_sizes, Tensor expert_offsets, Tensor sf_offsets) -> ()");

  // Compute NVFP4 block quantized tensor.
  ops.def(
      "scaled_fp4_quant(Tensor input,"
      "                 Tensor input_scale, bool "
      "is_sf_swizzled_layout) -> (Tensor, Tensor)");

  // Out variant
  // TODO: Add out_variant tag once PyTorch supports it (added in 2.11)
  // This registration is now migrated to stable ABI
  // at::Tag::out_variant is not available in the stable ABI (enum_tag.h is not
  // yet in torch/headeronly), the tag should be applied from Python
  // via torch.library.Library.define(..., tags=(torch.Tag.out_variant,))
  // with the .impl remaining in C++.
  // See pytorch/pytorch#176117.
  ops.def(
      "scaled_fp4_quant.out(Tensor input,"
      "                     Tensor input_scale, bool "
      "is_sf_swizzled_layout, *, Tensor(a!) output, Tensor(b!) output_scale) "
      "-> ()");

  // Compute NVFP4 experts quantization.
  ops.def(
      "scaled_fp4_experts_quant(Tensor! output, Tensor! output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts) -> ()");

  // Fused SiLU+Mul+NVFP4 experts quantization.
  ops.def(
      "silu_and_mul_scaled_fp4_experts_quant(Tensor! output, Tensor! "
      "output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts) -> ()");

  // Fused SiLU+Mul+NVFP4 quantization.
  ops.def(
      "silu_and_mul_nvfp4_quant(Tensor! result, Tensor! result_block_scale, "
      "Tensor input, Tensor input_global_scale) -> ()");

  // Check if cutlass_scaled_mm_fp4 is supported for CUDA devices
  // of the given capability
  ops.def("cutlass_scaled_mm_supports_fp4(int cuda_device_capability) -> bool");

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

  // SM100 CUTLASS MLA decode
  // conditionally compiled so impl registrations are in source file
  ops.def(
      "sm100_cutlass_mla_decode(Tensor! out, Tensor! lse, Tensor q_nope,"
      "                         Tensor q_pe, Tensor kv_c_and_k_pe_cache,"
      "                         Tensor seq_lens, Tensor page_table,"
      "                         Tensor workspace, float scale,"
      "                         int num_kv_splits) -> ()");

  ops.def(
      "sm100_cutlass_mla_get_workspace_size(int max_seq_len, int num_batches,"
      "                                     int sm_count, int num_kv_splits) "
      "-> int");
  // Quantized GEMM for AWQ.
  ops.def(
      "awq_gemm(Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters) -> Tensor");

  // Dequantization for AWQ.
  ops.def(
      "awq_dequantize(Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters, int thx, int thy) -> Tensor");

  // DeepSeek V3 fused A GEMM (SM 9.0+, bf16 only, 1-16 tokens).
  // conditionally compiled so impl registration is in source file
  ops.def(
      "dsv3_fused_a_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()");

  // reorder weight for AllSpark Ampere W8A16 Fused Gemm kernel
  ops.def(
      "rearrange_kn_weight_as_n32k16_order(Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_zeros, "
      "bool has_zp, Tensor! b_qweight_reorder, Tensor! b_scales_reorder, "
      "Tensor!? b_zeros_reorder, "
      "int K, int N, int N_32align) -> ()");

  // AllSpark quantization ops
  ops.def(
      "allspark_w8a16_gemm(Tensor a, Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_qzeros, "
      "SymInt n, SymInt group_size, SymInt sm_count, SymInt sm_version, SymInt "
      "CUBLAS_M_THRESHOLD, bool has_zp, bool n32k16_reorder) -> Tensor");
#endif

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
      "    Tensor suffix_lse,"
      "    int!? prefill_tokens_with_context,"
      "    Tensor? output_scale=None) -> ()");

  // Hadamard transforms
  // conditionally compiled so impl registration is in source file
  ops.def("hadacore_transform(Tensor! x, bool inplace) -> Tensor");

  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> "
      "()");

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");

  // Layernorm-quant
  // Apply Root Mean Square (RMS) Normalization to the input tensor.
  ops.def(
      "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor scale, float epsilon) -> "
      "()");

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");

  // Fused Layernorm + Quant kernels
  ops.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");

  // Fused Layernorm + Block quant kernels
  ops.def(
      "rms_norm_per_block_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual, int group_size, "
      "bool is_scale_transposed) -> ()");

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");

  // Function for fused QK Norm and RoPE
  ops.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, "
      "int num_heads_k, int num_heads_v, int head_dim, float eps, "
      "Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, "
      "bool is_neox, Tensor position_ids) -> ()");

  // Activation ops
  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");

  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");

  // Activation function used in GeGLU with `none` approximation.
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");

  // Activation function used in GeGLU with `tanh` approximation.
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");

  // FATReLU implementation.
  ops.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");

  ops.def(
      "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float "
      "limit=7.0) "
      "-> ()");

  // GELU implementation used in GPT-2.
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");

  // Approximate GELU implementation.
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");

  // Quick GELU implementation.
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");

  // Compute int8 quantized tensor for given scaling factor.
  ops.def(
      "static_scaled_int8_quant(Tensor! result, Tensor input, Tensor scale,"
      "Tensor? azp) -> ()");

  // Compute int8 quantized tensor and scaling factor
  ops.def(
      "dynamic_scaled_int8_quant(Tensor! result, Tensor input, Tensor! scale, "
      "Tensor!? azp) -> ()");

  // Compute FP8 quantized tensor for given scaling factor.
  // Supports per-tensor, per-channel, per-token, and arbitrary 2D group
  // scaling. Optional group_m/group_n specify the group shape explicitly;
  // required for 1D scales to disambiguate per-channel vs per-token.
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale, "
      "int[]? group_shape=None) -> ()");

  // Compute dynamic-per-tensor FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) "
      "-> "
      "()");

  // Compute dynamic-per-token FP8 quantized tensor and scaling factor.
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");

  // Quantized GEMM for GPTQ.
  // Note: even though the C++ inferred schema is correct for this op, it seems
  // to prevent the meta function registry.
  ops.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, "
      "Tensor b_gptq_scales, Tensor b_g_idx, bool use_exllama, bool "
      "use_v2_format, int bit) "
      "-> Tensor");

  // Post processing for GPTQ.
  ops.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");

  // Dequantization for GGML.
  ops.def(
      "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
      "dtype) -> Tensor");

  // mmvq kernel for GGML.
  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) "
      "-> Tensor");

  // mmq kernel for GGML.
  ops.def(
      "ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");

  // moe kernel for GGML.
  ops.def(
      "ggml_moe_a8(Tensor X, Tensor W, "
      "Tensor sorted_token_ids, Tensor expert_ids, Tensor "
      "num_tokens_post_padded, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");

  ops.def(
      "ggml_moe_a8_vec(Tensor X, Tensor W, "
      "Tensor topk_ids, int top_k, "
      "int type, SymInt row, SymInt tokens) -> Tensor");

  ops.def("ggml_moe_get_block_size(int type) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
#ifndef USE_ROCM
  ops.impl("permute_cols", TORCH_BOX(&permute_cols));
#endif

#ifndef USE_ROCM
  // Per-token group quantization
  ops.impl("per_token_group_fp8_quant", TORCH_BOX(&per_token_group_quant_fp8));
  ops.impl("per_token_group_fp8_quant_packed",
           TORCH_BOX(&per_token_group_quant_8bit_packed));
  ops.impl("per_token_group_quant_int8",
           TORCH_BOX(&per_token_group_quant_int8));

  // CUTLASS scaled_mm ops
  ops.impl("cutlass_scaled_mm", TORCH_BOX(&cutlass_scaled_mm));
  ops.impl("cutlass_scaled_mm_azp", TORCH_BOX(&cutlass_scaled_mm_azp));
  ops.impl("cutlass_moe_mm", TORCH_BOX(&cutlass_moe_mm));
  ops.impl("get_cutlass_moe_mm_data", TORCH_BOX(&get_cutlass_moe_mm_data));
  ops.impl("get_cutlass_moe_mm_problem_sizes_from_expert_offsets",
           TORCH_BOX(&get_cutlass_moe_mm_problem_sizes_from_expert_offsets));
  ops.impl("get_cutlass_batched_moe_mm_data",
           TORCH_BOX(&get_cutlass_batched_moe_mm_data));

  // FP4/NVFP4 ops
  ops.impl("cutlass_scaled_fp4_mm", TORCH_BOX(&cutlass_scaled_fp4_mm));
  ops.impl("scaled_fp4_quant", TORCH_BOX(&scaled_fp4_quant_func));
  ops.impl("scaled_fp4_quant.out", TORCH_BOX(&scaled_fp4_quant_out));
  ops.impl("scaled_fp4_experts_quant", TORCH_BOX(&scaled_fp4_experts_quant));
  ops.impl("silu_and_mul_scaled_fp4_experts_quant",
           TORCH_BOX(&silu_and_mul_scaled_fp4_experts_quant));
  ops.impl("silu_and_mul_nvfp4_quant", TORCH_BOX(&silu_and_mul_nvfp4_quant));

  // W4A8 ops: impl registrations are in the source files
  // (w4a8_mm_entry.cu and w4a8_grouped_mm_entry.cu)

  // AWQ ops
  ops.impl("awq_gemm", TORCH_BOX(&awq_gemm));
  ops.impl("awq_dequantize", TORCH_BOX(&awq_dequantize));

  // DSV3 fused A GEMM: conditionally compiled so impl registration is in
  // source file (dsv3_fused_a_gemm.cu)

  // AllSpark ops: conditionally compiled so impl registrations are in source
  // files (allspark_repack.cu and allspark_qgemm_w8a16.cu)
#endif

  ops.impl("merge_attn_states", TORCH_BOX(&merge_attn_states));

  // Layernorm kernels (shared CUDA/ROCm)
  ops.impl("rms_norm", TORCH_BOX(&rms_norm));
  ops.impl("fused_add_rms_norm", TORCH_BOX(&fused_add_rms_norm));

  // Layernorm-quant kernels (shared CUDA/ROCm)
  ops.impl("rms_norm_static_fp8_quant", TORCH_BOX(&rms_norm_static_fp8_quant));
  ops.impl("fused_add_rms_norm_static_fp8_quant",
           TORCH_BOX(&fused_add_rms_norm_static_fp8_quant));

  // Fused layernorm + dynamic per-token quant kernels (shared CUDA/ROCm)
  ops.impl("rms_norm_dynamic_per_token_quant",
           TORCH_BOX(&rms_norm_dynamic_per_token_quant));
  ops.impl("rms_norm_per_block_quant", TORCH_BOX(&rms_norm_per_block_quant));

  // Positional encoding kernels (shared CUDA/ROCm)
  ops.impl("rotary_embedding", TORCH_BOX(&rotary_embedding));
  ops.impl("fused_qk_norm_rope", TORCH_BOX(&fused_qk_norm_rope));

  // Activation kernels (shared CUDA/ROCm)
  ops.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
  ops.impl("mul_and_silu", TORCH_BOX(&mul_and_silu));
  ops.impl("gelu_and_mul", TORCH_BOX(&gelu_and_mul));
  ops.impl("gelu_tanh_and_mul", TORCH_BOX(&gelu_tanh_and_mul));
  ops.impl("fatrelu_and_mul", TORCH_BOX(&fatrelu_and_mul));
  ops.impl("swigluoai_and_mul", TORCH_BOX(&swigluoai_and_mul));
  ops.impl("gelu_new", TORCH_BOX(&gelu_new));
  ops.impl("gelu_fast", TORCH_BOX(&gelu_fast));
  ops.impl("gelu_quick", TORCH_BOX(&gelu_quick));

  // INT8 quantization kernels
  ops.impl("static_scaled_int8_quant", TORCH_BOX(&static_scaled_int8_quant));
  ops.impl("dynamic_scaled_int8_quant", TORCH_BOX(&dynamic_scaled_int8_quant));

  // FP8 quantization kernels
  ops.impl("static_scaled_fp8_quant", TORCH_BOX(&static_scaled_fp8_quant));
  ops.impl("dynamic_scaled_fp8_quant", TORCH_BOX(&dynamic_scaled_fp8_quant));
  ops.impl("dynamic_per_token_scaled_fp8_quant",
           TORCH_BOX(&dynamic_per_token_scaled_fp8_quant));

  // GPTQ kernels
  ops.impl("gptq_gemm", TORCH_BOX(&gptq_gemm));
  ops.impl("gptq_shuffle", TORCH_BOX(&gptq_shuffle));

  // GGML kernels
  ops.impl("ggml_dequantize", TORCH_BOX(&ggml_dequantize));
  ops.impl("ggml_mul_mat_vec_a8", TORCH_BOX(&ggml_mul_mat_vec_a8));
  ops.impl("ggml_mul_mat_a8", TORCH_BOX(&ggml_mul_mat_a8));
  ops.impl("ggml_moe_a8", TORCH_BOX(&ggml_moe_a8));
  ops.impl("ggml_moe_a8_vec", TORCH_BOX(&ggml_moe_a8_vec));
}

// These capability-check functions take only primitive args (no tensors), so
// there is no device to dispatch on. CompositeExplicitAutograd makes them
// available for all backends. This is the stable ABI equivalent of calling
// ops.impl("op_name", &func) without a dispatch key in the non-stable API.
STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, ops) {
#ifndef USE_ROCM
  ops.impl("cutlass_scaled_mm_supports_fp8",
           TORCH_BOX(&cutlass_scaled_mm_supports_fp8));
  ops.impl("cutlass_group_gemm_supported",
           TORCH_BOX(&cutlass_group_gemm_supported));
  ops.impl("cutlass_scaled_mm_supports_block_fp8",
           TORCH_BOX(&cutlass_scaled_mm_supports_block_fp8));
  ops.impl("cutlass_scaled_mm_supports_fp4",
           TORCH_BOX(&cutlass_scaled_mm_supports_fp4));
#endif

  // GGML block size lookup (no tensor args)
  ops.impl("ggml_moe_get_block_size", TORCH_BOX(&ggml_moe_get_block_size));
}

REGISTER_EXTENSION(_C_stable_libtorch)
