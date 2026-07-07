#include "ops.h"
#include "cuda_utils.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

// Register ops with STABLE_TORCH_LIBRARY for libtorch stable ABI compatibility.
// Note: We register under namespace "_C" so ops are accessible as
// torch.ops._C.<op_name> for compatibility with existing code.
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
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
  ops.def("permute_cols(Tensor A, Tensor perm) -> Tensor");

#ifndef USE_ROCM

  // TODO: Remove this once ROCm upgrade to torch 2.11.
  ops.def("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");

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

  // Marlin GEMM
  ops.def(
      "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none,Tensor b_scales, "
      "Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, "
      "Tensor? "
      "g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_type_id, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  // conditionally compiled so impl registrations are in source file

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
#endif

#ifndef USE_ROCM
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

  // cutlass mxfp4 block scaled group GEMM (MXFP4 x MXFP4 MoE)
  ops.def(
      "cutlass_mxfp4_group_mm(Tensor! out, Tensor a, Tensor b,"
      " Tensor a_blockscale, Tensor b_blockscales,"
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

  // Compute MXFP4 experts quantization (32-element blocks, E8M0 SFs).
  ops.def(
      "mxfp4_experts_quant(Tensor! output, Tensor! output_scale,"
      "Tensor input, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts, int n_experts) -> ()");

  // Fused SiLU+Mul+MXFP4 experts quantization.
  ops.def(
      "silu_and_mul_mxfp4_experts_quant(Tensor! output, Tensor! "
      "output_scale,"
      "Tensor input, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts, int n_experts) -> ()");

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

  // BF16/FP32 x FP32 -> FP32 router GEMM for H=3072, E=256, M<=32 (SM90+).
  // conditionally compiled so impl registration is in source file
  ops.def("fp32_router_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()");

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
      "rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) "
      "-> "
      "()");

  // In-place fused Add and RMS Normalization.
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, "
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

  // Fused SiLU+Mul + per-block quantization
  ops.def(
      "silu_and_mul_per_block_quant("
      "Tensor! out, "
      "Tensor input, "
      "Tensor! scales, "
      "int group_size, "
      "Tensor? scale_ub=None, "
      "bool is_scale_transposed=False) -> ()");

  // Rotary embedding
  // Apply GPT-NeoX or GPT-J style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox, int "
      "rope_dim_offset=0, bool inverse=False) -> ()");

  // Function for fused QK Norm and RoPE
  ops.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, "
      "int num_heads_k, int num_heads_v, int head_dim, float eps, "
      "Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, "
      "bool is_neox, Tensor position_ids, "
      "int forced_token_heads_per_warp=-1) -> ()");

  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert("
      "Tensor q_in, Tensor kv, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "int q_head_padded, float eps, int cache_block_size) -> Tensor");

  // FlashInfer V4 full-cache variants: write Q in place (bf16) or to a separate
  // FP8 tensor, and KV into a contiguous 512-wide token-strided cache.
  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert("
      "Tensor! q, Tensor kv, Tensor! k_cache, Tensor slot_mapping, "
      "Tensor position_ids, Tensor cos_sin_cache, float eps, "
      "int cache_block_size) -> ()");
  ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert("
      "Tensor q, Tensor kv, Tensor! q_fp8, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "Tensor fp8_scale, Tensor q_fp8_scale_inv, float eps, "
      "int cache_block_size) -> ()");

#ifndef USE_ROCM
  ops.def(
      "minimax_allreduce_rms_qk("
      "Tensor qkv, Tensor norm_weight_q, Tensor norm_weight_k, "
      "Tensor workspace, int q_size, int kv_size, int rank, int nranks, "
      "float eps) -> (Tensor, Tensor)");
#endif

  // Horizontally-fused MiniMax-M3 QK-norm + partial NeoX RoPE + KV-insert.
  ops.def(
      "fused_minimax_m3_qknorm_rope_kv_insert("
      "Tensor! qkv, Tensor q_norm_weight, Tensor k_norm_weight, "
      "Tensor cos_sin_cache, Tensor positions, int num_heads, "
      "int num_kv_heads, int rotary_dim, float eps, "
      "Tensor? index_q_norm_weight, Tensor? index_k_norm_weight, "
      "int num_index_heads, "
      "Tensor? slot_mapping, Tensor? index_slot_mapping, "
      "Tensor!? kv_cache, Tensor!? index_cache, "
      "int block_size, Tensor!? q_out, Tensor!? index_q_out, "
      "str kv_cache_dtype) -> ()");

  // Apply repetition penalties to logits in-place.
  ops.def(
      "apply_repetition_penalties_(Tensor! logits, Tensor prompt_mask, "
      "Tensor output_mask, Tensor repetition_penalties) -> ()");

  // Optimized top-k per row operations.
  ops.def(
      "top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, "
      "Tensor! indices, int numRows, int stride0, "
      "int stride1, int topK) -> ()");

  ops.def(
      "top_k_per_row_decode(Tensor logits, int next_n, "
      "Tensor seq_lens, Tensor! indices, "
      "int numRows, int stride0, int stride1, int topK) -> ()");

  ops.def(
      "persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
      "Tensor workspace, int k, int max_seq_len) -> ()");

#ifdef VLLM_ENABLE_COOPERATIVE_TOPK
  ops.def(
      "cooperative_topk(Tensor logits, Tensor lengths, Tensor! output, "
      "Tensor workspace, int k, int max_seq_len) -> ()");
#endif

  // Activation ops
  ops.def(
      "persistent_masked_m_silu_mul_quant(Tensor input, Tensor counts, Tensor! "
      "y_q, Tensor! y_s, bool use_ue8m0) -> ()");
  ops.def("weak_ref_tensor(Tensor input) -> Tensor");

  // Activation function used in SwiGLU.
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");

  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");

  // SwiGLU activation with input clamping.
  // alpha scales the sigmoid (gate * sigmoid(alpha * gate)); beta is added to
  // the up half (up + beta). Defaults alpha=1.0, beta=0.0 give silu(gate)*up.
  ops.def(
      "silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit, "
      "float alpha=1.0, float beta=0.0) -> ()");

  // SwiGLU activation with FP8 quantization.
  ops.def(
      "silu_and_mul_quant(Tensor! result, Tensor input, Tensor scale) -> ()");

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
      "int null_block_id,"
      "int block_size,"
      "Tensor? block_idx_first_scheduled_token,"
      "Tensor? block_idx_last_scheduled_token,"
      "Tensor? initial_state_idx,"
      "Tensor? cu_chunk_seqlen,"
      "Tensor? last_chunk_indices) -> ()");

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
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  // Per-token group quantization
  ops.impl("per_token_group_fp8_quant", TORCH_BOX(&per_token_group_quant_fp8));
  ops.impl("per_token_group_fp8_quant_packed",
           TORCH_BOX(&per_token_group_quant_8bit_packed));
  ops.impl("per_token_group_quant_int8",
           TORCH_BOX(&per_token_group_quant_int8));

  ops.impl("permute_cols", TORCH_BOX(&permute_cols));

#ifndef USE_ROCM
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
  // mxfp4_experts_quant: registered in mxfp4_experts_quant.cu (SM100 only).
  // W4A8 ops: registered in w4a8_mm_entry.cu / w4a8_grouped_mm_entry.cu.

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
  ops.impl("silu_and_mul_per_block_quant",
           TORCH_BOX(&silu_and_mul_per_block_quant));

  // Positional encoding kernels (shared CUDA/ROCm)
  ops.impl("rotary_embedding", TORCH_BOX(&rotary_embedding));
  ops.impl("fused_qk_norm_rope", TORCH_BOX(&fused_qk_norm_rope));
  ops.impl("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
           TORCH_BOX(&fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert));
  ops.impl(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert",
      TORCH_BOX(&fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert));
  ops.impl(
      "fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert",
      TORCH_BOX(&fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert));
#ifndef USE_ROCM
  ops.impl("minimax_allreduce_rms_qk", TORCH_BOX(&minimax_allreduce_rms_qk));
#endif
  ops.impl("fused_minimax_m3_qknorm_rope_kv_insert",
           TORCH_BOX(&fused_minimax_m3_qknorm_rope_kv_insert));

  // Sampler kernels (shared CUDA/ROCm)
  ops.impl("apply_repetition_penalties_",
           TORCH_BOX(&apply_repetition_penalties_));
  ops.impl("top_k_per_row_prefill", TORCH_BOX(&top_k_per_row_prefill));
  ops.impl("top_k_per_row_decode", TORCH_BOX(&top_k_per_row_decode));
  ops.impl("persistent_topk", TORCH_BOX(&persistent_topk));
#ifdef VLLM_ENABLE_COOPERATIVE_TOPK
  ops.impl("cooperative_topk", TORCH_BOX(&cooperative_topk));
#endif

  // Activation kernels (shared CUDA/ROCm)
  ops.impl("persistent_masked_m_silu_mul_quant",
           TORCH_BOX(&persistent_masked_m_silu_mul_quant));
  ops.impl("weak_ref_tensor", TORCH_BOX(&weak_ref_tensor));
  ops.impl("silu_and_mul_quant", TORCH_BOX(&silu_and_mul_quant));
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

  // Mamba kernels
  ops.impl("selective_scan_fwd", TORCH_BOX(&selective_scan_fwd));

  ops.impl("paged_attention_v1", TORCH_BOX(&paged_attention_v1));
  ops.impl("paged_attention_v2", TORCH_BOX(&paged_attention_v2));
}

// TODO: Remove this once ROCm upgrade to torch 2.11.
#ifndef USE_ROCM
STABLE_TORCH_LIBRARY_IMPL(_C, CPU, ops) {
  ops.impl("get_cuda_view_from_cpu_tensor",
           TORCH_BOX(&get_cuda_view_from_cpu_tensor));
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C_cuda_utils, cuda_utils) {
  cuda_utils.def("get_device_attribute(int attribute, int device_id) -> int");
  cuda_utils.def(
      "get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(_C_cuda_utils, CompositeExplicitAutograd,
                          cuda_utils) {
  cuda_utils.impl("get_device_attribute", TORCH_BOX(&get_device_attribute));
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute",
                  TORCH_BOX(&get_max_shared_memory_per_block_device_attribute));
}

#endif

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
}

// Cache ops
STABLE_TORCH_LIBRARY_FRAGMENT(_C_cache_ops, ops) {
  // Swap in (out) the cache blocks from src to dst.
  ops.def(
      "swap_blocks(Tensor src, Tensor! dst,"
      "            int block_size_in_bytes, Tensor block_mapping) -> ()");

  // Batch swap: submit all block copies in a single driver call.
  ops.def(
      "swap_blocks_batch(Tensor src_ptrs, Tensor dst_ptrs,"
      "                  Tensor sizes,"
      "                  bool is_src_access_order_any=False) -> ()");

  // Reshape the key and value tensors and cache them.
  ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale) -> ()");

  // Reshape the key and value tensors and cache them.
  ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");

  // Concat kv_c and k_pe and cache them.
  ops.def(
      "concat_and_cache_mla(Tensor kv_c, Tensor k_pe,"
      "                     Tensor! kv_cache,"
      "                     Tensor slot_mapping,"
      "                     str kv_cache_dtype,"
      "                     Tensor scale) -> ()");

  // Rotate Q and K, then write to kv cache for MLA
  ops.def(
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

  // Convert the key and value cache to fp8 data type.
  ops.def(
      "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
      "str kv_cache_dtype) -> ()");

  // Gather cache blocks from src_cache to dst, dequantizing from
  // src_cache's dtype to dst's dtype if necessary.
  ops.def(
      "gather_and_maybe_dequant_cache(Tensor src_cache, Tensor! dst, "
      "                               Tensor block_table, Tensor cu_seq_lens, "
      "                               Tensor token_to_seq, "
      "                               int num_tokens, "
      "                               str kv_cache_dtype, "
      "                               Tensor scale, Tensor? seq_starts) -> ()");

  ops.def(
      "cp_gather_cache(Tensor src_cache, Tensor! dst, Tensor block_table, "
      "Tensor cu_seq_lens, int batch_size, Tensor? seq_starts) -> ()");

  ops.def(
      "cp_gather_and_upconvert_fp8_kv_cache(Tensor src_cache, Tensor! dst, "
      "Tensor block_table, Tensor seq_lens, Tensor workspace_starts, int "
      "batch_size) -> ()");

  ops.def(
      "indexer_k_quant_and_cache(Tensor k, Tensor! kv_cache, Tensor "
      "slot_mapping, "
      "int quant_block_size, str kv_cache_dtype) -> ()");

  ops.def("concat_mla_q(Tensor ql_nope, Tensor q_pe, Tensor! q_out) -> ()");

  ops.def(
      "cp_gather_indexer_k_quant_cache(Tensor kv_cache, Tensor! dst_k, Tensor! "
      "dst_scale, Tensor block_table, Tensor cu_seq_lens) -> ()");
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C_custom_ar, custom_ar) {
  custom_ar.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool fully_connected) -> int");
  custom_ar.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ar.def("dispose(int fa) -> ()");
  custom_ar.def("custom_ar_close(int fa) -> ()");
  custom_ar.def("meta_size() -> int");
  custom_ar.def("register_buffer(int fa, int[] ipc_tensors) -> ()");
  custom_ar.def("get_graph_buffer_ipc_meta(int fa) -> (int[], int[])");
  custom_ar.def(
      "get_graph_buffer_ipc_meta_for_reinit(int fa) -> (int[], int[])");
  custom_ar.def(
      "register_graph_buffers(int fa, int[][] handles, int[][] offsets) -> ()");
  custom_ar.def("custom_ar_prepare_for_suspend(int fa) -> ()");
  custom_ar.def(
      "custom_ar_reinit_after_resume(int fa, int[][] handles, "
      "int[][] offsets, int[] signal_ptrs, int[] buffer_ptrs) -> ()");
  custom_ar.def("allocate_shared_buffer_and_handle(int size) -> (int, Tensor)");
  custom_ar.def("open_mem_handle(Tensor mem_handle) -> int");
  custom_ar.def("get_mem_handle(int ptr) -> Tensor");
  custom_ar.def("close_mem_handle(int ptr) -> ()");
  custom_ar.def("free_shared_buffer(int ptr) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CUDA, custom_ar) {
  custom_ar.impl("init_custom_ar", TORCH_BOX(&init_custom_ar));
  custom_ar.impl("all_reduce", TORCH_BOX(&all_reduce));
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CPU, custom_ar) {
  custom_ar.impl("open_mem_handle", TORCH_BOX(&open_mem_handle));
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CompositeExplicitAutograd, custom_ar) {
  custom_ar.impl("dispose", TORCH_BOX(&dispose));
  custom_ar.impl("custom_ar_close", TORCH_BOX(&custom_ar_close));
  custom_ar.impl("meta_size", TORCH_BOX(&meta_size));
  custom_ar.impl("register_buffer", TORCH_BOX(&register_buffer));
  custom_ar.impl("get_graph_buffer_ipc_meta",
                 TORCH_BOX(&get_graph_buffer_ipc_meta));
  custom_ar.impl("get_graph_buffer_ipc_meta_for_reinit",
                 TORCH_BOX(&get_graph_buffer_ipc_meta_for_reinit));
  custom_ar.impl("register_graph_buffers", TORCH_BOX(&register_graph_buffers));
  custom_ar.impl("custom_ar_prepare_for_suspend",
                 TORCH_BOX(&custom_ar_prepare_for_suspend));
  custom_ar.impl("custom_ar_reinit_after_resume",
                 TORCH_BOX(&custom_ar_reinit_after_resume));
  custom_ar.impl("allocate_shared_buffer_and_handle",
                 TORCH_BOX(&allocate_shared_buffer_and_handle));
  custom_ar.impl("get_mem_handle", TORCH_BOX(&get_mem_handle));
  custom_ar.impl("close_mem_handle", TORCH_BOX(&close_mem_handle));
  custom_ar.impl("free_shared_buffer", TORCH_BOX(&free_shared_buffer));
}

STABLE_TORCH_LIBRARY_IMPL(_C_cache_ops, CPU, ops) {
  ops.impl("swap_blocks_batch", TORCH_BOX(&swap_blocks_batch));
}

STABLE_TORCH_LIBRARY_IMPL(_C_cache_ops, CUDA, ops) {
  ops.impl("swap_blocks", TORCH_BOX(&swap_blocks));
  ops.impl("reshape_and_cache", TORCH_BOX(&reshape_and_cache));
  ops.impl("reshape_and_cache_flash", TORCH_BOX(&reshape_and_cache_flash));
  ops.impl("concat_and_cache_mla", TORCH_BOX(&concat_and_cache_mla));
  ops.impl("concat_and_cache_mla_rope_fused",
           TORCH_BOX(&concat_and_cache_mla_rope_fused));
  ops.impl("convert_fp8", TORCH_BOX(&convert_fp8));
  ops.impl("gather_and_maybe_dequant_cache",
           TORCH_BOX(&gather_and_maybe_dequant_cache));
  ops.impl("cp_gather_cache", TORCH_BOX(&cp_gather_cache));
  ops.impl("cp_gather_and_upconvert_fp8_kv_cache",
           TORCH_BOX(&cp_gather_and_upconvert_fp8_kv_cache));
  ops.impl("indexer_k_quant_and_cache", TORCH_BOX(&indexer_k_quant_and_cache));
  ops.impl("concat_mla_q", TORCH_BOX(&concat_mla_q));
  ops.impl("cp_gather_indexer_k_quant_cache",
           TORCH_BOX(&cp_gather_indexer_k_quant_cache));
}

REGISTER_EXTENSION(_C_stable_libtorch)
