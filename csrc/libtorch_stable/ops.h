#pragma once

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#ifndef USE_ROCM
torch::stable::Tensor permute_cols(torch::stable::Tensor const& A,
                                   torch::stable::Tensor const& perm);

void per_token_group_quant_fp8(const torch::stable::Tensor& input,
                               torch::stable::Tensor& output_q,
                               torch::stable::Tensor& output_s,
                               int64_t group_size, double eps, double fp8_min,
                               double fp8_max, bool scale_ue8m0,
                               bool dummy_is_scale_transposed,
                               bool dummy_is_tma_aligned);

// Fused activation quantisation + DeepGEMM-compatible UE8M0-packed scales.
void per_token_group_quant_8bit_packed(const torch::stable::Tensor& input,
                                       torch::stable::Tensor& output_q,
                                       torch::stable::Tensor& output_s_packed,
                                       int64_t group_size, double eps,
                                       double min_8bit, double max_8bit);

void per_token_group_quant_int8(const torch::stable::Tensor& input,
                                torch::stable::Tensor& output_q,
                                torch::stable::Tensor& output_s,
                                int64_t group_size, double eps, double int8_min,
                                double int8_max);

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability);
bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability);
bool cutlass_group_gemm_supported(int64_t cuda_device_capability);

void cutlass_scaled_mm(torch::stable::Tensor& out,
                       torch::stable::Tensor const& a,
                       torch::stable::Tensor const& b,
                       torch::stable::Tensor const& a_scales,
                       torch::stable::Tensor const& b_scales,
                       std::optional<torch::stable::Tensor> const& bias);

void cutlass_moe_mm(torch::stable::Tensor& out_tensors,
                    torch::stable::Tensor const& a_tensors,
                    torch::stable::Tensor const& b_tensors,
                    torch::stable::Tensor const& a_scales,
                    torch::stable::Tensor const& b_scales,
                    torch::stable::Tensor const& expert_offsets,
                    torch::stable::Tensor const& problem_sizes,
                    torch::stable::Tensor const& a_strides,
                    torch::stable::Tensor const& b_strides,
                    torch::stable::Tensor const& c_strides, bool per_act_token,
                    bool per_out_ch);

void cutlass_scaled_mm_azp(torch::stable::Tensor& out,
                           torch::stable::Tensor const& a,
                           torch::stable::Tensor const& b,
                           torch::stable::Tensor const& a_scales,
                           torch::stable::Tensor const& b_scales,
                           torch::stable::Tensor const& azp_adj,
                           std::optional<torch::stable::Tensor> const& azp,
                           std::optional<torch::stable::Tensor> const& bias);

void get_cutlass_moe_mm_data(
    const torch::stable::Tensor& topk_ids,
    torch::stable::Tensor& expert_offsets,
    torch::stable::Tensor& problem_sizes1,
    torch::stable::Tensor& problem_sizes2,
    torch::stable::Tensor& input_permutation,
    torch::stable::Tensor& output_permutation, const int64_t num_experts,
    const int64_t n, const int64_t k,
    const std::optional<torch::stable::Tensor>& blockscale_offsets,
    const bool is_gated);

void get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
    const torch::stable::Tensor& expert_first_token_offset,
    torch::stable::Tensor& problem_sizes1,
    torch::stable::Tensor& problem_sizes2, const int64_t n, const int64_t k,
    const bool swap_ab);

void get_cutlass_batched_moe_mm_data(
    torch::stable::Tensor& expert_offsets,
    torch::stable::Tensor& problem_sizes1,
    torch::stable::Tensor& problem_sizes2,
    const torch::stable::Tensor& expert_num_tokens,
    const int64_t num_local_experts, const int64_t padded_m, const int64_t n,
    const int64_t k);

// FP4/NVFP4 ops
bool cutlass_scaled_mm_supports_fp4(int64_t cuda_device_capability);

void cutlass_scaled_fp4_mm(torch::stable::Tensor& D,
                           torch::stable::Tensor const& A,
                           torch::stable::Tensor const& B,
                           torch::stable::Tensor const& A_sf,
                           torch::stable::Tensor const& B_sf,
                           torch::stable::Tensor const& alpha);

void cutlass_fp4_group_mm(torch::stable::Tensor& output,
                          const torch::stable::Tensor& a,
                          const torch::stable::Tensor& b,
                          const torch::stable::Tensor& a_blockscale,
                          const torch::stable::Tensor& b_blockscales,
                          const torch::stable::Tensor& alphas,
                          const torch::stable::Tensor& problem_sizes,
                          const torch::stable::Tensor& expert_offsets,
                          const torch::stable::Tensor& sf_offsets);

std::tuple<torch::stable::Tensor, torch::stable::Tensor> scaled_fp4_quant_func(
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_scale, bool is_sf_swizzled_layout);

void scaled_fp4_quant_out(torch::stable::Tensor const& input,
                          torch::stable::Tensor const& input_scale,
                          bool is_sf_swizzled_layout,
                          torch::stable::Tensor& output,
                          torch::stable::Tensor& output_scale);

void scaled_fp4_experts_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts);

void silu_and_mul_scaled_fp4_experts_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts);

void silu_and_mul_nvfp4_quant(torch::stable::Tensor& out,
                              torch::stable::Tensor& output_block_scale,
                              torch::stable::Tensor& input,
                              torch::stable::Tensor& input_global_scale);

// AWQ ops
torch::stable::Tensor awq_gemm(torch::stable::Tensor _in_feats,
                               torch::stable::Tensor _kernel,
                               torch::stable::Tensor _scaling_factors,
                               torch::stable::Tensor _zeros,
                               int64_t split_k_iters);

torch::stable::Tensor awq_dequantize(torch::stable::Tensor _kernel,
                                     torch::stable::Tensor _scaling_factors,
                                     torch::stable::Tensor _zeros,
                                     int64_t split_k_iters, int64_t thx,
                                     int64_t thy);

// DSV3 fused A GEMM: conditionally compiled so declaration and impl
// registration are in the source file (dsv3_fused_a_gemm.cu)

// AllSpark ops: declarations are in the source files
// (allspark_repack.cu and allspark_qgemm_w8a16.cu)

#endif

// Attention kernels (shared CUDA/ROCm)
void merge_attn_states(
    torch::stable::Tensor& output,
    std::optional<torch::stable::Tensor> output_lse,
    const torch::stable::Tensor& prefix_output,
    const torch::stable::Tensor& prefix_lse,
    const torch::stable::Tensor& suffix_output,
    const torch::stable::Tensor& suffix_lse,
    const std::optional<int64_t> prefill_tokens_with_context,
    const std::optional<torch::stable::Tensor>& output_scale);

torch::stable::Tensor hadacore_transform(torch::stable::Tensor& x,
                                         bool inplace);

// Layernorm kernels (shared CUDA/ROCm)
void rms_norm(torch::stable::Tensor& out, torch::stable::Tensor& input,
              torch::stable::Tensor& weight, double epsilon);

void fused_add_rms_norm(torch::stable::Tensor& input,
                        torch::stable::Tensor& residual,
                        torch::stable::Tensor& weight, double epsilon);

// Layernorm-quant kernels (shared CUDA/ROCm)
void rms_norm_static_fp8_quant(torch::stable::Tensor& out,
                               torch::stable::Tensor& input,
                               torch::stable::Tensor& weight,
                               torch::stable::Tensor& scale, double epsilon);

void fused_add_rms_norm_static_fp8_quant(torch::stable::Tensor& out,
                                         torch::stable::Tensor& input,
                                         torch::stable::Tensor& residual,
                                         torch::stable::Tensor& weight,
                                         torch::stable::Tensor& scale,
                                         double epsilon);

// Fused layernorm + dynamic per-token quant kernels (shared CUDA/ROCm)
void rms_norm_dynamic_per_token_quant(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor const& weight, torch::stable::Tensor& scales,
    double const var_epsilon, std::optional<torch::stable::Tensor> scale_ub,
    std::optional<torch::stable::Tensor> residual);

void rms_norm_per_block_quant(torch::stable::Tensor& out,
                              torch::stable::Tensor const& input,
                              torch::stable::Tensor const& weight,
                              torch::stable::Tensor& scales,
                              double const var_epsilon,
                              std::optional<torch::stable::Tensor> scale_ub,
                              std::optional<torch::stable::Tensor> residual,
                              int64_t group_size, bool is_scale_transposed);

// Positional encoding kernels (shared CUDA/ROCm)
void rotary_embedding(torch::stable::Tensor& positions,
                      torch::stable::Tensor& query,
                      std::optional<torch::stable::Tensor> key,
                      int64_t head_size, torch::stable::Tensor& cos_sin_cache,
                      bool is_neox);

void fused_qk_norm_rope(torch::stable::Tensor& qkv, int64_t num_heads_q,
                        int64_t num_heads_k, int64_t num_heads_v,
                        int64_t head_dim, double eps,
                        torch::stable::Tensor& q_weight,
                        torch::stable::Tensor& k_weight,
                        torch::stable::Tensor& cos_sin_cache, bool is_neox,
                        torch::stable::Tensor& position_ids);

// Activation kernels (shared CUDA/ROCm)
void silu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);
void mul_and_silu(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_tanh_and_mul(torch::stable::Tensor& out,
                       torch::stable::Tensor& input);
void fatrelu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input,
                     double threshold);
void swigluoai_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input,
                       double alpha = 1.702, double limit = 7.0);
void gelu_new(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_fast(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_quick(torch::stable::Tensor& out, torch::stable::Tensor& input);

// INT8 quantization kernels (shared CUDA/ROCm)
void static_scaled_int8_quant(torch::stable::Tensor& out,
                              torch::stable::Tensor const& input,
                              torch::stable::Tensor const& scale,
                              std::optional<torch::stable::Tensor> const& azp);

void dynamic_scaled_int8_quant(torch::stable::Tensor& out,
                               torch::stable::Tensor const& input,
                               torch::stable::Tensor& scales,
                               std::optional<torch::stable::Tensor> const& azp);

// FP8 quantization kernels (shared CUDA/ROCm)
void static_scaled_fp8_quant(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor const& scale,
    std::optional<torch::headeronly::IntHeaderOnlyArrayRef> group_shape =
        std::nullopt);

void dynamic_scaled_fp8_quant(torch::stable::Tensor& out,
                              torch::stable::Tensor const& input,
                              torch::stable::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor& scale,
    std::optional<torch::stable::Tensor> const& scale_ub);

// GPTQ kernels (shared CUDA/ROCm)
torch::stable::Tensor gptq_gemm(torch::stable::Tensor a,
                                torch::stable::Tensor b_q_weight,
                                torch::stable::Tensor b_gptq_qzeros,
                                torch::stable::Tensor b_gptq_scales,
                                torch::stable::Tensor b_g_idx, bool use_exllama,
                                bool use_v2_format, int64_t bit);

void gptq_shuffle(torch::stable::Tensor q_weight, torch::stable::Tensor q_perm,
                  int64_t bit);

// GGML kernels (shared CUDA/ROCm)
torch::stable::Tensor ggml_dequantize(
    torch::stable::Tensor W, int64_t type, int64_t m, int64_t n,
    std::optional<torch::headeronly::ScalarType> const& dtype);

torch::stable::Tensor ggml_mul_mat_vec_a8(torch::stable::Tensor W,
                                          torch::stable::Tensor X, int64_t type,
                                          int64_t row);

torch::stable::Tensor ggml_mul_mat_a8(torch::stable::Tensor W,
                                      torch::stable::Tensor X, int64_t type,
                                      int64_t row);

torch::stable::Tensor ggml_moe_a8(torch::stable::Tensor X,
                                  torch::stable::Tensor W,
                                  torch::stable::Tensor sorted_token_ids,
                                  torch::stable::Tensor expert_ids,
                                  torch::stable::Tensor num_tokens_post_padded,
                                  int64_t type, int64_t row, int64_t top_k,
                                  int64_t tokens);

torch::stable::Tensor ggml_moe_a8_vec(torch::stable::Tensor X,
                                      torch::stable::Tensor W,
                                      torch::stable::Tensor topk_ids,
                                      int64_t top_k, int64_t type, int64_t row,
                                      int64_t tokens);

int64_t ggml_moe_get_block_size(int64_t type);

void apply_repetition_penalties_(
    torch::stable::Tensor& logits, const torch::stable::Tensor& prompt_mask,
    const torch::stable::Tensor& output_mask,
    const torch::stable::Tensor& repetition_penalties);

void top_k_per_row_prefill(const torch::stable::Tensor& logits,
                           const torch::stable::Tensor& rowStarts,
                           const torch::stable::Tensor& rowEnds,
                           torch::stable::Tensor& indices, int64_t numRows,
                           int64_t stride0, int64_t stride1, int64_t topK);

void top_k_per_row_decode(const torch::stable::Tensor& logits, int64_t next_n,
                          const torch::stable::Tensor& seqLens,
                          torch::stable::Tensor& indices, int64_t numRows,
                          int64_t stride0, int64_t stride1, int64_t topK);

void large_context_topk(const torch::stable::Tensor& logits,
                        torch::stable::Tensor& indices,
                        const torch::stable::Tensor& seq_lens,
                        std::optional<torch::stable::Tensor> row_starts);

void paged_attention_v1(
    torch::stable::Tensor& out, torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    int64_t num_kv_heads, double scale, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::stable::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void paged_attention_v2(
    torch::stable::Tensor& out, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& key_cache,
    torch::stable::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::stable::Tensor& block_tables, torch::stable::Tensor& seq_lens,
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::stable::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void selective_scan_fwd(
    const torch::stable::Tensor& u, const torch::stable::Tensor& delta,
    const torch::stable::Tensor& A, const torch::stable::Tensor& B,
    const torch::stable::Tensor& C,
    const std::optional<torch::stable::Tensor>& D_,
    const std::optional<torch::stable::Tensor>& z_,
    const std::optional<torch::stable::Tensor>& delta_bias_,
    bool delta_softplus,
    const std::optional<torch::stable::Tensor>& query_start_loc,
    const std::optional<torch::stable::Tensor>& cache_indices,
    const std::optional<torch::stable::Tensor>& has_initial_state,
    const torch::stable::Tensor& ssm_states, int64_t null_block_id,
    int64_t block_size,
    const std::optional<torch::stable::Tensor>& block_idx_first_scheduled_token,
    const std::optional<torch::stable::Tensor>& block_idx_last_scheduled_token,
    const std::optional<torch::stable::Tensor>& initial_state_idx,
    const std::optional<torch::stable::Tensor>& cu_chunk_seqlen,
    const std::optional<torch::stable::Tensor>& last_chunk_indices);
