#pragma once

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>

// Gated activation functions (input: [..., 2*d] -> output: [..., d])
void silu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);
void mul_and_silu(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_tanh_and_mul(torch::stable::Tensor& out,
                       torch::stable::Tensor& input);
void fatrelu_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input,
                     double threshold);
void swigluoai_and_mul(torch::stable::Tensor& out, torch::stable::Tensor& input,
                       double alpha, double limit);

// Element-wise activation functions (input: [..., d] -> output: [..., d])
void gelu_new(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_fast(torch::stable::Tensor& out, torch::stable::Tensor& input);
void gelu_quick(torch::stable::Tensor& out, torch::stable::Tensor& input);

// Utility functions
torch::stable::Tensor get_cuda_view_from_cpu_tensor(
    torch::stable::Tensor& cpu_tensor);

#ifndef USE_ROCM
torch::stable::Tensor permute_cols(torch::stable::Tensor const& A,
                                   torch::stable::Tensor const& perm);
#endif

// Layernorm functions
void rms_norm(torch::stable::Tensor& out, torch::stable::Tensor input,
              const torch::stable::Tensor& weight, double epsilon);

void fused_add_rms_norm(torch::stable::Tensor& input,
                        torch::stable::Tensor& residual,
                        torch::stable::Tensor& weight, double epsilon);

// Layernorm + quantization functions
void rms_norm_static_fp8_quant(torch::stable::Tensor& out,
                               const torch::stable::Tensor& input,
                               const torch::stable::Tensor& weight,
                               const torch::stable::Tensor& scale,
                               double epsilon);

void fused_add_rms_norm_static_fp8_quant(torch::stable::Tensor& out,
                                         const torch::stable::Tensor& input,
                                         torch::stable::Tensor& residual,
                                         const torch::stable::Tensor& weight,
                                         const torch::stable::Tensor& scale,
                                         double epsilon);

// Fused layernorm + dynamic per-token quantization
void rms_norm_dynamic_per_token_quant(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor const& weight, torch::stable::Tensor& scales,
    double epsilon, std::optional<torch::stable::Tensor> scale_ub,
    std::optional<torch::stable::Tensor> residual);

// Fused layernorm + block quantization
void rms_norm_per_block_quant(torch::stable::Tensor& out,
                              torch::stable::Tensor const& input,
                              torch::stable::Tensor const& weight,
                              torch::stable::Tensor& scales,
                              double const epsilon,
                              std::optional<torch::stable::Tensor> scale_ub,
                              std::optional<torch::stable::Tensor> residual,
                              int64_t group_size, bool is_scale_transposed);

// Positional encoding functions
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

// Sampler functions
void apply_repetition_penalties_(
    torch::stable::Tensor& logits, torch::stable::Tensor const& prompt_mask,
    torch::stable::Tensor const& output_mask,
    torch::stable::Tensor const& repetition_penalties);

void top_k_per_row_prefill(torch::stable::Tensor const& logits,
                           torch::stable::Tensor const& rowStarts,
                           torch::stable::Tensor const& rowEnds,
                           torch::stable::Tensor& indices, int64_t numRows,
                           int64_t stride0, int64_t stride1, int64_t topK);

void top_k_per_row_decode(torch::stable::Tensor const& logits, int64_t next_n,
                          torch::stable::Tensor const& seqLens,
                          torch::stable::Tensor& indices, int64_t numRows,
                          int64_t stride0, int64_t stride1, int64_t topK);

// Quantized activation functions
void silu_and_mul_quant(torch::stable::Tensor& out,
                        torch::stable::Tensor& input,
                        torch::stable::Tensor& scale);

void persistent_masked_m_silu_mul_quant(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& tokens_per_expert, torch::stable::Tensor& y_q,
    torch::stable::Tensor& y_s, bool cast_scale_ue8m0);

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

// Compute int8 quantized tensor for given scaling factor.
void static_scaled_int8_quant(torch::stable::Tensor& out,
                              const torch::stable::Tensor& input,
                              const torch::stable::Tensor& scale,
                              const std::optional<torch::stable::Tensor>& azp);

// Compute int8 quantized tensor and scaling factor.
void dynamic_scaled_int8_quant(torch::stable::Tensor& out,
                               const torch::stable::Tensor& input,
                               torch::stable::Tensor& scales,
                               const std::optional<torch::stable::Tensor>& azp);

#ifndef USE_ROCM

void per_token_group_quant_fp8(const torch::stable::Tensor& input,
                               torch::stable::Tensor& output_q,
                               torch::stable::Tensor& output_s,
                               int64_t group_size, double eps, double fp8_min,
                               double fp8_max, bool scale_ue8m0);

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
#endif

// Attention kernels
void paged_attention_v1(
    torch::stable::Tensor& out, torch::stable::Tensor& query,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    int64_t num_kv_heads, double scale, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::stable::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, int64_t tp_rank,
    int64_t blocksparse_local_blocks, int64_t blocksparse_vert_stride,
    int64_t blocksparse_block_size, int64_t blocksparse_head_sliding_step);

void paged_attention_v2(
    torch::stable::Tensor& out, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& key_cache,
    torch::stable::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::stable::Tensor& block_tables, torch::stable::Tensor& seq_lens,
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::stable::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, int64_t tp_rank,
    int64_t blocksparse_local_blocks, int64_t blocksparse_vert_stride,
    int64_t blocksparse_block_size, int64_t blocksparse_head_sliding_step);

void merge_attn_states(torch::stable::Tensor& output,
                       std::optional<torch::stable::Tensor> output_lse,
                       const torch::stable::Tensor& prefix_output,
                       const torch::stable::Tensor& prefix_lse,
                       const torch::stable::Tensor& suffix_output,
                       const torch::stable::Tensor& suffix_lse);

#ifndef USE_ROCM
void convert_vertical_slash_indexes(
    torch::stable::Tensor& block_count, torch::stable::Tensor& block_offset,
    torch::stable::Tensor& column_count, torch::stable::Tensor& column_index,
    torch::stable::Tensor q_seqlens, torch::stable::Tensor kv_seqlens,
    torch::stable::Tensor vertical_indexes, torch::stable::Tensor slash_indexes,
    int64_t context_size, int64_t block_size_M, int64_t block_size_N,
    bool causal);

void convert_vertical_slash_indexes_mergehead(
    torch::stable::Tensor& block_count, torch::stable::Tensor& block_offset,
    torch::stable::Tensor& column_count, torch::stable::Tensor& column_index,
    torch::stable::Tensor q_seqlens, torch::stable::Tensor kv_seqlens,
    torch::stable::Tensor vertical_indexes, torch::stable::Tensor slash_indexes,
    torch::stable::Tensor vertical_indices_count,
    torch::stable::Tensor slash_indices_count, int64_t context_size,
    int64_t block_size_M, int64_t block_size_N, bool causal);
#endif
