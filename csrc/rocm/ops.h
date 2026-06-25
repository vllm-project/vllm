#pragma once

#include <torch/all.h>

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              torch::Tensor b_g_idx, bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales,
                                   torch::Tensor b_g_idx, bool use_v2_format);

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

// DeepSeek-V4 fused compressors (gfx950 / CDNA4 only). Impls in
// csrc/rocm/dsv4_compress_ops.cpp + dsv4_{csa,hca,indexer}_compress.cu.
void dsv4_csa_compress(torch::Tensor state_cache, int64_t num_actual,
                       torch::Tensor ape, torch::Tensor token_to_req_indices,
                       torch::Tensor positions, torch::Tensor slot_mapping,
                       torch::Tensor block_table, int64_t block_size,
                       torch::Tensor rms_norm_weight, double rms_norm_eps,
                       torch::Tensor cos_sin_cache, torch::Tensor kv_cache,
                       torch::Tensor kv_slot_mapping,
                       int64_t kv_cache_block_size, int64_t scale_dim);

void dsv4_hca_compress(torch::Tensor state_cache, int64_t num_actual,
                       torch::Tensor ape, torch::Tensor token_to_req_indices,
                       torch::Tensor positions, torch::Tensor slot_mapping,
                       torch::Tensor block_table, int64_t block_size,
                       torch::Tensor rms_norm_weight, double rms_norm_eps,
                       torch::Tensor cos_sin_cache, torch::Tensor kv_cache,
                       torch::Tensor kv_slot_mapping,
                       int64_t kv_cache_block_size, int64_t scale_dim,
                       torch::Tensor plan_scratch,
                       torch::Tensor counter_scratch);

void dsv4_indexer_compress(torch::Tensor state_cache, int64_t num_actual,
                           torch::Tensor ape,
                           torch::Tensor token_to_req_indices,
                           torch::Tensor positions, torch::Tensor slot_mapping,
                           torch::Tensor block_table, int64_t block_size,
                           torch::Tensor rms_norm_weight, double rms_norm_eps,
                           torch::Tensor cos_sin_cache, torch::Tensor kv_cache,
                           torch::Tensor kv_slot_mapping,
                           int64_t kv_cache_block_size, int64_t scale_dim,
                           bool use_fp4_cache);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type);
