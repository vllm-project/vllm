#pragma once

#include <torch/library.h>

void paged_attention_v1(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double kv_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void paged_attention_v2(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double kv_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      torch::Tensor& key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);

void batched_rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                              torch::Tensor& key, int64_t head_size,
                              torch::Tensor& cos_sin_cache, bool is_neox,
                              int64_t rot_dim,
                              torch::Tensor& cos_sin_cache_offsets);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

#ifndef USE_ROCM
torch::Tensor aqlm_gemm(const torch::Tensor& input, const torch::Tensor& codes,
                        const torch::Tensor& codebooks,
                        const torch::Tensor& scales,
                        const torch::Tensor& codebook_partition_sizes,
                        const std::optional<torch::Tensor>& bias);

torch::Tensor aqlm_dequant(const torch::Tensor& codes,
                           const torch::Tensor& codebooks,
                           const torch::Tensor& codebook_partition_sizes);

torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters);

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy);

torch::Tensor marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                          torch::Tensor& b_scales, torch::Tensor& workspace,
                          int64_t size_m, int64_t size_n, int64_t size_k);

torch::Tensor gptq_marlin_24_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                                  torch::Tensor& b_meta,
                                  torch::Tensor& b_scales,
                                  torch::Tensor& workspace, int64_t num_bits,
                                  int64_t size_m, int64_t size_n,
                                  int64_t size_k);

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& g_idx,
                               torch::Tensor& perm, torch::Tensor& workspace,
                               int64_t num_bits, int64_t size_m, int64_t size_n,
                               int64_t size_k, bool is_k_full);

torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
                                 int64_t size_k, int64_t size_n,
                                 int64_t num_bits);

void cutlass_scaled_mm_dq(torch::Tensor& out, torch::Tensor const& a,
                          torch::Tensor const& b, torch::Tensor const& a_scales,
                          torch::Tensor const& b_scales);

#endif

void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale);

void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales);

void squeezellm_gemm(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
                     torch::Tensor lookup_table);

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, int64_t bit);

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit);

void static_scaled_fp8_quant(torch::Tensor& out, torch::Tensor& input,
                             torch::Tensor& scale);

void dynamic_scaled_fp8_quant(torch::Tensor& out, torch::Tensor& input,
                              torch::Tensor& scale);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);

#ifndef USE_ROCM
using fptr_t = int64_t;
fptr_t init_custom_ar(torch::Tensor& meta, torch::Tensor& rank_data,
                      const std::vector<std::string>& handles,
                      const std::vector<int64_t>& offsets, int64_t rank,
                      bool full_nvlink);
bool should_custom_ar(torch::Tensor& inp, int64_t max_size, int64_t world_size,
                      bool full_nvlink);
void all_reduce_reg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out);
void all_reduce_unreg(fptr_t _fa, torch::Tensor& inp, torch::Tensor& reg_buffer,
                      torch::Tensor& out);
void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, torch::Tensor& t,
                     const std::vector<std::string>& handles,
                     const std::vector<int64_t>& offsets);
std::tuple<torch::Tensor, std::vector<int64_t>> get_graph_buffer_ipc_meta(
    fptr_t _fa);
void register_graph_buffers(fptr_t _fa, const std::vector<std::string>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);
#endif
