#pragma once

#include <torch/all.h>

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor mxfp4_w4a8_gemv_rdna3(const at::Tensor& a, const at::Tensor& b_q,
                                    const at::Tensor& b_scale);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales, bool use_v2_format);

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type, bool use_interleaved_v_cache);

// RDNA3.5 only: Fused Gated Delta Net prefill kernel.
// Mutates `out` and `final_state` in place.
void rdna35_gdn_chunked(torch::Tensor& q, torch::Tensor& k, torch::Tensor& v,
                        torch::Tensor& g, torch::Tensor& beta,
                        std::optional<torch::Tensor> initial_state,
                        torch::Tensor& cu_seqlens, torch::Tensor& out,
                        torch::Tensor& final_state, double scale);

// HipCustomCommunicator
// (vllm/distributed/device_communicators/hip_custom_communicator.py).
std::tuple<std::optional<int64_t>, int64_t, std::string>
hip_custom_communicator_open(const std::string& group, int64_t device,
                             int64_t max_rows, int64_t max_cols,
                             double timeout_seconds);
std::optional<int64_t> hip_custom_communicator_register_graph(int64_t state);
std::optional<int64_t> hip_custom_communicator_close(int64_t state, bool alone);
std::optional<int64_t> hip_custom_communicator_all_reduce(
    int64_t state, torch::Tensor& input, torch::Tensor& output,
    const std::string& algorithm, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization, int64_t pack_bytes);
std::optional<int64_t> hip_custom_communicator_all_reduce_add_rms_norm(
    int64_t state, torch::Tensor& input,
    const std::optional<torch::Tensor>& residual, torch::Tensor& weight,
    torch::Tensor& output, const std::optional<torch::Tensor>& residual_output,
    double eps, const std::string& algorithm, int64_t grid_x, int64_t grid_y,
    int64_t grid_z, int64_t block_x, int64_t block_y, int64_t block_z,
    int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> add_rms_norm_grid_x,
    std::optional<int64_t> add_rms_norm_grid_y,
    std::optional<int64_t> add_rms_norm_grid_z,
    std::optional<int64_t> add_rms_norm_block_x,
    std::optional<int64_t> add_rms_norm_block_y,
    std::optional<int64_t> add_rms_norm_block_z,
    std::optional<int64_t> add_rms_norm_waves_per_eu,
    std::optional<int64_t> add_rms_norm_tile_m,
    std::optional<int64_t> add_rms_norm_tile_n,
    std::optional<int64_t> add_rms_norm_pack_bytes);
std::optional<int64_t> hip_custom_communicator_all_reduce_add_attn_res_rms_norm(
    int64_t state, torch::Tensor& input,
    const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks,
    torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight,
    torch::Tensor& prefix_output, torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output, int64_t num_blocks,
    double eps, double output_eps, const std::string& algorithm, int64_t grid_x,
    int64_t grid_y, int64_t grid_z, int64_t block_x, int64_t block_y,
    int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    int64_t tile_k, const std::string& quantization, int64_t pack_bytes,
    std::optional<int64_t> add_attn_res_rms_norm_grid_x,
    std::optional<int64_t> add_attn_res_rms_norm_grid_y,
    std::optional<int64_t> add_attn_res_rms_norm_grid_z,
    std::optional<int64_t> add_attn_res_rms_norm_block_x,
    std::optional<int64_t> add_attn_res_rms_norm_block_y,
    std::optional<int64_t> add_attn_res_rms_norm_block_z,
    std::optional<int64_t> add_attn_res_rms_norm_waves_per_eu,
    std::optional<int64_t> add_attn_res_rms_norm_tile_m,
    std::optional<int64_t> add_attn_res_rms_norm_tile_n,
    std::optional<int64_t> add_attn_res_rms_norm_tile_k,
    std::optional<int64_t> add_attn_res_rms_norm_pack_bytes);
