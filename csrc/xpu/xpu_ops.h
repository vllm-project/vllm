#include <torch/extension.h>

void rotary_embedding(torch::Tensor &positions, torch::Tensor &query,
                          torch::Tensor &key, int head_size,
                          torch::Tensor &cos_sin_cache, bool is_neox);
void batched_rotary_embedding(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  torch::Tensor& cos_sin_cache,
  bool is_neox,
  int rot_dim,
  torch::Tensor& cos_sin_cache_offsets);

void silu_and_mul(torch::Tensor &out, torch::Tensor &input);
void gelu_and_mul(torch::Tensor &out, torch::Tensor &input);

void gelu_new(torch::Tensor &out, torch::Tensor &input);

void gelu_fast(torch::Tensor &out, torch::Tensor &input);


void gelu_tanh_and_mul(
  torch::Tensor& out,
  torch::Tensor& input);

void paged_attention_v1(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes,
    const std::string& kv_cache_dtype, const float kv_scale);

void paged_attention_v2(
    torch::Tensor &out, torch::Tensor &exp_sums, torch::Tensor &max_logits,
    torch::Tensor &tmp_out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes,
    const std::string& kv_cache_dtype, const float kv_scale);

torch::Tensor context_attention_forward_v1(
    torch::Tensor query,  // [num_tokens, num_kv_head, head_dim]
    torch::Tensor key,    // [num_tokens, num_kv_heads * head_size]
    torch::Tensor value,  // [num_tokens, num_kv_heads * head_size]
    torch::Tensor block_tables, torch::Tensor query_start_loc,
    torch::Tensor seq_lens, torch::Tensor context_lens, int max_input_length,
    int max_context_length);

torch::Tensor context_attention_forward_v2(
    torch::Tensor query,  // [num_tokens, num_kv_head, head_dim]
    torch::Tensor key,    // [num_tokens, num_kv_heads * head_size]
    torch::Tensor value,  // [num_tokens, num_kv_heads * head_size]
    torch::Tensor block_tables, torch::Tensor query_start_loc,
    torch::Tensor seq_lens, torch::Tensor context_lens, int max_input_length,
    int max_context_length);

void copy_blocks(
    std::vector<torch::Tensor> &key_caches,
    std::vector<torch::Tensor> &value_caches,
    const std::map<int64_t, std::vector<int64_t>> &block_mapping);

void reshape_and_cache(torch::Tensor &key, torch::Tensor &value,
                           torch::Tensor &key_cache, torch::Tensor &value_cache,
                           torch::Tensor &slot_mapping,
                           const std::string& kv_cache_dtype, const float kv_scale);

void moe_align_block_size(
  torch::Tensor topk_ids,
  int num_experts,
  int block_size,
  torch::Tensor sorted_token_ids,
  torch::Tensor experts_ids,
  torch::Tensor num_tokens_post_pad) {
  TORCH_CHECK(false, "moe_align_block_size is not supported on XPU.");
}
void swap_blocks(torch::Tensor &src, torch::Tensor &dst,
                     const std::map<int64_t, int64_t> &block_mapping);

void gather_cached_kv(torch::Tensor &key, torch::Tensor &value,
                          torch::Tensor &key_cache, torch::Tensor &value_cache,
                          torch::Tensor &slot_mapping);

void convert_fp8_e5m2(torch::Tensor& src_cache, torch::Tensor& dst_cache) {
  TORCH_CHECK(false, "Quantization is not supported on XPU.");
}

void rms_norm(torch::Tensor &out, torch::Tensor &input,
                  torch::Tensor &weight, float epsilon);

void fused_add_rms_norm(torch::Tensor &input, torch::Tensor &residual,
                            torch::Tensor &weight, float epsilon);

torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int split_k_iters) {
  TORCH_CHECK(false, "awq_gemm is not supported on XPU.");                            
}

torch::Tensor marlin_gemm(
    torch::Tensor& a, 
    torch::Tensor& b_q_weight,
    torch::Tensor& b_scales, 
    torch::Tensor& workspace,
    int64_t size_m, 
    int64_t size_n, 
    int64_t size_k) {
  TORCH_CHECK(false, "marlin_gemm is not supported on XPU.");                            
}

torch::Tensor awq_dequantize(torch::Tensor _kernel, 
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int split_k_iters,
    int thx,
    int thy);

void squeezellm_gemm(torch::Tensor vec, torch::Tensor mat,
                         torch::Tensor mul, torch::Tensor lookup_table) {
  TORCH_CHECK(false, "squeezellm_gemm is not supported on XPU.");
}

torch::Tensor gptq_gemm(
  torch::Tensor a,
  torch::Tensor b_q_weight,
  torch::Tensor b_gptq_qzeros,
  torch::Tensor b_gptq_scales,
  torch::Tensor b_g_idx,
  bool use_exllama,
  int bit) {
  TORCH_CHECK(false, "gptq_gemm is not supported on XPU.");
}

void gptq_shuffle(
  torch::Tensor q_weight,
  torch::Tensor q_perm,
  int bit) {
  TORCH_CHECK(false, "gptq_shuffle is not supported on XPU.");
}