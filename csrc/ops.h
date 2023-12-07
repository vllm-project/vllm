#include <torch/extension.h>

void paged_attention_v1(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& head_mapping,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  bool enable_quant = false,
  const float k_scale = 1.0f,
  const float k_zp = 0.0f,
  const float v_scale = 1.0f,
  const float v_zp = 0.0f);

void paged_attention_v2(
  torch::Tensor& out,
  torch::Tensor& exp_sums,
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& head_mapping,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  bool enable_quant = false,
  const float k_scale = 1.0f,
  const float k_zp = 0.0f,
  const float v_scale = 1.0f,
  const float v_zp = 0.0f);

void rms_norm(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon,
  bool use_quant);

void dequant_add_residual_rms_norm_quant(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& gamma,
  float scale, float epsilon);

void dequant_add_residual_rms_norm_quant(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& gamma,
  torch::Tensor& scale,
  float epsilon);

void fused_add_rms_norm(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& weight,
  float epsilon,
  bool use_quant);

void rotary_embedding(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  torch::Tensor& cos_sin_cache,
  bool is_neox,
  torch::Tensor& query_out,
  torch::Tensor& key_out,
  bool use_dequant = false,
  const float query_scale = 1.0f,
  const float key_scale = 1.0f);

void silu_and_mul(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_new(
  torch::Tensor& out,
  torch::Tensor& input);

void gelu_fast(
  torch::Tensor& out,
  torch::Tensor& input);

void dequant_silu_and_mul_quant(
  torch::Tensor& out,
  torch::Tensor& input,
  const float scale_gate,
  const float scale_up,
  const float scale_out);

void dequant_silu_and_mul_quant(
  torch::Tensor& out,
  torch::Tensor& input,
  const float scale_gate,
  const float scale_up,
  torch::Tensor& scale_out,
  torch::Tensor& tmp);

void dequant_add_residual(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& residual,
  float scale);

void dequant_add_residual(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& scale);

void dequant(
  torch::Tensor& out,
  torch::Tensor& input,
  float scale);

void dequant(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& scale);

void quant(
  torch::Tensor& out,
  torch::Tensor& input,
  float scale);

void quant(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& scale);

torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters);

void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table);
