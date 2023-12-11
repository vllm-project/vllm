#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "cpu/cpu_ops.h"
#include "dispatch_utils.h"
#include <torch/extension.h>

void rotary_embedding_dispatch(torch::Tensor &positions, torch::Tensor &query,
                               torch::Tensor &key, int head_size,
                               torch::Tensor &cos_sin_cache, bool is_neox) {
  VLLM_DISPATCH_DEVICES(key.device(), rotary_embedding, positions, query, key, head_size, cos_sin_cache, is_neox);
}

void paged_attention_v1_dispatch(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes) {
  VLLM_DISPATCH_DEVICES(out.device(), paged_attention_v1, out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, context_lens, block_size, max_context_len, alibi_slopes);
}

void paged_attention_v2_dispatch(torch::Tensor &out, torch::Tensor &exp_sums,
    torch::Tensor &max_logits, torch::Tensor &tmp_out, torch::Tensor &query, 
    torch::Tensor &key_cache, torch::Tensor &value_cache, int num_kv_heads, 
    float scale, torch::Tensor &block_tables, torch::Tensor &context_lens, int block_size,
    int max_context_len, const c10::optional<torch::Tensor> &alibi_slopes) {
  VLLM_DISPATCH_DEVICES(out.device(), paged_attention_v2, out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, context_lens, block_size,max_context_len, alibi_slopes);
}

void silu_and_mul_dispatch(torch::Tensor &out, torch::Tensor &input) {
  VLLM_DISPATCH_DEVICES(out.device(), silu_and_mul, out, input);
}

void gelu_new_dispatch(torch::Tensor &out, torch::Tensor &input) {
  VLLM_DISPATCH_DEVICES(out.device(), gelu_new, out, input);
}

void gelu_fast_dispatch(torch::Tensor &out, torch::Tensor &input) {
  VLLM_DISPATCH_DEVICES(out.device(), gelu_fast, out, input);
}

void rms_norm_dispatch(torch::Tensor &out, torch::Tensor &input, torch::Tensor &weight, float epsilon) {
  VLLM_DISPATCH_DEVICES(out.device(), rms_norm, out, input, weight, epsilon);
}

void fused_add_rms_norm_dispatch(torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, float epsilon) {
  VLLM_DISPATCH_DEVICES(input.device(), fused_add_rms_norm, input, residual, weight, epsilon);
}

torch::Tensor awq_gemm_dispatch(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _scaling_factors, torch::Tensor _zeros, int split_k_iters) {
  VLLM_DISPATCH_DEVICES(_in_feats.device(), awq_gemm, _in_feats, _kernel, _scaling_factors, _zeros, split_k_iters);
}

void squeezellm_gemm_dispatch(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul, torch::Tensor lookup_table) {
  VLLM_DISPATCH_DEVICES(vec.device(), squeezellm_gemm, vec, mat, mul, lookup_table);
}

void swap_blocks_dispatch(torch::Tensor& src, torch::Tensor& dst, const std::map<int64_t, int64_t>& block_mapping) {
  VLLM_DISPATCH_DEVICES(src.device(), swap_blocks, src, dst, block_mapping);
}

void copy_blocks_dispatch(std::vector<torch::Tensor>& key_caches, std::vector<torch::Tensor>& value_caches, const std::map<int64_t, std::vector<int64_t>>& block_mapping) {
  VLLM_DISPATCH_DEVICES(key_caches[0].device(), copy_blocks, key_caches, value_caches, block_mapping);
}

void reshape_and_cache_dispatch(torch::Tensor& key, torch::Tensor& value, torch::Tensor& key_cache, torch::Tensor& value_cache, torch::Tensor& slot_mapping) {
  VLLM_DISPATCH_DEVICES(key.device(), reshape_and_cache, key, value, key_cache, value_cache, slot_mapping);
}

void gather_cached_kv_dispatch(torch::Tensor& key, torch::Tensor& value, torch::Tensor& key_cache, torch::Tensor& value_cache, torch::Tensor& slot_mapping) {
  VLLM_DISPATCH_DEVICES(key.device(), gather_cached_kv, key, value, key_cache, value_cache, slot_mapping);
}

torch::Tensor gptq_gemm_dispatch(
  torch::Tensor a,
  torch::Tensor b_q_weight,
  torch::Tensor b_gptq_qzeros,
  torch::Tensor b_gptq_scales,
  torch::Tensor b_g_idx,
  bool use_exllama) {
    VLLM_DISPATCH_DEVICES(a.device(), gptq_gemm, a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_exllama);
}

void gptq_shuffle_dispatch(
  torch::Tensor q_weight,
  torch::Tensor q_perm) {
    VLLM_DISPATCH_DEVICES(q_weight.device(), gptq_shuffle, q_weight, q_perm);
}

#ifdef VLLM_BUILD_CPU_ONLY
int get_device_attribute(
    int attribute,
    int device_id) { return 94387; }
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // vLLM custom ops
  pybind11::module ops = m.def_submodule("ops", "vLLM custom operators");

  // Attention ops
  ops.def(
    "paged_attention_v1",
    &paged_attention_v1_dispatch,
    "Compute the attention between an input query and the cached keys/values using PagedAttention.");
  ops.def(
    "paged_attention_v2",
    &paged_attention_v2_dispatch,
    "PagedAttention V2.");

  // Activation ops
  ops.def(
    "silu_and_mul",
    &silu_and_mul_dispatch,
    "Activation function used in SwiGLU.");
  ops.def(
    "gelu_new",
    &gelu_new_dispatch,
    "GELU implementation used in GPT-2.");
  ops.def(
    "gelu_fast",
    &gelu_fast_dispatch,
    "Approximate GELU implementation.");

  // Layernorm
  ops.def(
    "rms_norm",
    &rms_norm_dispatch,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");

  ops.def(
    "fused_add_rms_norm",
    &fused_add_rms_norm_dispatch,
    "In-place fused Add and RMS Normalization");

  // Rotary embedding
  ops.def(
    "rotary_embedding",
    &rotary_embedding_dispatch,
    "Apply GPT-NeoX or GPT-J style rotary embedding to query and key");

#ifndef USE_ROCM
  // Quantization ops
  ops.def("awq_gemm", &awq_gemm_dispatch, "Quantized GEMM for AWQ");
#endif
  ops.def("gptq_gemm", &gptq_gemm_dispatch, "Quantized GEMM for GPTQ");
  ops.def("gptq_shuffle", &gptq_shuffle_dispatch, "Post processing for GPTQ");
  ops.def("squeezellm_gemm", &squeezellm_gemm_dispatch, "Quantized GEMM for SqueezeLLM");

  // Cache ops
  pybind11::module cache_ops = m.def_submodule("cache_ops", "vLLM cache ops");
  cache_ops.def(
    "swap_blocks",
    &swap_blocks_dispatch,
    "Swap in (out) the cache blocks from src to dst");
  cache_ops.def(
    "copy_blocks",
    &copy_blocks_dispatch,
    "Copy the cache blocks from src to dst");
  cache_ops.def(
    "reshape_and_cache",
    &reshape_and_cache_dispatch,
    "Reshape the key and value tensors and cache them");
  cache_ops.def(
    "gather_cached_kv",
    &gather_cached_kv_dispatch,
    "Gather key and value from the cache into contiguous QKV tensors");

  // Cuda utils
  pybind11::module cuda_utils = m.def_submodule("cuda_utils", "vLLM cuda utils");
  cuda_utils.def(
    "get_device_attribute",
    &get_device_attribute,
    "Gets the specified device attribute.");
}
