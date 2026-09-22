#include <torch/extension.h>

#include <optional>

namespace vllm::batch_invariant {

void top_k_per_row_prefill(const at::Tensor& logits,
                          const at::Tensor& row_starts,
                          const at::Tensor& row_ends, at::Tensor& indices,
                          int64_t num_rows, int64_t stride0, int64_t stride1,
                          int64_t top_k);

void fused_silu_mul_per_token_group_quant(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    double clamp_limit,
    bool round_scale,
    bool scale_ue8m0,
    bool fuse_silu_and_mul,
    const std::optional<torch::Tensor>& masked_m);

void combine_topk_swa_decode(torch::Tensor& combined_indices,
                             torch::Tensor& combined_lens,
                             const torch::Tensor& topk_indices,
                             const torch::Tensor& seq_lens,
                             const torch::Tensor& is_valid, int64_t M,
                             int64_t N, int64_t top_k, int64_t compress_ratio,
                             int64_t window_size);

void combine_c128_swa_decode(torch::Tensor& combined_indices,
                             torch::Tensor& combined_lens,
                             const torch::Tensor& seq_lens,
                             const torch::Tensor& is_valid, int64_t M,
                             int64_t N, int64_t top_k, int64_t compress_ratio,
                             int64_t window_size);

}  // namespace vllm::batch_invariant

TORCH_LIBRARY(vllm_batch_invariant, ops) {
  ops.def(
      "fused_silu_mul_per_token_group_quant(Tensor input, Tensor! output_q, "
      "Tensor! output_s, int group_size, float eps, float min_8bit, "
      "float max_8bit, float clamp_limit, bool round_scale, bool scale_ue8m0, "
      "bool fuse_silu_and_mul, Tensor? masked_m) -> ()");
  ops.def(
      "top_k_per_row_prefill(Tensor logits, Tensor row_starts, "
      "Tensor row_ends, Tensor(a!) indices, int num_rows, int stride0, "
      "int stride1, int top_k) -> ()");
  ops.def(
      "combine_topk_swa_decode(Tensor(a!) combined_indices, "
      "Tensor(b!) combined_lens, Tensor topk_indices, Tensor seq_lens, "
      "Tensor is_valid, int M, int N, int top_k, int compress_ratio, "
      "int window_size) -> ()");
  ops.def(
      "combine_c128_swa_decode(Tensor(a!) combined_indices, "
      "Tensor(b!) combined_lens, Tensor seq_lens, Tensor is_valid, int M, "
      "int N, int top_k, int compress_ratio, int window_size) -> ()");
}

TORCH_LIBRARY_IMPL(vllm_batch_invariant, CUDA, ops) {
  ops.impl("fused_silu_mul_per_token_group_quant",
           &vllm::batch_invariant::fused_silu_mul_per_token_group_quant);
  ops.impl("top_k_per_row_prefill",
           &vllm::batch_invariant::top_k_per_row_prefill);
  ops.impl("combine_topk_swa_decode",
           &vllm::batch_invariant::combine_topk_swa_decode);
  ops.impl("combine_c128_swa_decode",
           &vllm::batch_invariant::combine_c128_swa_decode);
}
