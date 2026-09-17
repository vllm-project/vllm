#include <torch/extension.h>

#include <optional>

namespace vllm::batch_invariant {

void fused_silu_mul_per_token_group_quant(
    torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s,
    int64_t group_size, double eps, double min_8bit, double max_8bit,
    double clamp_limit, bool round_scale, bool scale_ue8m0,
    bool fuse_silu_and_mul, const std::optional<torch::Tensor>& masked_m);

void top_k_per_row_prefill(const torch::Tensor& logits,
                           const torch::Tensor& row_starts,
                           const torch::Tensor& row_ends,
                           torch::Tensor& indices, int64_t num_rows,
                           int64_t stride0, int64_t stride1, int64_t top_k);

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
}

TORCH_LIBRARY_IMPL(vllm_batch_invariant, CUDA, ops) {
  ops.impl("fused_silu_mul_per_token_group_quant",
           &vllm::batch_invariant::fused_silu_mul_per_token_group_quant);
  ops.impl("top_k_per_row_prefill",
           &vllm::batch_invariant::top_k_per_row_prefill);
}
