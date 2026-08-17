#include <torch/extension.h>

#include <optional>

void sgl_per_token_group_quant_8bit_v2(
    torch::Tensor input,
    torch::Tensor output_q,
    torch::Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool round_scale,
    bool scale_ue8m0,
    bool fuse_silu_and_mul,
    const std::optional<torch::Tensor>& masked_m);

TORCH_LIBRARY(ds4_alignment, ops) {
  ops.def(
      "per_token_group_quant_8bit_v2(Tensor input, Tensor! output_q, "
      "Tensor! output_s, int group_size, float eps, float min_8bit, "
      "float max_8bit, bool round_scale, bool scale_ue8m0, "
      "bool fuse_silu_and_mul, "
      "Tensor? masked_m) -> ()");
}

TORCH_LIBRARY_IMPL(ds4_alignment, CUDA, ops) {
  ops.impl("per_token_group_quant_8bit_v2",
           &sgl_per_token_group_quant_8bit_v2);
}
