// Entry point of the fused MUSE-Spark qknorm/rope/kv-insert op, forwarding to
// the translation unit that instantiates the kernels for this head_dim. The
// per-lane units define the same entry under a suffixed name (see
// fused_muse_spark_qknorm_rope_kv_insert_hd<n>.cu), so the device code of each
// head_dim is generated in its own unit instead of all three landing on one
// core.
#include "torch_utils.h"

#include <optional>
#include <string>

void fused_muse_spark_qknorm_rope_kv_insert_hd64(
    torch::stable::Tensor& qkv, torch::stable::Tensor const& positions,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim, double eps,
    std::optional<torch::stable::Tensor> cos_sin_cache, int64_t rotary_dim,
    std::optional<torch::stable::Tensor> slot_mapping,
    std::optional<torch::stable::Tensor> kv_cache, int64_t block_size,
    const std::string& kv_cache_dtype,
    std::optional<torch::stable::Tensor> k_scale,
    std::optional<torch::stable::Tensor> v_scale,
    std::optional<torch::stable::Tensor> q_fp8_out,
    std::optional<torch::stable::Tensor> q_scale);
void fused_muse_spark_qknorm_rope_kv_insert_hd128(
    torch::stable::Tensor& qkv, torch::stable::Tensor const& positions,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim, double eps,
    std::optional<torch::stable::Tensor> cos_sin_cache, int64_t rotary_dim,
    std::optional<torch::stable::Tensor> slot_mapping,
    std::optional<torch::stable::Tensor> kv_cache, int64_t block_size,
    const std::string& kv_cache_dtype,
    std::optional<torch::stable::Tensor> k_scale,
    std::optional<torch::stable::Tensor> v_scale,
    std::optional<torch::stable::Tensor> q_fp8_out,
    std::optional<torch::stable::Tensor> q_scale);
void fused_muse_spark_qknorm_rope_kv_insert_hd256(
    torch::stable::Tensor& qkv, torch::stable::Tensor const& positions,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim, double eps,
    std::optional<torch::stable::Tensor> cos_sin_cache, int64_t rotary_dim,
    std::optional<torch::stable::Tensor> slot_mapping,
    std::optional<torch::stable::Tensor> kv_cache, int64_t block_size,
    const std::string& kv_cache_dtype,
    std::optional<torch::stable::Tensor> k_scale,
    std::optional<torch::stable::Tensor> v_scale,
    std::optional<torch::stable::Tensor> q_fp8_out,
    std::optional<torch::stable::Tensor> q_scale);

void fused_muse_spark_qknorm_rope_kv_insert(
    torch::stable::Tensor& qkv, torch::stable::Tensor const& positions,
    int64_t num_heads, int64_t num_kv_heads, int64_t head_dim, double eps,
    std::optional<torch::stable::Tensor> cos_sin_cache, int64_t rotary_dim,
    std::optional<torch::stable::Tensor> slot_mapping,
    std::optional<torch::stable::Tensor> kv_cache, int64_t block_size,
    const std::string& kv_cache_dtype,
    std::optional<torch::stable::Tensor> k_scale,
    std::optional<torch::stable::Tensor> v_scale,
    std::optional<torch::stable::Tensor> q_fp8_out,
    std::optional<torch::stable::Tensor> q_scale) {
  switch (head_dim) {
    case 64:
      return fused_muse_spark_qknorm_rope_kv_insert_hd64(
          qkv, positions, num_heads, num_kv_heads, head_dim, eps, cos_sin_cache,
          rotary_dim, slot_mapping, kv_cache, block_size, kv_cache_dtype,
          k_scale, v_scale, q_fp8_out, q_scale);
    case 128:
      return fused_muse_spark_qknorm_rope_kv_insert_hd128(
          qkv, positions, num_heads, num_kv_heads, head_dim, eps, cos_sin_cache,
          rotary_dim, slot_mapping, kv_cache, block_size, kv_cache_dtype,
          k_scale, v_scale, q_fp8_out, q_scale);
    case 256:
      return fused_muse_spark_qknorm_rope_kv_insert_hd256(
          qkv, positions, num_heads, num_kv_heads, head_dim, eps, cos_sin_cache,
          rotary_dim, slot_mapping, kv_cache, block_size, kv_cache_dtype,
          k_scale, v_scale, q_fp8_out, q_scale);
    default:
      STD_TORCH_CHECK(false, "unsupported head_dim: ", head_dim);
  }
}
