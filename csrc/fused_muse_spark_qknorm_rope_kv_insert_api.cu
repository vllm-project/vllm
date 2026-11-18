// Public entry point of the fused MUSE-Spark qknorm/rope/kv-insert op. The
// kernels are instantiated one unit per head_dim (see
// fused_muse_spark_qknorm_rope_kv_insert_hd<n>.cu); this unit only dispatches,
// so it includes the declaration header rather than the kernel header.
#include "fused_muse_spark_qknorm_rope_kv_insert_entry.h"

void fused_muse_spark_qknorm_rope_kv_insert(VLLM_MUSE_SPARK_ENTRY_SIG) {
  switch (head_dim) {
    case 64:
      return fused_muse_spark_qknorm_rope_kv_insert_typed<64>(
          qkv, positions, num_heads, num_kv_heads, head_dim, eps, cos_sin_cache,
          rotary_dim, slot_mapping, kv_cache, block_size, kv_cache_dtype,
          k_scale, v_scale, q_fp8_out, q_scale);
    case 128:
      return fused_muse_spark_qknorm_rope_kv_insert_typed<128>(
          qkv, positions, num_heads, num_kv_heads, head_dim, eps, cos_sin_cache,
          rotary_dim, slot_mapping, kv_cache, block_size, kv_cache_dtype,
          k_scale, v_scale, q_fp8_out, q_scale);
    case 256:
      return fused_muse_spark_qknorm_rope_kv_insert_typed<256>(
          qkv, positions, num_heads, num_kv_heads, head_dim, eps, cos_sin_cache,
          rotary_dim, slot_mapping, kv_cache, block_size, kv_cache_dtype,
          k_scale, v_scale, q_fp8_out, q_scale);
    default:
      STD_TORCH_CHECK(false, "unsupported head_dim: ", head_dim);
  }
}
