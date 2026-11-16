// Instantiates the fused MUSE-Spark qknorm/rope/kv-insert kernels for
// head_dim == 256 only; see fused_muse_spark_qknorm_rope_kv_insert_entry.h.
#include "fused_muse_spark_qknorm_rope_kv_insert_kernel.cuh"
VLLM_MUSE_SPARK_ENTRY(, 256);
