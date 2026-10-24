// Instantiates the fused MUSE-Spark qknorm/rope/kv-insert kernels for
// head_dim == 64 only; see FUSED_MUSE_SPARK_LANE in the main file.
#define FUSED_MUSE_SPARK_LANE 64
#define fused_muse_spark_qknorm_rope_kv_insert \
  fused_muse_spark_qknorm_rope_kv_insert_hd64
#include "fused_muse_spark_qknorm_rope_kv_insert_kernel.cu"
