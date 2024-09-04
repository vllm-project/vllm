#include "core/registration.h"
#include "custom_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, custom_ops) {
  custom_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, Tensor! out_c, int rows_per_block) -> "
      "()");
  custom_ops.impl("LLMM1", torch::kCUDA, &LLMM1);
  custom_ops.def(
      "LLMM_Silu(Tensor in_a, Tensor in_b, Tensor! out_c, int rows_per_block) "
      "-> ()");
  custom_ops.impl("LLMM_Silu", torch::kCUDA, &LLMM_Silu);
  custom_ops.def(
      "paged_attention_custom(Tensor! out, Tensor exp_sums,"
      "                       Tensor max_logits, Tensor tmp_out,"
      "                       Tensor query, Tensor key_cache,"
      "                       Tensor value_cache, int num_kv_heads,"
      "                       float scale, Tensor block_tables,"
      "                       Tensor context_lens, int block_size,"
      "                       int max_context_len,"
      "                       Tensor? alibi_slopes,"
      "                       str kv_cache_dtype) -> ()");
  custom_ops.impl("paged_attention_custom", torch::kCUDA,
                  &paged_attention_custom);
  custom_ops.def(
      "wvSpltK(Tensor in_a, Tensor in_b, Tensor! out_c, int N_in,"
      "        int CuCount) -> ()");
  custom_ops.impl("wvSpltK", torch::kCUDA, &wvSpltK);
}
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
