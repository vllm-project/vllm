// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Registers paged attention from _C_stable_libtorch into the _rocm_C
// namespace; the rest of _rocm_C is registered in rocm/torch_bindings.cpp.
#include <torch/csrc/stable/library.h>

#include "ops_stable.h"

STABLE_TORCH_LIBRARY_FRAGMENT(_rocm_C, rocm_ops) {
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type,"
      "                bool use_interleaved_v_cache=False) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_rocm_C, CUDA, rocm_ops) {
  rocm_ops.impl("paged_attention", TORCH_BOX(&paged_attention));
}
