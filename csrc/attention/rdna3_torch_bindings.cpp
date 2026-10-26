// Op registration for the RDNA3 (gfx1100) per-token-head INT8/INT4 attention
// kernels (legacy _C extension, full torch API).
#include <torch/all.h>
#include <torch/library.h>

#include "core/registration.h"

void paged_prefill_attn_rdna3_int8(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_chunk,
    torch::Tensor v_chunk, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor k_scale_cache, torch::Tensor v_scale_cache,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, int64_t max_query_len, double sm_scale,
    bool causal);

void paged_prefill_attn_rdna3_int4(
    torch::Tensor& out, torch::Tensor q, torch::Tensor k_cache,
    torch::Tensor v_cache, torch::Tensor k_scale_cache,
    torch::Tensor v_scale_cache, torch::Tensor rht_signs,
    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
    torch::Tensor seq_lens, int64_t max_query_len, double sm_scale,
    bool causal);

// Fused RHT + INT4 quantize + nibble pack for RDNA3 reshape_and_cache.
void reshape_cache_int4_rdna3(torch::Tensor key, torch::Tensor value,
                              torch::Tensor key_cache,
                              torch::Tensor value_cache,
                              torch::Tensor k_scale_cache,
                              torch::Tensor v_scale_cache,
                              torch::Tensor rht_signs,
                              torch::Tensor slot_mapping);

// Inplace RHT butterfly for decode Q rotation / output unrotation.
void rht_rotate_inplace_rdna3(torch::Tensor data, torch::Tensor rht_signs,
                              bool inverse, double post_scale);

void pth_decode_int4_rdna3(torch::Tensor out, torch::Tensor query,
                           torch::Tensor key_cache, torch::Tensor value_cache,
                           torch::Tensor k_scale_cache,
                           torch::Tensor v_scale_cache, torch::Tensor rht_signs,
                           torch::Tensor block_table, torch::Tensor q_to_req,
                           torch::Tensor q_to_klen, torch::Tensor mid_o_buf,
                           double sm_scale, int64_t num_kv_splits);

void pth_decode_int8_rdna3(torch::Tensor out, torch::Tensor query,
                           torch::Tensor key_cache, torch::Tensor value_cache,
                           torch::Tensor k_scale_cache,
                           torch::Tensor v_scale_cache,
                           torch::Tensor block_table, torch::Tensor q_to_req,
                           torch::Tensor q_to_klen, torch::Tensor mid_o_buf,
                           double sm_scale, int64_t num_kv_splits);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "paged_prefill_attn_rdna3_int8(Tensor! out, Tensor q, Tensor k_chunk, "
      "Tensor v_chunk, Tensor k_cache, Tensor v_cache, "
      "Tensor k_scale_cache, Tensor v_scale_cache, Tensor block_table, "
      "Tensor cu_seqlens_q, Tensor seq_lens, int max_query_len, "
      "float sm_scale, bool causal) -> ()");
  ops.impl("paged_prefill_attn_rdna3_int8", torch::kCUDA,
           &paged_prefill_attn_rdna3_int8);

  ops.def(
      "paged_prefill_attn_rdna3_int4(Tensor! out, Tensor q, "
      "Tensor k_cache, Tensor v_cache, "
      "Tensor k_scale_cache, Tensor v_scale_cache, Tensor rht_signs, "
      "Tensor block_table, "
      "Tensor cu_seqlens_q, Tensor seq_lens, int max_query_len, "
      "float sm_scale, bool causal) -> ()");
  ops.impl("paged_prefill_attn_rdna3_int4", torch::kCUDA,
           &paged_prefill_attn_rdna3_int4);

  ops.def(
      "reshape_cache_int4_rdna3(Tensor key, Tensor value, "
      "Tensor key_cache, Tensor value_cache, "
      "Tensor k_scale_cache, Tensor v_scale_cache, "
      "Tensor rht_signs, Tensor slot_mapping) -> ()");
  ops.impl("reshape_cache_int4_rdna3", torch::kCUDA, &reshape_cache_int4_rdna3);

  ops.def(
      "rht_rotate_inplace_rdna3(Tensor! data, Tensor rht_signs, "
      "bool inverse, float post_scale) -> ()");
  ops.impl("rht_rotate_inplace_rdna3", torch::kCUDA, &rht_rotate_inplace_rdna3);

  ops.def(
      "pth_decode_int4_rdna3(Tensor! out, Tensor query, "
      "Tensor key_cache, Tensor value_cache, "
      "Tensor k_scale_cache, Tensor v_scale_cache, "
      "Tensor rht_signs, Tensor block_table, "
      "Tensor q_to_req, Tensor q_to_klen, "
      "Tensor! mid_o_buf, float sm_scale, int num_kv_splits) -> ()");
  ops.impl("pth_decode_int4_rdna3", torch::kCUDA, &pth_decode_int4_rdna3);

  ops.def(
      "pth_decode_int8_rdna3(Tensor! out, Tensor query, "
      "Tensor key_cache, Tensor value_cache, "
      "Tensor k_scale_cache, Tensor v_scale_cache, "
      "Tensor block_table, "
      "Tensor q_to_req, Tensor q_to_klen, "
      "Tensor! mid_o_buf, float sm_scale, int num_kv_splits) -> ()");
  ops.impl("pth_decode_int8_rdna3", torch::kCUDA, &pth_decode_int8_rdna3);
}
