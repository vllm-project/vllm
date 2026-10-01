// Provides torch::Tensor for ops.h (previously included transitively via
// cache.h, which is no longer included here after cache ops moved to
// _C_stable_libtorch).
#include <torch/all.h>
#include "ops.h"
#include "core/registration.h"
#include <torch/library.h>
#include <torch/version.h>

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

// RDNA3 HIP kernels stay on the legacy _C extension (they use the full torch
// API, not the stable ABI). get_cuda_view_from_cpu_tensor and the CUDA/CPU ops
// migrated to the stable _C fragment in
// csrc/libtorch_stable/torch_bindings.cpp.
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // RDNA3 INT8 per-token-head paged prefill attention (gfx1100).
  ops.def(
      "paged_prefill_attn_rdna3_int8(Tensor! out, Tensor q, Tensor k_chunk, "
      "Tensor v_chunk, Tensor k_cache, Tensor v_cache, "
      "Tensor k_scale_cache, Tensor v_scale_cache, Tensor block_table, "
      "Tensor cu_seqlens_q, Tensor seq_lens, int max_query_len, "
      "float sm_scale, bool causal) -> ()");
  ops.impl("paged_prefill_attn_rdna3_int8", torch::kCUDA,
           &paged_prefill_attn_rdna3_int8);

  // RDNA3 INT4 per-token-head paged prefill attention (gfx1100).
  ops.def(
      "paged_prefill_attn_rdna3_int4(Tensor! out, Tensor q, "
      "Tensor k_cache, Tensor v_cache, "
      "Tensor k_scale_cache, Tensor v_scale_cache, Tensor rht_signs, "
      "Tensor block_table, "
      "Tensor cu_seqlens_q, Tensor seq_lens, int max_query_len, "
      "float sm_scale, bool causal) -> ()");
  ops.impl("paged_prefill_attn_rdna3_int4", torch::kCUDA,
           &paged_prefill_attn_rdna3_int4);

  // Fused RHT + INT4 reshape_and_cache for RDNA3.
  ops.def(
      "reshape_cache_int4_rdna3(Tensor key, Tensor value, "
      "Tensor key_cache, Tensor value_cache, "
      "Tensor k_scale_cache, Tensor v_scale_cache, "
      "Tensor rht_signs, Tensor slot_mapping) -> ()");
  ops.impl("reshape_cache_int4_rdna3", torch::kCUDA, &reshape_cache_int4_rdna3);

  // Inplace RHT butterfly for INT4 decode Q rotation / output unrotation.
  ops.def(
      "rht_rotate_inplace_rdna3(Tensor! data, Tensor rht_signs, "
      "bool inverse, float post_scale) -> ()");
  ops.impl("rht_rotate_inplace_rdna3", torch::kCUDA, &rht_rotate_inplace_rdna3);

  // HIP split-KV decode attention for INT4 per-token-head (RDNA3).
  ops.def(
      "pth_decode_int4_rdna3(Tensor! out, Tensor query, "
      "Tensor key_cache, Tensor value_cache, "
      "Tensor k_scale_cache, Tensor v_scale_cache, "
      "Tensor rht_signs, Tensor block_table, "
      "Tensor q_to_req, Tensor q_to_klen, "
      "Tensor! mid_o_buf, float sm_scale, int num_kv_splits) -> ()");
  ops.impl("pth_decode_int4_rdna3", torch::kCUDA, &pth_decode_int4_rdna3);

  // INT8 per-token-head decode for RDNA3.
  ops.def(
      "pth_decode_int8_rdna3(Tensor! out, Tensor query, "
      "Tensor key_cache, Tensor value_cache, "
      "Tensor k_scale_cache, Tensor v_scale_cache, "
      "Tensor block_table, "
      "Tensor q_to_req, Tensor q_to_klen, "
      "Tensor! mid_o_buf, float sm_scale, int num_kv_splits) -> ()");
  ops.impl("pth_decode_int8_rdna3", torch::kCUDA, &pth_decode_int8_rdna3);

  // SplitQ KV cache (RDNA3): store, split-KV decode, int8 expansion.
  ops.def(
      "splitq_cache_store(Tensor key, Tensor value, Tensor! cache, "
      "Tensor slot_mapping, Tensor nope_signs, Tensor v_signs, int bits) -> ()");
  ops.impl("splitq_cache_store", torch::kCUDA, &splitq_cache_store);
  ops.def(
      "splitq_decode(Tensor! out, Tensor query, Tensor cache, "
      "Tensor block_table, Tensor q_to_req, Tensor q_to_klen, "
      "Tensor! mid_o, Tensor nope_signs, Tensor v_signs, float sm_scale, "
      "int num_kv_splits, int bits, int query_group) -> ()");
  ops.impl("splitq_decode", torch::kCUDA, &splitq_decode);
  ops.def(
      "splitq_to_int8(Tensor cache, Tensor block_table, "
      "Tensor query_start_loc, Tensor seq_lens, int max_ctx_pad, "
      "Tensor! k_out, Tensor! v_out, Tensor! k_scale_out, "
      "Tensor! v_scale_out, int bits) -> ()");
  ops.impl("splitq_to_int8", torch::kCUDA, &splitq_to_int8);
}

#ifdef USE_ROCM
TORCH_LIBRARY_FRAGMENT(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  // Quick Reduce all-reduce kernels (ROCm-only; stays on legacy _C).
  custom_ar.def(
      "qr_all_reduce(int fa, Tensor inp, Tensor out, int quant_level, bool "
      "cast_bf2half) -> ()");
  custom_ar.impl("qr_all_reduce", torch::kCUDA, &qr_all_reduce);

  custom_ar.def("init_custom_qr", &init_custom_qr);
  custom_ar.def("qr_destroy", &qr_destroy);
  custom_ar.def("qr_get_handle", &qr_get_handle);

  custom_ar.def("qr_open_handles(int _fa, Tensor[](b!) handles) -> ()");
  custom_ar.impl("qr_open_handles", torch::kCPU, &qr_open_handles);

  custom_ar.def("qr_max_size", &qr_max_size);
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
