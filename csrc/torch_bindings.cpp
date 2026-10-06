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

// ROCm-only kernels on the legacy _C extension (they use the full torch API,
// not the stable ABI).
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // Octave KV cache (ROCm): store, split-KV decode, rotation, prefill.
  ops.def(
      "octave_cache_store(Tensor key, Tensor value, Tensor! cache, "
      "Tensor slot_mapping, Tensor k_signs, Tensor v_signs, int fmt) -> ()");
  ops.impl("octave_cache_store", torch::kCUDA, &octave_cache_store);
  ops.def(
      "octave_decode(Tensor! out, Tensor query, Tensor cache, "
      "Tensor block_table, Tensor q_to_req, Tensor q_to_klen, "
      "Tensor! mid_o, Tensor k_signs, Tensor v_signs, float sm_scale, "
      "int num_kv_splits, int fmt, int query_group, bool use_wmma) -> ()");
  ops.impl("octave_decode", torch::kCUDA, &octave_decode);
  ops.def(
      "octave_decode_sparse(Tensor! out, Tensor query, Tensor cache, "
      "Tensor block_table, Tensor q_to_req, Tensor indices, Tensor! mid_o, "
      "Tensor k_signs, Tensor v_signs, float sm_scale, int num_kv_splits, "
      "int fmt) -> ()");
  ops.impl("octave_decode_sparse", torch::kCUDA, &octave_decode_sparse);
  ops.def(
      "octave_rotate(Tensor! x, Tensor signs, bool k_layout, bool inverse) "
      "-> ()");
  ops.impl("octave_rotate", torch::kCUDA, &octave_rotate);
  ops.def(
      "octave_prefill(Tensor! out, Tensor q, Tensor k, Tensor v, Tensor cache, "
      "Tensor block_table, Tensor cu_seqlens_q, Tensor seq_lens, "
      "int max_query_len, float sm_scale, int fmt) -> ()");
  ops.impl("octave_prefill", torch::kCUDA, &octave_prefill);
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
