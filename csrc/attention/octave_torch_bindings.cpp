// Op registration for the Octave KV cache kernels (legacy _C extension, full
// torch API). Built only for gfx1100 and gfx942; see CMakeLists.txt.
#include <optional>

#include <torch/all.h>
#include <torch/library.h>

#include "core/registration.h"

// fmt: V bits (3 or 4), plus 16 for compact K.
void octave_cache_store(torch::Tensor key, torch::Tensor value,
                        torch::Tensor cache, torch::Tensor slot_mapping,
                        torch::Tensor k_signs, torch::Tensor v_signs,
                        int64_t fmt);

void octave_decode(torch::Tensor out, torch::Tensor query, torch::Tensor cache,
                   torch::Tensor block_table, torch::Tensor q_to_req,
                   torch::Tensor q_to_klen, torch::Tensor mid_o,
                   torch::Tensor k_signs, torch::Tensor v_signs,
                   double sm_scale, int64_t num_kv_splits, int64_t fmt,
                   int64_t query_group, bool use_wmma);

void octave_decode_sparse(torch::Tensor out, torch::Tensor query,
                          torch::Tensor cache, torch::Tensor block_table,
                          torch::Tensor q_to_req, torch::Tensor indices,
                          torch::Tensor mid_o, torch::Tensor k_signs,
                          torch::Tensor v_signs, double sm_scale,
                          int64_t num_kv_splits, int64_t fmt,
                          const std::optional<torch::Tensor>& positions,
                          const std::optional<torch::Tensor>& wtab,
                          const std::optional<torch::Tensor>& wtags,
                          const std::optional<torch::Tensor>& stab,
                          const std::optional<torch::Tensor>& stags);

void octave_window_store(torch::Tensor key, torch::Tensor value,
                         torch::Tensor slot_mapping, torch::Tensor positions,
                         torch::Tensor wtab, torch::Tensor wtags,
                         torch::Tensor wlocks, torch::Tensor stab,
                         torch::Tensor stags, torch::Tensor slocks);

void octave_rotate(torch::Tensor x, torch::Tensor signs, bool k_layout,
                   bool inverse);

void octave_prefill(torch::Tensor out, torch::Tensor q, torch::Tensor k,
                    torch::Tensor v, torch::Tensor cache,
                    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
                    torch::Tensor seq_lens, int64_t max_query_len,
                    double sm_scale, int64_t fmt);

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
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
      "int fmt, Tensor? positions=None, Tensor? wtab=None, "
      "Tensor? wtags=None, Tensor? stab=None, Tensor? stags=None) -> ()");
  ops.impl("octave_decode_sparse", torch::kCUDA, &octave_decode_sparse);
  ops.def(
      "octave_window_store(Tensor key, Tensor value, Tensor slot_mapping, "
      "Tensor positions, Tensor! wtab, Tensor! wtags, Tensor! wlocks, "
      "Tensor! stab, Tensor! stags, Tensor! slocks) -> ()");
  ops.impl("octave_window_store", torch::kCUDA, &octave_window_store);
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
