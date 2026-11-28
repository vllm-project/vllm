// Op registration for the Octave KV cache kernels (_C_stable_libtorch,
// LibTorch stable ABI). Built only for gfx1100 and gfx942; see CMakeLists.txt.
#include <optional>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

// fmt: V bits (3 or 4), plus 16 for compact K.
void octave_cache_store(torch::stable::Tensor key, torch::stable::Tensor value,
                        torch::stable::Tensor cache,
                        torch::stable::Tensor slot_mapping,
                        torch::stable::Tensor k_signs,
                        torch::stable::Tensor v_signs, int64_t fmt);

void octave_decode(torch::stable::Tensor out, torch::stable::Tensor query,
                   torch::stable::Tensor cache,
                   torch::stable::Tensor block_table,
                   torch::stable::Tensor q_to_req,
                   torch::stable::Tensor q_to_klen, torch::stable::Tensor mid_o,
                   torch::stable::Tensor k_signs, torch::stable::Tensor v_signs,
                   double sm_scale, int64_t num_kv_splits, int64_t fmt,
                   int64_t query_group, bool use_wmma);

void octave_decode_sparse(
    torch::stable::Tensor out, torch::stable::Tensor query,
    torch::stable::Tensor cache, torch::stable::Tensor block_table,
    torch::stable::Tensor q_to_req, torch::stable::Tensor indices,
    torch::stable::Tensor mid_o, torch::stable::Tensor k_signs,
    torch::stable::Tensor v_signs, double sm_scale, int64_t num_kv_splits,
    int64_t fmt, const std::optional<torch::stable::Tensor>& positions,
    const std::optional<torch::stable::Tensor>& wtab,
    const std::optional<torch::stable::Tensor>& wtags,
    const std::optional<torch::stable::Tensor>& stab,
    const std::optional<torch::stable::Tensor>& stags);

void octave_window_store(
    torch::stable::Tensor key, torch::stable::Tensor value,
    torch::stable::Tensor slot_mapping, torch::stable::Tensor positions,
    torch::stable::Tensor wtab, torch::stable::Tensor wtags,
    torch::stable::Tensor wlocks, torch::stable::Tensor stab,
    torch::stable::Tensor stags, torch::stable::Tensor slocks);

void octave_rotate(torch::stable::Tensor x, torch::stable::Tensor signs,
                   bool k_layout, bool inverse);

void octave_prefill(torch::stable::Tensor out, torch::stable::Tensor q,
                    torch::stable::Tensor k, torch::stable::Tensor v,
                    torch::stable::Tensor cache,
                    torch::stable::Tensor block_table,
                    torch::stable::Tensor cu_seqlens_q,
                    torch::stable::Tensor seq_lens, int64_t max_query_len,
                    double sm_scale, int64_t fmt);

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "octave_cache_store(Tensor key, Tensor value, Tensor! cache, "
      "Tensor slot_mapping, Tensor k_signs, Tensor v_signs, int fmt) -> ()");
  ops.def(
      "octave_decode(Tensor! out, Tensor query, Tensor cache, "
      "Tensor block_table, Tensor q_to_req, Tensor q_to_klen, "
      "Tensor! mid_o, Tensor k_signs, Tensor v_signs, float sm_scale, "
      "int num_kv_splits, int fmt, int query_group, bool use_wmma) -> ()");
  ops.def(
      "octave_decode_sparse(Tensor! out, Tensor query, Tensor cache, "
      "Tensor block_table, Tensor q_to_req, Tensor indices, Tensor! mid_o, "
      "Tensor k_signs, Tensor v_signs, float sm_scale, int num_kv_splits, "
      "int fmt, Tensor? positions=None, Tensor? wtab=None, "
      "Tensor? wtags=None, Tensor? stab=None, Tensor? stags=None) -> ()");
  ops.def(
      "octave_window_store(Tensor key, Tensor value, Tensor slot_mapping, "
      "Tensor positions, Tensor! wtab, Tensor! wtags, Tensor! wlocks, "
      "Tensor! stab, Tensor! stags, Tensor! slocks) -> ()");
  ops.def(
      "octave_rotate(Tensor! x, Tensor signs, bool k_layout, bool inverse) "
      "-> ()");
  ops.def(
      "octave_prefill(Tensor! out, Tensor q, Tensor k, Tensor v, Tensor cache, "
      "Tensor block_table, Tensor cu_seqlens_q, Tensor seq_lens, "
      "int max_query_len, float sm_scale, int fmt) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("octave_cache_store", TORCH_BOX(&octave_cache_store));
  ops.impl("octave_decode", TORCH_BOX(&octave_decode));
  ops.impl("octave_decode_sparse", TORCH_BOX(&octave_decode_sparse));
  ops.impl("octave_window_store", TORCH_BOX(&octave_window_store));
  ops.impl("octave_rotate", TORCH_BOX(&octave_rotate));
  ops.impl("octave_prefill", TORCH_BOX(&octave_prefill));
}
