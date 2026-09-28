#include "core/registration.h"
#include "flash_kda.h"

#include <torch/csrc/stable/library.h>

// Keep CUTLASS diagnostic changes local to its headers.
#if defined(__GNUG__)
  #pragma GCC diagnostic push
#endif
#include "flashkda_kcp.cuh"
#if defined(__GNUG__)
  #pragma GCC diagnostic pop
#endif

STABLE_TORCH_LIBRARY(_flashkda_C, m) {
  m.def(
      "kcp_flash_prepare(Tensor q, Tensor k, Tensor raw_g, Tensor beta_t, "
      "Tensor A_log, Tensor dt_bias, Tensor cu_seqlens, Tensor(a!) workspace, "
      "float scale, float lower_bound) -> ()");
  m.def(
      "kcp_flash_summary(Tensor v, Tensor beta_t, Tensor workspace, "
      "Tensor cu_seqlens, Tensor out_rows, Tensor(a!) summaries) -> ()");
  m.def(
      "kcp_flash_scan(Tensor v, Tensor beta_t, Tensor workspace, "
      "Tensor cu_seqlens, Tensor initial_state, Tensor(a!) out) -> ()");
  m.def("get_workspace_size(int T_total, int H, int N=1) -> int");
  m.def(
      "fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, "
      "Tensor(a!) out, Tensor(c!) workspace, Tensor A_log, Tensor dt_bias, "
      "float lower_bound, "
      "Tensor? initial_state=None, Tensor(b!)? final_state=None, "
      "Tensor? cu_seqlens=None, Tensor(d!)? checkpoint_state=None, "
      "Tensor? checkpoint_offsets=None, Tensor? segment_ids=None) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_flashkda_C, CompositeExplicitAutograd, m) {
  m.impl("get_workspace_size", TORCH_BOX(&get_workspace_size));
}

STABLE_TORCH_LIBRARY_IMPL(_flashkda_C, CUDA, m) {
  m.impl("fwd", TORCH_BOX(&fwd));
  m.impl("kcp_flash_prepare", TORCH_BOX(&kcp_flash_prepare));
  m.impl("kcp_flash_summary", TORCH_BOX(&kcp_flash_summary));
  m.impl("kcp_flash_scan", TORCH_BOX(&kcp_flash_scan));
}

REGISTER_EXTENSION(_flashkda_C)
