#include "core/registration.h"
#include "flash_kda.h"

TORCH_LIBRARY(_flashkda_C, m) {
  m.def("get_workspace_size(int T_total, int H, int N=1) -> int",
        &get_workspace_size);
  m.def(
      "fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, "
      "Tensor(a!) out, Tensor workspace, Tensor A_log, Tensor dt_bias, "
      "float lower_bound, "
      "Tensor? initial_state=None, Tensor(b!)? final_state=None, "
      "Tensor? cu_seqlens=None) -> ()");
}

TORCH_LIBRARY_IMPL(_flashkda_C, CUDA, m) { m.impl("fwd", &fwd); }

REGISTER_EXTENSION(_flashkda_C)
