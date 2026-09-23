#include "ops.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(_C_custom_ar, custom_ag_rs) {
  custom_ag_rs.def(
      "custom_all_gather(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ag_rs.def(
      "mnnvl_lamport_all_gather(int fa, Tensor inp, Tensor! out, int "
      "local_buffer, int multicast_buffer, int epoch_buffer, int "
      "stage_sz_bytes) -> ()");
  custom_ag_rs.def(
      "custom_reduce_scatter(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ag_rs.def(
      "mnnvl_lamport_reduce_scatter(int fa, Tensor inp, Tensor! out, int "
      "local_buffer, int epoch_buffer, int stage_sz_bytes) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CUDA, custom_ag_rs) {
  custom_ag_rs.impl("custom_all_gather", TORCH_BOX(&custom_all_gather));
  custom_ag_rs.impl("mnnvl_lamport_all_gather",
                    TORCH_BOX(&mnnvl_lamport_all_gather));
  custom_ag_rs.impl("custom_reduce_scatter", TORCH_BOX(&custom_reduce_scatter));
  custom_ag_rs.impl("mnnvl_lamport_reduce_scatter",
                    TORCH_BOX(&mnnvl_lamport_reduce_scatter));
}
