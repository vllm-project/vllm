#include "ops.h"
#include "core/registration.h"

#include <torch/csrc/stable/library.h>

// Register ops using STABLE_TORCH_LIBRARY for stable ABI compatibility.
// Note: We register under namespace "_C" so ops are accessible as
// torch.ops._C.<op_name> for compatibility with existing code.
STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
#ifndef USE_ROCM
  m.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
#ifndef USE_ROCM
  m.impl("permute_cols", TORCH_BOX(&permute_cols));
#endif
}

REGISTER_EXTENSION(_C_stable_libtorch)
