#include "core/registration.h"
#include "gradlib/ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, gradlib_ops) {
  // Gradlib custom ops

  gradlib_ops.def("hipb_create_extension", &hipb_create_extension);
  gradlib_ops.def("hipb_destroy_extension", &hipb_destroy_extension);
  gradlib_ops.def("hipb_mm", &hipb_mm);
  gradlib_ops.def("hipb_findallsols", &hipb_findallsols);

  gradlib_ops.def("rocb_create_extension", &rocb_create_extension);
  gradlib_ops.def("rocb_destroy_extension", &rocb_destroy_extension);
  gradlib_ops.def("rocb_mm", &RocSolIdxBlas);
  gradlib_ops.def("rocb_findallsols", &RocFindAllSolIdxBlas);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
