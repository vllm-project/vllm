#include "registration.h"
#include "punica_ops.h"

#include <git.h>

std::string githash() { return std::string{git::CommitSHA1()}; }

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def("githash() -> ()");
  m.impl("githash", torch::kCUDA, &githash);

  m.def(
      "dispatch_bgmv(Tensor! y, Tensor x, Tensor w, Tensor indicies, int "
      "layer_idx, float scale) -> ()");
  m.impl("dispatch_bgmv", torch::kCUDA, &dispatch_bgmv);

  m.def(
      "dispatch_bgmv_low_level(Tensor! y, Tensor x, Tensor w,"
      "Tensor indicies, int layer_idx,"
      "float scale, int h_in, int h_out,"
      "int y_offset) -> ()");
  m.impl("dispatch_bgmv_low_level", torch::kCUDA, &dispatch_bgmv_low_level);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
