#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>
#include <vector>

#ifdef USE_ROCM
// Defined in custom_quickreduce.cu.
using fptr_t = int64_t;
fptr_t init_custom_qr(int64_t rank, int64_t world_size,
                      std::optional<int64_t> qr_max_size = std::nullopt);
void qr_destroy(fptr_t _fa);
torch::stable::Tensor qr_get_handle(fptr_t _fa);
void qr_open_handles(fptr_t _fa,
                     const std::vector<torch::stable::Tensor>& handles);
void qr_all_reduce(fptr_t _fa, torch::stable::Tensor& inp,
                   torch::stable::Tensor& out, int64_t quant_level,
                   bool cast_bf2half = false);
int64_t qr_max_size();
#endif

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

#ifdef USE_ROCM
STABLE_TORCH_LIBRARY_FRAGMENT(_C_custom_ar, custom_ar) {
  // Quick Reduce all-reduce kernels (ROCm-only).
  custom_ar.def(
      "qr_all_reduce(int fa, Tensor inp, Tensor out, int quant_level, bool "
      "cast_bf2half) -> ()");
  custom_ar.def(
      "init_custom_qr(int rank, int world_size, int? qr_max_size) -> int");
  custom_ar.def("qr_destroy(int fa) -> ()");
  custom_ar.def("qr_get_handle(int fa) -> Tensor");
  custom_ar.def("qr_open_handles(int _fa, Tensor[](b!) handles) -> ()");
  custom_ar.def("qr_max_size() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CUDA, custom_ar) {
  custom_ar.impl("qr_all_reduce", TORCH_BOX(&qr_all_reduce));
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CPU, custom_ar) {
  custom_ar.impl("qr_open_handles", TORCH_BOX(&qr_open_handles));
}

STABLE_TORCH_LIBRARY_IMPL(_C_custom_ar, CompositeExplicitAutograd, custom_ar) {
  custom_ar.impl("init_custom_qr", TORCH_BOX(&init_custom_qr));
  custom_ar.impl("qr_destroy", TORCH_BOX(&qr_destroy));
  custom_ar.impl("qr_get_handle", TORCH_BOX(&qr_get_handle));
  custom_ar.impl("qr_max_size", TORCH_BOX(&qr_max_size));
}
#endif
