#include "machete_mm_launcher.cuh"
#include "machete_prepack_launcher.cuh"
#include "core/scalar_type.hpp"

#include "core/registration.h"

namespace machete {

using namespace vllm;

std::vector<std::string> supported_schedules(ScalarTypeTorchPtr const& btype) {
  return {};
}

torch::Tensor gemm(torch::Tensor const& A, torch::Tensor const& B,
                   ScalarTypeTorchPtr const& btype,
                   c10::optional<at::ScalarType> const& out_type,
                   c10::optional<torch::Tensor> const& scales,
                   c10::optional<torch::Tensor> const& zeros,
                   c10::optional<int64_t> group_size,
                   c10::optional<torch::Tensor> const& C,
                   c10::optional<double> alpha, c10::optional<double> beta,
                   c10::optional<std::string> schedule) {
  return gemm_dispatch({.A = A,
                        .B = B,
                        .btype = *btype,
                        .out_type = c10::nullopt,
                        .scales = scales,
                        .zeros = zeros,
                        .group_size = group_size,
                        .C = C,
                        .alpha = alpha,
                        .beta = beta,
                        .schedule = schedule});
}

torch::Tensor prepack_B(torch::Tensor const& B, at::ScalarType const& atype,
                        ScalarTypeTorchPtr const& btype) {
  return prepack_B_dispatch(B, atype, *btype);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("machete_prepack_B", &prepack_B);
  m.impl("machete_gemm", &gemm);
  m.impl("machete_supported_schedules", &supported_schedules);
}

};  // namespace machete
