#include "machete_mm_launcher.cuh"
#include "machete_prepack_launcher.cuh"
#include "core/scalar_type.hpp"

#include "core/registration.h"

namespace machete {

using namespace vllm;

std::vector<std::string> supported_schedules(
    at::ScalarType a_type, ScalarTypeTorchPtr b_type,
    c10::optional<at::ScalarType> maybe_group_scales_type,
    c10::optional<at::ScalarType> maybe_group_zeros_type,
    c10::optional<at::ScalarType> maybe_channel_scales_type,
    c10::optional<at::ScalarType> maybe_token_scales_type,
    c10::optional<at::ScalarType> maybe_out_type) {
  return supported_schedules_dispatch({
      .a_type = a_type,
      .b_type = *b_type,
      .maybe_group_scales_type = maybe_group_scales_type,
      .maybe_group_zeros_type = maybe_group_zeros_type,
      .maybe_channel_scales_type = maybe_channel_scales_type,
      .maybe_token_scales_type = maybe_token_scales_type,
      .maybe_out_type = maybe_out_type,
  });
}

torch::Tensor mm(torch::Tensor const& A, torch::Tensor const& B,
                 ScalarTypeTorchPtr const& btype,
                 c10::optional<at::ScalarType> const& maybe_out_type,
                 c10::optional<torch::Tensor> const& maybe_group_scales,
                 c10::optional<torch::Tensor> const& maybe_group_zeros,
                 c10::optional<int64_t> maybe_group_size,
                 c10::optional<torch::Tensor> const& maybe_channel_scales,
                 c10::optional<torch::Tensor> const& maybe_token_scales,
                 c10::optional<std::string> maybe_schedule) {
  return mm_dispatch({.A = A,
                      .B = B,
                      .btype = *btype,
                      .maybe_out_type = maybe_out_type,
                      .maybe_group_scales = maybe_group_scales,
                      .maybe_group_zeros = maybe_group_zeros,
                      .maybe_group_size = maybe_group_size,
                      .maybe_channel_scales = maybe_channel_scales,
                      .maybe_token_scales = maybe_token_scales,
                      .maybe_schedule = maybe_schedule});
}

torch::Tensor prepack_B(
    torch::Tensor const& B, at::ScalarType const& a_type,
    ScalarTypeTorchPtr const& b_type,
    c10::optional<at::ScalarType> const& maybe_group_scales_type) {
  return prepack_B_dispatch(
      {.B = B,
       .a_type = a_type,
       .b_type = *b_type,
       .maybe_group_scales_type = maybe_group_scales_type});
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("machete_prepack_B", &prepack_B);
  m.impl("machete_gemm", &gemm);
  m.impl("machete_supported_schedules", &supported_schedules);
}

};  // namespace machete
