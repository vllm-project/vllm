#include "machete_mm_launcher.cuh"
#include "machete_prepack_launcher.cuh"
#include "stable/core/scalar_type.hpp"

#include <torch/csrc/stable/library.h>

namespace machete {

using namespace vllm;

std::vector<std::string> supported_schedules(
    torch::headeronly::ScalarType a_type, int64_t b_type_id,
    std::optional<torch::headeronly::ScalarType> maybe_group_scales_type,
    std::optional<torch::headeronly::ScalarType> maybe_group_zeros_type,
    std::optional<torch::headeronly::ScalarType> maybe_channel_scales_type,
    std::optional<torch::headeronly::ScalarType> maybe_token_scales_type,
    std::optional<torch::headeronly::ScalarType> maybe_out_type) {
  ScalarType const b_type = ScalarType::from_id(b_type_id);
  return supported_schedules_dispatch({
      .a_type = a_type,
      .b_type = b_type,
      .maybe_group_scales_type = maybe_group_scales_type,
      .maybe_group_zeros_type = maybe_group_zeros_type,
      .maybe_channel_scales_type = maybe_channel_scales_type,
      .maybe_token_scales_type = maybe_token_scales_type,
      .maybe_out_type = maybe_out_type,
  });
}

torch::stable::Tensor mm(
    torch::stable::Tensor const& A, torch::stable::Tensor const& B,
    int64_t b_type_id,
    std::optional<torch::headeronly::ScalarType> const& maybe_out_type,
    std::optional<torch::stable::Tensor> const& maybe_group_scales,
    std::optional<torch::stable::Tensor> const& maybe_group_zeros,
    std::optional<int64_t> maybe_group_size,
    std::optional<torch::stable::Tensor> const& maybe_channel_scales,
    std::optional<torch::stable::Tensor> const& maybe_token_scales,
    std::optional<std::string> maybe_schedule) {
  ScalarType const b_type = ScalarType::from_id(b_type_id);
  return mm_dispatch({.A = A,
                      .B = B,
                      .b_type = b_type,
                      .maybe_out_type = maybe_out_type,
                      .maybe_group_scales = maybe_group_scales,
                      .maybe_group_zeros = maybe_group_zeros,
                      .maybe_group_size = maybe_group_size,
                      .maybe_channel_scales = maybe_channel_scales,
                      .maybe_token_scales = maybe_token_scales,
                      .maybe_schedule = maybe_schedule});
}

torch::stable::Tensor prepack_B(
    torch::stable::Tensor const& B, torch::headeronly::ScalarType const& a_type,
    int64_t b_type_id,
    std::optional<torch::headeronly::ScalarType> const&
        maybe_group_scales_type) {
  ScalarType const b_type = ScalarType::from_id(b_type_id);
  return prepack_B_dispatch(
      {.B = B,
       .a_type = a_type,
       .b_type = b_type,
       .maybe_group_scales_type = maybe_group_scales_type});
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("machete_prepack_B", TORCH_BOX(&prepack_B));
  m.impl("machete_mm", TORCH_BOX(&mm));
}

// use CompositeExplicitAutograd since supported_schedules has no tensor
// arguments
STABLE_TORCH_LIBRARY_IMPL(_C, CompositeExplicitAutograd, m) {
  m.impl("machete_supported_schedules", TORCH_BOX(&supported_schedules));
}

};  // namespace machete
