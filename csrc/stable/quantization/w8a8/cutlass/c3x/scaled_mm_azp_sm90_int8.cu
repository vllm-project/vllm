#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm90_int8_dispatch.cuh"
#include "stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

namespace vllm {

void cutlass_scaled_mm_azp_sm90_int8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, torch::stable::Tensor const& azp_adj,
    std::optional<torch::stable::Tensor> const& azp,
    std::optional<torch::stable::Tensor> const& bias) {
  if (azp) {
    return cutlass_scaled_mm_sm90_int8_epilogue<
        c3x::ScaledEpilogueBiasAzpToken>(out, a, b, a_scales, b_scales, azp_adj,
                                         *azp, bias);
  } else {
    return cutlass_scaled_mm_sm90_int8_epilogue<c3x::ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
}

}  // namespace vllm
