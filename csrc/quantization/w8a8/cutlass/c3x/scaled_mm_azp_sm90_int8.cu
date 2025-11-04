#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm90_int8_dispatch.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

namespace vllm {

void cutlass_scaled_mm_azp_sm90_int8(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     torch::Tensor const& a_scales,
                                     torch::Tensor const& b_scales,
                                     torch::Tensor const& azp_adj,
                                     std::optional<torch::Tensor> const& azp,
                                     std::optional<torch::Tensor> const& bias) {
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
