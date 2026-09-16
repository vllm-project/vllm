#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/stableivalue_conversions.h>

#include <array>

#include "libtorch_stable/ops.h"

void helion_cutlass_hybrid_scaled_mm(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    std::optional<torch::stable::Tensor> const& bias,
    int64_t helion_threshold) {
  if (a.size(0) <= helion_threshold) {
    // The Helion scaled_mm is a Python custom op registered on the dispatcher
    // as vllm_helion::scaled_mm. Re-enter the dispatcher to invoke it; the
    // stack order must match its schema:
    //   scaled_mm(Tensor(a0!) out, Tensor a, Tensor b, Tensor a_scales,
    //             Tensor b_scales, Tensor? bias=None) -> ()
    // out is mutated in place, so there is no return value to unpack.
    using torch::stable::detail::from;
    std::array<StableIValue, 6> stack{from(out),      from(a),
                                      from(b),        from(a_scales),
                                      from(b_scales), from(bias)};
    TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
        "vllm_helion::scaled_mm", "", stack.data(), TORCH_ABI_VERSION));
    return;
  }
  cutlass_scaled_mm(out, a, b, a_scales, b_scales, bias);
}
