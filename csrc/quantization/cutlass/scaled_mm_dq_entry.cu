#include <torch/extension.h>

void cutlass_scaled_mm_dq_sm80(torch::Tensor &out, torch::Tensor const &a,
                               torch::Tensor const &b,
                               torch::Tensor const &a_scales,
                               torch::Tensor const &b_scales);
void cutlass_scaled_mm_dq_sm89(torch::Tensor &out, torch::Tensor const &a,
                               torch::Tensor const &b,
                               torch::Tensor const &a_scales,
                               torch::Tensor const &b_scales);
void cutlass_scaled_mm_dq_sm90(torch::Tensor &out, torch::Tensor const &a,
                               torch::Tensor const &b,
                               torch::Tensor const &a_scales,
                               torch::Tensor const &b_scales);

void cutlass_scaled_mm_dq(torch::Tensor &out, torch::Tensor const &a,
                          torch::Tensor const &b, torch::Tensor const &a_scales,
                          torch::Tensor const &b_scales) {
  // TODO(tms): Hack. cudaGetDeviceProperties is very slow.
  static std::optional<int32_t> maybe_version_num = std::nullopt;
  if (!maybe_version_num.has_value()) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    maybe_version_num = properties.major * 10 + properties.minor;
  }
  int32_t version_num = maybe_version_num.value();

  if (version_num >= 90) /* H100 */ {
    // TODO: This kernel only works for sm90a
    //  -- figure out how to detect 90a vs 90

    cutlass_scaled_mm_dq_sm90(out, a, b, a_scales, b_scales);
  } else if (version_num == 89) /* Ada Lovelace */ {
    cutlass_scaled_mm_dq_sm89(out, a, b, a_scales, b_scales);
  } else if (version_num >= 80) /* Ampere */ {
    cutlass_scaled_mm_dq_sm80(out, a, b, a_scales, b_scales);
  } else {
    throw std::runtime_error("Unsupported GPU architecture");
  }
}
