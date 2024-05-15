#include <torch/extension.h>
#include <cuda_runtime.h>

void cutlass_scaled_mm_dq_sm75(torch::Tensor &out, torch::Tensor const &a,
                               torch::Tensor const &b,
                               torch::Tensor const &a_scales,
                               torch::Tensor const &b_scales);

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
  int32_t major_capability;
  int32_t minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor, 0);
  int32_t version_num = major_capability * 10 + minor_capability;

  if (version_num >= 90) /* H100 */ {
    // TODO: This kernel only works for sm90a
    //  -- figure out how to detect 90a vs 90

    cutlass_scaled_mm_dq_sm90(out, a, b, a_scales, b_scales);
  } else if (version_num == 89) /* Ada Lovelace */ {
    cutlass_scaled_mm_dq_sm89(out, a, b, a_scales, b_scales);
  } else if (version_num >= 80) /* Ampere */ {
    cutlass_scaled_mm_dq_sm80(out, a, b, a_scales, b_scales);
  } else {
    TORCH_CHECK(version_num >= 75);
    cutlass_scaled_mm_dq_sm75(out, a, b, a_scales, b_scales);
  }
}
