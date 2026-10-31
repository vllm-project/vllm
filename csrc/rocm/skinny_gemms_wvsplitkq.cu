// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-side launcher of the wvSplitKQ skinny GEMM; kernels in skinny_gemms.cuh.
#include "skinny_gemms.cuh"

#include "dispatch_utils.h"                  // VLLM_DISPATCH_FP8_TYPES
#include "quantization/w8a8/fp8/common.cuh"  // is_fp8_ocp()

void wvSplitKQ(const at::Tensor& in_b, const at::Tensor& in_a,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;
  auto M_in = in_b.size(0);
  auto K_in = in_b.size(1);
  auto N_in = in_a.size(0);
  auto Kap_in = in_a.stride(0);
  auto Kbp_in = in_b.stride(0);
  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(K_in % 16 == 0, "k % 16 == 0");
  TORCH_CHECK(in_a.dtype() == in_b.dtype() && in_a.dtype() == kFp8Type);
  TORCH_CHECK(out_c.dtype() == torch::kFloat16 ||
              out_c.dtype() == torch::kBFloat16);

  dim3 grid(CuCount);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int max_lds_len = get_lds_size();

#define WVSPLITKQ_IMPL(_THRDS, _WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N) \
  {                                                                            \
    dim3 block(_THRDS, _WvPrGrp);                                              \
    if ((Kap_in * N_in <= max_lds_len) && (M_in % _YTILEs == 0)) {             \
      int __wvPrGrp = min(_WvPrGrp, mindiv(M_in, CuCount * _YTILEs, 16));      \
      wvSplitKQ_hf_sml_<fptype, fp8_t, _THRDS, _YTILEs, _WvPrGrp, 16, _UNRLs,  \
                        _N><<<grid, block, 0, stream>>>(                       \
          K_in, Kap_in, Kbp_in, M_in, Bx_in, By_in, b_ptr, a_ptr, bias_ptr,    \
          c_ptr, s_a, s_b, __wvPrGrp, CuCount);                                \
    } else {                                                                   \
      int __wvPrGrp = min(_WvPrGrp, mindiv(M_in, CuCount * _YTILEm, 16));      \
      wvSplitKQ_hf_<fptype, fp8_t, _THRDS, _YTILEm, _WvPrGrp, 16, _UNRLm, _N>  \
          <<<grid, block, 0, stream>>>(K_in, Kap_in, Kbp_in, M_in, Bx_in,      \
                                       By_in, b_ptr, a_ptr, bias_ptr, c_ptr,   \
                                       s_a, s_b, __wvPrGrp, CuCount);          \
    }                                                                          \
  }

#define WVSPLITKQ(_WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N)      \
  if (on_gfx12())                                                      \
    WVSPLITKQ_IMPL(32, _WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N) \
  else                                                                 \
    WVSPLITKQ_IMPL(64, _WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N)

  AT_DISPATCH_REDUCED_FLOATING_TYPES(out_c.scalar_type(), "wvSplitKQ", [&] {
    using fptype = typename scalar<scalar_t>::type;
    auto c_ptr = reinterpret_cast<fptype*>(out_c.data_ptr());
    auto s_a = scale_a.data_ptr<float>();
    auto s_b = scale_b.data_ptr<float>();
    VLLM_DISPATCH_FP8_TYPES(in_a.scalar_type(), "wvSplitKQ", [&] {
      auto a_ptr = in_a.data_ptr<fp8_t>();
      auto b_ptr = in_b.data_ptr<fp8_t>();
      auto bias_ptr = (in_bias.has_value() && in_bias->numel() > 0)
                          ? reinterpret_cast<fptype*>(in_bias->data_ptr())
                          : nullptr;
      switch (N_in) {
        case 1:
          WVSPLITKQ(16, 2, 2, 2, 2, 1)
          break;
        case 2:
          WVSPLITKQ(16, 2, 2, 2, 2, 2)
          break;
        case 3:
          WVSPLITKQ(16, 2, 2, 1, 1, 3)
          break;
        case 4:
          WVSPLITKQ(16, 2, 2, 1, 1, 4)
          break;
        default:
          throw std::runtime_error(
              "Unsupported N value: " + std::to_string(M_in) + "," +
              std::to_string(K_in) + "," + std::to_string(N_in));
      }
    });
  });
}
