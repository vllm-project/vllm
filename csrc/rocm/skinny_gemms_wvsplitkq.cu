// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-side launcher of the wvSplitKQ skinny GEMM; kernels in skinny_gemms.cuh.
#include <stdexcept>
#include <string>

#include "skinny_gemms.cuh"

#include "quantization/w8a8/fp8/common.cuh"  // is_fp8_ocp()

void wvSplitKQ(const torch::stable::Tensor& in_b,
               const torch::stable::Tensor& in_a,
               const std::optional<torch::stable::Tensor>& in_bias,
               torch::stable::Tensor& out_c,
               const torch::stable::Tensor& scale_a,
               const torch::stable::Tensor& scale_b, const int64_t CuCount) {
  static torch::headeronly::ScalarType kFp8Type =
      is_fp8_ocp() ? torch::headeronly::ScalarType::Float8_e4m3fn
                   : torch::headeronly::ScalarType::Float8_e4m3fnuz;
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

  STD_TORCH_CHECK(K_in % 16 == 0, "k % 16 == 0");
  STD_TORCH_CHECK(in_a.scalar_type() == in_b.scalar_type() &&
                      in_a.scalar_type() == kFp8Type,
                  "in_a and in_b must both be ", kFp8Type, ", got ",
                  in_a.scalar_type(), " and ", in_b.scalar_type());
  STD_TORCH_CHECK(
      out_c.scalar_type() == torch::headeronly::ScalarType::Half ||
          out_c.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "out_c must be float16 or bfloat16, got ", out_c.scalar_type());

  dim3 grid(CuCount);
  const torch::stable::accelerator::DeviceGuard device_guard(
      in_a.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(in_a.get_device_index());
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
  if (on_gfx12() || on_gfx117x())                                      \
    WVSPLITKQ_IMPL(32, _WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N) \
  else                                                                 \
    WVSPLITKQ_IMPL(64, _WvPrGrp, _YTILEs, _YTILEm, _UNRLs, _UNRLm, _N)

  VLLM_STABLE_DISPATCH_HALF_TYPES(out_c.scalar_type(), "wvSplitKQ", [&] {
    using fptype = typename scalar<scalar_t>::type;
    auto c_ptr = reinterpret_cast<fptype*>(out_c.mutable_data_ptr());
    auto s_a = scale_a.const_data_ptr<float>();
    auto s_b = scale_b.const_data_ptr<float>();
    VLLM_STABLE_DISPATCH_FP8_TYPES(in_a.scalar_type(), "wvSplitKQ", [&] {
      auto a_ptr = in_a.const_data_ptr<fp8_t>();
      auto b_ptr = in_b.const_data_ptr<fp8_t>();
      auto bias_ptr =
          (in_bias.has_value() && in_bias->numel() > 0)
              ? reinterpret_cast<const fptype*>(in_bias->const_data_ptr())
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
