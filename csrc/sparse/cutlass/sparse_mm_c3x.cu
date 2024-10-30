// clang-format will break include orders
// clang-format off
#include <cudaTypedefs.h>

#if defined CUDA_VERSION && CUDA_VERSION >= 12000

#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "util/broadcast_load_epilogue_c3x.hpp"
#include "util/common.hpp"
// clang-format on

#include "util/common_gemm.cuh"

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_gemm_sm90_fp8_dispatch(torch::Tensor& out, torch::Tensor const& a,
                                    torch::Tensor const& e,
                                    torch::Tensor const& b,
                                    EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(e.dtype() == torch::kUInt8);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  using Cutlass3xGemmDefault =
      typename sm90_fp8_config_default<InType, OutType,
                                       Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM64 =
      typename sm90_fp8_config_M64<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM128 =
      typename sm90_fp8_config_M128<InType, OutType, Epilogue>::Cutlass3xGemm;

  uint32_t const m = a.size(0);
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(64), next_pow_2(m));  // next power of 2

  if (mp2 <= 64) {
    // m in [1, 64]
    return cutlass_test_gemm_caller<Cutlass3xGemmM64>(
        out, a, e, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // m in (64, 128]
    return cutlass_test_gemm_caller<Cutlass3xGemmM128>(
        out, a, e, b, std::forward<EpilogueArgs>(args)...);
  } else {
    // m in (128, inf)
    return cutlass_test_gemm_caller<Cutlass3xGemmDefault>(
        out, a, e, b, std::forward<EpilogueArgs>(args)...);
  }
}

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_gemm_sm90_int8_dispatch(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& e,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, int8_t>());
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(e.dtype() == torch::kUInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  using Cutlass3xGemmDefault =
      typename sm90_int8_config_default<InType, OutType,
                                        Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM128 =
      typename sm90_int8_config_M128<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM64 =
      typename sm90_int8_config_M64<InType, OutType, Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM32NBig =
      typename sm90_int8_config_M32_NBig<InType, OutType,
                                         Epilogue>::Cutlass3xGemm;
  using Cutlass3xGemmM32NSmall =
      typename sm90_int8_config_M32_NSmall<InType, OutType,
                                           Epilogue>::Cutlass3xGemm;

  uint32_t const n = out.size(1);
  bool const is_small_n = n < 8192;

  uint32_t const m = a.size(0);
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(32), next_pow_2(m));  // next power of 2

  if (mp2 <= 32) {
    // m in [1, 32]
    if (is_small_n) {
      return cutlass_test_gemm_caller<Cutlass3xGemmM32NSmall>(
          out, a, e, b, std::forward<EpilogueArgs>(args)...);
    } else {
      return cutlass_test_gemm_caller<Cutlass3xGemmM32NBig>(
          out, a, e, b, std::forward<EpilogueArgs>(args)...);
    }
  } else if (mp2 <= 64) {
    // m in (32, 64]
    return cutlass_test_gemm_caller<Cutlass3xGemmM64>(
        out, a, e, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // m in (64, 128]
    return cutlass_test_gemm_caller<Cutlass3xGemmM128>(
        out, a, e, b, std::forward<EpilogueArgs>(args)...);
  } else {
    // m in (128, inf)
    return cutlass_test_gemm_caller<Cutlass3xGemmDefault>(
        out, a, e, b, std::forward<EpilogueArgs>(args)...);
  }
}

template <template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_test_mm_sm90_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& e,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  if (a.dtype() == torch::kInt8) {
    TORCH_CHECK(e.dtype() == torch::kUInt8);
    TORCH_CHECK(b.dtype() == torch::kInt8);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::bfloat16_t,
                                             Epilogue>(
          out, a, e, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);
      return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::half_t, Epilogue>(
          out, a, e, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  } else {
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(e.dtype() == torch::kUInt8);
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::bfloat16_t, Epilogue>(
          out, a, e, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);
      return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::half_t, Epilogue>(
          out, a, e, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  }
}

void cutlass_scaled_test_mm_sm90(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& e,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias) {
    TORCH_CHECK(bias->dtype() == c.dtype(),
                "currently bias dtype must match output dtype ", c.dtype());
    return cutlass_scaled_test_mm_sm90_epilogue<ScaledEpilogueBias>(
        c, a, e, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_test_mm_sm90_epilogue<ScaledEpilogue>(c, a, e, b,
                                                           a_scales,
                                                           b_scales);
  }
}

void cutlass_scaled_test_mm_azp_sm90(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& e,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                c10::optional<torch::Tensor> const& azp,
                                c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (azp) {
    return cutlass_scaled_test_mm_sm90_epilogue<ScaledEpilogueBiasAzpToken>(
        out, a, e, b, a_scales, b_scales, azp_adj, *azp, bias);
  } else {
    return cutlass_scaled_test_mm_sm90_epilogue<ScaledEpilogueBiasAzp>(
        out, a, e, b, a_scales, b_scales, azp_adj, bias);
  }
}

#endif
