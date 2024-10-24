#include <stddef.h>
#include <torch/all.h>
#include "cutlass/cutlass.h"

#include "scaled_mm_c2x.cuh"
#include "scaled_mm_c2x_sm75_dispatch.cuh"
#include "scaled_mm_c2x_sm80_dispatch.cuh"
#include "scaled_mm_c2x_sm89_fp8_dispatch.cuh"
#include "scaled_mm_c2x_sm89_int8_dispatch.cuh"

#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp"

using namespace vllm;

/*
   This file defines quantized GEMM operations using the CUTLASS 2.x API, for
   NVIDIA GPUs with SM versions prior to sm90 (Hopper).
*/

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm75_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_sm75_dispatch<int8_t, cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_sm75_dispatch<int8_t, cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

void cutlass_scaled_mm_sm75(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm75_epilogue<c2x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm75_epilogue<c2x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_azp_sm75(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                c10::optional<torch::Tensor> const& azp,
                                c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (azp) {
    return cutlass_scaled_mm_sm75_epilogue<c2x::ScaledEpilogueBiasAzpToken>(
        out, a, b, a_scales, b_scales, azp_adj, *azp, bias);
  } else {
    return cutlass_scaled_mm_sm75_epilogue<c2x::ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
}

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm80_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_sm80_dispatch<int8_t, cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_sm80_dispatch<int8_t, cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

void cutlass_scaled_mm_sm80(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_azp_sm80(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                c10::optional<torch::Tensor> const& azp,
                                c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (azp) {
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogueBiasAzpToken>(
        out, a, b, a_scales, b_scales, azp_adj, *azp, bias);
  } else {
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
}

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm89_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  if (a.dtype() == torch::kInt8) {
    TORCH_CHECK(b.dtype() == torch::kInt8);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_sm89_int8_dispatch<int8_t, cutlass::bfloat16_t,
                                             Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      assert(out.dtype() == torch::kFloat16);
      return cutlass_gemm_sm89_int8_dispatch<int8_t, cutlass::half_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  } else {
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_sm89_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::bfloat16_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);
      return cutlass_gemm_sm89_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::half_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  }
}

void cutlass_scaled_mm_sm89(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_azp_sm89(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                c10::optional<torch::Tensor> const& azp,
                                c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (azp) {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBiasAzpToken>(
        out, a, b, a_scales, b_scales, azp_adj, *azp, bias);
  } else {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
}
