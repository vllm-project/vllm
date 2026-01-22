#include <stddef.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "cutlass/cutlass.h"

#include "scaled_mm_c2x.cuh"
#include "scaled_mm_c2x_sm75_dispatch.cuh"
#include "scaled_mm_c2x_sm80_dispatch.cuh"
#include "scaled_mm_c2x_sm89_fp8_dispatch.cuh"
#include "scaled_mm_c2x_sm89_int8_dispatch.cuh"

#include "stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp"
#include "stable/torch_utils.h"

using namespace vllm;

/*
   This file defines quantized GEMM operations using the CUTLASS 2.x API, for
   NVIDIA GPUs with SM versions prior to sm90 (Hopper).

*/

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm75_epilogue(torch::stable::Tensor& out,
                                     torch::stable::Tensor const& a,
                                     torch::stable::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Char);

  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    return cutlass_gemm_sm75_dispatch<int8_t, cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
    return cutlass_gemm_sm75_dispatch<int8_t, cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

void cutlass_scaled_mm_sm75(torch::stable::Tensor& out,
                            torch::stable::Tensor const& a,
                            torch::stable::Tensor const& b,
                            torch::stable::Tensor const& a_scales,
                            torch::stable::Tensor const& b_scales,
                            std::optional<torch::stable::Tensor> const& bias) {
  STD_TORCH_CHECK(a_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  if (bias) {
    STD_TORCH_CHECK(bias->scalar_type() == out.scalar_type(),
                    "currently bias dtype must match output dtype");
    return cutlass_scaled_mm_sm75_epilogue<c2x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm75_epilogue<c2x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_azp_sm75(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, torch::stable::Tensor const& azp_adj,
    std::optional<torch::stable::Tensor> const& azp,
    std::optional<torch::stable::Tensor> const& bias) {
  STD_TORCH_CHECK(a_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);

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
void cutlass_scaled_mm_sm80_epilogue(torch::stable::Tensor& out,
                                     torch::stable::Tensor const& a,
                                     torch::stable::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Char);

  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    return cutlass_gemm_sm80_dispatch<int8_t, cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
    return cutlass_gemm_sm80_dispatch<int8_t, cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

void cutlass_scaled_mm_sm80(torch::stable::Tensor& out,
                            torch::stable::Tensor const& a,
                            torch::stable::Tensor const& b,
                            torch::stable::Tensor const& a_scales,
                            torch::stable::Tensor const& b_scales,
                            std::optional<torch::stable::Tensor> const& bias) {
  STD_TORCH_CHECK(a_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  if (bias) {
    STD_TORCH_CHECK(bias->scalar_type() == out.scalar_type(),
                    "currently bias dtype must match output dtype");
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_azp_sm80(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, torch::stable::Tensor const& azp_adj,
    std::optional<torch::stable::Tensor> const& azp,
    std::optional<torch::stable::Tensor> const& bias) {
  STD_TORCH_CHECK(a_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);

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
void cutlass_scaled_mm_sm89_epilogue(torch::stable::Tensor& out,
                                     torch::stable::Tensor const& a,
                                     torch::stable::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  if (a.scalar_type() == torch::headeronly::ScalarType::Char) {
    STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Char);

    if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
      return cutlass_gemm_sm89_int8_dispatch<int8_t, cutlass::bfloat16_t,
                                             Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
      return cutlass_gemm_sm89_int8_dispatch<int8_t, cutlass::half_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  } else {
    STD_TORCH_CHECK(a.scalar_type() ==
                    torch::headeronly::ScalarType::Float8_e4m3fn);
    STD_TORCH_CHECK(b.scalar_type() ==
                    torch::headeronly::ScalarType::Float8_e4m3fn);

    if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
      return cutlass_gemm_sm89_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::bfloat16_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
      return cutlass_gemm_sm89_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::half_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  }
}

void cutlass_scaled_mm_sm89(torch::stable::Tensor& out,
                            torch::stable::Tensor const& a,
                            torch::stable::Tensor const& b,
                            torch::stable::Tensor const& a_scales,
                            torch::stable::Tensor const& b_scales,
                            std::optional<torch::stable::Tensor> const& bias) {
  STD_TORCH_CHECK(a_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  if (bias) {
    STD_TORCH_CHECK(bias->scalar_type() == out.scalar_type(),
                    "currently bias dtype must match output dtype");
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_azp_sm89(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, torch::stable::Tensor const& azp_adj,
    std::optional<torch::stable::Tensor> const& azp,
    std::optional<torch::stable::Tensor> const& bias) {
  STD_TORCH_CHECK(a_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);

  if (azp) {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBiasAzpToken>(
        out, a, b, a_scales, b_scales, azp_adj, *azp, bias);
  } else {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
}
