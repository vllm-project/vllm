#pragma once

#include "scaled_mm.cuh"
#include "cutlass_gemm_caller.cuh"
#include "stable/torch_utils.h"
#include <torch/headeronly/core/ScalarType.h>

/**
 * This file defines Gemm kernel configurations for SM120 (fp8) based on the
 * Gemm shape.
 *
 */

namespace vllm {

using c3x::cutlass_gemm_caller;

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm120_fp8_config_default {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;  // Only work with Shape<_1, _1, _1>
  using Cutlass3xGemm =
      cutlass_3x_gemm_sm120<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm120_fp8_dispatch(torch::stable::Tensor& out,
                                            torch::stable::Tensor const& a,
                                            torch::stable::Tensor const& b,
                                            EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  STD_TORCH_CHECK(a.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);

  using Cutlass3xGemmDefault =
      typename sm120_fp8_config_default<InType, OutType,
                                        Epilogue>::Cutlass3xGemm;
  return cutlass_gemm_caller<Cutlass3xGemmDefault>(
      out, a, b, std::forward<EpilogueArgs>(args)...);
}

template <template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm120_fp8_epilogue(torch::stable::Tensor& out,
                                          torch::stable::Tensor const& a,
                                          torch::stable::Tensor const& b,
                                          EpilogueArgs&&... epilogue_args) {
  STD_TORCH_CHECK(a.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);

  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    return cutlass_gemm_sm120_fp8_dispatch<cutlass::float_e4m3_t,
                                           cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
    return cutlass_gemm_sm120_fp8_dispatch<cutlass::float_e4m3_t,
                                           cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

}  // namespace vllm
