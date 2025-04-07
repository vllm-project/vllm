#pragma once

#include "scaled_mm.cuh"
#include "cutlass_gemm_caller.cuh"

/**
 * This file defines Gemm kernel configurations for SM90 (fp8) based on the Gemm
 * shape.
 */

namespace vllm {

using c3x::cutlass_gemm_caller;

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_default {
  // M in (128, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M128 {
  // M in (64, 128]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_fp8_config_M64 {
  // M in [1, 64]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _128>;
  using ClusterShape = Shape<_1, _8, _1>;

  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm90_fp8_dispatch(torch::Tensor& out,
                                           torch::Tensor const& a,
                                           torch::Tensor const& b,
                                           EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
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
    return cutlass_gemm_caller<Cutlass3xGemmM64>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // m in (64, 128]
    return cutlass_gemm_caller<Cutlass3xGemmM128>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else {
    // m in (128, inf)
    return cutlass_gemm_caller<Cutlass3xGemmDefault>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
}

template <template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm90_fp8_epilogue(torch::Tensor& out,
                                         torch::Tensor const& a,
                                         torch::Tensor const& b,
                                         EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                          cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                          cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

}  // namespace vllm