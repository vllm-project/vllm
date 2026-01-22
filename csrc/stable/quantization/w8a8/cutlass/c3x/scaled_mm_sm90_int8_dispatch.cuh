#pragma once

#include "scaled_mm.cuh"
#include "cutlass_gemm_caller.cuh"
#include "stable/torch_utils.h"
#include <torch/headeronly/core/ScalarType.h>

/**
 * This file defines Gemm kernel configurations for SM90 (int8) based on the
 * Gemm shape.
 *
 */

namespace vllm {

using c3x::cutlass_gemm_caller;

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_default {
  // For M > 128 and any N
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule =
      typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_M128 {
  // For M in (64, 128] and any N
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule =
      typename cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_M64 {
  // For M in (32, 64] and any N
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_M32_NBig {
  // For M in [1, 32] and N >= 8192
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _256>;
  using ClusterShape = Shape<_1, _4, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm90_int8_config_M32_NSmall {
  // For M in [1, 32] and N < 8192
  static_assert(std::is_same<InType, int8_t>());
  using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_1, _8, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm<InType, OutType, Epilogue, TileShape, ClusterShape,
                      KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm90_int8_dispatch(torch::stable::Tensor& out,
                                            torch::stable::Tensor const& a,
                                            torch::stable::Tensor const& b,
                                            EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, int8_t>());
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Char);

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
      return cutlass_gemm_caller<Cutlass3xGemmM32NSmall>(
          out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      return cutlass_gemm_caller<Cutlass3xGemmM32NBig>(
          out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  } else if (mp2 <= 64) {
    // m in (32, 64]
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
void cutlass_scaled_mm_sm90_int8_epilogue(torch::stable::Tensor& out,
                                          torch::stable::Tensor const& a,
                                          torch::stable::Tensor const& b,
                                          EpilogueArgs&&... epilogue_args) {
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Char);

  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::bfloat16_t,
                                           Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
    return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

}  // namespace vllm
