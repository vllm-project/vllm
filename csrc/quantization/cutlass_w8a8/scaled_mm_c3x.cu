#include <stddef.h>
#include <torch/all.h>
#include "cutlass/cutlass.h"
#include "scaled_mm_c3x.cuh"

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_gemm_sm90_fp8_dispatch(torch::Tensor& out, torch::Tensor const& a,
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

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_gemm_sm90_int8_dispatch(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, int8_t>());
  TORCH_CHECK(a.dtype() == torch::kInt8);
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
void cutlass_scaled_mm_sm90_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  if (a.dtype() == torch::kInt8) {
    TORCH_CHECK(b.dtype() == torch::kInt8);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::bfloat16_t,
                                             Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);
      return cutlass_gemm_sm90_int8_dispatch<int8_t, cutlass::half_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  } else {
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
}

void cutlass_scaled_mm_sm90(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias) {
    TORCH_CHECK(bias->dtype() == c.dtype(),
                "currently bias dtype must match output dtype ", c.dtype());
    return cutlass_scaled_mm_sm90_epilogue<ScaledEpilogueBias>(
        c, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm90_epilogue<ScaledEpilogue>(c, a, b, a_scales,
                                                           b_scales);
  }
}

void cutlass_scaled_mm_azp_sm90(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                c10::optional<torch::Tensor> const& azp,
                                c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (azp) {
    return cutlass_scaled_mm_sm90_epilogue<ScaledEpilogueBiasAzpToken>(
        out, a, b, a_scales, b_scales, azp_adj, *azp, bias);
  } else {
    return cutlass_scaled_mm_sm90_epilogue<ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
}

// hyper-parameter sweep kernels

void cutlass_scaled_mm_sm90_dispatch(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     torch::Tensor const& a_scales,
                                     torch::Tensor const& b_scales,
                                     c10::optional<torch::Tensor> const& bias) {
  assert(!bias);

  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _256>;
  using ClusterShape = Shape<_1, _4, _1>;
  using AccType = float;

  if (out.dtype() == torch::kBFloat16) {
    using Cutlass3xGemm =
        cutlass_3x_gemm<cutlass::float_e4m3_t, cutlass::bfloat16_t,
                        ScaledEpilogue, TileShape, ClusterShape, KernelSchedule,
                        EpilogueSchedule, AccType,
                        cutlass::gemm::PersistentScheduler,
                        cutlass::gemm::GemmUniversalMode::kGemm>;

    return cutlass_gemm_caller<Cutlass3xGemm>(out, a, b, a_scales, b_scales);

  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);

    using Cutlass3xGemm =
        cutlass_3x_gemm<cutlass::float_e4m3_t, cutlass::half_t, ScaledEpilogue,
                        TileShape, ClusterShape, KernelSchedule,
                        EpilogueSchedule, AccType,
                        cutlass::gemm::PersistentScheduler,
                        cutlass::gemm::GemmUniversalMode::kGemm>;

    return cutlass_gemm_caller<Cutlass3xGemm>(out, a, b, a_scales, b_scales);
  }
}

void cutlass_simple_gemm_sm90_dispatch(torch::Tensor& out,
                                       torch::Tensor const& a,
                                       torch::Tensor const& b) {
  using KernelSchedule = typename cutlass::gemm::KernelTmaWarpSpecialized;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _256>;
  using ClusterShape = Shape<_1, _4, _1>;
  using AccType = float;

  if (out.dtype() == torch::kBFloat16) {
    using Cutlass3xGemm =
        cutlass_3x_simple_gemm<cutlass::float_e4m3_t, cutlass::bfloat16_t,
                               TileShape, ClusterShape, KernelSchedule, AccType,
                               cutlass::gemm::PersistentScheduler,
                               cutlass::gemm::GemmUniversalMode::kGemm>;

    return cutlass_simple_gemm_caller<Cutlass3xGemm>(out, a, b);

  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);

    using Cutlass3xGemm =
        cutlass_3x_simple_gemm<cutlass::float_e4m3_t, cutlass::half_t,
                               TileShape, ClusterShape, KernelSchedule, AccType,
                               cutlass::gemm::PersistentScheduler,
                               cutlass::gemm::GemmUniversalMode::kGemm>;

    return cutlass_simple_gemm_caller<Cutlass3xGemm>(out, a, b);
  }
}
