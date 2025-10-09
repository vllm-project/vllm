#pragma once

#include "scaled_mm_c2x.cuh"

/**
 * This file defines Gemm kernel configurations for SM75 based on the Gemm
 * shape.
 */

namespace vllm {

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm75_config_default {
  // This config is used in 2 cases,
  // - M in (256, inf]
  // - M in (64, 128]
  // Shared memory required by this Gemm 32768
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<8, 8, 16>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm75, enable_sm75_to_sm80, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 2>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm75_config_M256 {
  // M in (128, 256]
  // Shared memory required by this Gemm 65536
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<8, 8, 16>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm75, enable_sm75_to_sm80, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 2>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm75_config_M64 {
  // M in (32, 64]
  // Shared memory required by this Gemm 49152
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = typename cutlass::gemm::GemmShape<64, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<8, 8, 16>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm75, enable_sm75_to_sm80, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 2>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm75_config_M32 {
  // M in [1, 32]
  // Shared memory required by this Gemm 49152
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = typename cutlass::gemm::GemmShape<32, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<8, 8, 16>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm75, enable_sm75_to_sm80, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 2>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm75_dispatch(torch::Tensor& out,
                                       torch::Tensor const& a,
                                       torch::Tensor const& b,
                                       EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, int8_t>());
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  using Cutlass2xGemmDefault =
      typename sm75_config_default<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM256 =
      typename sm75_config_M256<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM128 = Cutlass2xGemmDefault;
  using Cutlass2xGemmM64 =
      typename sm75_config_M64<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM32 =
      typename sm75_config_M32<InType, OutType, Epilogue>::Cutlass2xGemm;

  // Due to shared memory requirements, some Gemms may fail to run on some
  // GPUs. As the name indicates, the Fallback Gemm is used as an alternative
  // in such cases.
  // sm75_config_default has the least shared-memory requirements.
  using FallbackGemm = Cutlass2xGemmDefault;

  uint32_t const m = a.size(0);
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(32), next_pow_2(m));  // next power of 2
  if (mp2 <= 32) {
    // M in [1, 32]
    return fallback_cutlass_gemm_caller<Cutlass2xGemmM32, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 64) {
    // M in (32, 64]
    return fallback_cutlass_gemm_caller<Cutlass2xGemmM64, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // M in (64, 128]
    return fallback_cutlass_gemm_caller<Cutlass2xGemmM128, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 256) {
    // M in (128, 256]
    return fallback_cutlass_gemm_caller<Cutlass2xGemmM256, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else {
    // M in (256, inf)
    return fallback_cutlass_gemm_caller<Cutlass2xGemmDefault, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
}

}  // namespace vllm
