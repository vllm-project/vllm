#pragma once

#include "scaled_mm_c2x.cuh"

/**
 * This file defines Gemm kernel configurations for SM80 based on the Gemm
 * shape.
 */

namespace vllm {

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm80_config_default {
  // This config is used in 2 cases,
  //  - M in (128, inf)
  //  - M in (64, 128] and N >= 8192
  // Shared Memory required by this Gemm - 81920 bytes
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 5>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm80_config_M64 {
  // This config is used in 2 cases,
  // - M in (32, 64]
  // - M in (64, 128] and N < 8192
  // Shared Memory required by this Gemm - 122880 bytes
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = typename cutlass::gemm::GemmShape<64, 128, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 5>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm80_config_M32 {
  // M in (16, 32]
  // Shared Memory required by this Gemm - 61440 bytes
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = typename cutlass::gemm::GemmShape<32, 64, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 5>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm80_config_M16 {
  // M in [1, 16]
  // Shared Memory required by this Gemm - 51200 bytes
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = typename cutlass::gemm::GemmShape<16, 64, 128>;
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm80, enable_sm80_to_sm89, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 5>;
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm80_dispatch(torch::stable::Tensor& out,
                                       torch::stable::Tensor const& a,
                                       torch::stable::Tensor const& b,
                                       EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, int8_t>());
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Char);

  using Cutlass2xGemmDefault =
      typename sm80_config_default<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM128BigN =
      typename sm80_config_default<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM128SmallN =
      typename sm80_config_M64<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM64 =
      typename sm80_config_M64<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM32 =
      typename sm80_config_M32<InType, OutType, Epilogue>::Cutlass2xGemm;
  using Cutlass2xGemmM16 =
      typename sm80_config_M16<InType, OutType, Epilogue>::Cutlass2xGemm;

  // Due to shared memory requirements, some Gemms may fail to run on some
  // GPUs. As the name indicates, the Fallback Gemm is used as an alternative
  // in such cases.
  // sm80_config_M16 has the least shared-memory requirement. However,
  // based on some profiling, we select sm80_config_M32 as a better alternative
  // performance wise.
  using FallbackGemm =
      typename sm80_config_M32<InType, OutType, Epilogue>::Cutlass2xGemm;

  uint32_t const m = a.size(0);
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(16), next_pow_2(m));  // next power of 2
  if (mp2 <= 16) {
    // M in [1, 16]
    return fallback_cutlass_gemm_caller<Cutlass2xGemmM16, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 32) {
    // M in (16, 32]
    return fallback_cutlass_gemm_caller<Cutlass2xGemmM32, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 64) {
    // M in (32, 64]
    return fallback_cutlass_gemm_caller<Cutlass2xGemmM64, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // M in (64, 128]
    uint32_t const n = out.size(1);
    bool const small_n = n < 8192;
    if (small_n) {
      return fallback_cutlass_gemm_caller<Cutlass2xGemmM128SmallN,
                                          FallbackGemm>(
          out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      return fallback_cutlass_gemm_caller<Cutlass2xGemmM128BigN, FallbackGemm>(
          out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  } else {
    // M in (128, inf)
    return fallback_cutlass_gemm_caller<Cutlass2xGemmDefault, FallbackGemm>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
}

}  // namespace vllm
