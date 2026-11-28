#pragma once

#include <torch/headeronly/util/shim_utils.h>

#include <cuda_runtime.h>

#include "scaled_mm_c2x.cuh"

/**
 * This file defines Gemm kernel configurations for SM89 (int8) based on the
 * Gemm shape.
 */

namespace vllm {

// Number of SMs on the current device (cached per device).
inline int sm89_int8_sm_count() {
  static int counts[64] = {0};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 0 || dev >= 64) dev = 0;
  if (counts[dev] == 0) {
    cudaDeviceGetAttribute(&counts[dev], cudaDevAttrMultiProcessorCount, dev);
  }
  return counts[dev];
}

// For M in (16, 64], the 16-row tiles of sm89_int8_config_M16 launch 2x
// (vs the M32 config) or 4x (vs M64) as many CTAs. That wins while the extra
// CTAs fit in the same number of waves (one CTA per SM for these tile sizes)
// and loses once they spill into extra waves, which depends on the SM count:
// on an RTX 4090 (128 SMs) small-N layers stay in one wave, on 58-60 SM parts
// (L4, RTX 4500 Ada) they don't (vLLM #56924, #62128).
inline bool sm89_int8_use_m16_tiles(uint32_t m, uint32_t n) {
  if (next_pow_2(n) > 8192) return false;  // M16 switches to a 128-wide N tile
  uint32_t const sms = static_cast<uint32_t>(sm89_int8_sm_count());
  uint32_t const n_tiles = (n + 63) / 64;
  uint32_t const stock_rows = m <= 32 ? 32 : 64;  // M32 / M64 configs
  uint32_t const w16 = (((m + 15) / 16) * n_tiles + sms - 1) / sms;
  uint32_t const ws =
      (((m + stock_rows - 1) / stock_rows) * n_tiles + sms - 1) / sms;
  if (w16 == ws) return true;
  // Stock already needs several waves: the finer 16-row grid fills the tail
  // better, as long as it adds at most half again as many waves.
  return ws >= 2 && 2 * w16 <= 3 * ws;
}

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_int8_fallback_gemm {
  // Shared mem requirement : 61440
  static_assert(std::is_same<InType, int8_t>());
  using TileShape = cutlass::gemm::GemmShape<32, 64, 128>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static int32_t const MainLoopStages = 5;

  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 5>;
};

struct sm89_int8_config_default {
  // M in (256, inf)
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::stable::Tensor& out,
                       torch::stable::Tensor const& a,
                       torch::stable::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, int8_t>());
    STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);

    using FallbackGemm =
        typename sm89_int8_fallback_gemm<InType, OutType,
                                         Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 4096) {
      using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 8192) {
      using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 16384) {
      using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_int8_config_M256 {
  // M in (128, 256]
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::stable::Tensor& out,
                       torch::stable::Tensor const& a,
                       torch::stable::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, int8_t>());
    STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);

    using FallbackGemm =
        typename sm89_int8_fallback_gemm<InType, OutType,
                                         Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 4096) {
      using TileShape = cutlass::gemm::GemmShape<64, 128, 128>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 8192) {
      using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 16384) {
      using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_int8_config_M128 {
  // M in (64, 128]
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::stable::Tensor& out,
                       torch::stable::Tensor const& a,
                       torch::stable::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, int8_t>());
    STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);

    using FallbackGemm =
        typename sm89_int8_fallback_gemm<InType, OutType,
                                         Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 8192) {
      using TileShape = cutlass::gemm::GemmShape<64, 128, 128>;
      using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 16384) {
      using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
      using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = cutlass::gemm::GemmShape<64, 64, 128>;
      using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_int8_config_M64 {
  // M in (32, 64]
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::stable::Tensor& out,
                       torch::stable::Tensor const& a,
                       torch::stable::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, int8_t>());
    STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);

    using FallbackGemm =
        typename sm89_int8_fallback_gemm<InType, OutType,
                                         Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 8192) {
      using TileShape = cutlass::gemm::GemmShape<64, 64, 128>;
      using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = cutlass::gemm::GemmShape<64, 128, 128>;
      using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_int8_config_M32 {
  // M in (16, 32]
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::stable::Tensor& out,
                       torch::stable::Tensor const& a,
                       torch::stable::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, int8_t>());
    STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);

    using FallbackGemm =
        typename sm89_int8_fallback_gemm<InType, OutType,
                                         Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 8192) {
      using TileShape = cutlass::gemm::GemmShape<32, 64, 128>;
      using WarpShape = cutlass::gemm::GemmShape<16, 64, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = cutlass::gemm::GemmShape<32, 128, 128>;
      using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 4>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_int8_config_M16 {
  // M in [1, 16]
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::stable::Tensor& out,
                       torch::stable::Tensor const& a,
                       torch::stable::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, int8_t>());
    STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);

    using FallbackGemm =
        typename sm89_int8_fallback_gemm<InType, OutType,
                                         Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 8192) {
      using TileShape = cutlass::gemm::GemmShape<16, 64, 128>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = cutlass::gemm::GemmShape<16, 128, 128>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 4>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm89_int8_dispatch(torch::stable::Tensor& out,
                                            torch::stable::Tensor const& a,
                                            torch::stable::Tensor const& b,
                                            EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, int8_t>());
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Char);

  uint32_t const m = a.size(0);
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(16), next_pow_2(m));  // next power of 2

  if (mp2 <= 16) {
    // M in [1, 16]
    return sm89_int8_config_M16::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 32) {
    // M in (16, 32]
    if (sm89_int8_use_m16_tiles(m, out.size(1)))
      return sm89_int8_config_M16::dispatch<InType, OutType, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(args)...);
    return sm89_int8_config_M32::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 64) {
    // M in (32, 64]
    if (sm89_int8_use_m16_tiles(m, out.size(1)))
      return sm89_int8_config_M16::dispatch<InType, OutType, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(args)...);
    return sm89_int8_config_M64::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // M in (64, 128]
    return sm89_int8_config_M128::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 256) {
    // M in (128, 256]
    return sm89_int8_config_M256::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else {
    // M in (256, inf)
    return sm89_int8_config_default::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
}

}  // namespace vllm
