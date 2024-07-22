#pragma once

#include "scaled_mm_c2x.cuh"
#include "cutlass/float8.h"

/**
 * This file defines Gemm kernel configurations for SM89 based on the Gemm
 * shape.
 */

namespace vllm {

struct sm89_config_default {

  // M in (256, inf)

  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  static const int32_t MainLoopStages = 1;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out,
                       torch::Tensor const& a,
                       torch::Tensor const& b,
                       EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    uint32_t const n = a.size(1);
    uint32_t const np2 = next_pow_2(n);

#if 0
    using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
    static const int32_t MainLoopStages = 3;
    return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
        cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
        Epilogue, TileShape, WarpShape, InstructionShape,
        MainLoopStages, cutlass::arch::OpMultiplyAdd>>(out, a, b, std::forward<EpilogueArgs>(args)...);
#endif

    if (np2 <= 4096) {
      using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
      return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
          cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
          Epilogue, TileShape, WarpShape, InstructionShape,
          MainLoopStages, FP8MathOperator>>(out, a, b, std::forward<EpilogueArgs>(args)...);

    } else {
      using TileShape = typename cutlass::gemm::GemmShape<256, 128, 64>;
      return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
          cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
          Epilogue, TileShape, WarpShape, InstructionShape,
          MainLoopStages, FP8MathOperator>>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_config_M256 {

  // M in (128, 256]

  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  static const int32_t MainLoopStages = 1;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out,
                       torch::Tensor const& a,
                       torch::Tensor const& b,
                       EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    uint32_t const n = a.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 4096) {
      using TileShape = typename cutlass::gemm::GemmShape<128, 64, 64>;
      using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;

      return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
          cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
          Epilogue, TileShape, WarpShape, InstructionShape,
          MainLoopStages, FP8MathOperator>>(out, a, b, std::forward<EpilogueArgs>(args)...);

    } else if (np2 <= 8192) {
      using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;
      using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;

      return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
          cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
          Epilogue, TileShape, WarpShape, InstructionShape,
          MainLoopStages, FP8MathOperator>>(out, a, b, std::forward<EpilogueArgs>(args)...);

    } else {
      using TileShape = typename cutlass::gemm::GemmShape<256, 128, 64>;
      using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;

      return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
          cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
          Epilogue, TileShape, WarpShape, InstructionShape,
          MainLoopStages, FP8MathOperator>>(out, a, b, std::forward<EpilogueArgs>(args)...);

    }
  }
};

struct sm89_config_M128 {

  // M in (64, 128]

  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  static const int32_t MainLoopStages = 1;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out,
                       torch::Tensor const& a,
                       torch::Tensor const& b,
                       EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    uint32_t const n = a.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 4096) {
      using TileShape = cutlass::gemm::GemmShape<64, 64, 128>;
      using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;

      return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
          cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
          Epilogue, TileShape, WarpShape, InstructionShape,
          MainLoopStages, FP8MathOperator>>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 8192) {
      using TileShape = cutlass::gemm::GemmShape<128, 64, 64>;
      using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;

      return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
          cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
          Epilogue, TileShape, WarpShape, InstructionShape,
          MainLoopStages, FP8MathOperator>>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = cutlass::gemm::GemmShape<128, 128, 128>;
      using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;

      return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
          cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
          Epilogue, TileShape, WarpShape, InstructionShape,
          MainLoopStages, FP8MathOperator>>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_config_M64 {

  // M in (32, 64]

  using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;
  static const int32_t MainLoopStages = 1;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out,
                       torch::Tensor const& a,
                       torch::Tensor const& b,
                       EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    uint32_t const n = a.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 8192) {
      using TileShape = cutlass::gemm::GemmShape<64, 64, 128>;

      return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
          cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
          Epilogue, TileShape, WarpShape, InstructionShape,
          MainLoopStages, FP8MathOperator>>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = cutlass::gemm::GemmShape<64, 128, 128>;

      return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
          cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
          Epilogue, TileShape, WarpShape, InstructionShape,
          MainLoopStages, FP8MathOperator>>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_config_M32 {

  // M in [1, 32]

  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  static const int32_t MainLoopStages = 1;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out,
                       torch::Tensor const& a,
                       torch::Tensor const& b,
                       EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    uint32_t const n = a.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 8192) {
      using TileShape = typename cutlass::gemm::GemmShape<32, 64, 128>;
      using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;

      return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
          cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
          Epilogue, TileShape, WarpShape, InstructionShape,
          MainLoopStages, FP8MathOperator>>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = typename cutlass::gemm::GemmShape<32, 128, 128>;
      using FP8MathOperator = typename cutlass::arch::OpMultiplyAdd;

      return vllm::cutlass_gemm_caller<vllm::cutlass_2x_gemm<
          cutlass::arch::Sm89, vllm::enable_sm89_to_sm90, InType, OutType,
          Epilogue, TileShape, WarpShape, InstructionShape,
          MainLoopStages, FP8MathOperator>>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm89_dispatch(torch::Tensor& out,
                                       torch::Tensor const& a,
                                       torch::Tensor const& b,
                                       EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  // TODO check shared memory requirements

  uint32_t const m = a.size(0);
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(32), next_pow_2(m));  // next power of 2

  //return sm89_config_default::dispatch<InType, OutType, Epilogue>(out, a, b, std::forward<EpilogueArgs>(args)...);

#if 1
  if (mp2 <= 32) {
    // M in [1, 32]
    return sm89_config_M32::dispatch<InType, OutType, Epilogue>(out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 64) {
    // M in (32, 64]
    return sm89_config_M64::dispatch<InType, OutType, Epilogue>(out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // M in (64, 128]
    return sm89_config_M128::dispatch<InType, OutType, Epilogue>(out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 256) {
    // M in (128, 256]
    return sm89_config_M256::dispatch<InType, OutType, Epilogue>(out, a, b, std::forward<EpilogueArgs>(args)...);
  } else {
    // M in (256, inf)
    return sm89_config_default::dispatch<InType, OutType, Epilogue>(out, a, b, std::forward<EpilogueArgs>(args)...);
  }
#endif
}

}  // namespace vllm
