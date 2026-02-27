#pragma once

#include "scaled_mm.cuh"
#include "cutlass_gemm_caller.cuh"

/**
 * This file defines Gemm kernel configurations for SM120 (fp8) based on the
 * Gemm shape.
 */

namespace vllm {

using c3x::cutlass_gemm_caller;

// Custom wrapper to allow specifying EpilogueTile for small M
template <typename ElementAB_, typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule, typename EpilogueTile>
struct cutlass_3x_gemm_sm120_custom {
  using ElementAB = ElementAB_;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementAB>::value;

  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementAB>::value;

  using ElementC = void;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementD_>::value;

  using ElementD = ElementD_;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = AlignmentC;

  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;
  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;

  // MMA type
  using ElementAccumulator = float;

  // Epilogue types
  using ElementBias = cutlass::half_t;
  using ElementCompute = float;
  using ElementAux = ElementD;
  using LayoutAux = LayoutD;
  using ElementAmax = float;

  using EVTCompute = typename Epilogue::EVTCompute;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, EpilogueTile,  // Use custom EpilogueTile
          ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
          ElementD, LayoutD, AlignmentD, EpilogueSchedule,
          EVTCompute>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementAB,
          LayoutA, AlignmentA, ElementAB, LayoutB, AlignmentB,
          ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule, void>::CollectiveOp;

  using GemmKernel = enable_sm120_only<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>>;
};

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
          template <typename, typename, typename> typename Epilogue>
struct sm120_fp8_config_M64 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  // SM120 Cooperative kernel requires Tile M >= 128.
  // For M=64 tile, we use Pingpong schedule which is more flexible with small
  // tiles.
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _64, _128>;
  // CUTLASS 3.x on SM120 currently restricts programmatic multicast (Cluster >
  // 1) for certain schedules/types. Reverting to 1x1x1 to ensure compilation.
  using ClusterShape = Shape<_1, _1, _1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm_sm120<InType, OutType, Epilogue, TileShape, ClusterShape,
                            KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm120_fp8_config_M32 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_32, _64, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  // Use custom gemm to specify EpilogueTile M=32
  using Cutlass3xGemm =
      cutlass_3x_gemm_sm120_custom<InType, OutType, Epilogue, TileShape,
                                   ClusterShape, KernelSchedule,
                                   EpilogueSchedule, Shape<_32, _32>>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm120_fp8_config_M16 {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_16, _64, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  // Use custom gemm to specify EpilogueTile M=16
  using Cutlass3xGemm =
      cutlass_3x_gemm_sm120_custom<InType, OutType, Epilogue, TileShape,
                                   ClusterShape, KernelSchedule,
                                   EpilogueSchedule, Shape<_16, _32>>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm120_fp8_dispatch(torch::Tensor& out,
                                            torch::Tensor const& a,
                                            torch::Tensor const& b,
                                            EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  int M = a.size(0);

  if (M <= 16) {
    using Cutlass3xGemmM16 =
        typename sm120_fp8_config_M16<InType, OutType, Epilogue>::Cutlass3xGemm;
    return cutlass_gemm_caller<Cutlass3xGemmM16>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
  if (M <= 32) {
    using Cutlass3xGemmM32 =
        typename sm120_fp8_config_M32<InType, OutType, Epilogue>::Cutlass3xGemm;
    return cutlass_gemm_caller<Cutlass3xGemmM32>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }

  if (M <= 256) {
    using Cutlass3xGemmM64 =
        typename sm120_fp8_config_M64<InType, OutType, Epilogue>::Cutlass3xGemm;
    return cutlass_gemm_caller<Cutlass3xGemmM64>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }

  using Cutlass3xGemmDefault =
      typename sm120_fp8_config_default<InType, OutType,
                                        Epilogue>::Cutlass3xGemm;
  return cutlass_gemm_caller<Cutlass3xGemmDefault>(
      out, a, b, std::forward<EpilogueArgs>(args)...);
}

template <template <typename, typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm120_fp8_epilogue(torch::Tensor& out,
                                          torch::Tensor const& a,
                                          torch::Tensor const& b,
                                          EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_sm120_fp8_dispatch<cutlass::float_e4m3_t,
                                           cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_sm120_fp8_dispatch<cutlass::float_e4m3_t,
                                           cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

}  // namespace vllm
