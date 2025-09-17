#pragma once

#include "scaled_mm.cuh"
#include "cutlass_gemm_caller.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

/**
 * This file defines Gemm kernel configurations for SM90 (fp8) based on the Gemm
 * shape.
 */

namespace vllm {

using c3x::cutlass_gemm_caller;

template <typename ElementAB_, typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule, bool swap_ab_ = false>
struct cutlass_3x_gemm_sm90_fp8 {
  using ElementAB = ElementAB_;
  using ElementC = ElementD_;
  using ElementD = ElementD_;
  using ElementAcc =
      typename std::conditional<std::is_same_v<ElementAB, int8_t>, int32_t,
                                float>::type;

  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;

  using EVTCompute = typename Epilogue::EVTCompute;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  // Compile-time swap_ab flag
  static constexpr bool swap_ab = swap_ab_;

  // -----------------------------------------------------------
  // Layout definitions
  // -----------------------------------------------------------
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_T = typename cutlass::layout::LayoutTranspose<LayoutA>::type;

  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_T = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutD>::type;

  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;

  // -----------------------------------------------------------
  // Collective epilogue (conditionally swap operands and layouts)
  // -----------------------------------------------------------
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc, float, ElementC,
          conditional_t<swap_ab, LayoutC_Transpose, LayoutC>, AlignmentCD,
          ElementD, conditional_t<swap_ab, LayoutD_Transpose, LayoutD>,
          AlignmentCD, EpilogueSchedule, EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);

  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  // -----------------------------------------------------------
  // Collective mainloop (conditionally swap operands and layouts)
  // -----------------------------------------------------------
  using CollectiveMainloop = conditional_t<
      swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementAB,
          LayoutB_T, AlignmentAB,             // Swapped B (as A)
          ElementAB, LayoutA_T, AlignmentAB,  // Swapped A (as B)
          ElementAcc, TileShape, ClusterShape, Stages,
          KernelSchedule>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementAB,
          LayoutA, AlignmentAB, ElementAB, LayoutB, AlignmentAB, ElementAcc,
          TileShape, ClusterShape, Stages, KernelSchedule>::CollectiveOp>;

  // -----------------------------------------------------------
  // Kernel definition
  // -----------------------------------------------------------
  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>>;

  struct GemmKernel : public KernelType {};
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_default {
  // M in (128, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;

  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_M128 {
  // M in (64, 128]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_M64_N1280 {
  // M in (16, 64], N in [1 1280]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _16, _256>;
  using ClusterShape = Shape<_1, _4, _1>;

  // enable swap AB for M < 64
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueColumnBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule, true>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule,
                               true>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_M64_N8192 {
  // M in (16, 64], N > 1280
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_1, _1, _1>;

  // enable swap AB for M < 64
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueColumnBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule, true>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule,
                               true>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_M16_N1280 {
  // M in [1, 16], N in [1, 1280]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _16, _256>;
  using ClusterShape = Shape<_1, _2, _1>;

  // enable swap AB for M < 64
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueColumnBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule, true>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule,
                               true>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm90_fp8_config_M16_N8192 {
  // M in [1, 16], N > 1280
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = typename cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _16, _256>;
  using ClusterShape = Shape<_1, _1, _1>;

  // enable swap AB for M < 64
  using Cutlass3xGemm = conditional_t<
      EnableBias,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogueColumnBias,
                               TileShape, ClusterShape, KernelSchedule,
                               EpilogueSchedule, true>,
      cutlass_3x_gemm_sm90_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                               ClusterShape, KernelSchedule, EpilogueSchedule,
                               true>>;
};

template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller_sm90_fp8(torch::Tensor& out, torch::Tensor const& a,
                                  torch::Tensor const& b,
                                  EpilogueArgs&&... epilogue_params) {
  static constexpr bool swap_ab = Gemm::swap_ab;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;

  int32_t m = a.size(0), n = b.size(1), k = a.size(1);
  auto prob_shape =
      swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);

  StrideA a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(
      StrideC{},
      swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  auto c_ptr = static_cast<ElementD*>(out.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args =
      swap_ab ? typename GemmKernel::MainloopArguments{b_ptr, b_stride, a_ptr,
                                                       a_stride}
              : typename GemmKernel::MainloopArguments{a_ptr, a_stride, b_ptr,
                                                       b_stride};

  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptr, c_stride, c_ptr, c_stride};

  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                       epilogue_args);
}

template <typename InType, typename OutType, bool EnableBias,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm90_fp8_dispatch(torch::Tensor& out,
                                           torch::Tensor const& a,
                                           torch::Tensor const& b,
                                           torch::Tensor const& a_scales,
                                           torch::Tensor const& b_scales,
                                           EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  using Cutlass3xGemmDefault =
      typename sm90_fp8_config_default<InType, OutType,
                                       EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM128 =
      typename sm90_fp8_config_M128<InType, OutType, EnableBias>::Cutlass3xGemm;

  using Cutlass3xGemmM64_N1280 =
      typename sm90_fp8_config_M64_N1280<InType, OutType,
                                         EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM64_N8192 =
      typename sm90_fp8_config_M64_N8192<InType, OutType,
                                         EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM16_N1280 =
      typename sm90_fp8_config_M16_N1280<InType, OutType,
                                         EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM16_N8192 =
      typename sm90_fp8_config_M16_N8192<InType, OutType,
                                         EnableBias>::Cutlass3xGemm;

  uint32_t const m = a.size(0);
  uint32_t const n = b.size(1);

  if (m <= 16) {
    // m in [1, 16]
    if (n <= 1280) {
      return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmM16_N1280>(
          out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
    }
    return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmM16_N8192>(
        out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
  } else if (m <= 64) {
    // m in (16, 64]
    if (n <= 1280) {
      return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmM64_N1280>(
          out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
    }
    return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmM64_N8192>(
        out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
  } else if (m <= 128) {
    // m in (64, 128]
    return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmM128>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  } else {
    // m in (128, inf)
    return cutlass_gemm_caller_sm90_fp8<Cutlass3xGemmDefault>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  }
}

template <bool EnableBias, typename... EpilogueArgs>
void cutlass_scaled_mm_sm90_fp8_epilogue(torch::Tensor& out,
                                         torch::Tensor const& a,
                                         torch::Tensor const& b,
                                         torch::Tensor const& a_scales,
                                         torch::Tensor const& b_scales,
                                         EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                          cutlass::bfloat16_t, EnableBias>(
        out, a, b, a_scales, b_scales,
        std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_sm90_fp8_dispatch<cutlass::float_e4m3_t,
                                          cutlass::half_t, EnableBias>(
        out, a, b, a_scales, b_scales,
        std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

}  // namespace vllm
