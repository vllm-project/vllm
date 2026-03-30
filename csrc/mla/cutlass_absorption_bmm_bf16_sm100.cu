// clang-format off
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "core/math.hpp"
#include "cutlass_extensions/common.hpp"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

// For cutlass_gemm_caller
#include "quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh"
// clang-format on

/*
 * CUTLASS BF16×BF16→FP8 Batched GEMM for MLA Absorption BMM
 *
 * Computes: q_out[:, :, :N] = fp8(scale_a * (q_nope_bf16 @ W_UK_bf16^T))
 *
 * A = q_nope_bf16:  [L=N_heads, M=B, K=128]  bf16 (may be non-contiguous)
 * B = W_UK_bf16:    [L=N_heads, N=512, K=128] bf16 contiguous (pre-dequantized)
 * D = q_out view:   written with strides into [B, N_heads, 576]
 *
 * SM100 MMA requires both operands to be the same type, so W_UK is
 * pre-dequantized from FP8 to BF16 during model init (one-time cost,
 * ~8MB extra per layer — negligible on B200's 192GB HBM).
 *
 * The ScaledEpilogue applies scale_a and casts the FP32 accumulator to FP8.
 * This avoids runtime FP8 quantization of q_nope entirely.
 */

using namespace cute;

namespace vllm {

// ============================================================
// GEMM template: BF16×BF16→FP8 with ScaledEpilogue
// ============================================================

template <typename ElementAB_, typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape, typename ClusterShape, typename KernelSchedule,
          typename EpilogueSchedule>
struct cutlass_3x_bmm_sm100_bf16 {
  using ElementAB = ElementAB_;  // bfloat16_t for both A and B
  using ElementC = ElementD_;
  using ElementD = ElementD_;  // float_e4m3_t (output)
  using ElementAcc = float;

  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;
  using EVTCompute = typename Epilogue::EVTCompute;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutC = LayoutD;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, TileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc, float, ElementC, LayoutC, AlignmentCD, ElementD, LayoutD,
          AlignmentCD, EpilogueSchedule, EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);

  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  // BF16×BF16 mainloop — SM100 requires both operands same type
  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementAB,
          LayoutA, AlignmentAB, ElementAB, LayoutB, AlignmentAB, ElementAcc,
          TileShape, ClusterShape, Stages, KernelSchedule>::CollectiveOp;

  using GemmKernel = enable_sm100f_only<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>>;
};

// ============================================================
// Batched GEMM caller for BF16×BF16→FP8
// ============================================================

template <typename Gemm>
void cutlass_bmm_caller_sm100_bf16(
    torch::Tensor& out,            // [B, N_heads, D_cols] fp8
    torch::Tensor const& a,        // [N_heads, B, K] bf16 (may be strided)
    torch::Tensor const& b,        // [N_heads, N, K] bf16
    torch::Tensor const& scale_a,  // [1] float
    torch::Tensor const& scale_b   // [1] float
) {
  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = StrideC;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  int32_t L = a.size(0);         // N_heads (batch count)
  int32_t M = a.size(1);         // B_tokens
  int32_t K = a.size(2);         // qk_nope_head_dim (128)
  int32_t N = b.size(1);         // kv_lora_rank (512)
  int64_t D_cols = out.size(2);  // total output cols (e.g., 576)

  auto prob_shape = cute::make_shape(M, N, K, L);

  // A: [L, M, K] -- use tensor strides to avoid .contiguous() memcpy.
  StrideA a_stride;
  get<0>(a_stride) = a.stride(1);  // M stride
  // get<1> is Int<1>{} -- K stride, always contiguous
  get<2>(a_stride) = a.stride(0);  // batch stride

  // B: [L, N, K] contiguous — packed strides
  StrideB b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));

  // D: STRIDED output — write [M, N] per batch into [M, L, D_cols]
  //   D[m, n] for batch l  =>  out[m, l, n]
  //   out is [M, L, D_cols] C-contiguous with strides (L*D_cols, D_cols, 1)
  //   M_stride = L * D_cols,  N_stride = 1,  batch_stride = D_cols
  StrideD d_stride;
  get<0>(d_stride) = static_cast<int64_t>(L) * D_cols;
  // get<1> is Int<1>{} — already set by default
  get<2>(d_stride) = static_cast<int64_t>(D_cols);

  StrideC c_stride = d_stride;

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  auto d_ptr = static_cast<ElementD*>(out.data_ptr());

  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, a_stride, b_ptr,
                                                       b_stride};

  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(scale_a, scale_b), d_ptr, c_stride, d_ptr,
      d_stride};

  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                       epilogue_args);
}

// ============================================================
// Tile configs: BF16×BF16→FP8 on SM100
// ============================================================

using InType = cutlass::bfloat16_t;
using OutType = cutlass::float_e4m3_t;
using KS = cutlass::gemm::collective::KernelScheduleAuto;
using ES = cutlass::epilogue::collective::EpilogueScheduleAuto;

// M > 256
using BmmGemm_BF16_Default =
    cutlass_3x_bmm_sm100_bf16<InType, OutType, c3x::ScaledEpilogue,
                              Shape<_256, _128, _128>, Shape<_2, _2, _1>, KS,
                              ES>;

// 64 < M <= 256
using BmmGemm_BF16_M256 =
    cutlass_3x_bmm_sm100_bf16<InType, OutType, c3x::ScaledEpilogue,
                              Shape<_128, _128, _128>, Shape<_2, _1, _1>, KS,
                              ES>;

// M <= 64
using BmmGemm_BF16_M64 =
    cutlass_3x_bmm_sm100_bf16<InType, OutType, c3x::ScaledEpilogue,
                              Shape<_64, _64, _128>, Shape<_1, _1, _1>, KS, ES>;

// M <= 16
using BmmGemm_BF16_M16 =
    cutlass_3x_bmm_sm100_bf16<InType, OutType, c3x::ScaledEpilogue,
                              Shape<_128, _32, _128>, Shape<_1, _1, _1>, KS,
                              ES>;

}  // namespace vllm

// ============================================================
// Entry point (global namespace — matches ops.h declaration)
// ============================================================

void mla_absorption_bmm_bf16(torch::Tensor& out, torch::Tensor const& a,
                             torch::Tensor const& b,
                             torch::Tensor const& scale_a,
                             torch::Tensor const& scale_b) {
  TORCH_CHECK(a.dtype() == torch::kBFloat16, "a must be bf16, got ", a.dtype());
  TORCH_CHECK(b.dtype() == torch::kBFloat16, "b must be bf16, got ", b.dtype());
  TORCH_CHECK(out.dtype() == torch::kFloat8_e4m3fn,
              "out must be fp8_e4m3, got ", out.dtype());
  TORCH_CHECK(a.dim() == 3, "a must be 3D [L, M, K]");
  TORCH_CHECK(b.dim() == 3, "b must be 3D [L, N, K]");
  TORCH_CHECK(out.dim() == 3, "out must be 3D [M, L, D_cols]");
  // a may be non-contiguous (e.g., from transpose); K dim must be contiguous
  TORCH_CHECK(a.stride(2) == 1, "a innermost dim (K) must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
  TORCH_CHECK(scale_a.is_contiguous() && scale_b.is_contiguous());
  TORCH_CHECK(a.size(0) == b.size(0), "batch (L) mismatch: ", a.size(0), " vs ",
              b.size(0));
  TORCH_CHECK(a.size(2) == b.size(2), "K mismatch: ", a.size(2), " vs ",
              b.size(2));
  TORCH_CHECK(out.size(0) == a.size(1), "M mismatch: out[0]=", out.size(0),
              " vs a[1]=", a.size(1));
  TORCH_CHECK(out.size(1) == a.size(0), "L mismatch: out[1]=", out.size(1),
              " vs a[0]=", a.size(0));
  TORCH_CHECK(out.size(2) >= b.size(1), "out cols ", out.size(2),
              " must be >= N=", b.size(1));

  uint32_t M = a.size(1);

  if (M <= 16) {
    return vllm::cutlass_bmm_caller_sm100_bf16<vllm::BmmGemm_BF16_M16>(
        out, a, b, scale_a, scale_b);
  } else if (M <= 64) {
    return vllm::cutlass_bmm_caller_sm100_bf16<vllm::BmmGemm_BF16_M64>(
        out, a, b, scale_a, scale_b);
  } else if (M <= 256) {
    return vllm::cutlass_bmm_caller_sm100_bf16<vllm::BmmGemm_BF16_M256>(
        out, a, b, scale_a, scale_b);
  } else {
    return vllm::cutlass_bmm_caller_sm100_bf16<vllm::BmmGemm_BF16_Default>(
        out, a, b, scale_a, scale_b);
  }
}
