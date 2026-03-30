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

// Reuse the SM100 FP8 template (includes scaled_mm.cuh + caller)
#include "quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8_dispatch.cuh"
// clang-format on

/*
 * CUTLASS FP8×FP8→FP8 Batched GEMM for MLA Absorption BMM
 *
 * Computes: q_out[:, :, :N] = scale * (q_nope_fp8 @ W_UK_fp8^T)
 *
 * A = q_nope_fp8:  [L=N_heads, M=B, K=128]  fp8  contiguous
 * B = W_UK_fp8:    [L=N_heads, N=512, K=128] fp8  contiguous
 * D = q_out view:  written with strides into [B, N_heads, 576]
 *
 * Uses CUTLASS 3.x GemmUniversal with L > 1 for batched operation
 * and non-packed D strides for strided output writes.
 */

using namespace cute;

namespace vllm {

// ============================================================
// Batched GEMM caller for SM100 FP8→FP8
// ============================================================

template <typename Gemm>
void cutlass_bmm_caller_sm100_fp8(
    torch::Tensor& out,            // [B, N_heads, D_cols] fp8
    torch::Tensor const& a,        // [N_heads, B, K] fp8
    torch::Tensor const& b,        // [N_heads, N, K] fp8
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

  // A: [L, M, K] contiguous — packed strides
  StrideA a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));

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
// Tile configs: FP8×FP8→FP8 on SM100, no swap_ab
// ============================================================

using InType = cutlass::float_e4m3_t;
using OutType = cutlass::float_e4m3_t;
using KS = cutlass::gemm::collective::KernelScheduleAuto;
using ES = cutlass::epilogue::collective::EpilogueScheduleAuto;

// M > 256
using BmmGemm_Default =
    cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogue,
                              Shape<_256, _128, _128>, Shape<_2, _2, _1>, KS,
                              ES, false>;

// 64 < M <= 256
using BmmGemm_M256 =
    cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogue,
                              Shape<_128, _128, _128>, Shape<_2, _1, _1>, KS,
                              ES, false>;

// M <= 64
using BmmGemm_M64 =
    cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogue,
                              Shape<_64, _64, _128>, Shape<_1, _1, _1>, KS, ES,
                              false>;

// M <= 16
using BmmGemm_M16 =
    cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogue,
                              Shape<_128, _32, _128>, Shape<_1, _1, _1>, KS, ES,
                              false>;

}  // namespace vllm

// ============================================================
// Entry point (global namespace — matches ops.h declaration)
// ============================================================

void mla_absorption_bmm(torch::Tensor& out, torch::Tensor const& a,
                        torch::Tensor const& b, torch::Tensor const& scale_a,
                        torch::Tensor const& scale_b) {
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn, "a must be fp8_e4m3, got ",
              a.dtype());
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn, "b must be fp8_e4m3, got ",
              b.dtype());
  TORCH_CHECK(out.dtype() == torch::kFloat8_e4m3fn,
              "out must be fp8_e4m3, got ", out.dtype());
  TORCH_CHECK(a.dim() == 3, "a must be 3D [L, M, K]");
  TORCH_CHECK(b.dim() == 3, "b must be 3D [L, N, K]");
  TORCH_CHECK(out.dim() == 3, "out must be 3D [M, L, D_cols]");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
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
    return vllm::cutlass_bmm_caller_sm100_fp8<vllm::BmmGemm_M16>(
        out, a, b, scale_a, scale_b);
  } else if (M <= 64) {
    return vllm::cutlass_bmm_caller_sm100_fp8<vllm::BmmGemm_M64>(
        out, a, b, scale_a, scale_b);
  } else if (M <= 256) {
    return vllm::cutlass_bmm_caller_sm100_fp8<vllm::BmmGemm_M256>(
        out, a, b, scale_a, scale_b);
  } else {
    return vllm::cutlass_bmm_caller_sm100_fp8<vllm::BmmGemm_Default>(
        out, a, b, scale_a, scale_b);
  }
}
