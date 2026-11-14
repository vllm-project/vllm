// Blockwise-scaled FP8 GEMM for Ada (SM89) using CUTLASS 2.x.
//
//   out[M, N] = (A[M, K] * a_scales) @ (B[K, N] * b_scales)
//
// A is FP8 e4m3, row-major, with one fp32 scale per token per 128 values of K
// (a_scales: [M, ceil(K/128)], any strides; vLLM passes it column-major).
// B is FP8 e4m3, column-major, with one fp32 scale per 128x128 block
// (b_scales: [ceil(K/128), ceil(N/128)], any strides).
//
// The mainloop is adapted from CUTLASS's Ada blockwise GEMM (example 94),
// which applies one A scale per 128x128 block; here A scales are per row.

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "torch_utils.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/numeric_types.h"

#include "cutlass_extensions/gemm/kernel/gemm_blockwise_pertoken.h"

namespace vllm {

namespace {

constexpr int64_t kScaleBlock = 128;

template <typename ElementOut, typename TileShape, typename WarpShape,
          int Stages, int SwizzleN = 1>
using Sm89BlockwiseKernel =
    typename cutlass::gemm::kernel::GemmBlockwisePerTokenSm89<
        cutlass::float_e4m3_t, cutlass::layout::RowMajor, 16,
        cutlass::float_e4m3_t, cutlass::layout::ColumnMajor, 16, ElementOut,
        float, TileShape, WarpShape, cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<ElementOut, 8, float,
                                                     float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<SwizzleN>,
        Stages, cutlass::arch::OpMultiplyAdd>::GemmKernel;

template <typename GemmKernel, typename ElementOut>
void run_sm89_blockwise(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b,
    cutlass::gemm::threadblock::BlockwiseScaleArgs const& scales) {
  int const m = static_cast<int>(a.size(0));
  int const k = static_cast<int>(a.size(1));
  int const n = static_cast<int>(b.size(1));

  cutlass::gemm::GemmCoord problem(m, n, k);
  using Swizzle = typename GemmKernel::ThreadblockSwizzle;
  using Shape = typename GemmKernel::Mma::Shape;
  cutlass::gemm::GemmCoord tiled = Swizzle().get_tiled_shape(
      problem, {Shape::kM, Shape::kN, Shape::kK}, /*split_k_slices=*/1);
  dim3 grid = Swizzle().get_grid_shape(tiled);
  dim3 block(GemmKernel::kThreadCount, 1, 1);
  int const smem = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));

  auto* a_ptr = static_cast<cutlass::float_e4m3_t*>(a.data_ptr());
  auto* b_ptr = static_cast<cutlass::float_e4m3_t*>(b.data_ptr());
  auto* out_ptr = static_cast<ElementOut*>(out.data_ptr());

  typename GemmKernel::Params params(
      problem, tiled, {a_ptr, cutlass::layout::RowMajor(a.stride(0))},
      {b_ptr, cutlass::layout::ColumnMajor(b.stride(1))},
      {out_ptr, cutlass::layout::RowMajor(out.stride(0))},  // C (beta = 0)
      {out_ptr, cutlass::layout::RowMajor(out.stride(0))},  // D
      scales, {1.f, 0.f});

  if (smem >= (48 << 10)) {
    cudaError_t err =
        cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    STD_TORCH_CHECK(err == cudaSuccess,
                    "cudaFuncSetAttribute failed: ", cudaGetErrorString(err));
  }

  cudaStream_t stream = get_current_cuda_stream(a.get_device_index());
  cutlass::Kernel<GemmKernel><<<grid, block, smem, stream>>>(params);
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "sm89 blockwise FP8 GEMM launch failed: ",
                  cudaGetErrorString(err));
}

template <typename ElementOut>
void dispatch_sm89_blockwise(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b,
    cutlass::gemm::threadblock::BlockwiseScaleArgs const& scales) {
  // Tile configs by M, picked from an L4 sweep of 16 configs over Qwen3-8B
  // layer shapes (M = 1 to 4096). K tile is always 128 (one scale block).
  using cutlass::gemm::GemmShape;
  int64_t const m = a.size(0);
  if (m <= 16) {
    // Decode: 16-row tiles, deep pipeline to keep weight loads in flight.
    using Kernel = Sm89BlockwiseKernel<ElementOut, GemmShape<16, 128, 128>,
                                       GemmShape<16, 64, 128>, 5>;
    run_sm89_blockwise<Kernel, ElementOut>(out, a, b, scales);
  } else if (m <= 32) {
    using Kernel = Sm89BlockwiseKernel<ElementOut, GemmShape<32, 64, 128>,
                                       GemmShape<16, 32, 128>, 4>;
    run_sm89_blockwise<Kernel, ElementOut>(out, a, b, scales);
  } else if (m <= 64) {
    using Kernel = Sm89BlockwiseKernel<ElementOut, GemmShape<64, 128, 128>,
                                       GemmShape<32, 64, 128>, 3>;
    run_sm89_blockwise<Kernel, ElementOut>(out, a, b, scales);
  } else {
    using Kernel = Sm89BlockwiseKernel<ElementOut, GemmShape<128, 128, 128>,
                                       GemmShape<64, 32, 128>, 3, 4>;
    run_sm89_blockwise<Kernel, ElementOut>(out, a, b, scales);
  }
}

}  // namespace

void cutlass_scaled_mm_blockwise_sm89_fp8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(a.scalar_type() == ScalarType::Float8_e4m3fn &&
                  b.scalar_type() == ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(a_scales.scalar_type() == ScalarType::Float &&
                  b_scales.scalar_type() == ScalarType::Float);

  int64_t const m = a.size(0), k = a.size(1), n = b.size(1);
  int64_t const kb = (k + kScaleBlock - 1) / kScaleBlock;
  int64_t const nb = (n + kScaleBlock - 1) / kScaleBlock;

  STD_TORCH_CHECK(
      a_scales.dim() == 2 && a_scales.size(0) == m && a_scales.size(1) == kb,
      "a_scales must be [M, ceil(K/128)] (group shape [1, 128]).");
  STD_TORCH_CHECK(
      b_scales.dim() == 2 && b_scales.size(0) == kb && b_scales.size(1) == nb,
      "b_scales must be [ceil(K/128), ceil(N/128)] "
      "(group shape [128, 128]).");
  // 16-element alignment for the FP8 operands and the output.
  STD_TORCH_CHECK(k % 16 == 0 && n % 16 == 0,
                  "sm89 blockwise FP8 GEMM needs K and N divisible by 16.");

  cutlass::gemm::threadblock::BlockwiseScaleArgs scales;
  scales.ptr_a = static_cast<float const*>(a_scales.data_ptr());
  scales.a_stride_m = a_scales.stride(0);
  scales.a_stride_k = a_scales.stride(1);
  // b_scales is indexed [kb, nb]; the mainloop asks for (nb, kb).
  scales.ptr_b = static_cast<float const*>(b_scales.data_ptr());
  scales.b_stride_n = b_scales.stride(1);
  scales.b_stride_k = b_scales.stride(0);
  scales.num_k_blocks = static_cast<int>(kb);

  if (out.scalar_type() == ScalarType::BFloat16) {
    dispatch_sm89_blockwise<cutlass::bfloat16_t>(out, a, b, scales);
  } else {
    STD_TORCH_CHECK(out.scalar_type() == ScalarType::Half);
    dispatch_sm89_blockwise<cutlass::half_t>(out, a, b, scales);
  }
}

}  // namespace vllm
