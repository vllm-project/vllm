#include "scaled_mm_kernels.hpp"
#include "scaled_mm_blockwise_sm120_fp8_dispatch.cuh"
#include "libtorch_stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

namespace vllm {

namespace {

// CTA swizzle for SM 12.x parts whose L2 does not hold the weight operand.
//
// On a GB10 (24 MiB L2) the blockwise kernel loses most of its throughput once
// the weight is re-streamed from DRAM per row of M tiles: a 16384x2560 FP8
// weight runs at 163 TFLOPS at M=4096 but 95 at M=8192 and 52 at M>=16384;
// 5120x5120 collapses to 74 from M=6144. With the tile scheduler's
// max_swizzle_size = 8 the same launches run at 150-168 TFLOPS at every M,
// bit-identical to the default order. The default order is marginally faster
// while the weight still fits the L2 (M <= 4096 on GB10: 170 vs 155 TFLOPS on
// the widest weight), so the swizzle is applied only past that point. Parts
// whose L2 holds the weight (RTX PRO 6000 Blackwell / GB202: 96-128 MiB) keep
// the default order at every M.
constexpr int kBlockwiseFp8SwizzleSize = 8;
constexpr int64_t kBlockwiseFp8SwizzleMinRows = 4096;

int blockwise_fp8_swizzle_size(int64_t m, int64_t weight_bytes) {
  if (m <= kBlockwiseFp8SwizzleMinRows) return 1;
  const int64_t l2_bytes = get_device_prop()->l2CacheSize;
  return (l2_bytes > 0 && weight_bytes > l2_bytes) ? kBlockwiseFp8SwizzleSize
                                                   : 1;
}

}  // namespace

void cutlass_scaled_mm_blockwise_sm120_fp8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales) {
  // b is [K, N] FP8 (one byte per element).
  const int swizzle = blockwise_fp8_swizzle_size(a.size(0), b.size(1) * b.size(0));
  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    cutlass_gemm_blockwise_sm120_fp8_dispatch<cutlass::bfloat16_t>(
        out, a, b, a_scales, b_scales, swizzle);
  } else {
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
    cutlass_gemm_blockwise_sm120_fp8_dispatch<cutlass::half_t>(
        out, a, b, a_scales, b_scales, swizzle);
  }
}

}  // namespace vllm
