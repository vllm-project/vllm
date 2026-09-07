#include "scaled_mm_kernels.hpp"
#include "scaled_mm_blockwise_sm120_fp8_dispatch.cuh"
#include "libtorch_stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

namespace vllm {

namespace {

// CTA swizzle for SM 12.x parts whose L2 does not hold the weight operand.
//
// On a GB10 (24 MiB L2) the blockwise kernel loses most of its throughput once
// the weight is re-streamed from DRAM per row of M tiles: a 16384x2560 FP8
// weight runs at 165 TFLOPS at M=4096 but 90 at M=8192 and 52 at M>=16384;
// 8192x8192 is at 54 from M=6144. With the tile scheduler's max_swizzle_size =
// 8 the same launches run at 150-174 TFLOPS at every M, bit-identical to the
// default order (ten N/K shapes, M 2048-16384, all cells identical). The one
// place the default order is better is a narrow band around M=4096 on the
// 2560-wide weights (167 vs 153 at 16384x2560, 160 vs 154 at 12288x2560);
// elsewhere the swizzled order is equal or up to 3.3x faster, so it is used
// whenever the weight exceeds the L2. Parts whose L2 holds the weight (RTX PRO
// 6000 Blackwell / GB202: 96-128 MiB) keep the default order, which is also
// the faster one there (2560x6144 at 15 MiB: 178 vs 163 at M=2048).
constexpr int kBlockwiseFp8SwizzleSize = 8;

int blockwise_fp8_swizzle_size(int64_t weight_bytes) {
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
  const int swizzle = blockwise_fp8_swizzle_size(b.size(1) * b.size(0));
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
