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
// 5120x5120 is already at 117 at M=4096 and 74 from M=6144. With the tile
// scheduler's max_swizzle_size = 8 the same launches run at 150-168 TFLOPS at
// every M, bit-identical to the default order.
//
// The default order stays marginally faster while the activation slab is
// small enough for the weight tiles to survive in the L2 anyway: 16384x2560
// at M=4096 (10 MiB of A) is 165-170 vs 149-155 swizzled, 10240x2560 at
// M=4096 is 157 vs 152 and 16384x2560 at M=5120 (12.5 MiB) is 166 vs 153,
// while 5120x5120 at M=4096 (20 MiB of A) is 117 vs 160 and every shape at
// M=6144 (15 MiB) already gains. The measured crossover is the activation
// working set, not M alone, so the gate is A's byte count (empirical
// threshold between the 12.5 MiB loss and the 15 MiB gain: 14 MiB). Parts whose L2 holds the weight (RTX PRO 6000
// Blackwell / GB202: 96-128 MiB) keep the default order at every M.
constexpr int kBlockwiseFp8SwizzleSize = 8;
constexpr int64_t kBlockwiseFp8SwizzleMinActivationBytes = 14ll << 20;

int blockwise_fp8_swizzle_size(int64_t m, int64_t k, int64_t weight_bytes) {
  if (m * k < kBlockwiseFp8SwizzleMinActivationBytes) return 1;  // FP8: 1 B/elem
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
  const int swizzle =
      blockwise_fp8_swizzle_size(a.size(0), a.size(1), b.size(1) * b.size(0));
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
