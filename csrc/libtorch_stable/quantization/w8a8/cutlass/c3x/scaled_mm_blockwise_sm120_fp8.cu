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
// Above the L2 the swizzled order is not unconditionally better. On a GB10
// (24 MiB L2) it loses to the default order across a band of middling
// activation sizes -- worst at M = 2560 (5120x5120 -10.4 %, 10240x2560
// -9.4 %, 16384x2560 -6.4 %), and 3-5 % at M = 4096 on the 2560-wide weights.
// It wins again once the activation slab is large (up to 3.2x), and it wins at
// very small M. Over four starts on 6 shapes x M = 1024..6144 (66 cells),
// gating on the weight alone leaves 118.8 percentage points on the table, the
// activation term alone 54.0, and the two together 35.3.
//
// The small-M island stops at 1024 because it is K-dependent: at K = 2560 the
// swizzle wins there, at K = 5120 the default order does, so an island reaching
// M = 2048 gives back 6.9-8.5 % on the 5120-wide weights.
constexpr int64_t kBlockwiseFp8SwizzleMinActivationBytes = 14ll << 20;
constexpr int64_t kBlockwiseFp8SwizzleSmallM = 1024;

int blockwise_fp8_swizzle_size(int64_t m, int64_t k, int64_t weight_bytes) {
  const int64_t l2_bytes = get_device_prop()->l2CacheSize;
  if (l2_bytes <= 0 || weight_bytes <= l2_bytes) return 1;
  if (m <= kBlockwiseFp8SwizzleSmallM) return kBlockwiseFp8SwizzleSize;
  // FP8 activations: one byte per element.
  return (m * k >= kBlockwiseFp8SwizzleMinActivationBytes)
             ? kBlockwiseFp8SwizzleSize
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
