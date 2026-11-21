#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm120_fp8_dispatch.cuh"
#include "core/utils.hpp"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

namespace vllm {

namespace {

// CTA swizzle for prefill-sized GEMMs on SM 12.x parts.
//
// On a GB10 (DGX Spark, 24 MiB L2) the default CTA raster re-streams an operand
// from DRAM once both the weight and the activation slab outgrow the L2:
// Llama-3.1-8B gate_up (28672x4096) runs at ~185 TFLOPS at M=4096 but 52-65
// from M=6144, and 8192x8192 at ~42 from M=6144. With the tile scheduler's
// max_swizzle_size = 8 the gated launches run at 96-187 TFLOPS (1.0-4.6x,
// never slower), bit-identical to the default order (each output tile's
// K-reduction is unchanged). Below the gate the default order is equal or
// faster (the swizzle costs up to ~15% at M=1024 with K >= 14336, and up to
// ~8% when the weight fits in the L2), so it is only used when M >= 2048 and
// both the weight (N*K) and the activation slab (M*K) are at least 3/4 of the
// L2. With a 128 MiB L2 (RTX PRO 6000 Blackwell) the gate only opens for very
// large weights and prompts. Same scheduler argument as the SM 12.x blockwise
// FP8 kernel.
constexpr int kSm120Fp8SwizzleSize = 8;
constexpr int64_t kSm120Fp8SwizzleMinM = 2048;

int sm120_fp8_swizzle_size(int64_t m, int64_t n, int64_t k) {
  if (m < kSm120Fp8SwizzleMinM) {
    return 1;
  }
  const int64_t l2_bytes = get_device_prop()->l2CacheSize;
  const int64_t threshold = l2_bytes / 4 * 3;
  return (l2_bytes > 0 && m * k >= threshold && n * k >= threshold)
             ? kSm120Fp8SwizzleSize
             : 1;
}

}  // namespace

void cutlass_scaled_mm_sm120_fp8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    std::optional<torch::stable::Tensor> const& bias) {
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
  // a is [M, K] and b is [K, N], both FP8 (one byte per element).
  const int swizzle = sm120_fp8_swizzle_size(a.size(0), b.size(1), a.size(1));
  if (bias) {
    STD_TORCH_CHECK(bias->scalar_type() == out.scalar_type(),
                    "currently bias dtype must match output dtype ",
                    out.scalar_type());
    if (vllm_is_batch_invariant()) {
      return cutlass_scaled_mm_sm120_fp8_batch_invariant_epilogue<
          c3x::ScaledEpilogueBias>(out, a, b, a_scales, b_scales, *bias);
    }
    return cutlass_scaled_mm_sm120_fp8_epilogue<c3x::ScaledEpilogueBias>(
        out, a, b, swizzle, a_scales, b_scales, *bias);
  } else {
    if (vllm_is_batch_invariant()) {
      return cutlass_scaled_mm_sm120_fp8_batch_invariant_epilogue<
          c3x::ScaledEpilogue>(out, a, b, a_scales, b_scales);
    }
    return cutlass_scaled_mm_sm120_fp8_epilogue<c3x::ScaledEpilogue>(
        out, a, b, swizzle, a_scales, b_scales);
  }
}

}  // namespace vllm
