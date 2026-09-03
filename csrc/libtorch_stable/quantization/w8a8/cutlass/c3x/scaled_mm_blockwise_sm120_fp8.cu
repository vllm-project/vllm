#include <algorithm>

#include "scaled_mm_kernels.hpp"
#include "scaled_mm_blockwise_sm120_fp8_dispatch.cuh"
#include "libtorch_stable/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

namespace vllm {

namespace {

void blockwise_sm120_fp8_dispatch_dtype(torch::stable::Tensor& out,
                                        torch::stable::Tensor const& a,
                                        torch::stable::Tensor const& b,
                                        torch::stable::Tensor const& a_scales,
                                        torch::stable::Tensor const& b_scales) {
  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    cutlass_gemm_blockwise_sm120_fp8_dispatch<cutlass::bfloat16_t>(
        out, a, b, a_scales, b_scales);
  } else {
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
    cutlass_gemm_blockwise_sm120_fp8_dispatch<cutlass::half_t>(
        out, a, b, a_scales, b_scales);
  }
}

// Chunk M when the weight operand does not fit the device L2.
//
// On SM 12.x parts with a small L2 (GB10: 24 MiB) the blockwise kernel loses
// most of its throughput once the working set of a launch outgrows the L2 and
// the weight operand is re-streamed from DRAM per tile row: on a GB10 a
// 16384x2560 FP8 weight runs at 163 TFLOPS at M=4096 but 95 at M=8192 and 51
// at M=16384; a 5120x5120 weight collapses already at M=5120 (76 TFLOPS).
// Issuing the same GEMM in launches whose activation slice stays around
// kBlockwiseFp8ChunkBytes keeps it at 155-170 TFLOPS at every M. Chunking M is
// exact (each output element's K-reduction is unchanged; the chunked result
// is bit-identical to a single launch). Parts whose L2 holds the weight (RTX
// PRO 6000 Blackwell / GB202: 96-128 MiB) are not chunked.
constexpr int64_t kBlockwiseFp8ChunkBytes = 12ll
                                            << 20;  // activation bytes/launch
constexpr int64_t kBlockwiseFp8MaxChunkRows = 4096;
constexpr int64_t kBlockwiseFp8MinChunkRows = 512;

int64_t blockwise_fp8_chunk_rows(int64_t k) {
  int64_t rows = kBlockwiseFp8ChunkBytes / k;  // FP8: one byte per element
  rows = std::min(rows, kBlockwiseFp8MaxChunkRows);
  rows = std::max(rows, kBlockwiseFp8MinChunkRows);
  return rows / 4 * 4;
}

bool blockwise_fp8_should_chunk(int64_t m, int64_t k, int64_t weight_bytes) {
  // Degenerate shapes take the single-launch path (and its own checks).
  if (k <= 0 || m <= 0) return false;
  // Below ~1.5 chunks a single launch is still in its efficient regime.
  if (m < blockwise_fp8_chunk_rows(k) * 3 / 2) return false;
  const int64_t l2_bytes = get_device_prop()->l2CacheSize;
  return l2_bytes > 0 && weight_bytes > l2_bytes;
}

}  // namespace

void cutlass_scaled_mm_blockwise_sm120_fp8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales) {
  const int64_t m = a.size(0), k = a.size(1), n = b.size(1);
  // b is [K, N] FP8 (one byte per element).
  if (!blockwise_fp8_should_chunk(m, k, n * k)) {
    blockwise_sm120_fp8_dispatch_dtype(out, a, b, a_scales, b_scales);
    return;
  }
  // The activation scales are [M, K/128] stored column-major (M fastest), and
  // the kernel derives that layout from its own M, so each chunk needs its
  // scales re-laid out with the chunk's M as the leading dimension. A and
  // the output are row-major; row ranges of them are plain views.
  // Balance the chunks: the fewest launches of at most max_rows each, so no
  // chunk is tiny. Chunk starts stay multiples of 4 for the aligned dispatch.
  const int64_t max_rows = blockwise_fp8_chunk_rows(k);
  const int64_t num_chunks = (m + max_rows - 1) / max_rows;
  const int64_t chunk_rows = ((m + num_chunks - 1) / num_chunks + 3) / 4 * 4;
  torch::stable::Tensor a_v = a;
  torch::stable::Tensor out_v = out;
  torch::stable::Tensor a_scales_t = torch::stable::transpose(a_scales, 0, 1);
  for (int64_t i = 0; i < m; i += chunk_rows) {
    const int64_t rows = std::min(chunk_rows, m - i);
    torch::stable::Tensor a_c = torch::stable::narrow(a_v, 0, i, rows);
    torch::stable::Tensor out_c = torch::stable::narrow(out_v, 0, i, rows);
    torch::stable::Tensor scales_c = torch::stable::transpose(
        torch::stable::contiguous(
            torch::stable::narrow(a_scales_t, 1, i, rows)),
        0, 1);
    blockwise_sm120_fp8_dispatch_dtype(out_c, a_c, b, scales_c, b_scales);
  }
}

}  // namespace vllm
