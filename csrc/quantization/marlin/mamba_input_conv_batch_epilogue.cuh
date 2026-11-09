// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Batched request selection reuses the original short-convolution arithmetic.
// Its causal-conv attribution remains in mamba_input_conv_short_epilogue.cuh.
#pragma once

#include <limits>

#include "mamba_input_conv_short_epilogue.cuh"

namespace vllm_mamba_input_conv {

// Only the new .batched overload interprets reserved this way. Keep the
// original 80-byte descriptor and every B1 caller unchanged: low4 bits are
// physical H, the remaining positive bits are the first-column cache-index
// stride.
constexpr int kBatchHistoryBits = 4;
constexpr int kBatchIndexStrideMax =
    std::numeric_limits<int32_t>::max() >> kBatchHistoryBits;
static_assert(kBatchIndexStrideMax * 16 + 10 <=
              std::numeric_limits<int32_t>::max());

template <int Tokens>
__device__ __forceinline__ void epilogue_batch(void* shared_b, int slice_col,
                                               const ConvParams& params,
                                               uint16_t* projected, int rows) {
  static_assert(Tokens >= 1 && Tokens <= 4);
  // These CTAs have no convolution and must not read request metadata.
  if (slice_col < 64 || slice_col >= 144) return;
  const int batches = rows / Tokens;
  const int index_stride = params.reserved >> kBatchHistoryBits;
  const int history_len = params.reserved & ((1 << kBatchHistoryBits) - 1);
  // One CTA owns all M<=8 rows and one channel tile. Requests run sequentially;
  // every active epilogue ends at its original CTA barrier. Inactive returns
  // are CTA-uniform, before accepted/history access, so all lanes keep step.
  for (int batch = 0; batch < batches; ++batch) {
    ConvParams request = params;
    request.cache_indices += static_cast<int64_t>(batch) * index_stride;
    if (request.accepted_tokens) request.accepted_tokens += batch;
    request.reserved = history_len;
    auto* raw = reinterpret_cast<unsigned char*>(shared_b) +
                batch * Tokens * kRawPitch * sizeof(uint16_t);
    epilogue_short<Tokens>(
        raw, slice_col, request,
        projected + static_cast<int64_t>(batch) * Tokens * kColumns);
  }
}

// Shifting the reused history/conv scratch by at most7 raw rows stays entirely
// above the full raw8-row tile and inside the dead S3 B allocation.
static_assert(kSharedReuseBytes + 7 * kRawPitch * sizeof(uint16_t) <=
              kDepth3DeadBBytes);

}  // namespace vllm_mamba_input_conv
