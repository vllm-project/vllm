// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shapes and workspace layout shared by the fused MQA-logits + top-k kernel
// and its host wrapper. The Python side mirrors the workspace sizes in
// vllm/model_executor/kernels/attention/dsa/fused_mqa_topk.py (the host wrapper
// checks them).
#pragma once

#include <cstdint>

namespace fused_mqa_topk {

// Indexer shape: 32 heads of dimension 128.
constexpr uint32_t kNumHeads = 32, kHeadDim = 128;
// Query rows per block (the unit a CTA claims), threads per CTA, keys per
// scan tile.
constexpr uint32_t kBlockRows = 8, kNumThreads = 384, kTileKeys = 256;
// Largest key range per call.
constexpr uint32_t kMaxKeys = 1u << 20;
// Per-row workspace: the gated candidate slab (a row with more candidates is
// rescanned exactly) and the exact rescan's retained set.
constexpr uint32_t kCandidatesPerRow = 16384, kRetainedPerRow = 4096;
// Candidate / retained slots (of kBlockRows rows each) per persistent CTA: one
// block in selection, one claimed ahead.
constexpr uint32_t kSlotsPerCta = 2;

// Top-k sizes the kernel is compiled for.
constexpr bool is_supported_topk(int64_t k) {
  return k == 512 || k == 1024 || k == 2048;
}

}  // namespace fused_mqa_topk
