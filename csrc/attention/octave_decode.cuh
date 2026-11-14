// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cstdint>
#include <optional>
#include <torch/all.h>
#include <cuda_runtime_api.h>

namespace octave {

// mid_o stores float32 [query, head, split, D + 2] partials: unnormalized
// rotated output, the score maximum in log2 units, and the exponential sum.
struct DecodeArgs {
  torch::Tensor query, cache, block_table, q_to_req, q_to_klen, mid_o, k_signs;
  double sm_scale;
  // Upper bound on splits; native launchers may select fewer.
  int64_t num_kv_splits;
  int64_t fmt;
  int64_t query_group;
};

struct SparseDecodeArgs {
  // [num_q, topk] int32 logical positions; negative entries are skipped.
  std::optional<torch::Tensor> indices;
  // Exact-window metadata: exclude tagged recent/sink tokens from packed
  // attention so the caller can handle them in a separate window-attention
  // split.
  const int* qpos = nullptr;
  const int* wtags = nullptr;
  int wmask = 0;
  const int* stags = nullptr;
  int smask = 0;
};

}  // namespace octave

// Public entries validate inputs and guard the query device; common dispatch
// selects the target. Native decode rotates Q and writes partials to mid_o.
// Return the selected split count, or 0 without launching when ineligible.
// Eligible empty queries return the count without launching. The caller owns
// workspace, reduction, and the decision to fall back or reject the request.
namespace octave_rdna {
// WMMA on gfx1100. With sparse.indices, args.q_to_klen is ignored;
// otherwise each query attends to its first q_to_klen tokens.
int octave_decode(const octave::DecodeArgs& args,
                  const octave::SparseDecodeArgs& sparse);
}  // namespace octave_rdna

namespace octave_cdna {
// Dense MFMA on gfx942; query_group > 1 groups four query rows, otherwise one.
int octave_decode(const octave::DecodeArgs& args);

// Merge splits, normalize, and undo V rotation on the supplied stream.
// Returns false without launching if the shape or batch is unsuitable,
// leaving the caller to use the portable reducer.
bool octave_reduce(torch::Tensor out, torch::Tensor mid_o,
                   torch::Tensor v_signs, int num_splits, cudaStream_t stream);
}  // namespace octave_cdna
