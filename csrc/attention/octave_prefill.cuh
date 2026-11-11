// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cstdint>
#include <torch/all.h>

namespace octave {

struct PrefillArgs {
  torch::Tensor out, q, k, v, cache;
  torch::Tensor block_table, cu_seqlens_q, seq_lens;
  int64_t max_query_len;
  double sm_scale;
  int64_t fmt;
  // Prefix partials: [num_splits, q.size(0), q.size(1), D + 2]. Each row holds
  // unnormalized output, the score maximum in natural-log units, and exp sum.
  float* ws;
  int num_splits;
};

}  // namespace octave

// The public entry owns validation, target selection, and the device guard.
// Launchers use its split count and workspace; Q/K/V and output stay rotated.
namespace octave_rdna {
// Launch the gfx1100 prefix phase, then merge it with causal chunk attention.
void octave_prefill(const octave::PrefillArgs& args);
}  // namespace octave_rdna

namespace octave_cdna {
// Launch the gfx942 prefix phase, then merge it with causal chunk attention.
void octave_prefill(const octave::PrefillArgs& args);
}  // namespace octave_cdna
