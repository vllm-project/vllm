// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The one entry point the vendored MoE kernels call, so that their diff against
// upstream SGLang stays a single renamed function per stage.
//
// Split from moe_numa_shard.hpp because this one needs common.h (for
// parallel_2d and adjust_num_threads) and the weight loader in utils.cpp must
// be able to use the split rule without pulling the whole kernel header in.

#pragma once

#include "cpu/moe_numa_shard.hpp"
#include "cpu/sgl-kernels/common.h"
#include "cpu/sgl-kernels/gemm.h"

// The loader and the policy both round shard boundaries to kBlockN without
// including the kernel headers. Pin it here, where both are visible.
static_assert(block_size_n() == vllm_cpu_moe_numa::kBlockN,
              "vllm_cpu_moe_numa::kBlockN is out of sync with the kernel's "
              "block_size_n(); update it and BLOCK_N in cpu_numa_shard.py.");

// Dispatch to the NUMA-sharded loop, or to the stock one when sharding is off
// or does not apply. The stock path is the *same call as before*, not a
// reimplementation: when nothing enabled sharding, this change is inert.
//
// Sharding is skipped, and the plain loop used, when:
//   * no policy set a shard count (the default, so every existing deployment
//     keeps its current behaviour byte for byte);
//   * the thread count is not a multiple of the shard count, so "thread ith
//     belongs to node ith / (nth / shards)" would not hold;
//   * there are fewer blocks than shards, which would leave a memory controller
//     idle while its cores read from the others.
template <typename func_t>
inline void parallel_2d_numa_or_plain(int m, int n, const func_t& f) {
  const int shards = vllm_cpu_moe_numa::shard_count();
  if (shards > 1 && vllm_cpu_moe_numa::moe_numa_can_shard(n, shards)) {
    const int nth = adjust_num_threads(m);
    if (nth % shards == 0 && nth >= shards) {
      vllm_cpu_moe_numa::parallel_2d_numa(m, n, shards, nth, f);
      return;
    }
  }
  parallel_2d(m, n, f);
}
