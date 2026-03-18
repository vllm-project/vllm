/*
 * Persistent TopK Scheduler for DSA (DeepSeek Attention) Indexer
 *
 * Single persistent kernel with dynamic per-row path selection:
 *   - Medium path (seq_len <= 64K): CTA 0 runs radix select, others skip
 *   - Large path  (seq_len > 64K): all CTAs in group cooperate
 *
 * CUDAGraph-safe: fixed grid configuration handles all seq_lens.
 * The grid is always set up for the large path (worst case). For medium
 * rows, only CTA 0 of each group does work — others skip with no barrier
 * overhead. The inter-CTA barrier protocol naturally handles timing
 * differences between fast-skipping and working CTAs.
 *
 * Key optimizations:
 *   - __launch_bounds__(1024) with __noinline__ paths for register isolation
 *   - CTA groups with round-robin row assignment
 *   - __noinline__ on path functions for independent register allocation
 */

#ifndef PERSISTENT_TOPK_CUH_
#define PERSISTENT_TOPK_CUH_

#include "persistent_topk_common.cuh"
#include "persistent_topk_medium.cuh"
#include "persistent_topk_large.cuh"

namespace vllm {
namespace persistent {

// ============================================================================
// Persistent kernel — unified, CUDAGraph-safe
// ============================================================================

template <uint32_t VEC_SIZE = 1>
__global__ void __launch_bounds__(kThreadsPerBlock)
    persistent_topk_kernel(PersistentTopKParams params) {
  large_topk_cuda<VEC_SIZE>(params);
}

}  // namespace persistent
}  // namespace vllm

#endif  // PERSISTENT_TOPK_CUH_
