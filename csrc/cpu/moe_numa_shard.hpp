// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// NUMA-aware work splitting for the CPU MoE experts.
//
// The CPU MoE kernel parallelises each of its two GEMM stages over the output
// rows of the weight it reads: stage 1 over the ``2N`` rows of w1, stage 2 over
// the ``K`` rows of w2. Both are already the parallel axis, so giving a NUMA
// node a contiguous slice of that axis makes every thread read only pages that
// were placed on its own node -- no reduction, no barrier, no extra buffers.
//
// This header owns two things and nothing else:
//
//   * ``moe_numa_block_split``, the split rule, which the weight loader and the
//     kernel must agree on exactly. It is a pure function of (blocks, shards)
//     so both can call it instead of passing a table around.
//   * ``parallel_2d_numa``, the loop, which subdivides *each node's own slice*
//     among that node's threads.
//
// It lives outside ``sgl-kernels/`` because that directory is vendored from
// SGLang and re-synced wholesale; keeping this here leaves the vendored call
// sites a one-line change.

#pragma once

#include <algorithm>
#include <cmath>

#if defined(_OPENMP)
  #include <omp.h>
#endif

namespace vllm_cpu_moe_numa {

// The microkernel's column unit, i.e. ``block_size_n()`` in
// csrc/cpu/sgl-kernels/gemm.h. Duplicated here because the weight loader has to
// round shard boundaries to it without pulling the kernel headers in, and
// mirrored in cpu_numa_shard.py because the policy needs it too.
//
// moe_numa_parallel.hpp static_asserts this against the kernel's own value, so
// the three cannot drift apart silently: if gemm.h changes, the build breaks
// here rather than the loader placing pages on boundaries the kernel does not
// split on -- which would be slower and give no sign of it.
constexpr int kBlockN = 32;

// Number of shards the MoE weights were placed with, or 1 for the plain path.
// Set once from Python after the topology is known; read by the kernel on every
// call, so it must stay a plain load.
int& shard_count();

// Split ``blocks`` across ``shards``, in whole blocks, handing the remainder
// out one block at a time to the first shards.
//
// Whole blocks because the microkernel's unit is ``BLOCK_N`` columns, not one
// column: a slice that is not a multiple of it cannot be computed. Equal slices
// are *not* required, and insisting on them is what made an earlier draft of
// this decline for ``intermediate_size = 2880`` -- vLLM's own CPU MoE benchmark
// size -- because 720 is not a multiple of 32. Splitting 90 blocks as
// 23/23/22/22 instead bounds the imbalance to a single block (4.5%) rather than
// piling the remainder onto the last shard (9.5%).
inline void moe_numa_block_split(int blocks, int shards, int shard, int* begin,
                                 int* end) {
  const int base = blocks / shards;
  const int rem = blocks % shards;
  const int b = shard * base + std::min(shard, rem);
  *begin = b;
  *end = b + base + (shard < rem ? 1 : 0);
}

// True when ``blocks`` can be split across ``shards`` with every shard getting
// at least one block. Fewer blocks than shards would leave a memory controller
// with no work while its cores keep reading from the others, which measures
// worse than not sharding at all.
inline bool moe_numa_can_shard(int blocks, int shards) {
  return shards > 1 && blocks >= shards;
}

// ``parallel_2d`` with the ``n`` axis partitioned by NUMA node first.
//
// Thread ``ith`` is taken to belong to node ``ith / (nth / shards)``. That
// holds when the OpenMP threads are bound to CPUs in list order, which is what
// ``KMP_AFFINITY=...,explicit,proclist=[...]`` and ``GOMP_CPU_AFFINITY`` give.
// The generic ``OMP_PLACES`` fallback defines a single place containing every
// CPU and does not, so the policy that sets ``shard_count()`` has to verify the
// binding and leave it at 1 rather than assume.
//
// ``f`` is called exactly as ``parallel_2d`` calls it, with block indices into
// the full ``n`` range, so call sites only change which function they name.
template <typename func_t>
inline void parallel_2d_numa(int m, int n, int shards, int nth,
                             const func_t& f) {
#if defined(_OPENMP)
  const int threads_per_node = nth / shards;
  #pragma omp parallel num_threads(nth)
  {
    const int ith = omp_get_thread_num();
    const int node = ith / threads_per_node;
    const int ith_local = ith % threads_per_node;

    int node_nb0 = 0, node_nb1 = 0;
    moe_numa_block_split(n, shards, node, &node_nb0, &node_nb1);
    const int n_local = node_nb1 - node_nb0;

    // Same square-ish blocking as parallel_2d, but over this node's slice and
    // this node's threads. Because each node subdivides its own range, the
    // thread -> tile map does not need the swap a globally-partitioned version
    // would have needed: threads of a node cannot stray outside its pages.
    const float r = float(m) / float(std::max(1, n_local));
    int nth_m = int(std::ceil(std::sqrt(r * threads_per_node)));
    int nth_n = 1;
    for (; nth_m > 0; --nth_m) {
      nth_n = threads_per_node / nth_m;
      if (nth_m * nth_n == threads_per_node) {
        break;
      }
    }
    if (nth_m <= 0) {
      nth_m = 1;
      nth_n = threads_per_node;
    }

    const int ith_m = ith_local / nth_n;
    const int ith_n = ith_local % nth_n;

    const int block_m = (m + nth_m - 1) / nth_m;
    const int block_n = (n_local + nth_n - 1) / nth_n;

    const int begin_m = ith_m * block_m;
    const int end_m = std::min(m, begin_m + block_m);
    const int begin_n = node_nb0 + ith_n * block_n;
    const int end_n = std::min(node_nb1, begin_n + block_n);

    if (begin_m < end_m && begin_n < end_n) {
      f(begin_m, end_m, begin_n, end_n);
    }
  }
#else
  (void)shards;
  (void)nth;
  f(0, m, 0, n);
#endif
}

}  // namespace vllm_cpu_moe_numa
