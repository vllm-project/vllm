// SPDX-License-Identifier: MIT
#pragma once
#include <cutlass/arch/barrier.h>
#include <stdint.h>

// Each producer warp owns two banks. A single drainer visits producer warps
// in order, consuming one published batch at a time. Epochs span query blocks.
template <int P = 8, int R = 4, int S = 8>
struct alignas(16) CandidateHandoff {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  Barrier full[2][P], empty[2][P];
  uint32_t records[2][P][R][S][32];
  uint32_t counts[2][P][R][32];
  uint32_t bases[2][P];
  __device__ void init(int tid) {
    if (tid < 2 * P) {
      full[tid / P][tid % P].init(32);
      empty[tid / P][tid % P].init(32);
    }
    cutlass::arch::fence_barrier_init();
  }
  __device__ void acquire(int producer, unsigned epoch) {
    empty[epoch & 1][producer].wait(((epoch >> 1) & 1) ^ 1);
    __syncwarp();
  }
  __device__ void publish(int producer, unsigned epoch, unsigned base) {
    if ((threadIdx.x & 31) == 0) bases[epoch & 1][producer] = base;
    __syncwarp();
    full[epoch & 1][producer].arrive();
  }
  template <class Store>
  __device__ void drain(int producer, unsigned epoch, unsigned rows,
                        unsigned row_base, int* totals, unsigned cap,
                        Store store) {
    const int lane = threadIdx.x & 31, bank = epoch & 1;
    full[bank][producer].wait((epoch >> 1) & 1);
    __syncwarp();
    const unsigned base = bases[bank][producer];
    // Two 16-bit prefix sums share each shuffle. A row contributes at
    // most 32*S records, so the low half cannot carry into the high half.
    static_assert(R == 4 && 32 * S < 65536);
    unsigned n[4];
    unsigned my_total = 0;
#pragma unroll
    for (unsigned row = 0; row < 4; ++row) {
      n[row] = counts[bank][producer][row][lane];
      unsigned total = __reduce_add_sync(0xffffffffu, n[row]);
      if (lane == row) my_total = total;
    }
    unsigned my_start = 0;
    if (lane < 4 && my_total)
      my_start = atomicAdd(totals + row_base + lane, my_total);
    unsigned p01 = n[0] | (n[1] << 16), p23 = n[2] | (n[3] << 16);
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
      unsigned a = __shfl_up_sync(0xffffffffu, p01, d);
      unsigned b = __shfl_up_sync(0xffffffffu, p23, d);
      if (lane >= d) {
        p01 += a;
        p23 += b;
      }
    }
    unsigned prefix[4] = {p01 & 65535u, p01 >> 16, p23 & 65535u, p23 >> 16};
#pragma unroll
    for (unsigned row = 0; row < 4; ++row) {
      unsigned start = __shfl_sync(0xffffffffu, my_start, row);
      for (unsigned j = 0; j < n[row]; ++j) {
        unsigned slot = start + prefix[row] - n[row] + j;
        unsigned record = records[bank][producer][row][j][lane];
        if (slot < cap)
          store(row_base + row, slot, record, base, producer * 32 + lane);
      }
    }
    __syncwarp();
    empty[bank][producer].arrive();
  }
};
