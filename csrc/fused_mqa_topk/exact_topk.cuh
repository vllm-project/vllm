// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Exact radix top-k over one row of fp32 scores, used by the GPU recovery
// (fp8_recovery_kernel) and by selector rows whose pivot bin overflows the tie
// buffer, plus the recovery's scratch-slot semaphore.
#pragma once

#include "device_utils.cuh"
#include <cstdint>

namespace fused_mqa_topk::device::litetopk_recovery {

struct alignas(16) Scratch {
  uint32_t hist[256];
  uint32_t prefix, mask, rank, greater;
  uint32_t greater_written, equal_written;
  uint32_t min_key, max_key;
};

// A slot can only be owned by a resident CTA, so waiting CTAs cannot prevent
// its owner from finishing. The bitset is zero again when the grid completes.
FMT_DEVICE uint32_t acquire_slot(uint32_t* state, uint32_t slots) {
  const uint32_t allowed = slots == 32 ? 0xffffffffu : (1u << slots) - 1u;
  for (;;) {
    const uint32_t observed = atomicCAS(state, 0u, 0u);
    const uint32_t available = allowed & ~observed;
    if (available) {
      const uint32_t slot = __ffs(available) - 1;
      if (atomicCAS(state, observed, observed | (1u << slot)) == observed)
        return slot;
    }
    __nanosleep(256);
  }
}

FMT_DEVICE void release_slot(uint32_t* state, uint32_t slot) {
  atomicAnd(state, ~(1u << slot));
}

FMT_DEVICE uint32_t ordered_key(float value) {
  uint32_t bits = __float_as_uint(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return 0;  // NaNs rank below finite values.
  if (magnitude == 0) bits = 0;           // Both signed zeros are the same tie.
  return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

FMT_DEVICE uint32_t reserve_warp(uint32_t mask, uint32_t* counter) {
  const uint32_t lane = threadIdx.x & 31u;
  uint32_t base = 0;
  if (lane == 0 && mask) base = atomicAdd(counter, __popc(mask));
  base = __shfl_sync(0xffffffffu, base, 0);
  return base + __popc(mask & ((1u << lane) - 1u));
}

// All threads participate. Only a 256-bin histogram is needed, irrespective of
// row length or tie population. `Caller` gives each calling kernel its own
// copy: kernels of one translation unit sharing this __noinline__ callee get
// different SASS.
template <typename Caller>
__device__ __noinline__ void exact_topk(const float* scores,
                                        const int32_t* indices, uint32_t n,
                                        uint32_t topk, int32_t* out,
                                        Scratch& scratch) {
  const uint32_t tx = threadIdx.x;
  const uint32_t wanted = min(n, topk);
  for (uint32_t i = tx; i < topk; i += blockDim.x) out[i] = -1;
  if (tx == 0) {
    scratch.prefix = 0;
    scratch.mask = 0;
    scratch.rank = wanted;
    scratch.min_key = 0xffffffffu;
    scratch.max_key = 0;
  }
  __syncthreads();
  if (wanted == 0) return;
  if (n <= topk) {
    for (uint32_t i = tx; i < n; i += blockDim.x) {
      out[i] = indices ? __ldcg(indices + i) : int32_t(i);
    }
    __syncthreads();
    return;
  }

  for (int shift = 24; shift >= 0; shift -= 8) {
    for (uint32_t i = tx; i < 256; i += blockDim.x) scratch.hist[i] = 0;
    __syncthreads();
    const uint32_t prefix = scratch.prefix, mask = scratch.mask;
    const bool first_pass = shift == 24;
    uint32_t local_min = 0xffffffffu, local_max = 0;
    for (uint32_t base = 0; base < n; base += blockDim.x) {
      const uint32_t i = base + tx;
      const uint32_t key = i < n ? ordered_key(__ldcg(scores + i)) : 0;
      if (first_pass && i < n) {
        local_min = min(local_min, key);
        local_max = max(local_max, key);
      }
      const bool active = i < n && (key & mask) == prefix;
      const uint32_t voters = __ballot_sync(0xffffffffu, active);
      if (active) {
        const uint32_t bin = (key >> shift) & 255u;
        const uint32_t peers = __match_any_sync(voters, bin);
        if ((tx & 31u) == uint32_t(__ffs(peers) - 1)) {
          atomicAdd(scratch.hist + bin, __popc(peers));
        }
      }
    }
    if (first_pass) {
      local_min = __reduce_min_sync(0xffffffffu, local_min);
      local_max = __reduce_max_sync(0xffffffffu, local_max);
      if ((tx & 31u) == 0) {
        atomicMin(&scratch.min_key, local_min);
        atomicMax(&scratch.max_key, local_max);
      }
    }
    __syncthreads();
    if (first_pass && scratch.min_key == scratch.max_key) {
      for (uint32_t i = tx; i < wanted; i += blockDim.x) {
        out[i] = indices ? __ldcg(indices + i) : int32_t(i);
      }
      __syncthreads();
      return;
    }
    if (tx == 0) {
      uint32_t above = 0;
      for (int bin = 255; bin >= 0; --bin) {
        const uint32_t count = scratch.hist[bin];
        if (above + count >= scratch.rank) {
          scratch.prefix |= static_cast<uint32_t>(bin) << shift;
          scratch.mask |= 255u << shift;
          scratch.rank -= above;
          break;
        }
        above += count;
      }
    }
    __syncthreads();
  }
  if (tx == 0) {
    scratch.greater = wanted - scratch.rank;
    scratch.greater_written = 0;
    scratch.equal_written = 0;
  }
  __syncthreads();
  const uint32_t threshold = scratch.prefix;
  const uint32_t greater = scratch.greater;
  for (uint32_t base = 0; base < n; base += blockDim.x) {
    const uint32_t i = base + tx;
    const uint32_t key = i < n ? ordered_key(__ldcg(scores + i)) : 0;
    const bool gt = i < n && key > threshold, eq = i < n && key == threshold;
    const uint32_t gt_slot =
        reserve_warp(__ballot_sync(0xffffffffu, gt), &scratch.greater_written);
    const uint32_t eq_slot =
        reserve_warp(__ballot_sync(0xffffffffu, eq), &scratch.equal_written);
    if (gt && gt_slot < greater)
      out[gt_slot] = indices ? __ldcg(indices + i) : int32_t(i);
    if (eq && eq_slot < wanted - greater)
      out[greater + eq_slot] = indices ? __ldcg(indices + i) : int32_t(i);
  }
  __syncthreads();
}

}  // namespace fused_mqa_topk::device::litetopk_recovery
