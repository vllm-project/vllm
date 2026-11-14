// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Adapted from SGLang PR #47469
// (https://github.com/sgl-project/sglang/pull/47469, Apache-2.0) at
// 599923b6f48d638d2256abbd0550c8fde141f4f5:
// python/sglang/kernels/jit/csrc/dsa/litetopk_prefill/topk_candidates_common.cuh.
//
// Device helpers of the candidate top-k kernel.
#pragma once

#include "device_utils.cuh"
#include <cstdint>

namespace fused_mqa_topk {

namespace topk_cands {

// `v + shfl_down(v, kDelta)` where the source lane exists, `v` otherwise: the
// shuffle's in-range predicate guards the add (no lane compare and select).
template <uint32_t kDelta>
FMT_DEVICE uint32_t shfl_down_add(uint32_t v) {
  asm("{\n .reg .pred p;\n .reg .b32 t;\n shfl.sync.down.b32 t|p, %0, %1, 31, "
      "-1;\n @p add.u32 %0, %0, t;\n}"
      : "+r"(v)
      : "n"(kDelta));
  return v;
}

// Inclusive warp suffix sum: lane L gets the sum over lanes >= L.
FMT_DEVICE uint32_t suffix_sum(uint32_t v) {
  return shfl_down_add<16>(shfl_down_add<8>(
      shfl_down_add<4>(shfl_down_add<2>(shfl_down_add<1>(v)))));
}

// `v + shfl_up(v, kDelta)` where the source lane exists, `v` otherwise (see
// shfl_down_add).
template <uint32_t kDelta>
FMT_DEVICE uint32_t shfl_up_add(uint32_t v) {
  asm("{\n .reg .pred p;\n .reg .b32 t;\n shfl.sync.up.b32 t|p, %0, %1, 0, "
      "-1;\n @p add.u32 %0, %0, t;\n}"
      : "+r"(v)
      : "n"(kDelta));
  return v;
}

// Inclusive warp prefix sum: lane L gets the sum over lanes <= L.
FMT_DEVICE uint32_t prefix_sum(uint32_t v) {
  return shfl_up_add<16>(
      shfl_up_add<8>(shfl_up_add<4>(shfl_up_add<2>(shfl_up_add<1>(v)))));
}

// Warp-aggregated range reservation: every lane's `cnt` (packed counts) gets
// its exclusive prefix on top of one shared atomic per warp. Slots land in
// arrival order.
FMT_DEVICE uint32_t warp_reserve(uint32_t cnt, uint32_t* counter,
                                 uint32_t lane_id) {
  const uint32_t inc = prefix_sum(cnt);
  uint32_t base = 0;
  if (lane_id == device::kWarpThreads - 1) base = atomicAdd(counter, inc);
  base = __shfl_sync(device::warp::kFullMask, base, device::kWarpThreads - 1);
  return base + inc - cnt;
}

}  // namespace topk_cands

template <uint32_t N, uint32_t M, typename Fn>
FMT_DEVICE void static_loop(uint32_t id, Fn fn) {
  constexpr uint32_t kNumIters = N / M;
#pragma unroll
  for (uint32_t i = 0; i < kNumIters; ++i) {
    const uint32_t j = i * M + id;
    fn(j);
  }
  if constexpr (N % M != 0) {
    const uint32_t j = kNumIters * M + id;
    if (j < N) fn(j);
  }
}

}  // namespace fused_mqa_topk
