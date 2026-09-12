# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime-compiled HIP gather kernels for W4A16 cold-expert offload (arch A).

Two HIP-graph-capturable kernels (no host sync, fixed launch shapes):

  ec_plan : one block. Dedup the step's needed experts, freshen LRU on hits,
            pick a clock-LRU victim on misses (current-step protection is
            automatic — hits and just-loaded slots are freshened to the max
            clock this step, so argmin never selects them while an older slot
            exists), update slot_of_expert / expert_of_slot, emit an
            (expert -> slot) copy plan padded to max_m.
  ec_copy : one block per plan entry. Valid entries zero-copy the expert's row
            block from device-mapped pinned host master into its GPU slot;
            padded entries are skipped by a SIMT branch.

Measured on gfx1100: in-kernel zero-copy reads from device-mapped pinned host
run at ~26 GB/s (== DMA), and the all-hit path is ~0.007 ms/layer.

Compiled at import via torch cpp_extension so we do NOT rebuild vLLM's _rocm_C.
"""

from __future__ import annotations

import functools

import torch

_CPP_SRC = r"""
#include <torch/extension.h>
int64_t ec_device_ptr(torch::Tensor pinned);
void ec_plan(torch::Tensor topk_ids, torch::Tensor slot_of_expert,
             torch::Tensor expert_of_slot, torch::Tensor lru_last,
             torch::Tensor clk, int64_t cache_size, torch::Tensor plan_expert,
             torch::Tensor plan_slot);
void ec_copy(torch::Tensor plan_expert, torch::Tensor plan_slot,
             int64_t master_devptr, torch::Tensor cache, int64_t row_u4);
"""

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__global__ void ec_plan_k(const int* __restrict__ topk_ids, int n_sel,
                          int* __restrict__ slot_of_expert,
                          int* __restrict__ expert_of_slot,
                          long long* __restrict__ lru_last,
                          long long* __restrict__ clk, int cache_size, int max_m,
                          int* __restrict__ plan_expert,
                          int* __restrict__ plan_slot) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  long long now = *clk;
  int pc = 0;
  for (int i = 0; i < n_sel; i++) {
    int e = topk_ids[i];
    if (e < 0) continue;
    int s = slot_of_expert[e];
    if (s >= 0) { lru_last[s] = now; continue; }  // hit: freshen
    bool dup = false;
    for (int j = 0; j < pc; j++) if (plan_expert[j] == e) { dup = true; break; }
    if (dup) continue;
    int victim = 0; long long best = lru_last[0];
    for (int c = 1; c < cache_size; c++)
      if (lru_last[c] < best) { best = lru_last[c]; victim = c; }
    int ve = expert_of_slot[victim];
    if (ve >= 0) slot_of_expert[ve] = -1;
    expert_of_slot[victim] = e;
    slot_of_expert[e] = victim;
    lru_last[victim] = now;
    plan_expert[pc] = e; plan_slot[pc] = victim; pc++;
  }
  for (int j = pc; j < max_m; j++) { plan_expert[j] = -1; plan_slot[j] = -1; }
  *clk = now + 1;
}

__global__ void ec_copy_k(const int* __restrict__ plan_expert,
                          const int* __restrict__ plan_slot,
                          const int4* __restrict__ master,
                          int4* __restrict__ cache, int row_u4) {
  int m = blockIdx.x;
  int e = plan_expert[m];
  if (e < 0) return;  // padded: skip
  int s = plan_slot[m];
  const int4* src = master + (long long)e * row_u4;
  int4* dst = cache + (long long)s * row_u4;
  for (int i = threadIdx.x; i < row_u4; i += blockDim.x) dst[i] = src[i];
}

// Resolve a device-accessible pointer for a pinned host tensor (once, at setup).
int64_t ec_device_ptr(torch::Tensor pinned) {
  void* dptr = nullptr;
  cudaHostGetDevicePointer(&dptr, pinned.data_ptr(), 0);
  return reinterpret_cast<int64_t>(dptr);
}

// Plan the step's residency (dedup + evict + update maps + emit copy plan).
// Fully on-device -> HIP-graph capturable. Call ONCE per MoE step; the plan is
// shared by all weight planes.
void ec_plan(torch::Tensor topk_ids, torch::Tensor slot_of_expert,
             torch::Tensor expert_of_slot, torch::Tensor lru_last,
             torch::Tensor clk, int64_t cache_size, torch::Tensor plan_expert,
             torch::Tensor plan_slot) {
  int n_sel = topk_ids.numel();
  int max_m = plan_expert.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  ec_plan_k<<<1, 64, 0, stream>>>(
      topk_ids.data_ptr<int>(), n_sel, slot_of_expert.data_ptr<int>(),
      expert_of_slot.data_ptr<int>(),
      reinterpret_cast<long long*>(lru_last.data_ptr<int64_t>()),
      reinterpret_cast<long long*>(clk.data_ptr<int64_t>()), (int)cache_size,
      max_m, plan_expert.data_ptr<int>(), plan_slot.data_ptr<int>());
}

// Zero-copy the planned experts' rows for ONE weight plane (contiguous [C,..]).
// row_u4 = bytes_per_expert_row / 16. Call once per plane with the same plan.
void ec_copy(torch::Tensor plan_expert, torch::Tensor plan_slot,
             int64_t master_devptr, torch::Tensor cache, int64_t row_u4) {
  int max_m = plan_expert.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  ec_copy_k<<<max_m, 256, 0, stream>>>(
      plan_expert.data_ptr<int>(), plan_slot.data_ptr<int>(),
      reinterpret_cast<const int4*>(master_devptr),
      reinterpret_cast<int4*>(cache.data_ptr()), (int)row_u4);
}
"""


@functools.lru_cache(maxsize=1)
def _mod():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="vllm_expert_gather",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["ec_device_ptr", "ec_plan", "ec_copy"],
        with_cuda=True,
        verbose=False,
    )


def device_ptr(pinned: torch.Tensor) -> int:
    return _mod().ec_device_ptr(pinned)


def plan(
    topk_ids: torch.Tensor,
    slot_of_expert: torch.Tensor,
    expert_of_slot: torch.Tensor,
    lru_last: torch.Tensor,
    clk: torch.Tensor,
    cache_size: int,
    plan_expert: torch.Tensor,
    plan_slot: torch.Tensor,
) -> None:
    _mod().ec_plan(
        topk_ids, slot_of_expert, expert_of_slot, lru_last, clk,
        cache_size, plan_expert, plan_slot,
    )


def copy(
    plan_expert: torch.Tensor,
    plan_slot: torch.Tensor,
    master_devptr: int,
    cache: torch.Tensor,
    row_u4: int,
) -> None:
    _mod().ec_copy(plan_expert, plan_slot, master_devptr, cache, row_u4)


class ExpertOffloadCache:
    """Per-layer W4A16 cold-expert cache: pinned master of all E experts +
    fixed GPU cache of C slots, backed by the plan/copy HIP kernels.

    Each per-expert weight plane is stored as a contiguous, uint4-aligned pinned
    master ``[E, row_ints]`` (device-mapped) and a GPU cache ``[C, row_ints]``.
    :meth:`ensure` plans residency for a step's ``topk_ids`` (expert space) and
    returns ``topk_ids`` remapped to slot space ``[0, C)`` for the GEMM.
    """

    def __init__(self, num_experts: int, cache_size: int, max_sel: int,
                 device: torch.device):
        self.E = num_experts
        self.C = min(cache_size, num_experts)
        self.device = device
        self.planes: dict[str, dict] = {}  # name -> {master, cache, dptr, row_u4}
        self.slot_of_expert = torch.full((self.E,), -1, dtype=torch.int32,
                                         device=device)
        self.expert_of_slot = torch.full((self.C,), -1, dtype=torch.int32,
                                         device=device)
        self.lru_last = torch.zeros(self.C, dtype=torch.int64, device=device)
        self.clk = torch.ones(1, dtype=torch.int64, device=device)
        self.plan_e = torch.full((max_sel,), -1, dtype=torch.int32, device=device)
        self.plan_s = torch.full((max_sel,), -1, dtype=torch.int32, device=device)

    def add_plane(self, name: str, master_pinned: torch.Tensor,
                  cache_gpu: torch.Tensor) -> None:
        """master_pinned [E, *shape] (pinned), cache_gpu [C, *shape] (device).
        Both must be contiguous and uint4 (16B) aligned per expert row."""
        row_bytes = master_pinned[0].numel() * master_pinned.element_size()
        assert row_bytes % 16 == 0, f"{name} row {row_bytes} not 16B-aligned"
        self.planes[name] = {
            "master": master_pinned,
            "cache": cache_gpu,
            "dptr": device_ptr(master_pinned),
            "row_u4": row_bytes // 16,
        }

    def cache_tensor(self, name: str) -> torch.Tensor:
        return self.planes[name]["cache"]

    def warm(self, expert_ids: list[int]) -> None:
        """Preload experts into slots (host-side, one-time at load)."""
        for slot, e in enumerate(expert_ids[: self.C]):
            self.slot_of_expert[e] = slot
            self.expert_of_slot[slot] = e
            self.lru_last[slot] = 0
            for p in self.planes.values():
                p["cache"][slot].copy_(p["master"][e], non_blocking=True)

    def ensure(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Make the step's experts resident; return slot-space topk_ids."""
        flat = topk_ids.reshape(-1).to(torch.int32)
        plan(flat, self.slot_of_expert, self.expert_of_slot, self.lru_last,
             self.clk, self.C, self.plan_e, self.plan_s)
        for p in self.planes.values():
            copy(self.plan_e, self.plan_s, p["dptr"], p["cache"], p["row_u4"])
        return self.slot_of_expert[topk_ids.long()].to(topk_ids.dtype)
