# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER IPC all-gather for small per-layer payloads on ROCm.

vLLM's TP communicator keeps its AITER `CustomAllreduce` instance (when
`VLLM_ROCM_USE_AITER_CUSTOM_AR` is set) with IPC buffers already exported to
the peers. Its `all_gather` stays on the RCCL collective for TP, so callers
that gather small payloads outside the TP forward pass -- a CP candidate
exchange, a DCP index-key gather -- borrow that instance directly and fall
back to the group's own collective when it cannot take the payload.

AITER's IPC all-gather moves 16-byte packs, and its own `should_custom_ag`
decides whether a given payload fits that form and the IPC buffer. Payloads
that do not are left to the fallback. Callers keep the packed row a whole
number of packs when they choose the layout; this module does not pad.
"""

from functools import cache

import torch

from vllm.distributed.parallel_state import GroupCoordinator, get_tp_group


@cache
def _holds_tp_ranks(group: GroupCoordinator) -> bool:
    """Whether `group` is the TP group; the borrowed comm only covers TP ranks.

    Group membership cannot change after init, so this is resolved once per
    group rather than on every layer's gather.
    """
    tp_group = get_tp_group()
    return group.world_size == tp_group.world_size and list(group.ranks) == list(
        tp_group.ranks
    )


def aiter_all_gather(x: torch.Tensor, group: GroupCoordinator) -> torch.Tensor | None:
    """Gather contiguous `x` over `group` as `[world, *x.shape]`, or None.

    The payload is viewed as int32 so AITER's pybind layer (which has no
    float64 after it relabels integers as floats) accepts it -- int64 keys,
    fp8 index bytes and bf16 queries alike. Returns None when the TP AITER
    instance is missing or disabled, when `group` is not the TP group (the
    borrowed comm only covers TP ranks), or when AITER cannot take the payload;
    the caller then falls back to `group.all_gather`.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    comm = rocm_aiter_ops.get_aiter_allreduce()
    if comm is None or comm.disabled:
        return None
    if x.numel() == 0 or not x.is_contiguous():
        return None
    if (x.shape[-1] * x.element_size()) % 4:
        return None
    if not _holds_tp_ranks(group):
        return None
    packed = x.view(torch.int32)
    if not comm.should_custom_ag(packed):
        return None
    gathered = comm.custom_all_gather(packed, dim=0)
    if gathered is None:
        return None
    return gathered.view(x.dtype).view(group.world_size, *x.shape)


def all_gather(
    x: torch.Tensor, group: GroupCoordinator, *, use_aiter: bool = True
) -> torch.Tensor:
    """All-gather contiguous `x` over `group` to `[world, *x.shape]`.

    Tries AITER's IPC all-gather first unless `use_aiter` is off, and falls back
    to the group's own collective. Every rank must make the same choice: the
    decision reads only the payload's shape and dtype, the group size and the
    topology, never data or allocator state.
    """
    if use_aiter:
        gathered = aiter_all_gather(x, group)
        if gathered is not None:
            return gathered
    return group.all_gather(x, dim=0).view(group.world_size, *x.shape)
