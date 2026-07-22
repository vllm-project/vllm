# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for direct symmetric-memory DCP collectives."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.distributed.parallel_state import GroupCoordinator

try:
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem_available = True
except ImportError:
    symm_mem = None  # type: ignore[assignment]
    symm_mem_available = False


@cache
def _symm_mem_spans_group(group: GroupCoordinator) -> bool:
    """Whether symmetric memory actually reaches every rank in `group`.

    Node co-residency is only a proxy for this, and an outdated one: it holds
    when the symmetric-memory backend is CUDA IPC, but on an NVLink-fabric rack
    (NVL72/MNNVL) the fabric carries symmetric memory across nodes too. Treating
    a multi-node group as incapable silently demotes it to the generic NCCL
    path, which costs roughly 5x the latency of the direct path at the small
    payloads decode issues.

    Probes the same way `CustomAllreduce._init_mnnvl_buffer` does -- rendezvous
    a small buffer and require a usable multicast pointer -- because only an
    actual rendezvous proves mutual reachability; `has_multicast_support` is a
    per-device query and cannot tell whether these particular ranks can reach
    each other.

    Collective: every rank in `group` must call this, and all do, because the
    conditions guarding it in `_direct_dcp_enabled` are rank-invariant. Any
    failure returns False, falling back to today's behaviour.
    """
    if not symm_mem_available:
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        device = torch.device("cuda", torch.cuda.current_device())
        if not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device.index):
            return False
        probe = symm_mem.empty(8, dtype=torch.uint8, device=device)
        probe.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(probe, group.device_group.group_name)
        spans = handle is not None and handle.multicast_ptr != 0
    except Exception as error:
        logger.debug("Direct DCP symmetric-memory probe failed: %s", error)
        return False
    logger.debug_once(
        "Direct DCP symmetric memory across %d ranks: %s",
        group.world_size,
        "available" if spans else "unavailable",
    )
    return spans


def _direct_dcp_enabled(
    group: GroupCoordinator,
    dtype: torch.dtype,
    use_direct: bool | None,
    supported_dtypes: tuple[torch.dtype, ...] | None = None,
) -> bool:
    if use_direct is not None:
        return use_direct
    return (
        symm_mem_available
        and current_platform.is_cuda()
        and (supported_dtypes is None or dtype in supported_dtypes)
        and (
            all(in_the_same_node_as(group.cpu_group, source_rank=0))
            or _symm_mem_spans_group(group)
        )
    )


class _DirectDCPWorkspace:
    """Own symmetric allocations and their per-ubatch peer pointers."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        num_ubatches: int,
    ) -> None:
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.device = torch.device(device)
        self.num_ubatches = num_ubatches
        self.epoch = torch.zeros(num_ubatches, dtype=torch.int64, device=self.device)
        self._allocations: list[tuple[torch.Tensor, object, list[torch.Tensor]]] = []

    def _allocate(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        storage = symm_mem.empty(shape, device=self.device, dtype=dtype)
        storage.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(storage, self.group.group_name)
        assert handle is not None, "DCP symmetric memory rendezvous returned None"
        handle.barrier()
        views = [
            handle.get_buffer(peer, list(shape), dtype, 0)
            for peer in range(self.world_size)
        ]
        self.device = storage.device
        peer_ptrs = torch.tensor(
            [
                [view[ubatch].data_ptr() for view in views]
                for ubatch in range(self.num_ubatches)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._allocations.append((storage, handle, views))
        return storage, peer_ptrs

    def _multicast_ptrs(self, storage: torch.Tensor) -> list[int]:
        """Per-ubatch NVLS multicast pointers for `storage`, or zeros.

        A multimem store through the returned pointer replicates the payload
        into every rank's copy of the symmetric buffer. Returns all zeros when
        the fabric or the symmetric-memory backend has no multicast support,
        in which case callers fall back to per-peer unicast stores.
        """
        disabled = [0] * self.num_ubatches
        for allocated, handle, _ in self._allocations:
            if allocated is storage:
                break
        else:
            return disabled
        try:
            from torch._C._autograd import DeviceType
            from torch._C._distributed_c10d import _SymmetricMemory

            if not _SymmetricMemory.has_multicast_support(
                DeviceType.CUDA, storage.device.index
            ):
                return disabled
            mc_base = handle.multicast_ptr
        except Exception:
            return disabled
        if not mc_base:
            return disabled
        base = storage.data_ptr()
        return [
            mc_base + (storage[ubatch].data_ptr() - base)
            for ubatch in range(self.num_ubatches)
        ]
