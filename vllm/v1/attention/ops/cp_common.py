# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared symmetric-memory infrastructure for context-parallel attention."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.device_communicators.nvlink_fabric import (
    SymmetricMemoryTopology,
    get_symmetric_memory_topology,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)

try:
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem_available = True
except ImportError:
    symm_mem = None  # type: ignore[assignment]
    symm_mem_available = False


@functools.cache
def _symm_mem_spans_group(group: GroupCoordinator) -> bool:
    """Probe whether the group has NVLS symmetric memory."""
    if not symm_mem_available:
        return False
    topology = get_symmetric_memory_topology(group.cpu_group)
    if topology is SymmetricMemoryTopology.UNSUPPORTED:
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        device = torch.device("cuda", torch.accelerator.current_device_index())
        if not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device.index):
            return False
        probe = symm_mem.empty(8, dtype=torch.uint8, device=device)
        probe.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(probe, group.device_group.group_name)
        spans = handle is not None and handle.multicast_ptr != 0
    except Exception as error:
        logger.debug("Direct CP symmetric-memory probe failed: %s", error)
        return False
    logger.debug_once(
        "Direct CP symmetric memory across %d ranks: %s",
        group.world_size,
        "available" if spans else "unavailable",
    )
    return spans


def direct_cp_enabled(
    group: GroupCoordinator,
    dtype: torch.dtype,
    use_direct: bool | None,
    supported_dtypes: tuple[torch.dtype, ...] | None = None,
) -> bool:
    if use_direct is False:
        return False
    if use_direct is None and (
        not symm_mem_available
        or not current_platform.is_cuda()
        or (supported_dtypes is not None and dtype not in supported_dtypes)
    ):
        return False

    topology = get_symmetric_memory_topology(group.cpu_group)
    if topology is SymmetricMemoryTopology.UNSUPPORTED:
        logger.warning_once(
            "Direct CP is disabled because the ranks do not share a "
            "symmetric-memory fabric."
        )
        return False
    return use_direct is True or (
        topology is SymmetricMemoryTopology.INTRA_NODE or _symm_mem_spans_group(group)
    )


def direct_cp_multicast_enabled(
    group: GroupCoordinator,
    dtype: torch.dtype,
    use_direct: bool | None,
    supported_dtypes: tuple[torch.dtype, ...] | None = None,
) -> bool:
    return direct_cp_enabled(
        group, dtype, use_direct, supported_dtypes
    ) and _symm_mem_spans_group(group)


class DirectCPWorkspace:
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
        self._allocations: list[tuple[torch.Tensor, Any, list[torch.Tensor]]] = []

    def _allocate(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        storage = symm_mem.empty(shape, device=self.device, dtype=dtype)
        storage.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(storage, self.group.group_name)
        assert handle is not None, "CP symmetric memory rendezvous returned None"
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
            multicast_base = handle.multicast_ptr
        except Exception:
            return disabled
        if not multicast_base:
            return disabled
        storage_base = storage.data_ptr()
        return [
            multicast_base + (storage[ubatch].data_ptr() - storage_base)
            for ubatch in range(self.num_ubatches)
        ]
