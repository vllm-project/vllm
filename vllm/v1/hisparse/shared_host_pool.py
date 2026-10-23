# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned host pool allocated once and mapped by every local TP rank."""

from __future__ import annotations

import ctypes
import json
import os
import socket
import time
from collections.abc import Callable

import torch
import torch.distributed as dist
from cuda.bindings import driver

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_POSIX_FD = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR


def _check(result):
    err, *values = result
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return values[0] if len(values) == 1 else values


def _location(kind: driver.CUmemLocationType, id: int = 0) -> driver.CUmemLocation:
    location = driver.CUmemLocation()
    location.type = kind
    location.id = id
    return location


def _host_prop(numa_node: int | None) -> driver.CUmemAllocationProp:
    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = _POSIX_FD
    prop.location = (
        _location(driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT)
        if numa_node is None
        else _location(
            driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, numa_node
        )
    )
    return prop


class HiSparseSharedHostPool:
    """One pinned host allocation per NUMA node of the local TP GPUs.

    TP rank 0 creates the physical memory and passes POSIX fds to the other
    ranks over a Unix socket. Every rank maps the memory at its own virtual
    address and grants access to its GPU and the CPU. The kernel pins the pages
    once, so setup and teardown do not scale with the number of ranks.
    """

    def __init__(
        self,
        name: str,
        size: int,
        tp_group: GroupCoordinator,
        creator_memory_check: Callable[[int], None],
    ) -> None:
        self.size = size
        self._handles: list[int] = []
        self._ptr = 0
        self._reserved = 0
        self._mapped = 0
        self._base: torch.Tensor | None = None
        start = time.perf_counter()

        numa_node = current_platform.get_device_numa_node(
            torch.accelerator.current_device_index()
        )
        numa_nodes: list[int | None] = [None] * tp_group.world_size
        dist.all_gather_object(numa_nodes, numa_node, group=tp_group.cpu_group)
        nodes: list[int | None] = [
            *sorted({n for n in numa_nodes if n is not None})
        ] or [None]

        address = f"\0vllm_{name}".encode()
        try:
            if tp_group.rank_in_group == 0:
                creator_memory_check(size)
                chunk_sizes, fds = self._create(nodes)
                server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                server.bind(address)
                server.listen(tp_group.world_size)
                tp_group.barrier()
                message = json.dumps(chunk_sizes).encode()
                try:
                    for _ in range(tp_group.world_size - 1):
                        conn, _ = server.accept()
                        with conn:
                            socket.send_fds(conn, [message], fds)
                finally:
                    server.close()
                    for fd in fds:
                        os.close(fd)
            else:
                tp_group.barrier()
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                    conn.connect(address)
                    message, fds, _, _ = socket.recv_fds(conn, 4096, len(nodes))
                chunk_sizes = json.loads(message)
                for fd in fds:
                    try:
                        self._handles.append(
                            _check(driver.cuMemImportFromShareableHandle(fd, _POSIX_FD))
                        )
                    finally:
                        os.close(fd)
            self._map(chunk_sizes)
        except Exception:
            self.cleanup()
            raise
        tp_group.barrier()
        logger.info(
            "HiSparse: mapped %.1f GiB shared host pool across NUMA nodes %s in %.1fs.",
            size / 2**30,
            nodes,
            time.perf_counter() - start,
        )

    def _create(self, nodes: list[int | None]) -> tuple[list[int], list[int]]:
        granularity = _check(
            driver.cuMemGetAllocationGranularity(
                _host_prop(nodes[0]),
                driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )
        per_node = -(-self.size // len(nodes) // granularity) * granularity
        chunk_sizes, fds, remaining = [], [], self.size
        for node in nodes:
            chunk_size = -(-min(per_node, remaining) // granularity) * granularity
            remaining -= min(per_node, remaining)
            handle = _check(driver.cuMemCreate(chunk_size, _host_prop(node), 0))
            self._handles.append(handle)
            fds.append(
                _check(driver.cuMemExportToShareableHandle(handle, _POSIX_FD, 0))
            )
            chunk_sizes.append(chunk_size)
        return chunk_sizes, fds

    def _map(self, chunk_sizes: list[int]) -> None:
        reserved = sum(chunk_sizes)
        self._ptr = int(_check(driver.cuMemAddressReserve(reserved, 0, 0, 0)))
        self._reserved = reserved
        offset = 0
        for handle, chunk_size in zip(self._handles, chunk_sizes, strict=True):
            _check(driver.cuMemMap(self._ptr + offset, chunk_size, 0, handle, 0))
            offset += chunk_size
            self._mapped = offset
        rw = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        descs = []
        for location in (
            _location(
                driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
                torch.accelerator.current_device_index(),
            ),
            _location(driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST),
        ):
            desc = driver.CUmemAccessDesc()
            desc.location = location
            desc.flags = rw
            descs.append(desc)
        _check(driver.cuMemSetAccess(self._ptr, reserved, descs, len(descs)))
        buffer = (ctypes.c_int8 * self.size).from_address(self._ptr)
        self._base = torch.frombuffer(buffer, dtype=torch.int8)

    @property
    def base_tensor(self) -> torch.Tensor:
        if self._base is None:
            raise RuntimeError("HiSparse shared host pool has been released.")
        return self._base

    def views(self, tensor_sizes: list[int]) -> list[torch.Tensor]:
        offsets = [0]
        for tensor_size in tensor_sizes:
            offsets.append(offsets[-1] + tensor_size)
        assert offsets[-1] <= self.size
        return [self.base_tensor[start:end] for start, end in zip(offsets, offsets[1:])]

    def cleanup(self) -> None:
        start = time.perf_counter()
        self._base = None
        if self._mapped:
            _check(driver.cuMemUnmap(self._ptr, self._mapped))
            self._mapped = 0
        while self._handles:
            _check(driver.cuMemRelease(self._handles.pop()))
        if self._reserved:
            _check(driver.cuMemAddressFree(self._ptr, self._reserved))
            self._reserved = 0
            logger.info(
                "HiSparse: released %.1f GiB shared host pool in %.1fs.",
                self.size / 2**30,
                time.perf_counter() - start,
            )
