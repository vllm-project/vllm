# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned host pool allocated once and mapped by every local TP rank."""

from __future__ import annotations

import ctypes
import json
import os
import socket
import struct
import sys
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from cuda.bindings import driver

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_POSIX_FD = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
_REQUIRED_ATTRIBUTES = (
    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
)


def _check(result: tuple[Any, ...]) -> Any:
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


def _check_device_support(device: int) -> None:
    _check(driver.cuInit(0))
    cu_device = _check(driver.cuDeviceGet(device))
    for attribute in _REQUIRED_ATTRIBUTES:
        if not _check(driver.cuDeviceGetAttribute(attribute, cu_device)):
            raise RuntimeError(
                f"HiSparse shares one host pool across TP ranks, which needs "
                f"{attribute.name}; GPU {device} does not support it."
            )


def _peer_pid(conn: socket.socket) -> int:
    if sys.platform != "linux":
        raise RuntimeError("HiSparse's shared host pool requires Linux.")
    creds = conn.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
    )
    return struct.unpack("3i", creds)[0]


class HiSparseSharedHostPool:
    """One pinned host allocation per NUMA node of the local TP GPUs.

    TP rank 0 creates the memory and passes POSIX fds to the other ranks over a
    Unix socket. Every rank maps it and grants access to its GPU and the CPU, so
    the pages are pinned once instead of once per rank.
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

        device = torch.accelerator.current_device_index()
        _check_device_support(device)
        rank_info: list[tuple[int, int | None]] = [(0, None)] * tp_group.world_size
        dist.all_gather_object(
            rank_info,
            (os.getpid(), current_platform.get_device_numa_node(device)),
            group=tp_group.cpu_group,
        )
        known = sorted({node for _, node in rank_info if node is not None})
        nodes: list[int | None] = [*known] or [None]

        address = f"\0vllm_{name}".encode()
        try:
            if tp_group.rank_in_group == 0:
                creator_memory_check(size)
                chunk_sizes = self._create(nodes)
                self._serve_fds(
                    address, chunk_sizes, {pid for pid, _ in rank_info[1:]}, tp_group
                )
            else:
                tp_group.barrier()
                chunk_sizes = self._receive_fds(address, len(nodes))
            self._map(chunk_sizes, device)
        except Exception:
            self.cleanup()
            raise
        tp_group.barrier()
        logger.info(
            "HiSparse: mapped %.1f GiB shared host pool on NUMA nodes %s in %.1fs.",
            size / 2**30,
            nodes,
            time.perf_counter() - start,
        )

    def _create(self, nodes: list[int | None]) -> list[int]:
        granularity = _check(
            driver.cuMemGetAllocationGranularity(
                _host_prop(nodes[0]),
                driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            )
        )
        per_node = -(-self.size // len(nodes) // granularity) * granularity
        chunk_sizes, remaining = [], self.size
        for node in nodes:
            chunk_size = -(-min(per_node, remaining) // granularity) * granularity
            remaining -= min(per_node, remaining)
            self._handles.append(
                _check(driver.cuMemCreate(chunk_size, _host_prop(node), 0))
            )
            chunk_sizes.append(chunk_size)
        return chunk_sizes

    def _serve_fds(
        self,
        address: bytes,
        chunk_sizes: list[int],
        peer_pids: set[int],
        tp_group: GroupCoordinator,
    ) -> None:
        fds = [
            _check(driver.cuMemExportToShareableHandle(handle, _POSIX_FD, 0))
            for handle in self._handles
        ]
        message = json.dumps(chunk_sizes).encode()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(address)
                server.listen(len(peer_pids))
                tp_group.barrier()
                while peer_pids:
                    conn, _ = server.accept()
                    with conn:
                        pid = _peer_pid(conn)
                        if pid not in peer_pids:
                            logger.warning(
                                "HiSparse: refused host pool fds to pid %d.", pid
                            )
                            continue
                        socket.send_fds(conn, [message], fds)
                        peer_pids.remove(pid)
        finally:
            for fd in fds:
                os.close(fd)

    def _receive_fds(self, address: bytes, num_chunks: int) -> list[int]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.connect(address)
            message, fds, _, _ = socket.recv_fds(conn, 4096, num_chunks)
        try:
            for fd in fds:
                self._handles.append(
                    _check(driver.cuMemImportFromShareableHandle(fd, _POSIX_FD))
                )
        finally:
            for fd in fds:
                os.close(fd)
        return json.loads(message)

    def _map(self, chunk_sizes: list[int], device: int) -> None:
        reserved = sum(chunk_sizes)
        self._ptr = int(_check(driver.cuMemAddressReserve(reserved, 0, 0, 0)))
        self._reserved = reserved
        for handle, chunk_size in zip(self._handles, chunk_sizes, strict=True):
            _check(driver.cuMemMap(self._ptr + self._mapped, chunk_size, 0, handle, 0))
            self._mapped += chunk_size
        descs = []
        for location in (
            _location(driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, device),
            _location(driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST),
        ):
            desc = driver.CUmemAccessDesc()
            desc.location = location
            desc.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            descs.append(desc)
        _check(driver.cuMemSetAccess(self._ptr, reserved, descs, len(descs)))
        buffer = (ctypes.c_int8 * self.size).from_address(self._ptr)
        self._base = torch.frombuffer(buffer, dtype=torch.int8)

    @property
    def base_tensor(self) -> torch.Tensor:
        if self._base is None:
            raise RuntimeError("HiSparse shared host pool has been released.")
        return self._base

    def views(self, sizes: list[int]) -> list[torch.Tensor]:
        assert sum(sizes) <= self.size
        return list(self.base_tensor[: sum(sizes)].split(sizes))

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
