# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram host tables in HIP VMM host memory, shared across DP ranks by dmabuf FD.

The ROCm counterpart of `nvidia/ops/engram_host_table.py`. HIP cannot export
host allocations as FABRIC handles, so rank 0 sends POSIX FDs to its peers over
a unix socket. GPUs map these allocations with large pages, unlike the 4K pages
of a /dev/shm mapping on hosts without shmem THP, so random row lookups miss
the TLB less.
"""

import ctypes
import functools
import mmap
import os
import shutil
import socket
import tempfile
import weakref
from types import SimpleNamespace
from typing import NamedTuple

import psutil
import torch

from vllm.distributed.device_communicators.shm_broadcast import SHM_PATH
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import init_logger
from vllm.models.deepseek_v41.common.engram import describe_error, gather_errors
from vllm.platforms import current_platform
from vllm.utils.cpu_resource_utils import get_memory_node_info
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

# Tables are created in chunks, so the creator can stop before draining memory.
CHUNK_BYTES = 2 << 30
# Fraction of host memory left to other allocations.
NUMA_RESERVE_FRACTION = 0.05
# hipMemGetAllocationGranularity's recommended size for host allocations.
PAGE_BYTES = 2 << 20
SOCKET_TIMEOUT_S = 120.0

_PINNED = 1
_POSIX_FD = 1
_LOCATION_DEVICE = 1
_LOCATION_HOST_NUMA = 3
_PROT_READWRITE = 3
# Missing from the mmap module.
_PROT_NONE = 0
_MAP_FIXED = 0x10


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", ctypes.c_uint32),
    ]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


class ExportedHostTable(NamedTuple):
    """What a peer needs to map a table; the FDs travel over `socket_path`."""

    padded_bytes: int
    socket_path: str


@functools.cache
def _hip():
    hip = ctypes.CDLL("libamdhip64.so")
    ptr, size, flags = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulonglong
    out = ctypes.POINTER(ctypes.c_void_p)
    signatures = {
        "hipMemCreate": [out, size, ptr, flags],
        "hipMemExportToShareableHandle": [ptr, ptr, ctypes.c_int, flags],
        "hipMemImportFromShareableHandle": [out, ptr, ctypes.c_int],
        "hipMemRelease": [ptr],
        "hipMemAddressReserve": [out, size, size, ptr, flags],
        "hipMemAddressFree": [ptr, size],
        "hipMemMap": [ptr, size, size, ptr, flags],
        "hipMemUnmap": [ptr, size],
        "hipMemSetAccess": [ptr, size, ptr, size],
    }
    for name, argtypes in signatures.items():
        getattr(hip, name).argtypes = argtypes
        getattr(hip, name).restype = ctypes.c_int
    hip.hipGetErrorName.argtypes = [ctypes.c_int]
    hip.hipGetErrorName.restype = ctypes.c_char_p
    return hip


def _call(name: str, *args) -> None:
    if error := getattr(_hip(), name)(*args):
        raise RuntimeError(f"{name} failed: {_hip().hipGetErrorName(error).decode()}")


def _chunk_ranges(padded_bytes: int) -> list[tuple[int, int]]:
    return [
        (offset, min(CHUNK_BYTES, padded_bytes - offset))
        for offset in range(0, padded_bytes, CHUNK_BYTES)
    ]


def _gpu_numa_node() -> int:
    nodes = current_platform.get_all_device_numa_nodes()
    if nodes is None:
        raise RuntimeError("the GPU's NUMA node is unknown")
    return nodes[torch.accelerator.current_device_index()]


def _create_chunk(length: int, numa_node: int) -> ctypes.c_void_p:
    prop = _AllocationProp(
        type=_PINNED,
        requestedHandleTypes=_POSIX_FD,
        location=_Location(_LOCATION_HOST_NUMA, numa_node),
    )
    handle = ctypes.c_void_p()
    _call("hipMemCreate", ctypes.byref(handle), length, ctypes.byref(prop), 0)
    return handle


def _export_fd(handle: ctypes.c_void_p) -> int:
    fd = ctypes.c_int(-1)
    _call("hipMemExportToShareableHandle", ctypes.byref(fd), handle, _POSIX_FD, 0)
    return fd.value


def _release(handles: list) -> None:
    for handle in handles:
        _hip().hipMemRelease(handle)


def _close(fds: list[int]) -> None:
    for fd in fds:
        os.close(fd)


@functools.cache
def host_memory_supported() -> bool:
    """Whether HIP host allocations export POSIX FDs that the CPU can map."""
    try:
        handle = _create_chunk(PAGE_BYTES, _gpu_numa_node())
        try:
            fd = _export_fd(handle)
            try:
                mmap.mmap(fd, PAGE_BYTES).close()
            finally:
                os.close(fd)
        finally:
            _hip().hipMemRelease(handle)
    except Exception as exc:
        logger.info_once(
            "HIP VMM host memory is unavailable (%s); Engram DP replicas share "
            "tables through /dev/shm.",
            exc,
        )
        return False
    return True


def _check_reserve(numa_node: int, created: int, padded_bytes: int) -> None:
    """HIP may place host memory on another node, so watch the whole host too."""
    node = get_memory_node_info(numa_node)
    host = psutil.virtual_memory()
    for name, available, total in (
        (f"NUMA node {numa_node}", node.available_memory, node.total_memory),
        ("Host memory", host.available, host.total),
    ):
        reserve = total * NUMA_RESERVE_FRACTION
        if available < reserve:
            raise MemoryError(
                f"{name} fell below its {reserve / 2**30:.1f} GiB reserve after "
                f"{created / 2**30:.1f} of {padded_bytes / 2**30:.1f} GiB"
            )


def _create_table(num_bytes: int) -> tuple[list, list[int], int]:
    """Allocate a table and export one FD per chunk."""
    numa_node = _gpu_numa_node()
    padded_bytes = round_up(num_bytes, PAGE_BYTES)
    handles: list = []
    fds: list[int] = []
    try:
        for offset, length in _chunk_ranges(padded_bytes):
            handles.append(_create_chunk(length, numa_node))
            _check_reserve(numa_node, offset + length, padded_bytes)
        for handle in handles:
            fds.append(_export_fd(handle))
    except BaseException:
        _close(fds)
        _release(handles)
        raise
    return handles, fds, padded_bytes


def _receive_fds(path: str, count: int) -> list[int]:
    fds: list[int] = []
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(SOCKET_TIMEOUT_S)
        connection.connect(path)
        try:
            for _ in range(count):
                _, received, _, _ = socket.recv_fds(connection, 1, 1)
                if not received:
                    raise ConnectionError("Engram rank 0 closed the FD socket")
                fds.extend(received)
        except BaseException:
            _close(fds)
            raise
    return fds


def _unmap(
    host_address: int | None,
    gpu_address: int,
    size: int,
    mapped: list[tuple[int, int]],
) -> None:
    for offset, length in mapped:
        _hip().hipMemUnmap(gpu_address + offset, length)
    _hip().hipMemAddressFree(gpu_address, size)
    if host_address is not None:
        libc = ctypes.CDLL(None)
        libc.munmap(ctypes.c_void_p(host_address), ctypes.c_size_t(size))


def _map_table(
    table: ExportedHostTable, num_bytes: int, fds: list[int], handles: list
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a table on the GPU and the CPU, importing its FDs unless given handles.

    Imported handles are appended to `handles`; the caller releases them and
    closes the FDs, since the mappings keep the allocations alive. HIP grants
    no CPU access to imported host allocations, so the CPU maps the dmabuf FDs
    directly. Returns CPU and GPU uint8 views; either keeps the mappings alive.
    """
    device = torch.accelerator.current_device_index()
    size = table.padded_bytes
    ranges = _chunk_ranges(size)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    host_address: int | None = None
    gpu_address: int | None = None
    mapped: list[tuple[int, int]] = []
    try:
        if not handles:
            for fd in fds:
                handle = ctypes.c_void_p()
                _call(
                    "hipMemImportFromShareableHandle",
                    ctypes.byref(handle),
                    ctypes.c_void_p(fd),
                    _POSIX_FD,
                )
                handles.append(handle)
        address = ctypes.c_void_p()
        _call("hipMemAddressReserve", ctypes.byref(address), size, PAGE_BYTES, None, 0)
        gpu_address = address.value
        assert gpu_address is not None
        for (offset, length), handle in zip(ranges, handles):
            _call("hipMemMap", gpu_address + offset, length, 0, handle, 0)
            mapped.append((offset, length))
        access = _AccessDesc(_Location(_LOCATION_DEVICE, device), _PROT_READWRITE)
        _call("hipMemSetAccess", gpu_address, size, ctypes.byref(access), 1)

        host_address = libc.mmap(
            None, size, _PROT_NONE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0
        )
        if host_address == ctypes.c_void_p(-1).value:
            host_address = None
            raise OSError(ctypes.get_errno(), "mmap failed")
        for (offset, length), fd in zip(ranges, fds):
            chunk = libc.mmap(
                host_address + offset,
                length,
                mmap.PROT_READ | mmap.PROT_WRITE,
                mmap.MAP_SHARED | _MAP_FIXED,
                fd,
                0,
            )
            if chunk != host_address + offset:
                raise OSError(ctypes.get_errno(), "mmap of an Engram FD failed")
    except BaseException:
        if gpu_address is not None:
            _unmap(host_address, gpu_address, size, mapped)
        raise

    assert host_address is not None and gpu_address is not None
    host_buffer = (ctypes.c_uint8 * num_bytes).from_address(host_address)
    finalizer = weakref.finalize(
        host_buffer, _unmap, host_address, gpu_address, size, mapped
    )
    finalizer.atexit = False  # type: ignore[misc]
    # Torch retains this object, so the GPU view keeps the buffer alive too.
    # Name the device: an index-less "cuda" makes torch copy the table.
    gpu_view = torch.as_tensor(
        SimpleNamespace(
            __cuda_array_interface__={
                "shape": (num_bytes,),
                "typestr": "|u1",
                "data": (gpu_address, False),
                "version": 3,
            },
            owner=host_buffer,
        ),
        device=torch.device("cuda", device),
    )
    return torch.frombuffer(host_buffer, dtype=torch.uint8), gpu_view


def share_host_table(
    num_bytes: int, group: GroupCoordinator
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Map one table on every rank of a node-local group, or None on all of them.

    Rank 0 creates the table and sends its FDs to the others over a unix socket
    in a private directory. Returns CPU and GPU uint8 views of the table.
    """
    handles: list = []
    fds: list[int] = []
    table = error = server = directory = None
    if group.rank_in_group == 0 and host_memory_supported():
        try:
            handles, fds, padded_bytes = _create_table(num_bytes)
            # The /dev/shm fallback needs SHM_PATH shared too, so both reach
            # the same ranks.
            directory = tempfile.mkdtemp(prefix="vllm_engram_", dir=SHM_PATH)
            server = socket.socket(socket.AF_UNIX)
            server.settimeout(SOCKET_TIMEOUT_S)
            server.bind(path := os.path.join(directory, "fds"))
            server.listen(group.world_size)
            table = ExportedHostTable(padded_bytes, path)
        except Exception as exc:
            error = describe_error(exc)
    table, error = group.broadcast_object((table, error))
    views = None
    try:
        if table is not None:
            if server is not None:
                for _ in range(group.world_size - 1):
                    connection, _ = server.accept()
                    with connection:
                        for fd in fds:
                            socket.send_fds(connection, [b"\0"], [fd])
            else:
                count = len(_chunk_ranges(table.padded_bytes))
                fds = _receive_fds(table.socket_path, count)
            views = _map_table(table, num_bytes, fds, handles)
    except Exception as exc:
        error = describe_error(exc)
    finally:
        _close(fds)
        _release(handles)
        if server is not None:
            server.close()
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
    if table is not None and not (error := gather_errors(group, error)):
        return views
    if error and group.rank_in_group == 0:
        logger.warning(
            "Engram host table cannot use HIP VMM host memory (%s); falling back "
            "to /dev/shm, whose small GPU pages make lookups slower.",
            error,
        )
    return None
