# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram host tables in host-NUMA cuMem, shared across DP ranks by FABRIC handle.

GPUs map cuMem host memory with 2M pages instead of the 64K pages of a registered
/dev/shm mapping, so random row lookups miss the TLB far less.
"""

import ctypes
import weakref
from functools import cache
from types import SimpleNamespace
from typing import NamedTuple

import torch
from cuda.bindings import driver as cu

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import init_logger
from vllm.models.deepseek_v41.common.engram import describe_error, gather_errors
from vllm.utils.cpu_resource_utils import get_memory_node_info
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

FABRIC_HANDLE = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
# Tables are created in chunks, so the creator can stop before draining its node.
CHUNK_BYTES = 2 << 30
# Fraction of the NUMA node left to other allocations.
NUMA_RESERVE_FRACTION = 0.05


class ExportedHostTable(NamedTuple):
    """What a peer needs to map a table."""

    numa_node: int
    padded_bytes: int
    fabric_handles: list[bytes]


def _chunk_ranges(padded_bytes: int) -> list[tuple[int, int]]:
    return [
        (offset, min(CHUNK_BYTES, padded_bytes - offset))
        for offset in range(0, padded_bytes, CHUNK_BYTES)
    ]


def _call_driver(fn, *args):
    """Call a CUDA driver API and return its result, raising on failure."""
    error, *result = fn(*args)
    if error != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{fn.__name__} failed: {error.name}")
    return result[0] if result else None


def _gpu_numa_node() -> int:
    return _call_driver(
        cu.cuDeviceGetAttribute,
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
        torch.accelerator.current_device_index(),
    )


def _host_allocation_prop(numa_node: int) -> cu.CUmemAllocationProp:
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    prop.location.id = numa_node
    prop.requestedHandleTypes = FABRIC_HANDLE
    return prop


def _allocation_granularity(prop: cu.CUmemAllocationProp) -> int:
    return _call_driver(cu.cuMemGetAllocationGranularity, prop, 0)


def _export_fabric_handle(handle) -> bytes:
    exported = _call_driver(cu.cuMemExportToShareableHandle, handle, FABRIC_HANDLE, 0)
    return bytes(exported.data)


def _release(handles: list) -> None:
    for handle in handles:
        cu.cuMemRelease(handle)


def _read_write_access(location_type, location_id: int) -> cu.CUmemAccessDesc:
    access = cu.CUmemAccessDesc()
    access.location.type = location_type
    access.location.id = location_id
    access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return access


def _unmap(address: int, size: int, ranges: list[tuple[int, int]]) -> None:
    for offset, length in ranges:
        cu.cuMemUnmap(address + offset, length)
    cu.cuMemAddressFree(address, size)


@cache
def fabric_host_memory_supported() -> bool:
    """Whether host-NUMA cuMem allocations can export FABRIC handles here."""
    try:
        prop = _host_allocation_prop(_gpu_numa_node())
        handle = _call_driver(cu.cuMemCreate, _allocation_granularity(prop), prop, 0)
        try:
            _export_fabric_handle(handle)
        finally:
            cu.cuMemRelease(handle)
    except Exception as exc:
        logger.info_once(
            "Host-NUMA cuMem FABRIC handles are unavailable (%s); Engram DP "
            "replicas share tables through /dev/shm.",
            exc,
        )
        return False
    return True


def _create_table(num_bytes: int) -> tuple[list, ExportedHostTable]:
    """Allocate and export a table on this GPU's host NUMA node.

    Stops once the node would drop below its reserve. The kernel compacts to
    find 2M pages and fails the chunk, rather than OOM-killing, if it cannot.
    """
    numa_node = _gpu_numa_node()
    prop = _host_allocation_prop(numa_node)
    padded_bytes = round_up(num_bytes, _allocation_granularity(prop))
    reserve = get_memory_node_info(numa_node).total_memory * NUMA_RESERVE_FRACTION
    handles: list = []
    try:
        for offset, length in _chunk_ranges(padded_bytes):
            handles.append(_call_driver(cu.cuMemCreate, length, prop, 0))
            if get_memory_node_info(numa_node).available_memory < reserve:
                raise MemoryError(
                    f"NUMA node {numa_node} fell below its "
                    f"{reserve / 2**30:.1f} GiB reserve after "
                    f"{(offset + length) / 2**30:.1f} of "
                    f"{padded_bytes / 2**30:.1f} GiB"
                )
        exported = [_export_fabric_handle(handle) for handle in handles]
    except BaseException:
        _release(handles)
        raise
    return handles, ExportedHostTable(numa_node, padded_bytes, exported)


def _map_table(
    table: ExportedHostTable, num_bytes: int, handles: list
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a table, importing its handles unless given, and release the handles.

    Returns CPU and GPU uint8 views; either keeps the mapping alive.
    """
    device = torch.accelerator.current_device_index()
    address = None
    mapped: list[tuple[int, int]] = []
    try:
        if not handles:
            for exported in table.fabric_handles:
                handles.append(
                    _call_driver(
                        cu.cuMemImportFromShareableHandle, exported, FABRIC_HANDLE
                    )
                )
        address = int(_call_driver(cu.cuMemAddressReserve, table.padded_bytes, 0, 0, 0))
        for (offset, length), handle in zip(_chunk_ranges(table.padded_bytes), handles):
            _call_driver(cu.cuMemMap, address + offset, length, 0, handle, 0)
            mapped.append((offset, length))
        location = cu.CUmemLocationType
        access = [
            _read_write_access(location.CU_MEM_LOCATION_TYPE_DEVICE, device),
            _read_write_access(
                location.CU_MEM_LOCATION_TYPE_HOST_NUMA, table.numa_node
            ),
        ]
        _call_driver(
            cu.cuMemSetAccess, address, table.padded_bytes, access, len(access)
        )
    except BaseException:
        if address is not None:
            _unmap(address, table.padded_bytes, mapped)
        raise
    finally:
        # The mappings keep the allocations alive.
        _release(handles)

    host_buffer = (ctypes.c_uint8 * num_bytes).from_address(address)
    finalizer = weakref.finalize(
        host_buffer, _unmap, address, table.padded_bytes, mapped
    )
    finalizer.atexit = False  # type: ignore[misc]
    # Torch retains this object, so the GPU view keeps the buffer alive too.
    # Name the device: an index-less "cuda" makes torch copy the table.
    gpu_view = torch.as_tensor(
        SimpleNamespace(
            __cuda_array_interface__={
                "shape": (num_bytes,),
                "typestr": "|u1",
                "data": (address, False),
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

    Rank 0 creates the table and broadcasts its handles; the others import them.
    Returns CPU and GPU uint8 views of the table.
    """
    handles: list = []
    table = error = None
    if group.rank_in_group == 0 and fabric_host_memory_supported():
        try:
            handles, table = _create_table(num_bytes)
        except Exception as exc:
            error = describe_error(exc)
    table, error = group.broadcast_object((table, error))
    if table is not None:
        views = None
        try:
            views = _map_table(table, num_bytes, handles)
        except Exception as exc:
            error = describe_error(exc)
        if not (error := gather_errors(group, error)):
            return views
    if error and group.rank_in_group == 0:
        logger.warning(
            "Engram host table cannot use host-NUMA cuMem (%s); falling back "
            "to /dev/shm, whose 64K GPU pages make lookups slower.",
            error,
        )
    return None
