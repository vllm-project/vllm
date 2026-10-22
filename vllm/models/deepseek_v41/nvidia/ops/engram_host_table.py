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

FABRIC = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
# Tables are created in chunks, so the creator can stop before draining its node.
CHUNK_BYTES = 2 << 30
# Share of the NUMA node left to other allocations.
NODE_RESERVE = 0.05


class SharedHostTable(NamedTuple):
    """What a peer needs to map a host table: its node, layout and handles."""

    numa_id: int
    num_bytes: int
    size: int
    chunk_bytes: int
    handles: list[bytes]

    def chunks(self) -> list[tuple[int, int]]:
        return [
            (offset, min(self.chunk_bytes, self.size - offset))
            for offset in range(0, self.size, self.chunk_bytes)
        ]


def _check(result: tuple):
    err, *rest = result
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return rest[0] if rest else None


def _host_numa_id() -> int:
    attr = cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
    return _check(
        cu.cuDeviceGetAttribute(attr, torch.accelerator.current_device_index())
    )


def _host_prop(numa_id: int) -> cu.CUmemAllocationProp:
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    prop.location.id = numa_id
    prop.requestedHandleTypes = FABRIC
    return prop


def _granularity(prop: cu.CUmemAllocationProp) -> int:
    return _check(cu.cuMemGetAllocationGranularity(prop, 0))


def _export(handle) -> bytes:
    return bytes(_check(cu.cuMemExportToShareableHandle(handle, FABRIC, 0)).data)


def _read_write(location_type, location_id: int) -> cu.CUmemAccessDesc:
    desc = cu.CUmemAccessDesc()
    desc.location.type = location_type
    desc.location.id = location_id
    desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return desc


def _unmap(ptr: int, size: int, chunks: list[tuple[int, int]]) -> None:
    for offset, length in chunks:
        cu.cuMemUnmap(ptr + offset, length)
    cu.cuMemAddressFree(ptr, size)


@cache
def host_tables_supported() -> bool:
    """Whether host-NUMA cuMem allocations can export FABRIC handles here."""
    try:
        prop = _host_prop(_host_numa_id())
        handle = _check(cu.cuMemCreate(_granularity(prop), prop, 0))
        try:
            _export(handle)
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


def host_table_fits(num_bytes: int) -> bool:
    """Whether a table of this size can live on this GPU's host NUMA node."""
    if not host_tables_supported():
        return False
    total = get_memory_node_info(_host_numa_id()).total_memory
    return num_bytes <= total * (1 - NODE_RESERVE)


def create_host_table(num_bytes: int) -> tuple[list, SharedHostTable]:
    """Allocate and export a table on this GPU's host NUMA node.

    Stops once the node would drop below its reserve. The kernel compacts to
    find 2M pages and fails the chunk, rather than OOM-killing, if it cannot.
    """
    numa_id = _host_numa_id()
    prop = _host_prop(numa_id)
    size = round_up(num_bytes, _granularity(prop))
    table = SharedHostTable(numa_id, num_bytes, size, CHUNK_BYTES, [])
    reserve = get_memory_node_info(numa_id).total_memory * NODE_RESERVE
    handles: list = []
    try:
        for offset, length in table.chunks():
            handles.append(_check(cu.cuMemCreate(length, prop, 0)))
            if get_memory_node_info(numa_id).available_memory < reserve:
                raise MemoryError(
                    f"NUMA node {numa_id} fell below its {reserve / 2**30:.1f} GiB "
                    f"reserve after {(offset + length) / 2**30:.1f} of "
                    f"{size / 2**30:.1f} GiB"
                )
        table.handles.extend(map(_export, handles))
    except BaseException:
        for handle in handles:
            cu.cuMemRelease(handle)
        raise
    return handles, table


def map_host_table(
    table: SharedHostTable, handles: list
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a table on this rank, importing its handles unless given.

    Consumes the handles. Returns CPU and GPU uint8 views of the table; either
    keeps the mapping alive.
    """
    device = torch.accelerator.current_device_index()
    chunks = table.chunks()
    ptr = None
    num_mapped = 0
    try:
        if not handles:
            for exported in table.handles:
                handles.append(
                    _check(cu.cuMemImportFromShareableHandle(exported, FABRIC))
                )
        ptr = int(_check(cu.cuMemAddressReserve(table.size, 0, 0, 0)))
        for (offset, length), handle in zip(chunks, handles):
            _check(cu.cuMemMap(ptr + offset, length, 0, handle, 0))
            num_mapped += 1
        access = [
            _read_write(cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, device),
            _read_write(
                cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, table.numa_id
            ),
        ]
        _check(cu.cuMemSetAccess(ptr, table.size, access, len(access)))
    except BaseException:
        if ptr is not None:
            _unmap(ptr, table.size, chunks[:num_mapped])
        raise
    finally:
        # The mappings keep the allocations alive.
        for handle in handles:
            cu.cuMemRelease(handle)

    owner = (ctypes.c_uint8 * table.num_bytes).from_address(ptr)
    finalizer = weakref.finalize(owner, _unmap, ptr, table.size, chunks)
    finalizer.atexit = False  # type: ignore[misc]
    # Torch retains this object, so the GPU view keeps the owner alive too.
    # Name the device: an index-less "cuda" makes torch copy the table.
    gpu = torch.as_tensor(
        SimpleNamespace(
            __cuda_array_interface__={
                "shape": (table.num_bytes,),
                "typestr": "|u1",
                "data": (ptr, False),
                "version": 3,
            },
            owner=owner,
        ),
        device=torch.device("cuda", device),
    )
    return torch.frombuffer(owner, dtype=torch.uint8), gpu


def share_host_table(
    num_bytes: int, group: GroupCoordinator
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Map one table on every rank of a node-local group, or None on all of them.

    Rank 0 creates the table and broadcasts its handles; the others import them.
    Returns CPU and GPU uint8 views of the table.
    """
    handles: list = []
    table = error = None
    if group.rank_in_group == 0 and host_tables_supported():
        try:
            handles, table = create_host_table(num_bytes)
        except Exception as exc:
            error = describe_error(exc)
    table, error = group.broadcast_object((table, error))
    if table is not None:
        views = None
        try:
            views = map_host_table(table, handles)
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
