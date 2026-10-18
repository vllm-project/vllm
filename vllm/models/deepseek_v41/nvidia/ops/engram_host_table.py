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

import torch
from cuda.bindings import driver as cu

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)


def _check(result: tuple):
    err, *rest = result
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return rest[0] if rest else None


def _host_numa_prop(device: int) -> cu.CUmemAllocationProp:
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
    prop.location.id = _check(
        cu.cuDeviceGetAttribute(
            cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device
        )
    )
    prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    return prop


@cache
def fabric_host_tables_supported() -> bool:
    """Whether host-NUMA cuMem allocations can export FABRIC handles here."""
    try:
        prop = _host_numa_prop(torch.accelerator.current_device_index())
        size = _check(cu.cuMemGetAllocationGranularity(prop, 0))
        handle = _check(cu.cuMemCreate(size, prop, 0))
        try:
            _check(
                cu.cuMemExportToShareableHandle(handle, prop.requestedHandleTypes, 0)
            )
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


def _unmap(ptr: cu.CUdeviceptr, size: int) -> None:
    cu.cuMemUnmap(ptr, size)
    cu.cuMemAddressFree(ptr, size)


def allocate_fabric_host_table(
    num_bytes: int, group: GroupCoordinator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map one host allocation into every rank of a node-local group.

    Returns CPU and GPU uint8 views of the same memory; the GPU view is valid
    while the CPU view is alive.
    """
    device = torch.accelerator.current_device_index()
    prop = _host_numa_prop(device)
    handle = shared = error = None
    if group.rank_in_group == 0:
        try:
            size = round_up(
                num_bytes, _check(cu.cuMemGetAllocationGranularity(prop, 0))
            )
            handle = _check(cu.cuMemCreate(size, prop, 0))
            fabric = _check(
                cu.cuMemExportToShareableHandle(handle, prop.requestedHandleTypes, 0)
            )
            shared = (prop.location.id, size, bytes(fabric.data))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if handle is not None:
                cu.cuMemRelease(handle)
    shared, error = group.broadcast_object((shared, error))
    if error is not None:
        raise RuntimeError("Engram cuMem table creation failed on rank 0: " + error)
    numa, size, fabric = shared
    ptr = None
    try:
        if handle is None:
            handle = _check(
                cu.cuMemImportFromShareableHandle(fabric, prop.requestedHandleTypes)
            )
        ptr = _check(cu.cuMemAddressReserve(size, 0, 0, 0))
        _check(cu.cuMemMap(ptr, size, 0, handle, 0))
        access = []
        for location, index in (
            (cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, device),
            (cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, numa),
        ):
            desc = cu.CUmemAccessDesc()
            desc.location.type, desc.location.id = location, index
            desc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            access.append(desc)
        _check(cu.cuMemSetAccess(ptr, size, access, len(access)))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # The mapping keeps the allocation alive.
        if handle is not None:
            cu.cuMemRelease(handle)
    # A peer that failed to map must not leave the others waiting later on.
    errors: list[str | None] = [None] * group.world_size
    torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
    failures = "; ".join(
        f"rank {rank}: {error}" for rank, error in enumerate(errors) if error
    )
    if failures:
        if ptr is not None:
            # Unmapping a reserved but unmapped range just returns an error.
            _unmap(ptr, size)
        raise RuntimeError("Engram cuMem table mapping failed: " + failures)
    assert ptr is not None
    owner = (ctypes.c_uint8 * num_bytes).from_address(int(ptr))
    finalizer = weakref.finalize(owner, _unmap, ptr, size)
    finalizer.atexit = False  # type: ignore[misc]
    # Name the device: an index-less "cuda" makes torch copy the table.
    cai = {
        "shape": (num_bytes,),
        "typestr": "|u1",
        "data": (int(ptr), False),
        "version": 3,
    }
    gpu = torch.as_tensor(
        SimpleNamespace(__cuda_array_interface__=cai),
        device=torch.device("cuda", device),
    )
    return torch.frombuffer(owner, dtype=torch.uint8), gpu
