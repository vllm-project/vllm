# SPDX-License-Identifier: Apache-2.0
"""Opt-in CUDA fabric-handle KV cache allocator for cross-node MNNVL transfer.

On GB200/GB300 NVL72 (multi-node NVLink domains), NIXL/UCX ``cuda_ipc`` can move
KV cache across nodes over NVLink *only if the memory carries a CUDA fabric
handle*. Plain ``cudaMalloc`` memory (``torch.zeros(..., device="cuda")``) lacks
that property, so cross-node KV transfer silently downgrades to TCP. Allocating
the KV backing tensor via CUDA VMM with ``CU_MEM_HANDLE_TYPE_FABRIC`` fixes this
(measured on GB300 NVL72: NIXL KV transfer 110 MB/s -> 86 GB/s end to end).

Opt-in via ``VLLM_KV_CACHE_FABRIC=1``. NOTE: for the flag to reach the v1
``EngineCore``/worker process it should be registered in ``vllm/envs.py`` (vLLM
only forwards its *known* ``VLLM_*`` env vars to spawned workers) — see the PR
description. Falls back to ``torch.zeros`` when disabled or on any error, so
default behavior is unchanged.
"""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_KEEPALIVE: list = []  # fabric allocations must live for the process lifetime
_WARNED = False


def _enabled() -> bool:
    return os.environ.get("VLLM_KV_CACHE_FABRIC", "0") not in ("0", "", "false", "False")


def _fabric_alloc(size: int, dev_id: int) -> int:
    from cuda.bindings import driver as cu

    def chk(ret):
        err = ret[0]
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver error {err} in fabric KV alloc")
        return ret[1] if len(ret) == 2 else ret[1:]

    chk(cu.cuInit(0))
    dev = chk(cu.cuDeviceGet(dev_id))
    ctx = chk(cu.cuDevicePrimaryCtxRetain(dev))  # private ctx breaks NIXL registration
    chk(cu.cuCtxSetCurrent(ctx))
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev_id
    prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    gran = chk(cu.cuMemGetAllocationGranularity(
        prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
    asize = (size + gran - 1) // gran * gran
    handle = chk(cu.cuMemCreate(asize, prop, 0))
    va = chk(cu.cuMemAddressReserve(asize, gran, 0, 0))
    chk(cu.cuMemMap(va, asize, 0, handle, 0))
    acc = cu.CUmemAccessDesc()
    acc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    acc.location.id = dev_id
    acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    chk(cu.cuMemSetAccess(va, asize, [acc], 1))
    _KEEPALIVE.append((handle, va, asize))
    return int(va)


class _CudaArray:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,), "typestr": "|i1",
            "data": (ptr, False), "version": 3,
        }


def maybe_fabric_kv_tensor(size: int, device: torch.device) -> torch.Tensor:
    """Drop-in for ``torch.zeros(size, dtype=torch.int8, device=device)``.

    Uses CUDA VMM fabric-handle memory when ``VLLM_KV_CACHE_FABRIC`` is set,
    else (or on any failure) the standard allocation.
    """
    if not _enabled() or device.type != "cuda":
        return torch.zeros(size, dtype=torch.int8, device=device)
    global _WARNED
    try:
        dev_id = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.init()
        ptr = _fabric_alloc(size, dev_id)
        t = torch.as_tensor(_CudaArray(ptr, size), device=f"cuda:{dev_id}")
        t.zero_()
        torch.cuda.synchronize()
        return t
    except Exception as e:  # never break serving on allocator issues
        if not _WARNED:
            logger.warning("Fabric KV alloc failed (%s); falling back to torch.zeros", e)
            _WARNED = True
        return torch.zeros(size, dtype=torch.int8, device=device)
