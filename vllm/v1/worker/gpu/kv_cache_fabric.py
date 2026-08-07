# SPDX-License-Identifier: Apache-2.0
"""Opt-in CUDA fabric-handle KV cache allocator for cross-node MNNVL transfer.

On GB200/GB300 NVL72 (multi-node NVLink domains), NIXL/UCX ``cuda_ipc`` can move
the KV cache across nodes over NVLink *only if the memory carries a CUDA fabric
handle*. Plain ``cudaMalloc`` memory (``torch.zeros(..., device="cuda")``) lacks
that property, so cross-node KV transfer silently downgrades to TCP. Allocating
the KV backing tensor via CUDA VMM with ``CU_MEM_HANDLE_TYPE_FABRIC`` fixes this
(measured on GB300 NVL72: cross-node KV transfer 110 MB/s -> 86 GB/s end to end).

Enable with ``VLLM_KV_CACHE_FABRIC=1`` (registered in ``vllm/envs.py`` so vLLM
forwards it to the spawned EngineCore/worker process). Safe by construction:
falls back to ``torch.zeros`` when disabled, on non-CUDA devices, when the
platform cannot export fabric handles (capability probe), or on any allocation
error -- so default behavior and non-NVLink platforms are unaffected.

KV backing tensors live for the whole serving lifetime (vLLM does not free the
KV pool while the engine runs), so fabric allocations are intentionally retained
in ``_KEEPALIVE`` and released together via ``release_all()`` at shutdown.
"""
from __future__ import annotations

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_KEEPALIVE: list[tuple] = []
_SUPPORTED: bool | None = None
_WARNED = False


def _driver():
    from cuda.bindings import driver as cu
    return cu


def _chk(cu, ret):
    err = ret[0]
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error {err}")
    return ret[1] if len(ret) == 2 else ret[1:]


def _fabric_create(size: int, dev_id: int) -> tuple[int, int, object]:
    cu = _driver()
    _chk(cu, cu.cuInit(0))
    dev = _chk(cu, cu.cuDeviceGet(dev_id))
    ctx = _chk(cu, cu.cuDevicePrimaryCtxRetain(dev))  # primary ctx: private ctx breaks NIXL reg
    _chk(cu, cu.cuCtxSetCurrent(ctx))
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev_id
    prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    gran = _chk(cu, cu.cuMemGetAllocationGranularity(
        prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
    asize = (size + gran - 1) // gran * gran
    handle = _chk(cu, cu.cuMemCreate(asize, prop, 0))
    va = _chk(cu, cu.cuMemAddressReserve(asize, gran, 0, 0))
    _chk(cu, cu.cuMemMap(va, asize, 0, handle, 0))
    acc = cu.CUmemAccessDesc()
    acc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    acc.location.id = dev_id
    acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _chk(cu, cu.cuMemSetAccess(va, asize, [acc], 1))
    return int(va), asize, handle


def _fabric_supported(dev_id: int) -> bool:
    """One-time capability probe: can this platform export a fabric handle?"""
    global _SUPPORTED
    if _SUPPORTED is None:
        try:
            va, asize, handle = _fabric_create(2 << 20, dev_id)  # 2 MiB probe
            cu = _driver()
            cu.cuMemUnmap(va, asize)
            cu.cuMemAddressFree(va, asize)
            cu.cuMemRelease(handle)
            _SUPPORTED = True
        except Exception as e:
            logger.warning("Fabric KV cache disabled: platform cannot export "
                           "CU_MEM_HANDLE_TYPE_FABRIC (%s)", e)
            _SUPPORTED = False
    return _SUPPORTED


class _CudaArray:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,), "typestr": "|i1",
            "data": (ptr, False), "version": 3,
        }


def maybe_fabric_kv_tensor(size: int, device: torch.device) -> torch.Tensor:
    """Drop-in for ``torch.zeros(size, dtype=torch.int8, device=device)``.

    Returns fabric-handle VMM memory (MNNVL-exportable) when
    ``VLLM_KV_CACHE_FABRIC`` is set and the platform supports it; otherwise the
    standard allocation. Never raises -- any failure falls back to torch.zeros.
    """
    global _WARNED
    if not envs.VLLM_KV_CACHE_FABRIC or device.type != "cuda":
        return torch.zeros(size, dtype=torch.int8, device=device)
    dev_id = device.index if device.index is not None else torch.cuda.current_device()
    if not _fabric_supported(dev_id):
        return torch.zeros(size, dtype=torch.int8, device=device)
    try:
        torch.cuda.init()
        va, asize, handle = _fabric_create(size, dev_id)
        _KEEPALIVE.append((handle, va, asize))
        t = torch.as_tensor(_CudaArray(va, size), device=f"cuda:{dev_id}")
        t.zero_()
        torch.cuda.synchronize()
        return t
    except Exception as e:  # OOM or any driver error -> safe fallback
        if not _WARNED:
            logger.warning("Fabric KV alloc failed (%s); using torch.zeros", e)
            _WARNED = True
        return torch.zeros(size, dtype=torch.int8, device=device)


def release_all() -> None:
    """Release retained fabric allocations (call at engine shutdown)."""
    cu = _driver()
    while _KEEPALIVE:
        handle, va, asize = _KEEPALIVE.pop()
        try:
            cu.cuMemUnmap(va, asize)
            cu.cuMemAddressFree(va, asize)
            cu.cuMemRelease(handle)
        except Exception:
            pass
