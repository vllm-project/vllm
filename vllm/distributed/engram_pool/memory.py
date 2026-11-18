# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shard and staging memory: NUMA-local pinned host pages and cuMem HBM."""

import ctypes
import mmap
import os
import weakref
from types import SimpleNamespace

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.host_memory import exclude_from_core_dump, pin_tensor
from vllm.utils.math_utils import round_up
from vllm.utils.numa_utils import get_libnuma
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)


def _prefer_numa_node(address: int, num_bytes: int, node: int | None) -> None:
    libnuma = get_libnuma()
    if libnuma is None or node is None:
        logger.warning_once("Engram pool host shards may land off the GPU's NUMA node")
        return
    mask = ctypes.c_ulong(1 << node)
    # mbind(..., MPOL_PREFERRED = 1, nodemask, maxnode, flags)
    if libnuma.mbind(
        ctypes.c_void_p(address),
        ctypes.c_ulong(num_bytes),
        1,
        ctypes.byref(mask),
        ctypes.c_ulong(node + 2),
        0,
    ):
        logger.warning_once("Engram pool mbind to NUMA node %d failed", node)


def _unregister(mapping: mmap.mmap, pointer: int) -> None:
    # Keeps the mapping alive until CUDA has released the registration.
    cudart = torch.cuda.cudart()
    if (result := cudart.cudaHostUnregister(pointer)).value != 0:
        logger.warning("Engram pool cudaHostUnregister failed: %s", result)


class SharedHost:
    """Pinned, GPU-mapped host bytes near the current GPU, in a memfd that RDMA
    proxy processes map too."""

    def __init__(self, num_bytes: int, name: str) -> None:
        self.nbytes = round_up(max(num_bytes, 1), mmap.PAGESIZE)
        self.fd = os.memfd_create(f"vllm_engram_{name}", os.MFD_CLOEXEC)
        os.ftruncate(self.fd, self.nbytes)
        mapping = mmap.mmap(self.fd, self.nbytes, flags=mmap.MAP_SHARED)
        exclude_from_core_dump(mapping)
        owner = np.frombuffer(mapping, dtype=np.uint8)
        pointer = owner.ctypes.data
        device = torch.accelerator.current_device_index()
        node = current_platform.get_device_numa_node(device)
        _prefer_numa_node(pointer, self.nbytes, node)
        owner[:: mmap.PAGESIZE] = 0
        self.cpu = torch.from_numpy(owner)
        pin_tensor(self.cpu)
        finalizer = weakref.finalize(owner, _unregister, mapping, pointer)
        finalizer.atexit = False  # type: ignore[misc]
        self.gpu = get_accelerator_view_from_cpu_tensor(self.cpu)


def device_view(address: int, num_bytes: int, owner: object = None) -> torch.Tensor:
    """A uint8 tensor over mapped GPU memory; torch retains `owner` with it."""
    # Name the device: an index-less "cuda" makes torch copy the memory.
    return torch.as_tensor(
        SimpleNamespace(
            __cuda_array_interface__={
                "shape": (num_bytes,),
                "typestr": "|u1",
                "data": (address, False),
                "version": 3,
            },
            owner=owner,
        ),
        device=torch.device("cuda", torch.accelerator.current_device_index()),
    )


class CuMem:
    """cuMem HBM: FABRIC-exportable for NVLink peers, or GPUDirect-RDMA capable."""

    def __init__(self, num_bytes: int, fabric: bool = False, rdma: bool = False):
        from cuda.bindings import driver as cu

        self.cu = cu
        device = torch.accelerator.current_device_index()
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
        if fabric:
            prop.requestedHandleTypes = (
                cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            )
        if rdma:
            prop.allocFlags.gpuDirectRDMACapable = 1
        granularity = self.call(cu.cuMemGetAllocationGranularity, prop, 0)
        self.size = round_up(max(num_bytes, 1), granularity)
        self.handle = self.call(cu.cuMemCreate, self.size, prop, 0)

    def call(self, fn, *args):
        error, *result = fn(*args)
        if error != self.cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{fn.__name__} failed: {error.name}")
        return result[0] if result else None

    def export_fabric(self) -> bytes:
        fabric = self.cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        return bytes(
            self.call(self.cu.cuMemExportToShareableHandle, self.handle, fabric, 0).data
        )

    def map(self, handle=None) -> int:
        """Map `handle` read-write for this GPU; by default this allocation, at
        `self.address`."""
        cu = self.cu
        address = int(self.call(cu.cuMemAddressReserve, self.size, 0, 0, 0))
        self.call(cu.cuMemMap, address, self.size, 0, handle or self.handle, 0)
        access = cu.CUmemAccessDesc()
        access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = torch.accelerator.current_device_index()
        access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.call(cu.cuMemSetAccess, address, self.size, [access], 1)
        if handle is None:
            self.address = address
        return address

    def import_fabric(self, exported: bytes):
        fabric = self.cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        return self.call(self.cu.cuMemImportFromShareableHandle, exported, fabric)

    def release(self, handle) -> None:
        """Drop a handle once mapped: the mapping keeps the memory alive."""
        self.call(self.cu.cuMemRelease, handle)

    def dmabuf_fd(self) -> int:
        cu = self.cu
        kind = cu.CUmemRangeHandleType.CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
        handle = cu.cuMemGetHandleForAddressRange
        return int(self.call(handle, self.address, self.size, kind, 0))
