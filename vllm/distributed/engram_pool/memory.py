# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shard and staging memory: NUMA-local pinned host pages and cuMem HBM."""

import ctypes
import mmap
import os
import platform
import weakref
from types import SimpleNamespace

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

logger = init_logger(__name__)

_MPOL_PREFERRED = 1


def gpu_numa_node(device: int | None = None) -> int:
    """The NUMA node of a GPU's PCI device (current by default), or -1."""
    if device is None:
        device = torch.accelerator.current_device_index()
    props = torch.cuda.get_device_properties(device)
    bdf = (
        f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    )
    try:
        with open(f"/sys/bus/pci/devices/{bdf}/numa_node") as f:
            return int(f.read())
    except OSError:
        return -1


def _prefer_numa_node(address: int, num_bytes: int, node: int) -> None:
    sys_mbind = {"aarch64": 235, "x86_64": 237}.get(platform.machine())
    if sys_mbind is None or node < 0:
        return
    mask = ctypes.c_ulong(1 << node)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    if libc.syscall(
        ctypes.c_long(sys_mbind),
        ctypes.c_void_p(address),
        ctypes.c_ulong(num_bytes),
        ctypes.c_int(_MPOL_PREFERRED),
        ctypes.byref(mask),
        ctypes.c_ulong(node + 2),
        ctypes.c_uint(0),
    ):
        logger.warning_once(
            "Engram pool mbind to NUMA node %d failed: %s",
            node,
            os.strerror(ctypes.get_errno()),
        )


def _unregister(mapping: mmap.mmap, pointer: int) -> None:
    # Keeps the mapping alive until CUDA has released the registration.
    result = torch.cuda.cudart().cudaHostUnregister(pointer)
    if result.value != 0:
        logger.warning("Engram pool cudaHostUnregister failed: %s", result)


class SharedHost:
    """Pinned, GPU-mapped host bytes near the current GPU, in a memfd that the
    RDMA proxy process maps too."""

    def __init__(self, num_bytes: int, name: str) -> None:
        self.nbytes = round_up(max(num_bytes, 1), mmap.PAGESIZE)
        self.fd = os.memfd_create(f"vllm_engram_{name}", os.MFD_CLOEXEC)
        os.ftruncate(self.fd, self.nbytes)
        mapping = mmap.mmap(self.fd, self.nbytes, flags=mmap.MAP_SHARED)
        owner = np.frombuffer(mapping, dtype=np.uint8)
        pointer = owner.ctypes.data
        _prefer_numa_node(pointer, self.nbytes, gpu_numa_node())
        owner[:: mmap.PAGESIZE] = 0
        result = torch.cuda.cudart().cudaHostRegister(pointer, self.nbytes, 0)
        if result.value != 0:
            raise RuntimeError(f"cudaHostRegister({self.nbytes} B) failed: {result}")
        finalizer = weakref.finalize(owner, _unregister, mapping, pointer)
        finalizer.atexit = False  # type: ignore[misc]
        self.cpu = torch.from_numpy(owner)
        self.gpu = get_accelerator_view_from_cpu_tensor(self.cpu)

    def close(self) -> None:
        os.close(self.fd)


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
        """Map `handle` (this allocation by default) read-write for this GPU."""
        cu = self.cu
        address = int(self.call(cu.cuMemAddressReserve, self.size, 0, 0, 0))
        self.call(cu.cuMemMap, address, self.size, 0, handle or self.handle, 0)
        access = cu.CUmemAccessDesc()
        access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = torch.accelerator.current_device_index()
        access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.call(cu.cuMemSetAccess, address, self.size, [access], 1)
        return address

    def import_fabric(self, exported: bytes):
        fabric = self.cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        return self.call(self.cu.cuMemImportFromShareableHandle, exported, fabric)

    def release(self, handle) -> None:
        """Drop a handle once mapped: the mapping keeps the memory alive."""
        self.call(self.cu.cuMemRelease, handle)

    def dmabuf_fd(self, address: int) -> int:
        cu = self.cu
        kind = cu.CUmemRangeHandleType.CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
        return int(
            self.call(cu.cuMemGetHandleForAddressRange, address, self.size, kind, 0)
        )
