# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shard and staging memory: NUMA-local pinned host pages and cuMem HBM."""

import contextlib
import ctypes
import mmap
import os
import re
import uuid
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


MPOL_PREFERRED, MPOL_BIND = 1, 2
# Room left on a NUMA node below which host memory is no longer bound to it:
# a bound allocation that cannot be met invokes the OOM killer, and this
# node's other GPUs allocate theirs at the same time.
_BIND_HEADROOM = 4 << 30


def _claimable_bytes(node: int) -> int:
    """Free and clean page-cache bytes on NUMA node `node`."""
    fields = {}
    with open(f"/sys/devices/system/node/node{node}/meminfo") as f:
        for line in f:  # "Node 0 MemFree:   123 kB"
            key, value = line.split()[2:4]
            fields[key.rstrip(":")] = int(value) << 10
    cache = fields["Active(file)"] + fields["Inactive(file)"]
    return fields["MemFree"] + cache - fields["Dirty"] - fields["Writeback"]


def _fault_on_node(owner: np.ndarray, node: int | None) -> None:
    """Fault in `owner`'s pages on NUMA node `node`, 1 GB at a time.
    MPOL_PREFERRED takes another node's free memory before reclaiming page
    cache on `node` (with the default zone_reclaim_mode=0), so each chunk is
    bound to `node` while it clearly has room, and only preferred after."""
    libnuma = None if node is None else get_libnuma()
    if libnuma is None:
        logger.warning_once("Engram pool host shards may land off the GPU's NUMA node")
        owner[:: mmap.PAGESIZE] = 0
        return
    mask = ctypes.c_ulong(1 << node)
    bind = True
    for start in range(0, owner.size, 1 << 30):
        chunk = owner[start : start + (1 << 30)]
        with contextlib.suppress(OSError, KeyError, ValueError):
            bind = bind and _claimable_bytes(node) > chunk.size + _BIND_HEADROOM
        if libnuma.mbind(
            ctypes.c_void_p(chunk.ctypes.data),
            ctypes.c_ulong(chunk.size),
            MPOL_BIND if bind else MPOL_PREFERRED,
            ctypes.byref(mask),
            ctypes.c_ulong(node + 2),
            0,
        ):
            logger.warning_once("Engram pool mbind to NUMA node %d failed", node)
        chunk[:: mmap.PAGESIZE] = 0


def _check_pages(address: int, num_bytes: int, node: int | None, huge: bool) -> None:
    """Warn if a mapping's pages are mostly off `node`, or (`huge`) mostly not
    huge pages, as when memory was short or fragmented."""
    start = f"{address:x}"
    with open("/proc/self/numa_maps") as f:
        line = next((x for x in f if x.startswith(start + " ")), "")
    pages = {n: int(c) for n, c in re.findall(r" N(\d+)=(\d+)", line)}
    on_node = pages.get(str(node), 0) / max(sum(pages.values()), 1)
    huge_bytes = 0
    with open("/proc/self/smaps") as f:
        mine = False
        for x in f:
            if "-" in x.split()[0]:
                mine = x.startswith(start + "-")
            elif mine and x.startswith("ShmemPmdMapped:"):
                huge_bytes = int(x.split()[1]) << 10
    if node is not None and on_node < 0.9:
        logger.warning(
            "Engram pool: only %.0f%% of %.1f GiB of host memory is on the GPU's "
            "NUMA node %d",
            100 * on_node,
            num_bytes / (1 << 30),
            node,
        )
    if huge and huge_bytes < 0.9 * num_bytes:
        logger.warning(
            "Engram pool: only %.0f%% of %.1f GiB of host memory is on huge pages; "
            "RDMA gathers from it run slower",
            100 * huge_bytes / num_bytes,
            num_bytes / (1 << 30),
        )


def _unregister(mapping: mmap.mmap, pointer: int) -> None:
    # Keeps the mapping alive until CUDA has released the registration.
    cudart = torch.cuda.cudart()
    if (result := cudart.cudaHostUnregister(pointer)).value != 0:
        logger.warning("Engram pool cudaHostUnregister failed: %s", result)


# Not in Python's mmap module yet (Linux 6.1+).
MADV_COLLAPSE = 25


def _huge_page_bytes() -> int:
    """The size of a transparent huge page: 2 MB, or 512 MB on 64 KB pages."""
    try:
        with open("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size") as f:
            return int(f.read())
    except OSError:
        return mmap.PAGESIZE


class SharedHost:
    """Pinned, GPU-mapped host bytes near the current GPU, in a file that RDMA
    proxy processes map too: a memfd, or, with `thp_dir`, an unlinked file on
    that tmpfs.

    `thp_dir` must be on a tmpfs mounted with `huge=always`, so the shard gets
    transparent huge pages. A NIC translates every address it reads, and
    gathers rows at random: over a large shard on base pages it misses its
    translation cache on nearly every row."""

    def __init__(self, num_bytes: int, name: str, thp_dir: str | None = None) -> None:
        page = _huge_page_bytes() if thp_dir else mmap.PAGESIZE
        self.nbytes = round_up(max(num_bytes, 1), page)
        if thp_dir:
            path = os.path.join(thp_dir, f"vllm_engram_{name}_{uuid.uuid4().hex}")
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
            self.fd = os.open(path, flags, 0o600)
            # Peers open it through /proc/<pid>/fd, as they would a memfd.
            os.unlink(path)
        else:
            self.fd = os.memfd_create(f"vllm_engram_{name}", os.MFD_CLOEXEC)
        os.ftruncate(self.fd, self.nbytes)
        mapping = mmap.mmap(self.fd, self.nbytes, flags=mmap.MAP_SHARED)
        exclude_from_core_dump(mapping)
        owner = np.frombuffer(mapping, dtype=np.uint8)
        pointer = owner.ctypes.data
        device = torch.accelerator.current_device_index()
        node = current_platform.get_device_numa_node(device)
        if thp_dir:
            # Let the kernel compact memory to fault in huge pages...
            mapping.madvise(mmap.MADV_HUGEPAGE)
        _fault_on_node(owner, node)
        if thp_dir:
            # ...and collapse ranges that still fell back to base pages (e.g.
            # when memory was too fragmented), 1 GB at a time.
            for start in range(0, self.nbytes, 1 << 30):
                with contextlib.suppress(OSError):
                    mapping.madvise(
                        MADV_COLLAPSE, start, min(1 << 30, self.nbytes - start)
                    )
        _check_pages(pointer, self.nbytes, node, huge=thp_dir is not None)
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
