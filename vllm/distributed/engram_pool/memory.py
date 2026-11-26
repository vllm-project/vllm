# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shard and staging memory: NUMA-local pinned host pages and cuMem HBM."""

import contextlib
import ctypes
import mmap
import os
import uuid
import weakref
from types import SimpleNamespace

import numpy as np
import regex as re
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
    libnuma = get_libnuma()
    if node is None or libnuma is None:
        logger.warning_once("Engram pool host shards may land off the GPU's NUMA node")
        owner[:: mmap.PAGESIZE] = 0
        return
    mask = ctypes.c_ulong(1 << node)
    bind = True
    for start in range(0, owner.size, 1 << 30):
        chunk = owner[start : start + (1 << 30)]
        try:
            bind = bind and _claimable_bytes(node) > chunk.size + _BIND_HEADROOM
        except (OSError, KeyError, ValueError):
            bind = False  # Room unknown: never risk the OOM killer.
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
    """Warn if the pages of `[address, address + num_bytes)` are mostly off
    `node`, or (`huge`) mostly not huge pages, as when memory was short or
    fragmented. Chunked mbind may have split the range into several VMAs."""
    end = address + num_bytes

    def ours(line: str) -> bool:
        start = int(line.split(maxsplit=1)[0].split("-")[0], 16)
        return address <= start < end

    pages: dict[str, int] = {}
    with open("/proc/self/numa_maps") as f:
        for line in filter(ours, f):
            for n, count in re.findall(r" N(\d+)=(\d+)", line):
                pages[n] = pages.get(n, 0) + int(count)
    huge_bytes, mine = 0, False
    with open("/proc/self/smaps") as f:
        for line in f:
            if "-" in line.split(maxsplit=1)[0]:  # A VMA's header.
                mine = ours(line)
            elif mine and line.startswith("ShmemPmdMapped:"):
                huge_bytes += int(line.split()[1]) << 10
    total = sum(pages.values())
    if node is not None and total and pages.get(str(node), 0) < 0.9 * total:
        logger.warning(
            "Engram pool: only %.0f%% of %.1f GiB of host memory is on the GPU's "
            "NUMA node %d",
            100 * pages.get(str(node), 0) / total,
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
    """CUDA VMM backing in HBM or on a Grace host NUMA node."""

    def __init__(
        self,
        num_bytes: int,
        fabric: bool = False,
        rdma: bool = False,
        *,
        host_numa: bool = False,
    ):
        from cuda.bindings import driver as cu

        self.cu = cu
        device = torch.accelerator.current_device_index()
        prop = cu.CUmemAllocationProp()
        prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        if host_numa:
            prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
            prop.location.id = self.call(
                cu.cuDeviceGetAttribute,
                cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                device,
            )
        else:
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

    def map(self, handle=None, *, read_only: bool = False) -> int:
        """Map `handle` (this allocation by default) for this GPU."""
        cu = self.cu
        access = cu.CUmemAccessDesc()
        access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = torch.accelerator.current_device_index()
        access.flags = (
            cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
            if read_only
            else cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        )
        with contextlib.ExitStack() as rollback:
            address = int(self.call(cu.cuMemAddressReserve, self.size, 0, 0, 0))
            rollback.callback(self.call, cu.cuMemAddressFree, address, self.size)
            self.call(cu.cuMemMap, address, self.size, 0, handle or self.handle, 0)
            rollback.callback(self.call, cu.cuMemUnmap, address, self.size)
            self.call(cu.cuMemSetAccess, address, self.size, [access], 1)
            rollback.pop_all()
        if handle is None:
            self.address = address
        return address

    def unmap(self, address: int) -> None:
        self.call(self.cu.cuMemUnmap, address, self.size)
        self.call(self.cu.cuMemAddressFree, address, self.size)

    def map_fabric(self, exported: bytes, *, read_only: bool) -> int:
        fabric = self.cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        handle = self.call(self.cu.cuMemImportFromShareableHandle, exported, fabric)
        try:
            return self.map(handle, read_only=read_only)
        finally:
            self.release(handle)

    def release(self, handle) -> None:
        """Drop a handle once mapped: the mapping keeps the memory alive."""
        self.call(self.cu.cuMemRelease, handle)

    def dmabuf_fd(self) -> int:
        cu = self.cu
        kind = cu.CUmemRangeHandleType.CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD
        handle = cu.cuMemGetHandleForAddressRange
        return int(self.call(handle, self.address, self.size, kind, 0))
