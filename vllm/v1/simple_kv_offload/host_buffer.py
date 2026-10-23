# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One host mapping backing all of a rank's CPU KV caches.

Ordinary pages by default, or explicit 2 MiB / 1 GiB HugeTLB pages, which
never fall back to ordinary memory: a short pool or hugetlb cgroup limit
fails at startup.
"""

import ctypes
import mmap
import os
import platform

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

HUGEPAGE_SIZES = {"2MB": 2 << 20, "1GB": 1 << 30}

_MAP_HUGETLB = 0x40000
_MAP_HUGE_SHIFT = 26
_MADV_DONTFORK = getattr(mmap, "MADV_DONTFORK", 10)
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
_MPOL_PREFERRED = 1
_SYS_MBIND = {"x86_64": 237, "aarch64": 235}  # glibc does not export mbind


def parse_hugepage_size(value: object) -> int | None:
    """Parse ``cpu_hugepage_block_size``; ``None``/``""``/``"none"`` disable it."""
    key = str(value or "none").strip().upper().replace("IB", "B")
    key = {"2M": "2MB", "1G": "1GB"}.get(key, key)
    if key == "NONE":
        return None
    if key not in HUGEPAGE_SIZES:
        raise ValueError(
            f"Unsupported cpu_hugepage_block_size {value!r}; "
            f"expected one of {sorted(HUGEPAGE_SIZES)} or none"
        )
    return HUGEPAGE_SIZES[key]


class HostBuffer:
    """One anonymous mapping, viewed as an int8 CPU tensor.

    Keep this object alive as long as any tensor views the mapping.
    """

    def __init__(
        self,
        nbytes: int,
        hugepage_size: int | None = None,
        numa_node: int | None = None,
    ):
        self.page_size = hugepage_size or mmap.PAGESIZE
        self.nbytes = max(1, -(-nbytes // self.page_size)) * self.page_size
        kind = f"{self.page_size >> 20} MiB HugeTLB" if hugepage_size else "ordinary"
        flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        if hugepage_size:
            flags |= _MAP_HUGETLB | (hugepage_size.bit_length() - 1) << _MAP_HUGE_SHIFT
        try:
            # HugeTLB reserves every page from the host pool here.
            self._mmap = mmap.mmap(-1, self.nbytes, flags=flags)
        except OSError as e:
            raise RuntimeError(
                f"Cannot map {self.nbytes / 2**30:.2f} GiB of {kind} pages: {e}. "
                "Reserve the pages on the host and request them for the container."
            ) from e
        # Children never need the buffer, so fork() neither copies nor
        # copy-on-write shares the pinned pages.
        self._mmap.madvise(_MADV_DONTFORK)
        self.tensor = torch.frombuffer(memoryview(self._mmap), dtype=torch.int8)
        if numa_node is not None:
            self._prefer_node(numa_node)
        try:
            # Fault every page now: a cgroup or pool shortfall becomes an
            # error here instead of SIGBUS or an OOM kill while serving.
            self._mmap.madvise(_MADV_POPULATE_WRITE)
        except OSError as e:
            if e.errno == 22 and not hugepage_size:  # kernel < 5.14
                self.tensor[:: self.page_size].zero_()
                return
            raise RuntimeError(
                f"Cannot populate {self.nbytes / 2**30:.2f} GiB of {kind} pages: "
                f"{e}. The container's memory or hugetlb limit is too small."
            ) from e

    def _prefer_node(self, node: int) -> None:
        nr = _SYS_MBIND.get(platform.machine())
        if nr is None:
            return
        mask = (ctypes.c_ulong * (node // 64 + 1))()
        mask[node // 64] = 1 << (node % 64)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        ret = libc.syscall(
            ctypes.c_long(nr),
            ctypes.c_void_p(self.tensor.data_ptr()),
            ctypes.c_ulong(self.nbytes),
            ctypes.c_int(_MPOL_PREFERRED),
            mask,
            ctypes.c_ulong(len(mask) * 64 + 1),
            ctypes.c_uint(0),
        )
        if ret != 0:
            err = ctypes.get_errno()
            logger.warning("mbind to NUMA node %d failed: %s", node, os.strerror(err))
