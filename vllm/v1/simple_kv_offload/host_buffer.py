# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One host mapping backing all of a rank's CPU KV caches.

Ordinary pages by default, or explicit 2 MiB / 1 GiB HugeTLB pages, which
never fall back to ordinary memory: a short pool or hugetlb cgroup limit
fails at startup.
"""

import mmap
import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

HUGEPAGE_SIZES = {"2MB": 2 << 20, "1GB": 1 << 30}

_MAP_HUGETLB = 0x40000
_MAP_HUGE_SHIFT = 26
_MADV_HUGEPAGE = getattr(mmap, "MADV_HUGEPAGE", 14)
_MADV_DONTFORK = getattr(mmap, "MADV_DONTFORK", 10)
_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)


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
        if not hugepage_size and os.environ.get("THP_MEM_ALLOC_ENABLE") == "1":
            try:
                self._mmap.madvise(_MADV_HUGEPAGE)
            except OSError as e:
                logger.warning("MADV_HUGEPAGE failed: %s", e)
        self.tensor = torch.frombuffer(memoryview(self._mmap), dtype=torch.int8)
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
