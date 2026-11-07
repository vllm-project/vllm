# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One host mapping backing a rank's CPU KV caches."""

import mmap
import os

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.cpu.shared_offload_region import (
    _get_populate_write_fn,
    _madvise_populate_write,
)

logger = init_logger(__name__)

HUGEPAGE_SIZES = {"2MB": 2 << 20, "1GB": 1 << 30}

_MAP_HUGETLB = 0x40000
_MAP_HUGE_SHIFT = 26


def parse_hugepage_size(value: object) -> int | None:
    """Parse ``cpu_hugepage_block_size``; ``None`` or ``"none"`` disables it."""
    key = str(value or "none").upper()
    if key == "NONE":
        return None
    if key not in HUGEPAGE_SIZES:
        raise ValueError(
            f"Unsupported cpu_hugepage_block_size {value!r}; "
            f"expected one of {sorted(HUGEPAGE_SIZES)} or none"
        )
    return HUGEPAGE_SIZES[key]


def alloc_host_buffer(nbytes: int, hugepage_size: int | None = None) -> torch.Tensor:
    """Map and populate an int8 CPU tensor of whole backing pages.

    HugeTLB pages never fall back to ordinary pages.
    """
    page_size = hugepage_size or mmap.PAGESIZE
    nbytes = round_up(max(nbytes, 1), page_size)
    kind = f"{page_size >> 20} MiB HugeTLB" if hugepage_size else "ordinary"
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    if hugepage_size:
        flags |= _MAP_HUGETLB | (hugepage_size.bit_length() - 1) << _MAP_HUGE_SHIFT
    try:
        buf = mmap.mmap(-1, nbytes, flags=flags)
    except OSError as e:
        raise RuntimeError(
            f"Cannot map {nbytes / 2**30:.2f} GiB of {kind} pages: {e}. "
            "Reserve the pages on the host and request them for the container."
        ) from e
    # Keep pinned pages out of fork() children.
    buf.madvise(mmap.MADV_DONTFORK)
    if not hugepage_size and os.environ.get("THP_MEM_ALLOC_ENABLE") == "1":
        try:
            buf.madvise(mmap.MADV_HUGEPAGE)
        except OSError as e:
            logger.warning("MADV_HUGEPAGE failed: %s", e)
    populate = _madvise_populate_write if hugepage_size else _get_populate_write_fn(buf)
    try:
        # A pool or cgroup shortfall fails here instead of as SIGBUS or an
        # OOM kill while serving.
        populate(buf, 0, nbytes)
    except OSError as e:
        raise RuntimeError(
            f"Cannot populate {nbytes / 2**30:.2f} GiB of {kind} pages: "
            f"{e}. The container's memory or hugetlb limit is too small."
        ) from e
    return torch.frombuffer(buf, dtype=torch.int8)
