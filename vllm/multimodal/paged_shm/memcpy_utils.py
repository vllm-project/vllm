# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
CPU↔CPU memcpy primitives for paged shared memory.

Used by ``PagedShmStorage`` for the CPU-side paths:

  - ``_write_cpu``     : contiguous  -> paged blocks
  - ``read_to_numpy``  : paged blocks -> contiguous
  - ``read_to_tensor`` : paged blocks -> contiguous (``device="cpu"``)

Server CPUs usually need several cores to saturate copy bandwidth, so we
use a numba ``parallel=True`` kernel that copies independent segments
concurrently. Desktop CPUs already saturate with a single core, so the
multi-threading heuristic stays off there.

Heuristic — enable multi-threading only when the host looks like a server:

  - more than one NUMA node, or
  - more than 32 logical cores, or
  - more than 512 GiB of physical memory.

Parameters:

  - Block size: 1 MiB in the common case (``PagedShmStorage`` blocks),
    but the code adapts to whatever the caller passes.
  - Sub-chunk size: 8 KiB — the knee of the sub-chunk sweep. The actual
    sub-chunk is ``gcd(block_size, 8 KiB)`` so it divides every block
    evenly and every offset we compute is 8 KiB aligned whenever the block
    size is a multiple of 8 KiB.
  - Threads: 8 — the plateau of the thread sweep.

Fallback: without numba, a pure-numpy serial loop and no sub-chunking
(one segment per block).
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from typing import Final

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional numba import.
#
# ``NUMBA_NUM_THREADS`` must be set BEFORE ``import numba``: it caps the
# process-wide thread pool size, and ``set_num_threads`` can only reduce
# within this limit.
# ---------------------------------------------------------------------------

_DEFAULT_NUMBA_THREADS: Final[int] = 64
if not os.environ.get("NUMBA_NUM_THREADS"):
    os.environ["NUMBA_NUM_THREADS"] = str(_DEFAULT_NUMBA_THREADS)

try:
    from numba import get_num_threads, njit, prange, set_num_threads

    _HAS_NUMBA = True
    _MAX_NUMBA_THREADS: Final[int] = int(os.environ["NUMBA_NUM_THREADS"])
except ImportError:  # pragma: no cover - depends on environment
    _HAS_NUMBA = False
    _MAX_NUMBA_THREADS = 1
    logger.info("numba not available; paged_shm memcpy falls back to numpy")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# 8 KiB is the knee of the sub-chunk sweep: bandwidth is already saturated,
# and per-task overhead is still small.
_SUBCHUNK_BYTES: Final[int] = 8 * 1024

# 8 threads is the plateau of the thread sweep. The heuristic below caps at
# this value; more threads only add scheduling jitter.
_MAX_COPY_THREADS: Final[int] = 8


# ---------------------------------------------------------------------------
# Topology detection (cheap, done once at import)
# ---------------------------------------------------------------------------

def _detect_logical_cores() -> int:
    return os.cpu_count() or 1


def _detect_numa_nodes() -> int:
    """Return the number of NUMA nodes; ``1`` on non-Linux or on failure."""
    try:
        with open("/sys/devices/system/node/possible") as f:
            spec = f.read().strip()
    except OSError:
        return 1
    if not spec:
        return 1
    # Format is a CPU-list like "0-1" or "0-3,8-11".
    count = 0
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            count += int(hi) - int(lo) + 1
        else:
            count += 1
    return max(count, 1)


def _detect_memory_bytes() -> int:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except (ValueError, OSError, AttributeError):
        pass
    return 0


# ---------------------------------------------------------------------------
# Heuristic: enable multi-threaded copy?
# ---------------------------------------------------------------------------

def _auto_detect_mt(cores: int, numa: int, mem_bytes: int) -> bool:
    # Multiple NUMA nodes: almost certainly a server; a single thread cannot
    # saturate cross-node bandwidth.
    if numa > 1:
        return True
    # Many cores: covers high-end single-socket workstations with many
    # memory channels.
    if cores > 32:
        return True
    # Huge memory: a host with >512 GiB RAM is essentially always a
    # multi-channel server, even if NUMA is disabled or cores are capped.
    if mem_bytes > (1 << 39):  # 512 GiB
        return True
    return False


_CORES: Final[int] = _detect_logical_cores()
_NUMA: Final[int] = _detect_numa_nodes()
_MEM: Final[int] = _detect_memory_bytes()

# Without numba there is no thread pool to drive; always single-threaded.
_USE_MT: Final[bool] = _HAS_NUMBA and _auto_detect_mt(_CORES, _NUMA, _MEM)

_COPY_THREADS: Final[int] = (
    min(_CORES, _MAX_COPY_THREADS) if _USE_MT else 1
)


def use_multithread() -> bool:
    """Whether paged-shm CPU copies use multi-threading (cached)."""
    return _USE_MT


def get_copy_threads() -> int:
    """Thread count used when ``n_threads`` is not specified."""
    return _COPY_THREADS


def get_topology() -> tuple[int, int, int]:
    """``(logical_cores, numa_nodes, mem_bytes)`` for logging/diagnostics."""
    return _CORES, _NUMA, _MEM


def _log_decision_once() -> None:
    if not _HAS_NUMBA:
        logger.info(
            "paged_shm memcpy: numba unavailable -> single-thread numpy "
            "(cores=%d numa=%d mem=%.0fGiB)",
            _CORES,
            _NUMA,
            _MEM / (1 << 30),
        )
        return
    mode = "multithread" if _USE_MT else "single-thread"
    logger.info(
        "paged_shm memcpy: cores=%d numa=%d mem=%.0fGiB -> %s (%d threads, "
        "subchunk=%d KiB)",
        _CORES,
        _NUMA,
        _MEM / (1 << 30),
        mode,
        _COPY_THREADS,
        _SUBCHUNK_BYTES // 1024,
    )


_log_decision_once()


# ---------------------------------------------------------------------------
# Numba kernel (only defined when numba is importable)
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(nogil=True, parallel=True, cache=True)
    def _copy_segments_kernel(
        src: np.ndarray,          # 1-D uint8
        dst: np.ndarray,          # 1-D uint8
        src_offsets: np.ndarray,  # intp
        dst_offsets: np.ndarray,  # intp
        sizes: np.ndarray,        # intp
    ) -> None:
        n = sizes.shape[0]
        for i in prange(n):
            s = src_offsets[i]
            d = dst_offsets[i]
            k = sizes[i]
            dst[d : d + k] = src[s : s + k]


# ---------------------------------------------------------------------------
# Segment construction
# ---------------------------------------------------------------------------

def _pick_subchunk(block_size: int) -> int:
    """
    Segment size for the per-transfer decomposition.

    - With numba: ``gcd(block_size, SUBCHUNK_BYTES)``. This is always a
      power of two (since ``SUBCHUNK_BYTES`` is), divides ``block_size``
      evenly (no ragged tail), and stays as close to 8 KiB as possible.
      When ``block_size`` is itself a multiple of 8 KiB — the common case,
      e.g. the default 1 MiB block — every segment is exactly 8 KiB and
      every offset we compute inside both the flat and the contiguous
      buffer is 8 KiB aligned.
    - Without numba: one segment per block, to keep the Python loop
      shallow. A few hundred iterations is fine; tens of thousands is not.

    Note: this aligns the *offsets* we compute. The base addresses of the
    two buffers are outside our control (numpy ≥ 64 B, mmap ≥ 4 KiB),
    which is more than enough for peak memcpy bandwidth.
    """
    if _HAS_NUMBA:
        return math.gcd(block_size, _SUBCHUNK_BYTES)
    return block_size


def _build_offsets(
    blocks: Sequence[int],
    size: int,
    block_size: int,
    subchunk_bytes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a logical ``size``-byte transfer across ``blocks`` (each
    ``block_size`` bytes), further sliced into ``subchunk_bytes`` pieces
    that never cross a block boundary.

    Returns three ``intp`` arrays of equal length ``n_seg``:

      - ``contig_offsets[i]`` : byte offset inside the contiguous side
      - ``flat_offsets[i]``   : byte offset inside the flat paged side
      - ``seg_sizes[i]``      : number of bytes for this segment
    """
    n_full = size // block_size
    remainder = size - n_full * block_size
    n_used = n_full + (1 if remainder > 0 else 0)

    blocks_arr = np.asarray(blocks[:n_used], dtype=np.intp)

    # Effective bytes in each used block.
    eff = np.empty(n_used, dtype=np.intp)
    if n_full > 0:
        eff[:n_full] = block_size
    if remainder > 0:
        eff[n_full] = remainder

    # Sub-chunks per block.
    n_subs = (eff + subchunk_bytes - 1) // subchunk_bytes
    total = int(n_subs.sum())

    # Map each sub-chunk back to its block and its index within that block.
    block_of_sub = np.repeat(np.arange(n_used, dtype=np.intp), n_subs)

    block_start_sub = np.zeros(n_used, dtype=np.intp)
    if n_used > 1:
        np.cumsum(n_subs[:-1], out=block_start_sub[1:])
    j_in_block = np.arange(total, dtype=np.intp) - block_start_sub[block_of_sub]

    in_block_off = j_in_block * subchunk_bytes

    logical_base = block_of_sub * block_size
    physical_base = blocks_arr[block_of_sub] * block_size

    contig_offsets = logical_base + in_block_off
    flat_offsets = physical_base + in_block_off
    seg_sizes = np.minimum(eff[block_of_sub] - in_block_off, subchunk_bytes)

    return contig_offsets, flat_offsets, seg_sizes


# ---------------------------------------------------------------------------
# Serial numpy fallback
# ---------------------------------------------------------------------------

def _copy_segments_serial(
    src: np.ndarray,
    dst: np.ndarray,
    src_offsets: np.ndarray,
    dst_offsets: np.ndarray,
    sizes: np.ndarray,
) -> None:
    for i in range(sizes.shape[0]):
        s = int(src_offsets[i])
        d = int(dst_offsets[i])
        k = int(sizes[i])
        dst[d : d + k] = src[s : s + k]


# ---------------------------------------------------------------------------
# Thread dispatch
# ---------------------------------------------------------------------------

_last_threads: int | None = None


def _dispatch(n_threads: int | None) -> None:
    global _last_threads

    if not _HAS_NUMBA:
        return

    if n_threads is None:
        n_threads = _COPY_THREADS
    if n_threads < 1:
        raise ValueError(f"n_threads must be >= 1, got {n_threads}")
    if n_threads > _MAX_NUMBA_THREADS:
        n_threads = _MAX_NUMBA_THREADS

    if n_threads != _last_threads and get_num_threads() != n_threads:
        set_num_threads(n_threads)
    _last_threads = n_threads


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _check_args(contig: np.ndarray, flat: np.ndarray) -> None:
    if contig.dtype != np.uint8 or flat.dtype != np.uint8:
        raise TypeError("paged_shm memcpy requires uint8 arrays")
    if not contig.flags.c_contiguous:
        raise ValueError("contiguous side must be C-contiguous")
    if not flat.flags.c_contiguous:
        raise ValueError("flat view must be C-contiguous")


def copy_contig_to_blocks(
    src: np.ndarray,
    flat: np.ndarray,
    blocks: Sequence[int],
    block_size: int,
    *,
    n_threads: int | None = None,
) -> None:
    """
    Scatter ``src`` (``src.shape[0]`` bytes) into the given ``blocks`` of
    ``flat``.

    Args:
        src: contiguous 1-D ``uint8`` source; its length is the byte count.
        flat: 1-D ``uint8`` view of the paged segment
              (length ``n_block * block_size``).
        blocks: destination block indices.
        block_size: size of each block in bytes.
        n_threads: override the heuristic thread count; ``None`` uses
                   :func:`get_copy_threads`. Ignored without numba.
    """
    size = int(src.shape[0])
    if size == 0:
        return
    _check_args(src, flat)

    subchunk = _pick_subchunk(block_size)
    contig_off, flat_off, sizes = _build_offsets(
        blocks, size, block_size, subchunk
    )

    if _HAS_NUMBA:
        _dispatch(n_threads)
        _copy_segments_kernel(src, flat, contig_off, flat_off, sizes)
    else:
        _copy_segments_serial(src, flat, contig_off, flat_off, sizes)


def copy_blocks_to_contig(
    flat: np.ndarray,
    dst: np.ndarray,
    blocks: Sequence[int],
    block_size: int,
    *,
    n_threads: int | None = None,
) -> None:
    """
    Gather ``dst.shape[0]`` bytes from the given ``blocks`` of ``flat`` into
    ``dst``.

    Args:
        flat: 1-D ``uint8`` view of the paged segment.
        dst: contiguous 1-D ``uint8`` destination; its length is the byte count.
        blocks: source block indices.
        block_size: size of each block in bytes.
        n_threads: override the heuristic thread count; ``None`` uses
                   :func:`get_copy_threads`. Ignored without numba.
    """
    size = int(dst.shape[0])
    if size == 0:
        return
    _check_args(dst, flat)

    subchunk = _pick_subchunk(block_size)
    contig_off, flat_off, sizes = _build_offsets(
        blocks, size, block_size, subchunk
    )

    if _HAS_NUMBA:
        _dispatch(n_threads)
        _copy_segments_kernel(flat, dst, flat_off, contig_off, sizes)
    else:
        _copy_segments_serial(flat, dst, flat_off, contig_off, sizes)


# ---------------------------------------------------------------------------
# JIT warm-up
# ---------------------------------------------------------------------------

def warmup() -> None:
    """
    Force JIT compilation with tiny arrays so the first real copy is fast.

    Call this once during process startup (e.g. from
    ``PagedShmStorage.__init__``). With ``cache=True`` the compiled artifact
    is persisted to disk, so subsequent processes pay only the mmap cost.

    No-op when numba is not available.
    """
    if not _HAS_NUMBA:
        return

    dummy = np.zeros(64, dtype=np.uint8)
    one = np.zeros(1, dtype=np.intp)

    # Serial compilation.
    _dispatch(1)
    _copy_segments_kernel(dummy, dummy, one, one, one)

    # Parallel compilation (numba compiles the parallel variant of prange
    # under ``parallel=True``; this call makes sure it is cached).
    _dispatch(_COPY_THREADS)
    _copy_segments_kernel(dummy, dummy, one, one, one)


__all__ = [
    "copy_blocks_to_contig",
    "copy_contig_to_blocks",
    "get_copy_threads",
    "get_topology",
    "use_multithread",
    "warmup",
]