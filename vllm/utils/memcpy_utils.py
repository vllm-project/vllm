# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Server CPUs usually need several cores to saturate copy bandwidth, so we
use a numba ``parallel=True`` kernel that copies independent segments
concurrently. Desktop CPUs already saturate with a single core, so the
multi-threading heuristic stays off there.

Heuristic — enable multi-threading only when the host looks like a server:

  - more than one NUMA node, or
  - more than 32 logical cores, or
  - more than 500 GiB of physical memory.

Examples:
    >>> import numpy as np
    >>> src = np.arange(4096, dtype=np.uint8)
    >>> dst = np.empty_like(src)
    >>> memcpy_mt(src, dst, 4096)
    >>> np.array_equal(src, dst)
    True
"""

import os
from typing import Optional

import numpy as np

_GiB_BYTES = 1 << 30

_DEFAULT_NUMBA_THREADS = 64

# Normalize NUMBA_NUM_THREADS before importing numba, so an invalid
# user-provided value cannot break the import.
_raw_numba_threads = os.environ.get("NUMBA_NUM_THREADS")
if _raw_numba_threads is None:
    os.environ["NUMBA_NUM_THREADS"] = str(_DEFAULT_NUMBA_THREADS)
else:
    try:
        _parsed = int(_raw_numba_threads)
        if _parsed < 1:
            os.environ["NUMBA_NUM_THREADS"] = str(_DEFAULT_NUMBA_THREADS)
    except ValueError:
        os.environ["NUMBA_NUM_THREADS"] = str(_DEFAULT_NUMBA_THREADS)

try:
    from numba import get_num_threads, njit, prange, set_num_threads

    _HAS_NUMBA = True
    _MAX_NUMBA_THREADS = int(os.environ["NUMBA_NUM_THREADS"])
except ImportError:
    _HAS_NUMBA = False
    _MAX_NUMBA_THREADS = 1

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# 8 KiB is the knee of the sub-chunk sweep: bandwidth is already saturated,
# and per-task overhead is still small.
_SUBCHUNK_BYTES = 8 * 1024

# 8 threads is the plateau of the thread sweep. The heuristic below caps at
# this value; more threads only add scheduling jitter.
_MAX_COPY_THREADS = 8


# ---------------------------------------------------------------------------
# Topology detection (cheap, done once at import)
# ---------------------------------------------------------------------------

def _detect_logical_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
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


def _read_cgroup_memory_limit() -> Optional[int]:
    """Best-effort cgroup v1/v2 memory limit in bytes."""
    candidates = (
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    )
    for path in candidates:
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        # cgroup v1 often uses a huge sentinel for "unlimited".
        if 0 < limit < (1 << 60):
            return limit
    return None


def _detect_memory_bytes() -> int:
    host_bytes = 0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            host_bytes = int(pages) * int(page_size)
    except (ValueError, OSError, AttributeError):
        pass

    cgroup_limit = _read_cgroup_memory_limit()
    if host_bytes > 0 and cgroup_limit is not None:
        return min(host_bytes, cgroup_limit)
    if cgroup_limit is not None:
        return cgroup_limit
    return host_bytes


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
    # Huge memory: a host with >500 GiB RAM is essentially always a
    # multi-channel server, even if NUMA is disabled or cores are capped.
    if mem_bytes > 500 * _GiB_BYTES:
        return True
    return False


_CORES = _detect_logical_cores()
_NUMA = _detect_numa_nodes()
_MEM = _detect_memory_bytes()

# Without numba there is no thread pool to drive; always single-threaded.
# Also respect NUMBA_NUM_THREADS=1.
_USE_MT = (
    _HAS_NUMBA
    and _MAX_NUMBA_THREADS > 1
    and _auto_detect_mt(_CORES, _NUMA, _MEM)
)

_COPY_THREADS = (
    min(_CORES, _MAX_COPY_THREADS, _MAX_NUMBA_THREADS) if _USE_MT else 1
)


def use_multithread() -> bool:
    """Whether CPU copies use multi-threading (cached)."""
    return _USE_MT


def get_copy_threads() -> int:
    """Thread count used when ``max_copy_threads`` is not specified."""
    return _COPY_THREADS


def get_topology() -> tuple[int, int, int]:
    """``(logical_cores, numa_nodes, mem_bytes)`` for logging/diagnostics."""
    return _CORES, _NUMA, _MEM


def _log_decision_once() -> None:
    if not _HAS_NUMBA:
        logger.debug(
            "memcpy mt: numba unavailable -> single-thread numpy "
            "(cores=%d numa=%d mem=%.0fGiB)",
            _CORES,
            _NUMA,
            _MEM / _GiB_BYTES,
        )
        return
    mode = "multithread" if _USE_MT else "single-thread"
    logger.debug(
        "memcpy mt: cores=%d numa=%d mem=%.0fGiB -> %s (%d threads, "
        "subchunk=%d KiB)",
        _CORES,
        _NUMA,
        _MEM / _GiB_BYTES,
        mode,
        _COPY_THREADS,
        _SUBCHUNK_BYTES // 1024,
    )


_log_decision_once()


# ---------------------------------------------------------------------------
# Numba kernel
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(nogil=True, parallel=True, cache=True)
    def _copy_kernel(
        src: np.ndarray,  # 1-D uint8
        dst: np.ndarray,  # 1-D uint8
        size: int,
        chunk_bytes: int,
    ) -> None:
        n_subs = (size + chunk_bytes - 1) // chunk_bytes
        for i in prange(n_subs):
            s = i * chunk_bytes
            e = min(s + chunk_bytes, size)
            dst[s:e] = src[s:e]


# ---------------------------------------------------------------------------
# Thread dispatch
# ---------------------------------------------------------------------------

def _dispatch(n_threads: int) -> int:
    """Set the numba thread-pool size; return the previous value."""
    if n_threads < 1:
        raise ValueError(f"n_threads must be >= 1, got {n_threads}")
    if not _HAS_NUMBA:
        return n_threads
    n_threads = min(n_threads, _MAX_NUMBA_THREADS)
    prev = get_num_threads()
    if prev != n_threads:
        set_num_threads(n_threads)
    return prev


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_u8(arr: np.ndarray, *, require_contiguous: bool) -> np.ndarray:
    """Return a 1-D ``uint8`` view of ``arr``."""
    a = np.asarray(arr)
    if not a.flags.c_contiguous:
        if require_contiguous:
            raise ValueError(
                "dst must be C-contiguous for zero-copy flat byte access"
            )
        a = np.ascontiguousarray(a)
    if a.dtype == np.uint8 and a.ndim == 1:
        return a
    return a.reshape(-1).view(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def memcpy_mt(
    src: np.ndarray,
    dst: np.ndarray,
    size: int,
    *,
    sub_chunk_bytes: int = _SUBCHUNK_BYTES,
    max_copy_threads: int = _MAX_COPY_THREADS,
) -> None:
    if sub_chunk_bytes <= 0:
        raise ValueError(
            f"sub_chunk_bytes must be positive, got {sub_chunk_bytes}"
        )
    if max_copy_threads < 1:
        raise ValueError(
            f"max_copy_threads must be >= 1, got {max_copy_threads}"
        )

    size = int(size)
    if size <= 0:
        return

    src_flat = _flatten_u8(src, require_contiguous=False)
    dst_flat = _flatten_u8(dst, require_contiguous=True)

    if size > src_flat.size or size > dst_flat.size:
        raise ValueError(
            "size exceeds buffer capacity: "
            f"size={size}, |src|={src_flat.size}, |dst|={dst_flat.size}"
        )

    # Tiny copy: numpy slice assignment beats numba parallel dispatch.
    if not _HAS_NUMBA or size <= sub_chunk_bytes:
        dst_flat[:size] = src_flat[:size]
        return

    n_threads = min(_COPY_THREADS, max_copy_threads) if _USE_MT else 1
    if n_threads == 1:
        dst_flat[:size] = src_flat[:size]
        return

    prev_threads = _dispatch(n_threads)
    try:
        _copy_kernel(src_flat, dst_flat, size, sub_chunk_bytes)
    finally:
        # Runs after the parallel region: safe to shrink the pool back.
        if get_num_threads() != prev_threads:
            set_num_threads(prev_threads)