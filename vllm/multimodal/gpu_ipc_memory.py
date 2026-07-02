# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission control for frontend GPU-side multimodal work.

When multimodal media is decoded on the GPU in the API-server (frontend)
process, the decoded buffers compete for the same device memory that the
engine reserves for weights, activations, and the KV cache. To keep the
frontend's GPU usage within a sequestered budget (see
``MultiModalConfig.mm_ipc_gpu_memory_gb``), decode paths acquire the number of
bytes they need from a process-global :class:`MultiModalGPUMemoryPool` before
allocating on the device and release them once the device memory is freed.

The pool is a simple byte-counting semaphore: ``acquire`` blocks until enough
budget is free, so concurrent requests serialize rather than oversubscribe the
GPU. It lives only in the frontend process; the engine carves the matching
amount out of its KV-cache budget so the headroom physically exists.
"""

import threading

from vllm.logger import init_logger
from vllm.utils.mem_constants import GiB_bytes

logger = init_logger(__name__)


class MultiModalGPUMemoryLease:
    """A handle for bytes acquired from a :class:`MultiModalGPUMemoryPool`.

    Releasing is idempotent and the lease doubles as a context manager so the
    budget is returned even if the decode raises.
    """

    def __init__(self, pool: "MultiModalGPUMemoryPool", lease_id: int, nbytes: int):
        self.lease_id = lease_id
        self.nbytes = nbytes
        self._pool = pool

    def release(self) -> None:
        self._pool._release(self)

    def __enter__(self) -> "MultiModalGPUMemoryLease":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


class MultiModalGPUMemoryPool:
    """Blocking byte-counting semaphore for frontend GPU multimodal memory.

    Thread-safe in both directions: ``acquire`` (blocking) and ``release`` are
    typically called from the renderer's multimodal executor threads.
    """

    def __init__(self, total_bytes: int):
        if total_bytes <= 0:
            raise ValueError(f"total_bytes must be positive, got {total_bytes}")
        self._total_bytes = total_bytes
        self._available = total_bytes
        self._cond = threading.Condition()
        self._next_lease_id = 0
        # Outstanding lease ids, so a double release is a no-op.
        self._outstanding: set[int] = set()

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def available_bytes(self) -> int:
        with self._cond:
            return self._available

    def acquire(self, nbytes: int) -> MultiModalGPUMemoryLease:
        """Reserve ``nbytes``, blocking until that much budget is free.

        Raises ``ValueError`` if ``nbytes`` exceeds the pool's total capacity,
        since such a request could never be satisfied.
        """
        if nbytes < 0:
            raise ValueError(f"Cannot acquire negative bytes: {nbytes}")
        if nbytes > self._total_bytes:
            raise ValueError(
                f"Multimodal GPU decode requested {nbytes} bytes, which exceeds "
                f"the total pool size of {self._total_bytes} bytes. Increase "
                f"--mm-ipc-gpu-memory-gb or reduce the multimodal input size."
            )
        with self._cond:
            while self._available < nbytes:
                self._cond.wait()
            self._available -= nbytes
            lease_id = self._next_lease_id
            self._next_lease_id += 1
            self._outstanding.add(lease_id)
        return MultiModalGPUMemoryLease(self, lease_id, nbytes)

    def _release(self, lease: MultiModalGPUMemoryLease) -> None:
        with self._cond:
            if lease.lease_id not in self._outstanding:
                # Already released — idempotent.
                return
            self._outstanding.discard(lease.lease_id)
            self._available += lease.nbytes
            self._cond.notify_all()


_GLOBAL_POOL: MultiModalGPUMemoryPool | None = None


def set_mm_gpu_ipc_pool(pool: MultiModalGPUMemoryPool | None) -> None:
    """Install the process-global pool (frontend process only)."""
    global _GLOBAL_POOL
    _GLOBAL_POOL = pool


def get_mm_gpu_ipc_pool() -> MultiModalGPUMemoryPool | None:
    """Return the process-global pool, or ``None`` when gating is disabled."""
    return _GLOBAL_POOL


def maybe_init_mm_gpu_ipc_pool(
    mm_ipc_gpu_memory_gb: float,
    api_process_count: int = 1,
) -> MultiModalGPUMemoryPool | None:
    """Create and install the global pool from the configured GiB budget.

    Returns ``None`` (and leaves gating disabled) when the budget is 0. When
    multiple API-server processes share one engine, each process gets an equal
    slice of the user-provided frontend budget.
    """
    if mm_ipc_gpu_memory_gb <= 0:
        set_mm_gpu_ipc_pool(None)
        return None
    if api_process_count <= 0:
        raise ValueError(f"api_process_count must be positive, got {api_process_count}")
    total_bytes = int(mm_ipc_gpu_memory_gb * GiB_bytes) // api_process_count
    pool = MultiModalGPUMemoryPool(total_bytes)
    set_mm_gpu_ipc_pool(pool)
    logger.info(
        "Initialized multimodal GPU IPC memory pool with %d bytes for this API "
        "process (%.2f GiB total budget across %d API process(es)).",
        total_bytes,
        mm_ipc_gpu_memory_gb,
        api_process_count,
    )
    return pool
