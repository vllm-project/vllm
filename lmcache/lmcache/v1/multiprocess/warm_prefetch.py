# SPDX-License-Identifier: Apache-2.0
"""Warm-prefetch job table for the MP server.

A *warm* prefetch loads a caller-supplied set of keys from L2 into L1 and leaves
them resident, retained, and unpinned, so a subsequent real lookup hits L1. It
loads every requested key not already in L1, and pins nothing -- there is no
downstream reader to pin for.

The table tracks in-flight warm prefetches: ``submit`` starts one and returns an
opaque request id; ``poll`` reports its status (pending, then completed) and
drops it once the load finishes. Status is observed reactively by the caller --
there is no background polling -- and these calls do not block. A job whose
status is never polled to completion simply lingers until a later cleanup drops
it; since nothing is pinned, no L1 is held.
"""

# Standard
from dataclasses import dataclass
import threading
import uuid

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchMode,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.distributed.storage_manager import StorageManager

logger = init_logger(__name__)

# Job states reported by :meth:`WarmPrefetchJobs.poll`.
PENDING = "pending"
COMPLETED = "completed"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class WarmStatus:
    """Outcome of a status poll.

    Attributes:
        state: ``"pending"`` (load still running), ``"completed"`` (loaded), or
            ``"unknown"`` (no such request id -- already completed-and-consumed,
            or never submitted).
        found_keys: Keys loaded into L1 (only meaningful when ``completed``).
        total_keys: Keys originally requested (only meaningful when
            ``completed``).
    """

    state: str
    found_keys: int = 0
    total_keys: int = 0


class WarmPrefetchJobs:
    """Thread-safe table of in-flight warm-prefetch jobs (``request_id ->
    handle``).

    ``submit`` starts a no-lock retain prefetch and returns an opaque request
    id; ``poll`` reports progress and, on completion, drops the job
    (exactly-once: a subsequent poll for the same id returns ``UNKNOWN``).
    Since the warm holds no read lock, there is nothing to release on
    completion.
    """

    def __init__(self) -> None:
        """Initialize an empty table."""
        self._lock = threading.Lock()
        self._jobs: dict[str, PrefetchHandle] = {}

    def submit(
        self,
        storage_manager: StorageManager,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> str:
        """Start a no-lock retain prefetch and register its handle.

        Args:
            storage_manager: The MP server's storage manager.
            keys: Object keys to load from L2 into L1.
            layout_desc: Memory layout for the L1 write buffers.

        Returns:
            An opaque request id to pass to :meth:`poll`.
        """
        handle = storage_manager.submit_prefetch_task(
            PrefetchRequestSpec(
                keys=keys,
                layout_desc=layout_desc,
                mode=PrefetchMode.WARM,
                policy=TrimPolicy.SPARSE,
            )
        )
        request_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[request_id] = handle
        return request_id

    def poll(
        self,
        storage_manager: StorageManager,
        request_id: str,
    ) -> WarmStatus:
        """Report a job's status; drop it once the load completes.

        Args:
            storage_manager: The MP server's storage manager.
            request_id: The id returned by :meth:`submit`.

        Returns:
            A :class:`WarmStatus`. The first poll that observes completion drops
            the job (and pops the controller's result bookkeeping), so later
            polls for the same id return ``UNKNOWN``.
        """
        with self._lock:
            handle = self._jobs.get(request_id)
        if handle is None:
            return WarmStatus(state=UNKNOWN)

        found = storage_manager.query_prefetch_status(handle)
        if found is None:
            return WarmStatus(state=PENDING)

        with self._lock:
            self._jobs.pop(request_id, None)
        found_keys = found.popcount()
        logger.info(
            "Warm prefetch %s completed: %d/%d keys loaded into L1",
            request_id,
            found_keys,
            handle.total_requested_keys,
        )
        return WarmStatus(
            state=COMPLETED,
            found_keys=found_keys,
            total_keys=handle.total_requested_keys,
        )
