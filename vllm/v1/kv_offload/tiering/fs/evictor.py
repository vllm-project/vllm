# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Evictors for the fs tier where the storage is not shared between vllm
instances.

An Evictor controls whether a store job is allowed to proceed and reconciles
the in-memory view of which keys are resident on disk after each job completes.

Three concrete implementations are provided:

  NoOpEvictor       - Unbounded. Lookup delegates to the filesystem and all
                      lifecycle hooks are no-ops.  Use when no storage cap is
                      configured.

  LRUEvictor        - Bounded. Evicts the least-recently-used blocks to make
                      room for new stores.  May briefly overshoot the cap while
                      multiple stores are in-flight; eventual consistency is
                      guaranteed via complete_store.

  FailOnFullEvictor - Bounded. Refuses new store jobs (returns False from
                      prepare_store) when the cap is exhausted instead of
                      evicting existing blocks.

Lifecycle hooks called by FileSystemTierManager:
  prepare_store  - called before enqueueing a store task.
  complete_store - called after a store task finishes (success or failure).
  prepare_load   - called before enqueueing a load task.
  complete_load  - called after a load task finishes.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Literal

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import JobMetadata

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EvictorBase(ABC):
    """
    Common interface for all fs-tier evictors.

    Every method is called on the scheduler thread and must not block.
    """

    @abstractmethod
    def lookup(self, key: OffloadKey) -> bool:
        """Return True iff key is resident and ready to be loaded."""

    @abstractmethod
    def prepare_store(self, job_metadata: JobMetadata) -> bool:
        """
        Gate a store job before it is enqueued.

        Returns True if the job should proceed; False if it should be skipped
        (the caller will surface a failed JobResult without touching disk).
        """

    @abstractmethod
    def prepare_load(self, job_metadata: JobMetadata) -> None:
        """
        Called just before a load job is enqueued.

        May update ref-counts or recency information.
        """

    @abstractmethod
    def complete_store(self, job_metadata: JobMetadata) -> None:
        """
        Called once after a store job finishes (success or failure).

        Reconciles the in-memory state with what actually landed on disk.
        """

    @abstractmethod
    def complete_load(self, job_metadata: JobMetadata) -> None:
        """Called once after a load job finishes."""


# ---------------------------------------------------------------------------
# No-op evictor (unbounded storage)
# ---------------------------------------------------------------------------


class NoOpEvictor(EvictorBase):
    """
    Unbounded evictor used when no storage cap is configured.

    Lookup is delegated to the filesystem.  All lifecycle hooks are no-ops so
    there is zero overhead for the common uncapped case.
    """

    def __init__(self, file_mapper: FileMapper) -> None:
        self.file_mapper = file_mapper

    def lookup(self, key: OffloadKey) -> bool:
        return os.path.exists(self.file_mapper.get_file_name(key))

    def prepare_store(self, job_metadata: JobMetadata) -> bool:
        return True

    def prepare_load(self, job_metadata: JobMetadata) -> None:
        pass

    def complete_store(self, job_metadata: JobMetadata) -> None:
        pass

    def complete_load(self, job_metadata: JobMetadata) -> None:
        pass


# ---------------------------------------------------------------------------
# Shared base for capacity-limited evictors
# ---------------------------------------------------------------------------


class _BoundedEvictor(EvictorBase):
    """
    Abstract base for evictors that enforce a storage cap via a CachePolicy.

    Subclasses only need to implement _acquire_slots, which decides how (or
    whether) to make room for a new set of keys.

    Shared responsibilities handled here:
      - policy-backed lookup and ref-count tracking for in-flight loads
      - complete_store reconciliation (file-existence vs. policy state)
      - prepare_store skeleton (filter already-tracked keys, delegate to
        _acquire_slots for the remainder)
    """

    def __init__(
        self, file_mapper: FileMapper, policy: LRUCachePolicy, max_files: int
    ) -> None:
        logger.warning(
            "%s should only be used for testing, not in production.",
            type(self).__name__,
        )
        self.file_mapper = file_mapper
        self.policy = policy
        self.max_files = max_files

    # ------------------------------------------------------------------
    # Shared factory helper
    # ------------------------------------------------------------------

    @classmethod
    def make(
        cls,
        file_mapper: FileMapper,
        file_size_bytes: int,
        max_storage_size_gb: float,
    ) -> "_BoundedEvictor":
        """Construct from a storage budget expressed in gigabytes."""
        max_storage_bytes = int(max_storage_size_gb * 1024**3)
        max_files = max_storage_bytes // file_size_bytes
        assert max_files > 0, (
            f"Configured {max_storage_size_gb=} GB is too small for "
            f"{file_size_bytes=}-byte files"
        )
        return cls(file_mapper, LRUCachePolicy(max_files), max_files)

    # ------------------------------------------------------------------
    # EvictorBase interface
    # ------------------------------------------------------------------

    def lookup(self, key: OffloadKey) -> bool:
        block = self.policy.get(key)
        return block is not None and block.is_ready

    def prepare_load(self, job_metadata: JobMetadata) -> None:
        self.policy.touch(job_metadata.keys)
        for key in job_metadata.keys:
            block = self.policy.get(key)
            assert block is not None, f"Untracked file found for key {key!r}"
            assert block.is_ready, f"Block for {key!r} not yet ready"
            block.ref_cnt += 1

    def complete_load(self, job_metadata: JobMetadata) -> None:
        for key in job_metadata.keys:
            block = self.policy.get(key)
            assert block is not None, f"Untracked file found for key {key!r}"
            assert block.ref_cnt > 0
            block.ref_cnt -= 1

    def prepare_store(self, job_metadata: JobMetadata) -> bool:
        keys_to_store = [k for k in job_metadata.keys if self.policy.get(k) is None]

        if not keys_to_store:
            return True

        return self._acquire_slots(keys_to_store, protected=set(job_metadata.keys))

    def _complete_store(self, key: OffloadKey):
        # Determine store complete by checking file presence.
        # We allow multiple in-flight stores reconcile them here.

        def _maybe_materialize_slot(mkey: OffloadKey) -> bool:
            num_free_slots = self.max_files - len(self.policy.blocks)
            if num_free_slots > 0:
                block_status = BlockStatus(block_id=0)
                block_status.ref_cnt = 0
                self.policy.insert(mkey, block_status)
                return True
            return False

        key_file = self.file_mapper.get_file_name(key)
        key_file_exists = os.path.exists(key_file)
        block_meta = self.policy.get(key)

        if key_file_exists:
            if block_meta is not None:
                # Protect against multiple in-flight stores.
                if block_meta.ref_cnt == -1:
                    # Only the first reported store succeeds.
                    block_meta.ref_cnt = 0
            else:
                # Maybe a previous store-job failed and this job succeeded.
                # Try updating the self.policy.
                if not _maybe_materialize_slot(key):
                    # Cannot materialize. This block was likely evicted and
                    # a scheduled store job recreated it.
                    # Remove the file for consistency and to respect the FS size
                    # requirements.
                    try:
                        os.remove(key_file)
                        logger.warning("Removed a hash_file %s", key_file)
                    except Exception:
                        logger.warning("Cannot remove stray hash file %s", key_file)
            return

        # File does not exist
        if block_meta is not None:
            # Failed store job
            self.policy.remove(key)

    def complete_store(self, job_metadata: JobMetadata):
        for key in job_metadata.keys:
            self._complete_store(key)

    @abstractmethod
    def _acquire_slots(
        self, keys_to_store: list[OffloadKey], protected: set[OffloadKey]
    ) -> bool:
        """
        Reserve capacity for keys_to_store and insert them into the policy.

        keys_to_store  - keys that are not yet tracked by the policy.
        protected      - keys that must not be evicted (the full job key set).

        Returns True if the slots were acquired and the keys inserted;
        False if no space could be made (the store job will be skipped).
        """


# ---------------------------------------------------------------------------
# LRU evictor
# ---------------------------------------------------------------------------


class LRUEvictor(_BoundedEvictor):
    """
    Bounded evictor that evicts the least-recently-used blocks to make room.

    Stores may transiently overshoot the cap when multiple jobs are in-flight;
    complete_store restores consistency once each job settles.
    """

    def __init__(
        self, file_mapper: FileMapper, policy: LRUCachePolicy, max_files: int
    ) -> None:
        super().__init__(file_mapper, policy, max_files)
        logger.warning(
            "LRUEvictor: storage cap may be briefly exceeded while "
            "in-flight stores are outstanding; eventual consistency is guaranteed."
        )

    def _acquire_slots(
        self, keys_to_store: list[OffloadKey], protected: set[OffloadKey]
    ) -> bool:
        need = len(keys_to_store)
        free = self.max_files - len(self.policy.blocks)
        num_to_evict = max(0, need - free)

        if num_to_evict > 0:
            evicted = self.policy.evict(num_to_evict, protected=protected)
            if evicted is None:
                return False
            for evicted_key, _ in evicted:
                os.remove(self.file_mapper.get_file_name(evicted_key))

        for key in keys_to_store:
            self.policy.insert(key, BlockStatus(block_id=0))
        return True


# ---------------------------------------------------------------------------
# Fail-on-full evictor
# ---------------------------------------------------------------------------


class FailOnFullEvictor(_BoundedEvictor):
    """
    Bounded evictor that rejects new store jobs when the cap is exhausted.

    Unlike LRUEvictor, no existing blocks are ever evicted to make room.
    Stores for keys that do not fit are skipped immediately with a failed
    JobResult, leaving all currently resident blocks untouched.
    """

    def _acquire_slots(
        self, keys_to_store: list[OffloadKey], protected: set[OffloadKey]
    ) -> bool:
        free = self.max_files - len(self.policy.blocks)
        if len(keys_to_store) > free:
            return False
        for key in keys_to_store:
            self.policy.insert(key, BlockStatus(block_id=0))
        return True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_EVICTOR_CLASSES: dict[str, type[_BoundedEvictor]] = {
    "lru": LRUEvictor,
    "fail_on_full": FailOnFullEvictor,
}


def make_evictor(
    file_mapper: FileMapper,
    file_size_bytes: int,
    evictor_args: dict[str, Any] | None,
) -> EvictorBase:
    """
    Construct an evictor from a plain args dict.

    Pass None (or omit evictor_args) to get an unbounded NoOpEvictor.

    Recognised keys in evictor_args:
      max_storage_size_gb (float) - storage cap in GB.
      evictor_type        (str)   - "lru" (default) or "fail_on_full".

    Example::

        evictor = make_evictor(
            file_mapper,
            block_size,
            {"max_storage_size_gb": 100.0, "evictor_type": "lru"},
        )
    """
    if evictor_args is None:
        return NoOpEvictor(file_mapper)

    max_gb = evictor_args.get("max_storage_size_gb")
    if max_gb is None:
        return NoOpEvictor(file_mapper)

    evictor_type: Literal["lru", "fail_on_full"] = evictor_args.get(
        "evictor_type", "lru"
    )
    if evictor_type not in _EVICTOR_CLASSES:
        raise ValueError(
            f"Unknown evictor_type {evictor_type!r}. "
            f"Choose from: {list(_EVICTOR_CLASSES)}"
        )
    return _EVICTOR_CLASSES[evictor_type].make(
        file_mapper,
        file_size_bytes=file_size_bytes,
        max_storage_size_gb=float(max_gb),
    )
