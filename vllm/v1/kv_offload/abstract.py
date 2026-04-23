# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OffloadingManager class for managing KV data offloading in vLLM v1

This class runs in the scheduler, tracks which blocks are offloaded
and their address.

The class provides the following primitives:
    lookup() - check whether a single block is offloaded and ready.
    prepare_load() - prepare given blocks to be read.
        The given blocks will be protected from eviction.
        This function returns a LoadSpec which encapsulates
        information required for performing the load.
    touch() - marks the give blocks as recently used. Can be used
        to track block's LRU. This function is separated from the
        prepare_load function to allow setting block recency even
        for blocks which do not need reading from the cache, such as
        blocks that are cached by the GPU prefix cache.
    complete_load() - mark blocks which were previously prepared to be
        loaded as done loading. This is to re-allow their eviction.
    prepare_store() - prepare the given blocks to be written.
        Returns a StoreSpec encapsulating offloading information,
        as well as a list of blocks that were evicted as a result.
    complete_store() - marks a previous store as completed.
        Following this call, the given blocks will become loadable.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, NewType

# `OffloadKey` identifies an offloaded block. It combines a block hash with
# its KV cache group index, encoded as raw bytes to avoid tuple GC overhead.
# Use the helper functions below to construct / decompose keys.
OffloadKey = NewType("OffloadKey", bytes)


def make_offload_key(block_hash: bytes, group_idx: int) -> OffloadKey:
    """Pack a block hash and group index into an `OffloadKey`."""
    return OffloadKey(block_hash + group_idx.to_bytes(4, "big", signed=False))


def get_offload_block_hash(key: OffloadKey) -> bytes:
    """Extract the block hash from an `OffloadKey`."""
    return key[:-4]


def get_offload_group_idx(key: OffloadKey) -> int:
    """Extract the group index from an `OffloadKey`."""
    return int.from_bytes(key[-4:], "big", signed=False)


@dataclass
class ReqContext:
    kv_transfer_params: dict[str, Any] | None = None


# Type alias for job IDs used in async transfer tracking
JobId = int


class LoadStoreSpec(ABC):
    """
    Abstract metadata that encapsulates information allowing a worker
    to load, and optionally also to store, blocks of KV data.
    """

    @staticmethod
    @abstractmethod
    def medium() -> str:
        """
        Returns a string representation of the medium type
        this store/load targets.
        """
        pass


@dataclass
class PrepareStoreOutput:
    keys_to_store: list[OffloadKey]
    store_spec: LoadStoreSpec
    evicted_keys: list[OffloadKey]


@dataclass
class OffloadingEvent:
    keys: list[OffloadKey]
    medium: str
    # True if blocks are removed, False if stored
    removed: bool


@dataclass
class JobMetadata:
    """Metadata for an in-flight async transfer job."""

    job_id: JobId
    keys: list[OffloadKey]
    spec: LoadStoreSpec
    req_context: ReqContext = field(default_factory=ReqContext)


@dataclass
class JobResult:
    """Result of an async transfer job (successful or failed)."""

    job_id: JobId
    success: bool


class OffloadingManager(ABC):
    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Checks whether a single block is offloaded and ready to be read.

        Args:
            key: the key identifying the block to lookup.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            True if the block is offloaded and ready, False if not,
            or None if the lookup should be retried later.
            Returning None will delay the request handling by the vLLM
            scheduler.
        """
        pass

    @abstractmethod
    def prepare_load(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        """
        Prepare the given blocks to be read.
        The given blocks will be protected from eviction until
        complete_load is called.
        It assumes all given blocks are offloaded.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            A LoadStoreSpec that can be used by a worker to locate and load
            the actual offloaded KV data.
        """
        pass

    def touch(self, keys: Iterable[OffloadKey]):
        """
        Mark the given blocks as recently used.
        This could in practice mean moving them to the end of an LRU list.

        Args:
            keys: the keys identifying the blocks.
        """
        return

    def complete_load(self, keys: Iterable[OffloadKey]):
        """
        Marks previous blocks that were prepared to load as done loading.

        Args:
            keys: the keys identifying the blocks.
        """
        return

    @abstractmethod
    def prepare_store(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        """
        Prepare the given blocks to be offloaded.
        The given blocks will be protected from eviction until
        complete_store is called.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            A PrepareStoreOutput indicating which blocks need storing,
            where to store them (LoadStoreSpec), and list of blocks that
            were evicted as a result.
            None is returned if the blocks cannot be stored.
        """
        pass

    def complete_store(self, keys: Iterable[OffloadKey], success: bool = True):
        """
        Marks blocks which were previously prepared to be stored, as stored.
        Following this call, the blocks become loadable.
        If success is False, blocks that were not marked as stored will be
        removed.

        Args:
            keys: the keys identifying the blocks.
            success: whether the blocks were stored successfully.
        """
        return

    def take_events(self) -> Iterable[OffloadingEvent]:
        """
        Take the offloading events from the manager.

        Yields:
            New OffloadingEvents collected since the last call.
        """
        return ()

    def shutdown(self) -> None:
        """Shutdown the manager and release any resources."""
        return


class SecondaryTierManager(ABC):
    """
    Abstract interface for managing a single non-primary offloading tier.

    Secondary tiers cannot directly access GPU memory. All data transfers
    must go through the primary tier (implemented as CPU in current version):
      - Store: GPU → primary → secondary  (cascade)
      - Load:  secondary → primary → GPU  (promotion)

    IMPORTANT: All methods run in the Scheduler process and must be
    lightweight and non-blocking. submit_load() and submit_store() submit
    async jobs; get_finished() polls for completion.
    """

    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a block exists in this secondary tier.

        Args:
            key: Offload key to look up.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            True if the block is present and ready,
            False if not found,
            or None if the block is being transferred (retry later).
        """
        pass

    @abstractmethod
    def submit_store(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to store blocks from the primary tier to this
        secondary tier.

        This method is lightweight: it allocates metadata and submits the
        transfer job, but does NOT perform the actual data transfer on the
        calling thread.

        The caller (TieringOffloadingManager) must have already called
        primary.prepare_read(keys) to obtain job_metadata.spec and
        to increment ref_cnt on those blocks. ref_cnt will be decremented
        when get_finished() reports this job_id as complete and
        primary.unprepare_read() is called.

        This method is responsible for:
          1. Filtering out blocks already present in this secondary tier
          2. Evicting blocks from this secondary tier if needed (secondary
             tiers are responsible for their own evictions)
          3. Allocating space in this secondary tier
          4. Submitting the async transfer: primary → secondary

        Args:
            job_metadata: Job metadata including job_id, keys, and
                          spec for reading blocks from the primary tier
                          (obtained via primary.prepare_read()).
                          spec is a CPULoadStoreSpec with block_ids.
        """
        pass

    @abstractmethod
    def submit_load(self, job_metadata: JobMetadata) -> None:
        """
        Submit an async job to load blocks from this secondary tier to the
        primary tier.

        This method is lightweight: it marks blocks as in-flight and submits
        the transfer job, but does NOT perform the actual data transfer on
        the calling thread.

        The caller (TieringOffloadingManager) must have already called
        primary.prepare_write(keys) to obtain job_metadata.spec and
        to allocate space in the primary tier. When get_finished() reports
        this job_id as complete, primary.complete_write() is called to make
        the blocks available for GPU loads.

        Args:
            job_metadata: Job metadata including job_id, keys, and
                          spec for writing blocks into the primary tier
                          (obtained via primary.prepare_write()).
                          spec is a CPULoadStoreSpec with block_ids.
        """
        pass

    @abstractmethod
    def get_finished(self) -> Iterable[JobResult]:
        """
        Poll for finished async jobs (both loads and stores).

        This is the mechanism by which the TieringOffloadingManager learns
        that a transfer has finished and can:
          - Call primary.unprepare_read() to decrement ref_cnt (for stores)
          - Call primary.complete_write() to make blocks loadable (for loads)

        Returns:
            Iterable of JobResult objects for all jobs that have
            finished since the last call.
        """
        pass

    def set_primary_view(self, view: memoryview) -> None:
        """
        Provide a long-lived memoryview of the primary-tier CPU tensor.

        Called once by TieringOffloadingManager during initialisation.
        Override to store the view for use in `submit_store` and `submit_load`.
        Use `view.strides[0]` to obtain the byte stride between block slots.

        Args:
            view: Memoryview of the primary tier's CPU KV cache tensor.
        """
        return

    def touch(self, keys: Iterable[OffloadKey]):
        """
        Mark blocks as recently used for eviction policy.

        Args:
            keys: Offload keys to mark as recently used.
        """
        return

    @abstractmethod
    def get_tier_name(self) -> str:
        """
        Get the name of this tier (e.g., "Storage", "Network").

        Returns:
            Tier name string.
        """
        pass
