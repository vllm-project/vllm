# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingManager: Multi-tier KV cache offloading orchestrator.

This manager coordinates between a primary tier (with GPU access, currently
CPU-based) and zero or more secondary tiers (Storage, Network, etc.) to
provide hierarchical KV cache offloading.

Key Design Principles:
1. Always offload to all tiers — When a block is stored to the primary tier,
   it is cascaded to ALL secondary tiers
2. Primary tier is the gateway — Only the primary tier can directly access
   GPU memory (currently implemented using CPU memory)
3. Staged promotion — Blocks in secondary tiers must be promoted to the
   primary tier before GPU can access them
4. Transparent retry mechanism — Return None from lookup() to signal
   "data is being promoted, try later"
5. ref_cnt as eviction protection — primary.prepare_read() increments ref_cnt,
   protecting blocks from eviction until complete_read() is called
"""

from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import (
    JobId,
    JobMetadata,
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)


class CPUPrimaryTierOffloadingManager(CPUOffloadingManager):
    """CPUOffloadingManager with a primary/secondary transfer interface.

    The inherited prepare_store/complete_store/prepare_load/complete_load are the
    GPU-facing OffloadingManager interface. These aliases expose the same operations
    from the secondary tier perspective, where read/write refers to secondary
    accessing primary. This avoids confusion when reading TieringOffloadingManager
    code (e.g. calling prepare_load inside a cascade/store path would be misleading).
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: str = "lru",
        enable_events: bool = False,
        mmap_region: SharedOffloadRegion | None = None,
    ):
        super().__init__(
            num_blocks=num_blocks,
            cache_policy=cache_policy,  # type: ignore[arg-type]
            enable_events=enable_events,
        )
        self._mmap_region = mmap_region

    def prepare_write(self, keys, req_context: ReqContext) -> PrepareStoreOutput | None:
        """Allocate space in primary for a secondary->primary write (promotion)."""
        return self.prepare_store(keys, req_context)

    def complete_write(self, keys, success: bool = True) -> None:
        """Finalize secondary->primary write, making blocks available."""
        self.complete_store(keys, success)

    def prepare_read(self, keys, req_context: ReqContext) -> LoadStoreSpec:
        """Protect primary blocks for a primary->secondary read (cascade),
        incrementing ref_cnt."""
        return self.prepare_load(keys, req_context)

    def complete_read(self, keys) -> None:
        """Release protection after primary->secondary read completes,
        decrementing ref_cnt."""
        self.complete_load(keys)

    def get_primary_kv_tensor(self) -> torch.Tensor:
        """
        Get the primary tier's KV cache tensor.

        Returns a 2-D int8 tensor of shape (num_blocks, row_stride_bytes)
        backed by the SharedOffloadRegion mmap, where row_stride_bytes =
        cpu_page_size * world_size.  Secondary tiers address block b as
        view[b], and view.strides[0] gives the per-block byte stride.

        Returns:
            2-D int8 CPU tensor of shape (num_blocks, row_stride_bytes).
        """
        assert self._mmap_region is not None, (
            "mmap_region must be provided to CPUPrimaryTierOffloadingManager"
        )
        return self._mmap_region._base.view(
            self._mmap_region.num_blocks, self._mmap_region._row_stride
        )


class TieringOffloadingManager(OffloadingManager):
    """
    Orchestrates multi-tier KV cache offloading.

    This manager coordinates between a primary tier (with GPU access, currently
    CPU-based) and zero or more secondary tiers (Storage, Network, etc.) to
    provide hierarchical KV cache offloading.

    Key internal state:
      - Minimal state tracking; relies on secondary tiers to report completion
        via get_finished()
      - Secondary tiers return JobResult objects containing all necessary
        information
      - job_id_counter: monotonically increasing counter for job IDs
    """

    def __init__(
        self,
        primary_tier: CPUPrimaryTierOffloadingManager,
        secondary_tiers: list[SecondaryTierManager] | None = None,
        enable_events: bool = False,
    ):
        """
        Initialize the TieringOffloadingManager.

        Args:
            primary_tier: The primary tier manager (CPU-based).
            secondary_tiers: List of secondary tier managers (e.g., Storage,
                            Network). Can be None or empty list.
            enable_events: Whether to track offloading events
        """
        self.primary_tier: CPUPrimaryTierOffloadingManager = primary_tier
        self.secondary_tiers = secondary_tiers or []

        self._job_id_counter: int = 0
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

        # Job tracking: maps job_id to metadata for each transfer direction
        # Store jobs: primary → secondary transfers
        self._store_jobs: dict[JobId, JobMetadata] = {}
        # Load jobs: secondary → primary transfers (promotions)
        self._load_jobs: dict[JobId, JobMetadata] = {}

        # Wire each secondary tier with a long-lived memoryview of the primary
        # CPU tensor (one independent view per tier). Views are stored so they
        # can be released on shutdown().
        self._secondary_views: list[memoryview] = []
        cpu_tensor = primary_tier.get_primary_kv_tensor()
        for tier in self.secondary_tiers:
            view = memoryview(cpu_tensor.numpy())
            self._secondary_views.append(view)
            tier.set_primary_view(view)

    def _next_job_id(self) -> JobId:
        """Generate a unique job ID for async transfer tracking."""
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id

    def _process_finished_jobs(self):
        """
        Poll all secondary tiers for completed jobs and update state accordingly.

        This method:
        1. Calls get_finished() on each secondary tier
        2. For completed stores (primary→secondary): calls primary.complete_read()
           to decrement ref_cnt
        3. For completed loads (secondary→primary): calls primary.complete_write()
           to make blocks available
        """
        for tier in self.secondary_tiers:
            for completed_job in tier.get_finished():
                job_id = completed_job.job_id

                # Determine job type by checking which dictionary contains the job_id
                if job_id in self._store_jobs:
                    # primary→secondary transfer completed.
                    # Decrement ref_cnt on primary blocks.
                    job_metadata = self._store_jobs.pop(job_id)
                    self.primary_tier.complete_read(job_metadata.keys)
                elif job_id in self._load_jobs:
                    # secondary→primary transfer (promotion) completed.
                    # Make blocks available in primary tier.
                    job_metadata = self._load_jobs.pop(job_id)
                    self.primary_tier.complete_write(
                        job_metadata.keys, completed_job.success
                    )
                else:
                    # Job ID not found in either dictionary - this shouldn't happen
                    logger.error(
                        "Received finished job for unknown job_id %d from tier %s",
                        job_id,
                        tier.get_tier_name(),
                    )

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a single block is offloaded and ready.

        Algorithm:
        1. Process any completed async jobs first
        2. Check primary tier
        3. If not in primary, check secondary tiers and initiate promotion
           if the block is found there

        Args:
            key: Block hash to look up.
            req_context: Per-request context.

        Returns:
            True if the block is in the primary tier and ready,
            False if not found in any tier,
            None if the block is being transferred (retry later).
        """
        self._process_finished_jobs()

        # Step 1: Check primary tier
        primary_hit = self.primary_tier.lookup(key, req_context)
        if primary_hit is None:
            return None
        if primary_hit:
            return True

        # Step 2: Check secondary tiers
        for tier in self.secondary_tiers:
            secondary_hits = tier.lookup([key])
            if secondary_hits is None:
                # Tier is busy with this block
                return None
            if secondary_hits > 0:
                # Found in secondary — initiate promotion and signal retry
                self._initiate_promotion(tier, [key], req_context)
                return None

        return False

    def _initiate_promotion(
        self,
        tier: SecondaryTierManager,
        keys: list[OffloadKey],
        req_context: ReqContext,
    ):
        """
        Initiate promotion of blocks from a secondary tier to the primary tier.

        This method:
        1. Calls primary.prepare_write() to allocate space in primary tier
        2. Calls tier.submit_load() to start async transfer: secondary→primary
        3. Tracks the job in _load_jobs dictionary

        Args:
            tier: The secondary tier to promote from
            keys: Blocks to promote
            req_context: Per-request context forwarded to primary.prepare_write().
        """
        # Allocate space in primary tier for promoted blocks
        primary_store_result = self.primary_tier.prepare_write(keys, req_context)

        if primary_store_result is None:
            # Cannot allocate space in primary tier (full)
            # The next lookup() will retry
            return

        # Submit async load job: secondary→primary
        job_id = self._next_job_id()

        # Track this load job
        job_metadata = JobMetadata(
            job_id=job_id,
            keys=primary_store_result.keys_to_store,
            spec=primary_store_result.store_spec,
            req_context=req_context,
        )
        self._load_jobs[job_id] = job_metadata

        tier.submit_load(job_metadata)

    def prepare_load(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        """
        Prepare blocks to be loaded from primary tier to GPU.

        CRITICAL: This method calls _process_finished_jobs() FIRST to ensure
        that any completed promotions have been finalized and blocks are ready.

        This increments ref_cnt on the blocks in the primary tier, protecting
        them from eviction during the transfer.

        Args:
            keys: Blocks to prepare for loading.
            req_context: Per-request context.

        Returns:
            LoadStoreSpec for reading from primary tier.
        """
        # Process completed promotions to ensure blocks are ready
        self._process_finished_jobs()

        return self.primary_tier.prepare_load(keys, req_context)

    def touch(self, keys: Iterable[OffloadKey]):
        """
        Mark blocks as recently used in all tiers.

        Args:
            keys: Blocks to mark as recently used.
        """
        keys = list(keys)
        self.primary_tier.touch(keys)
        for tier in self.secondary_tiers:
            tier.touch(keys)

    def complete_load(self, keys: Iterable[OffloadKey]):
        """
        Mark blocks as done loading from primary tier to GPU.

        This decrements ref_cnt on the blocks in the primary tier, allowing
        them to be evicted again.

        Args:
            keys: Blocks that finished loading.
        """
        self.primary_tier.complete_load(keys)

    def prepare_store(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """
        Prepare blocks to be stored from GPU to primary tier.

        CRITICAL: This method calls _process_finished_jobs() FIRST to ensure
        that any completed async transfers have their ref_cnt decremented
        before the primary tier makes eviction decisions.

        Args:
            keys: Blocks to prepare for storing.
            req_context: Per-request context.

        Returns:
            PrepareStoreOutput describing where to store blocks and what was
            evicted, or None if store cannot proceed.
        """
        # Step 1: Poll for completed async jobs FIRST
        # This decrements ref_cnt on primary blocks that have been
        # successfully transferred to secondary tiers.
        self._process_finished_jobs()

        # Step 2: Store to primary tier
        primary_result = self.primary_tier.prepare_store(keys, req_context)

        # Note: Secondary tier cascading will happen in complete_store()
        # after the GPU→Primary transfer completes and blocks are ready.

        return primary_result

    def complete_store(
        self,
        keys: Iterable[OffloadKey],
        success: bool = True,
    ):
        """
        Mark blocks as done storing from GPU to primary tier.

        This is where secondary tier cascading happens — after blocks are
        confirmed to be in the primary tier, they are cascaded to ALL
        secondary tiers.

        For each secondary tier:
        1. Call primary.prepare_read() to get LoadStoreSpec AND increment
           ref_cnt (protecting blocks during async transfer)
        2. Call tier.submit_store() to start async transfer: primary→secondary
        3. Track the job in _store_jobs dictionary

        Args:
            keys: Blocks that finished storing.
            success: Whether the GPU→primary transfer succeeded.
            req_context: Per-request context forwarded to primary.prepare_read().
        """
        # Materialize only if success=True (needed for cascading to secondary tiers)
        keys_list = list(keys) if success else keys

        # Step 1: Complete store in primary tier (makes blocks loadable)
        self.primary_tier.complete_store(keys_list, success)

        if not success:
            # If GPU→Primary transfer failed, don't cascade to secondary tiers
            return

        # At this point, success=True is guaranteed, so keys_list
        # is list[OffloadKey]
        assert isinstance(keys_list, list)

        # Step 2: Cascade to ALL secondary tiers
        # For each secondary tier, call primary.prepare_read() to get the
        # LoadStoreSpec AND to increment ref_cnt (protecting blocks from
        # eviction during the async transfer). One prepare_read() call per
        # secondary tier.
        for tier in self.secondary_tiers:
            # Get spec for reading from primary tier AND increment ref_cnt
            # TODO: pass the actual req_context instead of None
            primary_blocks_spec = self.primary_tier.prepare_read(
                keys_list, ReqContext()
            )

            # Submit async store job: primary→secondary
            job_id = self._next_job_id()

            # Track this store job
            job_metadata = JobMetadata(
                job_id=job_id, keys=keys_list, spec=primary_blocks_spec
            )
            self._store_jobs[job_id] = job_metadata

            tier.submit_store(job_metadata)

        # Note: The async transfers are now in flight.
        # Their completion is tracked via get_finished() / _process_finished_jobs().

    def take_events(self) -> Iterable[OffloadingEvent]:
        """
        Take offloading events from the primary tier.

        Note: Currently only primary tier events are tracked. Secondary tier
        events could be added in the future if needed.

        Yields:
            New OffloadingEvents collected since the last call.
        """
        if self.events is not None:
            yield from self.events
            self.events.clear()

        # Also yield events from primary tier
        yield from self.primary_tier.take_events()

    def shutdown(self) -> None:
        """Release memoryviews and scheduler-side mmap."""
        for view in self._secondary_views:
            view.release()
        if self.primary_tier._mmap_region is not None:
            self.primary_tier._mmap_region.cleanup()
            self.primary_tier._mmap_region = None
