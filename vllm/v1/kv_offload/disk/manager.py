# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieredOffloadingManager: GPU → CPU (pinned) → Disk (NVMe)

Full three-tier with async prefetch (SGLang HiCache-inspired):

WRITES: Worker-side fire-and-forget after GPU→CPU completes.
READS:  Two-phase async prefetch:
  Phase 1 (lookup): Detect disk blocks, queue prefetch, DON'T mark ready
  Phase 2 (worker): Read disk→CPU in background between engine steps
  Phase 3 (complete_prefetch): Mark blocks ready in CPU policy
  Phase 4 (next lookup): Blocks found in CPU → counted as hits
"""
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec

logger = init_logger(__name__)


class DiskBlockIndex:
    """Tracks which block hashes are stored on disk."""

    def __init__(self, num_blocks: int):
        self._num_blocks = num_blocks
        self._num_allocated = 0
        self._free_list: list[int] = []
        self._index: dict[BlockHash, int] = {}

    def contains(self, bh: BlockHash) -> bool:
        return bh in self._index

    def get_block_id(self, bh: BlockHash) -> int | None:
        return self._index.get(bh)

    def allocate(self, bh: BlockHash) -> int | None:
        if bh in self._index:
            return self._index[bh]
        if self._free_list:
            bid = self._free_list.pop()
        elif self._num_allocated < self._num_blocks:
            bid = self._num_allocated
            self._num_allocated += 1
        else:
            return None
        self._index[bh] = bid
        return bid

    def free(self, bh: BlockHash) -> None:
        bid = self._index.pop(bh, None)
        if bid is not None:
            self._free_list.append(bid)

    @property
    def size(self) -> int:
        return len(self._index)


@dataclass
class PendingPrefetch:
    """A prefetch that has been dispatched to the worker but not yet completed."""
    block_hashes: list[BlockHash]
    cpu_block_ids: list[int]
    disk_block_ids: list[int]


class TieredOffloadingManager(OffloadingManager):
    """GPU → CPU (pinned) → Disk (NVMe) with async prefetch."""

    def __init__(
        self,
        block_size: int,
        num_cpu_blocks: int,
        num_disk_blocks: int,
        write_threshold: int = 1,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
    ):
        self._cpu = CPUOffloadingManager(
            block_size=block_size,
            num_blocks=num_cpu_blocks,
            cache_policy=cache_policy,
            enable_events=enable_events,
        )
        self._disk = DiskBlockIndex(num_disk_blocks)
        self.block_size = block_size

        # Prefetch queues
        # Outbound: scheduler → worker (to be read from disk)
        self._outbound_prefetches: list[PendingPrefetch] = []
        # Inbound: worker → scheduler (completed reads, need CPU policy insert)
        self._completed_prefetches: list[PendingPrefetch] = []
        # Track what's currently being prefetched to avoid duplicates
        self._inflight_prefetch_hashes: set[BlockHash] = set()

        logger.info(
            "TieredOffloadingManager: cpu=%d blocks, disk=%d blocks, "
            "write_threshold=%d",
            num_cpu_blocks, num_disk_blocks, write_threshold,
        )

    # ── OffloadingManager interface ──────────────────────────────

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """Count CPU-ready blocks. Queue prefetch for disk blocks."""
        # First: process any completed prefetches from previous cycle
        self._process_completed_prefetches()

        block_list = list(block_hashes)
        cpu_count = 0
        for bh in block_list:
            cpu_block = self._cpu._policy.get(bh)
            if cpu_block is not None and cpu_block.is_ready:
                cpu_count += 1
            else:
                break

        # Queue prefetch for consecutive disk blocks beyond CPU prefix
        remaining = block_list[cpu_count:]
        if remaining:
            self._maybe_queue_prefetch(remaining)

        return cpu_count

    def _maybe_queue_prefetch(self, block_hashes: list[BlockHash]) -> None:
        """Queue async prefetch for disk-resident blocks.

        Does NOT insert into CPU policy — blocks are only marked ready
        after the worker completes the disk read and signals back.
        """
        disk_hashes: list[BlockHash] = []
        disk_ids: list[int] = []

        for bh in block_hashes:
            if bh in self._inflight_prefetch_hashes:
                continue  # Already being prefetched
            # Check if already in CPU (might have been prefetched since last check)
            cpu_block = self._cpu._policy.get(bh)
            if cpu_block is not None:
                continue
            disk_id = self._disk.get_block_id(bh)
            if disk_id is not None:
                disk_hashes.append(bh)
                disk_ids.append(disk_id)
            else:
                break  # No more consecutive disk blocks

        if not disk_hashes:
            return

        # Allocate CPU slots for incoming data
        cpu_block_ids: list[int] = []
        valid_hashes: list[BlockHash] = []
        valid_disk_ids: list[int] = []

        for bh, disk_id in zip(disk_hashes, disk_ids):
            if self._cpu._get_num_free_blocks() == 0:
                evicted = self._cpu._policy.evict(1, set())
                if evicted:
                    for _, eblock in evicted:
                        self._cpu._free_block(eblock)
                else:
                    break

            blocks = self._cpu._allocate_blocks([bh])
            if not blocks:
                break

            cpu_block = blocks[0]
            # ref_cnt stays at -1 (NOT READY) until disk read completes.
            # This prevents lookup() from counting these blocks as hits.
            # Insert into policy so _process_completed_prefetches can
            # find and mark ready later.
            self._cpu._policy.insert(bh, cpu_block)
            cpu_block_ids.append(cpu_block.block_id)
            valid_hashes.append(bh)
            valid_disk_ids.append(disk_id)

        if not valid_hashes:
            return

        prefetch = PendingPrefetch(
            block_hashes=valid_hashes,
            cpu_block_ids=cpu_block_ids,
            disk_block_ids=valid_disk_ids,
        )
        self._outbound_prefetches.append(prefetch)
        for bh in valid_hashes:
            self._inflight_prefetch_hashes.add(bh)

        logger.debug("Queued prefetch: %d blocks disk→CPU", len(valid_hashes))

    def _process_completed_prefetches(self) -> None:
        """Insert completed prefetch blocks into CPU policy as ready.

        Called at the start of each lookup() cycle.
        """
        while self._completed_prefetches:
            prefetch = self._completed_prefetches.pop()
            for bh, cpu_bid in zip(
                prefetch.block_hashes, prefetch.cpu_block_ids
            ):
                self._inflight_prefetch_hashes.discard(bh)
                # Now insert into policy as ready (ref_cnt=0)
                cpu_block = self._cpu._policy.get(bh)
                if cpu_block is None:
                    # Block was evicted while prefetch was in flight — skip
                    continue
                if cpu_block.block_id == cpu_bid and not cpu_block.is_ready:
                    cpu_block.ref_cnt = 0  # Mark ready

    def take_outbound_prefetches(self) -> list[PendingPrefetch]:
        """Drain prefetch requests for the worker to process."""
        out = self._outbound_prefetches
        self._outbound_prefetches = []
        return out

    def mark_prefetches_completed(
        self, prefetches: list[PendingPrefetch]
    ) -> None:
        """Worker signals that disk reads completed."""
        self._completed_prefetches.extend(prefetches)

    def mark_prefetches_by_cpu_ids(self, cpu_block_ids: list[int]) -> None:
        """Mark prefetched blocks as ready by their CPU block IDs.

        Called from the scheduler when the worker reports completion.
        Finds the matching PendingPrefetch and moves it to completed.
        """
        cpu_id_set = set(cpu_block_ids)
        # Find matching inflight prefetch hashes and create a
        # synthetic PendingPrefetch for completion processing
        matched_hashes: list[BlockHash] = []
        matched_cpu_ids: list[int] = []
        for bh in list(self._inflight_prefetch_hashes):
            cpu_block = self._cpu._policy.get(bh)
            if cpu_block is not None and cpu_block.block_id in cpu_id_set:
                matched_hashes.append(bh)
                matched_cpu_ids.append(cpu_block.block_id)

        if matched_hashes:
            self._completed_prefetches.append(PendingPrefetch(
                block_hashes=matched_hashes,
                cpu_block_ids=matched_cpu_ids,
                disk_block_ids=[],  # Not needed for completion
            ))

    # ── Standard OffloadingManager methods ───────────────────────

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        return self._cpu.prepare_load(block_hashes)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self._cpu.touch(block_hashes)

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        self._cpu.complete_load(block_hashes)

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        result = self._cpu.prepare_store(block_hashes)
        if result is None:
            return None
        # Track in disk index — actual write happens on worker side
        for bh in result.block_hashes_to_store:
            self._disk.allocate(bh)
        return result

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        self._cpu.complete_store(block_hashes, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        yield from self._cpu.take_events()
