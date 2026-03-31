# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieredOffloadingManager: GPU → CPU (pinned) → Disk (NVMe)

Full three-tier with rate-limited async prefetch:

WRITES: Worker-side fire-and-forget after GPU→CPU completes.
READS:  Rate-limited async prefetch:
  - lookup() counts CPU blocks only (fast, no side effects)
  - lookup() records disk hits for prefetch consideration
  - Scheduler calls maybe_prefetch_top_requests() once per step
  - Only prefetches for the TOP N requests with most disk hits
  - Caps prefetch at 10% of CPU block capacity per step
  - Worker reads disk→CPU async (non-blocking)
  - Next step: completed prefetches marked ready → cache hits
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
    """A prefetch dispatched to worker, not yet completed."""
    block_hashes: list[BlockHash]
    cpu_block_ids: list[int]
    disk_block_ids: list[int]


class TieredOffloadingManager(OffloadingManager):
    """GPU → CPU (pinned) → Disk (NVMe) with rate-limited async prefetch."""

    # Max fraction of CPU blocks that can be used for prefetch per step
    PREFETCH_BUDGET_RATIO = 0.10  # 10% of CPU capacity
    # Max number of prefetch requests per step
    MAX_PREFETCHES_PER_STEP = 3

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
        self._num_cpu_blocks = num_cpu_blocks
        self._prefetch_budget = int(
            num_cpu_blocks * self.PREFETCH_BUDGET_RATIO
        )

        # Prefetch tracking
        self._outbound_prefetches: list[PendingPrefetch] = []
        self._completed_prefetches: list[PendingPrefetch] = []
        self._inflight_prefetch_hashes: set[BlockHash] = set()
        self._inflight_block_count = 0

        # Per-step disk hit tracking: populated by lookup(), consumed
        # by maybe_prefetch_top_requests()
        self._step_disk_hits: list[tuple[int, list[BlockHash], list[int]]] = []

        logger.info(
            "TieredOffloadingManager: cpu=%d blocks, disk=%d blocks, "
            "prefetch_budget=%d blocks/step",
            num_cpu_blocks, num_disk_blocks, self._prefetch_budget,
        )

    # ── OffloadingManager interface ──────────────────────────────

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """Count CPU-ready blocks. Record disk hits for prefetch."""
        self._process_completed_prefetches()

        block_list = list(block_hashes)
        cpu_count = 0
        for bh in block_list:
            cpu_block = self._cpu._policy.get(bh)
            if cpu_block is not None and cpu_block.is_ready:
                cpu_count += 1
            else:
                break

        # Record disk hits beyond CPU prefix (for prefetch ranking)
        remaining = block_list[cpu_count:]
        if remaining:
            disk_hashes = []
            disk_ids = []
            for bh in remaining:
                if bh in self._inflight_prefetch_hashes:
                    break  # Already being prefetched
                cpu_block = self._cpu._policy.get(bh)
                if cpu_block is not None:
                    break  # In CPU (maybe not ready yet — being prefetched)
                disk_id = self._disk.get_block_id(bh)
                if disk_id is not None:
                    disk_hashes.append(bh)
                    disk_ids.append(disk_id)
                else:
                    break

            if disk_hashes:
                # Record (num_disk_blocks, hashes, disk_ids) for ranking
                self._step_disk_hits.append(
                    (len(disk_hashes), disk_hashes, disk_ids)
                )

        return cpu_count

    def maybe_prefetch_top_requests(self) -> None:
        """Called once per scheduler step after all lookups complete.

        Picks the top requests by disk hit count and queues prefetch
        for them, respecting the per-step budget.
        """
        if not self._step_disk_hits:
            return

        # Sort by disk hit count (most hits first = most benefit)
        self._step_disk_hits.sort(key=lambda x: x[0], reverse=True)

        budget_remaining = self._prefetch_budget - self._inflight_block_count
        prefetches_this_step = 0

        for num_hits, disk_hashes, disk_ids in self._step_disk_hits:
            if prefetches_this_step >= self.MAX_PREFETCHES_PER_STEP:
                break
            if budget_remaining <= 0:
                break

            # Cap this prefetch to remaining budget
            cap = min(len(disk_hashes), budget_remaining)
            if cap < 4:  # Not worth prefetching less than 4 blocks
                continue
            capped_hashes = disk_hashes[:cap]
            capped_disk_ids = disk_ids[:cap]

            prefetch = self._allocate_prefetch(capped_hashes, capped_disk_ids)
            if prefetch is not None:
                self._outbound_prefetches.append(prefetch)
                budget_remaining -= len(prefetch.cpu_block_ids)
                prefetches_this_step += 1

        # Clear step hits for next cycle
        self._step_disk_hits.clear()

    def _allocate_prefetch(
        self, disk_hashes: list[BlockHash], disk_ids: list[int]
    ) -> PendingPrefetch | None:
        """Allocate CPU slots for a prefetch. Returns None if can't allocate."""
        cpu_block_ids: list[int] = []
        valid_hashes: list[BlockHash] = []
        valid_disk_ids: list[int] = []

        for bh, disk_id in zip(disk_hashes, disk_ids):
            if bh in self._inflight_prefetch_hashes:
                continue

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
            # ref_cnt=-1 (NOT READY) — prevents lookup() from counting
            self._cpu._policy.insert(bh, cpu_block)
            cpu_block_ids.append(cpu_block.block_id)
            valid_hashes.append(bh)
            valid_disk_ids.append(disk_id)

        if not valid_hashes:
            return None

        for bh in valid_hashes:
            self._inflight_prefetch_hashes.add(bh)
        self._inflight_block_count += len(valid_hashes)

        logger.debug(
            "Allocated prefetch: %d blocks disk→CPU (%d inflight total)",
            len(valid_hashes), self._inflight_block_count,
        )

        return PendingPrefetch(
            block_hashes=valid_hashes,
            cpu_block_ids=cpu_block_ids,
            disk_block_ids=valid_disk_ids,
        )

    def _process_completed_prefetches(self) -> None:
        """Mark completed prefetch blocks as ready in CPU policy."""
        while self._completed_prefetches:
            prefetch = self._completed_prefetches.pop()
            for bh, cpu_bid in zip(
                prefetch.block_hashes, prefetch.cpu_block_ids
            ):
                self._inflight_prefetch_hashes.discard(bh)
                self._inflight_block_count = max(
                    0, self._inflight_block_count - 1
                )
                cpu_block = self._cpu._policy.get(bh)
                if cpu_block is None:
                    continue
                if cpu_block.block_id == cpu_bid and not cpu_block.is_ready:
                    cpu_block.ref_cnt = 0  # READY

    # ── Prefetch transport ───────────────────────────────────────

    def take_outbound_prefetches(self) -> list[PendingPrefetch]:
        """Drain prefetch requests for the worker."""
        out = self._outbound_prefetches
        self._outbound_prefetches = []
        return out

    def mark_prefetches_by_cpu_ids(self, cpu_block_ids: list[int]) -> None:
        """Worker reports completion — find matching prefetches."""
        cpu_id_set = set(cpu_block_ids)
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
                disk_block_ids=[],
            ))

    # ── Standard OffloadingManager delegation ────────────────────

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
        for bh in result.block_hashes_to_store:
            self._disk.allocate(bh)
        return result

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        self._cpu.complete_store(block_hashes, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        yield from self._cpu.take_events()
