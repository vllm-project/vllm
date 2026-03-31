# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieredOffloadingManager: GPU → CPU (pinned) → Disk (NVMe)

Architecture:
  WRITE PATH: Worker-side fire-and-forget after GPU→CPU swap completes.
  READ PATH:  Scheduler detects disk hit on CPU miss → uses prepare_store
              to allocate CPU blocks → worker reads disk data into those
              blocks → scheduler calls complete_store → blocks become
              available in CPU for normal GPU load.

The read path reuses the existing prepare_store/complete_store mechanism.
The difference is the data comes from disk instead of GPU. This avoids
any special allocation paths or staging buffers.

When get_num_new_matched_tokens returns None (retry), the scheduler
defers the request. By the next scheduling step, the disk read has
completed and the blocks are in CPU.
"""
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
class DiskPrefetchOp:
    """A disk→CPU prefetch in progress."""
    block_hashes: list[BlockHash]
    cpu_block_ids: list[int]
    disk_block_ids: list[int]
    dispatched: bool = False
    completed: bool = False


class TieredOffloadingManager(OffloadingManager):
    """
    GPU → CPU (pinned) → Disk (NVMe).

    Uses prepare_store/complete_store for disk→CPU promotion,
    keeping all block management consistent with the CPU tier.
    """

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

        # Active prefetch operations (keyed by block hash)
        self._active_prefetches: dict[BlockHash, DiskPrefetchOp] = {}
        # Prefetches ready to dispatch to worker
        self._pending_dispatch: list[DiskPrefetchOp] = []
        # Prefetches completed by worker, need complete_store
        self._pending_completion: list[DiskPrefetchOp] = []
        # Total blocks currently reserved for prefetch (guards against
        # prefetches consuming too much of the CPU pool)
        self._prefetch_blocks_reserved = 0
        self._max_prefetch_blocks = num_cpu_blocks // 5  # 20% cap

        logger.info(
            "TieredOffloadingManager: cpu=%d blocks, disk=%d blocks, "
            "max_prefetch=%d blocks",
            num_cpu_blocks, num_disk_blocks, self._max_prefetch_blocks,
        )

    # ── Core interface ───────────────────────────────────────────

    _lookup_count = 0

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """Count CPU-ready blocks."""
        self._finalize_completed_prefetches()
        result = self._cpu.lookup(block_hashes)

        # Debug: log stats every 100 lookups
        TieredOffloadingManager._lookup_count += 1
        if TieredOffloadingManager._lookup_count % 100 == 0:
            stats = self.get_debug_stats()
            logger.info(
                "DISK_DEBUG lookup#%d result=%s stats=%s",
                TieredOffloadingManager._lookup_count, result, stats,
            )
        return result

    def get_debug_stats(self) -> dict:
        """Debug stats for diagnosing stalls."""
        total_blocks = self._cpu._num_blocks
        allocated = self._cpu._num_allocated_blocks
        free_list = len(self._cpu._free_list)
        # Count blocks by ref_cnt
        ref_counts = {-1: 0, 0: 0}
        for bh, block in self._cpu._policy.blocks.items():
            rc = block.ref_cnt
            ref_counts[rc] = ref_counts.get(rc, 0) + 1
        return {
            "total": total_blocks,
            "allocated": allocated,
            "free_list": free_list,
            "available": free_list + (total_blocks - allocated),
            "in_policy": len(self._cpu._policy.blocks),
            "ref_cnt_dist": ref_counts,
            "active_prefetches": len(self._active_prefetches),
            "pending_dispatch": len(self._pending_dispatch),
            "pending_completion": len(self._pending_completion),
            "disk_index_size": self._disk.size,
            "prefetch_reserved": self._prefetch_blocks_reserved,
        }

    def try_disk_prefetch(
        self,
        block_hashes: list[BlockHash],
        start_idx: int = 0,
    ) -> bool:
        """Attempt to start a disk→CPU prefetch for the given blocks.

        Returns True if prefetch was initiated (caller should return None
        to defer the request). Returns False if no disk data available.

        Uses prepare_store() to allocate CPU blocks, then queues a
        disk read for the worker to execute.
        """
        # Find consecutive disk-resident blocks
        disk_hashes: list[BlockHash] = []
        disk_ids: list[int] = []
        for bh in block_hashes:
            # Skip if already in CPU or already being prefetched
            if self._cpu._policy.get(bh) is not None:
                continue
            if bh in self._active_prefetches:
                return True  # Already prefetching, just wait
            disk_id = self._disk.get_block_id(bh)
            if disk_id is not None:
                disk_hashes.append(bh)
                disk_ids.append(disk_id)
            else:
                break

        if not disk_hashes:
            return False

        # Guard: don't let prefetch consume too much of the CPU pool
        if self._prefetch_blocks_reserved >= self._max_prefetch_blocks:
            return False
        # Cap this prefetch to remaining budget
        budget = self._max_prefetch_blocks - self._prefetch_blocks_reserved
        disk_hashes = disk_hashes[:budget]
        disk_ids = disk_ids[:budget]

        # Use prepare_store to allocate CPU blocks via the normal path.
        # This handles eviction, block allocation, policy insertion.
        logger.info(
            "DISK_DEBUG try_prefetch: %d disk blocks, budget=%d, reserved=%d",
            len(disk_hashes), budget, self._prefetch_blocks_reserved,
        )
        result = self._cpu.prepare_store(disk_hashes)
        if result is None:
            logger.warning("DISK_DEBUG prefetch prepare_store returned None")
            return False  # Can't allocate, fall back to re-prefill

        # Get the CPU block IDs that were allocated
        cpu_store_spec = result.store_spec
        # The store spec contains the CPU block IDs
        from vllm.v1.kv_offload.mediums import CPULoadStoreSpec
        assert isinstance(cpu_store_spec, CPULoadStoreSpec)
        cpu_block_ids = cpu_store_spec.block_ids.tolist()

        # Only prefetch the blocks that were actually allocated
        actual_count = len(result.block_hashes_to_store)
        if actual_count == 0:
            return False

        # Map allocated block hashes to their disk IDs
        allocated_disk_ids = []
        for bh in result.block_hashes_to_store:
            did = self._disk.get_block_id(bh)
            assert did is not None
            allocated_disk_ids.append(did)

        op = DiskPrefetchOp(
            block_hashes=list(result.block_hashes_to_store),
            cpu_block_ids=cpu_block_ids[:actual_count],
            disk_block_ids=allocated_disk_ids,
        )

        # Track the prefetch
        for bh in op.block_hashes:
            self._active_prefetches[bh] = op
        self._pending_dispatch.append(op)
        self._prefetch_blocks_reserved += len(op.block_hashes)

        logger.debug(
            "Initiated disk prefetch: %d blocks", len(op.block_hashes)
        )
        return True

    def take_pending_dispatches(self) -> list[DiskPrefetchOp]:
        """Drain prefetch ops ready for worker dispatch."""
        ops = self._pending_dispatch
        self._pending_dispatch = []
        for op in ops:
            op.dispatched = True
        return ops

    def mark_dispatch_completed(self, op: DiskPrefetchOp) -> None:
        """Worker signals that disk read completed for this op."""
        op.completed = True
        self._pending_completion.append(op)

    def mark_completed_by_cpu_ids(self, cpu_block_ids: list[int]) -> None:
        """Worker signals completion by CPU block IDs."""
        cpu_set = set(cpu_block_ids)
        for bh, op in list(self._active_prefetches.items()):
            if op.dispatched and not op.completed:
                if any(cid in cpu_set for cid in op.cpu_block_ids):
                    op.completed = True
                    if op not in self._pending_completion:
                        self._pending_completion.append(op)

    def _finalize_completed_prefetches(self) -> None:
        """Mark completed prefetch blocks as ready via complete_store."""
        while self._pending_completion:
            op = self._pending_completion.pop()
            # complete_store marks blocks as ready (ref_cnt=0)
            self._cpu.complete_store(op.block_hashes, success=True)
            # Release reservation and clean up tracking
            self._prefetch_blocks_reserved = max(
                0, self._prefetch_blocks_reserved - len(op.block_hashes)
            )
            for bh in op.block_hashes:
                self._active_prefetches.pop(bh, None)

    # ── Standard delegation ──────────────────────────────────────

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
        # Track in disk index for write-through
        for bh in result.block_hashes_to_store:
            self._disk.allocate(bh)
        return result

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        self._cpu.complete_store(block_hashes, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        yield from self._cpu.take_events()
