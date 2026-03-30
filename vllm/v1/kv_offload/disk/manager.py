# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieredOffloadingManager: GPU → CPU (pinned) → Disk (NVMe)

Write-through with hit threshold:
  - Every block stored to CPU is tracked
  - After a block is "hit" N times (confirming it's hot), it gets
    written to disk in the background
  - When CPU evicts a block, if a disk copy exists, it's not lost
  - When a lookup misses CPU but hits disk, it gets prefetched

Inspired by SGLang HiCache but optimized for vLLM's architecture:
  - Uses vLLM's OffloadingManager interface
  - Pinned memory for GPU↔CPU (4.8 GB/s)
  - Batched disk I/O via thread pool
"""
from collections.abc import Iterable
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
    """Tracks which blocks are stored on disk and their disk slot IDs."""

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


class TieredOffloadingManager(OffloadingManager):
    """
    GPU → CPU (pinned) → Disk (NVMe).

    Write-through policy:
      1. GPU→CPU store completes → block tracked with hit_count=1
      2. On subsequent cache hits (touch), hit_count increments
      3. When hit_count >= write_threshold, queue async CPU→Disk write
      4. CPU eviction of a disk-backed block is free (disk copy exists)
      5. Lookup miss on CPU but hit on disk → queue prefetch (Disk→CPU)
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
        self.write_threshold = write_threshold

        # Track hit counts for write-through decisions
        self._hit_counts: dict[BlockHash, int] = {}

        # Pending operations for the worker to execute
        # (block_hash, cpu_block_id, disk_block_id)
        self._pending_writes: list[tuple[BlockHash, int, int]] = []
        self._pending_reads: list[tuple[BlockHash, int, int]] = []

        logger.info(
            "TieredOffloadingManager: cpu=%d blocks, disk=%d blocks, "
            "write_threshold=%d",
            num_cpu_blocks, num_disk_blocks, write_threshold,
        )

    # ── OffloadingManager interface ──────────────────────────────

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        count = 0
        for bh in block_hashes:
            cpu_block = self._cpu._policy.get(bh)
            if cpu_block is not None and cpu_block.is_ready:
                count += 1
            elif self._disk.contains(bh):
                # Disk hit — block can be promoted to CPU on load
                count += 1
            else:
                break
        return count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        block_ids: list[int] = []
        for bh in block_hashes:
            cpu_block = self._cpu._policy.get(bh)
            if cpu_block is not None:
                # In CPU — fast path
                cpu_block.ref_cnt += 1
                block_ids.append(cpu_block.block_id)
                continue

            # Not in CPU — must be on disk. Promote it.
            disk_bid = self._disk.get_block_id(bh)
            if disk_bid is None:
                raise AssertionError(f"Block {bh!r} not in CPU or disk")

            # Make room in CPU if needed
            if self._cpu._get_num_free_blocks() == 0:
                evicted = self._cpu._policy.evict(1, set())
                if evicted:
                    for _, eblock in evicted:
                        self._cpu._free_block(eblock)

            # Allocate CPU slot and queue disk→CPU read.
            # BlockStatus initializes ref_cnt=-1 (not ready), so we
            # set it to 1 (ready + locked for this load).
            new_blocks = self._cpu._allocate_blocks([bh])
            assert len(new_blocks) == 1
            cpu_block = new_blocks[0]
            cpu_block.ref_cnt = 1
            self._cpu._policy.insert(bh, cpu_block)

            self._pending_reads.append((bh, cpu_block.block_id, disk_bid))
            block_ids.append(cpu_block.block_id)

        return CPULoadStoreSpec(block_ids)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        self._cpu.touch(block_hashes)
        # Increment hit counts — used for write-through threshold
        for bh in block_hashes:
            count = self._hit_counts.get(bh, 0) + 1
            self._hit_counts[bh] = count
            # If hit threshold reached and not already on disk, queue write
            if count == self.write_threshold and not self._disk.contains(bh):
                cpu_block = self._cpu._policy.get(bh)
                if cpu_block is not None and cpu_block.is_ready:
                    disk_bid = self._disk.allocate(bh)
                    if disk_bid is not None:
                        self._pending_writes.append(
                            (bh, cpu_block.block_id, disk_bid)
                        )

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        self._cpu.complete_load(block_hashes)

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        result = self._cpu.prepare_store(block_hashes)
        if result is None:
            return None

        # Initialize hit counts for newly stored blocks
        for bh in result.block_hashes_to_store:
            if bh not in self._hit_counts:
                self._hit_counts[bh] = 0

        # For write_threshold=1 (aggressive write-through):
        # immediately queue disk writes for new blocks
        if self.write_threshold <= 1:
            for bh in result.block_hashes_to_store:
                if not self._disk.contains(bh):
                    cpu_block = self._cpu._policy.get(bh)
                    if cpu_block is not None:
                        disk_bid = self._disk.allocate(bh)
                        if disk_bid is not None:
                            self._pending_writes.append(
                                (bh, cpu_block.block_id, disk_bid)
                            )
                            self._hit_counts[bh] = self.write_threshold

        return result

    def complete_store(
        self, block_hashes: Iterable[BlockHash], success: bool = True
    ) -> None:
        self._cpu.complete_store(block_hashes, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        yield from self._cpu.take_events()

    # ── Disk operation queues (consumed by worker) ───────────────

    def take_pending_disk_writes(
        self,
    ) -> list[tuple[BlockHash, int, int]]:
        """Drain (block_hash, cpu_block_id, disk_block_id) writes."""
        w = self._pending_writes
        self._pending_writes = []
        return w

    def take_pending_disk_reads(
        self,
    ) -> list[tuple[BlockHash, int, int]]:
        """Drain (block_hash, cpu_block_id, disk_block_id) reads."""
        r = self._pending_reads
        self._pending_reads = []
        return r
