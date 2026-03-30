# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieredOffloadingManager: GPU → CPU (pinned) → Disk (NVMe)

Full three-tier cache with async prefetch (inspired by SGLang HiCache):

WRITES (worker-side, fire-and-forget):
  GPU→CPU swap completes → callback writes same blocks to disk in background

READS / PREFETCH (async, non-blocking):
  1. lookup() counts CPU blocks only, never blocks
  2. Scheduler calls request_prefetch() for blocks on disk but not CPU
  3. Worker's PrefetchWorker thread reads disk→CPU in background
  4. On completion, blocks appear in CPU → next lookup() finds them
  5. Request gets scheduled with cached prefix

The scheduler NEVER blocks on disk I/O.
"""
import threading
from collections import deque
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


class PrefetchRequest:
    """A request to promote blocks from disk to CPU."""

    def __init__(
        self,
        block_hashes: list[BlockHash],
        disk_block_ids: list[int],
        cpu_block_ids: list[int],
    ):
        self.block_hashes = block_hashes
        self.disk_block_ids = disk_block_ids
        self.cpu_block_ids = cpu_block_ids
        self.completed = threading.Event()


class TieredOffloadingManager(OffloadingManager):
    """
    GPU → CPU (pinned) → Disk (NVMe) with async prefetch.
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

        # Prefetch state
        self._prefetch_queue: deque[PrefetchRequest] = deque()
        self._active_prefetches: dict[BlockHash, PrefetchRequest] = {}
        # Blocks that have been prefetched to CPU but not yet
        # "officially" stored (need complete_store equivalent)
        self._prefetched_blocks: set[BlockHash] = set()

        logger.info(
            "TieredOffloadingManager: cpu=%d blocks, disk=%d blocks, "
            "write_threshold=%d",
            num_cpu_blocks, num_disk_blocks, write_threshold,
        )

    # ── Core OffloadingManager interface ─────────────────────────

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """Count consecutive blocks available in CPU.

        Also queues prefetch for disk-resident blocks that follow
        the CPU-resident prefix. This is non-blocking — the prefetch
        happens in the background on the worker.
        """
        block_list = list(block_hashes)
        cpu_count = 0
        for bh in block_list:
            cpu_block = self._cpu._policy.get(bh)
            if cpu_block is not None and cpu_block.is_ready:
                cpu_count += 1
            else:
                break

        # Check if blocks beyond the CPU prefix are on disk
        # and trigger async prefetch if so
        remaining = block_list[cpu_count:]
        if remaining:
            disk_hashes = []
            disk_ids = []
            for bh in remaining:
                if bh in self._active_prefetches:
                    continue  # Already being prefetched
                if bh in self._prefetched_blocks:
                    continue  # Already prefetched, will be in CPU soon
                disk_id = self._disk.get_block_id(bh)
                if disk_id is not None:
                    disk_hashes.append(bh)
                    disk_ids.append(disk_id)
                else:
                    break  # No more consecutive disk blocks

            if disk_hashes:
                self._queue_prefetch(disk_hashes, disk_ids)

        return cpu_count

    def _queue_prefetch(
        self, block_hashes: list[BlockHash], disk_block_ids: list[int]
    ) -> None:
        """Queue an async prefetch: disk → CPU.

        Allocates CPU slots and creates a PrefetchRequest that the
        worker will consume.
        """
        # Allocate CPU blocks for the prefetched data
        cpu_block_ids: list[int] = []
        valid_hashes: list[BlockHash] = []
        valid_disk_ids: list[int] = []

        for bh, disk_id in zip(block_hashes, disk_block_ids):
            # Make room if needed
            if self._cpu._get_num_free_blocks() == 0:
                evicted = self._cpu._policy.evict(1, set())
                if evicted:
                    for _, eblock in evicted:
                        self._cpu._free_block(eblock)
                else:
                    break  # Can't make room

            blocks = self._cpu._allocate_blocks([bh])
            if not blocks:
                break
            cpu_block = blocks[0]
            # Mark as ready with ref_cnt=0 (data will arrive async)
            # The block won't be used until prefetch completes and
            # the next lookup finds it in CPU
            cpu_block.ref_cnt = 0
            self._cpu._policy.insert(bh, cpu_block)

            cpu_block_ids.append(cpu_block.block_id)
            valid_hashes.append(bh)
            valid_disk_ids.append(disk_id)

        if not valid_hashes:
            return

        req = PrefetchRequest(
            block_hashes=valid_hashes,
            disk_block_ids=valid_disk_ids,
            cpu_block_ids=cpu_block_ids,
        )
        self._prefetch_queue.append(req)
        for bh in valid_hashes:
            self._active_prefetches[bh] = req

        logger.debug(
            "Queued prefetch: %d blocks from disk to CPU", len(valid_hashes)
        )

    def take_prefetch_requests(self) -> list[PrefetchRequest]:
        """Drain pending prefetch requests (consumed by worker)."""
        reqs: list[PrefetchRequest] = []
        while self._prefetch_queue:
            reqs.append(self._prefetch_queue.popleft())
        return reqs

    def complete_prefetch(self, req: PrefetchRequest) -> None:
        """Called when a prefetch completes (data is now in CPU)."""
        for bh in req.block_hashes:
            self._active_prefetches.pop(bh, None)
            self._prefetched_blocks.discard(bh)

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
