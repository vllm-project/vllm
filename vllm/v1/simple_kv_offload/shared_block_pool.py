# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared publication and references for the simple offload BlockPool.

Allocation, hash indexing and LRU reuse the core BlockPool. Offloading's
ChunkStatus lives in a SharedOffloadRegion so peer loads protect owner slots.
Metadata is locked; DMA retains references after the lock is released.
"""

import contextlib
import fcntl
import hashlib
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import wraps
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar

import numpy as np

from vllm.distributed.kv_events import MEDIUM_CPU, AllBlocksCleared
from vllm.utils.math_utils import round_up
from vllm.v1.core.block_pool import BlockHashToBlockMap, BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadHandshake

if TYPE_CHECKING:
    from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

P = ParamSpec("P")
R = TypeVar("R")


def with_shared_pool_lock(
    method: Callable[Concatenate["SimpleCPUOffloadScheduler", P], R],
) -> Callable[Concatenate["SimpleCPUOffloadScheduler", P], R]:
    @wraps(method)
    def wrapped(
        self: "SimpleCPUOffloadScheduler", *args: P.args, **kwargs: P.kwargs
    ) -> R:
        pool = self.cpu_block_pool
        if isinstance(pool, SharedCPUBlockPool):
            with pool.locked():
                return method(self, *args, **kwargs)
        if self._shared_required:
            raise RuntimeError("Shared CPU offload worker handshake has not completed.")
        return method(self, *args, **kwargs)

    return wrapped


class SharedCPUBlockPool(BlockPool):
    """A local BlockPool writer with shared publication and read references."""

    def __init__(
        self,
        metadata: SimpleCPUOffloadHandshake,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        max_hashes_per_slot: int = 2,
    ):
        if metadata.blocks_per_rank < 2:
            raise ValueError("Shared CPU offload needs at least two slots per DP rank.")
        self.metadata = metadata
        self.local_rank = metadata.local_rank
        self._start = metadata.local_rank * metadata.blocks_per_rank
        self._end = self._start + metadata.blocks_per_rank
        self._lock_depth = 0
        self._closed = False
        self._peer_pinned: set[int] = set()
        count = metadata.blocks_per_rank * metadata.local_size
        super().__init__(
            count, True, hash_block_size, enable_kv_cache_events, medium=MEDIUM_CPU
        )
        for block in self.blocks:
            block.prev_free_block = block.next_free_block = None
            block.is_null = block.block_id % metadata.blocks_per_rank == 0
        self.free_block_queue = FreeKVCacheBlockQueue(
            self.blocks[self._start + 1 : self._end]
        )
        hash_bytes = 36 if metadata.hash_signature[0].startswith("sha256") else 20
        self._slot_dtype = np.dtype(
            [
                ("status", np.dtype(ChunkStatus), (count,)),
                ("version", np.uint64, (count,)),
                ("num_tokens", np.int64, (count,)),
                ("hash_count", np.uint64, (count,)),
                ("hashes", np.uint8, (count, max_hashes_per_slot, hash_bytes)),
            ]
        )
        page = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
        header_bytes = round_up(32 + (1 + metadata.local_size) * 8, page)
        size = round_up(header_bytes + self._slot_dtype.itemsize, page)
        identity = hashlib.sha256(metadata.region_id.encode()).hexdigest()
        signature = hashlib.sha256(
            repr(
                (
                    metadata.local_size,
                    metadata.blocks_per_rank,
                    metadata.hash_signature,
                    hash_block_size,
                    self._slot_dtype.descr,
                )
            ).encode()
        ).digest()
        self._region = SharedOffloadRegion(f"simple_index_{identity}", 1, None, size, 0)
        assert self._region.fd is not None
        self._lock_fd = self._region.fd
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        try:
            mapped = self._region.mmap_obj
            assert mapped is not None
            # The first locker may be a joiner whose mmap finished first.
            if mapped[:32] == bytes(32):
                mapped[:32] = signature
            elif mapped[:32] != signature:
                raise ValueError("Shared CPU offload scheduler configuration mismatch.")
            self._header = np.ndarray(
                (1 + metadata.local_size,),
                dtype=np.uint64,
                buffer=mapped,
                offset=32,
            )
            self._slots = np.ndarray(
                (), dtype=self._slot_dtype, buffer=mapped, offset=header_bytes
            )
            self._refs = self._slots["status"]["ref_cnt"]
            self._versions = self._slots["version"]
            self._num_tokens = self._slots["num_tokens"]
            self._hash_counts = self._slots["hash_count"]
            self._hashes = self._slots["hashes"]
            self._seen = np.zeros(count, dtype=np.uint64)
            self._seen_revision = 0
            if self._header[1 + self.local_rank]:
                raise ValueError("Shared CPU offload rank already attached.")
            self._header[1 + self.local_rank] = 1
            self._slots["status"]["chunk_id"][self._start : self._end] = np.arange(
                self._start, self._end
            )
            if self._header[1 : 1 + metadata.local_size].all():
                os.unlink(self._region.mmap_path)
            self._region._creator = False
        except BaseException:
            self._release_metadata()
            raise
        else:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        if self._lock_depth:
            yield
            return
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        self._lock_depth += 1
        try:
            self._refresh_hashes()
            yield
        finally:
            self._lock_depth -= 1
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _refresh_hashes(self) -> None:
        revision = int(self._header[0])
        if revision == self._seen_revision:
            return
        for slot in np.flatnonzero(self._versions != self._seen):
            block = self.blocks[slot]
            assert block.ref_cnt == 0
            super()._remove_cached_block_hashes(block)
            for i in range(int(self._hash_counts[slot])):
                key = BlockHashWithGroupId(self._hashes[slot, i].tobytes())
                num_tokens = int(self._num_tokens[slot]) if i == 0 else -1
                super()._insert_block_hash(
                    key, block, num_tokens if num_tokens >= 0 else None
                )
            self._seen[slot] = self._versions[slot]
        self._seen_revision = revision

    def _changed(self, slot: int) -> None:
        self._header[0] += 1
        self._versions[slot] = self._header[0]
        self._seen[slot] = self._versions[slot]

    def _sync_peer_pins(self) -> None:
        refs = self._refs[self._start + 1 : self._end]
        pinned = {
            self._start + 1 + int(i)
            for i in np.flatnonzero(refs > 0)
            if self.blocks[self._start + 1 + int(i)].ref_cnt == 0
        }
        for slot in pinned - self._peer_pinned:
            self.free_block_queue.remove(self.blocks[slot])
        for slot in self._peer_pinned - pinned:
            if self.blocks[slot].ref_cnt == 0:
                self.free_block_queue.append(self.blocks[slot])
        self._peer_pinned = pinned

    def get_num_free_blocks(self) -> int:
        with self.locked():
            self._sync_peer_pins()
            return super().get_num_free_blocks()

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        blocks = super().get_cached_block(block_hash, kv_cache_group_ids)
        if blocks is None:
            return None
        # A process can exit between invalidating hashes and publishing a new
        # revision. Check shared readiness even when the local index is current.
        for block in blocks:
            slot = block.block_id
            if (
                self._refs[slot] < 0
                or self._hash_counts[slot] == 0
                or self._versions[slot] != self._seen[slot]
            ):
                return None
        return blocks

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        with self.locked():
            blocks = super().get_new_blocks(num_blocks)
            for block in blocks:
                self._refs[block.block_id] = -1
            return blocks

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        slot = block.block_id
        assert self._start < slot < self._end
        self._hash_counts[slot] = 0
        self._changed(slot)
        return super()._maybe_evict_cached_block(block)

    def _insert_block_hash(
        self, key: BlockHashWithGroupId, block: KVCacheBlock, num_tokens: int | None
    ) -> None:
        with self.locked():
            slot = block.block_id
            assert self._start < slot < self._end
            if self.cached_block_hash_to_block.contain(key, slot):
                return
            count = int(self._hash_counts[slot])
            if count >= self._hashes.shape[1] or len(key) != self._hashes.shape[2]:
                raise ValueError(
                    "Shared CPU offload slot hash metadata exceeds its layout."
                )
            super()._insert_block_hash(key, block, num_tokens)
            self._hashes[slot, count] = np.frombuffer(key, dtype=np.uint8)
            if count == 0:
                self._num_tokens[slot] = num_tokens if num_tokens is not None else -1
                self._refs[slot] = block.ref_cnt
            self._hash_counts[slot] = count + 1
            self._changed(slot)

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        with self.locked():
            for block in blocks:
                if block.is_null:
                    continue
                slot = block.block_id
                assert self._refs[slot] >= 0
                if (
                    self._start < slot < self._end
                    and block.ref_cnt == 0
                    and slot not in self._peer_pinned
                ):
                    self.free_block_queue.remove(block)
                block.ref_cnt += 1
                self._refs[slot] += 1

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        with self.locked():
            owned = []
            for block in ordered_blocks:
                if block.is_null:
                    continue
                slot = block.block_id
                assert block.ref_cnt > 0
                refs = self._refs[slot]
                self._refs[slot] = 0 if refs == -1 else refs - 1
                if self._start < slot < self._end:
                    owned.append(block)
                else:
                    block.ref_cnt -= 1
            super().free_blocks(owned)
            self._peer_pinned.difference_update(
                b.block_id for b in owned if b.ref_cnt == 0
            )
            self._sync_peer_pins()

    def reset_prefix_cache(self) -> bool:
        with self.locked():
            if np.any(self._refs != 0):
                return False
            self._hash_counts[:] = 0
            self._header[0] += 1
            self._versions[:] = self._header[0]
            self._seen[:] = self._versions
            self._seen_revision = int(self._header[0])
            self.cached_block_hash_to_block = BlockHashToBlockMap()
            self.cached_block_hashes_by_block.clear()
            for block in self.blocks:
                block.reset_hash()
            if self.enable_kv_cache_events:
                self.kv_event_queue.append(AllBlocksCleared())
            return True

    def get_usage(self) -> float:
        return 1 - self.get_num_free_blocks() / (self.metadata.blocks_per_rank - 1)

    def _release_metadata(self) -> None:
        for name in (
            "_header",
            "_slots",
            "_refs",
            "_versions",
            "_num_tokens",
            "_hash_counts",
            "_hashes",
        ):
            setattr(self, name, None)
        if hasattr(self, "_region"):
            self._region.cleanup()

    def close(self) -> None:
        if self._closed:
            return
        self._release_metadata()
        self._closed = True
