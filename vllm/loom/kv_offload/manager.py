from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus


class Tier(str, Enum):
    CXL = "CXL"


@dataclass
class _Entry:
    cxl: BlockStatus


class LoomManager(OffloadingManager):
    def __init__(
        self,
        *,
        cxl_backend: Backend,
        enable_events: bool = False,
    ):
        self.cxl_backend = cxl_backend
        self.entries: dict[BlockHash, _Entry] = {}
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        hit_count = 0
        for block_hash in block_hashes:
            entry = self.entries.get(block_hash)
            if entry is None:
                break
            if not entry.cxl.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        blocks: list[BlockStatus] = []
        for block_hash in block_hashes:
            entry = self.entries.get(block_hash)
            assert entry is not None
            assert entry.cxl.is_ready
            entry.cxl.ref_cnt += 1
            blocks.append(entry.cxl)
        return self.cxl_backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        return

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for block_hash in block_hashes:
            entry = self.entries.get(block_hash)
            if entry is None:
                continue
            if entry.cxl.ref_cnt > 0:
                entry.cxl.ref_cnt -= 1

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        block_hashes_to_store: list[BlockHash] = []
        for block_hash in block_hashes:
            entry = self.entries.get(block_hash)
            if entry is None:
                block_hashes_to_store.append(block_hash)
                continue
            continue

        if not block_hashes_to_store:
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=self.cxl_backend.get_load_store_spec([], []),
                block_hashes_evicted=[],
            )

        num_needed = len(block_hashes_to_store)
        if self.cxl_backend.get_num_free_blocks() < num_needed:
            return None

        blocks = self.cxl_backend.allocate_blocks(block_hashes_to_store)
        for block_hash, block in zip(block_hashes_to_store, blocks):
            self.entries[block_hash] = _Entry(cxl=block)

        store_spec = self.cxl_backend.get_load_store_spec(block_hashes_to_store, blocks)
        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=[],
        )

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True):
        stored: list[BlockHash] = []
        if success:
            for block_hash in block_hashes:
                entry = self.entries.get(block_hash)
                if entry is None:
                    continue
                if not entry.cxl.is_ready:
                    entry.cxl.ref_cnt = 0
                    stored.append(block_hash)
        else:
            for block_hash in block_hashes:
                entry = self.entries.get(block_hash)
                if entry is None:
                    continue
                if not entry.cxl.is_ready:
                    self.cxl_backend.free(entry.cxl)
                    del self.entries[block_hash]

        if stored and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored,
                    block_size=self.cxl_backend.block_size,
                    medium=self.cxl_backend.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()
