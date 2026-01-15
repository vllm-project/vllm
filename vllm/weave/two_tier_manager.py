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
    DRAM = "DRAM"
    CXL = "CXL"


@dataclass
class _Entry:
    dram: BlockStatus | None
    cxl: BlockStatus | None
    committed: bool


class TwoTierOffloadingManager(OffloadingManager):
    def __init__(
        self,
        *,
        dram_backend: Backend,
        cxl_backend: Backend,
        enable_events: bool = False,
    ):
        self.dram_backend = dram_backend
        self.cxl_backend = cxl_backend
        self.entries: dict[BlockHash, _Entry] = {}
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int:
        hit_count = 0
        for block_hash in block_hashes:
            entry = self.entries.get(block_hash)
            if entry is None or entry.dram is None:
                break
            if not entry.dram.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        blocks: list[BlockStatus] = []
        for block_hash in block_hashes:
            entry = self.entries.get(block_hash)
            assert entry is not None
            assert entry.dram is not None
            assert entry.dram.is_ready
            entry.dram.ref_cnt += 1
            blocks.append(entry.dram)
        return self.dram_backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]):
        return

    def complete_load(self, block_hashes: Iterable[BlockHash]):
        for block_hash in block_hashes:
            entry = self.entries.get(block_hash)
            if entry is None:
                continue
            if entry.dram is None:
                continue
            if entry.dram.ref_cnt > 0:
                entry.dram.ref_cnt -= 1

    def prepare_store(self, block_hashes: Iterable[BlockHash]) -> PrepareStoreOutput | None:
        block_hashes_to_store: list[BlockHash] = []
        for block_hash in block_hashes:
            entry = self.entries.get(block_hash)
            if entry is None:
                block_hashes_to_store.append(block_hash)
                continue
            if entry.dram is not None:
                continue
            if entry.cxl is not None and entry.committed:
                continue
            block_hashes_to_store.append(block_hash)

        if not block_hashes_to_store:
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=self.dram_backend.get_load_store_spec([], []),
                block_hashes_evicted=[],
            )

        num_needed = len(block_hashes_to_store)
        if self.dram_backend.get_num_free_blocks() < num_needed:
            return None

        blocks = self.dram_backend.allocate_blocks(block_hashes_to_store)
        for block_hash, block in zip(block_hashes_to_store, blocks):
            prior = self.entries.get(block_hash)
            self.entries[block_hash] = _Entry(
                dram=block,
                cxl=prior.cxl if prior is not None else None,
                committed=prior.committed if prior is not None else False,
            )

        store_spec = self.dram_backend.get_load_store_spec(block_hashes_to_store, blocks)
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
                if entry.dram is None:
                    continue
                if not entry.dram.is_ready:
                    entry.dram.ref_cnt = 0
                    stored.append(block_hash)
        else:
            for block_hash in block_hashes:
                entry = self.entries.get(block_hash)
                if entry is None:
                    continue
                if entry.dram is None:
                    continue
                if not entry.dram.is_ready:
                    self.dram_backend.free(entry.dram)
                    entry.dram = None

                if entry.dram is None and entry.cxl is None:
                    del self.entries[block_hash]

        if stored and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored,
                    block_size=self.dram_backend.block_size,
                    medium=self.dram_backend.medium,
                    removed=False,
                )
            )

    def prepare_flush(self, block_hashes: Iterable[BlockHash]) -> tuple[LoadStoreSpec, LoadStoreSpec] | None:
        src_blocks: list[BlockStatus] = []
        block_hashes_list = list(block_hashes)
        for block_hash in block_hashes_list:
            entry = self.entries.get(block_hash)
            if entry is None or entry.dram is None:
                return None
            if entry.committed:
                return None
            if entry.cxl is not None:
                return None
            if not entry.dram.is_ready:
                return None
            entry.dram.ref_cnt += 1
            src_blocks.append(entry.dram)

        if self.cxl_backend.get_num_free_blocks() < len(src_blocks):
            for block_hash in block_hashes_list:
                entry = self.entries.get(block_hash)
                assert entry is not None and entry.dram is not None
                entry.dram.ref_cnt -= 1
            return None

        dst_blocks = self.cxl_backend.allocate_blocks(block_hashes_list)
        for block_hash, dst in zip(block_hashes_list, dst_blocks):
            entry = self.entries[block_hash]
            entry.cxl = dst

        src_spec = self.dram_backend.get_load_store_spec(block_hashes_list, src_blocks)
        dst_spec = self.cxl_backend.get_load_store_spec(block_hashes_list, dst_blocks)
        return src_spec, dst_spec

    def complete_flush(self, block_hashes: Iterable[BlockHash], success: bool = True):
        for block_hash in block_hashes:
            entry = self.entries.get(block_hash)
            if entry is None:
                continue

            if entry.dram is not None and entry.dram.ref_cnt > 0:
                entry.dram.ref_cnt -= 1

            if entry.cxl is None:
                continue

            if success and not entry.cxl.is_ready:
                entry.cxl.ref_cnt = 0
                entry.committed = True
            elif not success and not entry.cxl.is_ready:
                self.cxl_backend.free(entry.cxl)
                entry.cxl = None

            if entry.dram is None and entry.cxl is None:
                del self.entries[block_hash]

    def prepare_promotion(self, block_hashes: Iterable[BlockHash]) -> tuple[LoadStoreSpec, LoadStoreSpec] | None:
        src_blocks: list[BlockStatus] = []
        block_hashes_list = list(block_hashes)
        for block_hash in block_hashes_list:
            entry = self.entries.get(block_hash)
            if entry is None or entry.cxl is None or not entry.committed:
                return None
            if entry.dram is not None:
                return None
            if not entry.cxl.is_ready:
                return None
            entry.cxl.ref_cnt += 1
            src_blocks.append(entry.cxl)

        if self.dram_backend.get_num_free_blocks() < len(src_blocks):
            for block_hash in block_hashes_list:
                entry = self.entries.get(block_hash)
                assert entry is not None and entry.cxl is not None
                entry.cxl.ref_cnt -= 1
            return None

        dst_blocks = self.dram_backend.allocate_blocks(block_hashes_list)
        for block_hash, dst in zip(block_hashes_list, dst_blocks):
            entry = self.entries[block_hash]
            entry.dram = dst

        src_spec = self.cxl_backend.get_load_store_spec(block_hashes_list, src_blocks)
        dst_spec = self.dram_backend.get_load_store_spec(block_hashes_list, dst_blocks)
        return src_spec, dst_spec

    def complete_promotion(self, block_hashes: Iterable[BlockHash], success: bool = True):
        for block_hash in block_hashes:
            entry = self.entries.get(block_hash)
            if entry is None:
                continue

            if entry.cxl is not None and entry.cxl.ref_cnt > 0:
                entry.cxl.ref_cnt -= 1

            if entry.dram is None:
                continue

            if success and not entry.dram.is_ready:
                entry.dram.ref_cnt = 0
            elif not success and not entry.dram.is_ready:
                self.dram_backend.free(entry.dram)
                entry.dram = None

            if entry.dram is None and entry.cxl is None:
                del self.entries[block_hash]

    def probe_cxl(self, block_hashes: Iterable[BlockHash]) -> list[bool]:
        return [
            (entry is not None and entry.cxl is not None and entry.committed)
            for entry in (self.entries.get(b) for b in block_hashes)
        ]

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()
