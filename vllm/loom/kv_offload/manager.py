from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.mediums import CXLLoadStoreSpec


class Tier(str, Enum):
    CXL = "CXL"


@dataclass
class _Entry:
    cxl: BlockStatus


@dataclass(frozen=True)
class SharedPrefixExtent:
    base_block_id: int
    num_blocks: int
    layout_version: int


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
        self.shared_prefix_directory: dict[tuple[int, int], SharedPrefixExtent] = {}

    def allocate_shared_prefix_extent(
        self,
        *,
        prefix_id: int,
        layer_group_id: int,
        num_blocks: int,
        layout_version: int,
    ) -> SharedPrefixExtent:
        key = (prefix_id, layer_group_id)
        if key in self.shared_prefix_directory:
            raise ValueError(
                "shared prefix extent already allocated: "
                f"prefix_id={prefix_id} layer_group_id={layer_group_id}"
            )

        allocate_extent = getattr(self.cxl_backend, "allocate_extent", None)
        if allocate_extent is None:
            raise RuntimeError(
                "CXL backend does not support allocate_extent(); "
                "unable to use bump allocator for shared prefixes."
            )

        base_block_id, num_blocks = allocate_extent(num_blocks)
        extent = SharedPrefixExtent(
            base_block_id=int(base_block_id),
            num_blocks=int(num_blocks),
            layout_version=int(layout_version),
        )
        self.shared_prefix_directory[key] = extent
        return extent

    def get_shared_prefix_extent(
        self, *, prefix_id: int, layer_group_id: int
    ) -> SharedPrefixExtent | None:
        return self.shared_prefix_directory.get((prefix_id, layer_group_id))

    def lookup_prefix(
        self,
        *,
        prefix_id: int,
        start_block_idx: int,
        num_blocks: int,
        extra: Mapping[str, Any] | None = None,
    ) -> int:
        if start_block_idx < 0:
            raise ValueError(f"start_block_idx must be >= 0, got {start_block_idx}")
        if num_blocks <= 0:
            return 0
        if start_block_idx >= num_blocks:
            return 0

        group_ids: set[int] = set()
        min_blocks: int | None = None
        for (pid, layer_group_id), extent in self.shared_prefix_directory.items():
            if pid != prefix_id:
                continue
            group_ids.add(int(layer_group_id))
            blocks = int(extent.num_blocks)
            min_blocks = blocks if min_blocks is None else min(min_blocks, blocks)

        if not group_ids:
            return 0

        max_group_id = max(group_ids)
        if min(group_ids) != 0 or len(group_ids) != max_group_id + 1:
            return 0

        if min_blocks is None or min_blocks < num_blocks:
            return 0

        return num_blocks - start_block_idx

    def prepare_load_prefix(
        self,
        *,
        prefix_id: int,
        start_block_idx: int,
        num_blocks: int,
        extra: Mapping[str, Any] | None = None,
    ) -> LoadStoreSpec:
        if start_block_idx < 0:
            raise ValueError(f"start_block_idx must be >= 0, got {start_block_idx}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        if start_block_idx >= num_blocks:
            raise ValueError(
                "start_block_idx must be < num_blocks for prepare_load_prefix: "
                f"start_block_idx={start_block_idx} num_blocks={num_blocks}"
            )

        group_ids: set[int] = set()
        for (pid, layer_group_id) in self.shared_prefix_directory:
            if pid == prefix_id:
                group_ids.add(int(layer_group_id))
        if not group_ids:
            raise ValueError(f"shared prefix not found in directory: prefix_id={prefix_id}")

        # Phase-2 MVP: only support a single layer-group (group_id=0) so that
        # block-wise swapping does not overwrite non-target layers.
        if group_ids != {0}:
            raise RuntimeError(
                "prepare_load_prefix currently requires a single layer-group (id=0); "
                f"got group_ids={sorted(group_ids)} for prefix_id={prefix_id}"
            )

        extent = self.get_shared_prefix_extent(prefix_id=prefix_id, layer_group_id=0)
        if extent is None:
            raise ValueError(
                f"shared prefix extent missing for prefix_id={prefix_id} layer_group_id=0"
            )
        if int(extent.num_blocks) < num_blocks:
            raise ValueError(
                "shared prefix extent too small for requested load: "
                f"prefix_id={prefix_id} extent_blocks={extent.num_blocks} need_num_blocks={num_blocks}"
            )

        base = int(extent.base_block_id)
        block_ids = list(range(base + start_block_idx, base + num_blocks))
        return CXLLoadStoreSpec(block_ids)

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
