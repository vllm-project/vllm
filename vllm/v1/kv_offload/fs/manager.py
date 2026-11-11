# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for direct device <-> filesystem offloading."""

from collections.abc import Collection, Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.worker import FSLoadStoreSpec
from vllm.v1.kv_offload.tiering.fs.manager import FsAsyncLookupManager
from vllm.v1.metrics.cache_hit_source import CacheHitSource


class FSOffloadingManager(CPUOffloadingManager):
    """Tracks content-addressed KV files that the worker moves to and from
    device memory directly, unlike FileSystemTierManager, which stages
    through the CPU tier.

    The CPU manager's index holds the keys this instance knows about, with
    num_chunks capping its size. Keys missing from the index are looked up
    on disk asynchronously, and found files are adopted into the index, so
    files written by other instances or earlier runs are reused. Evicting a
    key only drops it from the index; its file stays on disk.
    """

    def __init__(self, file_mapper: FileMapper, read_only: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.medium = Medium.STORAGE
        self._read_only = read_only
        self._fs_lookup = FsAsyncLookupManager(file_mapper, tier_type="fs")

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = super().lookup(key, req_context)
        if result is not LookupResult.MISS:
            return result
        found = self._fs_lookup.lookup(key, req_context)
        if found is None:
            return LookupResult.RETRY
        if not found or not self._adopt(key, req_context):
            return LookupResult.MISS
        return LookupResult.HIT

    def _adopt(self, key: OffloadKey, req_context: ReqContext) -> bool:
        """Index an on-disk key as a ready, evictable chunk."""
        if self._get_num_free_chunks() == 0:
            evicted = self._policy.evict(1, protected=set())
            if evicted is None:
                return False
            self._num_evictable_cache_chunks -= len(evicted)
            for _, evicted_chunk in evicted:
                self._free_chunk(evicted_chunk)
        (chunk,) = self._allocate_chunks([key])
        self._policy.insert(key, chunk, req_context)
        chunk.ref_cnt = 0
        self._num_evictable_cache_chunks += 1
        self._policy.mark_evictable(key)
        return True

    @override
    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        chunks: Iterable[ChunkStatus],
    ) -> LoadStoreSpec:
        return FSLoadStoreSpec(list(keys))

    @override
    def get_load_source(
        self, key: OffloadKey, req_context: ReqContext
    ) -> CacheHitSource:
        return CacheHitSource.DISK

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        if self._read_only:
            # Empty, not None: None counts as an allocation failure.
            return PrepareStoreOutput(
                keys_to_store=[], store_spec=FSLoadStoreSpec([]), evicted_keys=[]
            )
        return super().prepare_store(keys, req_context)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        super().on_schedule_end(context)
        self._fs_lookup.flush()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        super().on_request_finished(req_context)
        self._fs_lookup.cleanup(req_context.req_id)

    @override
    def shutdown(self) -> None:
        self._fs_lookup.shutdown()
        super().shutdown()
