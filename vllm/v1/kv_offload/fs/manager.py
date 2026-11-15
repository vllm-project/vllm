# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for direct device <-> filesystem offloading."""

from collections import OrderedDict
from collections.abc import Collection

from typing_extensions import override

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.lookup import FsAsyncLookupManager
from vllm.v1.kv_offload.fs.worker import FSLoadStoreSpec
from vllm.v1.metrics.cache_hit_source import CacheHitSource

DEFAULT_INDEX_SIZE = 100_000


class FSOffloadingManager(OffloadingManager):
    """Tracks content-addressed KV files that the worker moves to and from
    device memory directly, unlike FileSystemTierManager, which stages
    through the CPU tier.

    Keys are looked up on disk asynchronously, so files written by other
    instances or earlier runs are reused. Keys found on disk or stored by
    this instance are remembered in a bounded LRU index, so later requests
    hit without probing again. Dropping a key from the index leaves its
    file on disk.
    """

    def __init__(
        self,
        file_mapper: FileMapper,
        index_size: int = DEFAULT_INDEX_SIZE,
        read_only: bool = True,
    ):
        assert index_size > 0
        self.medium = Medium.STORAGE
        self._index_size = index_size
        self._read_only = read_only
        self._index: OrderedDict[OffloadKey, None] = OrderedDict()
        # Keys whose store is in flight; their files may be incomplete.
        self._storing: set[OffloadKey] = set()
        self._fs_lookup = FsAsyncLookupManager(file_mapper, tier_type="fs")

    def _remember(self, key: OffloadKey) -> None:
        self._index[key] = None
        self._index.move_to_end(key)
        if len(self._index) > self._index_size:
            self._index.popitem(last=False)

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if key in self._index:
            self._index.move_to_end(key)
            return LookupResult.HIT
        if key in self._storing:
            return LookupResult.HIT_PENDING
        found = self._fs_lookup.lookup(key, req_context)
        if found is None:
            return LookupResult.RETRY
        if not found:
            return LookupResult.MISS
        self._remember(key)
        return LookupResult.HIT

    @override
    def get_load_source(
        self, key: OffloadKey, req_context: ReqContext
    ) -> CacheHitSource:
        return CacheHitSource.DISK

    @override
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        return FSLoadStoreSpec(list(keys))

    @override
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        keys_to_store = (
            []
            if self._read_only
            else [k for k in keys if k not in self._index and k not in self._storing]
        )
        self._storing.update(keys_to_store)
        # Empty, not None: None counts as an allocation failure.
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=FSLoadStoreSpec(keys_to_store),
            evicted_keys=[],
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        for key in keys:
            self._storing.discard(key)
            if success:
                self._remember(key)
        if success:
            self._fs_lookup.mark_present(keys)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._fs_lookup.flush()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        self._fs_lookup.cleanup(req_context.req_id)

    @override
    def reset_cache(self) -> None:
        # Pre-reset jobs never reach complete_store, so drop them here too.
        self._index.clear()
        self._storing.clear()

    @override
    def shutdown(self) -> None:
        self._fs_lookup.shutdown()
