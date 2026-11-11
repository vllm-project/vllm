# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDS offloading manager — unlimited capacity, no eviction."""

import os
import sys
from collections.abc import Collection, Iterable

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.fs.gds.common import GDSLoadStoreSpec

logger = init_logger(__name__)


class GDSOffloadingManager(CPUOffloadingManager):
    """Tracks offloaded blocks on GDS storage.

    Inherits ref-counting and load/store lifecycle from
    CPUOffloadingManager.  Overrides capacity to be unlimited (no
    eviction) and produces GDSLoadStoreSpec (keys) instead of
    CPULoadStoreSpec (block IDs).

    On lookup miss, falls back to a filesystem check so blocks
    persisted by previous sessions are discovered.
    """

    def __init__(
        self,
        file_mapper: FileMapper,
        enable_events: bool = False,
        read_only: bool = False,
    ):
        super().__init__(
            num_chunks=sys.maxsize,
            cache_policy="lru",
            enable_events=enable_events,
            store_threshold=0,
        )
        self.medium = GDSLoadStoreSpec.medium()
        self._file_mapper = file_mapper
        self._read_only = read_only

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = super().lookup(key, req_context)
        if result != LookupResult.MISS:
            return result
        # Check if block exists on disk from a previous session
        file_path = self._file_mapper.get_file_name(key)
        if os.path.exists(file_path):
            chunks = self._allocate_chunks([key])
            chunk = chunks[0]
            chunk.ref_cnt = 0
            self._policy.insert(key, chunk, req_context)
            self._num_evictable_cache_chunks += 1
            self._policy.mark_evictable(key)
            return LookupResult.HIT
        return LookupResult.MISS

    @override
    def _get_num_free_chunks(self) -> int:
        return sys.maxsize

    @override
    def _allocate_chunks(self, keys: list[OffloadKey]) -> list[ChunkStatus]:
        chunks: list[ChunkStatus] = []
        for _ in keys:
            chunks.append(ChunkStatus(self._num_allocated_chunks))
            self._num_allocated_chunks += 1
        return chunks

    @override
    def _get_load_store_spec(  # type: ignore[override]
        self,
        keys: Iterable[OffloadKey],
        chunks: Iterable[ChunkStatus],
    ) -> LoadStoreSpec:
        return GDSLoadStoreSpec(list(keys))

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        logger.debug(
            "prepare_load: %d keys, req=%s",
            len(list(keys)),
            req_context.req_id,
        )
        return super().prepare_load(keys, req_context)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        logger.debug(
            "prepare_store: %d keys, req=%s",
            len(list(keys)),
            req_context.req_id,
        )
        if self._read_only:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=GDSLoadStoreSpec([]),
                evicted_keys=[],
            )
        keys_to_store = [k for k in keys if self._policy.get(k) is None]
        if not keys_to_store:
            logger.debug(
                "prepare_store: %d keys (0 new, 0 chunks), req=%s",
                len(list(keys)),
                req_context.req_id,
            )
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=GDSLoadStoreSpec([]),
                evicted_keys=[],
            )
        chunks = self._allocate_chunks(keys_to_store)
        logger.debug(
            "prepare_store: %d keys (%d new, %d chunks), req=%s",
            len(list(keys)),
            len(keys_to_store),
            len(chunks),
            req_context.req_id,
        )
        for key, chunk in zip(keys_to_store, chunks):
            self._policy.insert(key, chunk, req_context)
        store_spec = self._get_load_store_spec(keys_to_store, chunks)
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=[],
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        logger.debug(
            "complete_store: %d keys, success=%s, req=%s",
            len(list(keys)),
            success,
            req_context.req_id,
        )
        super().complete_store(keys, req_context, success)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        logger.debug(
            "complete_load: %d keys, req=%s",
            len(list(keys)),
            req_context.req_id,
        )
        super().complete_load(keys, req_context)
