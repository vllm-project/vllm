# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    ReqId,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventsTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.state import (
    RequestOffloadState,
    SchedulerOffloadConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.base import (  # noqa: E501
    StoreJobSink,
    StoreTriggerHandler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.candidates import (  # noqa: E501
    block_key,
    select_store_candidates,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadKey,
    OffloadPolicy,
    StoreTrigger,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class GpuChunk:
    """A group chunk to store when the block pool gives out one of its blocks."""

    key: OffloadKey
    config_idx: int
    chunk_idx: int
    # The GPU block ids of the chunk, without the null blocks.
    block_ids: tuple[int, ...]
    # The key of the data in each block of block_ids.
    block_keys: tuple[OffloadKey, ...]
    # The position of block_ids[0] inside the chunk.
    start_idx: int


class OverwriteMap:
    """The group chunks on GPU blocks, to store before a block gets new data."""

    def __init__(self) -> None:
        # block_id -> the group chunks that hold the block
        self.by_block: dict[int, list[GpuChunk]] = {}

    def add(self, chunks: Iterable[GpuChunk]) -> None:
        """Remember which chunks sit on each block."""
        for chunk in chunks:
            # A chunk with no non-null block has nothing to store.
            if not chunk.block_ids:
                continue
            # A prefix hit offers the chunk of an existing entry again.
            if chunk in self.by_block.get(chunk.block_ids[-1], ()):
                continue
            for block_id in chunk.block_ids:
                self.by_block.setdefault(block_id, []).append(chunk)

    def take_overwritten(
        self, block_keys: dict[int, OffloadKey | None]
    ) -> list[GpuChunk]:
        """Remove and return the chunks whose blocks get different data.

        block_keys maps a block id to the key that its new owner expects, or
        to None for new data.
        """
        overwritten: list[GpuChunk] = []
        for block_id, expected_key in block_keys.items():
            for chunk in tuple(self.by_block.get(block_id, ())):
                if chunk.block_keys[chunk.block_ids.index(block_id)] == expected_key:
                    continue
                for chunk_block_id in chunk.block_ids:
                    chunks = self.by_block[chunk_block_id]
                    chunks.remove(chunk)
                    if not chunks:
                        del self.by_block[chunk_block_id]
                overwritten.append(chunk)
        return overwritten

    def clear(self) -> None:
        """Forget every chunk."""
        self.by_block.clear()


class OnOverwriteStoreHandler(StoreTriggerHandler):
    """Offers a chunk when the block pool gives one of its blocks away
    (StoreTrigger.ON_OVERWRITE)."""

    trigger = StoreTrigger.ON_OVERWRITE

    @classmethod
    @override
    def check_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        extra_config: Mapping[str, Any],
        events_tracker: OffloadingEventsTracker,
    ) -> None:
        """Refuse a config that an ON_OVERWRITE store cannot serve."""
        if events_tracker.self_describing_enabled:
            # A self-describing payload carries the tokens and the parent hash
            # of the chunk, read from the request that stored it. The request
            # of this store is usually finished by the time its block is
            # reused, so the payload cannot be built.
            raise ValueError(
                "self_describing_kv_events is not supported with ON_OVERWRITE stores"
            )
        if vllm_config.use_v2_model_runner and kv_cache_config.needs_kv_cache_zeroing:
            # The V2 model runner zeroes new blocks and applies the CoW copies
            # in update_requests, before handle_preemptions flushes the
            # ON_OVERWRITE jobs. Such a store would read zeroed or copied data
            # and save it under a valid key.
            raise ValueError(
                "ON_OVERWRITE stores are not supported with the V2 model runner "
                "when the KV cache needs zeroing (mixed-precision KV cache). "
                "Set VLLM_USE_V2_MODEL_RUNNER=0."
            )

    def __init__(
        self,
        config: SchedulerOffloadConfig,
        events_tracker: OffloadingEventsTracker,
    ):
        super().__init__(config, events_tracker)
        # A store waits until the data is about to be lost: the map holds the
        # chunks that sit on GPU blocks, and a chunk is offered when one of
        # its blocks gets different data.
        self._overwrite_map = OverwriteMap()
        # req_id -> per group, the first chunk not recorded in the map yet.
        self._first_unrecorded_chunk_idxs: dict[ReqId, list[int]] = {}
        # req_id -> the chunks of the scheduled load, added at its completion.
        self._pending_loads: dict[ReqId, list[GpuChunk]] = {}
        # block_id -> the key that the new owner expects in the block, or None
        # for new data, for the blocks given out in this step. Rebuilt every
        # step, in on_request_blocks_allocated and on_blocks_allocated.
        self._allocated_block_keys: dict[int, OffloadKey | None] = {}

    @override
    def on_request_blocks_allocated(
        self,
        req_status: RequestOffloadState,
        block_ids: Sequence[Sequence[int]],
        start_block_idxs: Sequence[int],
    ) -> None:
        """Record the key that the new owner of each block expects."""
        req = req_status.req
        for group_config, group_block_ids, start in zip(
            self._config.kv_group_configs, block_ids, start_block_idxs
        ):
            num_hashed_blocks = len(req.block_hashes) // group_config.hashes_per_block
            for idx, bid in enumerate(group_block_ids, start):
                # A null block holds nothing, and a block whose hash is not
                # complete cannot be a prefix hit. on_blocks_allocated marks
                # both as new data.
                if bid != 0 and idx < num_hashed_blocks:
                    self._allocated_block_keys[bid] = block_key(req, group_config, idx)

    @override
    def on_blocks_allocated(self, block_ids: Collection[int]) -> None:
        """Mark the blocks of this step that no request claimed as new data."""
        # A block of a group that is not offloaded, a block with no hash yet,
        # or a copy destination gets new data. setdefault keeps the key that
        # on_request_blocks_allocated recorded for the same block.
        for bid in block_ids:
            self._allocated_block_keys.setdefault(bid, None)

    @override
    def build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
        req_states: Mapping[ReqId, RequestOffloadState],
        sink: StoreJobSink,
    ) -> None:
        """Record this step's chunks, then store the chunks it overwrites."""
        # Record the computed chunks before the overwritten ones, so a request
        # that finishes in this step can offer a chunk on a block that the
        # pool gives out in the same step.
        self._record_computed_chunks(scheduler_output, req_states)
        self._build_overwrite_store_jobs(sink)
        self._allocated_block_keys = {}

    @override
    def on_load_scheduled(
        self,
        req_status: RequestOffloadState,
        num_cached_tokens: int,
        load_chunk_ranges: Sequence[tuple[int, int]],
        load_block_ids: Sequence[Sequence[int]],
    ) -> None:
        """Record the chunks a load brings in, to add when the load completes."""
        # A load skips the computation of its chunks, so they come in the map
        # when the load completes. A request has one load in flight, and the
        # job completes before the next load of the request starts.
        req = req_status.req
        loaded_chunks: list[GpuChunk] = []
        for config_idx, (start, end) in enumerate(load_chunk_ranges):
            load_block_ids_of_group = load_block_ids[config_idx]
            offload_keys = req_status.group_states[config_idx].offload_keys
            for chunk_idx in range(start, end):
                loaded_chunks.append(
                    self._make_gpu_chunk(
                        req,
                        config_idx,
                        chunk_idx,
                        offload_keys[chunk_idx],
                        load_block_ids_of_group,
                    )
                )
        self._pending_loads[req.request_id] = loaded_chunks
        # A loaded chunk is not computed by this request. The request-level
        # policy stores the whole request anyway, so nothing is skipped there.
        if req_status.offloading_context.policy != OffloadPolicy.CHUNK_LEVEL:
            return
        start_chunk_idxs = self._record_start_chunk_idxs(req_status)
        for group_idx, group_config in enumerate(self._config.kv_group_configs):
            start_chunk_idxs[group_idx] = max(
                start_chunk_idxs[group_idx],
                cdiv(num_cached_tokens, group_config.tokens_per_chunk),
            )

    @override
    def first_live_chunk_idx(
        self, req_status: RequestOffloadState, config_idx: int
    ) -> int | None:
        """The first chunk of the group that is not in the map yet."""
        start_chunk_idxs = self._first_unrecorded_chunk_idxs.get(
            req_status.req.request_id
        )
        if start_chunk_idxs is None:
            return 0
        return start_chunk_idxs[config_idx]

    @override
    def on_load_complete(self, job_status: TransferJobStatus) -> None:
        assert job_status.req_id is not None
        loaded_chunks = self._pending_loads.pop(job_status.req_id, None)
        # A failed load never wrote the blocks, so they hold no chunk.
        if loaded_chunks and not job_status.failed:
            self._overwrite_map.add(loaded_chunks)

    @override
    def on_request_removed(self, req_id: ReqId) -> None:
        """Forget where the request recorded; its chunks stay in the map."""
        self._first_unrecorded_chunk_idxs.pop(req_id, None)
        self._pending_loads.pop(req_id, None)

    @override
    def on_request_restarted(self, req_id: ReqId) -> None:
        """Record the request from chunk 0 again, as it recomputes them."""
        # The blocks of a pending load are gone with the preemption.
        self._first_unrecorded_chunk_idxs.pop(req_id, None)
        self._pending_loads.pop(req_id, None)

    @override
    def reset(self, req_states: Mapping[ReqId, RequestOffloadState]) -> None:
        """Drop the map, and record the active requests from chunk 0."""
        # The tracked blocks belonged to the cache generation that was reset,
        # and no pending load will ever report.
        self._overwrite_map.clear()
        self._pending_loads.clear()
        self._allocated_block_keys = {}
        self._first_unrecorded_chunk_idxs = {
            req_id: [0] * len(req_status.group_states)
            for req_id, req_status in req_states.items()
        }

    def _record_start_chunk_idxs(self, req_status: RequestOffloadState) -> list[int]:
        """The per-group chunk where recording of this request continues."""
        req_id = req_status.req.request_id
        start_chunk_idxs = self._first_unrecorded_chunk_idxs.get(req_id)
        if start_chunk_idxs is None:
            start_chunk_idxs = [0] * len(req_status.group_states)
            self._first_unrecorded_chunk_idxs[req_id] = start_chunk_idxs
        return start_chunk_idxs

    def _record_computed_chunks(
        self,
        scheduler_output: SchedulerOutput,
        req_states: Mapping[ReqId, RequestOffloadState],
    ) -> None:
        """Record this step's computed chunks and advance each group."""
        for req_id in chain(
            scheduler_output.num_scheduled_tokens,
            # A request that finishes in this step computes its last chunks
            # here, so they must be recorded before the overwrites are taken.
            scheduler_output.finished_req_ids or (),
        ):
            req_status = req_states.get(req_id)
            if req_status is None:
                continue
            start_chunk_idxs = self._record_start_chunk_idxs(req_status)
            candidates = select_store_candidates(
                self._config, scheduler_output, req_status, start_chunk_idxs
            )
            req = req_status.req
            chunks = [
                self._make_gpu_chunk(
                    req,
                    config_idx,
                    chunk_idx,
                    key,
                    req_status.group_states[config_idx].block_ids,
                )
                for key, (config_idx, chunk_idx) in zip(
                    candidates.keys, candidates.chunk_positions
                )
            ]
            self._overwrite_map.add(chunks)
            # Advance past the whole range that was considered, skipped
            # chunks included, so the next step starts where this one did.
            for group_idx, (_, num_chunks) in enumerate(candidates.group_ranges):
                start_chunk_idxs[group_idx] = max(
                    start_chunk_idxs[group_idx], num_chunks
                )

    def _make_gpu_chunk(
        self,
        req: Request,
        config_idx: int,
        chunk_idx: int,
        key: OffloadKey,
        group_block_ids: Sequence[int],
    ) -> GpuChunk:
        """The chunk at a position, on its non-null GPU blocks."""
        # A partial chunk can start and end on null blocks, and the worker
        # needs the offset of the first one inside the chunk.
        start = chunk_idx * self._config.blocks_per_chunk
        block_idxs = [
            idx
            for idx in range(start, start + self._config.blocks_per_chunk)
            if group_block_ids[idx]
        ]
        group_config = self._config.kv_group_configs[config_idx]
        return GpuChunk(
            key,
            config_idx,
            chunk_idx,
            tuple(group_block_ids[idx] for idx in block_idxs),
            tuple(block_key(req, group_config, idx) for idx in block_idxs),
            block_idxs[0] if block_idxs else 0,
        )

    def _build_overwrite_store_jobs(self, sink: StoreJobSink) -> None:
        """Offer the chunks whose blocks get new data in this step."""
        # Two block sets can hold the same key, for example after two requests
        # computed the same prefix in one step. Store one copy of each key.
        unique_chunks: dict[OffloadKey, GpuChunk] = {}
        for chunk in self._overwrite_map.take_overwritten(self._allocated_block_keys):
            other = unique_chunks.get(chunk.key)
            if other is None or len(chunk.block_ids) > len(other.block_ids):
                unique_chunks[chunk.key] = chunk
        # Group-major for the GPU spec. Deepest first, so that the prefix gets
        # the newest rank.
        chunks = sorted(
            unique_chunks.values(),
            key=lambda chunk: (chunk.config_idx, -chunk.chunk_idx),
        )
        # The worker skips null blocks only at the start of each group. So a
        # chunk with null blocks goes in a job of its own.
        blocks_per_chunk = self._config.blocks_per_chunk
        full_chunks = [
            chunk for chunk in chunks if len(chunk.block_ids) == blocks_per_chunk
        ]
        batches = [full_chunks] if full_chunks else []
        batches += [
            [chunk] for chunk in chunks if len(chunk.block_ids) < blocks_per_chunk
        ]

        for batch in batches:
            store_output = sink.prepare_store([chunk.key for chunk in batch], None)
            if store_output is None:
                logger.warning("Cannot store %d overwritten chunks", len(batch))
                continue
            accepted_keys = set(store_output.keys_to_store)
            accepted = [chunk for chunk in batch if chunk.key in accepted_keys]
            if not accepted:
                continue
            sink.add_job(
                None,
                [chunk.key for chunk in accepted],
                self._batch_src_spec(accepted),
                store_output.store_spec,
            )

    def _batch_src_spec(self, chunks: list[GpuChunk]) -> GPULoadStoreSpec:
        """The GPU source spec of one job."""
        num_groups = len(self._config.kv_group_configs)
        src_block_ids: list[int] = []
        group_sizes = [0] * num_groups
        block_indices = [0] * num_groups
        for chunk in chunks:
            if not group_sizes[chunk.config_idx]:
                # The blocks of a batch are several chunks, deepest first, so
                # they are not contiguous in the request. Only the offset of
                # the first block inside its chunk matters: the worker copies
                # the group's blocks consecutively from there.
                block_indices[chunk.config_idx] = (
                    chunk.chunk_idx * self._config.blocks_per_chunk + chunk.start_idx
                )
            group_sizes[chunk.config_idx] += len(chunk.block_ids)
            src_block_ids.extend(chunk.block_ids)
        return GPULoadStoreSpec(
            src_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )
