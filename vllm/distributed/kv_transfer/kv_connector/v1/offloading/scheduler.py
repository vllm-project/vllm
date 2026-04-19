# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, NamedTuple

from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.abstract import (
    OffloadingManager,
    OffloadKey,
    get_offload_block_hash,
    make_offload_key,
)
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    gpu_block_size: int
    offloaded_block_size: int
    hash_block_size_factor: int


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    block_size_factor: int
    num_workers: int

    @classmethod
    def from_spec(cls, spec: OffloadingSpec) -> "SchedulerOffloadConfig":
        return cls(
            num_workers=spec.vllm_config.parallel_config.world_size,
            kv_group_configs=tuple(
                GroupOffloadConfig(
                    group_idx=idx,
                    gpu_block_size=gpu_block_size,
                    offloaded_block_size=gpu_block_size * spec.block_size_factor,
                    hash_block_size_factor=(
                        (gpu_block_size * spec.block_size_factor)
                        // spec.hash_block_size
                    ),
                )
                for idx, gpu_block_size in enumerate(spec.gpu_block_size)
            ),
            block_size_factor=spec.block_size_factor,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # index of next block (of size offloaded_block_size) to offload
    next_stored_block_idx: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    load_job: int | None = None
    store_jobs: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hash_block_size_factor * len(group_state.offload_keys)
                + group_config.hash_block_size_factor
                - 1,
                None,
                group_config.hash_block_size_factor,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def is_idle(self) -> bool:
        return self.req.is_finished() and self.load_job is None and not self.store_jobs


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.config = SchedulerOffloadConfig.from_spec(spec)
        self.manager: OffloadingManager = spec.get_manager()

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[OffloadKey] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        if req_status := self._req_status.get(request.request_id):
            # make sure block IDs are cleared
            for group_state in req_status.group_states:
                group_state.block_ids.clear()
        else:
            req_status = RequestOffloadState(config=self.config, req=request)
            req_status.update_offload_keys()
            self._req_status[request.request_id] = req_status

        req_status.num_locally_computed_tokens = num_computed_tokens

        # Below assertions will be removed once this function supports HMA
        assert len(self.config.kv_group_configs) == 1
        assert len(req_status.group_states) == 1
        group_config = self.config.kv_group_configs[0]
        group_state = req_status.group_states[0]

        num_blocks = request.num_tokens // group_config.offloaded_block_size

        assert len(request.block_hashes) // self.config.block_size_factor == num_blocks
        offload_keys = group_state.offload_keys

        self.manager.touch(offload_keys)

        full_block_tokens = group_config.offloaded_block_size * num_blocks
        if full_block_tokens - num_computed_tokens < group_config.offloaded_block_size:
            # we can load less than a block, skip
            return 0, False

        start_block_idx = num_computed_tokens // group_config.offloaded_block_size
        hits = self.manager.lookup(offload_keys[start_block_idx:])
        if hits is None:
            # indicates a lookup that should be tried later
            return None, False
        if hits == 0:
            return 0, False

        num_hit_tokens = (
            group_config.offloaded_block_size * (start_block_idx + hits)
            - num_computed_tokens
        )
        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            request.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        if num_hit_tokens < group_config.offloaded_block_size:
            return 0, False

        if self._blocks_being_loaded and any(
            key in self._blocks_being_loaded
            for key in offload_keys[start_block_idx : start_block_idx + hits]
        ):
            # hit blocks are being loaded, delay request
            logger.debug(
                "Delaying request %s since some of its blocks are already being loaded",
                request.request_id,
            )
            return None, False

        return num_hit_tokens, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]
        block_groups = blocks.get_block_ids()

        # Below assertions will be removed once this function supports HMA
        assert len(self.config.kv_group_configs) == 1
        assert len(req_status.group_states) == 1
        assert len(block_groups) == 1
        block_ids = block_groups[0]
        group_config = self.config.kv_group_configs[0]
        group_state = req_status.group_states[0]

        num_computed_gpu_blocks = sum(
            block.block_hash is not None for block in blocks.blocks[0]
        )
        num_computed_tokens = num_computed_gpu_blocks * group_config.gpu_block_size
        full_block_tokens = num_computed_tokens + num_external_tokens
        assert full_block_tokens % group_config.offloaded_block_size == 0

        num_pending_gpu_blocks = len(block_ids) - num_computed_gpu_blocks
        assert (
            num_external_tokens == num_pending_gpu_blocks * group_config.gpu_block_size
        )

        start_block_idx = num_computed_tokens // group_config.offloaded_block_size
        num_blocks = full_block_tokens // group_config.offloaded_block_size

        assert len(request.block_hashes) // self.config.block_size_factor >= num_blocks
        offload_keys = group_state.offload_keys[start_block_idx:num_blocks]

        src_spec = self.manager.prepare_load(offload_keys)
        dst_spec = GPULoadStoreSpec(
            block_ids[num_computed_gpu_blocks:],
            group_sizes=(num_pending_gpu_blocks,),
            block_indices=(num_computed_gpu_blocks,),
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            transfer_spec=(src_spec, dst_spec),
        )
        assert req_status.load_job is None
        req_status.load_job = load_job_id
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(offload_keys),
        )
        group_state.next_stored_block_idx = num_blocks

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(offload_keys)

    def _build_store_jobs(
        self, scheduler_output: SchedulerOutput
    ) -> dict[int, TransferJob]:
        # Below assertion will be removed once this function supports HMA
        assert len(self.config.kv_group_configs) == 1
        group_config = self.config.kv_group_configs[0]

        store_jobs: dict[int, TransferJob] = {}
        # iterate over both new and cached requests
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                req_status.update_block_id_groups(new_block_id_groups)

            # Below assertion will be removed once this function supports HMA
            assert len(req_status.group_states) == 1
            group_state = req_status.group_states[0]

            block_ids = group_state.block_ids

            req = req_status.req
            new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            expected_tokens = req.num_computed_tokens + new_tokens
            # with async scheduling, some tokens may be missing
            total_tokens = min(expected_tokens, req.num_tokens)
            num_blocks = total_tokens // group_config.offloaded_block_size
            start_block_idx = group_state.next_stored_block_idx
            num_new_blocks = num_blocks - start_block_idx

            if num_new_blocks <= 0:
                continue

            num_gpu_blocks = num_blocks * self.config.block_size_factor
            assert len(req.block_hashes) >= num_gpu_blocks

            new_offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
            store_output = self.manager.prepare_store(new_offload_keys)
            if store_output is None:
                logger.warning(
                    "Request %s: cannot store %s blocks", req_id, num_new_blocks
                )
                continue

            group_state.next_stored_block_idx = num_blocks

            if not store_output.keys_to_store:
                continue
            keys_to_store = set(store_output.keys_to_store)

            self.manager.touch(group_state.offload_keys[:num_blocks])

            dst_spec = store_output.store_spec
            src_block_ids: list[int] = []
            for idx, key in enumerate(new_offload_keys):
                if key not in keys_to_store:
                    continue
                offloaded_block_idx = start_block_idx + idx
                gpu_block_idx = offloaded_block_idx * self.config.block_size_factor
                for i in range(self.config.block_size_factor):
                    src_block_ids.append(block_ids[gpu_block_idx + i])
            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=(len(src_block_ids),)
            )

            job_id = self._generate_job_id()
            req_status.store_jobs.add(job_id)
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
            )

            store_jobs[job_id] = TransferJob(
                req_id=req_id, transfer_spec=(src_spec, dst_spec)
            )

            logger.debug(
                "Request %s offloading %s blocks starting from block #%d (job %d)",
                req_id,
                len(keys_to_store),
                start_block_idx,
                job_id,
            )

        return store_jobs

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        jobs_to_flush: set[int] = set()
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is not None:
                jobs_to_flush.update(req_status.store_jobs)

        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=self._build_store_jobs(scheduler_output),
            jobs_to_flush=jobs_to_flush,
        )
        self._current_batch_load_jobs = {}
        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        meta = connector_output.kv_connector_worker_meta
        assert isinstance(meta, OffloadingWorkerMetadata)

        for job_id, count in meta.completed_jobs.items():
            job_status = self._jobs.get(job_id)
            if job_status is None:
                continue
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue

            # All TP workers reported — job is complete.
            self._jobs.pop(job_id)
            req_status = self._req_status.get(job_status.req_id)
            if req_status is None:
                continue

            if job_id in req_status.store_jobs:
                req_status.store_jobs.remove(job_id)
                self.manager.complete_store(job_status.keys)
            elif job_id == req_status.load_job:
                req_status.load_job = None
                self.manager.complete_load(job_status.keys)
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(job_status.keys)

            if req_status.is_idle():
                self._req_status.pop(job_status.req_id, None)

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        req_id = request.request_id

        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        req_status = self._req_status.get(req_id)
        if req_status is None:
            return False, None

        request_being_stored = bool(req_status.store_jobs)
        if req_status.is_idle():
            self._req_status.pop(req_id, None)
        return request_being_stored, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            block_hashes = [get_offload_block_hash(key) for key in event.keys]
            if event.removed:
                yield BlockRemoved(block_hashes=block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=event.block_size,
                    medium=event.medium,
                    lora_name=None,
                )

    def shutdown(self) -> None:
        self.manager.shutdown()
