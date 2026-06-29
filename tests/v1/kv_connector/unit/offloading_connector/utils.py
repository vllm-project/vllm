# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.utils import (
    EOS_TOKEN_ID,
    create_model_runner_output,
    create_vllm_config,
)
from vllm import SamplingParams
from vllm.config import (
    KVEventsConfig,
    KVTransferConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.forward_context import ForwardContext
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCaches,
    GroupTransfer,
    LookupResult,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
    OffloadKey,
    PrepareStoreOutput,
    RequestOffloadingContext,
    TransferResult,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


def to_key(int_hash: int) -> OffloadKey:
    return make_offload_key(str(int_hash).encode(), 0)


def to_keys(int_hashes: list[int]) -> list[OffloadKey]:
    return [to_key(i) for i in int_hashes]


class MockLoadStoreSpec(BlockIDsLoadStoreSpec):
    """Mock offloaded side spec that tracks keys rather than real block IDs."""

    def __init__(self, offload_keys: Iterable[OffloadKey]):
        # block_ids is left empty. The mock doesn't allocate real CPU blocks.
        super().__init__(block_ids=[])
        self.offload_keys: list[OffloadKey] = list(offload_keys)

    @staticmethod
    def medium() -> str:
        return "Mock"

    def __repr__(self) -> str:
        return repr(self.offload_keys)


class MockOffloadingWorker(OffloadingWorker):
    def __init__(self):
        # Maps job_id -> (groups, is_store)
        self.transfer_specs: dict[int, tuple[Sequence[GroupTransfer], bool]] = {}
        self.completed_transfers: list[TransferResult] = []
        self.waiting_jobs: set[int] = set()
        self.completed_jobs: list[int] = []
        self.flushed_jobs: set[int] = set()

    def get_finished(self) -> list[TransferResult]:
        finished = self.completed_transfers
        self.completed_transfers = []
        return finished

    def submit_store(self, job_id: int, groups: Sequence[GroupTransfer]) -> bool:
        self.transfer_specs[job_id] = (groups, True)
        self.waiting_jobs.add(job_id)
        return True

    def submit_load(self, job_id: int, groups: Sequence[GroupTransfer]) -> bool:
        self.transfer_specs[job_id] = (groups, False)
        self.waiting_jobs.add(job_id)
        return True

    def complete_jobs(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            if job_id in self.waiting_jobs:
                self.waiting_jobs.remove(job_id)
                self.completed_jobs.append(job_id)
                result = TransferResult(
                    job_id=job_id,
                    success=True,
                    transfer_size=None,
                    transfer_time=None,
                )
                self.completed_transfers.append(result)

    def wait(self, job_ids: set[int]) -> None:
        self.flushed_jobs |= job_ids
        self.complete_jobs(job_ids)


class MockOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        self.manager = MagicMock(spec=OffloadingManager)
        self.manager.lookup.return_value = LookupResult.MISS

        # prepare_load returns one MockLoadStoreSpec per KV group (partitioned by group
        # index). All test keys currently use group_idx=0 via to_key(), so group 0 gets
        # all the keys and remaining groups get empty specs.
        num_kv_groups = len(kv_cache_config.kv_cache_groups)

        def _mock_prepare_load(keys, req_context):
            keys_by_group: dict[int, list[OffloadKey]] = defaultdict(list)
            for key in keys:
                keys_by_group[get_offload_group_idx(key)].append(key)
            return [
                MockLoadStoreSpec(keys_by_group.get(g, []))
                for g in range(num_kv_groups)
            ]

        self.manager.prepare_load = _mock_prepare_load
        self.manager.lookup.return_value = False
        self.manager.on_new_request.return_value = RequestOffloadingContext()
        self.handler = MockOffloadingWorker()

    def get_manager(self) -> OffloadingManager:
        return self.manager

    def get_worker(self, _: CanonicalKVCaches) -> OffloadingWorker:
        return self.handler

    def complete_transfers(self):
        self.handler.complete_jobs(self.handler.waiting_jobs.copy())

    def get_completed_transfers(
        self,
    ) -> list[tuple[Sequence[GroupTransfer], bool]]:
        specs = [
            self.handler.transfer_specs[job_id]
            for job_id in self.handler.completed_jobs
        ]
        self.handler.completed_jobs.clear()
        return specs

    def get_flushed_transfers(
        self,
    ) -> list[tuple[Sequence[GroupTransfer], bool]]:
        specs = [
            self.handler.transfer_specs[job_id] for job_id in self.handler.flushed_jobs
        ]
        self.handler.flushed_jobs.clear()
        return specs


@dataclass(frozen=True)
class GPUBlock:
    group_idx: int
    request_block_offset: int


@dataclass
class TransferSummary:
    gpu_blocks: list[GPUBlock]
    offload_addresses: list[Any]


class RequestRunner:
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        block_size_factor: int = 1,
        async_scheduling: bool = True,
        kv_cache_groups: list[KVCacheGroupSpec] | None = None,
        extra_config_overrides: dict[str, Any] | None = None,
    ):
        assert block_size_factor == 1 or kv_cache_groups is None, (
            "block_size_factor > 1 requires all groups to have the same "
            "block size, so kv_cache_groups must be None (use default group)"
        )

        self.block_size_factor: int = block_size_factor
        self.block_size: int = block_size
        self.num_gpu_blocks: int = num_gpu_blocks
        self.async_scheduling: bool = async_scheduling

        self.req_id: int = -1

        vllm_config = create_vllm_config(
            block_size=block_size,
            max_num_batched_tokens=1000,
            disable_hybrid_kv_cache_manager=False,
        )
        vllm_config.scheduler_config.async_scheduling = async_scheduling

        extra_config: dict[str, Any] = {
            "spec_name": "MockOffloadingSpec",
            "spec_module_path": "tests.v1.kv_connector.unit.offloading_connector.utils",  # noqa: E501
            # Preserve legacy behavior for tests; new opt-in tests override.
            "offload_prompt_only": False,
            # Exercise the self-describing KV events path by default;
            # opt-out tests override this to cover the legacy placeholders.
            "self_describing_kv_events": True,
        }
        if block_size_factor > 1:
            extra_config["block_size"] = block_size * block_size_factor
        if extra_config_overrides:
            extra_config.update(extra_config_overrides)

        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config=extra_config,
        )
        vllm_config.kv_events_config = KVEventsConfig(
            # Enable so the offloading events tracker is active, but use the
            # null publisher: these tests drain take_events directly and a
            # real ZMQ publisher would bind a port per test.
            enable_kv_cache_events=True,
            publisher="null",
        )

        if kv_cache_groups is None:
            kv_cache_groups = [
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ]

        kv_cache_tensors = [
            KVCacheTensor(
                size=group.kv_cache_spec.page_size_bytes * num_gpu_blocks,
                shared_by=[layer_name],
            )
            for group in kv_cache_groups
            for layer_name in group.layer_names
        ]

        kv_cache_config = KVCacheConfig(
            num_blocks=num_gpu_blocks,
            kv_cache_tensors=kv_cache_tensors,
            kv_cache_groups=kv_cache_groups,
        )
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        self.num_kv_groups = len(kv_cache_config.kv_cache_groups)

        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )

        scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
        self.scheduler = scheduler_cls(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            log_stats=True,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )

        self.worker_connector = OffloadingConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

        # register worker kv_caches to enable OffloadingWorker creations
        # set_current_vllm_config is needed for get_kv_cache_layout() to work
        kv_caches: dict[str, torch.Tensor] = {}
        for group in kv_cache_groups:
            spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                # Shape follows FlashAttention layout:
                # Shape: (num_blocks, 2, block_size, num_kv_heads, head_size)
                kv_caches[layer_name] = torch.empty(
                    num_gpu_blocks,
                    2,
                    spec.block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                    dtype=spec.dtype,
                )

        with set_current_vllm_config(vllm_config):
            self.worker_connector.register_kv_caches(kv_caches)

        # extract connector of scheduler
        scheduler_connector = self.scheduler.connector
        assert scheduler_connector is not None
        assert isinstance(scheduler_connector, OffloadingConnector)
        self.scheduler_connector: OffloadingConnector = scheduler_connector

        # extract mocked OffloadingManager of scheduler connector
        self.connector_scheduler = scheduler_connector.connector_scheduler
        assert self.connector_scheduler is not None
        manager = self.connector_scheduler.manager
        assert isinstance(manager, MagicMock)
        self.manager: MagicMock = manager

        num_kv_groups = len(kv_cache_config.kv_cache_groups)
        assert len(self.connector_scheduler.config.kv_group_configs) == num_kv_groups
        for group_config, kv_cache_group in zip(
            self.connector_scheduler.config.kv_group_configs,
            kv_cache_config.kv_cache_groups,
        ):
            gpu_block_size = kv_cache_group.kv_cache_spec.block_size
            assert group_config.gpu_block_size == gpu_block_size
            assert (
                group_config.offloaded_block_size == gpu_block_size * block_size_factor
            )

        # extract OffloadingSpec of worker_connector
        connector_worker = self.worker_connector.connector_worker
        assert connector_worker is not None
        offloading_spec = connector_worker.spec
        assert isinstance(offloading_spec, MockOffloadingSpec)
        self.offloading_spec: MockOffloadingSpec = offloading_spec

        # mapping (offloading address) -> GPUBlock
        self.offloaded: dict[Any, GPUBlock] = {}

        self.completed_loads: list[TransferSummary] = []
        self.completed_stores: list[TransferSummary] = []
        self.flushed_gpu_blocks: set[GPUBlock] = set()

        # block_id -> GPUBlock
        self.gpu_blocks: dict[int, GPUBlock] = {}

        init_none_hash(sha256)
        self._block_hasher = get_request_block_hasher(block_size, sha256)

        self._dummy_ctx: ForwardContext = ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
        )

    def new_request(
        self,
        token_ids: list[int],
        kv_transfer_params: dict | None = None,
        skip_reading_prefix_cache: bool = False,
    ):
        self.req_id += 1

        sampling_params = SamplingParams(
            max_tokens=1000,
            skip_reading_prefix_cache=skip_reading_prefix_cache or None,
        )
        sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

        req = Request(
            request_id=str(self.req_id),
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=self._block_hasher,
        )
        if kv_transfer_params is not None:
            req.kv_transfer_params = kv_transfer_params

        self.scheduler.add_request(req)

    def _parse_transfers(self):
        for groups, _is_store in self.offloading_spec.get_flushed_transfers():
            for group in groups:
                for block_id in group.gpu_spec.block_ids:
                    self.flushed_gpu_blocks.add(self.gpu_blocks[block_id.item()])

        block_size_factor = self.block_size_factor

        for groups, store in self.offloading_spec.get_completed_transfers():
            assert len(groups) == self.num_kv_groups

            gpu_blocks: list[GPUBlock] = []
            offload_addresses: list[Any] = []

            for group in groups:
                gpu_spec = group.gpu_spec
                offload_spec = group.offload_spec
                assert isinstance(offload_spec, MockLoadStoreSpec)

                for block_id in gpu_spec.block_ids:
                    gpu_blocks.append(self.gpu_blocks[block_id.item()])

                # Expand offload keys to (key, sub_block_idx) addresses and
                # apply the alignment skip: the first offload_skip sub blocks
                # of the first offloaded block correspond to GPU blocks before
                # our transfer range and are excluded.
                group_size = len(gpu_spec.block_ids)
                offload_skip = offload_spec.gpu_block_offset % block_size_factor
                raw_addresses: list[Any] = [
                    (offload_key, sub_block_idx)
                    for offload_key in offload_spec.offload_keys
                    for sub_block_idx in range(block_size_factor)
                ]
                aligned = raw_addresses[offload_skip : offload_skip + group_size]
                offload_addresses.extend(aligned)

            transfer_summary = TransferSummary(gpu_blocks, offload_addresses)
            if store:
                self.completed_stores.append(transfer_summary)
            else:
                self.completed_loads.append(transfer_summary)

    def _update_gpu_blocks(self):
        for group_idx, manager in enumerate(
            self.scheduler.kv_cache_manager.coordinator.single_type_managers
        ):
            for blocks in manager.req_to_blocks.values():
                for block_idx, block in enumerate(blocks):
                    self.gpu_blocks[block.block_id] = GPUBlock(group_idx, block_idx)

    def _run(
        self,
        decoded_tokens: list[int],
        complete_transfers: bool,
        post_step_fn: Callable[[], None] | None = None,
    ):
        """
        Runs multiple engine (scheduler + worker) steps.
        Assumes a single request is running.

        Args:
            decoded_tokens: the tokens to yield at each step.
            complete_transfers: complete transfers immediately
            post_step_fn: optional callback invoked after each step's
                update_from_output(), before the next schedule().
        """

        tokens_iter = iter(decoded_tokens)
        token_id = next(tokens_iter, None)
        prev_scheduler_output = None
        prev_model_runner_output = None
        while True:
            # Strict-always-False frees the request immediately on EOS, but
            # the worker may still have a deferred store queued. In production
            # the next request's step drains it; in single-request tests we
            # must keep stepping until the scheduler sees no in-flight jobs.
            if not self.scheduler.requests and not self.connector_scheduler._jobs:
                break

            scheduler_output = self.scheduler.schedule()
            self._update_gpu_blocks()

            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            assert isinstance(kv_connector_metadata, OffloadingConnectorMetadata)

            self.worker_connector.handle_preemptions(kv_connector_metadata)

            self.worker_connector.bind_connector_metadata(kv_connector_metadata)
            self.worker_connector.start_load_kv(self._dummy_ctx)

            if complete_transfers:
                self.offloading_spec.complete_transfers()

            finished_sending, finished_recving = self.worker_connector.get_finished(
                scheduler_output.finished_req_ids
            )
            worker_meta = (
                self.worker_connector.build_connector_worker_meta()
                or OffloadingWorkerMetadata()
            )

            self.worker_connector.clear_connector_metadata()

            model_runner_output = create_model_runner_output(
                reqs=self.scheduler.running,
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                token_id=token_id or 0,
                kv_connector_worker_meta=worker_meta,
            )

            prev_token_id = token_id
            if self.scheduler.running:
                token_id = next(tokens_iter, None)

            if self.async_scheduling:
                # in async scheduling we update the output of the previous step
                if prev_model_runner_output is not None:
                    self.scheduler.update_from_output(
                        prev_scheduler_output, prev_model_runner_output
                    )
                prev_scheduler_output = scheduler_output
                prev_model_runner_output = model_runner_output
            else:
                self.scheduler.update_from_output(scheduler_output, model_runner_output)

            if post_step_fn is not None:
                post_step_fn()

            if (
                prev_token_id == EOS_TOKEN_ID
                and prev_token_id != token_id
                and (self.scheduler.requests or self.connector_scheduler._jobs)
            ):
                # continue for one more step to allow offloading to kick off
                continue

            if token_id is None:
                if self.async_scheduling:
                    # sample last token
                    self.scheduler.update_from_output(
                        prev_scheduler_output, prev_model_runner_output
                    )
                break

        self._parse_transfers()

        if EOS_TOKEN_ID in decoded_tokens:
            assert not self.scheduler.running

    def _to_gpu_blocks(
        self, blocks: tuple[int | tuple[int, int], ...]
    ) -> list[GPUBlock]:
        gpu_blocks: list[GPUBlock] = []
        for block in blocks:
            if isinstance(block, int):
                for group_idx in range(self.num_kv_groups):
                    gpu_blocks.append(
                        GPUBlock(group_idx=group_idx, request_block_offset=block)
                    )
            else:
                group_idx, offset = block
                gpu_blocks.append(
                    GPUBlock(group_idx=group_idx, request_block_offset=offset)
                )
        return gpu_blocks

    def run(
        self,
        decoded_tokens: list[int],
        complete_transfers: bool = True,
        expected_stored: tuple[int | tuple[int, int], ...] = (),
        expected_loaded: tuple[int | tuple[int, int], ...] = (),
        expected_flushed: tuple[int | tuple[int, int], ...] = (),
        post_step_fn: Callable[[], None] | None = None,
    ):
        """
        Runs multiple engine (scheduler + worker) steps.
        Assumes a single request is running.

        Args:
            decoded_tokens: the tokens to yield at each step.
            complete_transfers: complete transfers immediately
            expected_stored: GPU blocks
                that are expected to be written during the run.
            expected_loaded: GPU blocks
                that are expected to be loaded during the run.
            expected_flushed: GPU blocks
                that are expected to be flushed during the run.

            A GPU block is either a (group_idx: int, request_block_offset: int)
            or just request_block_offset: int.
            The latter case is a convenience for representing all groups.
        """

        expected_stored_gpu_blocks = self._to_gpu_blocks(expected_stored)
        expected_loaded_gpu_blocks = self._to_gpu_blocks(expected_loaded)
        expected_flushed_gpu_blocks = self._to_gpu_blocks(expected_flushed)

        self.manager.reset_mock()
        self._run(decoded_tokens, complete_transfers, post_step_fn=post_step_fn)

        loaded_gpu_blocks: set[GPUBlock] = set()
        for transfer in self.completed_loads:
            for gpu_block, offloaded_address in zip(
                transfer.gpu_blocks, transfer.offload_addresses
            ):
                loaded_gpu_blocks.add(gpu_block)
                assert gpu_block == self.offloaded[offloaded_address]

        assert set(expected_loaded_gpu_blocks) == loaded_gpu_blocks
        self.completed_loads.clear()

        stored_gpu_blocks: set[GPUBlock] = set()
        for transfer in self.completed_stores:
            for gpu_block, offloaded_address in zip(
                transfer.gpu_blocks, transfer.offload_addresses
            ):
                stored_gpu_blocks.add(gpu_block)
                self.offloaded[offloaded_address] = gpu_block

        assert set(expected_stored_gpu_blocks) == stored_gpu_blocks
        self.completed_stores.clear()

        assert set(expected_flushed_gpu_blocks) == self.flushed_gpu_blocks
        self.flushed_gpu_blocks.clear()


@pytest.fixture
def request_runner():
    runners = []

    def runner_factory(
        block_size,
        num_gpu_blocks,
        async_scheduling,
        block_size_factor=1,
        kv_cache_groups=None,
        extra_config_overrides=None,
    ):
        runner = RequestRunner(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            block_size_factor=block_size_factor,
            async_scheduling=async_scheduling,
            kv_cache_groups=kv_cache_groups,
            extra_config_overrides=extra_config_overrides,
        )
        runners.append(runner)
        return runner

    yield runner_factory  # pass factory to the test


def generate_store_output(keys: Iterable[OffloadKey], num_groups: int = 1):
    keys = list(keys)
    keys_by_group: dict[int, list[OffloadKey]] = defaultdict(list)
    for key in keys:
        keys_by_group[get_offload_group_idx(key)].append(key)
    store_specs = [
        MockLoadStoreSpec(keys_by_group.get(g, [])) for g in range(num_groups)
    ]
    return PrepareStoreOutput(
        keys_to_store=keys,
        store_specs=store_specs,
        evicted_keys=[],
    )
