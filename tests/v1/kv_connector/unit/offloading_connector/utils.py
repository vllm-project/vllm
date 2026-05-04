# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.v1.kv_connector.unit.utils import (
    EOS_TOKEN_ID,
    create_model_runner_output,
    create_vllm_config,
)
from vllm import SamplingParams
from vllm.config import KVTransferConfig, VllmConfig, set_current_vllm_config
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
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    PrepareStoreOutput,
    make_offload_key,
)
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


def to_key(int_hash: int) -> OffloadKey:
    return make_offload_key(str(int_hash).encode(), 0)


def to_keys(int_hashes: list[int]) -> list[OffloadKey]:
    return [to_key(i) for i in int_hashes]


class MockLoadStoreSpec(LoadStoreSpec):
    def __init__(self, offload_keys: Iterable[OffloadKey]):
        self.offload_keys: list[OffloadKey] = list(offload_keys)

    @staticmethod
    def medium() -> str:
        return "Mock"

    def __repr__(self) -> str:
        return repr(self.offload_keys)


class MockOffloadingHandler(OffloadingHandler):
    def __init__(self):
        self.transfer_specs: dict[int, TransferSpec] = {}
        self.completed_transfers: list[TransferResult] = []
        self.waiting_jobs: set[int] = set()
        self.completed_jobs: list[int] = []
        self.flushed_jobs: set[int] = set()

    def get_finished(self) -> list[TransferResult]:
        finished = self.completed_transfers
        self.completed_transfers = []
        return finished

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        self.transfer_specs[job_id] = spec
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
                    transfer_type=None,
                )
                self.completed_transfers.append(result)

    def wait(self, job_ids: set[int]) -> None:
        self.flushed_jobs |= job_ids
        self.complete_jobs(job_ids)


class MockOffloadingSpec(OffloadingSpec):
    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        self.manager = MagicMock(spec=OffloadingManager)
        self.manager.lookup.return_value = 0
        self.manager.prepare_load = lambda keys, req_context: MockLoadStoreSpec(keys)
        self.manager.lookup.return_value = False
        self.handler = MockOffloadingHandler()

    def get_manager(self) -> OffloadingManager:
        return self.manager

    def get_handlers(
        self, _
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        yield GPULoadStoreSpec, MockLoadStoreSpec, self.handler
        yield MockLoadStoreSpec, GPULoadStoreSpec, self.handler

    def complete_transfers(self):
        self.handler.complete_jobs(self.handler.waiting_jobs.copy())

    def get_completed_transfers(self) -> list[TransferSpec]:
        specs = [
            self.handler.transfer_specs[job_id]
            for job_id in self.handler.completed_jobs
        ]
        self.handler.completed_jobs.clear()
        return specs

    def get_flushed_transfers(self):
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
        }
        if block_size_factor > 1:
            extra_config["block_size"] = block_size * block_size_factor

        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config=extra_config,
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

        kv_cache_config = KVCacheConfig(
            num_blocks=num_gpu_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        )
        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        self.num_kv_groups = len(kv_cache_config.kv_cache_groups)

        scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
        self.scheduler = scheduler_cls(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            log_stats=True,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=block_size,
        )

        self.worker_connector = OffloadingConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )

        # register worker kv_caches to enable OffloadingWorker creations
        # set_current_vllm_config is needed for get_kv_cache_layout() to work
        # Mock get_layers_from_vllm_config so that mock layer names
        # resolve to layers whose get_attn_backend() returns
        # FlashAttentionBackend.
        def _mock_get_layers(_vllm_config, _layer_type, layer_names):
            mock_layer = MagicMock()
            mock_layer.get_attn_backend.return_value = FlashAttentionBackend
            return {name: mock_layer for name in layer_names}

        kv_caches: dict[str, torch.Tensor] = {}
        for group in kv_cache_groups:
            spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                # Shape follows FlashAttention layout:
                # (2, num_blocks, block_size, num_kv_heads, head_size)
                kv_caches[layer_name] = torch.empty(
                    2,
                    num_gpu_blocks,
                    spec.block_size,
                    spec.num_kv_heads,
                    spec.head_size,
                    dtype=spec.dtype,
                )

        with (
            set_current_vllm_config(vllm_config),
            patch(
                "vllm.distributed.kv_transfer.kv_connector.v1"
                ".offloading.worker.get_layers_from_vllm_config",
                side_effect=_mock_get_layers,
            ),
        ):
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
    ):
        self.req_id += 1

        sampling_params = SamplingParams(max_tokens=1000)
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
        for transfer_spec in self.offloading_spec.get_flushed_transfers():
            src_spec, dst_spec = transfer_spec
            assert isinstance(src_spec, GPULoadStoreSpec)

            for block_id in src_spec.block_ids:
                self.flushed_gpu_blocks.add(self.gpu_blocks[block_id.item()])

        block_size_factor = self.block_size_factor

        for transfer_spec in self.offloading_spec.get_completed_transfers():
            src_spec, dst_spec = transfer_spec

            if isinstance(src_spec, GPULoadStoreSpec):
                store = True
                gpu_spec = src_spec
                offload_spec = dst_spec
            else:
                store = False
                gpu_spec = dst_spec
                offload_spec = src_spec

            assert isinstance(offload_spec, MockLoadStoreSpec)
            assert isinstance(gpu_spec, GPULoadStoreSpec)
            assert len(gpu_spec.group_sizes) == self.num_kv_groups

            gpu_blocks: list[GPUBlock] = []
            for block_id in gpu_spec.block_ids:
                gpu_blocks.append(self.gpu_blocks[block_id.item()])

            # list of (offload_key, sub_block_offset)
            offload_addresses: list[Any] = []
            for offload_key in offload_spec.offload_keys:
                for sub_block_idx in range(block_size_factor):
                    offload_addresses.append((offload_key, sub_block_idx))

            assert gpu_spec.block_indices is not None
            assert len(gpu_spec.block_indices) == self.num_kv_groups

            gpu_block_offset = 0
            offload_address_offset = 0
            for group_size, logical_offset in zip(
                gpu_spec.group_sizes, gpu_spec.block_indices
            ):
                gpu_block_end_offset = gpu_block_offset + group_size
                assert gpu_block_end_offset <= len(gpu_blocks)

                offload_addresses_to_skip = logical_offset % block_size_factor
                offload_addresses_end_offset = (
                    offload_address_offset + offload_addresses_to_skip + group_size
                )
                assert offload_addresses_end_offset <= len(offload_addresses)

                offload_addresses = (
                    offload_addresses[:offload_address_offset]
                    + offload_addresses[
                        offload_address_offset + offload_addresses_to_skip :
                    ]
                )

                gpu_block_offset += group_size
                offload_address_offset += group_size

            assert gpu_block_offset == len(gpu_blocks)
            assert offload_address_offset == len(offload_addresses)

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

    def _run(self, decoded_tokens: list[int], complete_transfers: bool):
        """
        Runs multiple engine (scheduler + worker) steps.
        Assumes a single request is running.

        Args:
            decoded_tokens: the tokens to yield at each step.
            complete_transfers: complete transfers immediately
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

            if scheduler_output.total_num_scheduled_tokens > 0:
                self.worker_connector.wait_for_save()

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
        self._run(decoded_tokens, complete_transfers)

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
    ):
        runner = RequestRunner(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            block_size_factor=block_size_factor,
            async_scheduling=async_scheduling,
            kv_cache_groups=kv_cache_groups,
        )
        runners.append(runner)
        return runner

    yield runner_factory  # pass factory to the test


def generate_store_output(keys: Iterable[OffloadKey]):
    keys = list(keys)
    return PrepareStoreOutput(
        keys_to_store=list(keys),
        store_spec=MockLoadStoreSpec(keys),
        evicted_keys=[],
    )
