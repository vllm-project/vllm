# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections.abc import Iterable, Iterator
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
from vllm.config import KVTransferConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.forward_context import ForwardContext
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
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
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


def to_hash(int_hash: int) -> BlockHash:
    return BlockHash(str(int_hash).encode())


def to_hashes(int_hashes: list[int]) -> list[BlockHash]:
    return [to_hash(i) for i in int_hashes]


class MockLoadStoreSpec(LoadStoreSpec):
    def __init__(self, block_hashes: Iterable[BlockHash]):
        self.block_hashes: list[BlockHash] = list(block_hashes)

    @staticmethod
    def medium() -> str:
        return "Mock"

    def __repr__(self) -> str:
        return repr(self.block_hashes)


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
        self.manager.lookup.return_value = False
        self.manager.prepare_load = lambda block_hashes: (
            MockLoadStoreSpec(block_hashes)
        )
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


@dataclass
class TransferSummary:
    gpu_block_indices: list[int]
    offload_addresses: list[Any]


class RequestRunner:
    def __init__(
        self,
        offloaded_block_size: int,
        gpu_block_size: int,
        num_gpu_blocks: int,
        async_scheduling: bool = True,
    ):
        self.offloaded_block_size: int = offloaded_block_size
        self.gpu_block_size: int = gpu_block_size
        self.num_gpu_blocks: int = num_gpu_blocks
        self.async_scheduling: bool = async_scheduling

        self.req_id: int = -1

        vllm_config = create_vllm_config(
            block_size=gpu_block_size, max_num_batched_tokens=1000
        )
        vllm_config.scheduler_config.async_scheduling = async_scheduling
        vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "MockOffloadingSpec",
                "spec_module_path": "tests.v1.kv_connector.unit.offloading_connector.utils",  # noqa: E501
                "block_size": offloaded_block_size,
            },
        )

        block_size = vllm_config.cache_config.block_size
        kv_cache_config = KVCacheConfig(
            num_blocks=num_gpu_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
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
        with set_current_vllm_config(vllm_config):
            self.worker_connector.register_cross_layers_kv_cache(
                kv_cache=torch.empty(0),
                attn_backend=FlashAttentionBackend,
            )

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

        assert self.connector_scheduler.gpu_block_size == gpu_block_size
        assert self.connector_scheduler.offloaded_block_size == offloaded_block_size

        # extract OffloadingSpec of worker_connector
        connector_worker = self.worker_connector.connector_worker
        assert connector_worker is not None
        offloading_spec = connector_worker.spec
        assert isinstance(offloading_spec, MockOffloadingSpec)
        self.offloading_spec: MockOffloadingSpec = offloading_spec

        # mapping (offloading address) -> gpu_block_index
        self.offloaded: dict[Any, int] = {}

        self.completed_loads: list[TransferSummary] = []
        self.completed_stores: list[TransferSummary] = []
        self.flushed_gpu_block_indexes: set[int] = set()

        # maps {block_id: block_offset}
        self.gpu_block_index: dict[int, int] = {}

        init_none_hash(sha256)
        self._block_hasher = get_request_block_hasher(gpu_block_size, sha256)

        self._dummy_ctx: ForwardContext = ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
        )

    def new_request(self, token_ids: list[int]):
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

        self.scheduler.add_request(req)

    def _parse_transfers(self):
        for transfer_spec in self.offloading_spec.get_flushed_transfers():
            src_spec, dst_spec = transfer_spec
            assert isinstance(src_spec, GPULoadStoreSpec)

            for block_id in src_spec.block_ids:
                self.flushed_gpu_block_indexes.add(
                    self.gpu_block_index[block_id.item()]
                )

        block_size_factor = self.offloaded_block_size // self.gpu_block_size

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

            gpu_block_indices: list[int] = []
            for block_id in gpu_spec.block_ids:
                gpu_block_indices.append(self.gpu_block_index[block_id.item()])

            # list of (block_hash, sub_block_offset)
            offload_addresses: list[Any] = []
            for block_hash in offload_spec.block_hashes:
                for sub_block_idx in range(block_size_factor):
                    offload_addresses.append((block_hash, sub_block_idx))

            if store:
                assert len(gpu_block_indices) == len(offload_addresses)

                self.completed_stores.append(
                    TransferSummary(gpu_block_indices, offload_addresses)
                )
            else:
                remainder_sub_block_count = len(offload_addresses) - len(
                    gpu_block_indices
                )
                assert remainder_sub_block_count >= 0
                assert remainder_sub_block_count < block_size_factor
                offload_addresses = offload_addresses[remainder_sub_block_count:]

                self.completed_loads.append(
                    TransferSummary(gpu_block_indices, offload_addresses)
                )

    def _update_gpu_block_idx(self):
        for blocks in self.scheduler.kv_cache_manager.coordinator.single_type_managers[
            0
        ].req_to_blocks.values():
            for block_idx, block in enumerate(blocks):
                self.gpu_block_index[block.block_id] = block_idx

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
            assert self.scheduler.requests

            scheduler_output = self.scheduler.schedule()
            self._update_gpu_block_idx()

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

            self.worker_connector.clear_connector_metadata()

            model_runner_output = create_model_runner_output(
                reqs=self.scheduler.running,
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                token_id=token_id or 0,
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
                and self.scheduler.requests
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

        # run one more step to update finished stored
        if EOS_TOKEN_ID in decoded_tokens:
            assert not self.scheduler.running

            while self.scheduler.requests:
                scheduler_output = self.scheduler.schedule()

                finished_sending, finished_recving = self.worker_connector.get_finished(
                    scheduler_output.finished_req_ids
                )

                assert not finished_recving

                model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
                model_runner_output.kv_connector_output = KVConnectorOutput(
                    finished_sending=finished_sending
                )

                self.scheduler.update_from_output(scheduler_output, model_runner_output)

    def run(
        self,
        decoded_tokens: list[int],
        complete_transfers: bool = True,
        expected_stored_gpu_block_indexes: tuple[int, ...] = (),
        expected_loaded_gpu_block_indexes: tuple[int, ...] = (),
        expected_flushed_gpu_block_indexes: tuple[int, ...] = (),
    ):
        """
        Runs multiple engine (scheduler + worker) steps.
        Assumes a single request is running.

        Args:
            decoded_tokens: the tokens to yield at each step.
            complete_transfers: complete transfers immediately
            expected_stored_gpu_block_indexes: GPU block indexes
                that are expected to be written during the run.
            expected_loaded_gpu_block_indexes: GPU block indexes
                that are expected to be loaded during the run.
            expected_flushed_gpu_block_indexes: GPU block indexes
                that are expected to be flushed during the run.
        """

        self.manager.reset_mock()
        self._run(decoded_tokens, complete_transfers)

        loaded_gpu_block_indexes: set[int] = set()
        for transfer in self.completed_loads:
            for gpu_block_idx, offloaded_address in zip(
                transfer.gpu_block_indices, transfer.offload_addresses
            ):
                loaded_gpu_block_indexes.add(gpu_block_idx)
                assert gpu_block_idx == self.offloaded[offloaded_address]

        assert set(expected_loaded_gpu_block_indexes) == loaded_gpu_block_indexes
        self.completed_loads.clear()

        stored_gpu_block_indexes: set[int] = set()
        for transfer in self.completed_stores:
            for gpu_block_idx, offloaded_address in zip(
                transfer.gpu_block_indices, transfer.offload_addresses
            ):
                stored_gpu_block_indexes.add(gpu_block_idx)
                self.offloaded[offloaded_address] = gpu_block_idx

        assert set(expected_stored_gpu_block_indexes) == stored_gpu_block_indexes
        self.completed_stores.clear()

        assert set(expected_flushed_gpu_block_indexes) == self.flushed_gpu_block_indexes
        self.flushed_gpu_block_indexes.clear()


@pytest.fixture
def request_runner():
    runners = []

    def runner_factory(
        offloaded_block_size, gpu_block_size, num_gpu_blocks, async_scheduling
    ):
        runner = RequestRunner(
            offloaded_block_size=offloaded_block_size,
            gpu_block_size=gpu_block_size,
            num_gpu_blocks=num_gpu_blocks,
            async_scheduling=async_scheduling,
        )
        runners.append(runner)
        return runner

    yield runner_factory  # pass factory to the test


def generate_store_output(block_hashes: Iterable[BlockHash]):
    block_hashes = list(block_hashes)
    return PrepareStoreOutput(
        block_hashes_to_store=list(block_hashes),
        store_spec=MockLoadStoreSpec(block_hashes),
        block_hashes_evicted=[],
    )
