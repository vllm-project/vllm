# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Any, ClassVar

import torch

from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import OffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingWorker,
    TransferSpec,
    TransferType,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

ReqId = str

logger = init_logger(__name__)


@dataclass
class OffloadingOperationMetrics:
    op_size: float
    op_time: float
    op_type: str


@dataclass
class OffloadingConnectorStats(KVConnectorStats):
    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        self.data: dict[str, list[OffloadingOperationMetrics]] = {}

    def clone_and_reset(self) -> "OffloadingConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for k, v in other.data.items():
                if k in self.data:
                    accumulator = self.data[k]
                    assert isinstance(accumulator, list)
                    accumulator.extend(v)
        return self

    def reduce(self) -> dict[str, int | float]:
        """
        Reduce the observations collected during a time interval to one or
        more representative values (eg avg/median/sum of the series).
        This is meant to be called by the logger to produce a summary of the
        stats for the last time interval.
        """
        return_dict: dict[str, int | float] = {}
        for k, v in self.data.items():
            assert isinstance(v, list)
            return_dict[k] = len(v)
        return return_dict

    def is_empty(self) -> bool:
        return len(self.data.items()) == 0

    def record_transfer(self, num_bytes: int, time: float, transfer_type: TransferType):
        src, dst = transfer_type
        transfer_type_key = src + "_to_" + dst
        op = OffloadingOperationMetrics(num_bytes, time, transfer_type_key)
        if transfer_type_key in self.data:
            self.data[transfer_type_key].append(op)
        else:
            self.data[transfer_type_key] = [op]


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]


class OffloadingConnector(KVConnectorBase_V1):
    prefer_cross_layer_blocks: ClassVar[bool] = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        spec = OffloadingSpecFactory.create_spec(vllm_config)

        self.connector_scheduler: OffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            if kv_cache_config is None:
                raise ValueError("kv_cache_config cannot be None for WORKER role")
            self.connector_worker = OffloadingConnectorWorker(
                spec, self.calculate_bytes_per_block(kv_cache_config, vllm_config)
            )

    def calculate_bytes_per_block(
        self, kv_cache_config: KVCacheConfig, vllm_config: VllmConfig
    ) -> int:
        page_sizes = {
            kv_cache_group.kv_cache_spec.page_size_bytes
            for kv_cache_group in kv_cache_config.kv_cache_groups
        }
        assert len(page_sizes) == 1
        page_size_bytes = page_sizes.pop()
        kv_bytes_per_block = (
            page_size_bytes
            * len(kv_cache_config.kv_cache_tensors)
            * vllm_config.parallel_config.world_size
        )
        return kv_bytes_per_block

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        assert self.connector_worker is not None
        self.connector_worker.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, OffloadingConnectorMetadata)
        self.connector_worker.start_store_kv(self._connector_metadata)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def take_events(self) -> Iterable[KVCacheEvent]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.take_events()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return (
            OffloadingConnectorStats(data=data)
            if data is not None
            else OffloadingConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return OffloadPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, spec: OffloadingSpec):
        self.gpu_block_size = spec.gpu_block_size
        self.offloaded_block_size = spec.offloaded_block_size
        self.block_size_factor = self.offloaded_block_size // self.gpu_block_size
        self.manager: OffloadingManager = spec.get_manager()

        self._requests: dict[ReqId, Request] = {}
        # list of GPU block IDs per request
        self._request_block_ids: dict[ReqId, list[int]] = {}
        # requests to load for the current scheduler step
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        # request blocks are stored in order
        # index of next block (of size offloaded_block_size) to offload
        self._next_stored_block_idx: dict[ReqId, int] = {}

        # request ID -> set(block hashes being stored/load)
        self._reqs_being_stored = defaultdict[ReqId, set[BlockHash]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[BlockHash]](set)

    def _get_block_hashes(
        self,
        req: Request,
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> Iterable[BlockHash]:
        return islice(
            req.block_hashes,
            self.block_size_factor * start_idx + self.block_size_factor - 1,
            self.block_size_factor * end_idx if end_idx else None,
            self.block_size_factor,
        )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
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
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        num_blocks = request.num_tokens // self.offloaded_block_size

        assert len(request.block_hashes) // self.block_size_factor == num_blocks
        block_hashes = self._get_block_hashes(request)

        self.manager.touch(block_hashes)

        full_block_tokens = self.offloaded_block_size * num_blocks
        if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
            # we can load less than a block, skip
            return 0, False

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        hits = self.manager.lookup(
            self._get_block_hashes(request, start_idx=start_block_idx)
        )
        if hits == 0:
            return 0, False

        num_hit_tokens = (
            self.offloaded_block_size * (start_block_idx + hits) - num_computed_tokens
        )
        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            request.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )
        if num_hit_tokens < self.offloaded_block_size:
            return 0, False

        return num_hit_tokens, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        self._requests[request.request_id] = request
        # the block ids are updated in _get_reqs_to_store
        self._request_block_ids[request.request_id] = []

        if num_external_tokens == 0:
            return

        block_groups = blocks.get_block_ids()
        block_ids = block_groups[0]

        num_computed_gpu_blocks = sum(
            block.block_hash is not None for block in blocks.blocks[0]
        )
        num_computed_tokens = num_computed_gpu_blocks * self.gpu_block_size
        full_block_tokens = num_computed_tokens + num_external_tokens
        assert full_block_tokens % self.offloaded_block_size == 0

        num_pending_gpu_blocks = len(block_ids) - num_computed_gpu_blocks
        assert num_external_tokens == num_pending_gpu_blocks * self.gpu_block_size

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        num_blocks = full_block_tokens // self.offloaded_block_size

        assert len(request.block_hashes) // self.block_size_factor >= num_blocks
        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        src_spec = self.manager.prepare_load(block_hashes)
        dst_spec = GPULoadStoreSpec(block_ids[num_computed_gpu_blocks:])

        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        self._reqs_to_load[request.request_id] = (src_spec, dst_spec)
        self._reqs_being_loaded[request.request_id].update(block_hashes)
        self._next_stored_block_idx[request.request_id] = num_blocks

    def _get_reqs_to_store(self, scheduler_output: SchedulerOutput):
        reqs_to_store: dict[ReqId, TransferSpec] = {}
        # iterate over both new and cached requests
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            if preempted:
                self._request_block_ids[req_id] = []

            if new_block_id_groups:
                new_block_ids = new_block_id_groups[0]
                self._request_block_ids[req_id] += new_block_ids

            block_ids = self._request_block_ids[req_id]

            req = self._requests[req_id]
            new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            total_tokens = req.num_computed_tokens + new_tokens
            num_blocks = total_tokens // self.offloaded_block_size
            start_block_idx = self._next_stored_block_idx.get(req_id, 0)
            num_new_blocks = num_blocks - start_block_idx

            if num_new_blocks <= 0:
                continue

            # NOTE: In async scheduling, placeholders may temporarily make
            # len(req.block_hashes) < num_blocks * self.block_size_factor.

            new_block_hashes = self._get_block_hashes(
                req, start_idx=start_block_idx, end_idx=num_blocks
            )
            store_output = self.manager.prepare_store(new_block_hashes)
            if store_output is None:
                logger.warning(
                    "Request %s: cannot store %s blocks", req_id, num_new_blocks
                )
                continue

            self._next_stored_block_idx[req_id] = num_blocks

            if not store_output.block_hashes_to_store:
                continue
            block_hashes_to_store = set(store_output.block_hashes_to_store)

            block_hashes = self._get_block_hashes(req, end_idx=num_blocks)
            self.manager.touch(block_hashes)

            new_block_hashes = self._get_block_hashes(
                req, start_idx=start_block_idx, end_idx=num_blocks
            )
            dst_spec = store_output.store_spec
            src_block_ids: list[int] = []
            for idx, blk_hash in enumerate(new_block_hashes):
                if blk_hash not in block_hashes_to_store:
                    continue
                offloaded_block_idx = start_block_idx + idx
                gpu_block_idx = offloaded_block_idx * self.block_size_factor
                for i in range(self.block_size_factor):
                    src_block_ids.append(block_ids[gpu_block_idx + i])
            src_spec = GPULoadStoreSpec(src_block_ids)

            reqs_to_store[req_id] = (src_spec, dst_spec)
            self._reqs_being_stored[req_id] |= block_hashes_to_store

            logger.debug(
                "Request %s offloading %s blocks starting from block #%d",
                req_id,
                len(block_hashes_to_store),
                start_block_idx,
            )

        return reqs_to_store

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = OffloadingConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=self._get_reqs_to_store(scheduler_output),
        )
        self._reqs_to_load = {}
        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        for req_id in connector_output.finished_sending or []:
            block_hashes = self._reqs_being_stored.pop(req_id, None)
            if block_hashes:
                self.manager.complete_store(block_hashes)

        for req_id in connector_output.finished_recving or []:
            block_hashes = self._reqs_being_loaded.pop(req_id, None)
            if block_hashes:
                self.manager.complete_load(block_hashes)

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
        self._requests.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)

        request_being_stored = req_id in self._reqs_being_stored
        return request_being_stored, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Take the KV cache events from the connector.

        Returns:
            A list of KV cache events.
        """
        for event in self.manager.take_events():
            if event.removed:
                yield BlockRemoved(block_hashes=event.block_hashes, medium=event.medium)
            else:
                yield BlockStored(
                    block_hashes=event.block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=event.block_size,
                    medium=event.medium,
                )


class OffloadingConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, spec: OffloadingSpec, bytes_per_block: int):
        self.spec = spec
        self.worker = OffloadingWorker()

        self._job_counter = 0

        self.kv_connector_stats = OffloadingConnectorStats()
        # req_id -> (job_id, store)
        self._jobs: dict[int, tuple[ReqId, bool]] = {}
        # req_id -> active job IDs
        self._load_job: dict[ReqId, int] = {}
        # req_id -> set(active job IDs)
        self._store_jobs = defaultdict[ReqId, set[int]](set)

        self._finished_reqs_waiting_for_store: set[ReqId] = set()

        self._bytes_per_block = bytes_per_block

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter = job_id + 1
        return job_id

    def _register_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ):
        for src_cls, dst_cls, handler in self.spec.get_handlers(
            kv_caches, attn_backends
        ):
            self.worker.register_handler(src_cls, dst_cls, handler)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        layer_names = list(kv_caches.keys())
        layers = get_layers_from_vllm_config(
            self.spec.vllm_config, Attention, layer_names
        )
        attn_backends = {
            layer_name: layers[layer_name].get_attn_backend()
            for layer_name in layer_names
        }
        self._register_handlers(kv_caches, attn_backends)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        cross_layer_name = "ALL_LAYERS"
        kv_caches = {cross_layer_name: kv_cache}
        attn_backends = {cross_layer_name: attn_backend}
        self._register_handlers(kv_caches, attn_backends)

    def start_load_kv(self, metadata: OffloadingConnectorMetadata):
        for req_id, transfer_spec in metadata.reqs_to_load.items():
            job_id = self._generate_job_id()
            src_spec, dst_spec = transfer_spec
            assert isinstance(src_spec, BlockIDsLoadStoreSpec)
            self._jobs[job_id] = (req_id, False)
            assert req_id not in self._load_job
            self._load_job[req_id] = job_id
            assert self.worker.transfer_async(job_id, transfer_spec)

    def start_store_kv(self, metadata: OffloadingConnectorMetadata):
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            job_id = self._generate_job_id()
            src_spec, dst_spec = transfer_spec
            assert isinstance(dst_spec, BlockIDsLoadStoreSpec)
            self._jobs[job_id] = (req_id, True)
            self._store_jobs[req_id].add(job_id)
            assert self.worker.transfer_async(job_id, transfer_spec)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.
        Returns a list of request IDs that finished loading or storing.

        Returns:
            ids of requests that have finished asynchronous transfer
            tuple of (sending/saving ids, recving/loading ids).
        """
        finished_sending = set()
        finished_recving = set()
        for job_id, success in self.worker.get_finished():
            # we currently do not support job failures
            assert success
            req_id, store = self._jobs.pop(job_id)
            num_blocks, transfer_time, transfer_type = self.worker.get_stats(job_id)
            self.kv_connector_stats.record_transfer(
                num_blocks * self._bytes_per_block, transfer_time, transfer_type
            )
            if store:
                req_jobs = self._store_jobs[req_id]
                req_jobs.remove(job_id)
                if req_jobs:
                    continue

                if req_id in self._finished_reqs_waiting_for_store:
                    self._finished_reqs_waiting_for_store.remove(req_id)
                    finished_sending.add(req_id)
                    del self._store_jobs[req_id]
            else:
                req_job = self._load_job[req_id]
                assert job_id == req_job
                del self._load_job[req_id]
                finished_recving.add(req_id)

        for req_id in finished_req_ids:
            pending_req_jobs = self._store_jobs.get(req_id)
            if pending_req_jobs:
                self._finished_reqs_waiting_for_store.add(req_id)
            elif pending_req_jobs is not None:
                finished_sending.add(req_id)
                del self._store_jobs[req_id]

        return finished_sending, finished_recving

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """
        # Clear stats for next iteration
        if not self.kv_connector_stats.is_empty():
            return self.kv_connector_stats.clone_and_reset()
        return None


class OffloadPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self.total_cpu_to_gpu_time = 0
        self.total_gpu_to_cpu_time = 0
        self.total_cpu_to_gpu_count = 0
        self.total_gpu_to_cpu_count = 0
        self.total_cpu_to_gpu_bytes = 0
        self.total_gpu_to_cpu_bytes = 0

        buckets = [  # In bytes
            1e6,
            5e6,
            10e6,
            20e6,
            40e6,
            60e6,
            80e6,
            100e6,
            150e6,
            200e6,
        ]

        self.size_buckets = self.create_size_buckets(buckets)

        self.gauge_cpu_to_gpu_throughput_by_bucket = {}
        self.gauge_gpu_to_cpu_throughput_by_bucket = {}
        for bucket in self.size_buckets:
            gauge = self._gauge_cls(
                name=f"vllm:kv_offload_cpu_to_gpu_throughput_{bucket}",
                documentation=f"Average throughput for CPU-GPU transfers {bucket}",
                multiprocess_mode="mostrecent",
                labelnames=labelnames,
            )
            self.gauge_cpu_to_gpu_throughput_by_bucket[bucket] = self.make_per_engine(
                gauge
            )

            gauge = self._gauge_cls(
                name=f"vllm:kv_offload_gpu_to_cpu_throughput_{bucket}",
                documentation=f"Average throughput for GPU-CPU transfers {bucket}",
                multiprocess_mode="mostrecent",
                labelnames=labelnames,
            )
            self.gauge_gpu_to_cpu_throughput_by_bucket[bucket] = self.make_per_engine(
                gauge
            )

        self.cpu_to_gpu_bucket_stats = {
            bucket: [0, 0] for bucket in self.size_buckets
        }  # [total_bytes, total_time]
        self.gpu_to_cpu_bucket_stats = {bucket: [0, 0] for bucket in self.size_buckets}

        offload_histogram_size_cpu_to_gpu = self._histogram_cls(
            name="vllm:kv_offload_size_cpu_to_gpu",
            documentation="Histogram of CPU to GPU transfer size, in bytes.",
            buckets=buckets[:],
            labelnames=labelnames,
        )
        self.offload_histogram_size_cpu_to_gpu = self.make_per_engine(
            offload_histogram_size_cpu_to_gpu
        )

        offload_histogram_size_gpu_to_cpu = self._histogram_cls(
            name="vllm:kv_offload_size_gpu_to_cpu",
            documentation="Histogram of GPU to CPU transfer size, in bytes.",
            buckets=buckets[:],
            labelnames=labelnames,
        )
        self.offload_histogram_size_gpu_to_cpu = self.make_per_engine(
            offload_histogram_size_gpu_to_cpu
        )

        gauge_kv_gpu_to_cpu_throughput_avg = self._gauge_cls(
            name="vllm:kv_offload_gpu_to_cpu_throughput",
            documentation="Average throughput of a GPU-CPU transfers",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_kv_gpu_to_cpu_throughput_avg = self.make_per_engine(
            gauge_kv_gpu_to_cpu_throughput_avg
        )

        gauge_kv_cpu_to_gpu_throughput_avg = self._gauge_cls(
            name="vllm:kv_offload_cpu_to_gpu_throughput",
            documentation="Average throughput of a CPU-GPU transfers",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )
        self.gauge_kv_cpu_to_gpu_throughput_avg = self.make_per_engine(
            gauge_kv_cpu_to_gpu_throughput_avg
        )

        counter_num_cpu_to_gpu = self._counter_cls(
            name="vllm:kv_offload_num_cpu_to_gpu",
            documentation="Number of CPU-GPU transfers done",
            labelnames=labelnames,
        )
        self.counter_num_cpu_to_gpu = self.make_per_engine(counter_num_cpu_to_gpu)

        counter_num_gpu_to_cpu = self._counter_cls(
            name="vllm:kv_offload_num_gpu_to_cpu",
            documentation="Number of GPU-CPU transfers done",
            labelnames=labelnames,
        )
        self.counter_num_gpu_to_cpu = self.make_per_engine(counter_num_gpu_to_cpu)

    def create_size_buckets(self, buckets):
        current_bucket = "0m"
        size_buckets: list[str] = []
        for bucket in buckets:
            scale_char = "k"
            if bucket < 1e6:
                next_bucket_val = bucket / 1000
            elif bucket < 1e9:
                next_bucket_val = bucket / 1e6
                scale_char = "m"
            else:
                next_bucket_val = bucket / 1e9
                scale_char = "g"
            next_bucket = str(next_bucket_val)
            first, second = next_bucket.split(".")
            next_bucket = first + scale_char
            arg_to_append = current_bucket + "_" + next_bucket
            size_buckets.append(arg_to_append)
            current_bucket = next_bucket
        size_buckets.append(current_bucket + "_plus")
        return size_buckets

    def get_size_bucket(self, size_bytes):
        for bucket in self.size_buckets:
            lower_str, upper_str = bucket.split("_")
            if upper_str == "plus":
                return bucket
            multiplier = 1.0
            if upper_str.endswith("k"):
                multiplier = 1000.0
            elif upper_str.endswith("m"):
                multiplier = 1e6
            elif upper_str.endswith("g"):
                multiplier = 1e9
            if size_bytes < float(upper_str[:-1]) * multiplier:
                return bucket
        return self.size_buckets[-1]

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        """
        Observe transfer statistics from the new data structure.
        transfer_stats_data is expected to be a dict where:
        - keys are transfer type strings (e.g., "cpu_to_gpu", "gpu_to_cpu")
        - values are lists of OffloadingOperationMetrics objects
        """

        # Process cpu_to_gpu operations
        if "CPU_to_GPU" in transfer_stats_data:
            cpu_to_gpu_ops = transfer_stats_data["CPU_to_GPU"]
            for op in cpu_to_gpu_ops:
                # Observe size histogram
                self.offload_histogram_size_cpu_to_gpu[engine_idx].observe(
                    op["op_size"]
                )

                # Calculate and observe throughput histogram
                bucket = self.get_size_bucket(op["op_size"])
                self.cpu_to_gpu_bucket_stats[bucket][0] += op["op_size"]
                self.cpu_to_gpu_bucket_stats[bucket][1] += op["op_time"]
                self.total_cpu_to_gpu_time += op["op_time"]
                self.total_cpu_to_gpu_bytes += op["op_size"]

            # Update counter
            self.counter_num_cpu_to_gpu[engine_idx].inc(len(cpu_to_gpu_ops))

            # Update gauge with cumulative average

            avg_cpu_to_gpu_throughput = self.total_cpu_to_gpu_bytes / max(
                self.total_cpu_to_gpu_time, 1
            )
            self.gauge_kv_cpu_to_gpu_throughput_avg[engine_idx].set(
                avg_cpu_to_gpu_throughput
            )

            # Update bucket-specific gauges
            for bucket, (
                total_bytes,
                total_time,
            ) in self.cpu_to_gpu_bucket_stats.items():
                if total_time > 0:
                    avg_throughput = total_bytes / total_time
                    self.gauge_cpu_to_gpu_throughput_by_bucket[bucket][engine_idx].set(
                        avg_throughput
                    )

        # Process gpu_to_cpu operations
        if "GPU_to_CPU" in transfer_stats_data:
            gpu_to_cpu_ops = transfer_stats_data["GPU_to_CPU"]
            assert isinstance(gpu_to_cpu_ops, list)
            for op in gpu_to_cpu_ops:
                # Observe size histogram
                self.offload_histogram_size_gpu_to_cpu[engine_idx].observe(
                    op["op_size"]
                )

                # Calculate and observe throughput histogram
                bucket = self.get_size_bucket(op["op_size"])
                self.gpu_to_cpu_bucket_stats[bucket][0] += op["op_size"]
                self.gpu_to_cpu_bucket_stats[bucket][1] += op["op_time"]
                self.total_gpu_to_cpu_time += op["op_time"]
                self.total_gpu_to_cpu_bytes += op["op_size"]

            # Update counter
            self.counter_num_gpu_to_cpu[engine_idx].inc(len(gpu_to_cpu_ops))

            # Update gauge with cumulative average

            avg_gpu_to_cpu_throughput = self.total_gpu_to_cpu_bytes / max(
                self.total_gpu_to_cpu_time, 1
            )
            self.gauge_kv_gpu_to_cpu_throughput_avg[engine_idx].set(
                avg_gpu_to_cpu_throughput
            )

            for bucket, (
                total_bytes,
                total_time,
            ) in self.gpu_to_cpu_bucket_stats.items():
                if total_time > 0:
                    avg_throughput = total_bytes / total_time
                    self.gauge_gpu_to_cpu_throughput_by_bucket[bucket][engine_idx].set(
                        avg_throughput
                    )
