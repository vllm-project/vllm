# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
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
from vllm.v1.kv_offload.worker.worker import OffloadingWorker, TransferSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

ReqId = str

logger = init_logger(__name__)


@dataclass
class OffloadingConnectorStats(KVConnectorStats):
    gpu_block_size: int = 0
    offloaded_block_size: int = 0

    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        self.data: dict[str, list[float]] = {
            "load_size": [],
            "store_size": [],
            "load_throughput": [],
            "store_throughput": [],
            "store_time": [],
            "load_time": [],
            "total_transactions": [0, 0],  # num_stores, num_loads
        }

    def clone_and_reset(self) -> "OffloadingConnectorStats":
        old = copy.copy(self)
        self.reset()
        return old

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if not other.is_empty():
            for k, v in other.data.items():
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
        return {
            "Avg Load Time": sum(self.data["load_time"])
            / max(len(self.data["load_time"]), 1),
            "Avg Store Time": sum(self.data["store_time"])
            / max(len(self.data["store_time"]), 1),
        }

    def is_empty(self) -> bool:
        return len(self.data["store_size"]) == 0

    def record_transfer(self, num_bytes: int, time: float, is_store: bool):
        time = time / 1e6
        if is_store:
            self.data["store_throughput"].append(num_bytes / time)
            self.data["store_size"].append(num_bytes)
            self.data["store_time"].append(time)
            self.data["total_transactions"][0] += 1
        else:
            self.data["load_throughput"].append(num_bytes / time)
            self.data["load_size"].append(num_bytes)
            self.data["load_time"].append(time)
            self.data["total_transactions"][1] += 1


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec]
    reqs_to_store: dict[ReqId, TransferSpec]


class OffloadingConnector(KVConnectorBase_V1):
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
            self.connector_worker = OffloadingConnectorWorker(spec)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

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
        num_new_matched_tokens, hit = (
            self.connector_scheduler.get_num_new_matched_tokens(
                request, num_computed_tokens
            )
        )
        return num_new_matched_tokens, hit

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
        per_engine_labelvalues: dict[int, list[str]],
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

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        self.worker = OffloadingWorker()

        self._job_counter = 0

        self.kv_connector_stats = OffloadingConnectorStats(
            gpu_block_size=spec.gpu_block_size,
            offloaded_block_size=spec.offloaded_block_size,
        )

        # req_id -> (job_id, store, start_time, num_blocks)
        self._jobs: dict[int, tuple[ReqId, bool, float, int]] = {}
        # req_id -> active job IDs
        self._load_job: dict[ReqId, int] = {}
        # req_id -> set(active job IDs)
        self._store_jobs = defaultdict[ReqId, set[int]](set)

        self._finished_reqs_waiting_for_store: set[ReqId] = set()

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter = job_id + 1
        return job_id

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        for src_cls, dst_cls, handler in self.spec.get_handlers(kv_caches):
            self.worker.register_handler(src_cls, dst_cls, handler)

    def start_load_kv(self, metadata: OffloadingConnectorMetadata):
        for req_id, transfer_spec in metadata.reqs_to_load.items():
            job_id = self._generate_job_id()
            current_time = time.perf_counter()
            src_spec, dst_spec = transfer_spec
            assert isinstance(src_spec, BlockIDsLoadStoreSpec)
            self._jobs[job_id] = (req_id, False, current_time, src_spec.num_blocks)
            assert req_id not in self._load_job
            self._load_job[req_id] = job_id
            assert self.worker.transfer_async(job_id, transfer_spec)

    def start_store_kv(self, metadata: OffloadingConnectorMetadata):
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            job_id = self._generate_job_id()
            current_time = time.perf_counter()
            src_spec, dst_spec = transfer_spec
            assert isinstance(dst_spec, BlockIDsLoadStoreSpec)
            self._jobs[job_id] = (req_id, True, current_time, dst_spec.num_blocks)
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
            req_id, store, start_time, num_blocks = self._jobs.pop(job_id)
            block_size = (
                self.spec.gpu_block_size if store else self.spec.offloaded_block_size
            )  # * bytes_per_token
            self.kv_connector_stats.record_transfer(
                num_blocks * block_size, (time.perf_counter() - start_time), store
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


def yield_req_data(
    scheduler_output,
) -> Iterator[tuple[str, tuple[list[int], ...], bool]]:
    """
    Yields:
        (req_id, new_block_id_groups, preempted)
    """
    # new requests
    for req_data in scheduler_output.scheduled_new_reqs:
        yield req_data.req_id, req_data.block_ids, False

    # cached requests
    cached_reqs = scheduler_output.scheduled_cached_reqs
    yield from zip(
        cached_reqs.req_ids,
        cached_reqs.new_block_ids,
        (req_id in cached_reqs.resumed_req_ids for req_id in cached_reqs.req_ids),
    )


class OffloadPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)

        buckets = [  # In K tokens per sec
            10000000,
            100000000,
            500000000,
            1000000000,
            5000000000,
            10000000000,
            30000000000,
            70000000000,
        ]

        offload_histogram_throughput_load = self._histogram_cls(
            name="vllm:kv_offload_throughput_load",
            documentation="Histogram of load throughput, in K tokens.",
            buckets=buckets[:],
            labelnames=labelnames,
        )
        self.offload_histogram_throughput_load = self.make_per_engine(
            offload_histogram_throughput_load
        )

        offload_histogram_throughput_store = self._histogram_cls(
            name="vllm:kv_offload_throughput_store",
            documentation="Histogram of store throughput, in K tokens.",
            buckets=buckets[:],
            labelnames=labelnames,
        )
        self.offload_histogram_throughput_store = self.make_per_engine(
            offload_histogram_throughput_store
        )

        buckets = [  # In K tokens
            10,
            15,
            20,
            25,
            30,
            50,
            80,
            100,
            200,
            300,
            500,
            1000,
            10000,
            50000,
        ]

        offload_histogram_size_load = self._histogram_cls(
            name="vllm:kv_offload_size_load",
            documentation="Histogram of load size, in K tokens.",
            buckets=buckets[:],
            labelnames=labelnames,
        )
        self.offload_histogram_size_load = self.make_per_engine(
            offload_histogram_size_load
        )

        offload_histogram_size_store = self._histogram_cls(
            name="vllm:kv_offload_size_store",
            documentation="Histogram of store size, in K tokens.",
            buckets=buckets[:],
            labelnames=labelnames,
        )
        self.offload_histogram_size_store = self.make_per_engine(
            offload_histogram_size_store
        )

        counter_offload_block_queries = self._counter_cls(
            name="vllm:kv_offload_block_queries",
            documentation="Number of blocks queried by Offloading KV Connector.",
            labelnames=labelnames,
        )

        self.counter_offload_block_queries = self.make_per_engine(
            counter_offload_block_queries
        )
        counter_offload_block_hits = self._counter_cls(
            name="vllm:kv_offload_block_hits",
            documentation="Number of blocks queried and hit by Offloading"
            " KV Connector.",
            labelnames=labelnames,
        )
        self.counter_offload_block_hits = self.make_per_engine(
            counter_offload_block_hits
        )

        gauge_kv_load_time_avg = self._gauge_cls(
            name="vllm:kv_offload_load_time",
            documentation="Average time it takes to load a token from "
            "offloaded KV memory",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )

        self.gauge_kv_load_time_avg = self.make_per_engine(gauge_kv_load_time_avg)

        gauge_kv_store_time_avg = self._gauge_cls(
            name="vllm:kv_offload_store_time",
            documentation="Average time it takes to store a token to "
            "offloaded KV memory",
            multiprocess_mode="mostrecent",
            labelnames=labelnames,
        )

        self.gauge_kv_store_time_avg = self.make_per_engine(gauge_kv_store_time_avg)

        counter_num_loads = self._counter_cls(
            name="vllm:kv_offload_num_loads",
            documentation="Number of loads done from offloaded KV cache",
            labelnames=labelnames,
        )
        self.counter_num_loads = self.make_per_engine(counter_num_loads)

        counter_num_stores = self._counter_cls(
            name="vllm:kv_offload_num_stores",
            documentation="Number of stores done into offloaded KV cache",
            labelnames=labelnames,
        )
        self.counter_num_stores = self.make_per_engine(counter_num_stores)

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        # Gauges:
        avg_load_time = sum(transfer_stats_data["load_time"]) / max(
            len(transfer_stats_data["load_time"]), 1
        )  # Avoid division by 0
        avg_store_time = sum(transfer_stats_data["store_time"]) / max(
            len(transfer_stats_data["store_time"]), 1
        )
        self.gauge_kv_load_time_avg[engine_idx].set(avg_load_time)
        self.gauge_kv_store_time_avg[engine_idx].set(avg_store_time)

        # Histograms:
        for prom_obj, list_item_key in zip(
            [
                self.offload_histogram_size_load,
                self.offload_histogram_size_store,
                self.offload_histogram_throughput_load,
                self.offload_histogram_throughput_store,
            ],
            [
                "load_size",
                "store_size",
                "load_throughput",
                "store_throughput",
            ],
        ):
            for list_item in transfer_stats_data[list_item_key]:
                prom_obj[engine_idx].observe(list_item)

        # Counters:
        self.counter_num_stores[engine_idx].inc(
            transfer_stats_data["total_transactions"][0]
        )
        self.counter_num_loads[engine_idx].inc(
            transfer_stats_data["total_transactions"][1]
        )
