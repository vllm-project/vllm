# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SwapConnector: KV connector that swaps KV cache between CPU and GPU
on a per-layer basis.

Unlike the OffloadingConnector which loads/stores all layers at once,
the SwapConnector loads a single layer's KV cache from CPU to GPU
before that layer's attention computation, and stores it back after.
This keeps only one layer's KV cache on GPU at a time.
"""
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import OffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingWorker, TransferSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

ReqId = str

logger = init_logger(__name__)


@dataclass
class SwapConnectorMetadata(KVConnectorMetadata):
    # Blocks to load from CPU for prefix cache hits (same as offloading)
    reqs_to_load: dict[ReqId, TransferSpec]
    # New blocks to store to CPU (scheduler tracks these)
    reqs_to_store: dict[ReqId, TransferSpec]
    # GPU block IDs per active request (for per-layer swap)
    active_gpu_block_ids: dict[ReqId, list[int]]
    # GPU block ID -> CPU block ID mapping for blocks on CPU
    gpu_to_cpu_block_map: dict[int, int]
    # Requests that are new this step (no KV on CPU yet, skip loading)
    new_req_ids: set[ReqId] = field(default_factory=set)


class SwapConnector(KVConnectorBase_V1):
    """
    KV connector that swaps KV cache between CPU and GPU per-layer.

    On each forward pass:
    - Before each layer's attention: load that layer's KV from CPU->GPU
    - After each layer's attention: store that layer's KV from GPU->CPU
    - Only one layer's KV cache is on GPU at a time
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)

        self.connector_scheduler: SwapConnectorScheduler | None = None
        self.connector_worker: SwapConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = SwapConnectorScheduler(spec)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = SwapConnectorWorker(spec)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def handle_preemptions(self, preempted_req_ids: set[str]):
        assert self.connector_worker is not None
        self.connector_worker.handle_preemptions(preempted_req_ids)

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs
    ) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, SwapConnectorMetadata)
        self.connector_worker.start_kv_transfers(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, SwapConnectorMetadata)
        self.connector_worker.load_layer_from_cpu(
            layer_name, self._connector_metadata
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, SwapConnectorMetadata)
        self.connector_worker.store_layer_to_cpu(
            layer_name, self._connector_metadata
        )

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, SwapConnectorMetadata)
        self.connector_worker.wait_for_all_stores()
        # Also prepare deferred bulk stores (same as offloading connector)
        self.connector_worker.prepare_store_kv(self._connector_metadata)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str], set[str]]:
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
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
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


class SwapConnectorScheduler:
    """
    Scheduler-side implementation for SwapConnector.

    Nearly identical to OffloadingConnectorScheduler, but additionally
    tracks active block IDs and GPU<->CPU block mappings for per-layer swap,
    and uses SwapManager (no eviction).
    """

    def __init__(self, spec: OffloadingSpec):
        self.gpu_block_size = spec.gpu_block_size
        self.offloaded_block_size = spec.offloaded_block_size
        self.block_size_factor = self.offloaded_block_size // self.gpu_block_size
        self.manager: OffloadingManager = spec.get_manager()

        self._requests: dict[ReqId, Request] = {}
        # list of GPU block IDs per request
        self._request_block_ids: dict[ReqId, list[int]] = {}
        # requests to load for the current scheduler step (prefix cache hits)
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        # index of next block (of size offloaded_block_size) to offload
        self._next_stored_block_idx: dict[ReqId, int] = {}

        # request ID -> set(block hashes being stored/loaded)
        self._reqs_being_stored = defaultdict[ReqId, set[BlockHash]](set)
        self._reqs_being_loaded = defaultdict[ReqId, set[BlockHash]](set)

        # GPU block ID -> CPU block ID mapping (persistent across steps)
        self._gpu_to_cpu_block_map: dict[int, int] = {}
        # Track which requests are new (first time, no KV on CPU)
        self._new_req_ids: set[ReqId] = set()

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
        num_blocks = request.num_tokens // self.offloaded_block_size

        assert (
            len(request.block_hashes) // self.block_size_factor == num_blocks
        )
        block_hashes = self._get_block_hashes(request)

        self.manager.touch(block_hashes)

        full_block_tokens = self.offloaded_block_size * num_blocks
        if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
            return 0, False

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        hits = self.manager.lookup(
            self._get_block_hashes(request, start_idx=start_block_idx)
        )
        if hits == 0:
            return 0, False

        num_hit_tokens = (
            self.offloaded_block_size * (start_block_idx + hits)
            - num_computed_tokens
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
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ):
        req_id = request.request_id
        self._requests[req_id] = request
        self._request_block_ids[req_id] = []

        if num_external_tokens == 0:
            # New request with no prefix cache hit
            self._new_req_ids.add(req_id)
            return

        # Prefix cache hit: prepare load spec
        block_groups = blocks.get_block_ids()
        block_ids = block_groups[0]

        num_computed_gpu_blocks = sum(
            block.block_hash is not None for block in blocks.blocks[0]
        )
        num_computed_tokens = num_computed_gpu_blocks * self.gpu_block_size
        full_block_tokens = num_computed_tokens + num_external_tokens
        assert full_block_tokens % self.offloaded_block_size == 0

        num_pending_gpu_blocks = len(block_ids) - num_computed_gpu_blocks
        assert (
            num_external_tokens == num_pending_gpu_blocks * self.gpu_block_size
        )

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        num_blocks = full_block_tokens // self.offloaded_block_size

        assert (
            len(request.block_hashes) // self.block_size_factor >= num_blocks
        )
        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        src_spec = self.manager.prepare_load(block_hashes)
        dst_spec = GPULoadStoreSpec(block_ids[num_computed_gpu_blocks:])

        block_hashes = self._get_block_hashes(
            request, start_idx=start_block_idx, end_idx=num_blocks
        )

        self._reqs_to_load[req_id] = (src_spec, dst_spec)
        self._reqs_being_loaded[req_id].update(block_hashes)
        self._next_stored_block_idx[req_id] = num_blocks

    def _get_reqs_to_store(self, scheduler_output: SchedulerOutput):
        reqs_to_store: dict[ReqId, TransferSpec] = {}
        for req_id, new_block_id_groups, preempted in yield_req_data(
            scheduler_output
        ):
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

            new_block_hashes = self._get_block_hashes(
                req, start_idx=start_block_idx, end_idx=num_blocks
            )
            store_output = self.manager.prepare_store(new_block_hashes)
            if store_output is None:
                logger.warning(
                    "Request %s: cannot store %s blocks",
                    req_id,
                    num_new_blocks,
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

            # Track GPU->CPU block mapping from store specs
            from vllm.v1.kv_offload.mediums import (
                BlockIDsLoadStoreSpec,
                CPULoadStoreSpec,
            )

            if isinstance(dst_spec, BlockIDsLoadStoreSpec):
                gpu_ids = src_spec.block_ids
                cpu_ids = dst_spec.block_ids
                for gpu_id, cpu_id in zip(gpu_ids, cpu_ids):
                    self._gpu_to_cpu_block_map[int(gpu_id)] = int(cpu_id)

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
        reqs_to_store = self._get_reqs_to_store(scheduler_output)

        # Build active GPU block IDs for all requests in this step
        active_gpu_block_ids: dict[ReqId, list[int]] = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            if req_id in self._request_block_ids:
                active_gpu_block_ids[req_id] = self._request_block_ids[
                    req_id
                ]

        meta = SwapConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_store=reqs_to_store,
            active_gpu_block_ids=active_gpu_block_ids,
            gpu_to_cpu_block_map=dict(self._gpu_to_cpu_block_map),
            new_req_ids=set(self._new_req_ids),
        )
        self._reqs_to_load = {}
        self._new_req_ids.clear()

        # Handle preemptions
        for req_id in scheduler_output.preempted_req_ids or ():
            block_hashes = self._reqs_being_stored.get(req_id)
            if block_hashes:
                self.manager.complete_store(block_hashes)
                block_hashes.clear()

        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
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
        req_id = request.request_id
        # Clean up GPU->CPU block mapping for this request's blocks
        for block_id in self._request_block_ids.get(req_id, []):
            self._gpu_to_cpu_block_map.pop(block_id, None)

        self._requests.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)
        self._new_req_ids.discard(req_id)

        request_being_stored = req_id in self._reqs_being_stored
        return request_being_stored, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        for event in self.manager.take_events():
            if event.removed:
                yield BlockRemoved(
                    block_hashes=event.block_hashes, medium=event.medium
                )
            else:
                yield BlockStored(
                    block_hashes=event.block_hashes,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=event.block_size,
                    medium=event.medium,
                    lora_name=None,
                )


class SwapConnectorWorker:
    """
    Worker-side implementation for SwapConnector.

    Manages per-layer CPU<->GPU KV cache transfers.
    """

    def __init__(self, spec: OffloadingSpec):
        self.spec = spec
        # Reuse the OffloadingWorker for bulk prefix cache loads/stores
        self.worker = OffloadingWorker()

        self._job_counter = 0

        # job_id -> (req_id, store)
        self._jobs: dict[int, tuple[ReqId, bool]] = {}
        # req_id -> active load job ID
        self._load_job: dict[ReqId, int] = {}
        # req_id -> set(active store job IDs)
        self._store_jobs = defaultdict[ReqId, set[int]](set)
        # Deferred store jobs
        self._unsubmitted_store_jobs: list[tuple[int, TransferSpec]] = []
        self._finished_reqs_waiting_for_store: set[ReqId] = set()

        # Per-layer CPU tensors and GPU tensors
        self._gpu_tensors: dict[str, torch.Tensor] = {}
        self._cpu_tensors: dict[str, torch.Tensor] = {}
        self._kv_dim_before_num_blocks: dict[str, bool] = {}

        # CUDA streams for per-layer transfers
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._store_event: torch.Event | None = None

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter = job_id + 1
        return job_id

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Create per-layer CPU tensors and register handlers for bulk transfers.
        """
        layer_names = list(kv_caches.keys())
        layers = get_layers_from_vllm_config(
            self.spec.vllm_config, Attention, layer_names
        )
        attn_backends = {
            layer_name: layers[layer_name].get_attn_backend()
            for layer_name in layer_names
        }

        # Register handlers for bulk transfers (prefix cache loads/stores)
        for src_cls, dst_cls, handler in self.spec.get_handlers(
            kv_caches, attn_backends
        ):
            self.worker.register_handler(src_cls, dst_cls, handler)

        pin_memory = is_pin_memory_available()
        num_cpu_blocks = self.spec.num_blocks
        gpu_block_size = self.spec.gpu_block_size
        cpu_block_size = self.spec.offloaded_block_size
        block_size_factor = cpu_block_size // gpu_block_size

        # Create per-layer CPU tensors
        for layer_name, gpu_tensor in kv_caches.items():
            self._gpu_tensors[layer_name] = gpu_tensor

            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234,
                block_size=16,
                num_kv_heads=8,
                head_size=256,
            )

            if test_shape[0] == 1234:
                # shape is (num_blocks, ...)
                num_blocks_idx = 0
                self._kv_dim_before_num_blocks[layer_name] = False
            else:
                # shape should be (2, num_blocks, ...)
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2
                num_blocks_idx = 1
                self._kv_dim_before_num_blocks[layer_name] = True

            cpu_shape = list(gpu_shape)
            cpu_shape[num_blocks_idx] = num_cpu_blocks * block_size_factor

            logger.debug(
                "Allocating CPU tensor for layer %s of shape %r",
                layer_name,
                cpu_shape,
            )
            self._cpu_tensors[layer_name] = torch.zeros(
                cpu_shape,
                dtype=gpu_tensor.dtype,
                device="cpu",
                pin_memory=pin_memory,
            )

    def handle_preemptions(self, preempted_req_ids: set[str]):
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        for req_id in preempted_req_ids:
            job_ids = self._store_jobs.get(req_id)
            if job_ids:
                self.worker.wait(job_ids)

    def start_kv_transfers(self, metadata: SwapConnectorMetadata):
        """Submit deferred stores and start prefix cache loads."""
        # Submit deferred store jobs from the previous step
        for job_id, transfer_spec in self._unsubmitted_store_jobs:
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success
        self._unsubmitted_store_jobs.clear()

        # Start prefix cache loads (bulk transfer, same as offloading)
        for req_id, transfer_spec in metadata.reqs_to_load.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, False)
            assert req_id not in self._load_job
            self._load_job[req_id] = job_id
            success = self.worker.transfer_async(job_id, transfer_spec)
            assert success

    def load_layer_from_cpu(
        self,
        layer_name: str,
        metadata: SwapConnectorMetadata,
    ):
        """
        Load a single layer's KV cache from CPU to GPU for all active
        requests that already have KV on CPU (skip new requests).
        """
        if layer_name not in self._cpu_tensors:
            return

        # Collect all GPU block IDs that need loading (non-new requests)
        gpu_block_ids = []
        cpu_block_ids = []
        for req_id, block_ids in metadata.active_gpu_block_ids.items():
            if req_id in metadata.new_req_ids:
                continue
            for gpu_id in block_ids:
                cpu_id = metadata.gpu_to_cpu_block_map.get(gpu_id)
                if cpu_id is not None:
                    gpu_block_ids.append(gpu_id)
                    cpu_block_ids.append(cpu_id)

        if not gpu_block_ids:
            return

        # Deduplicate (multiple requests might share blocks)
        seen = set()
        unique_gpu = []
        unique_cpu = []
        for gpu_id, cpu_id in zip(gpu_block_ids, cpu_block_ids):
            if gpu_id not in seen:
                seen.add(gpu_id)
                unique_gpu.append(gpu_id)
                unique_cpu.append(cpu_id)

        # Build src_to_dst mapping for ops.swap_blocks
        src_to_dst = np.column_stack(
            [
                np.array(unique_cpu, dtype=np.int64),
                np.array(unique_gpu, dtype=np.int64),
            ]
        )
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        # Lazily create CUDA stream
        if self._load_stream is None:
            self._load_stream = torch.cuda.Stream()

        # Wait for any previous store to complete before loading
        if self._store_event is not None:
            self._load_stream.wait_event(self._store_event)

        cpu_tensor = self._cpu_tensors[layer_name]
        gpu_tensor = self._gpu_tensors[layer_name]
        kv_dim = self._kv_dim_before_num_blocks[layer_name]

        with torch.cuda.stream(self._load_stream):
            if kv_dim:
                ops.swap_blocks(cpu_tensor[0], gpu_tensor[0], src_to_dst_tensor)
                ops.swap_blocks(cpu_tensor[1], gpu_tensor[1], src_to_dst_tensor)
            else:
                ops.swap_blocks(cpu_tensor, gpu_tensor, src_to_dst_tensor)

        # Must synchronize: attention needs the data to be ready
        self._load_stream.synchronize()

    def store_layer_to_cpu(
        self,
        layer_name: str,
        metadata: SwapConnectorMetadata,
    ):
        """
        Store a single layer's KV cache from GPU to CPU for all active
        requests (including new requests).
        """
        if layer_name not in self._cpu_tensors:
            return

        # Collect all GPU block IDs that need storing
        gpu_block_ids = []
        cpu_block_ids = []
        for req_id, block_ids in metadata.active_gpu_block_ids.items():
            for gpu_id in block_ids:
                cpu_id = metadata.gpu_to_cpu_block_map.get(gpu_id)
                if cpu_id is not None:
                    gpu_block_ids.append(gpu_id)
                    cpu_block_ids.append(cpu_id)

        if not gpu_block_ids:
            return

        # Deduplicate
        seen = set()
        unique_gpu = []
        unique_cpu = []
        for gpu_id, cpu_id in zip(gpu_block_ids, cpu_block_ids):
            if gpu_id not in seen:
                seen.add(gpu_id)
                unique_gpu.append(gpu_id)
                unique_cpu.append(cpu_id)

        # Build src_to_dst mapping (GPU -> CPU)
        src_to_dst = np.column_stack(
            [
                np.array(unique_gpu, dtype=np.int64),
                np.array(unique_cpu, dtype=np.int64),
            ]
        )
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        # Lazily create CUDA stream
        if self._store_stream is None:
            self._store_stream = torch.cuda.Stream()

        # Wait for attention computation on the default stream
        self._store_stream.wait_stream(torch.cuda.current_stream())

        cpu_tensor = self._cpu_tensors[layer_name]
        gpu_tensor = self._gpu_tensors[layer_name]
        kv_dim = self._kv_dim_before_num_blocks[layer_name]

        with torch.cuda.stream(self._store_stream):
            if kv_dim:
                ops.swap_blocks(gpu_tensor[0], cpu_tensor[0], src_to_dst_tensor)
                ops.swap_blocks(gpu_tensor[1], cpu_tensor[1], src_to_dst_tensor)
            else:
                ops.swap_blocks(gpu_tensor, cpu_tensor, src_to_dst_tensor)
            # Record event for the load stream to wait on
            if self._store_event is None:
                self._store_event = torch.Event()
            self._store_event.record(self._store_stream)

        # Do NOT synchronize here: overlap with next layer's load

    def wait_for_all_stores(self):
        """Synchronize all outstanding GPU->CPU store streams."""
        if self._store_stream is not None:
            self._store_stream.synchronize()

    def prepare_store_kv(self, metadata: SwapConnectorMetadata):
        """Prepare bulk store jobs for the scheduler's reqs_to_store."""
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, True)
            self._store_jobs[req_id].add(job_id)
            self._unsubmitted_store_jobs.append((job_id, transfer_spec))

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str], set[str]]:
        finished_sending = set()
        finished_recving = set()
        for job_id, success in self.worker.get_finished():
            assert success
            req_id, store = self._jobs.pop(job_id)
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
