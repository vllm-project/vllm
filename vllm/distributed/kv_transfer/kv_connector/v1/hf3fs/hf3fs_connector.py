# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
HF3FS KV Connector Implementation for vLLM.

This module implements a KV connector that uses the 3FS for storing and retrieving KV cache data.

Key components:
1. HF3FSConnector: Main connector implementation
   2.1 AsyncOperationManager: Manages async save/load operations with background threads
   2.2 HF3FSConnectorMetadata: Container for connector metadata
3. HF3FSMetadataServer: Mini Metadata server for HF3FS connector
4. HF3FSClient: 3FS Client Implementation
"""

import atexit
import concurrent
from concurrent.futures import Future
import hashlib
import os
import queue
from queue import Empty
import signal
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils import (
    gather_scatter_helper,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.common import (
    AtomicCounter,
    RequestSchedulingState,
    HF3FSRequestMetadata,
    HF3FSConnectorMetadata,
    LoadBlockInfo,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.gather_scatter_helper import (
    CopyBufferAllocator,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from vllm.v1.attention.backend import AttentionBackend
from vllm.forward_context import ForwardContext
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.request import Request

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from dataclasses import dataclass
import copy
import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_metadata_server import (
    Hf3fsGlobalMetadataClient as Hf3fsMetadataClient,
)

HF3FS_AVAILABLE = True
try:
    from hf3fs_fuse.io import deregister_fd
    from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_client import (
        Hf3fsClient,
    )
except Exception as e:
    HF3FS_AVAILABLE = False
    from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.hf3fs_mock_client import (
        Hf3fsClient,
    )

# Constants
DEFAULT_MAX_IO_ENTRIES = 8

logger = init_logger(__name__)


# ============================================================================
# Async Operation Management
# ============================================================================


class AsyncOperationManager:
    """
    Manages async save/load operations with background threads.
    """

    def __init__(self, connector: "HF3FSKVConnector"):
        # Store connector reference and extract commonly used attributes
        self._connector = connector
        self._device = connector._device
        self._dtype = connector._dtype
        self._shape_per_page = connector._shape_per_page
        self._bytes_per_page = connector._bytes_per_page
        self._rank = connector._rank
        self._numjobs = connector._numjobs
        self._max_device_buffer_count = connector._max_device_buffer_count

        # Operation tracking
        self._save_futures: Dict[str, List[Future]] = {}
        self._load_futures: Dict[str, Future] = {}
        self._pending_finished_requests: Set[str] = set()

        # Initialize resources
        self._init_cuda_resources()
        self._init_worker_threads()

        # Metrics
        self.hf3fs_stats = HF3FSKVConnectorStats()

        logger.info("AsyncOperationManager initialized for rank %d", self._rank)

    def _init_cuda_resources(self) -> None:
        """Initialize CUDA streams, events and buffer allocators."""
        # CUDA streams for async operations
        self._save_stream = torch.cuda.Stream()
        self._load_stream = torch.cuda.Stream()
        self._save_event = torch.cuda.Event()

        # Buffer allocators for data copying
        self._save_buffer_allocator = CopyBufferAllocator(
            self._device,
            self._dtype,
            self._shape_per_page,
            self._max_device_buffer_count,
        )
        self._load_buffer_allocator = CopyBufferAllocator(
            self._device,
            self._dtype,
            self._shape_per_page,
            self._max_device_buffer_count,
        )

    def _init_worker_threads(self) -> None:
        """Initialize worker threads and I/O executor."""
        # Thread synchronization
        self._stop_event = threading.Event()
        self._save_queue = queue.Queue()
        self._load_queue = queue.Queue()

        # I/O thread pool
        self._io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._numjobs,
            thread_name_prefix=f"HF3FS-Rank{self._rank}",
        )

        # Background worker threads
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._load_thread = threading.Thread(target=self._load_worker, daemon=True)
        self._save_thread.start()
        self._load_thread.start()

    def submit_save_operation(self, request_id: str, block_ids, block_hashes) -> Future:
        """Submit a save operation for async execution."""
        future = Future()
        main_stream_event = torch.cuda.Event()
        main_stream_event.record()
        task = (request_id, block_ids, block_hashes, future, main_stream_event)
        self._save_queue.put(task)

        if request_id not in self._save_futures:
            self._save_futures[request_id] = []
        self._save_futures[request_id].append(future)
        return future

    def submit_load_operation(self, request_id: str, block_ids, block_hashes) -> Future:
        """Submit a load operation for async execution."""
        future = Future()
        task = (request_id, block_ids, block_hashes, future)
        self._load_queue.put(task)
        self._load_futures[request_id] = future
        return future

    def get_finished_operations(
        self, finished_req_ids: Set[str]
    ) -> Tuple[Set[str], Set[str]]:
        completed_saves = self._check_completed_saves(finished_req_ids)
        completed_loads = self._check_completed_loads()

        if completed_saves or completed_loads:
            logger.info(
                "HF3FS Connector Completed: %d saves, %d loads operations",
                len(completed_saves),
                len(completed_loads),
            )

        return completed_saves, completed_loads

    def _check_completed_saves(self, finished_req_ids: Set[str]) -> Set[str]:
        """Check for completed save operations."""
        completed = set()

        # Check pending finished requests first
        for request_id in list(self._pending_finished_requests):
            if request_id in self._save_futures and self._all_saves_done(request_id):
                completed.add(request_id)
                self._save_futures.pop(request_id)
                self._pending_finished_requests.remove(request_id)

        # Process newly finished requests
        for request_id in finished_req_ids:
            if request_id in self._save_futures:
                if self._all_saves_done(request_id):
                    completed.add(request_id)
                    self._save_futures.pop(request_id)
                else:
                    self._pending_finished_requests.add(request_id)
            else:
                completed.add(request_id)

        return completed

    def _check_completed_loads(self) -> Set[str]:
        """Check for completed load operations."""
        completed = set()
        for request_id in list(self._load_futures):
            if self._load_futures[request_id].done():
                completed.add(request_id)
                self._load_futures.pop(request_id)
        return completed

    def _all_saves_done(self, request_id: str) -> bool:
        """Check if all save operations for a request are completed."""
        return all(future.done() for future in self._save_futures[request_id])

    def _save_worker(self) -> None:
        """Background worker for handling save operations."""
        torch.cuda.set_device(self._device)
        while not self._stop_event.is_set():
            try:
                task = self._save_queue.get(block=True, timeout=1)
                self._handle_save_task(task)
            except Empty:
                continue
            except Exception as e:
                logger.error("Save worker error: %s", e)

    def _load_worker(self) -> None:
        """Background worker for handling load operations."""
        torch.cuda.set_device(self._device)
        while not self._stop_event.is_set():
            try:
                task = self._load_queue.get(block=True, timeout=1)
                self._handle_load_task(task)
            except Empty:
                continue
            except Exception as e:
                logger.error("Load worker error: %s", e)

    def _handle_save_task(self, task) -> None:
        """Handle individual save task with proper stream synchronization."""
        request_id, block_ids, block_hashes, future, main_stream_event = task
        start_time = time.perf_counter()
        buffers = None
        try:
            # Step1: Allocate storage pages
            key_pairs = [(hash_val, "") for hash_val in block_hashes]
            allocation_results = (
                self._connector._metadata_client.allocate_pages_for_keys(
                    self._rank, key_pairs
                )
            )

            if any(result[1] < 0 for result in allocation_results):
                return self._fail_task("Saved", "Page allocation failed", request_id, future)

            page_indices = [result[1] for result in allocation_results]
            offsets = [idx * self._bytes_per_page for idx in page_indices]

            # Step2: Allocate buffers and gather KV cache data
            buffers = self._save_buffer_allocator.alloc_buffer(len(block_ids))
            if buffers is None:
                return self._fail_task(
                    "Saved",
                    f"Buffer allocation failed for {len(block_ids)} blocks",
                    request_id,
                    future,
                )

            # Synchronize streams and gather data
            with torch.cuda.stream(self._save_stream):
                self._save_stream.wait_event(main_stream_event)  # Wait for main stream
                self._connector._gather_or_scatter_kv_caches(
                    block_ids, buffers, "gather"
                )

                save_stream_event = torch.cuda.Event()
                save_stream_event.record(self._save_stream)  # Record gather completion

            # Step3: Write data in batches
            write_futures = []
            for i in range(0, len(offsets), DEFAULT_MAX_IO_ENTRIES):
                batch_offsets = offsets[i : i + DEFAULT_MAX_IO_ENTRIES]
                batch_buffers = buffers[i : i + DEFAULT_MAX_IO_ENTRIES]
                client = self._connector._clients[self._connector._ac.next()]
                write_future = self._io_executor.submit(
                    client.batch_write, batch_offsets, batch_buffers, save_stream_event
                )
                write_futures.append(write_future)

            # Check write results
            write_success = all(
                result == self._bytes_per_page
                for write_future in write_futures
                for result in write_future.result()
            )

            # Step4: Confirm writes to metadata server
            if write_success:
                written_keys = list(zip(block_hashes, page_indices))
                self._connector._metadata_client.confirm_write_for_keys(
                    self._rank, written_keys, []
                )
                self._save_buffer_allocator.free_buffer(buffers)
                return self._succeed_task(
                    "Saved", start_time, request_id, len(block_ids), future
                )
            else:
                self._connector._metadata_client.confirm_write_for_keys(
                    self._rank, [], page_indices
                )
                self._save_buffer_allocator.free_buffer(buffers)
                return self._fail_task("Saved", "Write operation failed", request_id, future)

        except Exception as e:
            if buffers is not None:
                self._save_buffer_allocator.free_buffer(buffers)
            return self._fail_task("Saved", f"Task execution error: {e}", request_id, future)

    def _handle_load_task(self, task) -> None:
        """Handle individual load task."""
        request_id, block_ids, block_hashes, future = task
        start_time = time.perf_counter()
        buffers = None
        try:
            # Step1: Get block locations from metadata server
            page_indices = self._connector._metadata_client.get_key_locations(
                self._rank, block_hashes
            )

            if any(idx is None for idx in page_indices):
                return self._fail_task("Loaded", "Blocks not found", request_id, future)

            # Allocate read buffer
            buffers = self._load_buffer_allocator.alloc_buffer(len(block_ids))
            if buffers is None:
                return self._fail_task(
                    "Loaded",
                    f"Buffer allocation failed for {len(block_ids)} blocks",
                    request_id,
                    future,
                )

            # Step2: Read data in batches
            offsets = [idx * self._bytes_per_page for idx in page_indices]
            read_futures = []
            for i in range(0, len(offsets), DEFAULT_MAX_IO_ENTRIES):
                batch_offsets = offsets[i : i + DEFAULT_MAX_IO_ENTRIES]
                batch_buffers = buffers[i : i + DEFAULT_MAX_IO_ENTRIES]
                client = self._connector._clients[self._connector._ac.next()]
                read_future = self._io_executor.submit(
                    client.batch_read, batch_offsets, batch_buffers
                )
                read_futures.append(read_future)

            # Check read results
            read_success = all(
                result == self._bytes_per_page
                for read_future in read_futures
                for result in read_future.result()
            )

            if not read_success:
                self._load_buffer_allocator.free_buffer(buffers)
                return self._fail_task("Loaded", "Read operation failed", request_id, future)

            # Step3: Scatter data back to KV cache
            with torch.cuda.stream(self._load_stream):
                self._connector._gather_or_scatter_kv_caches(
                    block_ids, buffers, "scatter"
                )

            self._load_stream.synchronize()
            self._load_buffer_allocator.free_buffer(buffers)
            return self._succeed_task(
                "Loaded", start_time, request_id, len(block_ids), future
            )

        except Exception as e:
            if buffers is not None:
                self._load_buffer_allocator.free_buffer(buffers)
            return self._fail_task("Loaded", f"Task execution error: {e}", request_id, future)

    def _fail_task(
        self, 
        operation: str,
        error_msg: str, 
        request_id: str, 
        future: Future
    ) -> None:
        """Helper to fail task with error logging."""
        logger.error(
            "%s for %s request %s", 
            error_msg,
            operation, 
            request_id,
        )
        self.hf3fs_stats.record_failed_task_count(operation)
        future.set_result(False)

    def _succeed_task(
        self,
        operation: str,
        start_time: float,
        request_id: str,
        block_count: int,
        future: Future,
    ) -> None:
        """Helper to succeed task with logging."""
        duration = time.perf_counter() - start_time
        logger.info(
            "%s %s: %d blocks in %.2fs",
            operation,
            request_id,
            block_count,
            duration,
        )
        self.hf3fs_stats.record_success_task_duration(operation, duration)
        future.set_result(True)

    def shutdown(self) -> None:
        """Clean shutdown of all background threads and resources."""
        self._stop_event.set()
        self._save_thread.join()
        self._load_thread.join()
        self._io_executor.shutdown(wait=True)
        logger.info("AsyncOperationManager shutdown completed")


# ============================================================================
# HF3FS Connector
# ============================================================================


class HF3FSKVConnector(KVConnectorBase_V1):
    """HF3FS KV Connector implementation."""

    def __init__(
        self, 
        vllm_config: "VllmConfig", 
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config
        )

        # Core configuration
        self._vllm_config = vllm_config
        self._role = role
        self._block_size = vllm_config.cache_config.block_size
        self._use_mla = vllm_config.model_config.use_mla
        self._model_config = vllm_config.model_config

        logger.info("Using MLA: %s", self._use_mla)

        # HF3FS configuration
        kv_config = vllm_config.kv_transfer_config
        self._storage_path = kv_config.get_from_extra_config(
            "hf3fs_storage_path", "/vllm-workspace/mnt/hf3fs"
        )
        self._metadata_server_url = kv_config.get_from_extra_config(
            "hf3fs_metadata_server_url", "http://localhost:18000"
        )
        self._file_size = kv_config.get_from_extra_config(
            "hf3fs_file_size", 1024 * 1024 * 1024
        )
        self._numjobs = kv_config.get_from_extra_config("hf3fs_client_numjobs", 16)
        self._max_device_buffer_count = kv_config.get_from_extra_config(
            "hf3fs_max_device_buffer_count", 128
        )
        self._max_device_buffer_count = max(
            self._max_device_buffer_count, self._numjobs * DEFAULT_MAX_IO_ENTRIES
        )

        if self._role == KVConnectorRole.SCHEDULER:
            self._scheduling_states: Dict[str, RequestSchedulingState] = {}
            self._metadata_client = Hf3fsMetadataClient()
            self._metadata_client.initialize(0, role="scheduler")

        atexit.register(self.close)
        signal.signal(signal.SIGINT, lambda sig, frame: self.close())
        signal.signal(signal.SIGTERM, lambda sig, frame: self.close())
        signal.signal(signal.SIGQUIT, lambda sig, frame: self.close())

        logger.info(
            "HF3FSKVConnector initialized: path=%s, role=%s",
            self._storage_path,
            self._role.name,
        )

    ############################################################
    # Worker Side Methods
    ############################################################

    def register_kv_caches(self, kv_caches: Dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches
        self._setup_kv_cache_config()
        self._setup_storage_clients()
        self._async_manager = AsyncOperationManager(self)

    def _setup_kv_cache_config(self):
        first_cache = next(iter(self._kv_caches.values()))
        self._device = first_cache.device
        self._dtype = first_cache.dtype
        element_size = first_cache.element_size()

        if self._use_mla:
            assert len(first_cache.shape) == 3, "MLA format should have 3 dimensions"
            # MLA format: [num_blocks, block_size, head_size]
            num_blocks, block_size, head_size = first_cache.shape
            num_heads = 1
        else:
            # MHA format: [2, num_blocks, block_size, num_heads, head_size]
            _, num_blocks, block_size, num_heads, head_size = first_cache.shape

        self._local_total_tokens = num_blocks * block_size
        self._local_block_size = block_size

        if self._use_mla:
            layer_block_size = block_size * head_size * element_size
            self._bytes_per_page = layer_block_size * len(self._kv_caches)
            self._shape_per_page = [
                len(self._kv_caches),
                block_size,
                head_size,
            ]
        else:
            layer_block_size = 2 * block_size * num_heads * head_size * element_size
            self._bytes_per_page = layer_block_size * len(self._kv_caches)
            self._shape_per_page = [
                len(self._kv_caches),
                2,
                block_size,
                num_heads * head_size,
            ]

        self._kvcache_ptrs = torch.tensor(
            [cache.data_ptr() for cache in self._kv_caches.values()],
            dtype=torch.int64,
            device=self._device,
        )

    def _setup_storage_clients(self):
        os.makedirs(self._storage_path, exist_ok=True)

        self._rank = get_tensor_model_parallel_rank()
        file_path = os.path.join(
            self._storage_path, f"hf3fs_vllm_data_file_{self._rank}"
        )

        try:
            # Initialize HF3FS clients
            self._ac = AtomicCounter(self._numjobs)
            self._clients = [
                Hf3fsClient(
                    path=file_path,
                    size=self._file_size,
                    bytes_per_page=self._bytes_per_page,
                    entries=DEFAULT_MAX_IO_ENTRIES,
                )
                for _ in range(self._numjobs)
            ]

            # Initialize metadata client
            num_pages = self._file_size // self._bytes_per_page
            self._metadata_client = Hf3fsMetadataClient()
            self._metadata_client.initialize(self._rank, num_pages, role="worker")
        except Exception as e:
            logger.error("HF3FS client initialization failed: %s", e)
            raise

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """HF3FSConnector does not do layerwise saving."""
        pass

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, HF3FSConnectorMetadata):
            logger.error("Invalid metadata type: %s", type(metadata))
            return

        for request in metadata.requests:
            if request.save_block_op is None:
                continue

            skip_blocks = request.save_block_op.skip_leading_blocks
            block_hashes = self._generate_block_hashes(request.token_ids, skip_blocks)
            block_ids = request.block_ids[skip_blocks : skip_blocks + len(block_hashes)]

            for i in range(0, len(block_ids), self._max_device_buffer_count):
                batch_block_ids = block_ids[i : i + self._max_device_buffer_count]
                batch_block_hashes = block_hashes[i : i + self._max_device_buffer_count]
                self._async_manager.submit_save_operation(
                    request.request_id, batch_block_ids, batch_block_hashes
                )

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, HF3FSConnectorMetadata):
            logger.error("Invalid metadata type for loading")
            return

        for request in metadata.requests:
            if request.load_block_op is None:
                continue

            load_op = request.load_block_op
            block_ids = request.block_ids[: load_op.num_blocks_to_load]
            block_hashes = self._generate_block_hashes(
                request.token_ids, load_op.num_computed_blocks, len(block_ids)
            )

            for i in range(0, len(block_ids), self._max_device_buffer_count):
                batch_block_ids = block_ids[i : i + self._max_device_buffer_count]
                batch_block_hashes = block_hashes[i : i + self._max_device_buffer_count]
                self._async_manager.submit_load_operation(
                    request.request_id, batch_block_ids, batch_block_hashes
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def get_finished(
        self, finished_req_ids: Set[str]
    ) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        return self._async_manager.get_finished_operations(finished_req_ids)

    def get_kv_connector_stats(self) -> Optional["KVConnectorStats"]:
        """
        Get the KV connector stats collected during the last interval.
        """
        # Clear stats for next iteration
        if hasattr(self, "_async_manager"):
            if not self._async_manager.hf3fs_stats.is_empty():
                return self._async_manager.hf3fs_stats.clone_and_reset()
        return None

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        return True, None

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> Tuple[int, bool]:
        """Get number of new tokens that can be loaded from external cache."""
        try:
            state = self._get_or_create_scheduling_state(request.request_id)
            state.request = request

            num_tokens_to_check = self._align_to_block_size(
                len(request.prompt_token_ids) - 1
            )

            if num_tokens_to_check <= num_computed_tokens:
                state.load_op = LoadBlockInfo(
                    num_computed_blocks=num_computed_tokens // self._block_size,
                    num_blocks_to_load=0,
                    need_fetch_block_ids=[],
                )
                return 0, False

            token_ids_to_check = request.prompt_token_ids[:num_tokens_to_check]
            block_hashes = self._generate_block_hashes(token_ids_to_check, 0)

            # Check existence
            exists_results = self._metadata_client.batch_key_exists(block_hashes)

            # Count consecutive matches
            matched_blocks = next(
                (i for i, exists in enumerate(exists_results) if not exists),
                len(exists_results),
            )
            matched_tokens = matched_blocks * self._block_size
            new_hit_tokens = max(0, matched_tokens - num_computed_tokens)

            # Store load operation
            state.load_op = LoadBlockInfo(
                num_computed_blocks=num_computed_tokens // self._block_size,
                num_blocks_to_load=new_hit_tokens // self._block_size,
                need_fetch_block_ids=[],
            )

            logger.info(
                "Token matching for %s: %d matched (%d blocks), %d new hits, prompt len %d",
                request.request_id,
                matched_tokens,
                matched_blocks,
                new_hit_tokens,
                len(request.prompt_token_ids),
            )
            return new_hit_tokens, new_hit_tokens > 0

        except Exception as e:
            logger.error(
                "Error calculating matches for request %s: %s", request.request_id, e
            )
            return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        """Update state after block allocation."""
        state = self._get_or_create_scheduling_state(request.request_id)
        state.request = request

        if num_external_tokens <= 0 or not state.needs_loading():
            return

        # Validate block allocation
        expected_blocks = state.load_op.num_blocks_to_load
        actual_blocks = num_external_tokens // self._block_size
        assert (
            actual_blocks == expected_blocks
        ), f"Block count mismatch for {request.request_id}: expected {expected_blocks}, got {actual_blocks}"

        # Update load operation with allocated block IDs
        if actual_blocks > 0:
            local_block_ids = blocks.get_unhashed_block_ids()
            state.load_op.need_fetch_block_ids.extend(local_block_ids)
            state.phase = "WAITING_TO_LOAD"

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Build connector metadata for scheduling step."""
        metadata = HF3FSConnectorMetadata()

        for request_id in scheduler_output.finished_req_ids:
            self._scheduling_states.pop(request_id, None)

        # Process requests by phase
        self._process_waiting_to_load_requests(metadata)
        self._process_new_requests(scheduler_output, metadata)
        self._process_cached_requests(scheduler_output, metadata)

        return metadata

    def _process_waiting_to_load_requests(
        self, metadata: HF3FSConnectorMetadata
    ) -> None:
        """Process requests waiting to load."""
        for state in list(self._scheduling_states.values()):
            if not state.is_ready_to_load():
                continue

            # Create load request metadata
            num_cached_blocks = (
                state.load_op.num_computed_blocks + state.load_op.num_blocks_to_load
            )
            num_tokens_to_compute = num_cached_blocks * self._block_size

            # Initialize token_ids and allocated_block_ids for loading
            state.token_ids = state.request.prompt_token_ids[
                :num_tokens_to_compute
            ].copy()
            state.allocated_block_ids = state.load_op.need_fetch_block_ids.copy()

            request_metadata = HF3FSRequestMetadata.from_scheduling_state(
                state, self._block_size, state.load_op, num_cached_blocks
            )

            if request_metadata:
                metadata.add_request(request_metadata)
                state.phase = "ACTIVE"

    def _process_new_requests(
        self, scheduler_output: SchedulerOutput, metadata: HF3FSConnectorMetadata
    ) -> None:
        """Process new requests."""
        for request in scheduler_output.scheduled_new_reqs:
            state = self._get_or_create_scheduling_state(request.req_id)

            # Calculate tokens to compute
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            self._initialize_state_from_new_request(
                state, request, num_tokens_to_compute
            )

            # Create save metadata (skip cached blocks if any)
            num_cached_blocks = None
            if state.load_op:
                num_cached_blocks = (
                    state.load_op.num_computed_blocks + state.load_op.num_blocks_to_load
                )

            request_metadata = HF3FSRequestMetadata.from_scheduling_state(
                state, self._block_size, None, num_cached_blocks
            )

            if request_metadata:
                metadata.add_request(request_metadata)
                state.phase = "ACTIVE"

    def _process_cached_requests(
        self, scheduler_output: SchedulerOutput, metadata: HF3FSConnectorMetadata
    ) -> None:
        """Process cached requests."""
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, request_id in enumerate(cached_reqs.req_ids):
            state = self._get_or_create_scheduling_state(request_id)

            # Update with new tokens and blocks
            num_new_tokens = scheduler_output.num_scheduled_tokens[request_id]
            num_current_tokens = len(state.token_ids)
            new_token_ids = state.request.all_token_ids[
                num_current_tokens : num_current_tokens + num_new_tokens
            ]
            new_block_ids = cached_reqs.new_block_ids[i]

            state.update_tokens_and_blocks(new_token_ids, new_block_ids)

            # Create save metadata
            request_metadata = HF3FSRequestMetadata.from_scheduling_state(
                state, self._block_size, None
            )

            if request_metadata:
                metadata.add_request(request_metadata)

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> Optional["KVConnectorStats"]:
        """
        KVConnectorStats resolution method. This method allows dynamically
        registered connectors to return their own KVConnectorStats object,
        which can implement custom aggregation logic on the data dict.
        """
        return (
            HF3FSKVConnectorStats(data=data)
            if data is not None
            else HF3FSKVConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
    ) -> KVConnectorPromMetrics:
        return HF3FSPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    def close(self) -> None:
        try:
            if hasattr(self, "_async_manager"):
                self._async_manager.shutdown()

            if hasattr(self, "_clients"):
                for client in self._clients:
                    client.close()
                logger.info("HF3FS clients closed")
        except Exception as e:
            logger.error("Connector shutdown error: %s", e)

    ############################################################
    # Utility Methods
    ############################################################

    def _get_or_create_scheduling_state(
        self, request_id: str
    ) -> RequestSchedulingState:
        """Get existing or create new scheduling state."""
        if request_id not in self._scheduling_states:
            self._scheduling_states[request_id] = RequestSchedulingState(
                request_id=request_id
            )
        return self._scheduling_states[request_id]

    def _initialize_state_from_new_request(
        self, state: RequestSchedulingState, request, num_tokens_to_compute: int
    ) -> None:
        """Initialize state from new request data."""
        # Handle different block_ids formats in vLLM 0.9.0+
        if isinstance(request.block_ids[0], list):
            unfolded_block_ids = request.block_ids[0].copy()
        else:
            unfolded_block_ids = request.block_ids.copy()

        state.token_ids = request.prompt_token_ids[:num_tokens_to_compute].copy()
        state.allocated_block_ids = unfolded_block_ids
        state.num_saved_blocks = 0

    def _generate_block_hashes(
        self,
        token_ids: List[int],
        start_block_id: int,
        max_blocks_count: Optional[int] = None,
    ) -> List[str]:
        """Generate block hashes for token sequence."""
        block_hashes = []
        previous_hash = ""

        for start_idx in range(0, len(token_ids), self._block_size):
            if start_idx + self._block_size > len(token_ids):
                break

            end_idx = start_idx + self._block_size
            block_hash = self._compute_prefix_hash(
                token_ids[start_idx:end_idx], previous_hash
            )

            block_index = start_idx // self._block_size
            if block_index >= start_block_id:
                block_hashes.append(block_hash)

            if max_blocks_count and len(block_hashes) >= max_blocks_count:
                break
            previous_hash = block_hash

        return block_hashes

    def _gather_or_scatter_kv_caches(
        self, block_ids: List[int], block_buffers, operation: str
    ):
        for buffer_tensor, block_id in zip(block_buffers, block_ids):
            start_idx = block_id * self._local_block_size
            token_indices = list(range(start_idx, start_idx + self._local_block_size))
            if operation == "gather":
                gather_scatter_helper.gather_kv_caches(
                    self._kvcache_ptrs,
                    self._local_total_tokens,
                    buffer_tensor,
                    token_indices,
                    is_mla=self._use_mla,
                )
            else:
                gather_scatter_helper.scatter_kv_caches(
                    self._kvcache_ptrs,
                    self._local_total_tokens,
                    buffer_tensor,
                    token_indices,
                    is_mla=self._use_mla,
                )

    def _compute_prefix_hash(
        self, token_ids: List[int], previous_hash: str = ""
    ) -> str:
        """Compute prefix hash for token block."""
        combined_string = f"{previous_hash}_{token_ids}"
        return hashlib.md5(combined_string.encode()).hexdigest()

    def _align_to_block_size(self, num_tokens: int) -> int:
        """Align token count to block size."""
        return (num_tokens // self._block_size) * self._block_size

@dataclass
class HF3FSKVConnectorStats(KVConnectorStats):
    """Container for transfer performance metrics"""
    def __post_init__(self):
        if not self.data:
            # Empty container init, no data is passed in.
            self.reset()

    def reset(self):
        # Must be serializable
        self.data: dict[str, float | list[float]] = {
            "save_duration": [],
            "load_duration": [],
            "num_failed_save": 0,
            "num_failed_load": 0,
            "num_transfer_task": 0,
        }
    
    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        if not other.is_empty():
            for k, v in other.data.items():
                accumulator = self.data[k]
                if isinstance(accumulator, list):
                    accumulator.extend(v)
                else: # int
                    self.data[k] += v
        return self

    def reduce(self) -> dict[str, int | float]:
        # Compute compact representative stats suitable for CLI logging
        if self.is_empty():
            return {
                "Num transfers task": 0,
                "Num save task (success/failed)": "0/0",
                "Num load task (success/failed)": "0/0",
                "Avg save duration (ms)": 0,
                "P90 save duration (ms)": 0,
                "Avg load duration (ms)": 0,
                "P90 load duration (ms)": 0,
            }
        num_success_save = len(self.data["save_duration"])
        num_success_load = len(self.data["load_duration"])
        num_failed_save = self.data["num_failed_save"]
        num_failed_load = self.data["num_failed_load"]
        if num_success_save == 0:
            save_duration = np.zeros(1)
        else:
            save_duration = np.asarray(self.data["save_duration"])
        if num_success_load == 0:
            load_duration = np.zeros(1)
        else:
            load_duration = np.asarray(self.data["load_duration"])

        return {
            "Num transfers task": self.data["num_transfer_task"],
            "Num save task (success/failed)": f"{num_success_save}/{num_failed_save}",
            "Num load task (success/failed)": f"{num_success_load}/{num_failed_load}",
            "Avg save duration (ms)": round(save_duration.mean() * 1e3, 3),
            "P90 save duration (ms)": round(np.percentile(save_duration, 90) * 1e3, 3),
            "Avg load duration (ms)": round(load_duration.mean() * 1e3, 3),
            "P90 load duration (ms)": round(np.percentile(load_duration, 90) * 1e3, 3),
        }

    def is_empty(self) -> bool:
        return self.data["num_transfer_task"] == 0
        
    def record_success_task_duration(self, operation, duration):
        if operation == "Saved":
            self.data["save_duration"].append(duration)
        elif operation == "Loaded":
            self.data["load_duration"].append(duration)
        self.data["num_transfer_task"] += 1
 
    def record_failed_task_count(self, operation):
        if operation == "Saved":
            self.data["num_failed_save"] += 1
        elif operation == "Loaded":
            self.data["num_failed_load"] += 1
        self.data["num_transfer_task"] += 1

    def clone_and_reset(self):
        old = copy.copy(self)
        self.reset()
        return old

class HF3FSPromMetrics(KVConnectorPromMetrics):
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[str]],
    ):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        buckets = [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
            5.0,
        ]
        hf3fs_save_duration = self._histogram_cls(
            name="vllm:hf3fs_save_duration_seconds",
            documentation="Histogram of save duration for HF3FSKVConnector.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.hf3fs_save_duration = self.make_per_engine(hf3fs_save_duration)
        
        hf3fs_load_duration = self._histogram_cls(
            name="vllm:hf3fs_load_duration_seconds",
            documentation="Histogram of load duration for HF3FSKVConnector.",
            buckets=buckets,
            labelnames=labelnames,
        )
        self.hf3fs_load_duration = self.make_per_engine(hf3fs_load_duration)
        
        hf3fs_num_failed_save = self._counter_cls(
            name="vllm:hf3fs_num_failed_save",
            documentation="Number of failed HF3FS KV save.",
            labelnames=labelnames,
        )
        self.hf3fs_num_failed_save = self.make_per_engine(hf3fs_num_failed_save)
        
        hf3fs_num_failed_load = self._counter_cls(
            name="vllm:hf3fs_num_failed_load",
            documentation="Number of failed HF3FS KV load.",
            labelnames=labelnames,
        )
        self.hf3fs_num_failed_load = self.make_per_engine(hf3fs_num_failed_load)
    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        
        for prom_obj, list_item_key in zip(
            [
                self.hf3fs_save_duration,
                self.hf3fs_load_duration,
            ],
            [
                "save_duration",
                "load_duration",
            ],
        ):
            for list_item in transfer_stats_data[list_item_key]:
                prom_obj[engine_idx].observe(list_item)
        for counter_obj, counter_item_key in zip(
            [
                self.hf3fs_num_failed_save,
                self.hf3fs_num_failed_load,
            ],
            [
                "num_failed_save",
                "num_failed_load",
            ],
        ):
            counter_obj[engine_idx].inc(transfer_stats_data[counter_item_key])