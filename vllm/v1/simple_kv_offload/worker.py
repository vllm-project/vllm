# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import ConnectorInitState
from vllm.logger import init_logger
from vllm.utils.host_memory import HOST_REGISTER_CHUNK_SIZE_BYTES, host_unregister
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.async_host_buffer import AsyncHostBuffer
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
from vllm.v1.simple_kv_offload.disk_backend import DiskBackend
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class _CpuCaches(NamedTuple):
    """CPU KV tensors plus their registered ranges, for unpinning on release."""

    tensors: dict[str, torch.Tensor]
    registered_ptrs: list[int]


def _allocate_cpu_caches(
    plan: dict[str, tuple[tuple[int, ...], torch.dtype]],
    pin_memory: bool,
    chunk_size_bytes: int | None = None,
) -> _CpuCaches:
    """Allocate zeroed CPU tensors and page-lock them via cudaHostRegister."""
    tensors: dict[str, torch.Tensor] = {}
    registered: list[int] = []
    try:
        for name, (shape, dtype) in plan.items():
            tensor = torch.zeros(shape, dtype=dtype, device="cpu")
            if pin_memory:
                registered += pin_tensor(tensor, chunk_size_bytes)
            tensors[name] = tensor
    except BaseException:
        host_unregister(registered)
        raise
    return _CpuCaches(tensors, registered)


def _release_cpu_caches(caches: _CpuCaches) -> None:
    host_unregister(caches.registered_ptrs)


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        kv_offload_backend: str = "cpu",
        disk_path: str | None = None,
        disk_capacity_bytes: int = 0,
        disk_buffer_slots: int = 2,
        use_page_cache: bool = False,
        async_init: bool = False,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes
        self.disk_path = disk_path
        self.disk_capacity_bytes = disk_capacity_bytes
        self.disk_buffer_slots = disk_buffer_slots
        self.use_page_cache = use_page_cache
        self.disk_mode = kv_offload_backend == "disk"
        assert not (async_init and self.disk_mode)

        self._async_init = async_init
        self._cpu_init: AsyncHostBuffer[_CpuCaches] | None = None
        self._alloc_plan: dict[str, tuple[tuple[int, ...], torch.dtype]] | None = None
        self._pin_memory = False
        self._nothing_to_offload = False
        self._init_failed = False

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        self._backend: DmaCopyBackend | DiskBackend | None = None

        # Ordered (event_idx, Event). Events pre-allocated on main thread.
        self._load_events: list[tuple[int, torch.Event]] = []
        self._store_events: list[tuple[int, torch.Event]] = []
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Compute-done event recorded before each store; reused across steps
        # (get_finished runs once per step, copy queue is FIFO).
        self._store_compute_done: torch.Event | None = None

        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        # Completed store events to report via build_connector_worker_meta
        self._completed_store_events: dict[int, int] = {}

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register GPU KV caches and allocate pinned CPU tensors.
        The worker will infer the underlying raw storage from the kv_caches.

        Args:
            kv_caches: Per-layer GPU KV caches. Values are either a single
                tensor (attention layers) or a list of tensors (Mamba layers
                in hybrid models). All values are included for offloading
                by resolving to their underlying raw storage.
        """
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            self._nothing_to_offload = True
            return

        self.device = next(iter(kv_caches.values())).device

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks
        assert self.kv_cache_config.kv_cache_tensors
        logical_storage_bytes = self.kv_cache_config.kv_cache_tensors[0].size

        # The DMA backend copies whole blocks as base + block_id * stride(0),
        # so view each unique allocation as [num_blocks, block_bytes].
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        seen: set[tuple[torch.device, int]] = set()
        for name, tensor in kv_caches.items():
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr())
            if key in seen:
                continue
            seen.add(key)

            physical_per_block, remainder = divmod(tensor.shape[0], num_blocks)
            assert remainder == 0, (
                f"KV cache {name!r} has {tensor.shape[0]} physical blocks, which "
                f"is not divisible by {num_blocks} scheduler blocks"
            )
            block_bytes = tensor.stride(0) * tensor.element_size() * physical_per_block
            raw = torch.empty(0, dtype=torch.int8, device=tensor.device).set_(storage)
            assert raw.numel() >= logical_storage_bytes, (
                f"KV cache {name!r} storage has {raw.numel()} bytes, smaller "
                f"than the configured {logical_storage_bytes}-byte allocation"
            )
            regions = raw[:logical_storage_bytes].view(-1, num_blocks, block_bytes)
            for idx, region in enumerate(regions):
                key_name = name if len(regions) == 1 else f"{name}.{idx}"
                unique_gpu_caches[key_name] = region

        # Compute per-tensor bytes_per_block. Tensors may have different
        # page_size_bytes (e.g., UniformTypeKVCacheSpecs with varying head_size).
        per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in unique_gpu_caches.values()
        ]
        total_bytes_per_block = sum(per_tensor_bpb)

        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)

        # Use lowest priority so KV cache I/O yields to compute streams.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

        self.gpu_kv_caches = unique_gpu_caches

        if self.disk_mode:
            self._init_disk_mode(unique_gpu_caches, total_bytes_per_block, self.device)
        else:
            self._init_cpu_mode(unique_gpu_caches, total_bytes_per_block, self.device)

    def _init_disk_mode(
        self,
        unique_gpu_caches: dict[str, torch.Tensor],
        total_bytes_per_block: int,
        device: torch.device,
    ) -> None:
        num_disk_slots = max(1, self.disk_capacity_bytes // total_bytes_per_block)
        self.num_cpu_blocks = num_disk_slots

        logger.info(
            "SimpleCPUOffloadWorker [DISK]: %d tensors, %d disk slots (%.2f GB)",
            len(unique_gpu_caches),
            num_disk_slots,
            (num_disk_slots * total_bytes_per_block) / (1024**3),
        )

        assert self.disk_path is not None
        rank_path = f"{self.disk_path}.rank_{device.index or 0}"
        self._backend = DiskBackend()
        self._backend.init(
            unique_gpu_caches,
            device,
            self.load_stream,
            self.store_stream,
            rank_path,
            num_disk_slots,
            total_bytes_per_block,
            self.disk_buffer_slots,
            self.use_page_cache,
        )

    def _init_cpu_mode(
        self,
        unique_gpu_caches: dict[str, torch.Tensor],
        total_bytes_per_block: int,
        device: torch.device,
    ) -> None:
        logger.info(
            "SimpleCPUOffloadWorker [CPU]: %d tensors, %d CPU blocks (%.2f GB)",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

        pin_memory = PIN_MEMORY
        if not pin_memory:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        alloc_plan = {
            name: ((self.num_cpu_blocks,) + gpu_tensor.shape[1:], gpu_tensor.dtype)
            for name, gpu_tensor in unique_gpu_caches.items()
        }
        if self._async_init:
            # Defer the background build to the first get_init_state poll:
            # engine warmup captures CUDA graphs after register_kv_caches, and
            # a concurrent cudaHostRegister invalidates an in-progress capture.
            # get_init_state only runs from the step path, after all capture.
            self._alloc_plan = alloc_plan
            self._pin_memory = pin_memory
            return

        self.cpu_kv_caches = _allocate_cpu_caches(alloc_plan, pin_memory).tensors
        self._init_cpu_backend()

    def _init_cpu_backend(self) -> None:
        assert self.gpu_kv_caches is not None
        assert self.cpu_kv_caches is not None
        backend = DmaCopyBackend()
        backend.init(
            self.gpu_kv_caches,
            self.cpu_kv_caches,
            self.device,
            self.load_stream,
            self.store_stream,
        )
        self._backend = backend

    def get_init_state(self) -> ConnectorInitState | None:
        """Drive asynchronous CPU-cache initialization and report its state."""
        if not self._async_init:
            return None
        if self._init_failed:
            raise RuntimeError("Asynchronous CPU KV cache initialization failed")
        if self._backend is not None or self._nothing_to_offload:
            return ConnectorInitState.READY
        if self._cpu_init is None:
            if self._alloc_plan is None:
                return ConnectorInitState.INITIALIZING
            logger.info("Starting asynchronous CPU KV cache initialization")
            alloc_plan, pin_memory = self._alloc_plan, self._pin_memory
            self._alloc_plan = None
            self._cpu_init = AsyncHostBuffer(
                allocate=lambda: _allocate_cpu_caches(
                    alloc_plan, pin_memory, HOST_REGISTER_CHUNK_SIZE_BYTES
                ),
                cleanup=_release_cpu_caches,
                thread_name="vllm-simple-kv-offload-init",
            )
            return ConnectorInitState.INITIALIZING
        caches = self._cpu_init.poll()
        if self._cpu_init.failed:
            self._init_failed = True
            raise RuntimeError("Asynchronous CPU KV cache initialization failed")
        if caches is None:
            return ConnectorInitState.INITIALIZING
        self.cpu_kv_caches = caches.tensors
        try:
            self._init_cpu_backend()
        except BaseException:
            self._init_failed = True
            self.cpu_kv_caches = None
            if caches.registered_ptrs:
                host_unregister(caches.registered_ptrs)
            raise
        logger.info("Asynchronous CPU KV cache initialization completed")
        return ConnectorInitState.READY

    def shutdown(self) -> None:
        """Release CPU caches that were prepared but never adopted."""
        if self._cpu_init is not None:
            self._cpu_init.close()

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        # Invoked after the forward launch on steps without sync loads
        # (SchedulerOutput.has_sync_kv_loads), so the CPU-side block copy
        # op overhead (~5ms) stays hidden behind GPU compute. Stores are
        # issued in wait_for_save().
        metadata = self._connector_metadata
        if metadata is not None and metadata.load_cpu_blocks:
            backend = self._backend
            assert backend is not None
            backend.launch_copy(
                metadata.load_cpu_blocks,
                metadata.load_gpu_blocks,
                is_store=False,
                event_idx=metadata.load_event,
                events_list=self._load_events,
            )

    def wait_for_save(self) -> None:
        """Submit async stores.

        Stores (GPU->CPU) read the live KV cache, which the compute stream may
        still be writing under v1 overlapped execution, so they are ordered
        after a compute-done event recorded on the current stream. See
        #45704 for the bug and #39306 for the srcAccessOrder rationale.
        """
        metadata = self._connector_metadata
        if metadata is not None and metadata.store_gpu_blocks:
            backend = self._backend
            assert backend is not None
            if self._store_compute_done is None:
                self._store_compute_done = torch.Event()
            self._store_compute_done.record(torch.cuda.current_stream())
            backend.launch_copy(
                metadata.store_gpu_blocks,
                metadata.store_cpu_blocks,
                is_store=True,
                event_idx=metadata.store_event,
                events_list=self._store_events,
                wait_event=self._store_compute_done,
            )

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report completed transfer events to the scheduler.

        Loads are submitted in start_load_kv() and stores in wait_for_save().

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending: always None (stores use worker metadata).
            - finished_recving: req_ids whose loads have completed.
        """
        metadata = self._connector_metadata
        finished_recving: set[str] = set()

        if self._pending_load_event_indices:
            load_wm = self._poll_stream_events(is_store=False)
            for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
                self._pending_load_event_indices.discard(j)
                req_ids = (
                    metadata.load_event_to_reqs.get(j) if metadata is not None else None
                )
                if req_ids:
                    finished_recving.update(req_ids)

        if self._pending_store_event_indices:
            store_wm = self._poll_stream_events(is_store=True)
            for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
                self._pending_store_event_indices.discard(j)
                self._completed_store_events[j] = 1

        return None, finished_recving or None

    def build_connector_worker_meta(self) -> SimpleCPUOffloadWorkerMetadata | None:
        """Return completed store events since the last call."""
        if not self._completed_store_events:
            return None
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
        )
        self._completed_store_events = {}
        return meta

    def handle_preemptions(
        self, kv_connector_metadata: SimpleCPUOffloadMetadata
    ) -> None:
        """Sync all in-flight transfers before preempted blocks are reused."""
        if not kv_connector_metadata.need_flush:
            return
        self._flush_and_sync_all()

    def _flush_and_sync_all(self) -> None:
        """Synchronize all in-flight transfer events."""
        for event_idx, event in self._load_events:
            event.synchronize()
            self._load_hwm = event_idx
        self._load_events.clear()

        for event_idx, event in self._store_events:
            event.synchronize()
            self._store_hwm = event_idx
        self._store_events.clear()

    def _poll_stream_events(self, is_store: bool) -> int:
        """Non-blocking poll for completed events and return the high-water mark."""
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.pop(0)
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm
