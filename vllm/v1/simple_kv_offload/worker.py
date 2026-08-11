# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY
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
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes
        self.disk_path = disk_path
        self.disk_capacity_bytes = disk_capacity_bytes
        self.disk_buffer_slots = disk_buffer_slots
        self.use_page_cache = use_page_cache
        self.disk_mode = kv_offload_backend == "disk"

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
            kv_caches: Per-layer GPU KV caches.
        """
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        self.device = next(iter(kv_caches.values())).device

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks

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
            regions = raw.view(-1, num_blocks, block_bytes)
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

        self.cpu_kv_caches = {}
        for name, gpu_tensor in unique_gpu_caches.items():
            cpu_shape = (self.num_cpu_blocks,) + gpu_tensor.shape[1:]
            # Allocate non-pinned first, then pin via cudaHostRegister to
            # bypass PyTorch's CUDACachingHostAllocator which rounds up to
            # the next power of 2 (e.g. 100 GB -> 128 GB).
            tensor = torch.zeros(cpu_shape, dtype=gpu_tensor.dtype, device="cpu")
            if pin_memory:
                pin_tensor(tensor)
            self.cpu_kv_caches[name] = tensor

        self._backend = DmaCopyBackend()
        self._backend.init(
            unique_gpu_caches,
            self.cpu_kv_caches,
            device,
            self.load_stream,
            self.store_stream,
        )

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        # NOTE: we defer launching both load and store to get_finished(),
        # which runs after model execution. This hides the CPU-side
        # block copy op overhead (~5ms) behind GPU compute.
        pass

    def wait_for_save(self) -> None:
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Submit transfers and report completed events to the scheduler.

        Stores (GPU->CPU) read the live KV cache, which the compute stream may
        still be writing under v1 overlapped execution, so they are ordered
        after a compute-done event recorded on the current stream. Loads
        (CPU->GPU) read stable pinned host memory and launch immediately. See
        #45704 for the bug and #39306 for the srcAccessOrder rationale.

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending: always None (stores use worker metadata).
            - finished_recving: req_ids whose loads have completed.
        """
        # (1) Submit transfers
        metadata = self._connector_metadata
        if metadata is not None:
            backend = self._backend
            assert backend is not None
            if metadata.load_cpu_blocks:
                backend.launch_copy(
                    metadata.load_cpu_blocks,
                    metadata.load_gpu_blocks,
                    is_store=False,
                    event_idx=metadata.load_event,
                    events_list=self._load_events,
                )
            if metadata.store_gpu_blocks:
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

        # (2) Track completed transfer events
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
