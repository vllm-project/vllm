# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
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
        disk_path: str | None = None,
        disk_capacity_bytes: int = 0,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes
        self._disk_path = disk_path
        self._disk_capacity_bytes = disk_capacity_bytes

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # Disk backend (initialized in register_kv_caches)
        self._disk_backend: "DiskBackend | None" = None

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        self._backend = DmaCopyBackend()

        # Ordered (event_idx, Event). Events pre-allocated on main thread.
        self._load_events: list[tuple[int, torch.Event]] = []
        self._store_events: list[tuple[int, torch.Event]] = []
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

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
            return

        # Resolve each entry to a representative tensor for storage
        # deduplication. For attention layers the value is already a tensor;
        # for Mamba layers it is a list of tensors that all share the same
        # underlying raw storage, so we take the first one.
        def _repr_tensor(v: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
            assert isinstance(v, torch.Tensor | list)
            return v if isinstance(v, torch.Tensor) else v[0]

        any_tensor = _repr_tensor(next(iter(kv_caches.values())))
        self.device = any_tensor.device

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks

        # Deduplicate: multiple layers may share the same backing storage.
        seen_ptrs: dict[int, tuple[str, torch.Tensor]] = {}
        for name, value in kv_caches.items():
            tensor = _repr_tensor(value)
            ptr = tensor.untyped_storage().data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs[ptr] = (name, tensor)

        # Build [num_blocks, block_bytes] int8 views from each unique
        # storage so that stride(0) gives block_bytes for the copy op.
        #
        # The physical layout varies across attention backends:
        #   FlashAttn/ROCm:  (2, num_blocks, ...) -> K/V outermost, 2 segments
        #   FlashInfer/MLA:  (num_blocks, ...)    -> blocks outermost, 1 segment
        # We derive page_size_bytes = storage.nbytes() // num_blocks, then
        # classify dims: any dim whose byte-stride exceeds page_size_bytes
        # must be an outer segment dim (e.g. the K/V dim of size 2). A less
        # hacky way is to update the interface with the layout.
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        for name, tensor in seen_ptrs.values():
            storage = tensor.untyped_storage()
            raw = torch.empty(0, dtype=torch.int8, device=self.device).set_(
                storage, 0, (storage.nbytes(),)
            )
            el = tensor.element_size()
            page_size_bytes = storage.nbytes() // num_blocks
            outer_dims = [
                d for d in range(tensor.ndim) if tensor.stride(d) * el > page_size_bytes
            ]
            if not outer_dims:
                unique_gpu_caches[name] = raw.view(num_blocks, -1)
            else:
                seg_stride = tensor.stride(outer_dims[0]) * el
                for idx in range(tensor.shape[outer_dims[0]]):
                    offset = idx * seg_stride
                    chunk = raw[offset : offset + seg_stride]
                    unique_gpu_caches[f"{name}.{idx}"] = chunk.view(num_blocks, -1)

        # Compute per-tensor bytes_per_block. Tensors may have different
        # page_size_bytes (e.g., UniformTypeKVCacheSpecs with varying head_size).
        per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in unique_gpu_caches.values()
        ]
        total_bytes_per_block = sum(per_tensor_bpb)

        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)

        logger.info(
            "SimpleCPUOffloadWorker: %d unique GPU KV tensors, "
            "allocating %d CPU blocks (%.2f GB)",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

        pin_memory = is_pin_memory_available()
        if not pin_memory:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        self.gpu_kv_caches = unique_gpu_caches
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

        # Use lowest priority so KV cache I/O yields to compute streams.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

        # Initialize copy backend with caches and streams.
        self._backend.init(
            self.gpu_kv_caches,
            self.cpu_kv_caches,
            self.device,
            self.load_stream,
            self.store_stream,
        )

        # Initialize disk backend if configured.
        if self._disk_path and self._disk_capacity_bytes > 0:
            from vllm.v1.simple_kv_offload.disk import create_disk_backend
            bytes_per_block = {
                name: t.shape[1] * t.element_size()
                for name, t in self.cpu_kv_caches.items()
            }
            total_bpb = sum(bytes_per_block.values())
            num_disk_blocks = max(
                1, self._disk_capacity_bytes // total_bpb
            )
            self._disk_backend = create_disk_backend(
                disk_path=self._disk_path,
                tensor_names=list(self.cpu_kv_caches.keys()),
                bytes_per_block=bytes_per_block,
                num_disk_blocks=num_disk_blocks,
                try_gds=True,
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

        Called after model execution. The manager only schedules stores for
        blocks whose KV data is confirmed computed, so we launch both loads
        and stores immediately — no deferral or cross-stream sync needed.

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending: always None (stores use worker metadata).
            - finished_recving: req_ids whose loads have completed.
        """
        # (1) Submit transfers
        metadata = self._connector_metadata
        if metadata is not None:
            # Launch loads (CPU->GPU).
            if metadata.load_cpu_blocks:
                self._backend.launch_copy(
                    metadata.load_cpu_blocks,
                    metadata.load_gpu_blocks,
                    is_store=False,
                    event_idx=metadata.load_event,
                    events_list=self._load_events,
                )
            # Launch stores (GPU->CPU).
            if metadata.store_gpu_blocks:
                self._backend.launch_copy(
                    metadata.store_gpu_blocks,
                    metadata.store_cpu_blocks,
                    is_store=True,
                    event_idx=metadata.store_event,
                    events_list=self._store_events,
                )

            # Disk write-back: queue evicted CPU blocks for background write
            if (
                self._disk_backend is not None
                and metadata.disk_write_cpu_blocks
            ):
                assert self.cpu_kv_caches is not None
                self._disk_backend.write_blocks(
                    self.cpu_kv_caches,
                    metadata.disk_write_cpu_blocks,
                    metadata.disk_write_disk_blocks,
                )

            # Disk read (prefetch): read disk blocks into CPU (blocking)
            # Must complete before CPU→GPU load can use these blocks
            if (
                self._disk_backend is not None
                and metadata.disk_read_cpu_blocks
            ):
                assert self.cpu_kv_caches is not None
                # Issue fadvise WILLNEED hint to trigger kernel
                # readahead before the blocking pread calls
                if hasattr(self._disk_backend, 'prefetch_hint'):
                    self._disk_backend.prefetch_hint(
                        metadata.disk_read_disk_blocks
                    )
                self._disk_backend.read_blocks_to_cpu(
                    self.cpu_kv_caches,
                    metadata.disk_read_cpu_blocks,
                    metadata.disk_read_disk_blocks,
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
